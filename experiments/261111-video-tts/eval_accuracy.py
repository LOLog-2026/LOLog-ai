"""TTS 정확도 평가 (#33). testset.json의 대사 144개를 읽히고 받아쓰기·목소리·길이로 채점한다.

    python -I eval_accuracy.py gen   VOICES_DIR OUT [--xvec] [--model ...]   # 읽기 + 화자 벡터
    python -I eval_accuracy.py score OUT                                     # Whisper 받아쓰기 + 채점표

기준(README "평가 기준")
  읽기    받아쓴 글과 들려야 하는 말의 글자 오류율(CER). 대사 하나는 CER 10% 이하면 통과
  목소리  읽은 소리의 화자 벡터가 24명 인사 클립 중 자기 클립과 가장 가까우면 통과(같은 나이대·성별이면 묶음 통과)
  길이    글자 수 × 0.45초(최소 4초) 안이면 통과
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE if (HERE / "tts.py").exists() else HERE.parents[1] / "src" / "lolog_ai"))
import tts as T  # noqa: E402

CER_PASS = 0.10
WHISPER = "openai/whisper-large-v3-turbo"


def norm(s: str) -> str:
    return re.sub(r"[^가-힣a-z0-9]", "", s.lower())


def cer(ref: str, hyp: str) -> float:
    r, h = norm(ref), norm(hyp)
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i]
        for j, hc in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / len(r)


def gen(a) -> None:
    import librosa
    import numpy as np
    import soundfile as sf
    import torch

    out = Path(a.out)
    (out / "wav").mkdir(parents=True, exist_ok=True)
    items = json.loads((HERE / "testset.json").read_text(encoding="utf-8"))
    tts = T.Tts(a.voices, model=a.model, xvec_only=a.xvec)
    model = tts.load()
    spk_sr = model.model.speaker_encoder_sample_rate

    def embed(wav, sr):
        if sr != spk_sr:
            wav = librosa.resample(y=wav.astype(np.float32), orig_sr=sr, target_sr=spk_sr)
        e = model.model.extract_speaker_embedding(audio=wav, sr=spk_sr).float().flatten()
        return torch.nn.functional.normalize(e, dim=0)

    ids = sorted(tts.voices())
    refs = torch.stack([embed(*sf.read(Path(a.voices) / f"{v}.wav", dtype="float32")) for v in ids])

    rows, t_all = [], time.perf_counter()
    for it in items:
        t0 = time.perf_counter()
        wav, sr = tts.say(it["voice"], it["text"], seed=it["id"])
        sec = time.perf_counter() - t0
        sf.write(out / "wav" / f"{it['id']:03d}.wav", wav, sr)
        sims = (refs @ embed(wav, sr)).tolist()
        own = sims[ids.index(it["voice"])]
        top = ids[max(range(len(ids)), key=sims.__getitem__)]
        spoken = T.speakable(it["text"])
        limit = max(T.MIN_SECONDS, len(spoken) * T.MAX_SECONDS_PER_CHAR)
        row = {**it, "sec": round(sec, 2), "audio": round(len(wav) / sr, 2), "tries": tts.last_tries,
               "len_ok": len(wav) / sr <= limit, "sim_own": round(own, 3), "top_voice": top,
               "voice_ok": top == it["voice"], "group_ok": top[:-3] == it["voice"][:-3]}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    meta = {"model": a.model, "xvec_only": a.xvec, "total_sec": round(time.perf_counter() - t_all, 1),
            "torch_peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    (out / "gen.json").write_text(json.dumps({"meta": meta, "rows": rows}, ensure_ascii=False, indent=1),
                                  encoding="utf-8")


def score(a) -> None:
    import torch
    from transformers import pipeline

    out = Path(a.out)
    data = json.loads((out / "gen.json").read_text(encoding="utf-8"))
    asr = pipeline("automatic-speech-recognition", model=WHISPER, torch_dtype=torch.float16, device="cuda:0")
    rows = data["rows"]
    for r in rows:
        hyp = asr(str(out / "wav" / f"{r['id']:03d}.wav"),
                  generate_kwargs={"language": "korean", "task": "transcribe"})["text"].strip()
        r["heard"] = hyp
        r["cer"] = round(cer(r["expect"], hyp), 3)
        r["read_ok"] = r["cer"] <= CER_PASS
        r["ok"] = r["read_ok"] and r["len_ok"] and r["group_ok"]

    def rate(rs, key):
        return round(sum(bool(r[key]) for r in rs) / len(rs), 3)

    def block(rs):
        return {"n": len(rs), "read_ok": rate(rs, "read_ok"), "cer_mean": round(sum(r["cer"] for r in rs) / len(rs), 3),
                "voice_ok": rate(rs, "voice_ok"), "group_ok": rate(rs, "group_ok"), "len_ok": rate(rs, "len_ok"),
                "all_ok": rate(rs, "ok"), "retried": sum(r["tries"] > 1 for r in rs),
                "rtf": round(sum(r["sec"] for r in rs) / sum(r["audio"] for r in rs), 2)}

    summary = {"all": block(rows)}
    for cat in sorted({r["cat"] for r in rows}):
        summary[cat] = block([r for r in rows if r["cat"] == cat])
    by_voice = {v: block([r for r in rows if r["voice"] == v]) for v in sorted({r["voice"] for r in rows})}
    data.update(summary=summary, by_voice=by_voice)
    (out / "score.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(data["meta"], ensure_ascii=False))
    for k, v in summary.items():
        print(f"{k:13s} {json.dumps(v)}")
    print("-- 읽기 실패")
    for r in rows:
        if not r["read_ok"]:
            print(f"{r['id']:3d} {r['voice']:11s} {r['cat']:12s} cer {r['cer']:.2f} | {r['text']} -> {r['heard']}")
    print("-- 목소리 묶음 실패")
    for r in rows:
        if not r["group_ok"]:
            print(f"{r['id']:3d} {r['voice']} -> {r['top_voice']} (sim_own {r['sim_own']}) | {r['text']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("voices")
    g.add_argument("out")
    g.add_argument("--xvec", action="store_true")
    g.add_argument("--model", default=T.MODEL)
    s = sub.add_parser("score")
    s.add_argument("out")
    a = ap.parse_args()
    {"gen": gen, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()
