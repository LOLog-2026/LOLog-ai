"""TTS 정확도 평가 (#33). testset.json의 대사 144개를 읽히고 받아쓰기·목소리·길이로 채점한다.

    python -I eval_accuracy.py gen   VOICES_DIR OUT [--engine qwen|chatterbox|xtts] [--model ...] [--xvec]
                                     [--temp 0.6] [--punct] [--verify] [--only kid-f-04,young-m-03] [--limit N]
    python -I eval_accuracy.py score OUT [--voices VOICES_DIR]

gen은 그 엔진의 venv로, score는 qwen venv(transformers, librosa)로 돌린다. 엔진은 engines.py.

기준(EVAL_PLAN.md)
  읽기    Whisper(large-v3-turbo)로 받아쓴 글과 들려야 하는 말의 글자 오류율(CER). 10% 이하면 통과
  목소리  읽은 소리의 화자 벡터(WavLM-SV, TTS 모델과 상관없는 화자 인식 모델)가 24명 인사 클립 중 같은
          성별 클립과 가장 가까우면 통과(gender_ok). 같은 나이대·성별 묶음(group_ok), 자기 클립(voice_ok)은 참고.
          이 모델은 어른 목소리로 배워서 합성한 목소리의 나이대는 잘 가리지 못했다(틀린 것의 대부분이 같은 성별의
          옆 나이대)
  길이    글자 수 × 0.45초(최소 4초) 안이면 통과

--verify: 읽은 소리를 작은 받아쓰기 모델(whisper-small)로 확인해서 CER 10%를 넘으면 시드를 바꿔 3번까지 다시
읽고 가장 나은 것을 쓴다. 채점용 모델(large-v3-turbo)과 다른 모델로 골라 채점이 유리해지지 않게 한다.
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
sys.path.insert(0, str(HERE))
import tts as T  # noqa: E402

CER_PASS = 0.10
WHISPER = "openai/whisper-large-v3-turbo"
SELECTOR = "openai/whisper-small"
SPK_MODEL = "microsoft/wavlm-base-plus-sv"
VERIFY_TRIES = 3


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


def _asr(model: str):
    import librosa
    import torch
    from transformers import pipeline

    pipe = pipeline("automatic-speech-recognition", model=model, torch_dtype=torch.float16, device="cuda:0")

    def run(wav, sr) -> str:
        if sr != 16000:  # 파일 이름으로 주면 ffmpeg가 있어야 해서 소리를 직접 넘긴다
            wav = librosa.resample(y=wav, orig_sr=sr, target_sr=16000)
        return pipe({"raw": wav, "sampling_rate": 16000},
                    generate_kwargs={"language": "korean", "task": "transcribe"})["text"].strip()
    return run


def gen(a) -> None:
    import soundfile as sf
    import torch
    from engines import ENGINES

    out = Path(a.out)
    (out / "wav").mkdir(parents=True, exist_ok=True)
    items = json.loads((HERE / "testset.json").read_text(encoding="utf-8"))
    if a.only:  # 그 화자의 대사만(인사 클립을 바꿔 다시 볼 때)
        items = [it for it in items if it["voice"] in a.only.split(",")]
    if a.limit:
        items = items[:a.limit]
    t0 = time.perf_counter()
    eng = ENGINES[a.engine](a.voices, model=a.model, xvec=a.xvec, temp=a.temp, add_punct=a.punct)
    load_s = time.perf_counter() - t0
    select = _asr(SELECTOR) if a.verify else None
    torch.cuda.reset_peak_memory_stats()

    rows, t_all = [], time.perf_counter()
    for it in items:
        t0 = time.perf_counter()
        best, tries = None, 0
        for k in range(VERIFY_TRIES if select else 1):
            wav, sr = eng.say(it["voice"], it["text"], seed=it["id"] + 1000 * k)
            tries += 1
            c = cer(T.speakable(it["text"]), select(wav, sr)) if select else 0.0
            if best is None or c < best[0]:
                best = (c, wav, sr)
            if c <= CER_PASS:
                break
        _, wav, sr = best
        sec = time.perf_counter() - t0
        sf.write(out / "wav" / f"{it['id']:03d}.wav", wav, sr)
        inner = getattr(getattr(eng, "tts", None), "last_tries", 1)  # 길이 한도를 넘어 다시 읽은 횟수(qwen)
        limit = max(T.MIN_SECONDS, len(T.speakable(it["text"])) * T.MAX_SECONDS_PER_CHAR)
        row = {**it, "sec": round(sec, 2), "audio": round(len(wav) / sr, 2), "tries": max(tries, inner),
               "len_ok": len(wav) / sr <= limit}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    meta = {"engine": eng.name, "voices": str(Path(a.voices).resolve()), "xvec_only": a.xvec, "temp": a.temp,
            "punct": a.punct, "verify": a.verify, "load_s": round(load_s, 1),
            "total_sec": round(time.perf_counter() - t_all, 1),
            "torch_peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    (out / "gen.json").write_text(json.dumps({"meta": meta, "rows": rows}, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
    print("[meta] " + json.dumps(meta, ensure_ascii=False), flush=True)


def _speaker_embedder():
    import librosa
    import torch
    from transformers import AutoFeatureExtractor, WavLMForXVector

    fe = AutoFeatureExtractor.from_pretrained(SPK_MODEL)
    model = WavLMForXVector.from_pretrained(SPK_MODEL).to("cuda:0").eval()

    def embed(wav, sr):
        if sr != 16000:
            wav = librosa.resample(y=wav, orig_sr=sr, target_sr=16000)
        x = fe(wav, sampling_rate=16000, return_tensors="pt", padding=True).to("cuda:0")
        with torch.inference_mode():
            return torch.nn.functional.normalize(model(**x).embeddings[0], dim=0)
    return embed


def score(a) -> None:
    import soundfile as sf
    import torch

    out = Path(a.out)
    data = json.loads((out / "gen.json").read_text(encoding="utf-8"))
    voices = Path(a.voices or data["meta"].get("voices") or "")
    if not (voices / "kid-f-01.wav").exists():
        raise SystemExit("인사 클립 폴더를 알 수 없어요. --voices 로 주세요")
    asr, embed = _asr(WHISPER), _speaker_embedder()
    ids = sorted(p.stem for p in voices.glob("*.wav"))
    refs = torch.stack([embed(*sf.read(voices / f"{v}.wav", dtype="float32")) for v in ids])
    rows = data["rows"]
    for r in rows:
        wav, sr = sf.read(out / "wav" / f"{r['id']:03d}.wav", dtype="float32")
        hyp = asr(wav, sr)
        r["heard"] = hyp
        # Whisper는 "20분", "PT"처럼 넣은 글 그대로 받아쓰기도 한다. 들려야 하는 말과 넣은 글 중 가까운 쪽으로 센다
        r["cer"] = round(min(cer(r["expect"], hyp), cer(T.speakable(r["text"]), hyp)), 3)
        r["read_ok"] = r["cer"] <= CER_PASS
        sims = (refs @ embed(wav, sr)).tolist()
        top = ids[max(range(len(ids)), key=sims.__getitem__)]
        r.update(sim_own=round(sims[ids.index(r["voice"])], 3), top_voice=top,
                 voice_ok=top == r["voice"], group_ok=top[:-3] == r["voice"][:-3],
                 gender_ok=top.split("-")[1] == r["voice"].split("-")[1])
        r["ok"] = r["read_ok"] and r["len_ok"] and r["gender_ok"]

    def rate(rs, key):
        return round(sum(bool(r[key]) for r in rs) / len(rs), 3)

    def block(rs):
        if not rs:
            return {"n": 0}
        return {"n": len(rs), "read_ok": rate(rs, "read_ok"), "cer_mean": round(sum(r["cer"] for r in rs) / len(rs), 3),
                "voice_ok": rate(rs, "voice_ok"), "group_ok": rate(rs, "group_ok"), "gender_ok": rate(rs, "gender_ok"),
                "sim_own": round(sum(r["sim_own"] for r in rs) / len(rs), 3), "len_ok": rate(rs, "len_ok"),
                "all_ok": rate(rs, "ok"), "retried": sum(r["tries"] > 1 for r in rs),
                "rtf": round(sum(r["sec"] for r in rs) / sum(r["audio"] for r in rs), 2)}

    hard = [r for r in rows if r["cat"].startswith("hard-")]
    summary = {"normal": block([r for r in rows if r not in hard]),  # 합격 판단은 이 120개
               "hard": block(hard), "all": block(rows)}
    for cat in sorted({r["cat"] for r in rows}):
        summary[cat] = block([r for r in rows if r["cat"] == cat])
    by_voice = {v: block([r for r in rows if r["voice"] == v]) for v in sorted({r["voice"] for r in rows})}
    data.update(summary=summary, by_voice=by_voice, scorer={"asr": WHISPER, "speaker": SPK_MODEL})
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
    g.add_argument("--engine", default="qwen")
    g.add_argument("--model", default=T.MODEL)
    g.add_argument("--xvec", action="store_true")
    g.add_argument("--temp", type=float, default=None)
    g.add_argument("--punct", action="store_true", help="끝에 문장 부호가 없으면 마침표를 붙임")
    g.add_argument("--verify", action="store_true", help="작은 받아쓰기 모델로 확인하고 다시 읽기")
    g.add_argument("--only", default=None, help="쉼표로 나눈 화자 id")
    g.add_argument("--limit", type=int, default=0)
    s = sub.add_parser("score")
    s.add_argument("out")
    s.add_argument("--voices", default=None)
    a = ap.parse_args()
    {"gen": gen, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()
