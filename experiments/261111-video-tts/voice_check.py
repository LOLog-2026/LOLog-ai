"""목소리 채점 방법 자체를 점검한다 (#33). 채점 결과(score.json)가 있는 실행 폴더를 준다.

    python -I voice_check.py OUT --voices VOICES_DIR

화자 인식 모델이 2~3초짜리 인사 클립과 맞는 묶음을 못 찾을 수 있어서, 세 가지를 따로 잰다.
  ref     읽은 소리가 24명 인사 클립 중 같은 묶음과 가장 가까운가 (score.json의 group_ok와 같은 값)
  among   읽은 소리끼리 비교: 그 화자의 다른 대사들 평균과 가장 가까운가 (인사 클립 없이, 화자가 서로 구분되는가)
  pitch   읽은 소리의 음높이 중앙값이 인사 클립의 ±20% 안인가
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_accuracy import _speaker_embedder  # noqa: E402


def f0(wav, sr) -> float:
    import librosa
    f, voiced, _ = librosa.pyin(wav, fmin=70, fmax=600, sr=sr)
    f = f[voiced & ~np.isnan(f)] if f is not None else np.array([])
    return float(np.median(f)) if len(f) else float("nan")


def main() -> None:
    import soundfile as sf
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--voices", required=True)
    a = ap.parse_args()
    out, voices = Path(a.out), Path(a.voices)
    rows = json.loads((out / "score.json").read_text(encoding="utf-8"))["rows"]
    embed = _speaker_embedder()
    ref_f0 = {p.stem: f0(*sf.read(p, dtype="float32")) for p in sorted(voices.glob("*.wav"))}
    embs, pitch_ok, pitch_n = [], 0, 0
    for r in rows:
        wav, sr = sf.read(out / "wav" / f"{r['id']:03d}.wav", dtype="float32")
        embs.append(embed(wav, sr))
        p = f0(wav, sr)
        r["f0"] = p
        if not np.isnan(p):
            pitch_n += 1
            pitch_ok += abs(p / ref_f0[r["voice"]] - 1) <= 0.20
    E = torch.stack(embs)
    vs = sorted({r["voice"] for r in rows})
    among_v = among_g = 0
    for i, r in enumerate(rows):  # 자기 자신을 뺀 화자별 평균과 비교
        best, best_s = None, -2.0
        for v in vs:
            idx = [j for j, q in enumerate(rows) if q["voice"] == v and j != i]
            s = float(torch.nn.functional.normalize(E[idx].mean(0), dim=0) @ E[i])
            if s > best_s:
                best, best_s = v, s
        among_v += best == r["voice"]
        among_g += best[:-3] == r["voice"][:-3]
    n = len(rows)
    res = {"n": n, "ref_voice": round(sum(r["voice_ok"] for r in rows) / n, 3),
           "ref_group": round(sum(r["group_ok"] for r in rows) / n, 3),
           "among_voice": round(among_v / n, 3), "among_group": round(among_g / n, 3),
           "pitch_ok": round(pitch_ok / max(1, pitch_n), 3), "pitch_n": pitch_n}
    print(json.dumps(res))
    (out / "voice_check.json").write_text(json.dumps({"summary": res, "ref_f0": ref_f0,
                                                      "f0": {r["id"]: r["f0"] for r in rows}}, indent=1))


if __name__ == "__main__":
    main()
