"""입 싱크를 낱말 시작 시각으로 잰다. 정답은 gt/<카드>.json(음성 인식의 낱말별 시각).
    python sync_eval.py <방법> <카드...>      방법: uniform(지금) | dur | peaks | dp
낱말(띄어쓰기 단위)마다 "우리가 그 낱말의 첫 음절을 놓은 시각 − 실제 시작 시각"을 재서 절댓값 평균(ms)과 80ms 안 비율을 낸다.
"""
import json, os, re, sys, wave
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src")); sys.path.insert(0, HERE)
import numpy as np
from lolog_ai import lipsync as L
import aligners

method, cards = sys.argv[1], sys.argv[2:]
FPS, SR = 30, 24000
errs, used, total = [], 0, 0
for card in cards:
    gt = json.load(open(f"gt/{card}.json"))
    for i, g in enumerate(gt):
        total += 1
        words = [w for w in re.split(r"\s+", g["text"].strip()) if L.syllables(w)]
        gw = [w for w in g["words"] if L.syllables(w[0])]
        if len(words) != len(gw) or any(len(L.syllables(a)) != len(L.syllables(b[0])) for a, b in zip(words, gw)):
            continue                     # 받아쓴 낱말이 대사와 다르면 그 줄은 정답으로 쓰지 않는다
        used += 1
        with wave.open(f"tts/{card}/l{i}.wav") as w:
            a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        starts = aligners.ALIGN[method](g["text"], a, SR, FPS)     # 음절별 시작 시각(초)
        # 음성 인식은 첫 낱말 시작을 늘 0초로 준다(소리가 0.6초에 시작해도). 첫 낱말은 실제 소리가 시작하는 시각으로 바로잡는다
        n30 = int(np.ceil(len(a) / SR * FPS)); v = np.flatnonzero(L.envelope(a, SR, FPS, n30) > 0.12)
        onset = (v[0] / FPS) if len(v) else 0.0
        k = 0
        for j, (wd, (_, t0, _)) in enumerate(zip(words, gw)):
            errs.append(starts[k] - (max(t0, onset) if j == 0 else t0)); k += len(L.syllables(wd))
e = np.array(errs) * 1000
print(json.dumps({"method": method, "cards": len(cards), "lines_used": used, "lines": total, "words": len(e),
                  "mae_ms": round(float(np.abs(e).mean()), 1), "median_ms": round(float(np.median(np.abs(e))), 1),
                  "within80": round(float((np.abs(e) <= 80).mean()), 3), "within150": round(float((np.abs(e) <= 150).mean()), 3),
                  "bias_ms": round(float(e.mean()), 1)}))
