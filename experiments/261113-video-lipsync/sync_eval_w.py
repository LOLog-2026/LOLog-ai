"""작은 음성 인식 모델(gt_base)의 낱말 시각을 그대로 쓰고, 받아쓰기가 대사와 다르면 지금 방식으로 놓았을 때의 점수(정답은 gt)."""
import json, re, sys, wave
sys.path.insert(0, "../../src"); sys.path.insert(0, ".")
import numpy as np
from lolog_ai import lipsync as L
import aligners
e = []; used = tot = fb = 0
for card in sys.argv[1:]:
    g0 = json.load(open(f"gt/{card}.json")); g1 = json.load(open(f"gt_base/{card}.json"))
    for i, (a, b) in enumerate(zip(g0, g1)):
        tot += 1
        words = [w for w in re.split(r"\s+", a["text"].strip()) if L.syllables(w)]
        wa = [w for w in a["words"] if L.syllables(w[0])]; wb = [w for w in b["words"] if L.syllables(w[0])]
        if len(wa) != len(words) or any(len(L.syllables(x)) != len(L.syllables(y[0])) for x, y in zip(words, wa)): continue
        used += 1
        with wave.open(f"tts/{card}/l{i}.wav") as w: au = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        v = np.flatnonzero(L.envelope(au, 24000, 30, int(np.ceil(len(au) / 24000 * 30))) > 0.12); onset = v[0] / 30 if len(v) else 0
        ok = len(wb) == len(words) and all(len(L.syllables(x)) == len(L.syllables(y[0])) for x, y in zip(words, wb))
        if ok:
            pred = [max(y[1], onset) if j == 0 else y[1] for j, y in enumerate(wb)]
        else:
            fb += 1; st = aligners.uniform(a["text"], au, 24000, 30); k = 0; pred = []
            for x in words: pred.append(st[k]); k += len(L.syllables(x))
        e += [p - (max(y[1], onset) if j == 0 else y[1]) for j, (p, y) in enumerate(zip(pred, wa))]
e = np.array(e) * 1000
print(json.dumps({"method": "whisper-base+fallback", "lines_used": used, "lines": tot, "fallback_lines": fb, "words": len(e), "mae_ms": round(float(np.abs(e).mean()), 1),
                  "within80": round(float((np.abs(e) <= 80).mean()), 3), "within150": round(float((np.abs(e) <= 150).mean()), 3)}))
