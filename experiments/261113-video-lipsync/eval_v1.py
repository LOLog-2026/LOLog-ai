"""서비스에 올라가 있는 코드(v1)로 같은 카드의 립싱크 적용률·FLUX 비율을 잰다(그림은 저장해 둔 것)."""
import glob, os, sys, copy, wave
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, HERE)
os.environ["LIPSYNC_FACE_MODEL"] = os.path.expanduser("~/.cache/lolog/models/mediapipe/face_landmarker.task")
import numpy as np
from PIL import Image
from lolog_ai import lipsync as L
from lolog_ai.video_storyboard import all_lines, timeline
from cards import CARDS
tot = []
for card in sys.argv[2:]:
    sb = copy.deepcopy(CARDS[card]); A = f"{HERE}/assets/{card}"
    durs = []
    for i in range(len(all_lines(sb))):
        with wave.open(f"{HERE}/tts/{card}/l{i}.wav") as w: durs.append(w.getnframes() / w.getframerate())
    timeline(sb, durs)
    images = [Image.open(f"{A}/scene{i + 1}.png").convert("RGB") for i in range(len(sb["scenes"]))]
    ins = {p[:-7]: np.asarray(Image.open(p).convert("RGB")).astype(int) for p in glob.glob(f"{A}/s*_in.png")}
    miss = [0]
    def edit(crop, k):
        a = np.asarray(crop).astype(int); best = min(ins, key=lambda p: np.abs(ins[p] - a).mean())
        p = f"{best}_{k}.png"
        if np.abs(ins[best] - a).mean() >= 2 or not os.path.exists(p):
            miss[0] += 1                      # 저장해 둔 FLUX 그림이 없는 얼굴(v1이 다른 얼굴을 화자로 짝지음): 실패로 센다
            return Image.new("RGB", crop.size, (128, 128, 128))
        return Image.open(p).convert("RGB")
    mouths = L.make_mouths(sb, images, edit)
    lines = [(s, d) for s in sb["scenes"] for d in s["lines"] if d["speaker"] in [c["key"] for c in s["cast"]]]
    mapped = sum(1 for s, d in lines if (fi := L.face_of(s, mouths[s["index"] - 1]["faces"], d["speaker"])) is not None and fi in mouths[s["index"] - 1]["variants"])
    src = [set(v.values()) for mo in mouths for v in mo["source"].values()]
    r = {"card": card, "lip_coverage": round(mapped / len(lines), 3), "flux_face_rate": round(sum("flux" in v for v in src) / max(1, len(src)), 3),
         "mouth_faces": len(src), "drawn_faces": sum("drawn" in v for v in src), "cache_miss": miss[0]}
    print(r); tot.append(r)
print("MEAN", {k: round(float(np.mean([t[k] for t in tot])), 3) for k in ["lip_coverage", "flux_face_rate", "mouth_faces", "drawn_faces"]})
