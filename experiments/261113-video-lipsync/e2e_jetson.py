"""2호 올리기 전 시험: 워커의 video()와 같은 순서(장면 → FLUX 입·눈 → 계획 → mp4)를 카드 하나로 돈다.
    python e2e2.py T
"""
import io, json, logging, os, sys, time, wave
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src")); sys.path.insert(0, HERE)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
import copy
import numpy as np
from lolog_ai import lipsync
from lolog_ai.video_storyboard import all_lines, timeline
from lolog_ai.video_render import render
from lolog_ai.worker.draw import draw_scenes_lips
from cards import CARDS

card = sys.argv[1]; sb = copy.deepcopy(CARDS[card]); out = f"{HERE}/e2e2_{card}"; os.makedirs(out, exist_ok=True)
SR = 24000; clips = []
for i in range(len(all_lines(sb))):
    with wave.open(f"{HERE}/tts/{card}/l{i}.wav") as w:
        clips.append(np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768)
total = timeline(sb, [len(c) / SR for c in clips])
track = np.zeros(int(total / 1000 * SR) + 1, dtype=np.float32)
for d, c in zip(all_lines(sb), clips):
    s = int(d["startMs"] / 1000 * SR); track[s:s + len(c)] = c[:len(track) - s]
buf = io.BytesIO()
with wave.open(buf, "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((track * 32767).astype(np.int16).tobytes())
wav = buf.getvalue(); open(f"{out}/audio.wav", "wb").write(wav)
t0 = time.perf_counter()
images, mouths = draw_scenes_lips(sb, seed=11)
t1 = time.perf_counter()
assert mouths is not None, "입모양 만들기가 실패해서 None"
lips = lipsync.plan(sb, wav, mouths)
t2 = time.perf_counter()
res = render(sb, images, f"{out}/audio.wav", f"{out}/{card}.mp4", lips=lips)
t3 = time.perf_counter()
print(json.dumps({"card": card, "video_s": total / 1000, "draw_and_flux_s": round(t1 - t0), "plan_s": round(t2 - t1, 1), "render_s": round(t3 - t2, 1),
                  "mouth_faces": [len(m["variants"]) for m in mouths], "eye_faces": [len(m.get("eyes") or {}) for m in mouths],
                  "sources": [{str(k): "".join(sorted(set(v.values()))) for k, v in m["source"].items()} for m in mouths],
                  "moves": sum(len(l["moves"]) for l in lips), "eye_frames": sum(len(l["eye_frames"]) for l in lips)}, ensure_ascii=False))
