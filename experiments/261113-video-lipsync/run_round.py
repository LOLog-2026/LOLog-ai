"""한 회차: 카드 하나를 저장해 둔 그림(assets/<카드>)으로 합치고 채점한다(CPU만 씀).
    python run_round.py A r1_base                  # 기준(입모양만)
    python run_round.py A r1_head head=1 blend=1   # 방안 적용
결과: rounds/<이름>/<카드>.mp4, strip.jpg, metrics.json
"""
import glob, io, json, os, sys, time, wave
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src")); sys.path.insert(0, HERE)
os.environ.setdefault("LIPSYNC_FACE_MODEL", os.path.expanduser("~/.cache/lolog/models/mediapipe/face_landmarker.task"))
os.environ.setdefault("ANIMEFACE_MODEL", os.path.expanduser("~/.cache/lolog/models/animeface/model.onnx"))

import imageio_ffmpeg; os.environ.setdefault("FFMPEG", imageio_ffmpeg.get_ffmpeg_exe())
import copy, subprocess
import numpy as np
from PIL import Image
from lolog_ai import lipsync as L
from lolog_ai import video_render as VR
from lolog_ai.video_storyboard import all_lines, timeline
from cards import CARDS

card, tag = sys.argv[1], sys.argv[2]
opt = dict(a.split("=") for a in sys.argv[3:])
head, blend, blink = float(opt.get("head", 0)), opt.get("blend", "0") == "1", opt.get("blink", "0") == "1"
sb = copy.deepcopy(CARDS[card]); A = f"{HERE}/assets/{card}"; out = f"{HERE}/rounds/{tag}"; os.makedirs(out, exist_ok=True)
SR = 24000
clips = []
for i in range(len(all_lines(sb))):
    with wave.open(f"{HERE}/{opt.get('tts', 'tts')}/{card}/l{i}.wav") as w:
        assert w.getframerate() == SR
        clips.append(np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768)
total = timeline(sb, [len(c) / SR for c in clips])
track = np.zeros(int(total / 1000 * SR) + 1, dtype=np.float32)
for d, c in zip(all_lines(sb), clips):
    s0 = int(d["startMs"] / 1000 * SR); track[s0:s0 + len(c)] = c[:len(track) - s0]
buf = io.BytesIO()
with wave.open(buf, "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((track * 32767).astype(np.int16).tobytes())
wav = buf.getvalue(); open(f"{out}/audio.wav", "wb").write(wav)
images = [Image.open(f"{A}/scene{i + 1}.png").convert("RGB") for i in range(len(sb["scenes"]))]
def _load(d):
    return {p[:-7]: np.asarray(Image.open(p).convert("RGB")).astype(int) for p in glob.glob(f"{d}/s*_in.png")}
ins, ins_hair = _load(A), _load(A + opt.get("hairtag", "_p4"))
def edit(crop, k):  # 2호에서 뽑아 둔 FLUX 그림을 찾아 준다(넣은 조각이 같은 것). "X+"는 수염 지키기 글로 뽑은 폴더에서
    src = ins_hair if "+" in k and ins_hair else ins
    k = k.split("+")[0]
    a = np.asarray(crop).astype(int)
    best = min(src, key=lambda p: np.abs(src[p] - a).mean())
    assert np.abs(src[best] - a).mean() < 2, "저장해 둔 조각과 다름(장면이나 얼굴 찾기가 바뀜)"
    return Image.open(f"{best}_{k}.png").convert("RGB") if os.path.exists(f"{best}_{k}.png") else None
L.EYE_KINDS[:] = [k for k in opt.get("eyekinds", "B").split(",") if k]
t0 = time.perf_counter()
mouths = L.make_mouths(sb, images, edit)
t1 = time.perf_counter()
lips = L.plan(sb, wav, mouths, blink=blink, blend=blend, head=head, eyes=opt.get("eyes", "0") == "1",
              brows=opt.get("brows", "0") == "1", nod=float(opt.get("nod", 0)))
t2 = time.perf_counter()
res = VR.render(sb, images, f"{out}/audio.wav", f"{out}/{card}.mp4", lips=lips)
t3 = time.perf_counter()

# ---- 채점
fps = sb["size"]["fps"]
m = {"card": card, "tag": tag, "opt": opt, "video_s": total / 1000}
spoken = [(s, d) for s in sb["scenes"] for d in s["lines"] if d.get("spoken", True)]
in_cast = [(s, d) for s, d in spoken if d["speaker"] in [c["key"] for c in s["cast"]] and not L.mouth_hidden(s, d["speaker"])]
m["hidden_mouth_lines"] = sum(1 for s, d in spoken if L.mouth_hidden(s, d["speaker"]))
corr, closed, switches, speech_frames, mapped = [], [], 0, 0, 0
for (s, d) in in_cast:
    mo, lp = mouths[s["index"] - 1], lips[s["index"] - 1]
    fi = L.face_of(s, mo["faces"], d["speaker"])
    if fi is None or fi not in mo["variants"]:
        continue
    mapped += 1
    f0 = round(d["startMs"] / 1000 * fps); n = int(np.ceil((d["endMs"] - d["startMs"]) / 1000 * fps))
    seq = [lp["frames"].get(f0 + k, {}).get(fi, "X") for k in range(n)]
    clip = track[int(d["startMs"] / 1000 * SR):int(d["endMs"] / 1000 * SR)]
    env = L.envelope(clip, SR, fps, n)
    corr.append(L.openness_corr(seq, clip, SR, fps))
    quiet = env <= 0.12
    if quiet.any(): closed.append(float(np.mean([seq[k] == "X" for k in np.flatnonzero(quiet)])))
    switches += sum(a != b for a, b in zip(seq, seq[1:])); speech_frames += n
src = [v for mo in mouths for d in mo["source"].values() for v in [set(d.values())]]
m["sync_corr"] = round(float(np.mean(corr)), 3) if corr else None
m["closed_in_silence"] = round(float(np.mean(closed)), 3) if closed else None
m["lip_coverage"] = round(mapped / max(1, len(in_cast)), 3)          # 화자가 장면에 있는 대사 중 입이 움직인 비율
m["flux_face_rate"] = round(sum("flux" in v for v in src) / max(1, len(src)), 3)
m["switch_per_s"] = round(switches / max(1, speech_frames) * fps, 2)
# 입 밖 움직임: 말하는 동안 그 얼굴 머리 둘레(입 상자 빼고)에서 앞 프레임과 달라진 정도(0~255 평균)
views = [VR._Scene(s, im, 720, 1280, lips[i], fps) for i, (s, im) in enumerate(zip(sb["scenes"], images))]
mot = []
for (s, d) in in_cast:
    i = s["index"] - 1; mo = mouths[i]; fi = L.face_of(s, mo["faces"], d["speaker"])
    if fi is None or fi not in mo["variants"]: continue
    f = mo["faces"][fi]; W = views[i].img.width
    x0, y0, x1, y1 = [int(v) for v in (max(0, f.pts[1][0] - f.face_w), max(0, f.pts[1][1] - 1.3 * f.face_w), min(W, f.pts[1][0] + f.face_w), min(W, f.pts[1][1] + f.face_w))]
    bx = lips[i]["boxes"][fi]
    prev = None
    for n in range(round(d["startMs"] / 1000 * fps), round(d["endMs"] / 1000 * fps), 2):
        a = np.asarray(views[i]._art(n * 1000 / fps)).astype(np.int16)[y0:y1, x0:x1].copy()
        a[max(0, bx[1] - y0):max(0, bx[3] - y0), max(0, bx[0] - x0):max(0, bx[2] - x0)] = 0
        if prev is not None: mot.append(float(np.abs(a - prev).mean()))
        prev = a
m["outside_motion"] = round(float(np.mean(mot)), 3) if mot else 0.0
m["cpu_s"] = {"mouth_paste": round(t1 - t0, 1), "plan": round(t2 - t1, 1), "render": round(t3 - t2, 1)}
m["eye_faces"] = [{str(k): "".join(sorted(v)) for k, v in mo.get("eyes", {}).items()} for mo in mouths]
m["blinks"] = sum(1 for lp in lips for n, d in lp.get("eye_frames", {}).items() for v in d.values() if v == "B") // 2
m["sources"] = [{str(k): "".join(sorted(set(v.values()))) for k, v in mo["source"].items()} for mo in mouths]
json.dump(m, open(f"{out}/metrics.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(m, ensure_ascii=False))
subprocess.run([os.environ["FFMPEG"], "-y", "-loglevel", "error", "-i", f"{out}/{card}.mp4", "-vf",
                f"fps=3,crop=720:720:0:170,scale=240:240,tile=8x{int(np.ceil(total / 1000 * 3 / 8))}", "-frames:v", "1", f"{out}/strip.jpg"])
