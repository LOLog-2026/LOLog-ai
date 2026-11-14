"""2호에서 돈다: 카드 하나의 장면 그림과, 얼굴마다 FLUX가 다시 그린 조각(입모양 5종 + 추가 편집)을 뽑아 둔다.
붙이기·타이밍·움직임 실험은 이 그림들로 CPU에서 반복한다(2호 워커를 오래 끄지 않으려고).

    python gen_assets.py A [seed]       → assets/A/scene<i>.png, assets/A/s<i>_f<j>_{in,X,M,A,O,E,B,...}.png, meta.json
"""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import logging; logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
from pathlib import Path
from PIL import Image
from lolog_ai import lipsync as L
from lolog_ai.render import VIDEO_STYLE, video_prompt
from lolog_ai.worker import draw as D
from cards import CARDS

name = sys.argv[1]; seed = int(sys.argv[2]) if len(sys.argv) > 2 else 11
sb = CARDS[name]
out = Path("assets") / (name + os.environ.get("ASSET_TAG", "")); out.mkdir(parents=True, exist_ok=True)
KEEP = ("Redraw image 1 exactly as it is: same person, same face, same hair, same clothes, same background, same framing."
        + L.KEEP_LIGHT)
EXTRA = {  # 입 말고 바꿔 볼 것들
    "B": f"{VIDEO_STYLE}. {KEEP} Change only the eyes of the person: both eyes gently closed, as if blinking. Keep the mouth the same. No text.",
    "R": f"{VIDEO_STYLE}. {KEEP} Change only the eyebrows of the person: eyebrows raised a little, as if emphasizing a word. Keep the mouth the same. No text.",
}
EXTRA = {k: v for k, v in EXTRA.items() if k in os.environ.get("EXTRA", "B,R").split(",")}
mouth = L.mouth_prompts(sb)   # 수염 난 인물이 있으면 "X+" 같은 글도 들어 있다
t0 = time.perf_counter()
prompts = [video_prompt(sb, s, set()) for s in sb["scenes"]]
embeds = D.encode(prompts + list(mouth.values()) + list(EXTRA.values()))
emb = dict(zip(list(mouth) + list(EXTRA), embeds[len(prompts):]))
t1 = time.perf_counter()
pipe = D._image_pipe(); t2 = time.perf_counter()
meta = {"card": name, "seed": seed, "faces": [], "times": {}}
images = []
for i, s in enumerate(sb["scenes"]):
    p = out / f"scene{i + 1}.png"
    if p.exists():
        images.append(Image.open(p).convert("RGB"))
    else:
        im = D._render(pipe, embeds[i], seed + i); im.save(p); images.append(im)
t3 = time.perf_counter()
n = 0
for i, (s, im) in enumerate(zip(sb["scenes"], images)):
    im = L.square(im)
    faces = L.find_faces(im)
    sp = L.speaking_faces(s, faces) if s.get("lines") else {}
    people = len(s.get("cast") or []) + len(s.get("extras") or [])
    big = sorted(range(len(faces)), key=lambda j: -faces[j].face_w)[:people]
    for fi in sorted(big):
        f = faces[fi]; box = L.face_box(f, *im.size)
        crop = im.crop(box).resize((L.MOUTH_SIZE, L.MOUTH_SIZE), Image.LANCZOS)
        crop.save(out / f"s{i + 1}_f{fi}_in.png")
        looks = {c.get("key"): c.get("appearance", "") for c in sb["characters"]}
        hairy = {L.face_of(s, faces, c["key"]): L.hair_of(looks.get(c["key"], "")) for c in s.get("cast") or []
                 if L.hair_of(looks.get(c["key"], ""))}
        if os.environ.get("ONLY_HAIRY") and fi not in hairy:
            continue
        kinds = (list(L.FLUX_SHAPES) if fi in sp else []) + ([] if os.environ.get("ONLY_HAIRY") else list(EXTRA))
        for k in kinds:
            p = out / f"s{i + 1}_f{fi}_{k}.png"
            if not p.exists():
                key = f"{k}+{hairy[fi]}" if (fi in hairy and k in L.FLUX_SHAPES) else k
                D._render(pipe, emb[key], seed, [crop], size=L.MOUTH_SIZE).save(p); n += 1
        meta["faces"].append({"scene": i + 1, "face": fi, "speaks": fi in sp, "box": list(box), "kinds": kinds, "hairy": fi in hairy,
                              "face_w": round(f.face_w, 1)})
        print("face", i + 1, fi, "speaks" if fi in sp else "-", kinds, flush=True)
t4 = time.perf_counter()
meta["times"] = {"encode_s": round(t1 - t0), "load_s": round(t2 - t1), "scenes_s": round(t3 - t2), "edits_s": round(t4 - t3), "edits": n}
json.dump(meta, open(out / "meta.json", "w"), ensure_ascii=False, indent=1)
print("DONE", json.dumps(meta["times"]))
