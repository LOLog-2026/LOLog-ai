"""말풍선 얼굴 피하기 비교 (이미 그린 컷으로, 다시 생성하지 않음).

bubble_method_test.py가 저장한 B 방식 원본 컷(*_bubble_code_p*_raw.png)에 말풍선을 두 방식으로 넣는다.
  before : 선이 적은 곳만 고름 (글자 32px)
  after  : 얼굴 검출로 얼굴을 피하고 꼬리를 화자 얼굴로 (글자 36px)
얼굴 가림은 "말풍선이 얼굴 상자의 10% 이상을 덮은 얼굴 수"로 센다(같은 검출기라 눈으로도 확인한다).

예)  .venv/bin/python scripts/bubble_face_test.py \
        --storyboards experiments/261105-flux-klein-korean-bubble/storyboards/*.json \
        --raw outputs/bubble-method --out outputs/bubble-face
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from PIL import Image, ImageDraw

from lolog_ai.bubble import _overlap, draw_bubbles, place_bubbles
from lolog_ai.faces import detect_faces, match_speakers


def compose(panels: list[Image.Image], gap: int = 16) -> Image.Image:
    s = panels[0].width
    out = Image.new("RGB", (s + 2 * gap, 4 * s + 5 * gap), "white")
    for i, p in enumerate(panels):
        out.paste(p, (gap, gap + i * (s + gap)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--storyboards", nargs="+", type=Path, required=True)
    ap.add_argument("--raw", type=Path, default=Path("outputs/bubble-method"))
    ap.add_argument("--out", type=Path, default=Path("outputs/bubble-face"))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    variants = {"before": dict(font_size=32, use_faces=False), "after": dict(font_size=36, use_faces=True)}
    rows = []
    for path in args.storyboards:
        sb = json.loads(path.read_text())
        story = path.stem
        for seed in args.seeds:
            finals = {}
            for name, v in variants.items():
                panels = []
                for panel in sb["panels"]:
                    raw = Image.open(args.raw / f"{story}_s{seed}_bubble_code_p{panel['index']}_raw.png")
                    pos = {c["key"]: c["position"] for c in panel["cast"]}
                    t = time.perf_counter()
                    faces = detect_faces(raw)
                    t_face = time.perf_counter() - t
                    if not panel["dialogue"]:
                        panels.append(raw.convert("RGB"))
                        continue
                    kw = {}
                    if v["use_faces"]:
                        kw = dict(faces=faces, speaker_faces=match_speakers(faces, pos))
                    t = time.perf_counter()
                    placed = place_bubbles(raw, panel["dialogue"], pos, font_size=v["font_size"], **kw)
                    img = draw_bubbles(raw, panel["dialogue"], placed, font_size=v["font_size"])
                    t_place = time.perf_counter() - t
                    covered = sum(any(_overlap((p.x0, p.y0, p.x1, p.y1), f) >= 0.1 for p in placed)
                                  for f in faces)
                    rows.append({"story": story, "seed": seed, "variant": name, "panel": panel["index"],
                                 "cast": len(pos), "faces": len(faces), "faces_covered": covered,
                                 "speakers_matched": len(kw.get("speaker_faces", {})),
                                 "face_s": f"{t_face:.3f}", "place_s": f"{t_place:.2f}"})
                    print(rows[-1])
                    panels.append(img)
                finals[name] = compose(panels)
                finals[name].save(args.out / f"{story}_s{seed}_{name}.png")
            a, b = finals["before"], finals["after"]
            cmp = Image.new("RGB", (a.width + b.width + 24, a.height + 40), "white")
            cmp.paste(a, (0, 40))
            cmp.paste(b, (a.width + 24, 40))
            d = ImageDraw.Draw(cmp)
            d.text((10, 10), "before", fill="black")
            d.text((a.width + 34, 10), "after (faces)", fill="black")
            cmp.save(args.out / f"{story}_s{seed}_compare.png")

    with open(args.out / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
