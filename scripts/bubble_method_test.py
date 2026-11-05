"""말풍선 만드는 방식 비교: 그림 모델이 빈 말풍선을 그림(A) vs 코드가 말풍선을 그림(B, 팀 명세 TR-07).

콘티 JSON(팀 명세 형식)을 받아 컷마다 따로 그리고 세로 4칸으로 합친다.
  A bubble_model : 프롬프트로 빈 말풍선을 그리게 하고, find_bubbles로 찾아 대사를 넣는다
  B bubble_code  : 글자·말풍선 없이 그리고, place_bubbles로 자리를 골라 말풍선과 대사를 그린다
같은 시드면 두 방식은 프롬프트의 말풍선 문장만 다르다.

예)  HF_HUB_OFFLINE=1 .venv/bin/python scripts/bubble_method_test.py \
        --storyboards experiments/261105-flux-klein-korean-bubble/storyboards/*.json \
        --out outputs/bubble-method
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from diffusers import Flux2KleinPipeline
from PIL import Image, ImageDraw

from lolog_ai.bubble import (POSITION_X, draw_bubbles, draw_dialogue, find_bubbles,
                             place_bubbles)

STYLE = ("Korean webtoon style comic, clean black line art, flat pastel colors, "
         "simple and cute characters")
NO_TEXT = "No text, no letters, no speech bubbles."


def cast_text(sb: dict, panel: dict) -> str:
    look = {c["key"]: c["appearance"] for c in sb["characters"]}
    parts = []
    for c in panel["cast"]:
        expr = f", {c['expression']}" if c.get("expression") else ""
        where = {"left": "on the left", "center": "in the center", "right": "on the right"}[c["position"]]
        parts.append(f"{where}: {look[c['key']]}{expr}")
    return "Characters " + "; ".join(parts)


def bubble_text(panel: dict) -> str:
    pos = {c["key"]: c["position"] for c in panel["cast"]}
    n = len(panel["dialogue"])
    if n == 0:
        return "No text, no letters, no speech bubbles."
    who = [f"the character {pos[d['speaker']]}" if pos[d["speaker"]] == "center"
           else f"the character on the {pos[d['speaker']]}" for d in panel["dialogue"]]
    if n == 1:
        return f"There is one large empty white speech bubble with no text inside, pointing at {who[0]}."
    return (f"There are two large empty white speech bubbles with no text inside, "
            f"one pointing at {who[0]} and one pointing at {who[1]}.")


def prompt(sb: dict, panel: dict, method: str) -> str:
    tail = bubble_text(panel) if method == "bubble_model" else NO_TEXT
    return (f"{STYLE}. A single comic panel. Setting: {sb['place']['prompt']}. "
            f"{panel['scene']} {cast_text(sb, panel)}. {tail}")


def model_bubbles(img: Image.Image, panel: dict) -> tuple[Image.Image, int]:
    """A: 찾은 말풍선(왼쪽→오른쪽)과 대사(화자 위치 왼쪽→오른쪽)를 짝짓는다."""
    if not panel["dialogue"]:
        return img, 0
    pos = {c["key"]: c["position"] for c in panel["cast"]}
    found = sorted(find_bubbles(img), key=lambda b: b.cx)
    order = sorted(panel["dialogue"], key=lambda d: POSITION_X[pos[d["speaker"]]])
    return draw_dialogue(img, found, [d["text"] for d in order]), len(found)


def code_bubbles(img: Image.Image, panel: dict, font_size: int) -> tuple[Image.Image, int]:
    if not panel["dialogue"]:
        return img, 0
    pos = {c["key"]: c["position"] for c in panel["cast"]}
    placed = place_bubbles(img, panel["dialogue"], pos, font_size=font_size)
    return draw_bubbles(img, panel["dialogue"], placed, font_size=font_size), len(placed)


def compose(panels: list[Image.Image], gap: int = 16) -> Image.Image:
    s = panels[0].width
    out = Image.new("RGB", (s + 2 * gap, 4 * s + 5 * gap), "white")
    for i, p in enumerate(panels):
        out.paste(p, (gap, gap + i * (s + gap)))
    return out


def side_by_side(a: Image.Image, b: Image.Image, labels=("A model bubble", "B code bubble")) -> Image.Image:
    out = Image.new("RGB", (a.width + b.width + 24, a.height + 40), "white")
    out.paste(a, (0, 40))
    out.paste(b, (a.width + 24, 40))
    d = ImageDraw.Draw(out)
    d.text((10, 10), labels[0], fill="black")
    d.text((a.width + 34, 10), labels[1], fill="black")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--storyboards", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("outputs/bubble-method"))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--panel-size", type=int, default=768)
    # 세로 4칸(가로 800px)을 폰 1080px로 볼 때 36px(약 12sp) 이상 → 원본 27px 이상. 여유 있게 32
    ap.add_argument("--font-size", type=int, default=32)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B",
                                              torch_dtype=torch.bfloat16).to("cuda")

    def gen(p: str, seed: int) -> tuple[Image.Image, float]:
        torch.cuda.synchronize()
        t = time.perf_counter()
        img = pipe(prompt=p, width=args.panel_size, height=args.panel_size, guidance_scale=1.0,
                   num_inference_steps=4, generator=torch.Generator("cuda").manual_seed(seed)).images[0]
        torch.cuda.synchronize()
        return img, time.perf_counter() - t

    rows = []
    for path in args.storyboards:
        sb = json.loads(path.read_text())
        story = path.stem
        for seed in args.seeds:
            finals = {}
            for method in ("bubble_model", "bubble_code"):
                panels = []
                for panel in sb["panels"]:
                    raw, sec = gen(prompt(sb, panel, method), seed)
                    raw.save(args.out / f"{story}_s{seed}_{method}_p{panel['index']}_raw.png")
                    t = time.perf_counter()
                    if method == "bubble_model":
                        img, got = model_bubbles(raw, panel)
                    else:
                        img, got = code_bubbles(raw, panel, args.font_size)
                    post = time.perf_counter() - t
                    panels.append(img)
                    rows.append({"story": story, "seed": seed, "method": method, "panel": panel["index"],
                                 "want": len(panel["dialogue"]), "got": got,
                                 "gen_s": f"{sec:.2f}", "post_s": f"{post:.2f}"})
                    print(rows[-1])
                finals[method] = compose(panels)
                finals[method].save(args.out / f"{story}_s{seed}_{method}.png")
            side_by_side(finals["bubble_model"], finals["bubble_code"]).save(
                args.out / f"{story}_s{seed}_compare.png")

    with open(args.out / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (args.out / "config.json").write_text(json.dumps(vars(args), default=str, indent=2))


if __name__ == "__main__":
    main()
