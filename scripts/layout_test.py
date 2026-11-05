"""컷 배치 비교와 말풍선 대사 최대 길이 측정.

같은 이야기(냉장고 케이크, schemas/storyboard.example.json과 같은 내용)를 세 방식으로 그린다.
  single2x2 : 한 장(1024×1024)에 2×2 4컷을 한 번에 그림
  single4v  : 한 장(512×2048)에 세로 4칸을 한 번에 그림
  panels    : 컷마다 따로(768×768) 그려 코드로 2×2 / 세로 4칸으로 합침
모두 빈 말풍선으로 그리고, 찾은 말풍선마다 "읽을 수 있는 글자 크기"로 몇 자까지 들어가는지 잰다.

예)  HF_HUB_OFFLINE=1 .venv/bin/python scripts/layout_test.py --out outputs/layout-test
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

from lolog_ai.bubble import DEFAULT_FONT, draw_dialogue, find_bubbles, fit_text

STYLE = ("Korean webtoon style comic, clean black line art, flat pastel colors, "
         "simple and cute characters")
WOMAN = "a young woman with long brown hair and a pink sweater"
MAN = "a young man with short black hair and a blue t-shirt"
PLACE = "a small apartment kitchen with a white fridge"
PANELS = [
    (f"{WOMAN} opens the fridge and finds a strawberry cake inside, eyes sparkling", 1,
     ["오 케이크다"]),
    (f"{WOMAN} sits at the table eating the whole cake happily with a fork", 1,
     ["한 입만 먹어야지"]),
    (f"on the left, {MAN} opens the fridge and sees an empty plate, shocked; "
     f"on the right, {WOMAN} looks nervous", 2, ["내 케이크 어디 갔어?", "글쎄..."]),
    (f"on the left, {MAN} glares; on the right, {WOMAN} looks away whistling "
     f"with cream on her cheek", 1, ["볼에 크림 묻었거든"]),
]
EMPTY = {1: "There is one large empty white speech bubble with no text inside.",
         2: "There are two large empty white speech bubbles with no text inside, "
            "one pointing at each character."}
# 길이 측정용 대사(띄어쓰기 포함 실제 말투)
SAMPLE = "아니 진짜 어제 편의점 갔다가 지갑 놓고 와서 다시 갔는데 문 닫았더라 완전 어이없어"


def panel_prompt(scene: str, n: int) -> str:
    return f"{STYLE}. A single comic panel. {PLACE}. {scene}. {EMPTY[n]}"


def strip_prompt(kind: str) -> str:
    grid = ("A four-panel comic page in a 2x2 grid with clear panel borders"
            if kind == "2x2" else
            "A four-panel vertical comic strip, four panels stacked top to bottom with clear borders")
    body = " ".join(f"Panel {i + 1}: {s}. {EMPTY[n]}" for i, (s, n, _) in enumerate(PANELS))
    return f"{STYLE}. {grid}. Setting: {PLACE}. {body}"


def max_chars(bubble, min_font: int) -> int:
    """이 말풍선에 min_font px 이상으로 넣을 수 있는 최대 글자 수(공백 포함)."""
    best = 0
    for n in range(1, len(SAMPLE) + 1):
        font, lines, _ = fit_text(SAMPLE[:n], bubble.text_box(), DEFAULT_FONT, min_size=8)
        if font.size >= min_font:
            best = n
        else:
            break
    return best


def compose(panels: list[Image.Image], kind: str, gap: int = 16) -> Image.Image:
    s = panels[0].width
    if kind == "2x2":
        out = Image.new("RGB", (2 * s + 3 * gap, 2 * s + 3 * gap), "white")
        for i, p in enumerate(panels):
            out.paste(p, (gap + (i % 2) * (s + gap), gap + (i // 2) * (s + gap)))
    else:
        out = Image.new("RGB", (s + 2 * gap, 4 * s + 5 * gap), "white")
        for i, p in enumerate(panels):
            out.paste(p, (gap, gap + i * (s + gap)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("outputs/layout-test"))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--panel-size", type=int, default=768)
    ap.add_argument("--display-width", type=int, default=1080, help="폰에서 보이는 가로 픽셀")
    ap.add_argument("--min-font-display", type=int, default=36,
                    help="폰 화면(display-width) 기준 최소 글자 크기 px. 1080px에서 약 12sp")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B",
                                              torch_dtype=torch.bfloat16).to("cuda")

    def gen(prompt: str, w: int, h: int, seed: int) -> tuple[Image.Image, float]:
        torch.cuda.synchronize()
        t = time.perf_counter()
        img = pipe(prompt=prompt, width=w, height=h, guidance_scale=1.0, num_inference_steps=4,
                   generator=torch.Generator("cuda").manual_seed(seed)).images[0]
        torch.cuda.synchronize()
        return img, time.perf_counter() - t

    rows = []

    def measure(method: str, layout: str, seed: int, final_w: int, bubbles_found: int,
                bubbles_want: int, bubbles: list, sec: float) -> None:
        # 최종 이미지를 폰 가로(display_width)에 맞춰 보여 줄 때 최소 글자 크기를 원본 px로 환산
        min_font = round(args.min_font_display * final_w / args.display_width)
        chars = [max_chars(b, min_font) for b in bubbles]
        rows.append({"method": method, "layout": layout, "seed": seed, "time_s": f"{sec:.1f}",
                     "bubbles": f"{bubbles_found}/{bubbles_want}",
                     "min_font_px": min_font,
                     "max_chars_min": min(chars) if chars else "",
                     "max_chars_median": sorted(chars)[len(chars) // 2] if chars else ""})
        print(rows[-1])

    want_total = sum(n for _, n, _ in PANELS)
    texts = [t for _, _, ts in PANELS for t in ts]

    for seed in args.seeds:
        # 1) 한 장 2x2
        img, sec = gen(strip_prompt("2x2"), 1024, 1024, seed)
        img.save(args.out / f"single2x2_seed{seed}.png")
        bs = find_bubbles(img, min_area_ratio=0.004)
        measure("single", "2x2", seed, 1024, len(bs), want_total, bs, sec)
        draw_dialogue(img, bs, texts).save(args.out / f"single2x2_seed{seed}_text.png")

        # 2) 한 장 세로 4칸
        img, sec = gen(strip_prompt("vertical4"), 512, 2048, seed)
        img.save(args.out / f"single4v_seed{seed}.png")
        bs = find_bubbles(img, min_area_ratio=0.004)
        measure("single", "vertical4", seed, 512, len(bs), want_total, bs, sec)

        # 3) 컷마다 따로 그려 합치기
        panels, all_bs, total, found = [], [], 0.0, 0
        for i, (scene, n, ts) in enumerate(PANELS):
            p, sec = gen(panel_prompt(scene, n), args.panel_size, args.panel_size, seed)
            total += sec
            bs = find_bubbles(p)
            found += min(len(bs), n)
            all_bs += bs[:n]
            p.save(args.out / f"panel{i + 1}_seed{seed}.png")
            panels.append(draw_dialogue(p, bs, ts))
        for layout in ("2x2", "vertical4"):
            final = compose(panels, layout)
            final.save(args.out / f"panels_{layout}_seed{seed}.png")
            # 합친 이미지의 가로 = 2컷(2x2) 또는 1컷(세로) 너비
            final_w = final.width
            measure("panels", layout, seed, final_w, found, want_total, all_bs, total)

    with open(args.out / "layout_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (args.out / "config.json").write_text(json.dumps(vars(args), default=str, indent=2))


if __name__ == "__main__":
    main()
