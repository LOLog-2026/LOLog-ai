"""빈 말풍선 이미지에 대사를 넣어 보는 테스트.

예)
  .venv/bin/python scripts/bubble_overlay_test.py \
      --images outputs/flux-klein-bubble-control --out outputs/bubble-overlay
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from lolog_ai.bubble import DEFAULT_FONT, draw_dialogue, find_bubbles

# 이미지 id → 넣을 대사 (말풍선 순서: 왼쪽→오른쪽)
CASES = {
    "b1": [
        ["배고파"],
        ["편의점 갔다가 지갑 놓고 와서 다시 갔어"],
        ["아니 그러니까 내 말은 그게 아니라고 몇 번을 말해!"],
    ],
    "b2": [
        ["너 왜 웃어?", "아무것도 아니야"],
        ["헐 대박", "진짜?"],
        ["내가 그럴 줄 알았어", "오늘 점심 뭐 먹지?"],
    ],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("outputs/bubble-overlay"))
    ap.add_argument("--font", type=Path, default=DEFAULT_FONT)
    ap.add_argument("--debug", action="store_true", help="찾은 말풍선 영역을 빨간 상자로 표시")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for cid, variants in CASES.items():
        for seed in range(3):
            src = args.images / f"{cid}_seed{seed}.png"
            img = Image.open(src)
            bubbles = find_bubbles(img)
            want = len(variants[0])
            status = "OK" if len(bubbles) == want else f"MISMATCH (want {want})"
            print(f"{src.name}: {len(bubbles)} bubbles {status} "
                  + " ".join(f"[{b.x0},{b.y0},{b.x1},{b.y1} fill={b.fill:.2f}]" for b in bubbles))
            for v, texts in enumerate(variants):
                out = draw_dialogue(img, bubbles, texts, args.font)
                if args.debug:
                    d = ImageDraw.Draw(out)
                    for b in bubbles:
                        d.rectangle((b.x0, b.y0, b.x1, b.y1), outline="red", width=3)
                        d.rectangle(b.text_box(), outline="blue", width=2)
                out.save(args.out / f"{cid}_seed{seed}_v{v}.png")


if __name__ == "__main__":
    main()
