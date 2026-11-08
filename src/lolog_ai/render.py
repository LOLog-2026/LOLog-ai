"""콘티 → 그림 프롬프트, 컷 → 세로 4칸 합치기 (spec/storyboard.md).

#6 실험의 scripts/bubble_method_test.py에서 "B 코드 말풍선" 방식만 옮겼다. 그림 모델은 글자·말풍선 없이
그리고, 말풍선은 bubble.place_bubbles/draw_bubbles가 그린다.
"""

from __future__ import annotations

from PIL import Image

STYLE = ("Korean webtoon style comic, clean black line art, flat pastel colors, "
         "simple and cute characters")
NO_TEXT = "No text, no letters, no speech bubbles."
WHERE = {"left": "on the left", "center": "in the center", "right": "on the right"}


def cast_text(sb: dict, panel: dict) -> str:
    look = {c["key"]: c["appearance"] for c in sb["characters"]}
    parts = []
    for c in panel["cast"]:
        expr = f", {c['expression']}" if c.get("expression") else ""
        parts.append(f"{WHERE[c['position']]}: {look[c['key']]}{expr}")
    return "Characters " + "; ".join(parts)


def prompt(sb: dict, panel: dict) -> str:
    return (f"{STYLE}. A single comic panel. Setting: {sb['place']['prompt']}. "
            f"{panel['scene']} {cast_text(sb, panel)}. {NO_TEXT}")


def compose(panels: list[Image.Image], gap: int = 16) -> Image.Image:
    """세로 4칸. 컷 768이면 800×3152."""
    s = panels[0].width
    out = Image.new("RGB", (s + 2 * gap, 4 * s + 5 * gap), "white")
    for i, p in enumerate(panels):
        out.paste(p, (gap, gap + i * (s + gap)))
    return out
