"""눈 항목(VIDEO_EVAL_PLAN.md 4장)을 세기 위한 확인 그림. 영상 3개씩 한 장: 장면 그림 + 콘티(인물 수, cast, extras, 장면 글).

    python make_check_sheets.py BATCH_DIR [--chars DEFAULT_CHARACTERS_DIR] [--font FONT.ttf]
"""
import argparse
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("batch")
ap.add_argument("--chars", default=None)
ap.add_argument("--font", default=str(Path.home() / ".cache/lolog/fonts/NotoSansKR.ttf"))
a = ap.parse_args()
B = Path(a.batch)
font = ImageFont.truetype(a.font, 13)
ids = sorted(p.name for p in B.iterdir() if (p / "storyboard.json").exists() and (p / "scene_1.png").exists())
S, CAP = 300, 95
for g in range(0, len(ids), 3):
    grp = ids[g:g + 3]
    out = Image.new("RGB", (9 * (S + 6), len(grp) * (S + CAP + 10)), "white")
    d = ImageDraw.Draw(out)
    for j, c in enumerate(grp):
        sb = json.loads((B / c / "storyboard.json").read_text(encoding="utf-8"))
        y, x = j * (S + CAP + 10), 0
        for ch in sb["characters"]:
            kind, _, rid = (ch.get("ref") or "").partition(":")
            if a.chars and kind == "default":
                out.paste(Image.open(Path(a.chars) / f"{rid}.png").resize((S // 2, S // 2)), (x, y))
                d.text((x, y + S // 2 + 2), f"REF {ch['key']}", fill="red", font=font)
                x += S // 2 + 6
        looks = "\n".join(f"{ch['key']}:{ch.get('role', '?')[0]} {ch['appearance'][:60]}" for ch in sb["characters"])
        extras = ", ".join(e["id"] + ":" + e["appearance"][:30] for e in sb.get("extras", []))
        d.text((x + 4, y + 4), f"{c}\n{looks}\nextras: {extras}", fill="black", font=font)
        for i, s in enumerate(sb["scenes"]):
            X = (i + 1) * (S + 6)
            out.paste(Image.open(B / c / f"scene_{s['index']}.png").resize((S, S)), (X, y))
            n = len(s["cast"]) + len(s.get("extras", []))
            cap = (f"#{s['index']} {'F' if s['flashback'] else '-'} n={n} cast={[q['key'][-1] for q in s['cast']]} "
                   f"ex={[e['id'] for e in s.get('extras', [])]}\n" + "\n".join(textwrap.wrap(s["scene"], 48)[:4]))
            d.text((X + 2, y + S + 2), cap, fill="black", font=font)
    out.save(B / f"check_{g // 3}.png")
    print(B / f"check_{g // 3}.png")
