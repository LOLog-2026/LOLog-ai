"""카드 하나로 영상을 끝까지 만들어 본다 (#33). 워커(3호·2호)가 할 일을 단계별로 손으로 돌리는 스크립트다.

    # 1) 3호가 할 일: 콘티(LLM) + 대사 TTS + 시간표 → OUT/storyboard.json, OUT/audio.wav   (qwen-tts venv)
    PYTHONPATH=src python scripts/video_demo.py plan CARD.json OUT --voices VOICES_DIR [--llm http://127.0.0.1:8000/v1]
    # 2) 2호가 할 일: 장면 그림 → OUT/scene_N.png                                          (FLUX venv)
    PYTHONPATH=src python scripts/video_demo.py draw OUT            # --placeholder 면 FLUX 없이 색 판만
    # 3) 2호가 할 일: 합치기 → OUT/video.mp4, OUT/thumb.jpg
    PYTHONPATH=src python scripts/video_demo.py render OUT

CARD.json은 VideoSource 모양(spec/video.md 3장, ComicSource와 같음).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path


def plan(a) -> None:
    import numpy as np
    import soundfile as sf

    from lolog_ai import tts as T
    from lolog_ai.video_storyboard import all_lines, timeline
    from lolog_ai.worker.llm import Llm
    from lolog_ai.worker.video_gen import make_video_storyboard

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    src = json.loads(Path(a.card).read_text(encoding="utf-8"))
    t0 = time.perf_counter()
    sb = make_video_storyboard(src, Llm(a.llm))
    t1 = time.perf_counter()

    tts = T.Tts(a.voices, device=a.device)
    have = tts.voices()
    for i, c in enumerate(sb["characters"]):  # 기본 화자가 붙어 있으면 그 목소리, 아니면 외모로 고른다
        ref = c.get("ref") or ""
        c["voice"] = ref.split(":", 1)[1] if ref.startswith("default:") and ref.split(":", 1)[1] in have \
            else T.voice_for(c["appearance"], seed=int(src["cardId"]) + i, available=have)
    voice = {c["key"]: c["voice"] for c in sb["characters"]}
    clips, sr = [], 24000
    for i, d in enumerate(all_lines(sb)):
        res = tts.say(voice[d["speaker"]], d["text"], seed=i)
        clips.append(None if res is None else res[0])
        if res is not None:
            sr = res[1]
    t2 = time.perf_counter()
    total = timeline(sb, [0.0 if c is None else len(c) / sr for c in clips])
    track = np.zeros(int(total / 1000 * sr) + 1, dtype=np.float32)
    for d, c in zip(all_lines(sb), clips):
        if c is not None:
            s = int(d["startMs"] / 1000 * sr)
            track[s:s + len(c)] = c[:len(track) - s]
    sf.write(out / "audio.wav", track, sr)
    (out / "storyboard.json").write_text(json.dumps(sb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[plan] 장면 {len(sb['scenes'])}개, 대사 {len(clips)}개, {total / 1000:.1f}초 | "
          f"콘티 {t1 - t0:.0f}초, TTS {t2 - t1:.0f}초 | 목소리 {voice}")


def draw(a) -> None:
    out = Path(a.out)
    sb = json.loads((out / "storyboard.json").read_text(encoding="utf-8"))
    if a.placeholder:
        from PIL import Image, ImageDraw
        for s in sb["scenes"]:
            img = Image.new("RGB", (768, 768), (40 + 25 * s["index"], 90, 200 - 20 * s["index"]))
            d = ImageDraw.Draw(img)
            for k in range(0, 768, 96):  # 움직임이 보이게 격자를 그린다
                d.line([(k, 0), (k, 768)], fill="white")
                d.line([(0, k), (768, k)], fill="white")
            d.text((40, 40), f"scene {s['index']}  {s['motion']} / {s['transition']}", fill="white")
            img.save(out / f"scene_{s['index']}.png")
        return
    from PIL import Image as PILImage

    from lolog_ai.worker import draw as D
    refs = {}
    if a.chars:  # 기본 화자 그림 폴더(LOLog-docs assets/default-characters)를 주면 ref가 있는 인물을 참조로 그린다
        for c in sb["characters"]:
            kind, _, rid = (c.get("ref") or "").partition(":")
            if kind == "default" and (Path(a.chars) / f"{rid}.png").exists():
                refs[c["key"]] = PILImage.open(Path(a.chars) / f"{rid}.png")
    t0 = time.perf_counter()
    for s, img in zip(sb["scenes"], D.draw_scenes(sb, seed=a.seed, refs=refs)):
        img.save(out / f"scene_{s['index']}.png")
    print(f"[draw] 장면 {len(sb['scenes'])}장 {time.perf_counter() - t0:.0f}초, 참조 {len(refs)}명")


def render(a) -> None:
    from PIL import Image

    from lolog_ai.video_render import render as R

    out = Path(a.out)
    sb = json.loads((out / "storyboard.json").read_text(encoding="utf-8"))
    images = [Image.open(out / f"scene_{s['index']}.png") for s in sb["scenes"]]
    t0 = time.perf_counter()
    res = R(sb, images, out / "audio.wav", out / "video.mp4", out / "thumb.jpg")
    print(f"[render] {res} {time.perf_counter() - t0:.0f}초 → {out / 'video.mp4'}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("card")
    p.add_argument("out")
    p.add_argument("--voices", required=True)
    p.add_argument("--llm", default="http://127.0.0.1:8000/v1")
    p.add_argument("--device", default=None)
    d = sub.add_parser("draw")
    d.add_argument("out")
    d.add_argument("--placeholder", action="store_true")
    d.add_argument("--seed", type=int, default=33)
    d.add_argument("--chars", default=None, help="기본 화자 그림 폴더(참조로 그림)")
    r = sub.add_parser("render")
    r.add_argument("out")
    a = ap.parse_args()
    {"plan": plan, "draw": draw, "render": render}[a.cmd](a)


if __name__ == "__main__":
    main()
