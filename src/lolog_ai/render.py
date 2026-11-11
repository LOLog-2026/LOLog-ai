"""콘티 → 그림 프롬프트, 컷 → 세로 4칸 합치기 (spec/storyboard.md).

#6 실험의 scripts/bubble_method_test.py에서 "B 코드 말풍선" 방식만 옮겼다. 그림 모델은 글자·말풍선 없이
그리고, 말풍선은 bubble.place_bubbles/draw_bubbles가 그린다.
"""

from __future__ import annotations

import re

from PIL import Image

STYLE = ("Korean webtoon style comic, clean black line art, flat pastel colors, "
         "simple and cute characters")
NO_TEXT = "No text, no letters, no speech bubbles."
WHERE = {"left": "on the left", "center": "in the center", "right": "on the right"}


# "glasses"를 쓰면 안경 없는 참조 인물에게도 안경을 그렸다(#33 3차 s31). 안경은 참조 그림을 따른다
KEEP_REF = "Keep each referenced person's face, hairstyle, accessories and clothes exactly as in the image."
PORTRAIT = ("Character portrait of {look}. Front view, upper body, facing the viewer, gentle smile, "
            "centered, plain white background.")


def ref_keys(panel: dict, with_ref: set[str] | frozenset[str]) -> list[str]:
    """이 컷에 참조 그림으로 넣을 인물 키. 순서가 프롬프트의 image 1, image 2…와 같다."""
    return [c["key"] for c in panel["cast"] if c["key"] in with_ref]


_NEGATED = re.compile(r",?\s*\b(no|without)\s+(glasses|hat|beard|mustache|earrings)\b", re.IGNORECASE)


def positive(appearance: str) -> str:
    """"no glasses"처럼 없는 것을 적은 말을 뺀다. 그림 모델은 부정을 못 읽고 "glasses"만 보고 안경을 그렸다
    (#33 2차: 안경 없는 기본 화자 middle-m-01이 모든 장면에서 안경을 씀. 기본 화자 24명 중 12명이 "no glasses")."""
    return _NEGATED.sub("", appearance).strip(" ,")


def cast_text(sb: dict, panel: dict, with_ref: set[str] | frozenset[str] = frozenset()) -> str:
    """참조 그림이 있는 인물은 'the person from image N (외모 글)'으로 부른다(#ref 벤치: 외모 글을 같이 넣어야
    안경 같은 특징이 참조 없는 인물에게 옮겨 붙는 일이 줄었다)."""
    look = {c["key"]: positive(c["appearance"]) for c in sb["characters"]}
    order = {k: i + 1 for i, k in enumerate(ref_keys(panel, with_ref))}
    parts = []
    for c in panel["cast"]:
        expr = f", {c['expression']}" if c.get("expression") else ""
        who = look[c["key"]]
        if c["key"] in order:
            who = f"the person from image {order[c['key']]} ({who})"
        parts.append(f"{WHERE[c['position']]}: {who}{expr}")
    return "Characters " + "; ".join(parts)


def prompt(sb: dict, panel: dict, with_ref: set[str] | frozenset[str] = frozenset()) -> str:
    keep = f" {KEEP_REF}" if ref_keys(panel, with_ref) else ""
    return (f"{STYLE}. A single comic panel. Setting: {sb['place']['prompt']}. "
            f"{panel['scene']} {cast_text(sb, panel, with_ref)}.{keep} {NO_TEXT}")


_CROWD = re.compile(r"\b(crowd|students|people|group|coworkers|classmates|audience|passengers|customers)\b",
                    re.IGNORECASE)

# 영상 장면(#33): STYLE의 "comic"이 있으면 한 장을 두 칸으로 나눠 그리는 일이 있어 화풍 문장을 따로 둔다
VIDEO_STYLE = ("Korean webtoon style illustration, clean black line art, flat pastel colors, "
               "simple and cute characters")
NO_SIGN_TEXT = ("One single full-frame illustration, not divided into panels, no borders. "
                "Any signs and storefronts are plain without writing. "
                "Phone and computer screens are blank or show only soft color.")


def video_prompt(sb: dict, scene: dict, with_ref: set[str] | frozenset[str] = frozenset()) -> str:
    """영상 장면 하나(video_storyboard). 화자(cast)에 더해 화자가 아닌 이야기 속 인물(extras)도 넣는다."""
    parts = []
    if scene.get("cast"):
        parts.append(cast_text(sb, scene, with_ref).removeprefix("Characters "))
    look = {e["id"]: positive(e["appearance"]) for e in sb.get("extras") or []}
    for e in scene.get("extras") or []:
        expr = f", {e['expression']}" if e.get("expression") else ""
        parts.append(f"{WHERE[e['position']]}: {look[e['id']]}{expr}")
    # 사람 수와 사람을 장면 글보다 먼저 쓴다. 장면 글의 "the man"과 인물 설명을 따로 읽어 같은 사람을 둘로
    # 그리는 일이 많았다(#33 3차 B3 같은 사람 둘 10장)
    n = len(parts)
    crowd = bool(_CROWD.search(scene["scene"]))
    if not parts:
        who = "No people in this picture."
    else:
        count = "one person" if n == 1 else f"{['two', 'three'][n - 2]} people"
        lead = "The picture shows" if crowd else "The picture shows exactly"
        who = f"{lead} {count}, each appears only once: {'; '.join(parts)}."
    keep = f" {KEEP_REF}" if ref_keys(scene, with_ref) else ""
    return (f"{VIDEO_STYLE}. Setting: {sb['place']['prompt']}. {who} Scene: {scene['scene']}{keep} "
            f"{NO_SIGN_TEXT} {NO_TEXT}")


def portrait_prompt(appearance: str, from_photo: bool = False) -> str:
    """기본 화자·프로필 캐릭터 그림. 프로필 캐릭터는 사진을 image 1로 넣고 그 사람을 이 화풍으로 다시 그린다."""
    look = f"the person from image 1 ({appearance})" if from_photo else appearance
    return f"{STYLE}. {PORTRAIT.format(look=look)} {NO_TEXT}"


def compose(panels: list[Image.Image], gap: int = 16) -> Image.Image:
    """세로 4칸. 컷 768이면 800×3152."""
    s = panels[0].width
    out = Image.new("RGB", (s + 2 * gap, 4 * s + 5 * gap), "white")
    for i, p in enumerate(panels):
        out.paste(p, (gap, gap + i * (s + gap)))
    return out
