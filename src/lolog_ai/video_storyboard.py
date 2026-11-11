"""영상 콘티 검사와 시간표 (#33, LOLog-docs spec/video.md 5장).

4컷 콘티(storyboard.py)와 같은 뼈대에 장면 수가 4~8개로 늘고, 장면마다 움직임(motion)과 앞 장면에서
넘어오는 전환 효과(transition)가 붙는다. 대사는 말풍선이 아니라 자막과 TTS로 나간다.

    errors = validate(sb, card)             # 빈 목록이면 통과
    timeline(sb, seconds)                   # 대사별 TTS 길이로 장면·대사의 startMs·endMs를 채운다

표준 라이브러리만 쓴다.
"""

from __future__ import annotations

from typing import Any

import re

from .storyboard import HANGUL_RE, INTERNAL_NAME_RE, KEY_RE

QUOTE_RE = re.compile(r"[\"“”‘’()]|(?<![A-Za-z])'|'(?![A-Za-z])")  # 따옴표(말한 내용)·괄호(이름) → 그림에 글자가 생겼다(#33). man's 같은 소유격은 통과

MIN_SCENES, MAX_SCENES = 4, 8
MAX_LINES = 3        # 장면 하나의 대사 수
MAX_TEXT = 40        # 자막 한 장의 글자 수(공백 포함). 720px 너비에 46px 글씨로 세 줄(줄당 약 14자)
MAX_CAST = 3
MOTIONS = ["zoom_in", "zoom_out", "pan_left", "pan_right", "pan_up", "shake", "still"]
TRANSITIONS = ["cut", "fade", "slide_left", "slide_up", "zoom", "flash"]
SIZE = {"width": 720, "height": 1280, "fps": 30}

LEAD_MS = 350        # 장면이 바뀐 뒤 첫 대사까지
GAP_MS = 250         # 대사 사이
TAIL_MS = 450        # 마지막 대사 뒤
SILENT_SCENE_MS = 2200  # 대사가 없는 장면
TRANSITION_MS = 400  # 전환 효과 길이. 장면 경계를 가운데 두고 겹친다(소리 시간표는 바뀌지 않는다)


def validate(sb: dict[str, Any], card: dict[str, Any] | None = None) -> list[str]:
    errs: list[str] = []
    if sb.get("version") != 1:
        errs.append("version은 1이어야 함")

    chars = sb.get("characters") or []
    keys = [c.get("key") for c in chars]
    if not chars:
        errs.append("characters가 비어 있음")
    if len(set(keys)) != len(keys):
        errs.append("characters의 key가 겹침")
    for c in chars:
        if not KEY_RE.match(str(c.get("key", ""))):
            errs.append(f"character key 형식 오류: {c.get('key')!r}")
        errs += _check_prompt(f"characters[{c.get('key')}].appearance", c.get("appearance", ""), sb)
    errs += _check_prompt("place.prompt", (sb.get("place") or {}).get("prompt", ""), sb)

    scenes = sb.get("scenes") or []
    if not MIN_SCENES <= len(scenes) <= MAX_SCENES:
        errs.append(f"scenes는 {MIN_SCENES}~{MAX_SCENES}개여야 함 (지금 {len(scenes)}개)")
    total_lines = 0
    for i, s in enumerate(scenes):
        where = f"scenes[{i}]"
        if s.get("index") != i + 1:
            errs.append(f"{where}.index는 {i + 1}이어야 함")
        if s.get("motion") not in MOTIONS:
            errs.append(f"{where}.motion은 {MOTIONS} 중 하나")
        if s.get("transition") not in TRANSITIONS:
            errs.append(f"{where}.transition은 {TRANSITIONS} 중 하나")
        errs += _check_prompt(f"{where}.scene", s.get("scene", ""), sb)
        if QUOTE_RE.search(s.get("scene", "")):
            errs.append(f"{where}.scene에 따옴표나 괄호가 있음 (그림에 글자·말풍선이 생김. 말한 내용은 쓰지 않는다)")
        if s.get("flashback"):
            listeners = {c.get("key") for c in chars if c.get("role") == "listener"}
            for c in s.get("cast") or []:
                if c.get("key") in listeners:
                    errs.append(f"{where}는 flashback인데 듣는 사람 {c.get('key')}가 cast에 있음")

        cast = s.get("cast") or []
        cast_keys = [c.get("key") for c in cast]
        if not 1 <= len(cast) <= MAX_CAST:
            errs.append(f"{where}.cast는 1~{MAX_CAST}명")
        if s.get("flashback") is None:
            errs.append(f"{where}.flashback이 없음")
        for k in cast_keys:
            if k not in keys:
                errs.append(f"{where}.cast에 없는 인물 {k!r}")
        positions = [c.get("position") for c in cast]
        if len(set(positions)) != len(positions):
            errs.append(f"{where}.cast의 position이 겹침")

        lines = s.get("lines") or []
        total_lines += len(lines)
        if len(lines) > MAX_LINES:
            errs.append(f"{where}.lines는 최대 {MAX_LINES}개 (장면을 나눈다)")
        for j, d in enumerate(lines):
            text = d.get("text", "")
            if d.get("speaker") not in cast_keys:
                errs.append(f"{where}.lines[{j}] 화자 {d.get('speaker')!r}가 이 장면 cast에 없음")
            if not text.strip():
                errs.append(f"{where}.lines[{j}] 대사가 비어 있음")
            if len(text) > MAX_TEXT:
                errs.append(f"{where}.lines[{j}] 대사가 {len(text)}자 (최대 {MAX_TEXT}자): {text!r}")
            if INTERNAL_NAME_RE.search(text):
                errs.append(f"{where}.lines[{j}] 대사에 내부 이름이 섞임: {text!r}")
    if scenes and total_lines == 0:
        errs.append("대사가 하나도 없음")

    if card is not None:
        card_keys = {s.get("key") for s in card.get("speakers", [])}
        for k in keys:
            if k not in card_keys:
                errs.append(f"카드에 없는 화자 key {k!r}")
    return errs


def _check_prompt(where: str, text: str, sb: dict[str, Any]) -> list[str]:
    errs = []
    if not text.strip():
        return [f"{where}가 비어 있음"]
    if INTERNAL_NAME_RE.search(text):
        errs.append(f"{where}에 내부 이름(화자1, speaker A 등)이 섞임")
    for c in sb.get("characters") or []:
        name = c.get("name")
        if name and name in text:
            errs.append(f"{where}에 별명 {name!r}이 들어감 (외모 설명으로 바꿔야 함)")
    if HANGUL_RE.search(text):
        errs.append(f"{where}에 한글이 섞임 (그림 모델용 설명은 영어)")
    return errs


def all_lines(sb: dict) -> list[dict]:
    """읽는 순서대로 모든 대사(장면 순, 장면 안 순)."""
    return [d for s in sb["scenes"] for d in s.get("lines") or []]


def timeline(sb: dict, seconds: list[float]) -> int:
    """대사별 소리 길이(all_lines 순서, 초. 읽지 않는 대사는 0)로 시간표를 채우고 전체 길이(ms)를 돌려준다.

    읽지 않는 대사("ㅋㅋㅋ"만 있는 줄)는 자막만 1.2초 보여 준다.
    """
    if len(seconds) != len(all_lines(sb)):
        raise ValueError(f"대사 {len(all_lines(sb))}개인데 길이는 {len(seconds)}개")
    it = iter(seconds)
    t = 0
    for s in sb["scenes"]:
        s["startMs"] = t
        lines = s.get("lines") or []
        if not lines:
            t += SILENT_SCENE_MS
        else:
            t += LEAD_MS
            for j, d in enumerate(lines):
                dur = round(next(it) * 1000)
                d["startMs"], d["endMs"], d["spoken"] = t, t + (dur or 1200), dur > 0
                t = d["endMs"] + (GAP_MS if j < len(lines) - 1 else TAIL_MS)
        s["endMs"] = t
    sb["durationMs"] = t
    sb["size"] = dict(SIZE)
    return t
