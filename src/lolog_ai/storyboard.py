"""4컷 콘티 검사.

LLM이 만든 콘티를 그림 단계로 넘기기 전에 확인한다(기획_v3 "4컷만화 만드는 법" 3. 검사).
콘티 형식의 기준은 LOLog-docs 레포의 spec/storyboard.md, spec/storyboard.schema.json이다.
https://github.com/LOLog-2026/LOLog-docs/blob/main/spec/storyboard.md
형식을 바꿀 때는 이 파일의 규칙(BEATS, MAX_BUBBLES, MAX_TEXT 등)도 같이 고친다.

    errors = validate(storyboard_dict, card_dict)
    if errors: LLM에 errors를 알려 한 번 다시 묻고, 그래도 있으면 실패 처리

jsonschema 없이 표준 라이브러리만 쓴다(Jetson 워커에서도 그대로 사용).
"""

from __future__ import annotations

import re
from typing import Any

BEATS = ["buildup", "development", "punchline", "reaction"]
MAX_BUBBLES = 2
# 인물이 이만큼 이상인 컷은 말풍선 1개까지. 3명 + 대사 2개면 두 번째 말풍선이 다른 인물 옆으로
# 밀려 그 사람 말처럼 보였다(6장 중 4장). 대사 1개면 6/6 화자가 분명했다(#6, #10)
CROWDED_CAST = 3
MAX_BUBBLES_CROWDED = 1
MAX_TEXT = 25  # 말풍선 대사 최대 글자 수(공백 포함). 세로 4칸 실측 최소 27자에서 여유를 둠(#6)
KEY_RE = re.compile(r"^[A-Z]$")
# 그림 설명에 섞이면 안 되는 내부 이름: "화자1", "speaker A", "Speaker 2" 등
INTERNAL_NAME_RE = re.compile(r"화자\s*\d|\bspeaker\s*[A-Z0-9]\b", re.IGNORECASE)
HANGUL_RE = re.compile(r"[가-힣]")


def validate(sb: dict[str, Any], card: dict[str, Any] | None = None) -> list[str]:
    """문제 목록을 돌려준다. 빈 목록이면 통과.

    card를 주면 카드와의 일관성(화자 키, 별명 노출)도 본다. card는 GET /api/cards/{id} 모양.
    """
    errs: list[str] = []

    if sb.get("version") != 1:
        errs.append("version은 1이어야 함")
    if sb.get("layout") not in ("vertical4", "2x2"):
        errs.append("layout은 vertical4 또는 2x2")

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

    place = sb.get("place") or {}
    errs += _check_prompt("place.prompt", place.get("prompt", ""), sb)

    panels = sb.get("panels") or []
    if len(panels) != 4:
        errs.append(f"panels는 정확히 4개여야 함 (지금 {len(panels)}개)")
    for i, p in enumerate(panels):
        where = f"panels[{i}]"
        if p.get("index") != i + 1:
            errs.append(f"{where}.index는 {i + 1}이어야 함")
        if i < len(BEATS) and p.get("beat") != BEATS[i]:
            errs.append(f"{where}.beat는 {BEATS[i]}이어야 함")
        errs += _check_prompt(f"{where}.scene", p.get("scene", ""), sb)

        cast = p.get("cast") or []
        cast_keys = [c.get("key") for c in cast]
        if not cast:
            errs.append(f"{where}.cast가 비어 있음")
        for k in cast_keys:
            if k not in keys:
                errs.append(f"{where}.cast에 없는 인물 {k!r}")
        positions = [c.get("position") for c in cast]
        if len(set(positions)) != len(positions):
            errs.append(f"{where}.cast의 position이 겹침 (말풍선과 화자를 짝지을 수 없음)")

        dialogue = p.get("dialogue") or []
        if len(dialogue) > MAX_BUBBLES:
            errs.append(f"{where}.dialogue는 최대 {MAX_BUBBLES}개")
        elif len(cast) >= CROWDED_CAST and len(dialogue) > MAX_BUBBLES_CROWDED:
            errs.append(f"{where}: 인물이 {len(cast)}명인 컷은 dialogue 최대 {MAX_BUBBLES_CROWDED}개"
                        " (말풍선이 다른 인물 말처럼 보임. 대사를 다른 컷으로 옮기거나 인물을 줄임)")
        for j, d in enumerate(dialogue):
            text = d.get("text", "")
            if d.get("speaker") not in cast_keys:
                errs.append(f"{where}.dialogue[{j}] 화자 {d.get('speaker')!r}가 이 컷 cast에 없음")
            if not text.strip():
                errs.append(f"{where}.dialogue[{j}] 대사가 비어 있음")
            if len(text) > MAX_TEXT:
                errs.append(f"{where}.dialogue[{j}] 대사가 {len(text)}자 (최대 {MAX_TEXT}자): {text!r}")
            if INTERNAL_NAME_RE.search(text):
                errs.append(f"{where}.dialogue[{j}] 대사에 내부 이름이 섞임: {text!r}")

    if card is not None:
        card_keys = {s.get("key") for s in card.get("speakers", [])}
        for k in keys:
            if k not in card_keys:
                errs.append(f"카드에 없는 화자 key {k!r}")
    return errs


def _check_prompt(where: str, text: str, sb: dict[str, Any]) -> list[str]:
    """그림 모델용 설명(영어)에 내부 이름·별명·한글이 섞였는지 본다."""
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
