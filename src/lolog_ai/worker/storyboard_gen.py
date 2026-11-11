"""카드(ComicSource) → 4컷 콘티 (spec/storyboard.md, FR-COMIC-02·06).

LLM이 장소·인물 외모·컷 4개를 JSON 스키마에 맞춰 만들고, 코드가 version·comicId·cardId·layout·index·
name을 채운다. storyboard.validate로 검사해서 틀리면 문제 목록을 알려 한 번 다시 묻고, 그래도 틀리면 실패.
화자 키는 카드와 같은 "화자1"…을 쓴다(api-dto.md 콘티).

source 화자에 character(기본 화자·친구 아바타)가 붙어 있으면 그 화자의 외모는 LLM이 정하지 않고
character.appearance를 그대로 쓴다. 콘티에는 characters[].ref로 어떤 그림을 썼는지 남긴다.
"""

from __future__ import annotations

import json
import logging

from ..storyboard import BEATS, MAX_TEXT, validate
from .llm import Llm

log = logging.getLogger(__name__)


class StoryboardInvalid(Exception):
    """다시 물어도 검사를 통과하지 못함. 재시도하지 않고 실패로 알린다."""


SYSTEM = f"""너는 대화 카드로 4컷 만화 콘티를 만든다. 결과는 JSON 하나다.

구성
- 4컷: 1 buildup(상황 소개) → 2 development(전개) → 3 punchline(웃음 포인트) → 4 reaction(반응)
- 카드의 대사와 요약에 있는 일만 그린다. 없는 사건을 지어내지 않는다

인물(characters)
- key는 카드의 화자 키(화자1, 화자2…) 그대로. 카드에 있는 화자만 쓴다
- appearance는 그 이야기에 어울리는 외모를 영어로 짧게 정한다(나이대, 머리, 옷 색). 모든 컷에 그대로 들어간다
- 화자 목록에 "외모 정해짐"이 있는 화자는 그 외모를 그대로 appearance에 쓰고, scene에서도 그 외모에 맞게 부른다

그림 설명(place.prompt, appearance, scene, expression)은 모두 영어
- 이름, 별명, "화자1" 같은 키를 쓰지 않는다. "the woman", "the man", "the boy"처럼 쓴다
- scene은 그 컷만 보고 그릴 수 있게 쓴다. "again", "the same" 같은 앞 컷에 기대는 말을 쓰지 않는다
- 한글을 섞지 않는다

컷의 인물(cast)
- 한 컷에 1~3명. position은 left/center/right 중 하나이고 한 컷에서 겹치지 않는다

대사(dialogue)
- 컷당 0~2개, 인물이 3명인 컷은 1개까지. speaker는 그 컷 cast에 있는 key
- text는 한국어 {MAX_TEXT}자 이하(공백 포함). 원래 대사를 뜻이 남게 줄인다
- 대화에 나온 실명과 주소는 빼거나 "친구", "우리 집"처럼 바꾼다
- sourceLineIds는 근거가 된 카드 대사 id

장소(place)
- text는 한국어 장소 이름. 카드에 장소가 있으면 그대로 쓴다. 없으면 이야기에서 정한다
- prompt는 그 장소의 영어 배경 설명"""


def _schema(keys: list[str]) -> dict:
    s = {"type": "string"}
    en = {"type": "string", "pattern": "^[ -~]+$"}  # 그림 모델용 영어 설명: 출력 단계에서 한글을 막음
    key = {"type": "string", "enum": keys}
    panel = {
        "type": "object",
        "properties": {
            "beat": {"type": "string", "enum": BEATS},
            "scene": en,
            "cast": {"type": "array", "minItems": 1, "maxItems": 3, "items": {
                "type": "object",
                "properties": {"key": key, "position": {"type": "string", "enum": ["left", "center", "right"]},
                               "expression": en},
                "required": ["key", "position", "expression"]}},
            "dialogue": {"type": "array", "maxItems": 2, "items": {
                "type": "object",
                "properties": {"speaker": key, "text": s,
                               "sourceLineIds": {"type": "array", "items": {"type": "integer"}}},
                "required": ["speaker", "text", "sourceLineIds"]}},
        },
        "required": ["beat", "scene", "cast", "dialogue"],
    }
    return {
        "type": "object",
        "properties": {
            "place": {"type": "object", "properties": {"text": s, "prompt": en}, "required": ["text", "prompt"]},
            "characters": {"type": "array", "minItems": 1, "maxItems": len(keys), "items": {
                "type": "object", "properties": {"key": key, "appearance": en},
                "required": ["key", "appearance"]}},
            "panels": {"type": "array", "minItems": 4, "maxItems": 4, "items": panel},
        },
        "required": ["place", "characters", "panels"],
    }


def fixed_looks(src: dict) -> dict[str, dict]:
    """화자 키 → character. 사용자가 그림을 고른 화자만."""
    return {sp["key"]: sp["character"] for sp in src["speakers"] if sp.get("character")}


def card_text(src: dict) -> str:
    names = {sp["key"]: sp.get("name") for sp in src["speakers"]}
    fixed = fixed_looks(src)

    def who(k: str, n: str | None) -> str:
        out = k + (f"(별명 {n})" if n else "")
        return out + (f"[외모 정해짐: {fixed[k]['appearance']}]" if k in fixed else "")

    speakers = ", ".join(who(k, n) for k, n in names.items())
    lines = "\n".join(f"[{l['id']}] {l['speaker']}: {l['text']}" for l in src["lines"])
    return (f"종류: {src['kind']}\n요약: {src['summary']}\n장소: {src.get('place') or '(없음, 정해 줘)'}\n"
            f"화자: {speakers}\n대사:\n{lines}")


def assemble(raw: dict, src: dict) -> dict:
    names = {sp["key"]: sp.get("name") for sp in src["speakers"]}
    fixed = fixed_looks(src)
    place = raw["place"]
    if src.get("place"):
        place = {**place, "text": src["place"]}  # 사용자가 적은 장소는 그대로
    sb = {
        "version": 1, "comicId": src["comicId"], "cardId": src["cardId"], "layout": "vertical4",
        "place": place,
        "characters": [_character(c, names, fixed) for c in raw["characters"]],
        "panels": [],
    }
    for i, p in enumerate(raw["panels"][:4]):
        cast = [{k: v for k, v in c.items() if k != "expression" or v} for c in p["cast"]]
        dialogue = [{k: v for k, v in d.items() if k != "sourceLineIds" or v} for d in p["dialogue"]]
        sb["panels"].append({"index": i + 1, "beat": p["beat"], "scene": p["scene"],
                             "cast": cast, "dialogue": dialogue})
    return sb


def _character(c: dict, names: dict, fixed: dict) -> dict:
    out = {"key": c["key"], "name": names.get(c["key"]), "appearance": c["appearance"]}
    if c["key"] in fixed:  # 고른 그림의 외모 글로 덮어씀(LLM이 바꿔 써도 무시)
        out["appearance"] = fixed[c["key"]]["appearance"]
        out["ref"] = fixed[c["key"]]["ref"]
    return out


def make_storyboard(src: dict, llm: Llm) -> dict:
    keys = [sp["key"] for sp in src["speakers"]] or ["화자1"]
    schema = _schema(keys)
    user = card_text(src)
    errs: list[str] = []
    for attempt in (1, 2, 3):
        msg = user if attempt == 1 else (
            user + "\n\n앞에서 만든 콘티에 아래 문제가 있었다. 고쳐서 다시 만든다.\n- " + "\n- ".join(errs[:15]))
        raw, stat = llm.chat_json(SYSTEM, msg, schema, "storyboard", temperature=0.4, max_tokens=3072)
        sb = assemble(raw, src)
        errs = validate(sb, src)
        log.info("comic %s 콘티 %d번째: 문제 %d개 %s", src["comicId"], attempt, len(errs), stat)
        if not errs:
            return sb
    raise StoryboardInvalid("콘티 검사 실패: " + json.dumps(errs[:5], ensure_ascii=False))
