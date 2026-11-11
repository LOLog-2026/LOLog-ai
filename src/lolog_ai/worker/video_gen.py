"""카드(VideoSource) → 영상 콘티 (#33, spec/video.md 5장).

4컷 콘티(storyboard_gen.py)와 같은 방식이다. LLM이 장소·인물 외모·장면 4~8개를 JSON 스키마에 맞춰 만들고,
코드가 version·videoId·cardId·index·name을 채운 뒤 video_storyboard.validate로 검사한다. 틀리면 문제 목록을
알려 다시 묻고(3번까지), 그래도 틀리면 실패. 시간표(startMs·endMs)는 TTS를 읽은 뒤 timeline()이 채운다.
"""

from __future__ import annotations

import json
import logging

from ..video_storyboard import MAX_LINES, MAX_SCENES, MAX_TEXT, MIN_SCENES, MOTIONS, TRANSITIONS, validate
from .llm import Llm
from .storyboard_gen import StoryboardInvalid, _character, card_text, fixed_looks

log = logging.getLogger(__name__)

SYSTEM = f"""너는 대화 카드로 30초 안팎의 세로 영상(움직이는 만화) 콘티를 만든다. 결과는 JSON 하나다.

구성
- 장면(scenes) {MIN_SCENES}~{MAX_SCENES}개. 상황 소개 → 전개 → 웃음 포인트 → 반응 순서로 이야기가 흐른다
- 카드의 대사와 요약에 있는 일만 그린다. 없는 사건을 지어내지 않는다
- 장면 하나는 그림 한 장이다. 장소나 행동, 표정이 바뀌면 장면을 나눈다

인물(characters)
- key는 카드의 화자 키(화자1, 화자2…) 그대로. 카드에 있는 화자만 쓴다
- appearance는 그 이야기에 어울리는 외모를 영어로 짧게 정한다(성별, 나이대, 머리, 옷 색). 모든 장면에 그대로 들어간다
- 화자 목록에 "외모 정해짐"이 있는 화자는 그 외모를 그대로 appearance에 쓴다

그림 설명(place.prompt, appearance, scene, expression)은 모두 영어
- 이름, 별명, "화자1" 같은 키를 쓰지 않는다. "the woman", "the man", "the boy"처럼 쓴다
- scene은 그 장면만 보고 그릴 수 있게 쓴다. "again", "the same" 같은 앞 장면에 기대는 말을 쓰지 않는다
- 한글을 섞지 않는다
- 그림에 글자가 들어가지 않게 한다. 말한 내용, 간판·표지판의 글, 따옴표 안의 말을 scene에 쓰지 않는다

장면의 인물(cast)
- 한 장면에 1~3명. position은 left/center/right 중 하나이고 한 장면에서 겹치지 않는다

대사(lines): 자막으로 뜨고 목소리로 읽힌다
- 장면당 0~{MAX_LINES}개. speaker는 그 장면 cast에 있는 key
- text는 한국어 {MAX_TEXT}자 이하(공백 포함). 원래 대사의 말투를 살려 줄인다. 카드의 대사 순서를 지킨다
- 소리 내어 읽을 수 있게 쓴다. "ㅋㅋㅋ", "ㅠㅠ" 같은 낱자모와 이모티콘은 쓰지 않는다. 숫자는 "세 시", "오만 원"처럼 한글로 쓴다
- 대화에 나온 실명과 주소는 빼거나 "친구", "우리 집"처럼 바꾼다
- sourceLineIds는 근거가 된 카드 대사 id

움직임(motion): 그 장면 그림이 어떻게 움직이는지
- zoom_in 인물이나 물건에 집중할 때 / zoom_out 상황 전체를 보여 줄 때 / pan_left·pan_right 둘이 주고받을 때
- pan_up 위를 올려다보거나 등장할 때 / shake 놀람·충격·웃음 포인트 / still 조용한 순간

전환(transition): 앞 장면에서 이 장면으로 넘어올 때의 효과. 첫 장면은 fade
- cut 같은 자리에서 바로 이어질 때 / fade 시간이 흐르거나 분위기가 바뀔 때 / slide_left·slide_up 장소가 바뀔 때
- zoom 웃음 포인트로 들어갈 때 / flash 충격·반전
- 같은 효과만 계속 쓰지 않는다

장소(place)
- text는 한국어 장소 이름. 카드에 장소가 있으면 그대로 쓴다. 없으면 이야기에서 정한다
- prompt는 그 장소의 영어 배경 설명"""


def _schema(keys: list[str]) -> dict:
    s = {"type": "string"}
    en = {"type": "string", "pattern": "^[ -~]+$"}  # 그림 모델용 영어 설명: 출력 단계에서 한글을 막음
    key = {"type": "string", "enum": keys}
    scene = {
        "type": "object",
        "properties": {
            "scene": en,
            "cast": {"type": "array", "minItems": 1, "maxItems": 3, "items": {
                "type": "object",
                "properties": {"key": key, "position": {"type": "string", "enum": ["left", "center", "right"]},
                               "expression": en},
                "required": ["key", "position", "expression"]}},
            "lines": {"type": "array", "maxItems": MAX_LINES, "items": {
                "type": "object",
                "properties": {"speaker": key, "text": s,
                               "sourceLineIds": {"type": "array", "items": {"type": "integer"}}},
                "required": ["speaker", "text", "sourceLineIds"]}},
            "motion": {"type": "string", "enum": MOTIONS},
            "transition": {"type": "string", "enum": TRANSITIONS},
        },
        "required": ["scene", "cast", "lines", "motion", "transition"],
    }
    return {
        "type": "object",
        "properties": {
            "place": {"type": "object", "properties": {"text": s, "prompt": en}, "required": ["text", "prompt"]},
            "characters": {"type": "array", "minItems": 1, "maxItems": len(keys), "items": {
                "type": "object", "properties": {"key": key, "appearance": en},
                "required": ["key", "appearance"]}},
            "scenes": {"type": "array", "minItems": MIN_SCENES, "maxItems": MAX_SCENES, "items": scene},
        },
        "required": ["place", "characters", "scenes"],
    }


def assemble(raw: dict, src: dict) -> dict:
    names = {sp["key"]: sp.get("name") for sp in src["speakers"]}
    fixed = fixed_looks(src)
    place = raw["place"]
    if src.get("place"):
        place = {**place, "text": src["place"]}  # 사용자가 적은 장소는 그대로
    sb = {
        "version": 1, "videoId": src.get("videoId"), "cardId": src["cardId"],
        "place": place,
        "characters": [_character(c, names, fixed) for c in raw["characters"]],
        "scenes": [],
    }
    for i, p in enumerate(raw["scenes"][:MAX_SCENES]):
        cast = [{k: v for k, v in c.items() if k != "expression" or v} for c in p["cast"]]
        lines = [{k: v for k, v in d.items() if k != "sourceLineIds" or v} for d in p["lines"]]
        sb["scenes"].append({"index": i + 1, "scene": p["scene"], "cast": cast, "lines": lines,
                             "motion": p["motion"], "transition": "fade" if i == 0 else p["transition"]})
    return sb


def make_video_storyboard(src: dict, llm: Llm) -> dict:
    keys = [sp["key"] for sp in src["speakers"]] or ["화자1"]
    schema = _schema(keys)
    user = card_text(src)
    errs: list[str] = []
    for attempt in (1, 2, 3):
        msg = user if attempt == 1 else (
            user + "\n\n앞에서 만든 콘티에 아래 문제가 있었다. 고쳐서 다시 만든다.\n- " + "\n- ".join(errs[:15]))
        raw, stat = llm.chat_json(SYSTEM, msg, schema, "video_storyboard", temperature=0.4, max_tokens=4096)
        sb = assemble(raw, src)
        errs = validate(sb, src)
        log.info("video %s 콘티 %d번째: 장면 %d개, 문제 %d개 %s", src.get("videoId"), attempt,
                 len(sb["scenes"]), len(errs), stat)
        if not errs:
            return sb
    raise StoryboardInvalid("영상 콘티 검사 실패: " + json.dumps(errs[:5], ensure_ascii=False))
