"""직접 쓴 글(일기·일지·일화)과 블로그 글로 카드 한 장 만들기 (LOLog-docs spec/texts.md 5장).

녹음과 같은 `lolog:detect` 스트림으로 `{ jobId, kind: "text", textId, title, body, place?, situation? }`가 온다.
녹음과 달리 이야기 찾기를 건너뛰고, LLM 한 번으로 글 전체를 카드 한 장(종류·요약·화자·대사)으로 바꾼다.

- 제목·장소·상황은 LLM에 배경으로만 넘긴다. 장소는 백엔드가 글에 적은 값을 그대로 카드 장소(cards.place)로
  저장해 만화·쇼츠 콘티의 배경(Setting)으로 이어지고, 상황은 카드에 저장하지 않는다(texts.md 3장 입력 칸)
- 글 카드는 시간 값이 없어 카드·대사의 startMs·endMs를 null로 보낸다
- 로그에 글 내용을 남기지 않는다(TR-12)
"""

from __future__ import annotations

import logging

from .detect import KINDS, SPEAKERS, SUMMARY_MAX
from .llm import Llm

log = logging.getLogger(__name__)

MAX_LINES = 40  # 녹음 이야기 한 개의 줄 수 상한(detect.MAX_STORY_SEGS)과 맞춘다

TEXT_SYSTEM = f"""너는 사용자가 직접 쓴 글(일기, 일지, 기억나는 일화)을 4컷 만화와 쇼츠 영상의 재료가 될 카드 한 장으로 바꾼다.

입력에는 글의 제목과 본문이 있고, 사용자가 적었다면 장소와 상황(언제, 누구와, 왜 같은 배경)도 있다.
장소와 상황은 사용자가 직접 적은 사실이니 본문과 함께 믿고 쓴다. 본문과 어긋나는 내용을 지어내지 않는다.

이야기 종류(kind)
- 웃김: 농담, 장난, 다 같이 웃을 만한 일
- 황당: 어이없거나 뜻밖인 일
- 추억: 지난 일을 떠올리며 쓴 얘기
- 셋 중 가장 가까운 것을 고른다

요약(summary)
- 카드 제목으로 쓰는 짧은 한국어 명사구, 공백 포함 20자 안팎(30자를 넘으면 잘림). 예: "마트에서 카트를 잃어버린 동생"
- 사람 이름(연예인, 친구 이름 모두)과 주소를 쓰지 않는다. "한 친구", "동생"처럼 쓴다

대사(lines)
- 글을 이야기에 나오는 사람들이 주고받는 대사로 바꾼다. 따옴표 안의 말이나 "~라고 했다" 같은 말은 그 말을 한 사람의 대사로 옮긴다
- 화자1은 글을 쓴 사람(나)이다. 다른 사람은 처음 말하는 순서대로 화자2, 화자3, 화자4로 붙이고 같은 사람은 끝까지 같은 이름을 쓴다. 최대 4명이다
- 직접 말한 대사가 적으면 글쓴이(화자1)가 그 일을 친구에게 들려주듯 말하는 대사로 바꿔 상황 소개 → 전개 → 웃음 포인트 → 반응이 이어지게 한다
- 대사 한 줄은 말풍선 하나에 들어갈 만큼 짧게(40자 안팎) 쓰고, 전체는 {MAX_LINES}줄 이하로 한다. 글이 길면 웃음 포인트와 그 앞뒤 흐름에 필요한 부분만 남긴다
- 사람 이름(연예인 포함)과 주소는 "친구", "동생", "우리 동네"처럼 바꾼다
- 글에 없는 사건을 더하지 않는다"""

TEXT_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": KINDS},
        "summary": {"type": "string", "minLength": 1, "maxLength": SUMMARY_MAX},
        "lines": {"type": "array", "minItems": 1, "maxItems": MAX_LINES, "items": {
            "type": "object",
            "properties": {
                "speaker": {"type": "string", "enum": SPEAKERS},
                "text": {"type": "string", "minLength": 1},
            },
            "required": ["speaker", "text"],
        }},
    },
    "required": ["kind", "summary", "lines"],
}


class TextCardInvalid(ValueError):
    """LLM 결과로 카드를 만들지 못함. 워커가 실패로 세고 다시 시도한다."""


def _clean(v) -> str | None:
    """공백만 있는 값은 비운 것으로 본다(texts.md 결정: 공백만 있는 장소·상황)."""
    if not isinstance(v, str):
        return None
    return v.strip() or None


def text_input(msg: dict) -> str:
    """LLM에 넣을 글. 비어 있는 장소·상황 줄은 넣지 않는다."""
    parts = [f"제목: {_clean(msg.get('title')) or '(없음)'}"]
    if place := _clean(msg.get("place")):
        parts.append(f"장소: {place}")
    if situation := _clean(msg.get("situation")):
        parts.append(f"상황: {situation}")
    parts.append("본문:\n" + (msg.get("body") or "").strip())
    return "\n".join(parts)


def to_text_card(raw: dict) -> dict:
    """LLM 결과 → TextCard(api TextDtos). 화자는 처음 나온 순서대로 화자1부터 다시 붙인다."""
    lines = [l for l in raw.get("lines") or [] if (l.get("text") or "").strip()][:MAX_LINES]
    order: list[str] = []
    for l in lines:
        if l["speaker"] not in order:
            order.append(l["speaker"])
    rename = {old: SPEAKERS[i] for i, old in enumerate(order[: len(SPEAKERS)])}
    out = [{"lineNo": i + 1, "speaker": rename[l["speaker"]], "text": l["text"].strip(),
            "startMs": None, "endMs": None} for i, l in enumerate(lines)]
    return {
        "kind": raw["kind"],
        "summary": (raw.get("summary") or "").strip()[:SUMMARY_MAX],
        "startMs": None,
        "endMs": None,
        "speakers": SPEAKERS[: len(rename)],
        "lines": out,
    }


def check_text_card(card: dict) -> list[str]:
    errs = []
    if card["kind"] not in KINDS:
        errs.append(f"kind {card['kind']}")
    if not 1 <= len(card["summary"]) <= SUMMARY_MAX:
        errs.append(f"summary 길이 {len(card['summary'])}")
    if not 1 <= len(card["speakers"]) <= len(SPEAKERS):
        errs.append(f"speakers {card['speakers']}")
    if not card["lines"]:
        errs.append("대사 없음")
    return errs


def text_card(msg: dict, llm: Llm) -> dict:
    """TextMessage → TextResult `{ jobId, cards: [카드 1장] }`."""
    if not (msg.get("body") or "").strip():
        raise TextCardInvalid("본문이 비어 있음")
    raw, stat = llm.chat_json(TEXT_SYSTEM, text_input(msg), TEXT_SCHEMA, "text_card", max_tokens=4096)
    card = to_text_card(raw)
    errs = check_text_card(card)
    if errs:
        raise TextCardInvalid(f"카드 형식이 틀림: {errs}")
    log.info("job %s 글 카드: 화자 %d명 대사 %d줄, 장소 %s, 상황 %s %s", msg["jobId"], len(card["speakers"]),
             len(card["lines"]), "있음" if _clean(msg.get("place")) else "없음",
             "있음" if _clean(msg.get("situation")) else "없음", stat)
    return {"jobId": msg["jobId"], "cards": [card]}

