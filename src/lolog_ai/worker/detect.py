"""이야기 찾기와 화자 나누기 (spec/functional-spec.md FR-CARD-01~04).

1) 구간마다 번호를 붙여 `[12] 글` 줄로 LLM에 넣는다. 시간(ms)은 넣지 않는다(토큰 절약, 시간 지어내기 방지)
2) LLM이 이야기마다 종류·요약·시작/끝 줄 번호·끝났는지를 돌려준다
3) 묶음 경계 규칙으로 카드로 만들 이야기와 다음 묶음으로 넘길 이야기를 나눈다
4) 카드로 만들 이야기만 다시 LLM에 보내 대사마다 화자를 붙인다
5) 줄 번호를 startMs/endMs로 바꿔 DetectResult(api-dto.md 5장)를 만든다

앞 묶음에서 이미 카드가 된 이야기 버리기(묶음 경계 ③)는 아직 하지 않는다. 넘겨받은 글은 앞 묶음에서
카드가 되지 않은 부분이라(넘길 시작점이 카드보다 뒤) 겹치는 경우가 드물다. 겹치면 백엔드에서 거르기로 한다.
"""

from __future__ import annotations

import logging

from .llm import Llm

log = logging.getLogger(__name__)

KINDS = ["웃김", "황당", "추억"]
SPEAKERS = ["화자1", "화자2", "화자3", "화자4"]
CARRY_WINDOW_MS = 5 * 60 * 1000

DETECT_SYSTEM = """너는 대화 녹음을 글로 바꾼 기록에서 보관할 만한 이야기를 찾는다.

입력은 `[번호] 글` 줄 목록이다. 폰의 음성 인식 결과라서
- 문장부호가 거의 없고 글자가 10% 넘게 틀린다. 소리가 비슷한 말로 짐작해서 읽는다
- 한 줄에 여러 사람의 말이 섞여 있을 수 있다
- 한두 글자뿐이거나 외국 문자인 줄은 잡음이니 무시한다

이야기 종류
- 웃김: 농담, 장난, 다 같이 웃을 만한 일
- 황당: 어이없거나 뜻밖인 일
- 추억: 지난 일을 떠올리며 나눈 얘기

규칙
- 사람들이 직접 나눈 대화만 본다. TV·유튜브·라디오처럼 틀어 둔 방송 소리로 보이면(진행자 멘트, 코너 소개, 광고, 노래 가사) 이야기로 잡지 않는다
- 평범한 안부, 일정 얘기, 짧은 맞장구만 있는 부분은 이야기가 아니다. 찾을 게 없으면 빈 목록을 돌려준다
- 한 녹음에 이야기가 여럿이면 따로 나눈다
- from, to는 이야기의 첫 줄과 마지막 줄 번호다. 웃음 포인트 앞의 상황 설명(빌드업)부터 넣는다
- 4컷 만화 한 편이 될 크기로 잡는다. 보통 30초~3분, 40줄 이하. 길게 이어지는 대화면 가장 재밌는 부분만 잡는다
- 이야기끼리 줄 번호가 겹치지 않게 한다
- finished는 이야기가 to에서 끝났으면 true, 입력의 마지막 줄까지 이어지고 있어 끝을 알 수 없으면 false
- summary는 무슨 일이었는지 한국어 한두 문장(200자 이하)
- summary에 사람 이름(연예인, 친구 이름 모두)과 주소를 쓰지 않는다. "한 친구", "배우", "동생"처럼 쓴다"""

SPEAKER_SYSTEM = """다음은 대화 녹음을 글로 바꾼 기록 중 한 이야기 부분이다. 문맥으로 대사마다 누가 말했는지 나눈다.

- 한 줄에 여러 사람의 말이 섞여 있으면 대사 단위로 쪼갠다. 쪼갠 대사에도 원래 줄 번호(seg)를 적는다
- 화자는 등장 순서대로 화자1, 화자2…로 붙이고 최대 4명이다. 같은 사람은 끝까지 같은 이름을 쓴다
- 글자 오류는 문맥으로 자연스럽게 고치되 뜻을 바꾸지 않는다. 잡음 줄은 뺀다
- 사람 이름(연예인 포함)과 주소는 "친구", "배우", "우리 동네"처럼 바꾼다"""

DETECT_SCHEMA = {
    "type": "object",
    "properties": {"stories": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": KINDS},
            "summary": {"type": "string"},
            "from": {"type": "integer"},
            "to": {"type": "integer"},
            "finished": {"type": "boolean"},
        },
        "required": ["kind", "summary", "from", "to", "finished"],
    }}},
    "required": ["stories"],
}

SPEAKER_SCHEMA = {
    "type": "object",
    "properties": {"lines": {"type": "array", "minItems": 1, "items": {
        "type": "object",
        "properties": {
            "seg": {"type": "integer"},
            "speaker": {"type": "string", "enum": SPEAKERS},
            "text": {"type": "string"},
        },
        "required": ["seg", "speaker", "text"],
    }}},
    "required": ["lines"],
}


MAX_STORY_SEGS = 40  # 4컷에 비해 너무 긴 이야기는 끝쪽(펀치라인·반응)만 남긴다


def tidy(stories: list[dict]) -> list[dict]:
    """길이 제한과 겹침 정리. 겹치면 짧은 쪽을 버린다."""
    out: list[dict] = []
    for s in stories:
        if s["to"] - s["from"] + 1 > MAX_STORY_SEGS:
            s = {**s, "from": s["to"] - MAX_STORY_SEGS + 1}
        out.append(s)
    out.sort(key=lambda s: s["to"] - s["from"], reverse=True)
    kept: list[dict] = []
    for s in out:
        if all(s["to"] < k["from"] or s["from"] > k["to"] for k in kept):
            kept.append(s)
    return sorted(kept, key=lambda s: s["from"])


def numbered(sents: list[dict], lo: int = 0, hi: int | None = None) -> str:
    hi = len(sents) - 1 if hi is None else hi
    return "\n".join(f"[{i}] {sents[i]['text'].strip()}" for i in range(lo, hi + 1))


def check_card(card: dict) -> list[str]:
    """api-dto.md DetectedCard 제한."""
    errs = []
    if not 1 <= len(card["summary"]) <= 200:
        errs.append(f"summary 길이 {len(card['summary'])}")
    if card["endMs"] <= card["startMs"]:
        errs.append("endMs<=startMs")
    sp = card["speakers"]
    if not 1 <= len(sp) <= 4 or sp != SPEAKERS[: len(sp)]:
        errs.append(f"speakers {sp}")
    if not card["lines"] or any(not l["text"].strip() for l in card["lines"]):
        errs.append("빈 대사")
    return errs


def to_card(story: dict, lines: list[dict], sents: list[dict]) -> dict:
    order: list[str] = []
    for l in lines:
        if l["speaker"] not in order:
            order.append(l["speaker"])
    rename = {old: SPEAKERS[i] for i, old in enumerate(order[: len(SPEAKERS)])}
    per_seg: dict[int, int] = {}
    for l in lines:
        per_seg[l["seg"]] = per_seg.get(l["seg"], 0) + 1
    out = []
    for l in lines:
        if l["speaker"] not in rename:
            continue
        whole = per_seg[l["seg"]] == 1  # 한 구간이 대사 하나면 그 구간의 시간을 그대로 씀
        out.append({
            "lineNo": len(out) + 1,
            "speaker": rename[l["speaker"]],
            "text": l["text"].strip(),
            "startMs": sents[l["seg"]]["startMs"] if whole else None,
            "endMs": sents[l["seg"]]["endMs"] if whole else None,
        })
    return {
        "kind": story["kind"],
        "summary": story["summary"].strip()[:200],
        "startMs": sents[story["from"]]["startMs"],
        "endMs": sents[story["to"]]["endMs"],
        "speakers": [rename[s] for s in order[: len(SPEAKERS)]],
        "lines": out,
    }


def detect(msg: dict, llm: Llm) -> dict:
    """DetectMessage → DetectResult."""
    sents = sorted(msg["sentences"], key=lambda s: s["startMs"])
    is_last = bool(msg["isLast"])
    result = {"jobId": msg["jobId"], "partNo": msg["partNo"], "cards": [], "carryFromMs": None}
    if not sents:
        return result

    det, stat = llm.chat_json(DETECT_SYSTEM, numbered(sents), DETECT_SCHEMA, "detect")
    stories = tidy([s for s in det.get("stories", []) if 0 <= s["from"] <= s["to"] < len(sents)])
    log.info("job %s 이야기 찾기: 구간 %d개 → 이야기 %d개 %s",
             msg["jobId"], len(sents), len(stories), stat)

    chunk_end = max(s["endMs"] for s in sents)
    window = chunk_end - CARRY_WINDOW_MS
    keep, carry = [], []
    for s in stories:
        late = sents[s["to"]]["endMs"] >= window or not s["finished"]
        (carry if late and not is_last else keep).append(s)
    if not is_last:
        result["carryFromMs"] = min([window] + [sents[s["from"]]["startMs"] for s in carry])

    for s in keep:
        res, stat = llm.chat_json(SPEAKER_SYSTEM, numbered(sents, s["from"], s["to"]),
                                  SPEAKER_SCHEMA, "speakers")
        lines = [l for l in res.get("lines", []) if s["from"] <= l["seg"] <= s["to"]]
        if not lines:
            log.warning("job %s 화자 나누기 결과가 비어 이야기 하나를 버림", msg["jobId"])
            continue
        card = to_card(s, lines, sents)
        errs = check_card(card)
        if errs:
            log.warning("job %s 형식이 틀린 카드를 버림: %s", msg["jobId"], errs)
            continue
        result["cards"].append(card)
        log.info("job %s 화자 나누기: 화자 %d명 대사 %d줄 %s",
                 msg["jobId"], len(card["speakers"]), len(card["lines"]), stat)
    return result
