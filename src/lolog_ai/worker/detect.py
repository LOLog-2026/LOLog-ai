"""이야기 찾기와 화자 나누기 (spec/functional-spec.md FR-CARD-01~04).

1) 구간마다 번호를 붙여 `[12] 글` 줄로 LLM에 넣는다. 시간(ms)은 넣지 않는다(토큰 절약, 시간 지어내기 방지)
2) 8분 창(3분 겹침)마다 따로 LLM에 넣어 이야기마다 종류·요약·시작/끝 줄 번호·끝났는지를 받는다.
   한 번에 넣으면 입력 길이와 상관없이 이야기를 9개 안팎만 내서 긴 묶음의 가운데를 놓쳤다(감지 실험 C절)
3) 묶음 경계 규칙으로 카드로 만들 이야기와 다음 묶음으로 넘길 이야기를 나눈다
4) 카드로 만들 이야기마다 그 구간만 다시 보여 주고 "이야기인가"를 1~5점으로 받아 낮은 것을 버린다(D·F절)
5) 남은 이야기만 다시 LLM에 보내 대사마다 화자를 붙인다. JSON은 문법으로 공백 없이 받는다(G절)
6) 줄 번호를 startMs/endMs로 바꿔 DetectResult(api-dto.md 5장)를 만든다

창 길이와 확인 기준은 환경 변수로 끌 수 있다(DETECT_WINDOW_MIN=0, DETECT_VERIFY_MIN=0이면 예전처럼 한 번에, 거르지 않음).

앞 묶음에서 이미 카드가 된 이야기 버리기(묶음 경계 ③)는 아직 하지 않는다. 넘겨받은 글은 앞 묶음에서
카드가 되지 않은 부분이라(넘길 시작점이 카드보다 뒤) 겹치는 경우가 드물다. 겹치면 백엔드에서 거르기로 한다.
"""

from __future__ import annotations

import logging
import os

from .llm import Llm, LlmError

log = logging.getLogger(__name__)

KINDS = ["웃김", "황당", "추억"]
SUMMARY_MAX = 30  # api-dto.md DetectedCard: 1~30자, 카드 제목
SPEAKERS = ["화자1", "화자2", "화자3", "화자4"]
CARRY_WINDOW_MS = 5 * 60 * 1000


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


# 이야기 찾기 창(감지 실험 C절 chunk8): 창 8분, 겹침 3분. 겹침은 이야기 최대 길이(약 3분)보다 길게 둔다. 0이면 한 번에
WINDOW_MS = int(_env_float("DETECT_WINDOW_MIN", 8) * 60 * 1000)
OVERLAP_MS = 3 * 60 * 1000
# 확인 단계(D·F절): 기대 점수가 이 값보다 낮은 후보는 버린다. 0이면 거르지 않음.
# 기대 점수를 못 받으면(서버가 logprobs를 안 줌) 뽑힌 점수 4점 이상을 남긴다
VERIFY_MIN = _env_float("DETECT_VERIFY_MIN", 3.9)
VERIFY_MIN_INT = 4

DETECT_SYSTEM = """너는 대화 녹음을 글로 바꾼 기록에서 보관할 만한 이야기를 찾는다.

입력은 `[번호] 글` 줄 목록이다. 폰의 음성 인식 결과라서
- 문장부호가 거의 없고 글자가 10% 넘게 틀린다. 소리가 비슷한 말로 짐작해서 읽는다
- 한 줄에 여러 사람의 말이 섞여 있을 수 있다
- 한두 글자뿐이거나 외국 문자인 줄은 잡음이니 무시한다

이야기 종류 (하나만 고른다. 위에서부터 맞는 것을 고른다)
- 추억: 몇 년 전 일(어린 시절, 학창 시절, 군대, 예전에 함께 한 일)을 함께 떠올리는 이야기. 웃기거나 뜻밖이어도 오래전 일을 돌아보는 게 중심이면 추억
- 황당: 최근에 겪은 일이 예상과 다르게 흘러가 어이없거나 당황스러운 이야기. 듣는 사람 반응이 "말이 돼", "어이없다" 쪽이면 황당
- 웃김: 최근 일 중에서 누군가의 말·행동·실수가 우스워서 다 같이 웃는 이야기. 반응이 웃음 쪽이면 웃김

규칙
- 사람들이 직접 나눈 대화만 본다. TV·유튜브·라디오처럼 틀어 둔 방송 소리로 보이면(진행자 멘트, 코너 소개, 광고, 노래 가사) 이야기로 잡지 않는다
- 평범한 안부, 일정 얘기, 짧은 맞장구만 있는 부분은 이야기가 아니다. 찾을 게 없으면 빈 목록을 돌려준다
- 한 녹음에 이야기가 여럿이면 따로 나눈다
- from, to는 이야기의 첫 줄과 마지막 줄 번호다. 웃음 포인트 앞의 상황 설명(빌드업)부터 넣는다
- 4컷 만화 한 편이 될 크기로 잡는다. 보통 30초~3분, 40줄 이하. 길게 이어지는 대화면 가장 재밌는 부분만 잡는다
- 이야기끼리 줄 번호가 겹치지 않게 한다
- finished는 이야기가 to에서 끝났으면 true, 입력의 마지막 줄까지 이어지고 있어 끝을 알 수 없으면 false
- summary는 카드 제목으로 쓰는 짧은 한국어 명사구, 공백 포함 20자 안팎(30자를 넘으면 잘림). 예: "버스에서 졸다가 종점까지 간 친구"
- summary에 사람 이름(연예인, 친구 이름 모두)과 주소를 쓰지 않는다. "한 친구", "배우", "동생"처럼 쓴다"""

SPEAKER_SYSTEM = """다음은 대화 녹음을 글로 바꾼 기록 중 한 이야기 부분이다. 문맥으로 대사마다 누가 말했는지 나눈다.

- 한 줄에 여러 사람의 말이 섞여 있으면 대사 단위로 쪼갠다. 쪼갠 대사에도 원래 줄 번호(seg)를 적는다
- 화자는 등장 순서대로 화자1, 화자2…로 붙이고 최대 4명이다. 같은 사람은 끝까지 같은 이름을 쓴다
- 글자 오류는 문맥으로 자연스럽게 고치되 뜻을 바꾸지 않는다. 잡음 줄은 뺀다
- 사람 이름(연예인 포함)과 주소는 "친구", "배우", "우리 동네"처럼 바꾼다"""

VERIFY_SYSTEM = """다음은 친구들 대화를 폰 음성 인식으로 받아 적은 기록 중 한 부분이다. 문장부호가 없고 글자가 자주 틀리니 소리가 비슷한 말로 짐작해서 읽는다.
이 부분이 나중에 4컷 만화 카드로 다시 보고 싶을 만한 "이야기"인지 판단한다.

이야기
- 누군가 겪었거나 벌어진 일이 있고, 웃기거나 어이없거나 오래 기억할 만한 지점이 있다
- 듣는 사람이 웃거나 놀라거나 되묻는 등 반응한다(웃음소리가 없어도 된다)

이야기가 아님
- 무엇을 먹을지, 언제 어디서 만날지, 무엇을 살지 정하는 대화
- 물건이나 정보를 묻고 답하는 대화, 지금 하는 일이나 할 일을 알리는 말, 안부, 날씨
- 맞장구나 감탄, 웃음만 있고 일어난 일이 없는 대화
- TV·유튜브·라디오처럼 틀어 둔 방송 소리(진행자 멘트, 광고, 노래 가사)

score: 1 확실히 이야기가 아님, 2 아마 아님, 3 애매함, 4 아마 이야기, 5 확실한 이야기"""

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {"score": {"type": "integer", "enum": [1, 2, 3, 4, 5]}},
    "required": ["score"],
}

# 화자 나누기 출력을 공백 없는 JSON으로 강제한다. 내용은 SPEAKER_SCHEMA와 같고 출력 토큰이 약 26% 준다(G절 spk-json)
SPEAKER_GRAMMAR = r'''root ::= "{\"lines\":[" line ("," line)* "]}"
line ::= "{\"seg\":" [0-9]+ ",\"speaker\":" spk ",\"text\":" str "}"
spk ::= "\"화자1\"" | "\"화자2\"" | "\"화자3\"" | "\"화자4\""
str ::= "\"" ( [^"\\\x7F\x00-\x1F] | "\\" ["\\/bfnrt] )* "\""
'''

DETECT_SCHEMA = {
    "type": "object",
    "properties": {"stories": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": KINDS},
            "summary": {"type": "string", "minLength": 1, "maxLength": 30},
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
    if not 1 <= len(card["summary"]) <= SUMMARY_MAX:
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
        "summary": story["summary"].strip()[:SUMMARY_MAX],
        "startMs": sents[story["from"]]["startMs"],
        "endMs": sents[story["to"]]["endMs"],
        "speakers": [rename[s] for s in order[: len(SPEAKERS)]],
        "lines": out,
    }


def windows(sents: list[dict]) -> list[tuple[int, int]]:
    """이야기 찾기 창: 구간 번호 (처음, 끝) 목록. 시작 시각으로 WINDOW_MS 창을 OVERLAP_MS씩 겹쳐 민다."""
    if WINDOW_MS <= 0:
        return [(0, len(sents) - 1)]
    step = max(WINDOW_MS - OVERLAP_MS, 60 * 1000)
    t0, t_end = sents[0]["startMs"], sents[-1]["startMs"]
    out, w = [], t0
    while True:
        idx = [i for i, s in enumerate(sents) if w <= s["startMs"] < w + WINDOW_MS]
        if idx and (not out or idx[-1] > out[-1][1]):
            out.append((idx[0], idx[-1]))
        if w + WINDOW_MS > t_end:
            return out
        w += step


def find_stories(sents: list[dict], llm: Llm, job) -> list[dict]:
    """창마다 이야기를 찾아 합친다. 줄 번호는 원래 번호 그대로라 다시 맞출 필요가 없고,
    겹친 곳에서 두 번 잡힌 이야기는 tidy()가 긴 쪽만 남긴다."""
    wins = windows(sents)
    stories: list[dict] = []
    for n, (lo, hi) in enumerate(wins):
        det, stat = llm.chat_json(DETECT_SYSTEM, numbered(sents, lo, hi), DETECT_SCHEMA, "detect")
        found = [s for s in det.get("stories", []) if lo <= s["from"] <= s["to"] <= hi]
        if n < len(wins) - 1:
            # 가운데 창의 끝에서 "안 끝남"은 창이 잘린 것이지 묶음이 잘린 것이 아니다. 다음 창이 겹쳐서 다시 본다
            found = [{**s, "finished": True} for s in found]
        stories += found
        log.info("job %s 이야기 찾기 창 %d/%d(구간 %d~%d): 이야기 %d개 %s",
                 job, n + 1, len(wins), lo, hi, len(found), stat)
    return tidy(stories)


def verify(story: dict, sents: list[dict], llm: Llm, job) -> bool:
    """그 구간만 다시 보여 주고 카드로 남길 이야기인지 점수로 판단한다. 호출이 실패하면 남긴다."""
    if VERIFY_MIN <= 0:
        return True
    try:
        res, stat = llm.chat_json(VERIFY_SYSTEM, numbered(sents, story["from"], story["to"]),
                                  VERIFY_SCHEMA, "verify", max_tokens=64, logprobs=True)
    except LlmError as e:
        log.warning("job %s 확인 단계 실패로 거르지 않고 남김: %s", job, e)
        return True
    escore = stat.get("escore")
    ok = escore >= VERIFY_MIN if escore is not None else res.get("score", 5) >= VERIFY_MIN_INT
    log.info("job %s 확인: 구간 %d~%d 점수 %s 기대 %s → %s", job, story["from"], story["to"],
             res.get("score"), escore, "남김" if ok else "버림")
    return ok


def detect(msg: dict, llm: Llm) -> dict:
    """DetectMessage → DetectResult."""
    sents = sorted(msg["sentences"], key=lambda s: s["startMs"])
    is_last = bool(msg["isLast"])
    job = msg["jobId"]
    result = {"jobId": job, "partNo": msg["partNo"], "cards": [], "carryFromMs": None}
    if not sents:
        return result

    stories = find_stories(sents, llm, job)
    log.info("job %s 이야기 찾기: 구간 %d개 → 이야기 %d개", job, len(sents), len(stories))

    chunk_end = max(s["endMs"] for s in sents)
    window = chunk_end - CARRY_WINDOW_MS
    keep, carry = [], []
    for s in stories:
        late = sents[s["to"]]["endMs"] >= window or not s["finished"]
        (carry if late and not is_last else keep).append(s)
    if not is_last:
        result["carryFromMs"] = min([window] + [sents[s["from"]]["startMs"] for s in carry])

    # 넘길 이야기는 다음 묶음에서 다시 찾으므로 카드로 만들 것만 확인한다
    keep = [s for s in keep if verify(s, sents, llm, job)]

    for s in keep:
        # 화자 나누기 출력이 max_tokens에서 잘리거나 JSON이 깨지면(같은 말 되풀이 등) 이 이야기만 버리고 나머지 카드는 살린다.
        # 예전엔 예외가 detect() 밖으로 나가 녹음 하나의 카드가 통째로 사라졌다(평가셋 짧은 녹음 75개 중 2개).
        # 서버가 안 받는 등 LlmError가 아닌 오류는 그대로 올려 작업을 다시 시도하게 한다
        try:
            res, stat = llm.chat_json(SPEAKER_SYSTEM, numbered(sents, s["from"], s["to"]),
                                      SPEAKER_SCHEMA, "speakers", grammar=SPEAKER_GRAMMAR)
        except LlmError as e:
            log.warning("job %s 화자 나누기 실패로 이야기 하나를 버림(구간 %d~%d): %s",
                        job, s["from"], s["to"], e)
            continue
        lines = [l for l in res.get("lines", []) if s["from"] <= l["seg"] <= s["to"]]
        if not lines:
            log.warning("job %s 화자 나누기 결과가 비어 이야기 하나를 버림", job)
            continue
        card = to_card(s, lines, sents)
        errs = check_card(card)
        if errs:
            log.warning("job %s 형식이 틀린 카드를 버림: %s", job, errs)
            continue
        result["cards"].append(card)
        log.info("job %s 화자 나누기: 화자 %d명 대사 %d줄 %s",
                 job, len(card["speakers"]), len(card["lines"]), stat)
    return result
