"""카드(VideoSource) → 영상 콘티 (#33, spec/video.md 5장).

4컷 콘티(storyboard_gen.py)와 같은 방식이다. LLM이 장소·인물 외모·장면 4~8개를 JSON 스키마에 맞춰 만들고,
코드가 version·videoId·cardId·index·name을 채운 뒤 video_storyboard.validate로 검사한다. 틀리면 문제 목록을
알려 다시 묻고(3번까지), 그래도 틀리면 실패. 시간표(startMs·endMs)는 TTS를 읽은 뒤 timeline()이 채운다.
"""

from __future__ import annotations

import json
import logging
import re

from ..sounds import BGM, MAX_SFX, NONE, SFX, to_null
from ..video_storyboard import (
    MAX_LINES,
    MAX_SCENES,
    MAX_TEXT,
    MIN_SCENES,
    MOTIONS,
    TRANSITIONS,
    validate,
)
from .llm import Llm, LlmError
from .storyboard_gen import StoryboardInvalid, _character, card_text, fixed_looks

log = logging.getLogger(__name__)

# 영상 길이 목표는 15~40초(VIDEO_EVAL_PLAN A5). 어림 길이(est_seconds)는 실제와 몇 초 어긋나서 여유를 둔다
TARGET_MAX_S = 36.0
BUDGET_LINES, BUDGET_CHARS = 10, 170

SYSTEM = f"""너는 대화 카드로 30초 안팎의 세로 영상(움직이는 만화) 콘티를 만든다. 결과는 JSON 하나다.

구성
- 장면(scenes) {MIN_SCENES}~{MAX_SCENES}개. 상황 소개 → 전개 → 웃음 포인트 → 반응 순서로 이야기가 흐른다
- 카드의 대사와 요약에 있는 일만 그린다. 없는 사건을 지어내지 않는다
- 장면 하나는 그림 한 장이다. 장소나 행동, 표정이 바뀌면 장면을 나눈다
- 대화는 보통 한 사람(teller)이 겪은 일을 다른 사람(listener)에게 들려주는 것이다
- 장면마다 flashback을 정한다. true는 들려주는 일(그때 그 자리)을 그린 장면, false는 지금 대화하는 자리
- flashback 장면에는 그 일을 겪은 화자만 cast에 넣는다. listener는 넣지 않는다. listener가 말하는 대사는 flashback이 아닌 장면에 둔다
- 이야기에 나오지만 화자가 아닌 사람(할머니, 엄마, 기사님, 선생님 등)은 extras에 한 번 정의하고(id는 영어 소문자, appearance는 영어 외모) 나오는 장면의 extras에 넣는다. 화자를 그 사람 대신 그리지 않는다
- 그 일을 겪은 사람이 화자가 아니면(할머니 이야기 등) 그 회상 장면의 cast는 비워도 된다. 그때 대사는 화면 밖에서 들려주는 말(내레이션)이 된다
- 마지막 장면은 지금 대화하는 자리로 돌아와 듣는 사람의 반응을 그려도 된다

인물(characters)
- key는 카드의 화자 키(화자1, 화자2…) 그대로. 카드에 있는 화자만 쓴다
- appearance는 그 이야기에 어울리는 외모를 영어로 짧게 정한다(성별, 나이대, 머리, 옷 색). 모든 장면에 그대로 들어간다
- appearance는 지금 말하고 있는 그 화자 본인의 외모다. 이야기에 나오는 다른 사람(할머니, 엄마 등)의 외모를 화자에게 쓰지 않는다. 화자의 나이대는 말투로 정하고 모르면 20대로 한다
- role은 teller(이야기를 들려주는 사람) 또는 listener(듣고 반응하는 사람)
- gender(f·m)와 age(kid 어린이, young 10~30대, middle 40대 이상)는 그 화자의 목소리를 고르는 데 쓴다. 말투와 내용으로 정하고, "외모 정해짐"이 있으면 그 외모에 맞춘다
- 화자 목록에 "외모 정해짐"이 있는 화자는 그 외모를 그대로 appearance에 쓴다

그림 설명(place.prompt, appearance, scene, expression)은 모두 영어
- 이름, 별명, "화자1" 같은 키를 쓰지 않는다. "the woman", "the man", "the boy"처럼 쓴다
- scene은 그 장면만 보고 그릴 수 있게 쓴다. "again", "the same" 같은 앞 장면에 기대는 말을 쓰지 않는다
- 한글을 섞지 않는다
- 그림에 글자가 들어가지 않게 한다. 말한 내용, 간판·표지판의 글, 따옴표 안의 말을 scene에 쓰지 않는다
- 휴대폰·컴퓨터 화면 속 내용(채팅, 사진, 앱)을 그리지 않는다. 화면을 보는 사람의 표정과 행동을 그린다
- scene에는 그 장면 cast와 extras에 있는 사람만 쓴다. 화자를 부를 때는 그 화자의 appearance와 같은 성별 말(the man, the woman)을 쓴다. 다른 사람(학생, 아이, 손님)이 필요하면 extras에 정의해 넣는다. 여러 사람이 배경에 필요하면 "a crowd"처럼만 쓴다
- extras는 사람이나 동물만. 물건(키오스크, 상자)은 scene에 쓴다

장면의 인물(cast)
- cast는 화자만, 0~3명. extras와 합쳐 한 장면에 3명까지. position은 left/center/right 중 하나이고 cast와 extras를 합쳐 한 장면에서 겹치지 않는다
- flashback이 아닌 장면은 화자가 한 명 이상 나온다

대사(lines): 자막으로 뜨고 목소리로 읽힌다
- 영상 전체 대사는 모두 합쳐 {BUDGET_LINES}개, {BUDGET_CHARS}자(공백 빼고) 안팎으로 한다. 카드 대화가 길면 상황 소개와 웃음 포인트에 필요한 대사만 남긴다
- 장면당 0~{MAX_LINES}개. flashback이 아닌 장면에서는 speaker가 그 장면 cast에 있어야 한다. flashback 장면의 대사는 들려주는 사람의 목소리라 cast에 없어도 된다
- text는 반드시 한국어로, 카드의 대사를 그대로 쓰거나 줄인다. 영어로 옮기지 않는다. {MAX_TEXT}자 이하(공백 포함). 원래 대사의 말투를 살려 줄인다. 카드의 대사 순서를 지킨다
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

소리(sfx, bgm): 영상에 작게 섞이는 효과음과 배경음. 아래 값 중에서만 고른다
- sfx는 장면마다 하나 또는 "none". 그 장면의 행동·반응에 딱 맞는 소리가 있을 때만 고르고, 영상 전체에 {MAX_SFX}개까지. 나머지는 "none"
- bgm은 영상 전체 분위기 하나 또는 "none"
- 효과음: {", ".join(f"{k}({v})" for k, v in SFX.items())}
- 배경음: {", ".join(f"{k}({v})" for k, v in BGM.items())}

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
            "flashback": {"type": "boolean"},
            "scene": en,
            "extras": {"type": "array", "maxItems": 2, "items": {
                "type": "object", "properties": {"id": {"type": "string", "pattern": "^[a-z][a-z0-9_]*$"},
                                                 "position": {"type": "string", "enum": ["left", "center", "right"]},
                                                 "expression": en},
                "required": ["id", "position", "expression"]}},
            "cast": {"type": "array", "minItems": 0, "maxItems": 3, "items": {
                "type": "object",
                "properties": {"key": key, "position": {"type": "string", "enum": ["left", "center", "right"]},
                               "expression": en},
                "required": ["key", "position", "expression"]}},
            "lines": {"type": "array", "maxItems": MAX_LINES, "items": {
                "type": "object",
                "properties": {"speaker": key, "text": {"type": "string", "pattern": "^[^A-Za-z]+$"},
                               "sourceLineIds": {"type": "array", "items": {"type": "integer"}}},
                "required": ["speaker", "text", "sourceLineIds"]}},
            "motion": {"type": "string", "enum": MOTIONS},
            "transition": {"type": "string", "enum": TRANSITIONS},
            "sfx": {"type": "string", "enum": [*SFX, NONE]},
        },
        "required": ["flashback", "scene", "cast", "extras", "lines", "motion", "transition", "sfx"],
    }
    return {
        "type": "object",
        "properties": {
            "place": {"type": "object", "properties": {"text": s, "prompt": en}, "required": ["text", "prompt"]},
            "characters": {"type": "array", "minItems": 1, "maxItems": len(keys), "items": {
                "type": "object", "properties": {"key": key, "appearance": en,
                                                 "role": {"type": "string", "enum": ["teller", "listener"]},
                                                 "gender": {"type": "string", "enum": ["f", "m"]},
                                                 "age": {"type": "string", "enum": ["kid", "young", "middle"]}},
                "required": ["key", "role", "gender", "age", "appearance"]}},
            "extras": {"type": "array", "maxItems": 3, "items": {
                "type": "object", "properties": {"id": {"type": "string", "pattern": "^[a-z][a-z0-9_]*$"},
                                                 "appearance": en},
                "required": ["id", "appearance"]}},
            "scenes": {"type": "array", "minItems": MIN_SCENES, "maxItems": MAX_SCENES, "items": scene},
            "bgm": {"type": "string", "enum": [*BGM, NONE]},
        },
        "required": ["place", "characters", "extras", "scenes", "bgm"],
    }


_QUOTED = re.compile(r"\s*(\"[^\"]*\"|“[^”]*”|‘[^’]*’|(?<![A-Za-z])'[^']*'(?![A-Za-z])|\([^)]*\))")


def clean_scene(text: str) -> str:
    """그림 설명에서 따옴표로 묶은 말과 괄호(이름)를 뺀다. 넣으면 그림에 글자·말풍선이 생긴다(#33).
    LLM이 규칙을 자주 어겨서 다시 묻지 않고 코드가 지운다."""
    out = _QUOTED.sub("", text)
    out = re.sub(r"(,\s*|\band\s+|\bwith\s+)?(a\s+)?(bright\s+|colorful\s+|large\s+|small\s+)?(neon\s+)?"
                 r"(signboards?|signs?|banners?|posters?|menu\s+boards?|menus?|white\s*boards?|black\s*boards?|"
                 r"chalk\s*boards?|bulletin\s+boards?|notice\s+boards?|calendars?)\b"
                 r"(\s+(filled|covered)\s+with\s+\w+(\s+\w+)?)?", "", out, flags=re.IGNORECASE)
    # 문자·메시지·채팅 화면을 그리라고 하면 말풍선과 가짜 글씨가 생겼다(#33 3차 s31 3장)
    out = re.sub(r"(,\s*|\s+)?(showing|shows|with|displaying)?\s*(an?\s+|the\s+)?(text\s+)?(messages?|chat|"
                 r"notifications?)(\s+(screen|window|app|bubble)s?)?\b[^,.]*", "", out, flags=re.IGNORECASE)
    # 글자를 그리라는 말(쪽지·문자·번호판, #33 2차 B1 실패 8장)과 말하는 동사(말풍선을 부름)를 지운다
    out = re.sub(r"(,\s*|\s+)?(with\s+)?(something\s+|words\s+|text\s+)?written\b[^,.]*", "", out, flags=re.IGNORECASE)
    out = re.sub(r"(,\s*|\s+)?(that|which)\s+(reads|says)\b[^,.]*", "", out, flags=re.IGNORECASE)
    out = re.sub(r"(,\s*|\s+)?(the\s+)?(text|message|note|sign|numbers?|screen|words?|letters?)\s+"
                 r"(reads|says|shows|is|are|appears?|flashes)\b[^.]*", "", out, flags=re.IGNORECASE)
    # 무엇이든 "reads"·"labeled"면 글자를 그린다(#33 5차 "The building reads.")
    out = re.sub(r"(,\s*|\s+)?\b(that\s+|which\s+)?(reads|labell?ed|titled|inscribed)\b[^,.]*", "", out,
                 flags=re.IGNORECASE)
    out = re.sub(r"(,\s*|\s+)?(and\s+)?\b(says|saying|said|asks|asking|shouts|whispers|speaks|speaking)\b"
                 r"(\s+(something|softly|quietly|loudly|clearly|to\s+\w+(\s+\w+)?))*", "", out, flags=re.IGNORECASE)
    # 한 장면에 시간·장소가 둘이면 칸이 나뉘어 그려졌다(B2). 문장마다 "then" 앞만 남기고 장면이 바뀌는 문장은 뺀다
    sentences = [t.strip() for t in re.split(r"(?<=[.!?])\s+", out) if t.strip()]
    kept = []
    for t in sentences:
        if re.match(r"(the\s+scene\s+(shifts|changes|cuts)|later|meanwhile|afterwards)\b", t, flags=re.IGNORECASE):
            continue
        t = re.split(r",?\s+(and\s+)?then\b", t, maxsplit=1, flags=re.IGNORECASE)[0].rstrip(" ,.") + "."
        kept.append(t)
    out = " ".join(kept) if kept else out
    out = re.sub(r",?\s+(with|and|of|to|showing|as|while|when)\s*\.", ".", out, flags=re.IGNORECASE)  # 지우고 남은 끝말
    out = re.sub(r"(^|\.\s+)(of|with|and)\s+[^.]*\.", r"\1", out, flags=re.IGNORECASE)
    return re.sub(r"\s+([,.])", r"\1", re.sub(r"\s{2,}", " ", out)).strip()


def _extras(p: dict, defined: list[dict], cast: list[dict]) -> list[dict]:
    """장면의 extras에서 정의되지 않은 id와 자리가 겹치는 사람을 뺀다(한 장면 3명까지)."""
    ids = {e["id"] for e in defined}
    taken = {c["position"] for c in cast}
    out = []
    for e in p.get("extras") or []:
        if e.get("id") in ids and e.get("position") not in taken and len(cast) + len(out) < 3:
            taken.add(e["position"])
            out.append({k: v for k, v in e.items() if k != "expression" or v})
    return out


def _head(appearance: str) -> str:
    """외모 글의 첫 덩이에서 사람을 가리키는 말. "an old woman, white hair, ..." → "old woman" """
    first = re.split(r"[,(]| with | wearing ", appearance.lower())[0]
    words = [w for w in re.findall(r"[a-z]+(?:-[a-z]+)*", first) if w not in ("a", "an", "the", "in", "her", "his", "s")
             and not w.isdigit()]
    return " ".join(words[-2:])


def _fix_flashback(scene: dict, sb: dict, drop_cast: bool = True) -> None:
    """회상 장면에서 LLM이 이야기 속 인물(extras)을 scene 글에만 쓰고 extras에 넣지 않으면, 화자가 그 사람 대신
    그려진다(#33 할머니 이야기). scene 글에 extras의 외모 머리말("old woman")이 있으면 언급 안 된 화자를 빼고
    그 인물을 넣는다. 역할 id("driver")로만 나오면 화자는 두고 그 인물만 더한다."""
    text = scene["scene"].lower()
    looks = {c["key"]: _head(c["appearance"]) for c in sb["characters"]}
    for e in sb["extras"]:
        head = _head(e["appearance"])
        named = re.search(rf"\b{re.escape(e['id'].replace('_', ' '))}\b", text)  # id가 "driver"처럼 역할이면
        if not ((head and head in text) or named) or any(x["id"] == e["id"] for x in scene["extras"]):
            continue
        if drop_cast and head and head in text:  # 그 사람의 외모("old woman")로 부르면 이 장면의 주인공. 언급 안 된 화자는 뺀다
            scene["cast"] = [c for c in scene["cast"] if looks.get(c["key"]) and looks[c["key"]] in text]
        taken = {c["position"] for c in scene["cast"]} | {x["position"] for x in scene["extras"]}
        free = [p for p in ("center", "left", "right") if p not in taken]
        if free and len(scene["cast"]) + len(scene["extras"]) < 3:
            scene["extras"].append({"id": e["id"], "position": free[0]})


POSITIONS = ("left", "right", "center")
_PERSON = re.compile(r"\b(man|woman|boy|girl|male|female|guy|lady|grandma|grandpa|grandmother|grandfather)\b")
_COLOR = re.compile(r"\b(black|white|gray|grey|navy|blue|red|green|yellow|pink|purple|brown|beige|orange|"
                    r"cream|khaki|mint|olive|maroon|denim|striped|plaid|checkered|floral)\b")
_TOP = re.compile(r"\b(hoodie|t-shirt|shirt|sweater|jacket|cardigan|blouse|dress|coat|sweatshirt|blazer|uniform)\b")
TOP_COLORS = ["navy", "mustard yellow", "forest green", "coral", "lavender", "sky blue"]
# 장면 글이 사람을 옷으로 부르는 말("the woman in a purple cardigan"). 1번 묶음은 색, 2번은 옷
_DRESSED = re.compile(r"\b(?:man|woman|boy|girl|person|guy|lady)\s+(?:in|wearing|with)\s+(?:an?\s+|the\s+|his\s+|her\s+)?"
                      rf"(?:{_COLOR.pattern}\s+)?\b(hoodie|t-shirt|shirt|sweater|jacket|cardigan|blouse|dress|coat|"
                      r"sweatshirt|blazer|uniform|jeans|pants|skirt|apron|scarf|cap|hat|suit|vest|overalls)\b")


def fix_appearance(c: dict, i: int) -> str:
    """외모 글을 모든 장면에서 같은 사람으로 그릴 수 있게 다듬는다(#33 12개 카드: 성별이 없거나 옷 색이 없는
    인물은 장면마다 성별·옷이 바뀌었다). 성별 말이 없으면 앞에 붙이고, 윗옷에 색이 없으면 색을 붙인다."""
    a = c.get("appearance", "").strip()
    low = a.lower()
    if not _PERSON.search(low) and c.get("gender") in ("f", "m"):
        noun = {"kid": ("girl", "boy"), "middle": ("middle-aged woman", "middle-aged man")}.get(
            c.get("age"), ("young woman", "young man"))[c["gender"] == "m"]
        a = f"{noun}, {a}"
    top = _TOP.search(a.lower())
    # 윗옷 바로 앞(머리 색과 "blue jeans and a hoodie"의 바지 색은 빼고). 바지 색을 윗옷 색으로 보면 장면마다
    # 후드 색이 바뀌었다(#33 6차 s59, 7차 s82)
    seg = re.split(r",|\band\b|\bwith\b|\bover\b", a.lower()[:top.start()])[-1] if top else ""
    if top and not _COLOR.search(seg):
        a = a[:top.start()] + TOP_COLORS[i % len(TOP_COLORS)] + " " + a[top.start():]
    elif not top:  # "casual clothes"처럼 윗옷이 없으면 장면마다 옷이 바뀌었다(#33 4차 s52)
        a = re.sub(r",?\s*(wearing\s+)?casual\s+(clothes|outfit)", "", a, flags=re.IGNORECASE).rstrip(" ,")
        a += f", wearing a {TOP_COLORS[i % len(TOP_COLORS)]} sweatshirt"
    return a


def split_line(d: dict, limit: int) -> list[dict]:
    """자막 한도를 넘는 대사를 띄어쓰기에서 둘로 나눈다(같은 화자). 나눌 수 없으면 그대로."""
    t = d["text"]
    if len(t) <= limit or " " not in t:
        return [d]
    cuts = [i for i, ch in enumerate(t) if ch == " "]
    i = min(cuts, key=lambda k: abs(k - len(t) / 2))
    a, b = t[:i].strip(), t[i + 1:].strip()
    if max(len(a), len(b)) > limit:
        return [d]
    return [{**d, "text": a}, {**d, "text": b}]


def fix_scene(sc: dict, limit: int, max_lines: int) -> None:
    """LLM이 자주 틀리는 것을 코드가 맞춘다(#33 12개 카드 중 3개가 이것들로 실패).
    대사가 긴 것은 나누고, 지금 장면에서 말하는 화자가 cast에 없으면 넣고, 자리가 겹치면 다시 정한다."""
    lines = []
    for d in sc["lines"]:
        lines += split_line(d, limit) if len(sc["lines"]) + len(lines) < max_lines + 1 else [d]
    sc["lines"] = lines[:max(max_lines, len(sc["lines"]))]
    if not sc["flashback"]:
        keys = [c["key"] for c in sc["cast"]]
        for d in sc["lines"]:
            if d["speaker"] not in keys and len(sc["cast"]) + len(sc["extras"]) < 3:
                sc["cast"].append({"key": d["speaker"], "position": "center"})
                keys.append(d["speaker"])
            elif d["speaker"] not in keys and sc["extras"]:
                sc["extras"].pop()
                sc["cast"].append({"key": d["speaker"], "position": "center"})
                keys.append(d["speaker"])
    people = sc["cast"] + sc["extras"]
    if len({p["position"] for p in people}) != len(people):
        order = POSITIONS if len(people) < 3 else ("left", "center", "right")
        for p, pos in zip(people, order):
            p["position"] = pos


def drop_phantoms(scene: dict, sb: dict) -> None:
    """장면 글에서 그 장면 cast·extras에 없는 사람(종류)을 부르는 문장을 뺀다.
    그 사람이 그림에 생기는 것(B3)을 막고, 콘티를 다시 묻지 않아 완성률과 시간을 지킨다(#33 4차: 다시 묻게 했더니
    완성률 100% → 83%)."""
    from ..video_storyboard import _KINDS, _ROLE, missing_kinds

    pats = {**_KINDS, "p": _ROLE}

    missing = missing_kinds(scene, sb["characters"], sb.get("extras") or [])
    if not missing:
        return
    sentences = [t for t in re.split(r"(?<=[.!?])\s+", scene["scene"]) if t.strip()]
    kept = [t for t in sentences if not any(pats[k].search(t) for k in missing)]
    if kept:
        scene["scene"] = " ".join(kept)
    else:  # 모든 문장이 부르면 그 말만 "person"으로 바꾼다
        out = scene["scene"]
        for k in missing:
            out = pats[k].sub("person", out)
        scene["scene"] = out


def add_dressed(scene: dict, sb: dict) -> None:
    """장면 글이 그 장면에 없는 인물을 그 인물만 입은 옷으로 부르면("the woman in a purple cardigan") 그 인물을
    넣는다. 넣지 않으면 FLUX가 그 옷을 입은 사람을 하나 더 그리거나 있는 화자에게 그 옷을 입혔다(#33 7차 B3·B4,
    L11·L32·s76·s82·s84). 같은 옷을 입은 인물이 둘 이상이거나 이미 장면에 있으면 두고, 회상에는 듣는 사람을 넣지 않는다."""
    text = scene["scene"].lower()
    people = [(c["key"], c["appearance"].lower(), "cast", c.get("role")) for c in sb["characters"]]
    people += [(e["id"], e["appearance"].lower(), "extras", None) for e in sb.get("extras") or []]
    present = {c["key"] for c in scene["cast"]} | {x["id"] for x in scene["extras"]}
    for m in _DRESSED.finditer(text):
        phrase = f"{m.group(1)} {m.group(2)}" if m.group(1) else m.group(2)
        owners = [p for p in people if re.search(rf"\b{re.escape(phrase)}\b", p[1])]
        if len(owners) != 1 or owners[0][0] in present:
            continue
        key, _, kind, role = owners[0]
        if scene["flashback"] and role == "listener":
            continue
        taken = {c["position"] for c in scene["cast"]} | {x["position"] for x in scene["extras"]}
        free = [p for p in ("center", "left", "right") if p not in taken]
        if not free or len(scene["cast"]) + len(scene["extras"]) >= 3:
            return
        scene[kind].append({"key" if kind == "cast" else "id": key, "position": free[0]})
        present.add(key)


def est_seconds(sb: dict) -> float:
    """대사 글자 수(공백 빼고)와 대사 수로 어림한 영상 길이(초). #33 개발용 영상 76개에 맞춘 직선."""
    lines = [d for s in sb["scenes"] for d in s["lines"]]
    chars = sum(len(re.sub(r"\s", "", d["text"])) for d in lines)
    return 2.9 + 0.109 * chars + 1.44 * len(lines)


def trim_lines(sb: dict, limit_s: float = TARGET_MAX_S) -> None:
    """어림 길이가 limit_s를 넘으면 가운데 장면의 대사를 하나씩 뺀다(#33 6차: 긴 대화 카드 3/12가 40초를 넘음).
    첫 장면(상황 소개)과 마지막 두 장면(웃음 포인트·반응)은 건드리지 않는다. 대사가 많은 장면부터, 같으면 긴 대사부터
    뺀다. 대사가 모두 빠진 장면은 장면 수가 최소보다 많을 때 없앤다."""
    while est_seconds(sb) > limit_s:
        mid = sb["scenes"][1:-2]
        cands = [(len(s["lines"]), len(d["text"]), s, d) for s in mid for d in s["lines"]]
        if not cands:
            break
        _, _, s, d = max(cands, key=lambda x: (x[0], x[1]))
        s["lines"].remove(d)
        if not s["lines"] and len(sb["scenes"]) > MIN_SCENES:
            sb["scenes"].remove(s)
    for i, s in enumerate(sb["scenes"]):
        s["index"] = i + 1
        if i == 0:
            s["transition"] = "fade"


def fix_sounds(sb: dict, bgm: str | None) -> None:
    """효과음·배경음 값을 맞춘다. 목록에 없는 값은 지우고, 효과음은 앞에서부터 MAX_SFX개까지,
    바로 앞 장면과 같은 효과음은 지운다(같은 소리가 연달아 나면 어색함). LLM이 넘쳐도 다시 묻지 않는다."""
    sb["bgm"] = bgm if bgm in BGM else None
    kept, prev = 0, None
    for sc in sb["scenes"]:
        v = sc.get("sfx")
        if v not in SFX or v == prev or kept >= MAX_SFX:
            v = None
        sc["sfx"], prev = v, v
        kept += v is not None


def assemble(raw: dict, src: dict) -> dict:
    names = {sp["key"]: sp.get("name") for sp in src["speakers"]}
    fixed = fixed_looks(src)
    place = raw["place"]
    if src.get("place"):
        place = {**place, "text": src["place"]}  # 사용자가 적은 장소는 그대로
    sb = {
        "version": 1, "videoId": src.get("videoId"), "cardId": src["cardId"],
        "place": place,
        "characters": [{**_character(c, names, fixed), "role": c.get("role", "teller"),
                        "gender": c.get("gender"), "age": c.get("age")} for c in raw["characters"]],
        "extras": [{"id": e["id"], "appearance": e["appearance"]} for e in raw.get("extras") or []],
        "scenes": [],
    }
    listeners = {c["key"] for c in raw["characters"] if c.get("role") == "listener"}
    for i, p in enumerate(raw["scenes"][:MAX_SCENES]):
        cast = [{k: v for k, v in c.items() if k != "expression" or v} for c in p["cast"]]
        lines = [{k: v for k, v in d.items() if k != "sourceLineIds" or v} for d in p["lines"]]
        flashback = bool(p.get("flashback"))
        if flashback:  # 회상 장면 규칙은 LLM이 자주 어겨서 코드가 맞춘다(#33)
            if any(d["speaker"] in listeners for d in lines):
                flashback = False  # 듣는 사람이 말하는 장면은 지금 자리
            else:
                cast = [c for c in cast if c["key"] not in listeners]
        sb["scenes"].append({"index": i + 1, "flashback": flashback, "scene": clean_scene(p["scene"]),
                             "cast": cast, "extras": _extras(p, sb["extras"], cast), "lines": lines,
                             "motion": p["motion"], "transition": "fade" if i == 0 else p["transition"],
                             "sfx": to_null(p.get("sfx"))})
    fix_sounds(sb, raw.get("bgm"))
    for i, c in enumerate(sb["characters"]):
        if not c.get("ref"):  # 고른 그림의 외모 글은 그대로 둔다
            c["appearance"] = fix_appearance(c, i)
    for sc in sb["scenes"]:
        # 회상이 아닌 장면도 이야기 속 인물이 장면 글에만 있으면 넣는다(#33 2차 B3: 할아버지·할머니가 빠진 7장).
        # 화자를 빼는 것은 회상 장면에서만
        _fix_flashback(sc, sb, drop_cast=sc["flashback"])
        fix_scene(sc, MAX_TEXT, MAX_LINES)
        add_dressed(sc, sb)
        drop_phantoms(sc, sb)
    trim_lines(sb)
    sb["place"]["prompt"] = clean_scene(sb["place"]["prompt"])
    return sb


def make_video_storyboard(src: dict, llm: Llm) -> dict:
    keys = [sp["key"] for sp in src["speakers"]] or ["화자1"]
    schema = _schema(keys)
    user = card_text(src)
    errs: list[str] = []
    last_llm_err: LlmError | None = None
    for attempt in (1, 2, 3):
        msg = user if not errs else (
            user + "\n\n앞에서 만든 콘티에 아래 문제가 있었다. 고쳐서 다시 만든다.\n- " + "\n- ".join(errs[:15]))
        try:
            raw, stat = llm.chat_json(SYSTEM, msg, schema, "video_storyboard", temperature=0.4, max_tokens=4096)
        except LlmError as e:  # 출력이 잘리거나 깨짐, 서버 형식 오류(500): 다음 번에 다시 묻는다
            log.warning("video %s 콘티 %d번째: LLM 오류 %s", src.get("videoId"), attempt, e)
            last_llm_err = e
            continue
        sb = assemble(raw, src)
        errs = validate(sb, src)
        log.info("video %s 콘티 %d번째: 장면 %d개, 문제 %d개 %s", src.get("videoId"), attempt,
                 len(sb["scenes"]), len(errs), stat)
        if not errs:
            return sb
    if not errs and last_llm_err:  # 세 번 다 LLM 오류: 예전처럼 작업을 다시 시도하게 그대로 올린다
        raise last_llm_err
    raise StoryboardInvalid("영상 콘티 검사 실패: " + json.dumps(errs[:5], ensure_ascii=False))
