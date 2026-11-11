"""카드(VideoSource) → 영상 콘티 (#33, spec/video.md 5장).

4컷 콘티(storyboard_gen.py)와 같은 방식이다. LLM이 장소·인물 외모·장면 4~8개를 JSON 스키마에 맞춰 만들고,
코드가 version·videoId·cardId·index·name을 채운 뒤 video_storyboard.validate로 검사한다. 틀리면 문제 목록을
알려 다시 묻고(3번까지), 그래도 틀리면 실패. 시간표(startMs·endMs)는 TTS를 읽은 뒤 timeline()이 채운다.
"""

from __future__ import annotations

import json
import logging
import re

from ..video_storyboard import (
    MAX_LINES,
    MAX_SCENES,
    MAX_TEXT,
    MIN_SCENES,
    MOTIONS,
    TRANSITIONS,
    validate,
)
from .llm import Llm
from .storyboard_gen import StoryboardInvalid, _character, card_text, fixed_looks

log = logging.getLogger(__name__)

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

장면의 인물(cast)
- cast는 화자만, 0~3명. extras와 합쳐 한 장면에 3명까지. position은 left/center/right 중 하나이고 cast와 extras를 합쳐 한 장면에서 겹치지 않는다
- flashback이 아닌 장면은 화자가 한 명 이상 나온다

대사(lines): 자막으로 뜨고 목소리로 읽힌다
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
        },
        "required": ["flashback", "scene", "cast", "extras", "lines", "motion", "transition"],
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
        },
        "required": ["place", "characters", "extras", "scenes"],
    }


_QUOTED = re.compile(r"\s*(\"[^\"]*\"|“[^”]*”|‘[^’]*’|(?<![A-Za-z])'[^']*'(?![A-Za-z])|\([^)]*\))")


def clean_scene(text: str) -> str:
    """그림 설명에서 따옴표로 묶은 말과 괄호(이름)를 뺀다. 넣으면 그림에 글자·말풍선이 생긴다(#33).
    LLM이 규칙을 자주 어겨서 다시 묻지 않고 코드가 지운다."""
    out = _QUOTED.sub("", text)
    out = re.sub(r"(,\s*|\band\s+|\bwith\s+)?(a\s+)?(bright\s+|colorful\s+)?(neon\s+)?(signs?|signboards?|banners?|"
                 r"posters?|menus? boards?)\b", "", out, flags=re.IGNORECASE)  # 간판·현수막은 글씨가 그려진다
    out = re.sub(r"\b(says|saying|said|asks|asking|shouts|whispers)\s*[,:]?(?=\s|[.]|$)", "speaks", out)
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


def _fix_flashback(scene: dict, sb: dict) -> None:
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
        if head and head in text:  # 그 사람의 외모("old woman")로 부르면 이 장면의 주인공. 언급 안 된 화자는 뺀다
            scene["cast"] = [c for c in scene["cast"] if looks.get(c["key"]) and looks[c["key"]] in text]
        taken = {c["position"] for c in scene["cast"]} | {x["position"] for x in scene["extras"]}
        free = [p for p in ("center", "left", "right") if p not in taken]
        if free and len(scene["cast"]) + len(scene["extras"]) < 3:
            scene["extras"].append({"id": e["id"], "position": free[0]})


POSITIONS = ("left", "right", "center")
_PERSON = re.compile(r"\b(man|woman|boy|girl|male|female|guy|lady|grandma|grandpa|grandmother|grandfather)\b")
_COLOR = re.compile(r"\b(black|white|gray|grey|navy|blue|red|green|yellow|pink|purple|brown|beige|orange|"
                    r"cream|khaki|mint|olive|maroon|denim|striped|plaid|checkered)\b")
_TOP = re.compile(r"\b(hoodie|t-shirt|shirt|sweater|jacket|cardigan|blouse|dress|coat|sweatshirt|blazer|uniform)\b")
TOP_COLORS = ["navy", "mustard yellow", "forest green", "coral", "lavender", "sky blue"]


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
    seg = a.lower()[:top.start()].rsplit(",", 1)[-1] if top else ""  # 윗옷 바로 앞(머리 색은 빼고)
    if top and not _COLOR.search(seg):
        a = a[:top.start()] + TOP_COLORS[i % len(TOP_COLORS)] + " " + a[top.start():]
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
                             "motion": p["motion"], "transition": "fade" if i == 0 else p["transition"]})
    for i, c in enumerate(sb["characters"]):
        if not c.get("ref"):  # 고른 그림의 외모 글은 그대로 둔다
            c["appearance"] = fix_appearance(c, i)
    for sc in sb["scenes"]:
        if sc["flashback"]:
            _fix_flashback(sc, sb)
        fix_scene(sc, MAX_TEXT, MAX_LINES)
    sb["place"]["prompt"] = clean_scene(sb["place"]["prompt"])
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
