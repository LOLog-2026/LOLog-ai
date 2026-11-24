"""콘티 검사 실패 메시지에 대사 원문·별명이 없어야 한다(TR-12, #62).

StoryboardInvalid 메시지는 워커 로그(journald)와 fail_reason(comics·jobs)으로 나간다.
원문은 LLM에 다시 물을 때만 붙인다(다시 묻는 글에는 앞에서 만든 콘티가 없어 원문이 유일한 단서).
"""

from __future__ import annotations

import pytest

from lolog_ai import storyboard, video_storyboard
from lolog_ai.worker.storyboard_gen import StoryboardInvalid, make_storyboard
from lolog_ai.worker.video_gen import make_video_storyboard

NICK = "Bbobbi"
SECRET = "화자1 비밀번호는 고구마야"                      # 내부 이름이 섞인 대사
LONG = "아니근데진짜어제엄마가계단올라갈때숨안차냐고물어봐서내가깜짝놀라가지고대답도못했잖아"  # 띄어쓰기 없이 40자 초과

SRC = {
    "comicId": 7, "videoId": 9, "cardId": 3, "kind": "웃김", "summary": "카페에서 생긴 일", "place": None,
    "speakers": [{"key": "화자1", "name": NICK}, {"key": "화자2", "name": None}],
    "lines": [{"id": 1, "speaker": "화자1", "text": "이거 봐"}, {"id": 2, "speaker": "화자2", "text": "헐 대박"}],
}
CHARACTERS = [
    {"key": "화자1", "role": "teller", "gender": "f", "age": "young", "appearance": "short black hair, hoodie"},
    {"key": "화자2", "role": "listener", "gender": "m", "age": "young", "appearance": "glasses, purple cardigan"},
]


def comic_raw() -> dict:
    def panel(beat: str, scene: str, text: str) -> dict:
        return {"beat": beat, "scene": scene,
                "cast": [{"key": "화자1", "position": "left", "expression": "smiling"}],
                "dialogue": [{"speaker": "화자1", "text": text, "sourceLineIds": [1]}]}

    return {
        "place": {"text": "카페", "prompt": "a cozy cafe"},
        "characters": CHARACTERS,
        "panels": [panel("buildup", f"{NICK} sits at a table.", "이거 봐"),
                   panel("development", "The woman sits at a table.", SECRET),
                   panel("punchline", "The woman sits at a table.", "뭐야"),
                   panel("reaction", "The woman sits at a table.", "헐")],
    }


def video_raw() -> dict:
    def scene(desc: str, text: str) -> dict:
        return {"flashback": False, "scene": desc, "extras": [],
                "cast": [{"key": "화자1", "position": "left", "expression": "smiling"}],
                "lines": [{"speaker": "화자1", "text": text, "sourceLineIds": [1]}],
                "motion": "still", "transition": "cut", "sfx": "none"}

    return {
        "place": {"text": "카페", "prompt": "a cozy cafe"},
        "characters": CHARACTERS,
        "extras": [],
        "scenes": [scene(f"{NICK} sits at a table.", LONG),
                   scene("The woman sits at a table.", SECRET),
                   scene("The woman sits at a table.", "뭐야"),
                   scene("The woman sits at a table.", "헐")],
        "bgm": "none",
    }


class AlwaysBad:
    """늘 같은 틀린 콘티를 준다. 다시 물을 때 받은 글을 모아 둔다."""

    def __init__(self, raw: dict):
        self.raw = raw
        self.asked: list[str] = []

    def chat_json(self, system, user, schema, name, **kw):
        self.asked.append(user)
        return self.raw, {}


def test_comic_failure_message_has_no_dialogue_or_nickname():
    llm = AlwaysBad(comic_raw())
    with pytest.raises(StoryboardInvalid) as e:
        make_storyboard(SRC, llm)
    msg = str(e.value)
    assert "내부 이름이 섞임" in msg and "별명이 들어감" in msg  # 어떤 규칙에 걸렸는지는 남는다
    assert "고구마" not in msg and NICK not in msg
    assert len(llm.asked) == 3
    feedback = llm.asked[1].split("고쳐서 다시 만든다.")[1]         # 다시 묻는 글에는 원문이 그대로 있다
    assert repr(SECRET) in feedback and f"별명 {NICK!r}이 들어감" in feedback


def test_video_failure_message_has_no_dialogue_or_nickname():
    llm = AlwaysBad(video_raw())
    with pytest.raises(StoryboardInvalid) as e:
        make_video_storyboard(SRC, llm)
    msg = str(e.value)
    assert f"대사가 {len(LONG)}자" in msg and "내부 이름이 섞임" in msg and "별명이 들어감" in msg
    assert LONG not in msg and "고구마" not in msg and NICK not in msg
    feedback = llm.asked[1].split("고쳐서 다시 만든다.")[1]
    assert repr(LONG) in feedback and repr(SECRET) in feedback and f"별명 {NICK!r}이 들어감" in feedback


@pytest.mark.parametrize("mod, lines_key, limit", [(storyboard, "dialogue", storyboard.MAX_TEXT),
                                                    (video_storyboard, "lines", video_storyboard.MAX_TEXT)])
def test_validate_quotes_only_when_asked(mod, lines_key, limit):
    long = "가" * (limit + 1)
    unit = {"index": 1, "scene": f"{NICK} waves.", "cast": [{"key": "화자1", "position": "left"}],
            lines_key: [{"speaker": "화자1", "text": long}, {"speaker": "화자1", "text": SECRET}]}
    sb = {"version": 1, "characters": [{"key": "화자1", "name": NICK, "appearance": "young woman"}],
          "place": {"prompt": "a cafe"}, "panels": [unit], "scenes": [unit]}
    safe = "\n".join(mod.validate(sb))
    assert f"대사가 {limit + 1}자 (최대 {limit}자)" in safe and "내부 이름이 섞임" in safe and "별명이 들어감" in safe
    assert long not in safe and "고구마" not in safe and NICK not in safe
    quoted = mod.validate(sb, quote=True)
    # quote=True는 고치기 전과 같은 문구(LLM에 다시 묻는 글이 달라지지 않게)
    assert any(e.endswith(f"대사가 {limit + 1}자 (최대 {limit}자): {long!r}") for e in quoted)
    assert any(e.endswith(f"대사에 내부 이름이 섞임: {SECRET!r}") for e in quoted)
    assert any(f"에 별명 {NICK!r}이 들어감 (외모 설명으로 바꿔야 함)" in e for e in quoted)
    assert len(quoted) == len(mod.validate(sb))
