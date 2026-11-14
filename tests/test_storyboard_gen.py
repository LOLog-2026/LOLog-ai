"""4컷 콘티(storyboard_gen.py): 영상에서 옮긴 그림 설명 보정과 컷 프롬프트."""

from __future__ import annotations

import io
import urllib.error
import urllib.request

import pytest

from lolog_ai import render
from lolog_ai.storyboard import validate
from lolog_ai.worker.llm import Llm, LlmError
from lolog_ai.worker.storyboard_gen import assemble, make_storyboard

SRC = {
    "comicId": 7, "cardId": 3, "kind": "웃김", "summary": "카페에서 쪽지를 잘못 준 친구", "place": None,
    "speakers": [{"key": "화자1", "name": None}, {"key": "화자2", "name": None}],
    "lines": [{"id": 1, "speaker": "화자1", "text": "이거 봐"}, {"id": 2, "speaker": "화자2", "text": "헐 대박"}],
}


def panel(beat: str, scene: str, cast: list[tuple[str, str]], dialogue: list[str]) -> dict:
    return {"beat": beat, "scene": scene,
            "cast": [{"key": k, "position": pos, "expression": "smiling"} for k, pos in cast],
            "dialogue": [{"speaker": cast[0][0], "text": t, "sourceLineIds": [1]} for t in dialogue]}


RAW = {
    "place": {"text": "카페", "prompt": "a cozy cafe with a menu board on the wall"},
    "characters": [
        {"key": "화자1", "gender": "f", "age": "young", "appearance": "short black hair, hoodie"},
        {"key": "화자2", "gender": "m", "age": "young", "appearance": "young man, glasses, purple cardigan"},
    ],
    "panels": [
        panel("buildup", 'The woman hands over a note that reads "sorry".', [("화자1", "left")], ["이거 봐"]),
        panel("development", "The man in a purple cardigan looks surprised.", [("화자1", "left")], ["뭐야"]),
        panel("punchline", "The woman laughs. A waiter walks by.", [("화자1", "left"), ("화자2", "right")], ["헐"]),
        panel("reaction", "They both laugh and then leave the cafe.", [("화자1", "left"), ("화자2", "right")], []),
    ],
}


def test_assemble_cleans_and_fixes():
    sb = assemble(RAW, SRC)
    assert validate(sb, SRC) == []
    p1, p2, p3, p4 = sb["panels"]
    assert "reads" not in p1["scene"] and "sorry" not in p1["scene"]  # 글자를 그리라는 말
    assert sb["place"]["prompt"] == "a cozy cafe on the wall."                 # 간판·메뉴판
    assert [c["key"] for c in p2["cast"]] == ["화자1", "화자2"]            # 옷으로 부른 인물을 넣음
    assert "waiter" not in p3["scene"]                                     # 컷에 없는 사람 문장
    assert "then" not in p4["scene"]                                       # 한 컷에 두 순간
    a1 = sb["characters"][0]["appearance"]
    assert a1.startswith("young woman") and "navy hoodie" in a1            # 성별, 윗옷 색
    assert set(sb["characters"][0]) == {"key", "name", "appearance"}       # gender·age는 콘티에 남기지 않음


def test_add_dressed_respects_crowded_bubble_rule():
    raw = {**RAW, "panels": [*RAW["panels"][:1],
                             panel("development", "The man in a purple cardigan watches.",
                                   [("화자1", "left")], ["하나", "둘"]),
                             *RAW["panels"][2:]]}
    sb = assemble(raw, SRC)
    assert validate(sb, SRC) == []


def test_ref_appearance_kept():
    src = {**SRC, "speakers": [{"key": "화자1", "name": None,
                                "character": {"appearance": "young woman, long hair", "ref": "young-f-01"}},
                               {"key": "화자2", "name": None}]}
    sb = assemble(RAW, src)
    assert sb["characters"][0]["appearance"] == "young woman, long hair"


def test_prompt_counts_people_before_scene():
    sb = assemble(RAW, SRC)
    text = render.prompt(sb, sb["panels"][2])
    assert "exactly two people, each appears only once" in text
    assert text.index("each appears only once") < text.index("Scene:")
    assert "signs and storefronts are plain" in text


class OnceLlm:
    def chat_json(self, system, user, schema, name, **kw):
        assert schema["properties"]["characters"]["items"]["required"] == ["key", "gender", "age", "appearance"]
        return RAW, {}


def test_make_storyboard_passes_first_try():
    sb = make_storyboard(SRC, OnceLlm())
    assert len(sb["panels"]) == 4


class FlakyLlm:
    """첫 번째는 서버 형식 오류(500 → LlmError), 두 번째는 정상."""

    def __init__(self, fails: int = 1):
        self.fails = fails
        self.calls = 0

    def chat_json(self, system, user, schema, name, **kw):
        self.calls += 1
        if self.calls <= self.fails:
            raise LlmError("LLM 서버 500: model produced output that does not match the expected format")
        return RAW, {}


def test_make_storyboard_retries_after_llm_error():
    llm = FlakyLlm()
    assert len(make_storyboard(SRC, llm)["panels"]) == 4
    assert llm.calls == 2


def test_make_storyboard_reraises_when_all_llm_errors():
    with pytest.raises(LlmError):
        make_storyboard(SRC, FlakyLlm(fails=3))


def test_post_500_becomes_llm_error(monkeypatch):
    def boom(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 500, "err", {}, io.BytesIO(b'{"error":"does not match"}'))
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(LlmError, match="500"):
        Llm("http://x/v1")._post("/chat/completions", {})
