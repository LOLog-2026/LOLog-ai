"""글 카드(texts.md 5장): LLM 입력, 결과 모양, 감지 워커 분기."""

from __future__ import annotations

import pytest

from lolog_ai.worker.text_card import TextCardInvalid, text_card, text_input


class FakeLlm:
    def __init__(self, out: dict):
        self.out = out
        self.calls: list[tuple[str, str]] = []

    def chat_json(self, system, user, schema, name, **kw):
        self.calls.append((system, user))
        return self.out, {}


MSG = {"jobId": 101, "kind": "text", "textId": 8, "title": "동생이랑 마트 간 날",
       "body": "오늘 동생이랑 마트에 갔는데 동생이 카트를 잃어버렸다.", "place": "마트",
       "situation": "주말 저녁, 장 보러 감"}

RAW = {"kind": "웃김", "summary": "  마트에서 카트를 잃어버린 동생  ", "lines": [
    {"speaker": "화자2", "text": "언니 카트 어디 갔어?"},
    {"speaker": "화자1", "text": "네가 끌고 있었잖아!"},
    {"speaker": "화자2", "text": "  "},
    {"speaker": "화자2", "text": "아 맞다"},
]}


def test_input_has_place_and_situation():
    text = text_input(MSG)
    assert "제목: 동생이랑 마트 간 날" in text
    assert "장소: 마트" in text
    assert "상황: 주말 저녁, 장 보러 감" in text
    assert text.endswith("본문:\n" + MSG["body"])


def test_input_skips_blank_place_and_situation():
    text = text_input({**MSG, "place": "   ", "situation": None})
    assert "장소" not in text and "상황" not in text


def test_card_shape():
    llm = FakeLlm(RAW)
    res = text_card(MSG, llm)
    assert res["jobId"] == 101 and len(res["cards"]) == 1
    card = res["cards"][0]
    assert card["summary"] == "마트에서 카트를 잃어버린 동생"
    assert card["startMs"] is None and card["endMs"] is None
    assert card["speakers"] == ["화자1", "화자2"]  # 처음 나온 순서로 다시 붙임
    assert [(l["lineNo"], l["speaker"], l["text"]) for l in card["lines"]] == [
        (1, "화자1", "언니 카트 어디 갔어?"), (2, "화자2", "네가 끌고 있었잖아!"), (3, "화자1", "아 맞다")]
    assert all(l["startMs"] is None and l["endMs"] is None for l in card["lines"])
    assert "장소: 마트" in llm.calls[0][1]


def test_empty_lines_is_invalid():
    with pytest.raises(TextCardInvalid):
        text_card(MSG, FakeLlm({**RAW, "lines": [{"speaker": "화자1", "text": " "}]}))


def test_detect_worker_routes_text(monkeypatch):
    from lolog_ai.worker import __main__ as m

    sent = {}

    class Api:
        def post_text_cards(self, text_id, result):
            sent["path"] = (text_id, result)

        def post_cards(self, *a):
            raise AssertionError("녹음 결과 주소로 보내면 안 됨")

    w = object.__new__(m.Worker)
    w.cfg = type("C", (), {"role": "detect"})()
    w.api, w.llm = Api(), FakeLlm(RAW)
    w.run_job(MSG)
    assert sent["path"][0] == 8 and sent["path"][1]["cards"][0]["kind"] == "웃김"
    assert "TEXT_NOT_FOUND" in m.GONE_CODES
