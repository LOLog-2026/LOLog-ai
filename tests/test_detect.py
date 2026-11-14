"""이야기 찾기(detect.py): 창 나누기, 확인 단계, 화자 나누기 문법."""

from __future__ import annotations

import itertools
import math

import pytest

from lolog_ai.worker import detect as det
from lolog_ai.worker.llm import expected_score


def sents(n: int, gap_ms: int = 10_000) -> list[dict]:
    return [{"startMs": i * gap_ms, "endMs": i * gap_ms + 5_000, "text": f"말 {i}"} for i in range(n)]


def seg_range(user: str) -> tuple[int, int]:
    nums = [int(l.split("]")[0][1:]) for l in user.splitlines()]
    return nums[0], nums[-1]


class FakeLlm:
    """창마다 정해 둔 이야기를 돌려주고, 확인 단계는 구간 시작 줄로 점수를 정한다."""

    def __init__(self, stories: list[dict], escore: dict[int, float | None] | None = None,
                 score: int = 5):
        self.stories = stories
        self.escore = escore or {}
        self.score = score
        self.calls: list[tuple[str, str, dict]] = []

    def chat_json(self, system, user, schema, name, **kw):
        self.calls.append((name, user, kw))
        lo, hi = seg_range(user)
        if name == "detect":
            # 창 밖 이야기도 돌려줘서 detect가 걸러내는지 본다
            return {"stories": [dict(s) for s in self.stories]}, {}
        if name == "verify":
            return {"score": self.score}, {"escore": self.escore.get(lo)}
        return {"lines": [{"seg": k, "speaker": "화자1", "text": f"말 {k}"} for k in range(lo, hi + 1)]}, {}


def story(lo: int, hi: int, finished: bool = True) -> dict:
    return {"kind": "웃김", "summary": f"이야기 {lo}", "from": lo, "to": hi, "finished": finished}


def msg(ss: list[dict], is_last: bool = True) -> dict:
    return {"jobId": 1, "partNo": 1, "isLast": is_last, "sentences": ss}


def test_windows_cover_all_with_overlap():
    ss = sents(180)  # 30분, 10초마다 한 구간
    wins = det.windows(ss)
    assert wins[0][0] == 0 and wins[-1][1] == 179
    for (_, a1), (b0, _) in itertools.pairwise(wins):
        assert b0 <= a1  # 겹침
        assert (a1 - b0 + 1) * 10 >= 170  # 겹침이 약 3분
    assert all((hi - lo + 1) * 10 <= 8 * 60 for lo, hi in wins)


def test_windows_off(monkeypatch):
    monkeypatch.setattr(det, "WINDOW_MS", 0)
    assert det.windows(sents(500)) == [(0, 499)]


def test_short_recording_one_window():
    assert det.windows(sents(20)) == [(0, 19)]


def test_stories_outside_window_dropped_and_dupes_merged():
    ss = sents(90)  # 15분 → 창 3개
    llm = FakeLlm([story(10, 20), story(40, 45), story(70, 85)])
    out = det.detect(msg(ss), llm)
    spans = [(c["startMs"] // 10_000, c["endMs"] // 10_000) for c in out["cards"]]
    assert spans == [(10, 20), (40, 45), (70, 85)]  # 창마다 다시 나와도 한 장씩
    assert sum(1 for n, _, _ in llm.calls if n == "detect") == len(det.windows(ss))


def test_unfinished_in_middle_window_is_not_carried():
    ss = sents(90)
    _, hi0 = det.windows(ss)[0]
    llm = FakeLlm([story(hi0 - 3, hi0, finished=False)])
    out = det.detect(msg(ss, is_last=False), llm)
    # 첫 창 끝에서 "안 끝남"이어도 묶음 끝이 아니라서 넘기지 않는다
    assert out["carryFromMs"] == max(s["endMs"] for s in ss) - det.CARRY_WINDOW_MS
    assert len(out["cards"]) == 1


def test_verify_filters_by_expected_score():
    ss = sents(30)
    llm = FakeLlm([story(2, 6), story(10, 14), story(20, 24)], escore={2: 4.2, 10: 3.85, 20: 3.9})
    out = det.detect(msg(ss), llm)
    assert [c["startMs"] // 10_000 for c in out["cards"]] == [2, 20]
    v = [kw for n, _, kw in llm.calls if n == "verify"]
    assert v and all(kw.get("logprobs") for kw in v)


def test_verify_falls_back_to_score_without_logprobs():
    ss = sents(30)
    assert len(det.detect(msg(ss), FakeLlm([story(2, 6)], score=4))["cards"]) == 1
    assert det.detect(msg(ss), FakeLlm([story(2, 6)], score=3))["cards"] == []


def test_verify_off(monkeypatch):
    monkeypatch.setattr(det, "VERIFY_MIN", 0)
    llm = FakeLlm([story(2, 6)], escore={2: 1.0})
    assert len(det.detect(msg(sents(30)), llm)["cards"]) == 1
    assert not [n for n, _, _ in llm.calls if n == "verify"]


def test_carried_stories_are_not_verified():
    ss = sents(60)  # 10분. 마지막 5분 안에서 끝나는 이야기는 넘긴다
    llm = FakeLlm([story(5, 8), story(50, 55)])
    out = det.detect(msg(ss, is_last=False), llm)
    verified = [seg_range(u)[0] for n, u, _ in llm.calls if n == "verify"]
    assert verified == [5]
    assert len(out["cards"]) == 1


def test_speakers_use_grammar():
    llm = FakeLlm([story(2, 6)])
    det.detect(msg(sents(30)), llm)
    spk = [kw for n, _, kw in llm.calls if n == "speakers"]
    assert spk and spk[0]["grammar"] == det.SPEAKER_GRAMMAR


def test_kind_rules_in_prompt():
    assert "추억: 몇 년 전 일" in det.DETECT_SYSTEM
    assert det.DETECT_SYSTEM.index("- 추억") < det.DETECT_SYSTEM.index("- 황당") < det.DETECT_SYSTEM.index("- 웃김")


def lp(tok: str, top: dict[str, float]) -> dict:
    return {"token": tok, "top_logprobs": [{"token": k, "logprob": math.log(v)} for k, v in top.items()]}


def test_expected_score():
    content = [lp("{\"", {}), lp("score", {}), lp("\":", {}), lp("4", {"4": 0.6, "5": 0.3, "3": 0.1})]
    assert expected_score({"content": content}) == pytest.approx(4.2)
    assert expected_score({"content": [lp("4", {"4": 1.0})]}) is None  # "score" 키 앞의 숫자는 보지 않음
    assert expected_score({}) is None
