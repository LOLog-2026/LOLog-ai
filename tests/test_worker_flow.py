"""워커의 재시도·실패 흐름(worker/__main__.py handle·_failed, #65).

큐 규칙(spec/api-spec.md "큐 메시지")을 가짜 Redis와 가짜 api로 지킨다.
- 끝나면 XACK 후 XDEL, 실패 횟수 키를 지움
- 같은 jobId가 3번 실패하면 fail 알림 후 메시지를 지움. 횟수는 Redis에 있어 워커가 다시 켜져도 이어짐
- 다시 해도 같은 실패(콘티·녹음 검사, 400)는 바로 fail
- 작업이 지워졌으면(404 + *_NOT_FOUND) fail 없이 메시지만 지움
- fail 알림도 실패하면 메시지를 남겨 둠(XACK 하지 않음 → XAUTOCLAIM으로 다시)
"""

from __future__ import annotations

import json
import urllib.error

import pytest

pytest.importorskip("redis")

from lolog_ai.worker import __main__ as wm
from lolog_ai.worker.api import ApiError
from lolog_ai.worker.storyboard_gen import StoryboardInvalid
from lolog_ai.worker.voice_clone import VoiceSampleInvalid

JOB = 41
KEY = wm.FAIL_KEY.format(JOB)
MID = "1700000000000-0"
STREAM, GROUP = "lolog:comic", "jetson-workers"


class FakeRedis:
    """실패 횟수(incr·expire·delete)와 XACK·XDEL만 흉내 낸다."""

    def __init__(self, **store: int):
        self.store = dict(store)
        self.ttl: dict[str, int] = {}
        self.sent: list[tuple] = []   # execute()까지 간 XACK·XDEL

    def incr(self, key: str) -> int:
        self.store[key] = int(self.store.get(key, 0)) + 1
        return self.store[key]

    def expire(self, key: str, seconds: int) -> None:
        self.ttl[key] = seconds

    def delete(self, key: str) -> None:
        self.store.pop(key, None)

    def pipeline(self):
        redis, queued = self, []

        class Pipe:
            def xack(self, stream, group, mid):
                queued.append(("xack", stream, group, mid))

            def xdel(self, stream, mid):
                queued.append(("xdel", stream, mid))

            def execute(self):
                redis.sent.extend(queued)

        return Pipe()


class FakeApi:
    def __init__(self, fail_error: Exception | None = None):
        self.fail_error = fail_error
        self.fails: list[tuple[int, str]] = []

    def fail_job(self, job_id: int, reason: str) -> None:
        self.fails.append((job_id, reason))
        if self.fail_error:
            raise self.fail_error


def api_error(status: int, code: str | None = None) -> ApiError:
    return ApiError(status, json.dumps({"code": code, "message": "x"}) if code else "oops")


@pytest.fixture
def sleeps(monkeypatch) -> list[float]:
    out: list[float] = []
    monkeypatch.setattr(wm.time, "sleep", out.append)
    return out


def make_worker(outcomes: list, redis: FakeRedis | None = None, api: FakeApi | None = None):
    """outcomes: run_job을 부를 때마다 차례로 쓴다. 예외면 던지고, None이면 성공."""
    w = wm.Worker.__new__(wm.Worker)
    w.cfg = type("Cfg", (), {"role": "comic", "stream": STREAM, "max_attempts": 3})()
    w.r = redis or FakeRedis()
    w.api = api or FakeApi()
    w.stopping = False
    w.ran = []

    def run_job(msg: dict) -> None:
        w.ran.append(msg)
        outcome = outcomes[len(w.ran) - 1]
        if outcome is not None:
            raise outcome

    w.run_job = run_job
    return w


def handle(w, payload: dict | None = None) -> None:
    w.handle(MID, {"payload": json.dumps(payload or {"jobId": JOB, "comicId": 7})})


ACKED = [("xack", STREAM, GROUP, MID), ("xdel", STREAM, MID)]


# ---------- 성공·버리기 ----------

def test_success_acks_and_clears_fail_count(sleeps):
    w = make_worker([None], FakeRedis(**{KEY: 1}))
    handle(w)
    assert len(w.ran) == 1 and w.r.sent == ACKED   # XACK 다음에 XDEL(대화 글이 스트림에 남지 않게)
    assert KEY not in w.r.store and w.api.fails == [] and sleeps == []


@pytest.mark.parametrize("fields", [
    {},                                      # payload 없음
    {"payload": "{json 아님"},
    {"payload": json.dumps({"comicId": 7})},  # jobId 없음
    {"payload": json.dumps({"jobId": None})},
])
def test_malformed_message_is_dropped(fields, sleeps):
    w = make_worker([])
    w.handle(MID, fields)
    assert w.ran == [] and w.r.sent == ACKED and w.api.fails == []


@pytest.mark.parametrize("code", sorted(wm.GONE_CODES))
def test_gone_job_is_dropped_without_fail(code, sleeps):
    w = make_worker([api_error(404, code)], FakeRedis(**{KEY: 2}))
    handle(w)
    assert len(w.ran) == 1 and w.api.fails == [] and sleeps == []
    assert w.r.sent == ACKED and KEY not in w.r.store


def test_other_404_counts_as_failure(sleeps):
    """404여도 지워진 작업 코드가 아니면(경로 오류 등) 버리지 않고 실패로 센다."""
    w = make_worker([api_error(404, "NOT_FOUND"), api_error(404), None])
    handle(w)
    assert len(w.ran) == 3 and sleeps == [5, 30] and w.api.fails == []


def test_stopping_worker_does_not_start_job(sleeps):
    w = make_worker([None])
    w.stopping = True
    handle(w)
    assert w.ran == [] and w.r.sent == []   # 메시지는 그대로 남아 다음에 다시 온다


# ---------- 다시 시도 ----------

def test_retries_then_succeeds(sleeps):
    w = make_worker([RuntimeError("a"), OSError("b"), None])
    handle(w)
    assert len(w.ran) == 3 and sleeps == [5, 30]
    assert w.api.fails == [] and w.r.sent == ACKED and KEY not in w.r.store


def test_third_failure_reports_fail_once(sleeps):
    w = make_worker([RuntimeError("a"), RuntimeError("b"), RuntimeError("boom")])
    handle(w)
    assert len(w.ran) == 3 and sleeps == [5, 30]      # 세 번째 뒤에는 쉬지 않는다
    assert w.api.fails == [(JOB, "RuntimeError: boom")]
    assert w.r.sent == ACKED and KEY not in w.r.store


def test_fail_count_survives_restart(sleeps):
    """실패 횟수는 Redis에 있다. 두 번 실패한 뒤 워커가 다시 켜졌으면 한 번 더 실패로 끝난다."""
    w = make_worker([RuntimeError("boom")], FakeRedis(**{KEY: 2}))
    handle(w)
    assert len(w.ran) == 1 and sleeps == [] and len(w.api.fails) == 1


def test_fail_count_expires_in_a_day(sleeps):
    w = make_worker([RuntimeError("a"), None])
    handle(w)
    assert w.r.ttl[KEY] == 86400


# ---------- 다시 해도 같은 실패 ----------

@pytest.mark.parametrize("error, reason", [
    (StoryboardInvalid("콘티 검사 실패: []"), "StoryboardInvalid: 콘티 검사 실패: []"),
    (VoiceSampleInvalid("VOICE_TOO_QUIET", "peak -50dB"), "VOICE_TOO_QUIET"),   # 녹음은 코드만(앱이 문구를 고른다)
    (api_error(400, "INVALID_STORYBOARD"), None),
])
def test_permanent_failure_fails_at_once(error, reason, sleeps):
    w = make_worker([error])
    handle(w)
    assert len(w.ran) == 1 and sleeps == []
    assert len(w.api.fails) == 1 and w.api.fails[0][0] == JOB
    if reason:
        assert w.api.fails[0][1] == reason
    assert w.r.sent == ACKED and KEY not in w.r.store


@pytest.mark.parametrize("status", [401, 409, 500, 503])
def test_other_api_errors_are_retried(status, sleeps):
    w = make_worker([api_error(status), None])
    handle(w)
    assert len(w.ran) == 2 and sleeps == [5] and w.api.fails == []


# ---------- fail 알림이 실패할 때 ----------

def test_fail_notice_400_means_job_is_gone(sleeps):
    """fail에 400이면 작업이 이미 없다. 메시지만 지운다."""
    w = make_worker([StoryboardInvalid("x")], api=FakeApi(api_error(400, "JOB_NOT_FOUND")))
    handle(w)
    assert len(w.api.fails) == 1 and w.r.sent == ACKED and KEY not in w.r.store


@pytest.mark.parametrize("fail_error", [
    api_error(500),
    api_error(404, "COMIC_NOT_FOUND"),
    urllib.error.URLError("터널 끊김"),
    TimeoutError(),
])
def test_fail_notice_failure_keeps_message(fail_error, sleeps):
    """fail 알림도 실패하면 XACK 하지 않는다. 10분 뒤 XAUTOCLAIM으로 다시 온다."""
    redis = FakeRedis(**{KEY: 2})
    w = make_worker([RuntimeError("boom")], redis, FakeApi(fail_error))
    handle(w)
    assert len(w.ran) == 1 and len(w.api.fails) == 1
    assert redis.sent == [] and redis.store[KEY] == 3   # 메시지도 횟수도 남는다


def test_reclaimed_message_runs_job_again(sleeps):
    """남겨 둔 메시지가 다시 오면 작업을 한 번 더 해 본다. 되면 끝, 또 실패하면 fail을 다시 알린다."""
    redis = FakeRedis(**{KEY: 3})
    w = make_worker([None], redis)
    handle(w)
    assert len(w.ran) == 1 and w.api.fails == [] and redis.sent == ACKED and KEY not in redis.store

    redis = FakeRedis(**{KEY: 3})
    w = make_worker([RuntimeError("boom")], redis)
    handle(w)
    assert len(w.ran) == 1 and len(w.api.fails) == 1 and sleeps == [] and redis.sent == ACKED
