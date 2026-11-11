"""워커 실행: python -m lolog_ai.worker --role detect|comic

큐 규칙 (spec/api-spec.md "큐 메시지")
- 켜질 때 컨슈머 그룹 jetson-workers를 XGROUP CREATE ... MKSTREAM으로 만든다. 이미 있으면 넘어간다
- 켜질 때와 5분마다 XAUTOCLAIM으로 10분 넘게 멈춘 메시지(죽은 워커 몫)를 가져와 처리한다
- 처리가 끝나면 XACK 후 XDEL(대화 글이 스트림에 남지 않게)
- 같은 jobId가 3번 실패하면 POST /internal/jobs/{id}/fail 후 XACK·XDEL. 실패 횟수는 Redis에 둬서
  워커가 죽었다 켜져도 이어서 센다
- 로그에 대화 글을 남기지 않는다(TR-12). jobId, 개수, 시간만 남긴다
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import time

import redis

from . import config as cfgmod
from .api import ApiError, InternalApi
from .comic import StoryboardInvalid, avatar, comic
from .detect import detect
from .llm import Llm

log = logging.getLogger("lolog_ai.worker")
GONE_CODES = {"RECORDING_NOT_FOUND", "COMIC_NOT_FOUND", "CHARACTER_NOT_FOUND"}  # 작업이 지워짐: 다시 하지 않음
FAIL_KEY = "lolog:worker:fail:{}"
RETRY_DELAYS = [5, 30]  # 1번째, 2번째 실패 뒤 기다리는 초


class Worker:
    def __init__(self, cfg: cfgmod.Config):
        self.cfg = cfg
        self.r = redis.Redis.from_url(cfg.redis_url, decode_responses=True,
                                      socket_timeout=30, health_check_interval=30)
        self.api = InternalApi(cfg.api_url, cfg.internal_token)
        self.llm = Llm(cfg.llm_base_url, cfg.llm_model, cfg.llm_api_key)
        self.stopping = False
        self.last_claim = 0.0

    # ---------- 큐 ----------

    def ensure_group(self) -> None:
        try:
            self.r.xgroup_create(self.cfg.stream, cfgmod.GROUP, id="0", mkstream=True)
            log.info("컨슈머 그룹 만듦: %s %s", self.cfg.stream, cfgmod.GROUP)
        except redis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

    def claim_stale(self) -> list[tuple[str, dict]]:
        self.last_claim = time.time()
        out, start = [], "0-0"
        while True:
            res = self.r.xautoclaim(self.cfg.stream, cfgmod.GROUP, self.cfg.worker_name,
                                    min_idle_time=self.cfg.claim_idle_ms, start_id=start, count=10)
            start, msgs = res[0], res[1]
            out += [(mid, f) for mid, f in msgs if f]
            if start == "0-0" or not msgs:
                return out

    def read_new(self) -> list[tuple[str, dict]]:
        res = self.r.xreadgroup(cfgmod.GROUP, self.cfg.worker_name, {self.cfg.stream: ">"},
                                count=1, block=5000)
        return [(mid, f) for _, msgs in res or [] for mid, f in msgs]

    def done(self, mid: str) -> None:
        pipe = self.r.pipeline()
        pipe.xack(self.cfg.stream, cfgmod.GROUP, mid)
        pipe.xdel(self.cfg.stream, mid)
        pipe.execute()

    # ---------- 처리 ----------

    def run_job(self, msg: dict) -> None:
        if self.cfg.role == "detect":
            result = detect(msg, self.llm)
            self.api.post_cards(msg["recordingId"], result)
            log.info("job %s 감지 완료: 카드 %d개, carryFromMs=%s",
                     msg["jobId"], len(result["cards"]), result["carryFromMs"])
        elif msg.get("kind") == "avatar":
            avatar(msg, self.api)
            log.info("job %s 프로필 캐릭터 완료: user %s", msg["jobId"], msg["userId"])
        else:
            comic(msg, self.api, self.llm)
            log.info("job %s 만화 완료: comic %s", msg["jobId"], msg["comicId"])

    def handle(self, mid: str, fields: dict) -> None:
        try:
            msg = json.loads(fields["payload"])
            job_id = int(msg["jobId"])
        except (KeyError, ValueError, TypeError):
            log.error("메시지 %s 형식이 틀려 버림", mid)
            self.done(mid)
            return
        key = FAIL_KEY.format(job_id)
        while not self.stopping:
            t0 = time.perf_counter()
            try:
                self.run_job(msg)
                self.r.delete(key)
                self.done(mid)
                log.info("job %s 처리 %.1f초", job_id, time.perf_counter() - t0)
                return
            except ApiError as e:
                if not (e.status == 404 and e.code in GONE_CODES):
                    if self._failed(mid, job_id, key, e):
                        return
                    continue
                # 사용자가 만드는 중인 만화를 지웠거나 탈퇴해서 작업이 없어짐(api-spec "작업이 지워졌을 때").
                # 다시 하지 않고 fail도 보내지 않고 메시지만 지운다
                log.info("job %s 작업이 지워져 버림: %s", job_id, e.code)
                self.r.delete(key)
                self.done(mid)
                return
            except Exception as e:  # noqa: BLE001 - 어떤 실패든 횟수를 세고 다시 시도
                if self._failed(mid, job_id, key, e):
                    return

    def _failed(self, mid: str, job_id: int, key: str, e: Exception) -> bool:
        """실패 한 번을 세고, 끝났으면(메시지를 지웠거나 남겨 두기로 했으면) True. 아니면 잠깐 쉬고 False."""
        permanent = isinstance(e, StoryboardInvalid) or (isinstance(e, ApiError) and e.status == 400)
        n = self.cfg.max_attempts if permanent else self.r.incr(key)
        self.r.expire(key, 86400)
        log.warning("job %s 실패 %d/%d: %s", job_id, n, self.cfg.max_attempts,
                    type(e).__name__ + ": " + str(e)[:200])
        if n < self.cfg.max_attempts:
            time.sleep(RETRY_DELAYS[min(n - 1, len(RETRY_DELAYS) - 1)])
            return False
        try:
            self.api.fail_job(job_id, f"{type(e).__name__}: {e}")
        except ApiError as e2:
            if e2.status != 400:
                log.error("job %s fail 알림도 실패, 메시지를 남겨 둠: %s", job_id, e2)
                return True  # XACK 하지 않음 → 10분 뒤 XAUTOCLAIM으로 다시 시도
            log.info("job %s fail에 400(작업 없음), 메시지만 지움", job_id)
        except Exception as e2:  # noqa: BLE001
            log.error("job %s fail 알림도 실패, 메시지를 남겨 둠: %s", job_id, e2)
            return True
        self.r.delete(key)
        self.done(mid)
        return True

    def run(self) -> None:
        log.info("LLM 서버 기다리는 중: %s", self.cfg.llm_base_url)  # 2호도 콘티에 1호 LLM을 씀
        self.llm.wait_ready()
        while not self.stopping:  # 서버 Redis가 아직 없거나 터널이 끊겼으면 기다린다(로그는 30초에 한 번)
            try:
                self.ensure_group()
                break
            except redis.ConnectionError as e:
                log.warning("Redis에 못 붙음(서버 Redis·터널 확인), 30초 뒤 다시: %s", e)
                time.sleep(30)
        if self.stopping:
            return
        log.info("시작: %s · %s · 컨슈머 %s", self.cfg.role, self.cfg.stream, self.cfg.worker_name)
        while not self.stopping:
            try:
                batch = self.claim_stale() if time.time() - self.last_claim > 300 else []
                if batch:
                    log.info("멈춘 메시지 %d개 다시 가져옴", len(batch))
                for mid, fields in batch + self.read_new():
                    if self.stopping:
                        break
                    self.handle(mid, fields)
            except redis.ConnectionError as e:
                log.warning("Redis 연결 끊김(터널 확인), 5초 뒤 다시: %s", e)
                time.sleep(5)
        log.info("멈춤")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--role", choices=list(cfgmod.STREAMS), required=True)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    w = Worker(cfgmod.load(args.role))

    def stop(*_):
        log.info("종료 신호: 지금 작업을 끝내고 멈춤")
        w.stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    w.run()


if __name__ == "__main__":
    main()
