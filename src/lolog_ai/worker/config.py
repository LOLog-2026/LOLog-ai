"""환경 변수 (spec/deployment.md 7장). 값은 /etc/lolog/worker.env에 두고 systemd가 읽는다."""

from __future__ import annotations

import os
from dataclasses import dataclass

STREAMS = {"detect": "lolog:detect", "comic": "lolog:comic"}
GROUP = "jetson-workers"


@dataclass(frozen=True)
class Config:
    role: str
    redis_url: str
    api_url: str
    internal_token: str
    llm_base_url: str
    llm_api_key: str | None
    worker_name: str
    llm_model: str
    claim_idle_ms: int = 10 * 60 * 1000  # 10분 넘게 멈춘 메시지는 다시 가져옴 (api-spec "다시 시도")
    max_attempts: int = 3                 # 같은 jobId가 3번 실패하면 fail로 알림

    @property
    def stream(self) -> str:
        return STREAMS[self.role]


def load(role: str) -> Config:
    def need(name: str) -> str:
        v = os.environ.get(name)
        if not v:
            raise SystemExit(f"환경 변수 {name}가 없어요 (/etc/lolog/worker.env 확인)")
        return v

    return Config(
        role=role,
        redis_url=need("REDIS_URL"),
        api_url=need("API_INTERNAL_URL").rstrip("/"),
        internal_token=need("INTERNAL_TOKEN"),
        llm_base_url=need("LLM_BASE_URL").rstrip("/"),
        llm_api_key=os.environ.get("LLM_API_KEY") or None,
        worker_name=need("WORKER_NAME"),
        llm_model=os.environ.get("LLM_MODEL", "local"),
    )
