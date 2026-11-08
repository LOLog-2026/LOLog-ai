"""LLM 호출. llama-server(OpenAI 호환 /v1/chat/completions)를 HTTP로 부른다."""

from __future__ import annotations

import json
import time
import urllib.request


class LlmError(Exception):
    pass


class Llm:
    def __init__(self, base_url: str, model: str = "local", api_key: str | None = None,
                 timeout: float = 1800):
        self.base = base_url  # .../v1
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def wait_ready(self, timeout: float = 600) -> None:
        """서버가 모델을 다 올릴 때까지 기다린다(젯슨에서 1~2분)."""
        url = self.base.removesuffix("/v1") + "/health"
        end = time.time() + timeout
        while True:
            req = urllib.request.Request(url)
            if self.api_key:
                req.add_header("Authorization", f"Bearer {self.api_key}")
            try:
                with urllib.request.urlopen(req, timeout=5) as r:
                    ready = json.loads(r.read()).get("status") == "ok"
            except (OSError, ValueError):  # 서버가 아직 뜨는 중(모델 로드 중엔 503)
                ready = False
            if ready:
                return
            if time.time() > end:
                raise LlmError(f"LLM 서버가 {timeout:.0f}초 안에 준비되지 않음: {url}")
            time.sleep(5)

    def chat_json(self, system: str, user: str, schema: dict, name: str,
                  temperature: float = 0.2, max_tokens: int = 2048) -> tuple[dict, dict]:
        """JSON 스키마로 출력을 강제해 부르고 (결과, 통계)를 돌려준다."""
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_schema", "json_schema": {"name": name, "schema": schema}},
        }
        t0 = time.perf_counter()
        resp = self._post("/chat/completions", body)
        choice = resp["choices"][0]
        stat = {
            "wall_s": round(time.perf_counter() - t0, 1),
            "prompt_tokens": resp.get("usage", {}).get("prompt_tokens"),
            "completion_tokens": resp.get("usage", {}).get("completion_tokens"),
            "finish_reason": choice.get("finish_reason"),
        }
        if choice.get("finish_reason") == "length":
            raise LlmError(f"출력이 max_tokens({max_tokens})에서 잘림")
        try:
            return json.loads(choice["message"]["content"]), stat
        except json.JSONDecodeError as e:
            raise LlmError(f"JSON 파싱 실패: {e}") from None
