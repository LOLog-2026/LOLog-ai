"""LLM 호출. llama-server(OpenAI 호환 /v1/chat/completions)를 HTTP로 부른다."""

from __future__ import annotations

import json
import math
import time
import urllib.error
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
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            # llama-server는 모델 출력이 응답 형식 해석기와 안 맞으면 500을 준다
            # ("model produced output that does not match the expected ... format", Qwen3.5에서 봄).
            # 출력 문제라서 JSON 깨짐과 같이 LlmError로 올려, 부르는 쪽이 다시 묻거나 그 이야기만 버리게 한다.
            # 503(모델 로드 중) 같은 다른 오류는 그대로 올려 작업 전체를 다시 시도하게 한다
            if e.code == 500:
                detail = e.read()[:300].decode("utf-8", "replace")
                raise LlmError(f"LLM 서버 500: {detail}") from None
            raise

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
                  temperature: float = 0.2, max_tokens: int = 2048,
                  grammar: str | None = None, logprobs: bool = False) -> tuple[dict, dict]:
        """JSON 스키마로 출력을 강제해 부르고 (결과, 통계)를 돌려준다.

        grammar: 스키마 대신 llama.cpp GBNF 문법으로 출력을 강제한다(공백 없는 JSON으로 출력 토큰 줄이기).
        logprobs: 결과의 "score"(1~5) 토큰 확률로 기대 점수를 구해 stat["escore"]에 넣는다(못 구하면 None).
        """
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if grammar:
            body["grammar"] = grammar
        else:
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": name, "schema": schema}}
        if logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = 10
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
            out = json.loads(choice["message"]["content"])
        except json.JSONDecodeError as e:
            raise LlmError(f"JSON 파싱 실패: {e}") from None
        if logprobs:
            stat["escore"] = expected_score(choice.get("logprobs") or {})
        return out, stat


def expected_score(logprobs: dict) -> float | None:
    """"score" 키 뒤에 처음 나오는 1~5 토큰의 확률로 기대 점수(1~5 사이 실수)를 낸다.

    뽑힌 점수 하나만 쓰면 애매한 잡담과 진짜 이야기가 같은 4점에 몰린다(감지 실험 F절).
    """
    seen = False
    for t in logprobs.get("content") or []:
        tok = t.get("token", "").strip()
        if tok == "score":
            seen = True
        elif seen and tok in ("1", "2", "3", "4", "5"):
            ps: dict[str, float] = {}
            for x in t.get("top_logprobs") or []:
                k = x.get("token", "").strip()
                if k in ("1", "2", "3", "4", "5"):
                    ps[k] = ps.get(k, 0.0) + math.exp(x["logprob"])
            if not ps:
                return None
            return round(sum(int(k) * v for k, v in ps.items()) / sum(ps.values()), 3)
    return None
