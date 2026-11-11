"""컨텐츠포인트 감지 채점. 워커의 detect.py를 그대로 불러 평가셋에 돌리고 평가 기준 문서 2절 지표를 낸다.

표준 라이브러리만 쓴다. LLM은 llama-server(OpenAI 호환 /v1)를 HTTP로 부른다.

지표 (평가 기준 2절)
  P / R / F1      찾은 카드와 정답 이야기를 짝짓기. 줄 번호 IoU 0.3 이상이거나 정답 펀치라인 구간을 모두 품으면 맞음
  punch           맞은 카드 중 정답 펀치라인 구간이 모두 from~to 안에 든 비율
  start / end     맞은 카드의 시작·끝 시간 차이 중앙값(초, 절댓값). late = 시작이 3초 넘게 늦은(빌드업 빠짐) 비율
  kind            맞은 카드 중 종류(웃김/황당/추억)가 같은 비율
  spk             맞은 카드의 대사 화자 정확도(한 사람만 말한 구간, 화자 이름 바꿈 허용)
  fp              정답과 짝이 없는 카드 수. nostory·tv 세트는 이것이 곧 오탐
  fmt             JSON·길이 오류로 녹음 하나가 통째로 실패한 수 + 형식 검사로 버려진 카드 수
  time            녹음당 처리 시간 합(초)
  --skip-speakers 면 spk는 비고(-), 나머지 지표는 같다

예)
  python3 run_eval.py --sets sets/short-stt sets/long-stt sets/nostory-stt sets/tv --runs 3 --tag base
  python3 run_eval.py --sets sets/short-stt --variant temp0 --tag temp0 --runs 1 --skip-speakers
  python3 run_eval.py --sets sets/short-stt --mock gold --runs 1 --tag selftest   # LLM 없이 채점기 확인 (F1=1이 나와야 함)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import itertools
import json
import logging
import math
import os
import random
import statistics
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

import variants

from lolog_ai.worker import detect as det
from lolog_ai.worker.llm import Llm, LlmError

IOU_MIN = 0.3
LATE_S = 3.0


# ---------- LLM 감싸기: 호출을 기록하고 온도를 바꾼다 ----------

class RecLlm(Llm):
    def __init__(self, *a, temperature: float = 0.2, skip_speakers: bool = False,
                 chunk: dict | None = None, verify: dict | None = None, speaker: dict | None = None,
                 gold_detect: bool = False, **kw):
        super().__init__(*a, **kw)
        self.temperature = temperature
        self.skip_speakers = skip_speakers
        self.chunk = chunk          # {"min": 창 길이(분), "overlap": 겹침(분)}. 있으면 이야기 찾기를 시간 창으로 나눠 부른다
        self.speaker = speaker      # {"system", "mode": "notext"|"pairs"}. 있으면 화자 나누기가 대사를 다시 쓰지 않는다
        self.gold_detect = gold_detect  # 이야기 찾기는 정답을 그대로 돌려줌(화자 나누기만 따로 잴 때)
        self.gold = None
        self.verify = verify        # {"system", "reason": bool, "min_score"}. 있으면 후보마다 1~5점을 매겨 거른다
        self.cur_sents: list[dict] = []  # run_set이 녹음마다 채움(detect()와 같은 startMs 정렬)
        self.calls: list[dict] = []
        self.rng = random.Random()

    def _post(self, path, body):
        # 요청마다 seed를 새로 뽑는다. 안 주면 같은 실행 안의 반복끼리 비슷하게 나와(서버 seed 고정)
        # 3번 범위가 실제 흔들림보다 좁게 잡혔다(같은 설정의 nostory2 오탐이 실행마다 7 / 13 / 19)
        if path == "/chat/completions":
            body = {**body, "seed": self.rng.randrange(1, 2**31)}
        return super()._post(path, body)

    def chat_json(self, system, user, schema, name, temperature=None, max_tokens=2048):
        if name == "speakers" and self.skip_speakers:  # 이야기 찾기만 잴 때: 화자는 부르지 않고 한 명으로 채움
            res = {"lines": [{"seg": first_seg(user), "speaker": "화자1", "text": "-"}]}
            stat = {"wall_s": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "finish_reason": "stop"}
            self.calls.append({"name": name, "user": user, "result": res, "stat": stat, "skipped": True})
            return res, stat
        if name == "detect" and self.gold_detect:
            res = {"stories": [{"kind": g["kind"], "summary": g["id"], "from": g["from"], "to": g["to"],
                                "finished": True} for g in self.gold["stories"]]}
            return res, {"wall_s": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "finish_reason": "stop"}
        if name == "speakers" and self.speaker:
            return self._speakers_compact(user, max_tokens)
        if name == "detect" and self.chunk:
            res, stat = self._detect_chunked(system, user, schema, max_tokens)
        else:
            res, stat = super().chat_json(system, user, schema, name,
                                          temperature=self.temperature, max_tokens=max_tokens)
            self.calls.append({"name": name, "user": user, "result": res, "stat": stat})
        if name == "detect" and self.verify:
            res = self._verify(user, res)
        return res, stat

    def _verify(self, user, res):
        """확인 단계 실험: 후보(워커와 같은 tidy를 거친 것)마다 그 구간만 보여 주고 1~5점을 받는다.
        점수와 구간을 calls에 남겨 sweep_verify.py가 기준 점수별 결과를 다시 계산한다. 호출이 실패하면 거르지 않고 남긴다."""
        lines = user.split("\n")
        stories = det.tidy([s for s in res.get("stories", []) if 0 <= s["from"] <= s["to"] < len(lines)])
        props = {"score": {"type": "integer", "enum": [1, 2, 3, 4, 5]}}
        if self.verify.get("reason"):
            props = {"reason": {"type": "string", "maxLength": 60}, **props}
        if self.verify.get("broadcast"):  # 방송 소리인지 먼저 따로 묻는다
            props = {"broadcast": {"type": "boolean"}, **props}
        schema = {"type": "object", "properties": props, "required": list(props)}
        kept = []
        for s in stories:
            sub = "\n".join(lines[s["from"]:s["to"] + 1])
            escore = None
            try:
                if self.verify.get("logprobs"):
                    r, st, escore = self._verify_logprobs(sub, schema)
                else:
                    r, st = super().chat_json(self.verify["system"], sub, schema, "verify",
                                              temperature=self.temperature, max_tokens=1024)
                score = r.get("score")
            except (LlmError, KeyError, ValueError) as e:
                r, st, score = {"error": str(e)}, {}, None
            self.calls.append({"name": "verify", "user": sub, "result": r, "stat": st,
                               "span": [s["from"], s["to"]], "score": score, "escore": escore,
                               "broadcast": r.get("broadcast")})
            if score is None or score >= self.verify.get("min_score", 1):
                kept.append(s)
        return {"stories": kept}

    PAIRS_GRAMMAR = 'root ::= "[" pair ("," pair)* "]"\npair ::= "[" [0-9]+ "," [1-4] "]"\n'

    LINES_GRAMMAR = r'''root ::= "{\"lines\":[" line ("," line)* "]}"
line ::= "{\"seg\":" [0-9]+ ",\"speaker\":" spk ",\"text\":" str "}"
spk ::= "\"화자1\"" | "\"화자2\"" | "\"화자3\"" | "\"화자4\""
str ::= "\"" ( [^"\\\x7F\x00-\x1F] | "\\" ["\\/bfnrt] )* "\""
'''

    def _speakers_compact(self, user, max_tokens):
        """화자 나누기 출력 줄이기 실험: 줄 번호와 화자만 받고 대사는 입력 글을 그대로 쓴다.
        notext = 지금 스키마에서 text만 뺌. pairs = [[줄 번호, 화자 번호], ...]를 문법으로 공백 없이 강제."""
        text_of = {int(l.split("]")[0][1:]): l.split("] ", 1)[1] if "] " in l else "" for l in user.split("\n")}
        mode = self.speaker["mode"]
        body = {"model": self.model, "temperature": self.temperature, "max_tokens": max_tokens,
                "messages": [{"role": "system", "content": self.speaker["system"] or det.SPEAKER_SYSTEM},
                             {"role": "user", "content": user}]}
        if mode == "pairs":
            body["grammar"] = self.PAIRS_GRAMMAR
        elif mode == "json":  # 지금 출력과 같은 내용(대사 포함)을 공백 없는 JSON으로만 강제
            body["grammar"] = self.LINES_GRAMMAR
        else:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "speakers", "schema": {
                "type": "object", "required": ["lines"], "properties": {"lines": {"type": "array", "minItems": 1, "items": {
                    "type": "object", "required": ["seg", "speaker"],
                    "properties": {"seg": {"type": "integer"}, "speaker": {"type": "string", "enum": det.SPEAKERS}}}}}}}}
        t0 = time.perf_counter()
        resp = self._post("/chat/completions", body)
        ch = resp["choices"][0]
        stat = {"wall_s": round(time.perf_counter() - t0, 1), "prompt_tokens": resp.get("usage", {}).get("prompt_tokens"),
                "completion_tokens": resp.get("usage", {}).get("completion_tokens"), "finish_reason": ch.get("finish_reason")}
        if ch.get("finish_reason") == "length":
            raise LlmError(f"출력이 max_tokens({max_tokens})에서 잘림")
        try:
            raw = json.loads(ch["message"]["content"])
        except json.JSONDecodeError as e:
            raise LlmError(f"JSON 파싱 실패: {e}") from None
        if mode == "json":
            res = {"lines": raw.get("lines", [])}
        else:
            pairs = [(a, f"화자{b}") for a, b in raw] if mode == "pairs" else [(l["seg"], l["speaker"]) for l in raw.get("lines", [])]
            res = {"lines": [{"seg": a, "speaker": sp, "text": text_of[a]} for a, sp in pairs if text_of.get(a)]}
        self.calls.append({"name": "speakers", "user": user, "result": res, "stat": stat})
        return res, stat

    def _verify_logprobs(self, sub, schema):
        """점수 토큰의 확률(top_logprobs)로 기대 점수(1~5 사이 실수)를 낸다. 출력 토큰은 늘지 않는다."""
        body = {"model": self.model, "temperature": self.temperature, "max_tokens": 1024,
                "logprobs": True, "top_logprobs": 10,
                "messages": [{"role": "system", "content": self.verify["system"]}, {"role": "user", "content": sub}],
                "response_format": {"type": "json_schema", "json_schema": {"name": "verify", "schema": schema}}}
        t0 = time.perf_counter()
        resp = self._post("/chat/completions", body)
        ch = resp["choices"][0]
        st = {"wall_s": round(time.perf_counter() - t0, 1), "prompt_tokens": resp.get("usage", {}).get("prompt_tokens"),
              "completion_tokens": resp.get("usage", {}).get("completion_tokens"), "finish_reason": ch.get("finish_reason")}
        r = json.loads(ch["message"]["content"])
        toks = ch["logprobs"]["content"]
        escore, seen = None, False
        for t in toks:  # "score" 키 뒤에 처음 나오는 숫자 토큰
            if t["token"].strip() == "score":
                seen = True
            elif seen and t["token"].strip() in ("1", "2", "3", "4", "5"):
                ps = {}
                for x in t.get("top_logprobs", []):
                    k = x["token"].strip()
                    if k in ("1", "2", "3", "4", "5"):
                        ps[k] = ps.get(k, 0.0) + math.exp(x["logprob"])
                if ps:
                    escore = round(sum(int(k) * v for k, v in ps.items()) / sum(ps.values()), 3)
                break
        return r, st, escore

    def _detect_chunked(self, system, user, schema, max_tokens):
        """긴 입력 나누기 실험: 시간 창(min분, overlap분 겹침)마다 따로 찾고 합친다.
        줄 번호는 원래 번호를 그대로 넣으므로 다시 맞출 필요가 없고, 겹친 곳의 중복은 워커의 tidy()가 긴 쪽만 남긴다."""
        lines = user.split("\n")
        sents = self.cur_sents
        assert len(lines) == len(sents), "cur_sents가 입력과 맞지 않음"
        win, step = self.chunk["min"] * 60000, (self.chunk["min"] - self.chunk["overlap"]) * 60000
        t0, t_end = sents[0]["startMs"], sents[-1]["startMs"]
        windows, w = [], t0
        while True:
            idx = [i for i, s in enumerate(sents) if w <= s["startMs"] < w + win]
            if idx:
                windows.append(idx)
            if w + win > t_end:
                break
            w += step
        stories, stat_sum = [], {"wall_s": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "finish_reason": "stop"}
        for idx in windows:
            sub = "\n".join(lines[i] for i in idx)
            res, stat = super().chat_json(system, sub, schema, "detect",
                                          temperature=self.temperature, max_tokens=max_tokens)
            self.calls.append({"name": "detect", "user": sub, "result": res, "stat": stat})
            lo, hi = idx[0], idx[-1]
            stories += [s for s in res.get("stories", []) if lo <= s["from"] <= s["to"] <= hi]
            for k in ("wall_s", "prompt_tokens", "completion_tokens"):
                stat_sum[k] += stat.get(k) or 0
            if stat.get("finish_reason") != "stop":
                stat_sum["finish_reason"] = stat.get("finish_reason")
        stat_sum["windows"] = len(windows)
        return {"stories": stories}, stat_sum


class GoldLlm:
    """채점기 확인용: 정답을 그대로 돌려준다."""

    def __init__(self):
        self.gold = None
        self.calls: list[dict] = []

    def chat_json(self, system, user, schema, name, **kw):
        if name == "detect":
            res = {"stories": [{"kind": g["kind"], "summary": g["id"], "from": g["from"], "to": g["to"],
                                "finished": True} for g in self.gold["stories"]]}
        else:
            segs = [int(l.split("]")[0][1:]) for l in user.splitlines()]
            g = next(g for g in self.gold["stories"] if g["from"] == segs[0])
            name_of = {}
            lines = []
            for k in segs:
                spk = g["speakers"].get(str(k))
                if spk:
                    name_of.setdefault(spk, f"화자{len(name_of) + 1}")
                    lines.append({"seg": k, "speaker": name_of[spk], "text": "x"})
            res = {"lines": lines or [{"seg": segs[0], "speaker": "화자1", "text": "x"}]}
        stat = {"wall_s": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "finish_reason": "stop"}
        self.calls.append({"name": name, "user": user, "result": res, "stat": stat})
        return res, stat


class DropCounter(logging.Handler):
    """워커가 버린 카드 수(경고 로그)를 센다. 녹음을 여러 스레드로 돌려도 섞이지 않게 스레드마다 따로 센다."""

    def __init__(self):
        super().__init__(logging.WARNING)
        self._local = threading.local()

    @property
    def n(self):
        return getattr(self._local, "n", 0)

    @n.setter
    def n(self, v):
        self._local.n = v

    def emit(self, record):
        self.n += 1


# ---------- 채점 ----------

def iou(a, b, c, d) -> float:
    inter = max(0, min(b, d) - max(a, c) + 1)
    return inter / (max(b, d) - min(a, c) + 1)


def first_seg(user: str) -> int:
    return int(user.splitlines()[0].split("]")[0][1:])


def predictions(result: dict, calls: list[dict], sents: list[dict]) -> list[dict]:
    by_start = {s["startMs"]: i for i, s in enumerate(sents)}
    by_end = {s["endMs"]: i for i, s in enumerate(sents)}
    spk_calls = {first_seg(c["user"]): c["result"].get("lines", []) for c in calls if c["name"] == "speakers"}
    skipped = {first_seg(c["user"]) for c in calls if c.get("skipped")}
    out = []
    for c in result["cards"]:
        f, t = by_start[c["startMs"]], by_end[c["endMs"]]
        seg_spk: dict[int, list[str]] = {}
        for l in spk_calls.get(f, []):
            seg_spk.setdefault(l["seg"], []).append(l["speaker"])
        out.append({"from": f, "to": t, "kind": c["kind"], "summary": c["summary"],
                    "speakers": None if f in skipped else
                    {k: max(set(v), key=v.count) for k, v in seg_spk.items()}})
    return out


def match(preds: list[dict], golds: list[dict]) -> list[tuple[int, int]]:
    cand = []
    for i, p in enumerate(preds):
        for j, g in enumerate(golds):
            v = iou(p["from"], p["to"], g["from"], g["to"])
            covers = g["punch"] and all(p["from"] <= k <= p["to"] for k in g["punch"])
            if v >= IOU_MIN or covers:
                cand.append((v, i, j))
    used_p, used_g, pairs = set(), set(), []
    for v, i, j in sorted(cand, reverse=True):
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
            pairs.append((i, j))
    return pairs


def speaker_score(p: dict, g: dict) -> tuple[int, int]:
    keys = [k for k in g["speakers"] if int(k) in p["speakers"]]
    total = len(g["speakers"])  # 정답 대사 중 화자를 못 붙인 것도 틀림으로 센다
    if not keys:
        return 0, total
    gnames = sorted({g["speakers"][k] for k in g["speakers"]})
    pnames = sorted({p["speakers"][int(k)] for k in keys})
    best = 0
    for perm in itertools.permutations(pnames + [None] * max(0, len(gnames) - len(pnames)), len(gnames)):
        m = dict(zip(gnames, perm))
        best = max(best, sum(1 for k in keys if m[g["speakers"][k]] == p["speakers"][int(k)]))
    return best, total


def score_rec(preds, gold, sents) -> dict:
    golds = gold["stories"]
    pairs = match(preds, golds)
    r = {"n_pred": len(preds), "n_gold": len(golds), "n_match": len(pairs), "punch_ok": 0, "kind_ok": 0,
         "start_err": [], "end_err": [], "late": 0, "spk_ok": 0, "spk_total": 0, "pairs": []}
    for i, j in pairs:
        p, g = preds[i], golds[j]
        r["punch_ok"] += all(p["from"] <= k <= p["to"] for k in g["punch"])
        r["kind_ok"] += p["kind"] == g["kind"]
        ds = (sents[p["from"]]["startMs"] - sents[g["from"]]["startMs"]) / 1000
        de = (sents[p["to"]]["endMs"] - sents[g["to"]]["endMs"]) / 1000
        r["start_err"].append(abs(ds))
        r["end_err"].append(abs(de))
        r["late"] += ds > LATE_S
        ok, tot = speaker_score(p, g) if p["speakers"] is not None else (0, 0)
        r["spk_ok"] += ok
        r["spk_total"] += tot
        r["pairs"].append({"gold": g["id"], "pred": [p["from"], p["to"]], "gold_span": [g["from"], g["to"]],
                           "punch": g["punch"], "kind": [p["kind"], g["kind"]], "start_s": ds, "end_s": de,
                           "spk": [ok, tot]})
    matched_p = {i for i, _ in pairs}
    r["fp"] = [{"span": [p["from"], p["to"]], "kind": p["kind"], "summary": p["summary"],
                "tv": any(k in gold.get("tvSegs", []) for k in range(p["from"], p["to"] + 1))}
               for i, p in enumerate(preds) if i not in matched_p]
    r["missed"] = [g["id"] for j, g in enumerate(golds) if j not in {j for _, j in pairs}]
    return r


def aggregate(recs: list[dict]) -> dict:
    s = lambda k: sum(r[k] for r in recs)
    npred, ngold, nm = s("n_pred"), s("n_gold"), s("n_match")
    p = nm / npred if npred else (1.0 if ngold == 0 else 0.0)
    rcl = nm / ngold if ngold else None
    f1 = (2 * p * rcl / (p + rcl) if p + rcl else 0.0) if rcl is not None else None
    st = [x for r in recs for x in r["start_err"]]
    en = [x for r in recs for x in r["end_err"]]
    dur_h = sum(r["dur_s"] for r in recs) / 3600
    return {
        "P": round(p, 3), "R": rcl if rcl is None else round(rcl, 3), "F1": f1 if f1 is None else round(f1, 3),
        "punch": round(s("punch_ok") / nm, 3) if nm else None,
        "start": round(statistics.median(st), 1) if st else None,
        "end": round(statistics.median(en), 1) if en else None,
        "late": round(s("late") / nm, 3) if nm else None,
        "kind": round(s("kind_ok") / nm, 3) if nm else None,
        "spk": round(s("spk_ok") / s("spk_total"), 3) if s("spk_total") else None,
        "fp": sum(len(r["fp"]) for r in recs), "fp_per_h": round(sum(len(r["fp"]) for r in recs) / dur_h, 1),
        "fmt": s("fmt_fail") + s("dropped"),
        "time": round(s("wall_s"), 1),
        "prompt_tok": s("prompt_tok"),
    }


def run_set(set_dir: Path, llm0, mock: bool, drop: DropCounter, limit: int | None, jobs: int = 1) -> list[dict]:
    files = [f for f in sorted(set_dir.glob("*.json")) if not f.name.endswith(".gold.json")]

    def one(f):
        llm = copy.copy(llm0)  # 녹음마다 따로(호출 기록·정답이 스레드끼리 섞이지 않게)
        msg = json.loads(f.read_text())
        gold = json.loads(f.with_name(f.stem + ".gold.json").read_text())
        sents = sorted(msg["sentences"], key=lambda s: s["startMs"])
        llm.calls = []
        llm.cur_sents = sents
        if mock or getattr(llm, "gold_detect", False):
            llm.gold = gold
        drop.n = 0
        t0 = time.perf_counter()
        fmt_fail, err = 0, None
        try:
            result = det.detect(msg, llm)
        except (LlmError, KeyError, ValueError, TypeError) as e:  # 실서비스에선 이 녹음이 다시 시도됨
            fmt_fail, err, result = 1, str(e), {"cards": []}
        wall = time.perf_counter() - t0
        preds = predictions(result, llm.calls, sents)
        r = score_rec(preds, gold, sents)
        r.update(rec=f.stem, fmt_fail=fmt_fail, error=err, dropped=drop.n, wall_s=round(wall, 1),
                 dur_s=(sents[-1]["endMs"] - sents[0]["startMs"]) / 1000 if sents else 0,
                 prompt_tok=sum(c["stat"].get("prompt_tokens") or 0 for c in llm.calls),
                 raw_detect=[c["result"] for c in llm.calls if c["name"] == "detect"],
                 verify=[{"span": c["span"], "score": c["score"], "escore": c.get("escore"), "broadcast": c.get("broadcast"), "reason": c["result"].get("reason") or c["result"].get("error"),
                          "wall_s": c["stat"].get("wall_s"), "out_tok": c["stat"].get("completion_tokens")}
                         for c in llm.calls if c["name"] == "verify"],
                 # 호출별 토큰·시간. 젯슨은 출력 속도(약 11토큰/초)가 병목이라 어디서 출력을 쓰는지 본다
                 call_stats=[{"name": c["name"], "in": c["stat"].get("prompt_tokens"),
                              "out": c["stat"].get("completion_tokens"), "wall_s": c["stat"].get("wall_s")}
                             for c in llm.calls if not c.get("skipped")])
        print(f"  {set_dir.name}/{f.stem}: 카드 {r['n_pred']} 정답 {r['n_gold']} 맞음 {r['n_match']} "
              f"오탐 {len(r['fp'])} {wall:.0f}s{' 실패: ' + err if err else ''}", flush=True)
        return r

    if jobs <= 1:
        return [one(f) for f in files[:limit]]
    # 정확도는 그대로고 시간(wall_s, time)만 서로 섞인다. 서버를 -np jobs 이상으로 띄워야 한다
    with concurrent.futures.ThreadPoolExecutor(jobs) as ex:
        return list(ex.map(one, files[:limit]))


def fmt_range(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    m = statistics.mean(vals)
    if len(vals) == 1 or max(vals) == min(vals):
        return f"{m:.3g}"
    return f"{m:.3g} ({min(vals):.3g}~{max(vals):.3g})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--variant", default="base", choices=sorted(variants.VARIANTS))
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--api-key", default=os.environ.get("LLM_API_KEY"))
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL", "local"))
    ap.add_argument("--mock", choices=["gold"])
    ap.add_argument("--skip-speakers", action="store_true",
                    help="화자 나누기를 부르지 않음(이야기 찾기만 비교할 때, 시간 약 1/3)")
    ap.add_argument("--gold-detect", action="store_true",
                    help="이야기 찾기는 정답을 그대로 쓰고 화자 나누기만 실제 모델로 잼(같은 이야기로 화자 방식 비교)")
    ap.add_argument("--jobs", type=int, default=1,
                    help="녹음을 동시에 몇 개 돌릴지(연구 서버에서 빨리 잴 때. llama-server를 -np N으로 띄워야 함. 시간 지표는 못 씀)")
    ap.add_argument("--limit", type=int, help="세트마다 앞에서 N개 녹음만 (빠른 확인용)")
    ap.add_argument("--out", default=str(HERE / "results"))
    args = ap.parse_args()

    v = variants.VARIANTS[args.variant]
    variants.apply(det, v)
    drop = DropCounter()
    logging.getLogger(det.__name__).addHandler(drop)
    logging.getLogger(det.__name__).setLevel(logging.INFO)

    if args.mock:
        llm = GoldLlm()
    else:
        llm = RecLlm(args.base_url, args.model, args.api_key, temperature=v.get("temperature", 0.2),
                     skip_speakers=args.skip_speakers, chunk=v.get("detect_chunk"), verify=v.get("verify"),
                     speaker=v.get("speaker"), gold_detect=args.gold_detect)
        llm.wait_ready(60)

    out = Path(args.out) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    table: dict[str, list[dict]] = {}
    for run in range(1, args.runs + 1):
        print(f"[{args.tag}] {run}/{args.runs}번째", flush=True)
        for sd in args.sets:
            sd = Path(sd)
            recs = run_set(sd, llm, bool(args.mock), drop, args.limit, args.jobs)
            agg = aggregate(recs)
            table.setdefault(sd.name, []).append(agg)
            (out / f"{sd.name}.run{run}.json").write_text(
                json.dumps({"agg": agg, "recs": recs}, ensure_ascii=False, indent=1))
            with open(Path(args.out) / "results.jsonl", "a") as fp:
                fp.write(json.dumps({"tag": args.tag, "variant": args.variant, "set": sd.name, "run": run,
                                     "skip_speakers": args.skip_speakers, "limit": args.limit,
                                     "time": time.strftime("%Y-%m-%d %H:%M"), **agg}, ensure_ascii=False) + "\n")

    keys = ["F1", "P", "R", "punch", "start", "end", "late", "kind", "spk", "fp", "fp_per_h", "fmt", "time"]
    print(f"\n## {args.tag} (variant={args.variant}, runs={args.runs})\n")
    print("| set | " + " | ".join(keys) + " |")
    print("|---" * (len(keys) + 1) + "|")
    for name, aggs in table.items():
        print(f"| {name} | " + " | ".join(fmt_range([a[k] for a in aggs]) for k in keys) + " |")
    summary = {name: {k: [a[k] for a in aggs] for k in keys} for name, aggs in table.items()}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
