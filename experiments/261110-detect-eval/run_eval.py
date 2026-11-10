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

예)
  python3 run_eval.py --sets sets/short-stt sets/long-stt sets/nostory-stt sets/tv --runs 3 --tag base
  python3 run_eval.py --sets sets/short-clean --variant temp0 --tag temp0
  python3 run_eval.py --sets sets/short-stt --mock gold --runs 1 --tag selftest   # LLM 없이 채점기 확인 (F1=1이 나와야 함)
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import statistics
import sys
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
    def __init__(self, *a, temperature: float = 0.2, **kw):
        super().__init__(*a, **kw)
        self.temperature = temperature
        self.calls: list[dict] = []

    def chat_json(self, system, user, schema, name, temperature=None, max_tokens=2048):
        res, stat = super().chat_json(system, user, schema, name,
                                      temperature=self.temperature, max_tokens=max_tokens)
        self.calls.append({"name": name, "user": user, "result": res, "stat": stat})
        return res, stat


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
    def __init__(self):
        super().__init__(logging.WARNING)
        self.n = 0

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
    out = []
    for c in result["cards"]:
        f, t = by_start[c["startMs"]], by_end[c["endMs"]]
        seg_spk: dict[int, list[str]] = {}
        for l in spk_calls.get(f, []):
            seg_spk.setdefault(l["seg"], []).append(l["speaker"])
        out.append({"from": f, "to": t, "kind": c["kind"], "summary": c["summary"],
                    "speakers": {k: max(set(v), key=v.count) for k, v in seg_spk.items()}})
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
        ok, tot = speaker_score(p, g)
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


def run_set(set_dir: Path, llm, mock: bool, drop: DropCounter) -> list[dict]:
    recs = []
    for f in sorted(set_dir.glob("*.json")):
        if f.name.endswith(".gold.json"):
            continue
        msg = json.loads(f.read_text())
        gold = json.loads(f.with_name(f.stem + ".gold.json").read_text())
        sents = sorted(msg["sentences"], key=lambda s: s["startMs"])
        llm.calls = []
        if mock:
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
                 raw_detect=[c["result"] for c in llm.calls if c["name"] == "detect"])
        recs.append(r)
        print(f"  {set_dir.name}/{f.stem}: 카드 {r['n_pred']} 정답 {r['n_gold']} 맞음 {r['n_match']} "
              f"오탐 {len(r['fp'])} {wall:.0f}s{' 실패: ' + err if err else ''}", flush=True)
    return recs


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
        llm = RecLlm(args.base_url, args.model, args.api_key, temperature=v.get("temperature", 0.2))
        llm.wait_ready(60)

    out = Path(args.out) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    table: dict[str, list[dict]] = {}
    for run in range(1, args.runs + 1):
        print(f"[{args.tag}] {run}/{args.runs}번째", flush=True)
        for sd in args.sets:
            sd = Path(sd)
            recs = run_set(sd, llm, bool(args.mock), drop)
            agg = aggregate(recs)
            table.setdefault(sd.name, []).append(agg)
            (out / f"{sd.name}.run{run}.json").write_text(
                json.dumps({"agg": agg, "recs": recs}, ensure_ascii=False, indent=1))
            with open(Path(args.out) / "results.jsonl", "a") as fp:
                fp.write(json.dumps({"tag": args.tag, "variant": args.variant, "set": sd.name, "run": run,
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
