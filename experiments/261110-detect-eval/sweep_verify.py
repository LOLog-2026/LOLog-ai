"""확인 단계 기준 점수 고르기. 확인 단계 변경(verify)을 min_score 1(다 남김)로 돌린 결과에서
"몇 점 이상만 카드로 남길지"를 1~5로 바꿔 가며 지표를 다시 계산한다(LLM을 다시 부르지 않음).

카드가 확인 단계에서 빠지면: 정답과 짝이던 카드는 그 정답을 놓친 것으로, 오탐 카드는 그냥 사라진 것으로 센다.
점수를 못 받은 카드(호출 실패)는 남긴다. P/R/F1은 run_eval.aggregate와 같은 방식(녹음 합산)이고 실행마다 따로 낸 뒤 평균(최소~최대)을 적는다.

  python3 sweep_verify.py --tag srv-kc8+verify-r1
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent


def score_at(recs: list[dict], t: float, field: str = "score", drop_bc: bool = False) -> dict:
    tp = fp = ngold = 0
    dur = 0.0
    for r in recs:
        sc = {tuple(v["span"]): v.get(field) for v in r.get("verify", [])}
        bc = {tuple(v["span"]): v.get("broadcast") for v in r.get("verify", [])}
        keep = lambda span: ((sc.get(tuple(span)) is None) or sc[tuple(span)] >= t) and not (drop_bc and bc.get(tuple(span)))
        tp += sum(1 for p in r["pairs"] if keep(p["pred"]))
        fp += sum(1 for f in r["fp"] if keep(f["span"]))
        ngold += r["n_gold"]
        dur += r["dur_s"]
    p = tp / (tp + fp) if tp + fp else (1.0 if ngold == 0 else 0.0)
    rc = tp / ngold if ngold else None
    f1 = (2 * p * rc / (p + rc) if p + rc else 0.0) if rc is not None else None
    return {"P": p, "R": rc, "F1": f1, "fp_per_h": fp / (dur / 3600) if dur else 0.0}


def fmt(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    m = statistics.mean(vals)
    return f"{m:.3g}" if len(vals) == 1 or max(vals) == min(vals) else f"{m:.3g} ({min(vals):.3g}~{max(vals):.3g})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--field", default="score", choices=["score", "escore"], help="escore = 점수 토큰 확률로 낸 기대 점수")
    ap.add_argument("--drop-broadcast", action="store_true", help="확인 단계가 방송 소리(broadcast=true)라고 한 카드도 버림")
    ap.add_argument("--cuts", type=float, nargs="+", help="기준 점수 목록(기본 1~5)")
    args = ap.parse_args()
    d = Path(args.out) / args.tag
    sets = sorted({Path(f).name.split(".run")[0] for f in glob.glob(str(d / "*.run*.json"))})
    print(f"## {args.tag}: 확인 단계 기준 점수별 (이상만 남김, 1 = 거르지 않음)\n")
    print("| set | 기준 | F1 | P | R | fp/h |")
    print("|---|---|---|---|---|---|")
    for s in sets:
        runs = [json.load(open(f))["recs"] for f in sorted(glob.glob(str(d / f"{s}.run*.json")))]
        for t in (args.cuts or range(1, 6)):
            res = [score_at(recs, t, args.field, args.drop_broadcast) for recs in runs]
            print(f"| {s} | {t} | {fmt([x['F1'] for x in res])} | {fmt([x['P'] for x in res])} | "
                  f"{fmt([x['R'] for x in res])} | {fmt([x['fp_per_h'] for x in res])} |")
    # 확인 단계 비용(3090): 호출 수와 출력 토큰
    calls = [v for f in glob.glob(str(d / "*.run*.json")) for r in json.load(open(f))["recs"] for v in r.get("verify", [])]
    if calls:
        fails = sum(1 for v in calls if v["score"] is None)
        toks = [v["out_tok"] for v in calls if v.get("out_tok")]
        walls = [v["wall_s"] for v in calls if v.get("wall_s") is not None]
        print(f"\n확인 호출 {len(calls)}번, 실패 {fails}번, 출력 토큰 중앙값 {statistics.median(toks) if toks else '-'}, "
              f"호출당 시간 중앙값 {statistics.median(walls) if walls else '-'}초")


if __name__ == "__main__":
    main()
