"""호출 종류별 토큰과 젯슨 예상 시간. run_eval.py 결과의 call_stats를 모은다.

젯슨 1호(Orin Nano 8GB, llama.cpp b12201, --cache-ram 0) 실측 속도로 환산한다:
입력 약 300토큰/초, 출력 약 10.9토큰/초(2026-11-10). 출력이 시간 대부분이다.

  python3 token_cost.py --tag tok-base [--in-tps 300 --out-tps 10.9]
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", nargs="+", required=True)
    ap.add_argument("--in-tps", type=float, default=300.0)
    ap.add_argument("--out-tps", type=float, default=10.9)
    args = ap.parse_args()
    print("| tag | set | 녹음 분 | 호출 | 호출 수 | 입력 토큰 | 출력 토큰 | 호출당 출력 | 젯슨 예상(초) | 1시간 환산(분) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for tag in args.tag:
        for f in sorted(glob.glob(str(HERE / "results" / tag / "*.run1.json"))):
            recs = json.load(open(f))["recs"]
            minutes = sum(r["dur_s"] for r in recs) / 60
            agg = collections.defaultdict(lambda: [0, 0, 0])
            for r in recs:
                for c in r.get("call_stats", []):
                    a = agg[c["name"]]
                    a[0] += 1
                    a[1] += c["in"] or 0
                    a[2] += c["out"] or 0
            total = 0.0
            for name in ("detect", "verify", "speakers"):
                if name not in agg:
                    continue
                n, i, o = agg[name]
                sec = i / args.in_tps + o / args.out_tps
                total += sec
                print(f"| {tag} | {Path(f).name.split('.run')[0]} | {minutes:.0f} | {name} | {n} | {i} | {o} | {o / n:.0f} | "
                      f"{sec:.0f} | {sec / minutes:.1f} |")
            print(f"| {tag} | | | **합계** | | | | | **{total:.0f}** | **{total / minutes:.1f}** |")


if __name__ == "__main__":
    main()
