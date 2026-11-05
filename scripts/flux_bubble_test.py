"""FLUX.2 klein 한글 말풍선 생성 테스트.

prompts.yaml의 케이스마다 여러 시드로 이미지를 만들고, 생성 시간과 GPU 최대 메모리를
results.csv에 기록한다. 이미지와 csv는 --out 폴더(기본 outputs/, git 제외)에 저장한다.

예)
  .venv/bin/python scripts/flux_bubble_test.py \
      --prompts experiments/261105-flux-klein-korean-bubble/prompts.yaml \
      --out outputs/flux-klein-bubble --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
import yaml
from diffusers import Flux2KleinPipeline


def build_prompt(cfg: dict, case: dict) -> str:
    bubbles = case["bubbles"]
    if case.get("bubble_text"):  # 대조 실험용: 말풍선 문장을 직접 지정
        text = case["bubble_text"]
    elif len(bubbles) == 1:
        text = f'There is one speech bubble that says "{bubbles[0]}".'
    else:
        parts = [f'speech bubble {i + 1} says "{b}"' for i, b in enumerate(bubbles)]
        text = f"There are {len(bubbles)} speech bubbles: " + ", ".join(parts) + "."
    if case.get("raw"):  # 대조 실험용: 한글 지시 문장을 붙이지 않는다
        return f"{cfg['style']}. {case['scene'].strip()}. {text}"
    return cfg["template"].format(style=cfg["style"], scene=case["scene"].strip(), bubbles=text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True, type=Path)
    ap.add_argument("--out", default=Path("outputs/flux-klein-bubble"), type=Path)
    ap.add_argument("--model", default="black-forest-labs/FLUX.2-klein-4B")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--offload", action="store_true", help="enable_model_cpu_offload (메모리 절약)")
    ap.add_argument("--only", nargs="*", help="이 id만 실행")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.prompts.read_text(encoding="utf-8"))
    cases = [dict(c, kind="single") for c in cfg["cases"]]
    cases += [dict(c, kind="four_panel", length="four_panel") for c in cfg.get("four_panel", [])]
    if args.only:
        cases = [c for c in cases if c["id"] in set(args.only)]

    args.out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    pipe = Flux2KleinPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    if args.offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    print(f"load: {time.perf_counter() - t0:.1f}s")

    rows = []
    for case in cases:
        prompt = build_prompt(cfg, case)
        for seed in args.seeds:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t = time.perf_counter()
            image = pipe(
                prompt=prompt,
                height=args.size,
                width=args.size,
                guidance_scale=1.0,
                num_inference_steps=args.steps,
                generator=torch.Generator(device="cuda").manual_seed(seed),
            ).images[0]
            torch.cuda.synchronize()
            sec = time.perf_counter() - t
            peak = torch.cuda.max_memory_allocated() / 2**30

            name = f"{case['id']}_seed{seed}.png"
            image.save(args.out / name)
            rows.append(
                {
                    "id": case["id"],
                    "kind": case["kind"],
                    "length": case["length"],
                    "seed": seed,
                    "expected": " | ".join(case["bubbles"]),
                    "time_s": f"{sec:.2f}",
                    "peak_gpu_gb": f"{peak:.2f}",
                    "file": name,
                    "prompt": prompt,
                }
            )
            print(f"{name}: {sec:.2f}s, peak {peak:.2f}GB")

    with open(args.out / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {len(rows)} images -> {args.out}")


if __name__ == "__main__":
    main()
