"""Orin Nano 메모리·속도 측정: FLUX.2 klein 4컷 생성 + 얼굴 검출 + 말풍선 (실제 서비스 순서).

8GB 통합 메모리에 넣기 위해 모델을 한 번에 하나만 올린다.
  1) 텍스트 인코더만 올려 4컷 프롬프트를 한 번에 인코딩 → 인코더 삭제
  2) 이미지 모델 + VAE 올림 → 준비 실행 1번(버림) → 시드마다 4컷 생성
  3) 컷마다 얼굴 검출 → 말풍선 자리 고르기 → 그리기 → 세로 4칸 합치기
Jetson은 CPU와 GPU가 같은 메모리를 쓰므로 enable_model_cpu_offload는 메모리를 줄이지 못한다(쓰지 않음).

메모리는 /proc/meminfo(시스템 전체, 0.2초 간격)로 단계별 최댓값을 잰다. torch.cuda 최대값은 torch가 잡은
양만이라 참고용으로 같이 적는다. tegrastats가 있으면(Jetson) 원본 로그도 저장한다(GPU 사용률, 온도, 전력).

설정 하나당 한 번 실행한다(프로세스를 새로 띄워야 메모리 측정이 섞이지 않음). 결과는 --out/<tag>/에 저장:
  panels.csv     컷마다 생성·디코딩·얼굴·말풍선 시간
  summary.json   단계별 시간과 최대 메모리, 환경(전원 모드, L4T, 패키지 버전)
  tegrastats.log, *_final.png

예)
  # 3090 서버에서 스크립트 확인
  HF_HUB_OFFLINE=1 .venv/bin/python scripts/jetson_bench.py --tag server-bf16
  # Orin Nano (양자화 패키지는 따로 설치: torchao / bitsandbytes / gguf)
  python scripts/jetson_bench.py --tag int8 --te-quant int8 --quant int8
  python scripts/jetson_bench.py --tag nf4 --te-quant nf4 --quant nf4
  python scripts/jetson_bench.py --tag gguf-q4 --te-quant nf4 --quant gguf --gguf path/to/klein-4b-Q4_K_M.gguf
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import platform
import shutil
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from bubble_method_test import compose, prompt  # noqa: E402  (같은 프롬프트로 B 방식 컷을 그린다)

from lolog_ai.bubble import draw_bubbles, place_bubbles  # noqa: E402
from lolog_ai.faces import detect_faces, match_speakers  # noqa: E402

MODEL = "black-forest-labs/FLUX.2-klein-4B"
DEFAULT_SB = Path("experiments/261105-flux-klein-korean-bubble/storyboards/cake.json")


# ---------- 메모리 기록 ----------

def meminfo_mb() -> tuple[int, int]:
    """시스템 전체 사용 메모리와 스왑(MB). 통합 메모리라 GPU 사용분도 여기 포함된다."""
    m = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, v = line.split(":")
        m[k] = int(v.split()[0])
    return (m["MemTotal"] - m["MemAvailable"]) // 1024, (m["SwapTotal"] - m["SwapFree"]) // 1024


class MemSampler:
    """백그라운드에서 메모리를 기록하고, stage() 구간마다 최댓값을 남긴다."""

    def __init__(self, interval: float = 0.2):
        self.interval = interval
        self.samples: list[tuple[float, int, int]] = []
        self.stages: dict[str, dict] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self.samples.append((time.perf_counter(), *meminfo_mb()))
            self._stop.wait(self.interval)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join()

    @contextmanager
    def stage(self, name: str):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        yield
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        self.samples.append((t1, *meminfo_mb()))
        win = [s for s in self.samples if t0 <= s[0] <= t1] or [self.samples[-1]]
        self.stages[name] = {
            "sec": round(t1 - t0, 3),
            "ram_peak_mb": max(s[1] for s in win),
            "swap_peak_mb": max(s[2] for s in win),
            "torch_peak_mb": (torch.cuda.max_memory_allocated() // 2**20
                              if torch.cuda.is_available() else None),
        }
        print(f"[{name}] {self.stages[name]}", flush=True)


def start_tegrastats(path: Path) -> subprocess.Popen | None:
    exe = shutil.which("tegrastats")
    if not exe:
        return None
    f = open(path, "w")
    return subprocess.Popen([exe, "--interval", "500"], stdout=f, stderr=subprocess.STDOUT)


def run_quiet(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


def environment() -> dict:
    vers = {}
    for p in ("torch", "diffusers", "transformers", "onnxruntime", "torchao", "bitsandbytes", "gguf"):
        try:
            vers[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    l4t = Path("/etc/nv_tegra_release")
    return {
        "host": platform.node(),
        "machine": platform.machine(),
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "l4t": l4t.read_text().splitlines()[0] if l4t.exists() else None,
        "nvpmodel": run_quiet(["nvpmodel", "-q"]),  # sudo 없이 안 되면 None → README에 손으로 적기
        "packages": vers,
    }


# ---------- 모델 불러오기 (양자화 선택) ----------

def load_text_encoder(quant: str):
    from transformers import Qwen3ForCausalLM

    kw = {}
    if quant == "int8":
        from torchao.quantization import Int8WeightOnlyConfig
        from transformers import TorchAoConfig
        kw["quantization_config"] = TorchAoConfig(Int8WeightOnlyConfig())
    elif quant == "nf4":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    te = Qwen3ForCausalLM.from_pretrained(MODEL, subfolder="text_encoder", torch_dtype=torch.bfloat16,
                                          device_map="cuda", **kw)
    return te.eval()


def load_transformer(quant: str, gguf: Path | None):
    from diffusers import Flux2Transformer2DModel

    if quant == "gguf":
        from diffusers import GGUFQuantizationConfig
        return Flux2Transformer2DModel.from_single_file(
            str(gguf), config=MODEL, subfolder="transformer", torch_dtype=torch.bfloat16,
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16)).to("cuda")
    kw = {}
    if quant == "int8":
        from diffusers import TorchAoConfig
        from torchao.quantization import Int8WeightOnlyConfig
        kw["quantization_config"] = TorchAoConfig(Int8WeightOnlyConfig())
    elif quant == "nf4":
        from diffusers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = Flux2Transformer2DModel.from_pretrained(MODEL, subfolder="transformer",
                                                    torch_dtype=torch.bfloat16, **kw)
    # bitsandbytes는 로드할 때 GPU에 올라가지만, torchao는 CPU에 남으므로 직접 옮긴다
    return model if quant == "nf4" else model.to("cuda")


def free():
    """참조를 None으로 끊은 뒤 부른다."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------- 본체 ----------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="설정 이름. 결과 폴더 이름이 된다")
    ap.add_argument("--storyboard", type=Path, default=DEFAULT_SB)
    ap.add_argument("--out", type=Path, default=Path("outputs/jetson-bench"))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--panel-size", type=int, default=768)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--font-size", type=int, default=36)
    ap.add_argument("--quant", choices=["none", "int8", "nf4", "gguf"], default="none",
                    help="이미지 모델(transformer) 양자화")
    ap.add_argument("--te-quant", choices=["none", "int8", "nf4"], default="none",
                    help="텍스트 인코더(Qwen3) 양자화")
    ap.add_argument("--gguf", type=Path, help="--quant gguf일 때 .gguf 파일 경로")
    args = ap.parse_args()
    if args.quant == "gguf" and not args.gguf:
        ap.error("--quant gguf에는 --gguf 경로가 필요하다")

    out = args.out / args.tag
    out.mkdir(parents=True, exist_ok=True)
    sb = json.loads(args.storyboard.read_text(encoding="utf-8"))
    prompts = [prompt(sb, p, "bubble_code") for p in sb["panels"]]

    tegra = start_tegrastats(out / "tegrastats.log")
    mem = MemSampler().start()
    time.sleep(1.0)
    baseline = meminfo_mb()
    print(f"baseline RAM {baseline[0]}MB, swap {baseline[1]}MB", flush=True)

    summary = {"args": {k: str(v) for k, v in vars(args).items()}, "env": environment(),
               "baseline_ram_mb": baseline[0], "baseline_swap_mb": baseline[1], "status": "ok"}
    rows: list[dict] = []
    try:
        # 1) 텍스트 인코딩: 인코더만 올렸다가 지운다
        from diffusers import Flux2KleinPipeline

        with mem.stage("te_load"):
            te = load_text_encoder(args.te_quant)
            pipe = Flux2KleinPipeline.from_pretrained(MODEL, text_encoder=te, transformer=None,
                                                      vae=None, torch_dtype=torch.bfloat16)
        with mem.stage("te_encode"):
            with torch.inference_mode():
                embeds = [pipe.encode_prompt(p, device="cuda")[0] for p in prompts]
        with mem.stage("te_free"):
            pipe = te = None
            free()

        # 2) 이미지 모델 + VAE
        with mem.stage("dit_load"):
            transformer = load_transformer(args.quant, args.gguf)
            pipe = Flux2KleinPipeline.from_pretrained(MODEL, transformer=transformer, text_encoder=None,
                                                      tokenizer=None, torch_dtype=torch.bfloat16)
            pipe.vae.to("cuda")

        def generate(i: int, seed: int) -> tuple[Image.Image, float, float]:
            torch.cuda.synchronize()
            t = time.perf_counter()
            latents = pipe(prompt_embeds=embeds[i], width=args.panel_size, height=args.panel_size,
                           guidance_scale=1.0, num_inference_steps=args.steps, output_type="latent",
                           generator=torch.Generator("cuda").manual_seed(seed)).images
            torch.cuda.synchronize()
            t_denoise = time.perf_counter() - t
            t = time.perf_counter()
            with torch.inference_mode():
                img = pipe.vae.decode(latents.to(pipe.vae.dtype), return_dict=False)[0]
                img = pipe.image_processor.postprocess(img, output_type="pil")[0]
            torch.cuda.synchronize()
            return img, t_denoise, time.perf_counter() - t

        with mem.stage("warmup"):  # 첫 실행은 커널 준비로 느리다. 결과에서 뺀다
            generate(0, 12345)
            detect_faces(Image.new("RGB", (args.panel_size, args.panel_size), "white"))

        # 3) 시드마다 4컷 전체
        for seed in args.seeds:
            with mem.stage(f"comic_s{seed}"):
                panels = []
                for i, panel in enumerate(sb["panels"]):
                    raw, t_dn, t_dec = generate(i, seed)
                    t = time.perf_counter()
                    faces = detect_faces(raw)
                    t_face = time.perf_counter() - t
                    t = time.perf_counter()
                    img = raw
                    if panel["dialogue"]:
                        pos = {c["key"]: c["position"] for c in panel["cast"]}
                        placed = place_bubbles(raw, panel["dialogue"], pos, font_size=args.font_size,
                                               faces=faces, speaker_faces=match_speakers(faces, pos))
                        img = draw_bubbles(raw, panel["dialogue"], placed, font_size=args.font_size)
                    t_bubble = time.perf_counter() - t
                    panels.append(img)
                    rows.append({"seed": seed, "panel": panel["index"], "denoise_s": round(t_dn, 3),
                                 "decode_s": round(t_dec, 3), "face_s": round(t_face, 3),
                                 "bubble_s": round(t_bubble, 3), "faces": len(faces),
                                 "dialogue": len(panel["dialogue"])})
                    print(rows[-1], flush=True)
                compose(panels).save(out / f"{args.storyboard.stem}_s{seed}_final.png")
    except Exception as e:  # 메모리 부족 등으로 죽어도 그때까지 잰 값은 남긴다
        summary["status"] = f"failed: {type(e).__name__}: {e}"
        print(summary["status"], flush=True)
        raise
    finally:
        mem.stop()
        if tegra:
            tegra.terminate()
        summary["stages"] = mem.stages
        summary["ram_peak_mb"] = max(s[1] for s in mem.samples)
        summary["swap_peak_mb"] = max(s[2] for s in mem.samples)
        comics = [v["sec"] for k, v in mem.stages.items() if k.startswith("comic_")]
        if comics:
            summary["comic_sec_median"] = sorted(comics)[len(comics) // 2]
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        if rows:
            with open(out / "panels.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
        print(f"saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
