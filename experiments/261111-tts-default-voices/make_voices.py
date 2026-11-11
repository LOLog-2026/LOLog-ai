"""기본 화자 24명의 TTS 목소리 만들기 (Qwen3-TTS).

1) design   : VoiceDesign 모델로 화자마다 목소리 후보 N개를 만든다 (영어 목소리 설명 + 한국어 기준 문장)
2) pick     : candidates/index.html 을 듣고 picks.txt 에 `kid-f-01 1` 처럼 24줄 적는다 (없으면 0번)
3) finalize : 고른 후보를 참조 음성으로 Base(클론) 모델에 넣어 미리 듣기 인사(<id>_preview.wav)와 예시 문장을 읽히고,
              default-voices/ 에 참조 음성·예시·voices.json 을 만든다

    python -I make_voices.py design   OUT [--only kid-f-01,young-m-01] [--n 3]
    python -I make_voices.py finalize OUT [--picks picks.txt] [--clone-model Qwen/Qwen3-TTS-12Hz-0.6B-Base]
"""

from __future__ import annotations

import argparse
import html
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
DESIGN_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
CLONE_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
SEED_BASE = 2000  # 후보 k 의 시드 = SEED_BASE + 화자 순번*10 + k


def pick_device(name: str | None) -> tuple[str, torch.dtype]:
    if name is None:
        name = "cuda:0" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    # MPS 는 bf16 연산이 일부 느리거나 빠져 있어 fp32 가 안전하다 (1.7B fp32 ≈ 7GB)
    dtype = torch.bfloat16 if name.startswith("cuda") else torch.float32
    return name, dtype


def load(model_id: str, device: str, dtype: torch.dtype):
    from qwen_tts import Qwen3TTSModel

    kw = dict(device_map=device, dtype=dtype)
    if device.startswith("cuda"):
        try:
            import flash_attn  # noqa: F401
            kw["attn_implementation"] = "flash_attention_2"
        except ImportError:
            kw["attn_implementation"] = "sdpa"
    else:
        kw["attn_implementation"] = "sdpa"
    t = time.time()
    m = Qwen3TTSModel.from_pretrained(model_id, **kw)
    print(f"[load] {model_id} on {device} {dtype} ({time.time() - t:.1f}s)", flush=True)
    return m


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def norm(wav: np.ndarray, peak_db: float = -1.0) -> np.ndarray:
    """앞뒤 무음을 자르고 최대값을 -1dBFS 로 맞춘다."""
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    thr = 10 ** (-45 / 20) * max(np.abs(wav).max(), 1e-6)
    idx = np.where(np.abs(wav) > thr)[0]
    if len(idx):
        wav = wav[max(idx[0] - 2400, 0): idx[-1] + 4800]
    peak = np.abs(wav).max()
    if peak > 0:
        wav = wav * (10 ** (peak_db / 20) / peak)
    return wav


def load_design() -> dict:
    return json.loads((HERE / "voice_design.json").read_text(encoding="utf-8"))


def cmd_design(a) -> None:
    cfg = load_design()
    ids = list(cfg["voices"])
    only = set(a.only.split(",")) if a.only else None
    out = Path(a.out) / "candidates"
    out.mkdir(parents=True, exist_ok=True)
    device, dtype = pick_device(a.device)
    model = load(a.model, device, dtype)
    log = out / "timing.jsonl"
    for i, cid in enumerate(ids):
        if only and cid not in only:
            continue
        v = cfg["voices"][cid]
        for k in range(a.n):
            path = out / f"{cid}_c{k}.wav"
            if path.exists():
                continue
            seed = SEED_BASE + i * 10 + k
            seed_all(seed)
            t = time.time()
            wavs, sr = model.generate_voice_design(
                text=v["refText"], language=cfg["language"], instruct=v["instruct"])
            dt = time.time() - t
            wav = norm(wavs[0])
            sf.write(path, wav, sr)
            dur = len(wav) / sr
            rec = dict(id=cid, cand=k, seed=seed, sec=round(dt, 2), audio=round(dur, 2),
                       rtf=round(dt / max(dur, 1e-3), 3), device=device)
            print(json.dumps(rec), flush=True)
            with log.open("a") as f:
                f.write(json.dumps(rec) + "\n")
    write_index(Path(a.out), cfg, a.n)


def write_index(root: Path, cfg: dict, n: int) -> None:
    """후보를 나란히 들어 보는 페이지 (candidates/index.html)."""
    rows = []
    for cid, v in cfg["voices"].items():
        cells = "".join(
            f'<td>c{k}<br><audio controls preload="none" src="{cid}_c{k}.wav"></audio></td>'
            for k in range(n) if (root / "candidates" / f"{cid}_c{k}.wav").exists())
        rows.append(
            f"<tr><td><b>{cid}</b><br><small>{html.escape(v['instruct'])}</small><br>"
            f"{html.escape(v['refText'])}</td>{cells}</tr>")
    page = ("<!doctype html><meta charset=utf-8><title>목소리 후보</title>"
            "<style>body{font-family:sans-serif}td{border-bottom:1px solid #ccc;padding:6px;"
            "vertical-align:top;max-width:420px}</style><table>" + "".join(rows) + "</table>")
    (root / "candidates" / "index.html").write_text(page, encoding="utf-8")


def read_picks(path: str | None) -> dict[str, int]:
    picks: dict[str, int] = {}
    if path and Path(path).exists():
        for line in Path(path).read_text().splitlines():
            parts = line.split()
            if len(parts) == 2 and not line.startswith("#"):
                picks[parts[0]] = int(parts[1])
    return picks


def cmd_finalize(a) -> None:
    cfg = load_design()
    ids = list(cfg["voices"])
    picks = read_picks(a.picks)
    root = Path(a.out)
    out = root / "default-voices"
    out.mkdir(parents=True, exist_ok=True)
    device, dtype = pick_device(a.device)
    model = load(a.clone_model, device, dtype)
    manifest = {"version": 1, "language": cfg["language"], "designModel": DESIGN_MODEL,
                "cloneModel": a.clone_model, "sampleRate": None, "voices": []}
    for i, cid in enumerate(ids):
        v = cfg["voices"][cid]
        k = picks.get(cid, 0)
        ref, sr = sf.read(root / "candidates" / f"{cid}_c{k}.wav", dtype="float32")
        sf.write(out / f"{cid}.wav", ref, sr)
        prompt = model.create_voice_clone_prompt(ref_audio=(ref, sr), ref_text=v["refText"])
        samples = []
        lines = [("preview", cfg["previewLine"])] + [
            (f"sample{j}", t) for j, t in enumerate(cfg["sampleLines"], 1)]
        for j, (tag, line) in enumerate(lines):
            name = f"{cid}_{tag}.wav"
            seed_all(SEED_BASE + i * 10 + j)
            t = time.time()
            wavs, osr = model.generate_voice_clone(
                text=line, language=cfg["language"], voice_clone_prompt=prompt)
            wav = norm(wavs[0])
            sf.write(out / name, wav, osr)
            print(json.dumps(dict(id=cid, sample=tag, sec=round(time.time() - t, 2),
                                  audio=round(len(wav) / osr, 2))), flush=True)
            if tag == "preview":
                preview = {"file": name, "text": line}
            else:
                samples.append({"file": name, "text": line})
        manifest["sampleRate"] = int(sr)
        manifest["voices"].append({
            "id": cid,
            "refAudio": f"{cid}.wav",
            "refText": v["refText"],
            "instruct": v["instruct"],
            "designSeed": SEED_BASE + i * 10 + k,
            "preview": preview,
            "samples": samples,
        })
    (out / "voices.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[done] {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("design")
    d.add_argument("out")
    d.add_argument("--only")
    d.add_argument("--n", type=int, default=3)
    d.add_argument("--model", default=DESIGN_MODEL)
    d.add_argument("--device")
    f = sub.add_parser("finalize")
    f.add_argument("out")
    f.add_argument("--picks")
    f.add_argument("--clone-model", default=CLONE_MODEL)
    f.add_argument("--device")
    a = p.parse_args()
    {"design": cmd_design, "finalize": cmd_finalize}[a.cmd](a)


if __name__ == "__main__":
    main()
