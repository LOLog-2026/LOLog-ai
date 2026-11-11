"""기본 화자 24명의 TTS 목소리 만들기 (Qwen3-TTS VoiceDesign).

모든 화자가 같은 인사("안녕하세요! 만나서 반가워요.")를 읽고, 목소리만 그림에 맞게 다르다.
목소리 설명(instruct)은 voice_design.json 에 화자별로 있다.

1) design   : 화자마다 후보 N개를 만든다 → candidates/index.html 로 듣고 고른다
2) finalize : picks.txt(`kid-f-01 1` 처럼 24줄, 없으면 0번)대로 고른 후보를
              default-voices/<id>.wav 로 복사하고 voices.json 을 만든다

    python -I make_voices.py design   OUT [--only kid-f-01,young-m-01] [--n 3]
    python -I make_voices.py finalize OUT [--picks picks.txt]
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DESIGN_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
SEED_BASE = 2000  # 후보 k 의 시드 = SEED_BASE + 화자 순번*10 + k


def load_design() -> dict:
    return json.loads((HERE / "voice_design.json").read_text(encoding="utf-8"))


def seed_of(i: int, k: int) -> int:
    return SEED_BASE + i * 10 + k


def pick_device(name: str | None):
    import torch

    if name is None:
        name = "cuda:0" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    # MPS·CPU 는 fp32 가 안전하다 (1.7B fp32 ≈ 7GB)
    dtype = torch.bfloat16 if name.startswith("cuda") else torch.float32
    return name, dtype


def load_model(model_id: str, device: str, dtype):
    from qwen_tts import Qwen3TTSModel

    kw = dict(device_map=device, dtype=dtype, attn_implementation="sdpa")
    if device.startswith("cuda"):
        try:
            import flash_attn  # noqa: F401
            kw["attn_implementation"] = "flash_attention_2"
        except ImportError:
            pass
    t = time.time()
    m = Qwen3TTSModel.from_pretrained(model_id, **kw)
    print(f"[load] {model_id} on {device} {dtype} ({time.time() - t:.1f}s)", flush=True)
    return m


def norm(wav: np.ndarray, sr: int, peak_db: float = -1.0) -> np.ndarray:
    """앞뒤 무음을 자르고(앞 0.1초·뒤 0.2초 남김) 최대값을 -1dBFS 로 맞춘다."""
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    thr = 10 ** (-45 / 20) * max(float(np.abs(wav).max()), 1e-6)
    idx = np.where(np.abs(wav) > thr)[0]
    if len(idx):
        wav = wav[max(idx[0] - int(0.1 * sr), 0): idx[-1] + int(0.2 * sr)]
    peak = float(np.abs(wav).max())
    if peak > 0:
        wav = wav * (10 ** (peak_db / 20) / peak)
    return wav


def cmd_design(a) -> None:
    import soundfile as sf
    import torch

    cfg = load_design()
    only = set(a.only.split(",")) if a.only else None
    out = Path(a.out) / "candidates"
    out.mkdir(parents=True, exist_ok=True)
    device, dtype = pick_device(a.device)
    model = load_model(a.model, device, dtype)
    log = out / "timing.jsonl"
    for i, cid in enumerate(cfg["voices"]):
        if only and cid not in only:
            continue
        instruct = cfg["voices"][cid]["instruct"]
        for k in range(a.n):
            path = out / f"{cid}_c{k}.wav"
            if path.exists():
                continue
            seed = seed_of(i, k)
            torch.manual_seed(seed)
            np.random.seed(seed)
            t = time.time()
            wavs, sr = model.generate_voice_design(
                text=cfg["text"], language=cfg["language"], instruct=instruct)
            dt = time.time() - t
            wav = norm(wavs[0], sr)
            sf.write(path, wav, sr, subtype="PCM_16")
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
        rows.append(f"<tr><td><b>{cid}</b><br><small>{html.escape(v['instruct'])}</small></td>"
                    f"{cells}</tr>")
    page = ("<!doctype html><meta charset=utf-8><title>목소리 후보</title>"
            "<style>body{font-family:sans-serif}td{border-bottom:1px solid #ccc;padding:6px;"
            f"vertical-align:top;max-width:420px}}</style><p>{html.escape(cfg['text'])}</p>"
            "<table>" + "".join(rows) + "</table>")
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
    import soundfile as sf

    cfg = load_design()
    picks = read_picks(a.picks)
    root = Path(a.out)
    out = root / "default-voices"
    out.mkdir(parents=True, exist_ok=True)
    voices = []
    sr = None
    for i, (cid, v) in enumerate(cfg["voices"].items()):
        k = picks.get(cid, 0)
        src = root / "candidates" / f"{cid}_c{k}.wav"
        shutil.copyfile(src, out / f"{cid}.wav")
        info = sf.info(src)
        sr = info.samplerate
        voices.append({"id": cid, "file": f"{cid}.wav", "seconds": round(info.duration, 2),
                       "instruct": v["instruct"], "seed": seed_of(i, k)})
    manifest = {"version": 1, "model": DESIGN_MODEL, "language": cfg["language"],
                "text": cfg["text"], "sampleRate": sr, "voices": voices}
    (out / "voices.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[done] {out} ({len(voices)} voices)")


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
    a = p.parse_args()
    {"design": cmd_design, "finalize": cmd_finalize}[a.cmd](a)


if __name__ == "__main__":
    main()
