"""STT 벤치용 데이터 만들기 (오픈소스 데이터만 씀).

FLEURS ko_kr(Google, CC-BY-4.0)을 받아 아래 세트를 만든다.

  clean      FLEURS test 그대로 (짧은 발화, 낭독체)
  babble10   test 발화 + 다른 화자 3명 섞은 웅성거림, SNR 10dB (dev 발화로 만듦)
  babble5    같은 방식, SNR 5dB (시끄러운 카페·TV 옆 상황)
  quiet      test 발화를 -20dB로 줄임 (폰이 멀리 있거나 주머니 속)
  longform   test 발화를 0.3~2.5초 쉼(약한 배경 소음)으로 이어 붙인 5분짜리 긴 녹음.
             앱처럼 VAD로 잘라서 인식해야 하는 세트. 기준 글은 이어 붙인 문장 전부.

Zeroth-Korean(OpenSLR 40, CC-BY-4.0) 테스트셋은 --zeroth 를 주면 HF에서 받아 같이 만든다.

쓰는 법:
  python -I prepare_data.py --out data/stt
  python -I prepare_data.py --out data/stt --zeroth   # Zeroth도 (HF 접속 필요)
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

FLEURS_URL = "https://storage.googleapis.com/xtreme_translations/FLEURS102/ko_kr.tar.gz"
# tar 안 순서가 dev -> test -> train 이라 앞 350MB만 받으면 dev·test가 다 들어 있다(전체 1.8GB).
FLEURS_HEAD_BYTES = 350_000_000
SR = 16000


def download(url: str, dst: Path, max_bytes: int = 0) -> Path:
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"받는 중: {url}")
    req = urllib.request.Request(url)
    if max_bytes:
        req.add_header("Range", f"bytes=0-{max_bytes - 1}")
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(dst)
    return dst


def load_fleurs(tar_path: Path, raw_dir: Path) -> dict[str, list[dict]]:
    """split -> [{id, wav, text}] . 오디오는 raw_dir/fleurs/<split>/ 에 푼다."""
    out_root = raw_dir / "fleurs"
    marker = out_root / ".done"
    if not marker.exists():
        with tarfile.open(tar_path, mode="r|gz") as tf:
          try:
            for m in tf:
                if "/train" in m.name or m.name.endswith("train.tsv"):
                    break  # train은 쓰지 않는다(웅성거림은 dev 발화로 만든다)
                if not m.isfile():
                    continue
                parts = Path(m.name).parts  # ko_kr/audio/test/x.wav , ko_kr/test.tsv
                if ".." in parts or m.name.startswith("/"):
                    continue
                if parts[-1].endswith(".tsv"):
                    dst = out_root / parts[-1]
                elif len(parts) == 4 and parts[1] == "audio" and parts[3].endswith(".wav"):
                    dst = out_root / parts[2] / parts[3]
                else:
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                with tf.extractfile(m) as src, open(dst, "wb") as f:
                    f.write(src.read())
          except (EOFError, tarfile.ReadError):
              pass  # 앞부분만 받은 tar라 끝이 잘려 있음
        marker.touch()

    splits: dict[str, list[dict]] = {}
    for split in ("dev", "test"):
        rows = []
        tsv = out_root / f"{split}.tsv"
        if not tsv.exists():  # 앞부분만 받은 tar에는 test.tsv가 없다 -> 저장소에 둔 사본
            tsv = Path(__file__).resolve().parent / "fleurs_meta" / f"{split}.tsv"
        with open(tsv, encoding="utf-8") as f:
            for r in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
                # id, file_name, raw_transcription, transcription, phonemes, num_samples, gender
                wav = out_root / split / r[1]
                if wav.exists():
                    rows.append({"id": f"fleurs-{split}-{Path(r[1]).stem}", "wav": str(wav),
                                 "text": r[2]})
        splits[split] = rows
    return splits


def read16k(path: str) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:  # FLEURS는 16k라 여기 안 옴. 혹시 몰라 단순 선형 보간.
        n = int(len(x) * SR / sr)
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
    return x


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12)


def make_babble(n: int, pool: list[str], rng: random.Random, k: int = 3) -> np.ndarray:
    acc = np.zeros(n, dtype=np.float32)
    for p in rng.sample(pool, k):
        y = read16k(p)
        if len(y) < n:
            y = np.tile(y, n // len(y) + 1)
        off = rng.randint(0, len(y) - n)
        y = y[off:off + n]
        acc += y / rms(y)
    return acc / k


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{path}: {len(rows)}개")


def build_variants(name: str, test: list[dict], babble_pool: list[str], out: Path,
                   seed: int) -> None:
    rng = random.Random(seed)
    write_manifest(out / f"{name}-clean.jsonl", test)
    for snr in (10, 5):
        rows = []
        d = out / "audio" / f"{name}-babble{snr}"
        d.mkdir(parents=True, exist_ok=True)
        for r in test:
            x = read16k(r["wav"])
            b = make_babble(len(x), babble_pool, rng)
            y = x + b * (rms(x) / rms(b)) / (10 ** (snr / 20))
            y = y / max(1.0, float(np.max(np.abs(y))))
            p = d / (r["id"] + ".wav")
            sf.write(p, y, SR, subtype="PCM_16")
            rows.append({**r, "wav": str(p)})
        write_manifest(out / f"{name}-babble{snr}.jsonl", rows)
    rows = []
    d = out / "audio" / f"{name}-quiet"
    d.mkdir(parents=True, exist_ok=True)
    for r in test:
        p = d / (r["id"] + ".wav")
        sf.write(p, read16k(r["wav"]) * 0.1, SR, subtype="PCM_16")
        rows.append({**r, "wav": str(p)})
    write_manifest(out / f"{name}-quiet.jsonl", rows)


def build_longform(name: str, test: list[dict], out: Path, seed: int,
                   minutes: float = 5.0) -> None:
    rng = random.Random(seed)
    items = test[:]
    rng.shuffle(items)
    d = out / "audio" / f"{name}-longform"
    d.mkdir(parents=True, exist_ok=True)
    rows, buf, texts, idx = [], [], [], 0
    nrng = np.random.default_rng(seed)

    def flush():
        nonlocal buf, texts, idx
        if not buf:
            return
        y = np.concatenate(buf)
        p = d / f"{name}-long-{idx:02d}.wav"
        sf.write(p, y, SR, subtype="PCM_16")
        rows.append({"id": p.stem, "wav": str(p), "text": " ".join(texts)})
        buf, texts, idx = [], [], idx + 1

    total = 0
    for r in items:
        x = read16k(r["wav"])
        gap = int(rng.uniform(0.3, 2.5) * SR)
        floor = rms(x) * 10 ** (-35 / 20)  # 말소리보다 35dB 낮은 바닥 소음
        buf.append((nrng.standard_normal(gap) * floor).astype(np.float32))
        buf.append(x)
        texts.append(r["text"])
        total += gap + len(x)
        if total >= minutes * 60 * SR:
            flush()
            total = 0
    flush()
    write_manifest(out / f"{name}-longform.jsonl", rows)


def load_zeroth(raw_dir: Path) -> list[dict]:
    """HF kresnik/zeroth_korean 테스트셋(457발화)을 wav로 푼다."""
    from huggingface_hub import hf_hub_download, list_repo_files  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415

    repo = "kresnik/zeroth_korean"
    files = [f for f in list_repo_files(repo, repo_type="dataset")
             if f.endswith(".parquet") and "test" in f]
    d = raw_dir / "zeroth" / "test"
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in files:
        t = pq.read_table(hf_hub_download(repo, f, repo_type="dataset")).to_pylist()
        for i, r in enumerate(t):
            audio = r["audio"]
            x, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
            p = d / f"{Path(f).stem}-{i:04d}.wav"
            if sr != SR:
                n = int(len(x) * SR / sr)
                x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
            sf.write(p, x, SR, subtype="PCM_16")
            rows.append({"id": f"zeroth-{p.stem}", "wav": str(p), "text": r["text"]})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/stt")
    ap.add_argument("--zeroth", action="store_true")
    ap.add_argument("--seed", type=int, default=20261109)
    a = ap.parse_args()
    out = Path(a.out)
    raw = out / "raw"
    tar = download(FLEURS_URL, raw / "ko_kr-head.tar.gz", FLEURS_HEAD_BYTES)
    sp = load_fleurs(tar, raw)
    pool = [r["wav"] for r in sp["dev"]]  # test와 겹치지 않는 화자·문장
    build_variants("fleurs", sp["test"], pool, out, a.seed)
    build_longform("fleurs", sp["test"], out, a.seed)
    # 프롬프트·설정 튜닝은 dev로 하고 test는 확인용으로만 쓴다(과적합 방지)
    write_manifest(out / "fleurs-dev.jsonl", sp["dev"])
    if a.zeroth:
        z = load_zeroth(raw)
        build_variants("zeroth", z, pool, out, a.seed)


if __name__ == "__main__":
    main()
