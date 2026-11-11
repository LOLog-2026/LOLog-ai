"""TTS 클론 속도·메모리 측정 (#33). 3090과 Jetson에서 같은 파일로 잰다.

기본 화자 인사 클립(<id>.wav)을 참조 음성으로 넣어 대사 8개를 읽고, 대사마다 걸린 시간과 소리 길이,
끝에 모델 올리는 시간·최대 메모리를 적는다. 결과 wav와 result.json은 OUT 폴더에 남는다.

    python -I bench_clone.py VOICES_DIR OUT [--model Qwen/Qwen3-TTS-12Hz-0.6B-Base] [--device cuda:0]

tts.py는 이 파일 옆에 있으면 그것을, 없으면 레포의 src/lolog_ai/tts.py를 쓴다.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE if (HERE / "tts.py").exists() else HERE.parents[1] / "src" / "lolog_ai"))
import tts as T  # noqa: E402

LINES = [
    ("young-f-02", "아니 그걸 왜 지금 말해?"),
    ("young-m-01", "나 어제 편의점에서 삼각김밥 데우다가 터졌어."),
    ("middle-f-01", "그래서 내가 뭐라 그랬어, 우산 챙기라 했지?"),
    ("kid-m-02", "엄마 나 숙제 다 했어! 진짜야!"),
    ("young-f-02", "ㅋㅋㅋ 진짜 미치겠다"),
    ("middle-m-03", "어이구, 그게 벌써 십 년 전 일이네."),
    ("young-m-01", "헐."),
    ("young-f-02", "근데 그 사람이 갑자기 뒤돌아보더니 나한테 인사를 하는 거야."),
]


def mem_available_mb() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) // 1024
    return -1


class MemWatch(threading.Thread):
    """남은 시스템 메모리의 최저값. Jetson은 GPU와 RAM이 한 메모리라 이 값이 실제 여유다."""

    def __init__(self):
        super().__init__(daemon=True)
        self.start_mb = self.low_mb = mem_available_mb()
        self.stop = False

    def run(self):
        while not self.stop:
            self.low_mb = min(self.low_mb, mem_available_mb())
            time.sleep(0.5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("voices")
    ap.add_argument("out")
    ap.add_argument("--model", default=T.MODEL)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    import soundfile as sf
    import torch

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    watch = MemWatch()
    watch.start()
    tts = T.Tts(a.voices, model=a.model, device=a.device)
    t0 = time.perf_counter()
    tts.load()
    load_s = time.perf_counter() - t0
    print(f"[load] {a.model} {load_s:.1f}s", flush=True)

    rows = []
    for i, (voice, text) in enumerate(LINES):
        first = voice not in tts._prompts
        t0 = time.perf_counter()
        wav, sr = tts.say(voice, text, seed=i)
        sec = time.perf_counter() - t0
        audio = len(wav) / sr
        sf.write(out / f"{i}_{voice}.wav", wav, sr)
        row = {"i": i, "voice": voice, "text": text, "chars": len(text), "first_use_of_voice": first,
               "sec": round(sec, 2), "audio": round(audio, 2), "rtf": round(sec / audio, 2)}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    watch.stop = True
    total_sec, total_audio = sum(r["sec"] for r in rows), sum(r["audio"] for r in rows)
    cuda = torch.cuda.is_available()
    summary = {
        "model": a.model, "machine": platform.machine(), "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "load_s": round(load_s, 1), "lines": len(rows),
        "total_sec": round(total_sec, 1), "total_audio": round(total_audio, 1),
        "rtf": round(total_sec / total_audio, 2),
        "torch_peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if cuda else None,
        "process_peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2),
        "mem_available_start_mb": watch.start_mb, "mem_available_low_mb": watch.low_mb,
    }
    print("[summary] " + json.dumps(summary, ensure_ascii=False), flush=True)
    (out / "result.json").write_text(json.dumps({"summary": summary, "lines": rows}, ensure_ascii=False, indent=2),
                                     encoding="utf-8")


if __name__ == "__main__":
    main()
