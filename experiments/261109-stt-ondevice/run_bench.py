"""폰 안 STT(sherpa-onnx) 설정을 하나씩 바꿔 가며 CER·속도를 재는 벤치.

앱과 같은 흐름을 흉내 낸다: (선택) 입력 소리 크기 맞춤 -> silero VAD로 말소리 구간 자르기
-> 구간 앞에 pad초 붙이기(앞 구간 끝은 넘지 않음) -> 구간마다 OfflineRecognizer.

폰처럼 CPU를 묶으려고 --threads 개수만큼만 코어를 쓴다(리눅스는 코어 고정까지, 맥은 스레드 수만).

결과는 results.jsonl 에 한 줄씩 쌓인다. 예:
  python -I run_bench.py --models models --data data/stt --set fleurs-dev \
      --model sensevoice-int8 --lang ko --itn 1 --vad silero --threads 2 --tag baseline
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import sherpa_onnx
import soundfile as sf

SR = 16000

# 모델 이름 -> 폴더와 만드는 법. fetch_models.sh 로 받은 폴더 이름과 맞춘다.
MODELS = {
    "sensevoice-int8": ("sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17", "sense_voice", "model.int8.onnx"),
    "sensevoice-fp32": ("sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17", "sense_voice", "model.onnx"),
    "sensevoice-2025-int8": ("sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09", "sense_voice", "model.int8.onnx"),
    "zipformer-ko": ("sherpa-onnx-zipformer-korean-2024-06-24", "transducer", "int8"),
    "zipformer-ko-fp32": ("sherpa-onnx-zipformer-korean-2024-06-24", "transducer", ""),
    "whisper-tiny": ("sherpa-onnx-whisper-tiny", "whisper", "tiny"),
    "whisper-base": ("sherpa-onnx-whisper-base", "whisper", "base"),
    "whisper-small": ("sherpa-onnx-whisper-small", "whisper", "small"),
    "whisper-turbo": ("sherpa-onnx-whisper-turbo", "whisper", "turbo"),
}


def pick(d: Path, pattern: str) -> str:
    c = sorted(d.glob(pattern))
    if not c:
        raise FileNotFoundError(f"{d}/{pattern}")
    return str(c[0])


def make_recognizer(a) -> sherpa_onnx.OfflineRecognizer:
    folder, kind, arg = MODELS[a.model]
    d = Path(a.models) / folder
    if kind == "sense_voice":
        return sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(d / arg), tokens=str(d / "tokens.txt"), num_threads=a.threads,
            language=a.lang, use_itn=bool(a.itn), rule_fsts=a.rule_fsts)
    if kind == "transducer":
        suf = f"*.{arg}.onnx" if arg else "*[!8].onnx"
        return sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=pick(d, "encoder" + suf), decoder=pick(d, "decoder" + suf),
            joiner=pick(d, "joiner" + suf), tokens=str(d / "tokens.txt"),
            num_threads=a.threads, decoding_method=a.decoding,
            max_active_paths=a.beam, rule_fsts=a.rule_fsts)
    if kind == "whisper":
        q = ".int8" if a.whisper_int8 else ""
        return sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=pick(d, f"{arg}-encoder{q}.onnx"), decoder=pick(d, f"{arg}-decoder{q}.onnx"),
            tokens=pick(d, f"{arg}-tokens.txt"), num_threads=a.threads, language="ko",
            task="transcribe", tail_paddings=a.whisper_tail)
    raise ValueError(kind)


def make_vad(a) -> sherpa_onnx.VoiceActivityDetector:
    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = str(Path(a.models) / "silero_vad.onnx")
    cfg.silero_vad.threshold = a.vad_threshold
    cfg.silero_vad.min_silence_duration = a.min_silence
    cfg.silero_vad.min_speech_duration = a.min_speech
    cfg.silero_vad.max_speech_duration = a.max_speech
    cfg.silero_vad.window_size = 512
    cfg.sample_rate = SR
    cfg.num_threads = 1
    return sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=max(60, a.max_speech * 2))


def apply_gain(x: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return x
    if mode == "peak":  # 파일 전체 최대값을 0.9로
        return x * (0.9 / max(1e-4, float(np.max(np.abs(x)))))
    if mode == "rms":  # 말소리 크기를 -23dBFS 근처로 (구간 단위로 쓸 때)
        r = float(np.sqrt(np.mean(x ** 2)) + 1e-8)
        g = min(20.0, (10 ** (-23 / 20)) / r)
        return np.clip(x * g, -1, 1)
    if mode == "agc":  # 폰에서 그대로 쓸 수 있는 실시간 자동 크기 맞춤(앞만 보고 결정)
        return agc(x)
    raise ValueError(mode)


def agc(x: np.ndarray, target_db: float = -23.0, max_gain: float = 20.0,
        frame: int = 512) -> np.ndarray:
    """32ms 프레임마다 말소리 크기를 따라가며 천천히 키운다. 파일 전체를 미리 보지 않는다.

    - 바닥 소음(noise)은 최솟값 쪽으로 빨리, 위로는 아주 천천히 따라간다.
    - 바닥보다 6dB 넘게 큰 프레임만 말소리 크기(level)를 갱신한다.
    - 이득은 target/level, 1~max_gain 사이, 프레임 사이에서 부드럽게 바꾼다.
    """
    tgt = 10 ** (target_db / 20)
    out = np.empty_like(x)
    noise, level, g = 1e-4, tgt, 1.0
    for i in range(0, len(x), frame):
        f = x[i:i + frame]
        r = float(np.sqrt(np.mean(f * f)) + 1e-8)
        noise = r if r < noise else noise * 1.002
        if r > noise * 2.0:
            level = 0.9 * level + 0.1 * r
        g_new = min(max_gain, max(1.0, tgt / level))
        ramp = np.linspace(g, g_new, len(f), dtype=np.float32)
        out[i:i + frame] = np.clip(f * ramp, -1, 1)
        g = g_new
    return out


def segments(x: np.ndarray, a, vad) -> list[np.ndarray]:
    """앱처럼 VAD로 잘라 앞에 pad초를 붙인 구간들. vad가 None이면 통째로."""
    if vad is None:
        return [x]
    vad.reset()
    spans = []
    win = 512
    for i in range(0, len(x), win):
        vad.accept_waveform(x[i:i + win])
        while not vad.empty():
            s = vad.front
            spans.append((s.start, s.start + len(s.samples)))
            vad.pop()
    vad.flush()
    while not vad.empty():
        s = vad.front
        spans.append((s.start, s.start + len(s.samples)))
        vad.pop()
    out, prev_end = [], 0
    pad = int(a.pad * SR)
    tail = int(a.tail_pad * SR)
    for st, en in spans:
        st2 = max(prev_end, st - pad)
        en2 = min(len(x), en + tail)
        out.append(x[st2:en2])
        prev_end = en2
    return out


# ---- 채점 ----
_keep = re.compile(r"[^0-9a-z가-힣ㄱ-ㆎ ]")


def norm(t: str, ref: bool = False) -> str:
    t = unicodedata.normalize("NFKC", t).lower()
    t = re.sub(r"<\|[^|]*\|>", " ", t)  # 혹시 남은 SenseVoice 태그
    if ref:  # FLEURS 기준 글의 괄호 속 원어 표기(예: 비슈케크(Bishkek))는 읽지 않으므로 뺀다.
        # 단, 괄호 안에 한글이 있으면 실제로 읽은 내용이라 남긴다
        t = re.sub(r"\(([^)]*)\)", lambda m: f" {m.group(1)} " if re.search(r"[\uac00-\ud7a3]", m.group(1)) else " ", t)
    t = re.sub(r"(?<=\d),(?=\d)", "", t)  # 1,000 -> 1000
    t = _keep.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def edit(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="models")
    ap.add_argument("--data", default="data/stt")
    ap.add_argument("--set", required=True, help="fleurs-dev, fleurs-clean, fleurs-babble5, ...")
    ap.add_argument("--model", default="sensevoice-int8", choices=sorted(MODELS))
    ap.add_argument("--lang", default="ko", help="SenseVoice: ko / auto")
    ap.add_argument("--itn", type=int, default=1)
    ap.add_argument("--rule-fsts", default="", help="후처리용 FST(쉼표로 여러 개)")
    ap.add_argument("--decoding", default="greedy_search")
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--whisper-int8", type=int, default=1)
    ap.add_argument("--whisper-tail", type=int, default=-1)
    ap.add_argument("--vad", default="silero", choices=["silero", "none"])
    ap.add_argument("--vad-threshold", type=float, default=0.5)
    ap.add_argument("--min-silence", type=float, default=0.5)
    ap.add_argument("--min-speech", type=float, default=0.25)
    ap.add_argument("--max-speech", type=float, default=20.0)
    ap.add_argument("--pad", type=float, default=0.3, help="구간 앞에 붙이는 초 (앱 링버퍼)")
    ap.add_argument("--tail-pad", type=float, default=0.0, help="구간 뒤에 붙이는 초")
    ap.add_argument("--gain", default="none", choices=["none", "peak", "rms", "agc"])
    ap.add_argument("--gain-scope", default="segment", choices=["file", "segment"])
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="results.jsonl")
    ap.add_argument("--dump", default="", help="발화별 결과를 쓸 jsonl")
    a = ap.parse_args()

    if hasattr(os, "sched_setaffinity"):  # 리눅스: 폰처럼 코어 수 자체를 묶는다
        os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[: a.threads]))
    os.environ["OMP_NUM_THREADS"] = str(a.threads)

    rows = [json.loads(l) for l in open(Path(a.data) / f"{a.set}.jsonl", encoding="utf-8")]
    if a.limit:
        rows = rows[: a.limit]

    t0 = time.perf_counter()
    rec = make_recognizer(a)
    load_s = time.perf_counter() - t0
    vad = make_vad(a) if a.vad == "silero" else None

    errs = errs_sp = n_ref = n_ref_sp = 0
    audio_s = proc_s = 0.0
    n_seg, max_seg, empty = 0, 0.0, 0
    dump = open(a.dump, "w", encoding="utf-8") if a.dump else None
    for r in rows:
        x, sr = sf.read(r["wav"], dtype="float32")
        assert sr == SR, r["wav"]
        audio_s += len(x) / SR
        t = time.perf_counter()
        if a.gain_scope == "file":
            x = apply_gain(x, a.gain)
        segs = segments(x, a, vad)
        hyps = []
        for s in segs:
            if a.gain_scope == "segment":
                s = apply_gain(s, a.gain)
            st = rec.create_stream()
            st.accept_waveform(SR, s)
            rec.decode_stream(st)
            hyps.append(st.result.text)
        proc_s += time.perf_counter() - t
        n_seg += len(segs)
        max_seg = max([max_seg] + [len(s) / SR for s in segs])
        hyp, ref = norm(" ".join(hyps)), norm(r["text"], ref=True)
        empty += not hyp
        e_sp = edit(ref.replace(" ", ""), hyp.replace(" ", ""))
        errs_sp += e_sp
        n_ref_sp += len(ref.replace(" ", ""))
        errs += edit(ref, hyp)
        n_ref += len(ref)
        if dump:
            dump.write(json.dumps({"id": r["id"], "ref": ref, "hyp": hyp,
                                   "cer": e_sp / max(1, len(ref.replace(' ', '')))},
                                  ensure_ascii=False) + "\n")

    res = {
        "tag": a.tag, "set": a.set, "n": len(rows), "model": a.model,
        "cer": round(100 * errs_sp / max(1, n_ref_sp), 2),  # 띄어쓰기 뺀 CER(주 지표)
        "cer_with_space": round(100 * errs / max(1, n_ref), 2),
        "rtf": round(proc_s / max(1e-9, audio_s), 4),
        "audio_min": round(audio_s / 60, 1), "load_s": round(load_s, 2),
        "segments": n_seg, "max_seg_s": round(max_seg, 1), "empty_hyp": empty,
        "cfg": {k: v for k, v in vars(a).items()
                if k not in ("models", "data", "out", "dump", "tag", "set", "limit")},
        "host": f"{platform.system()}-{platform.machine()}",
        "sherpa": sherpa_onnx.__version__,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    print(json.dumps(res, ensure_ascii=False))
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(res, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
