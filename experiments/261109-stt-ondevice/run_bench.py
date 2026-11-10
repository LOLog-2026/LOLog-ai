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


def make_vad(a, threshold: float | None = None) -> sherpa_onnx.VoiceActivityDetector:
    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = str(Path(a.models) / "silero_vad.onnx")
    cfg.silero_vad.threshold = a.vad_threshold if threshold is None else threshold
    cfg.silero_vad.min_silence_duration = a.min_silence
    cfg.silero_vad.min_speech_duration = a.min_speech
    cfg.silero_vad.max_speech_duration = a.max_speech
    cfg.silero_vad.window_size = 512
    cfg.sample_rate = SR
    cfg.num_threads = 1
    buf = a.vad_buffer or (300.0 if a.hard_max_speech else max(60, a.max_speech * 2))
    return sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=buf)


def hard_split(x: np.ndarray, st: int, en: int, limit: int, frame: int = 512) -> list[tuple[int, int]]:
    """limit 샘플보다 긴 구간을 limit 이하로 강제로 자른다.

    자르는 곳은 [st+limit/2, st+limit] 안에서 32ms 프레임 에너지가 가장 낮은 곳(말 사이 쉼일 확률이 큼).
    VAD가 웅성거림·TV 소리 때문에 닫히지 않을 때 구간이 수십~수백 초로 길어지는 것을 막는다.
    """
    out = []
    while en - st > limit:
        lo, hi = st + limit // 2, st + limit
        best, best_e = hi, float("inf")
        for j in range(lo, hi - frame + 1, frame):
            e = float(np.mean(x[j:j + frame] ** 2))
            if e < best_e:
                best, best_e = j + frame // 2, e
        out.append((st, best))
        st = best
    out.append((st, en))
    return out


def apply_gain(x: np.ndarray, mode: str, a=None) -> np.ndarray:
    if mode == "none":
        return x
    if mode == "peak":  # 파일 전체 최대값을 0.9로
        return x * (0.9 / max(1e-4, float(np.max(np.abs(x)))))
    if mode == "rms":  # 말소리 크기를 -23dBFS 근처로 (구간 단위로 쓸 때)
        r = float(np.sqrt(np.mean(x ** 2)) + 1e-8)
        g = min(20.0, (10 ** (-23 / 20)) / r)
        return np.clip(x * g, -1, 1)
    if mode == "agc":  # 폰에서 그대로 쓸 수 있는 실시간 자동 크기 맞춤(앞만 보고 결정)
        if a is None:
            return agc(x)
        return agc(x, max_gain=a.agc_max_gain, noise_ceil_db=a.agc_noise_ceil_db,
                   noise_init=a.agc_noise_init)
    raise ValueError(mode)


def agc(x: np.ndarray, target_db: float = -23.0, max_gain: float = 20.0,
        frame: int = 512, noise_ceil_db: float | None = None,
        noise_init: str = "fixed") -> np.ndarray:
    """32ms 프레임마다 말소리 크기를 따라가며 천천히 키운다. 파일 전체를 미리 보지 않는다.

    - 바닥 소음(noise)은 최솟값 쪽으로 빨리, 위로는 아주 천천히 따라간다.
    - 바닥보다 6dB 넘게 큰 프레임만 말소리 크기(level)를 갱신한다.
    - 이득은 target/level, 1~max_gain 사이, 프레임 사이에서 부드럽게 바꾼다.
    """
    tgt = 10 ** (target_db / 20)
    out = np.empty_like(x)
    noise, level, g = 1e-4, tgt, 1.0
    if noise_init == "first" and len(x):  # 1e-4(-80dBFS)에서 천천히 오르면 짧은 녹음에선 바닥 소음을 못 따라감
        noise = float(np.sqrt(np.mean(x[:frame] ** 2)) + 1e-8)
    for i in range(0, len(x), frame):
        f = x[i:i + frame]
        r = float(np.sqrt(np.mean(f * f)) + 1e-8)
        noise = r if r < noise else noise * 1.002
        if r > noise * 2.0:
            level = 0.9 * level + 0.1 * r
        g_new = min(max_gain, max(1.0, tgt / level))
        if noise_ceil_db is not None:  # 키운 뒤 바닥 소음이 ceil 을 넘지 않게
            g_new = max(1.0, min(g_new, 10 ** (noise_ceil_db / 20) / noise))
        ramp = np.linspace(g, g_new, len(f), dtype=np.float32)
        out[i:i + frame] = np.clip(f * ramp, -1, 1)
        g = g_new
    return out


def track_noise_level(x: np.ndarray, frame: int = 512, target_db: float = -23.0) -> tuple[np.ndarray, np.ndarray]:
    """agc와 같은 방식으로 프레임마다 바닥 소음(noise)과 말소리 크기(level)를 따라간 값(dBFS). 크기 맞춤 전 원래 소리 기준.

    noise: 첫 프레임에서 시작, 더 작으면 바로 내려가고 아니면 x1.002씩 오름.
    level: -23dBFS에서 시작, noise의 2배(6dB)를 넘는 프레임에서만 0.9/0.1로 따라감.
    """
    n = (len(x) + frame - 1) // frame
    nz, lv = np.empty(n, dtype=np.float32), np.empty(n, dtype=np.float32)
    noise = float(np.sqrt(np.mean(x[:frame] ** 2)) + 1e-8) if len(x) else 1e-4
    level = 10 ** (target_db / 20)
    for k, i in enumerate(range(0, len(x), frame)):
        f = x[i:i + frame]
        r = float(np.sqrt(np.mean(f * f)) + 1e-8)
        noise = r if r < noise else noise * 1.002
        if r > noise * 2.0:
            level = 0.9 * level + 0.1 * r
        nz[k] = 20 * np.log10(noise)
        lv[k] = 20 * np.log10(level)
    return nz, lv


def noise_floor_db(x: np.ndarray, frame: int = 512) -> np.ndarray:
    return track_noise_level(x, frame)[0]


def adapt_threshold(noise_db: float, lo_db: float, hi_db: float, th_lo: float = 0.4, th_hi: float = 0.6) -> float:
    if noise_db <= lo_db:
        return th_lo
    if noise_db >= hi_db:
        return th_hi
    return round(th_lo + (th_hi - th_lo) * (noise_db - lo_db) / (hi_db - lo_db), 3)


def adapt_threshold_snr(snr_db: float, lo_db: float, hi_db: float, th_quiet: float = 0.4, th_noisy: float = 0.6) -> float:
    """SNR이 HI 이상이면 0.4(조용한 곳), LO 이하면 0.6(시끄러운 곳), 사이는 선형."""
    if snr_db >= hi_db:
        return th_quiet
    if snr_db <= lo_db:
        return th_noisy
    return round(th_noisy + (th_quiet - th_noisy) * (snr_db - lo_db) / (hi_db - lo_db), 3)


def threshold_schedule(metric_db: np.ndarray, lo_db: float, hi_db: float, frame: int = 512,
                       first_s: float = 10.0, every_s: float = 30.0, fn=None) -> list[tuple[int, float]]:
    """[(시작 샘플, 민감도)]. 처음 10초 중앙값으로 정하고, 이후 30초마다 직전 30초 중앙값으로 다시 정한다(앞만 봄).

    metric_db는 바닥 소음(adapt_threshold) 또는 SNR 추정치(adapt_threshold_snr, fn으로 넘김)."""
    fn = fn or adapt_threshold
    nz_db = metric_db
    fps = SR / frame
    n_first = max(1, int(first_s * fps))
    sched = [(0, fn(float(np.median(nz_db[:n_first])), lo_db, hi_db))]
    step = int(every_s * fps)
    for k in range(step, len(nz_db), step):
        th = fn(float(np.median(nz_db[k - step:k])), lo_db, hi_db)
        if th != sched[-1][1]:
            sched.append((k * frame, th))
    return sched


VAD_TH_USED: list[float] = []  # 결과에 남길 민감도 분포 (파일마다, 바뀐 구간마다 1개)
ADAPT_SWITCHES: list[int] = []  # --adapt-defer: 파일마다 실제로 VAD를 다시 만든 횟수


def _segments_deferred(x: np.ndarray, a, sched: list[tuple[int, float]], win: int) -> list[tuple[int, int]]:
    """민감도가 새로 정해져도 바로 바꾸지 않는다(--adapt-defer).

    - 직전 적용값과 0.05 이상 차이 날 때만 바꾼다.
    - VAD가 닫혀 있을 때(is_speech_detected False) 바꾸고, 열려 있으면 닫힐 때까지 기다린다.
      단 열린 채로 hard_max_speech(없으면 max_speech)초가 지나면 그때 바꾼다.
    """
    spans: list[tuple[int, int]] = []
    limit = int((a.hard_max_speech or a.max_speech) * SR)
    cur_th = sched[0][1]
    cur, st0 = make_vad(a, cur_th), 0
    VAD_TH_USED.append(cur_th)
    switches, pending, k = 0, None, 1
    open_since = None

    def drain(v, base):
        while not v.empty():
            s = v.front
            spans.append((base + s.start, base + s.start + len(s.samples)))
            v.pop()

    for i in range(0, len(x), win):
        while k < len(sched) and sched[k][0] <= i:  # 이 시점까지 새로 정해진 민감도(가장 최근 것)
            pending = sched[k][1]
            k += 1
        cur.accept_waveform(x[i:i + win])
        drain(cur, st0)
        speaking = cur.is_speech_detected()
        if speaking and open_since is None:
            open_since = i
        elif not speaking:
            open_since = None
        if pending is not None:
            if abs(pending - cur_th) < 0.05:
                pending = None
            elif not speaking or (open_since is not None and i + win - open_since >= limit):
                cur.flush()
                drain(cur, st0)
                cur_th, pending = pending, None
                st0 = i + win
                cur = make_vad(a, cur_th)
                VAD_TH_USED.append(cur_th)
                switches += 1
                open_since = None
    cur.flush()
    drain(cur, st0)
    ADAPT_SWITCHES.append(switches)
    return spans


def segments(x: np.ndarray, a, vad, sched: list[tuple[int, float]] | None = None) -> list[np.ndarray]:
    """앱처럼 VAD로 잘라 앞에 pad초를 붙인 구간들. vad가 None이면 통째로.

    sched가 있으면 그 시점마다 민감도를 바꿔 VAD를 새로 만든다(sherpa VAD는 도중에 문턱을 못 바꿈).
    """
    if vad is None:
        return [x]
    spans = []
    win = 512
    if sched is not None and getattr(a, "adapt_defer", False):
        spans = _segments_deferred(x, a, sched, win)
        sched = []  # 아래 일반 경로는 건너뛴다
        cur = None
    elif sched is None:
        sched = [(0, a.vad_threshold)]
        vad.reset()
        cur = vad
    else:
        cur = None
    bounds = [s for s, _ in sched[1:]] + [len(x)]
    for (st0, th), en0 in zip(sched, bounds):
        if cur is None:
            cur = make_vad(a, th)
            VAD_TH_USED.append(th)
        for i in range(st0, en0, win):
            cur.accept_waveform(x[i:min(i + win, en0)])
            while not cur.empty():
                s = cur.front
                spans.append((st0 + s.start, st0 + s.start + len(s.samples)))
                cur.pop()
        cur.flush()
        while not cur.empty():
            s = cur.front
            spans.append((st0 + s.start, st0 + s.start + len(s.samples)))
            cur.pop()
        cur = None
    if a.hard_max_speech:
        limit = int(a.hard_max_speech * SR)
        spans = [p for st, en in spans for p in hard_split(x, st, en, limit)]
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
    ap.add_argument("--hard-max-speech", type=float, default=0.0,
                    help="VAD 구간이 이 초보다 길면 에너지가 낮은 곳에서 강제로 자름 (0=끔)")
    ap.add_argument("--vad-threshold-adapt", default="",
                    help="LO_DB,HI_DB: 바닥 소음(dBFS)이 LO 이하면 민감도 0.4, HI 이상이면 0.6, 사이는 선형. "
                         "처음 10초 중앙값으로 정하고 30초마다 다시 정해 바뀌면 VAD를 새로 만든다")
    ap.add_argument("--vad-threshold-adapt-snr", default="",
                    help="LO_DB,HI_DB: SNR 추정(agc식 말소리 크기 - 바닥 소음)이 HI 이상이면 민감도 0.4, LO 이하면 0.6, "
                         "사이는 선형. 정하는 시점은 --vad-threshold-adapt와 같다")
    ap.add_argument("--adapt-defer", action="store_true",
                    help="adapt 민감도를 바로 바꾸지 않고 VAD가 닫혔을 때(또는 열린 채 hard-max초가 지났을 때), "
                         "직전 값과 0.05 이상 다를 때만 바꾼다")
    ap.add_argument("--vad-buffer", type=float, default=0.0,
                    help="VAD 버퍼 초 (0이면 hard-max 켤 때 300, 아니면 max(60, max_speech*2))")
    ap.add_argument("--gain", default="none", choices=["none", "peak", "rms", "agc"])
    ap.add_argument("--gain-scope", default="segment", choices=["file", "segment"])
    ap.add_argument("--agc-max-gain", type=float, default=20.0, help="agc 최대 이득(배)")
    ap.add_argument("--agc-noise-ceil-db", type=float, default=None,
                    help="agc: 키운 뒤 바닥 소음이 이 dBFS를 넘지 않게 이득 제한 (기본 끔)")
    ap.add_argument("--agc-noise-init", default="fixed", choices=["fixed", "first"],
                    help="agc 바닥 소음 시작값: fixed=1e-4, first=첫 프레임 크기")
    ap.add_argument("--denoise", default="none", choices=["none", "gtcrn"],
                    help="VAD 전에 파일 전체 잡음 줄이기 (순서: denoise -> gain -> VAD)")
    ap.add_argument("--denoise-model", default="", help="기본 models/gtcrn_simple.onnx")
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
    adapt = tuple(float(v) for v in a.vad_threshold_adapt.split(",")) if a.vad_threshold_adapt else None
    adapt_snr = tuple(float(v) for v in a.vad_threshold_adapt_snr.split(",")) if a.vad_threshold_adapt_snr else None
    denoiser = None
    if a.denoise == "gtcrn":
        dcfg = sherpa_onnx.OfflineSpeechDenoiserConfig(
            model=sherpa_onnx.OfflineSpeechDenoiserModelConfig(
                gtcrn=sherpa_onnx.OfflineSpeechDenoiserGtcrnModelConfig(
                    model=a.denoise_model or str(Path(a.models) / "gtcrn_simple.onnx")),
                num_threads=a.threads))
        denoiser = sherpa_onnx.OfflineSpeechDenoiser(dcfg)

    errs = errs_sp = n_ref = n_ref_sp = 0
    audio_s = proc_s = 0.0
    n_seg, max_seg, empty = 0, 0.0, 0
    dump = open(a.dump, "w", encoding="utf-8") if a.dump else None
    for r in rows:
        x, sr = sf.read(r["wav"], dtype="float32")
        assert sr == SR, r["wav"]
        audio_s += len(x) / SR
        t = time.perf_counter()
        if denoiser is not None:  # 처리 시간(RTF)에 포함
            x = np.asarray(denoiser.run(x, SR).samples, dtype=np.float32)
        sched = None
        if adapt and vad is not None:  # 크기 맞춤 전 원래 소리의 바닥 소음으로 민감도를 정한다
            sched = threshold_schedule(noise_floor_db(x), *adapt)
        elif adapt_snr and vad is not None:  # 말소리 크기 - 바닥 소음(SNR 추정)으로 정한다
            nz_db, lv_db = track_noise_level(x)
            sched = threshold_schedule(lv_db - nz_db, *adapt_snr, fn=adapt_threshold_snr)
        if a.gain_scope == "file":
            x = apply_gain(x, a.gain, a)
        segs = segments(x, a, vad, sched)
        hyps = []
        for s in segs:
            if a.gain_scope == "segment":
                s = apply_gain(s, a.gain, a)
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
        **({"vad_th": {"mean": round(float(np.mean(VAD_TH_USED)), 3), "min": min(VAD_TH_USED),
                       "max": max(VAD_TH_USED), "n": len(VAD_TH_USED)}} if (adapt or adapt_snr) and VAD_TH_USED else {}),
        **({"adapt_switches": {"total": int(sum(ADAPT_SWITCHES)), "files": len(ADAPT_SWITCHES),
                               "per_file_mean": round(float(np.mean(ADAPT_SWITCHES)), 2),
                               "max": int(max(ADAPT_SWITCHES))}} if ADAPT_SWITCHES else {}),
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
