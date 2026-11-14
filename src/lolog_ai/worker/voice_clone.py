"""내 목소리 (#43, LOLog-docs spec/characters.md "내 목소리 (FR-CHAR-05)"). Jetson 3호.

사용자가 정해진 문장(RECORD_SENTENCE)을 6~8초 읽어 올리면, 쇼츠 TTS와 같은 모델(Qwen3-TTS 0.6B Base
클론)이 그 녹음을 참조 음성으로, 읽은 문장을 ref_text로 써서 같은 목소리로 대사를 읽는다. 학습·외부 API는 없다.

voice_clone(msg)  lolog:video, kind "voice_clone" { jobId, userId, text }
    녹음 받기(GET .../voice-clone/sample) → 24kHz 모노 → 길이·소리 크기 검사 → 미리 듣기 한 문장 → PUT .../preview

쇼츠에서는 worker/video.py가 load_sample()·check_sample()을 같이 쓴다.
로그에 녹음 내용·문장을 남기지 않는다(TR-12). 회원 ID와 길이·시간만 남긴다.
"""

from __future__ import annotations

import io
import logging
import os
import shutil
import subprocess
import tempfile
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # numpy는 소리를 다룰 때만 불러, numpy가 없는 1·2호에서도 이 모듈을 import할 수 있게 한다
    import numpy as np

log = logging.getLogger(__name__)

SR = 24000  # 기본 화자 클립과 같은 24kHz 모노

# 앱이 보여 주고 읽게 하는 문장 하나(11-14 정함). 40글자라 보통 빠르기(초당 5~7글자)로 6~8초.
# 받침·된소리·여러 모음이 고루 들어 있다. 문장이 하나라 앱이 따로 보내지 않고, 서버가 이 글자를
# voice_clone_text로 저장해 큐 text·쇼츠 voiceText로 준다. 워커는 늘 받은 text를 ref_text로 쓰므로,
# 나중에 문장을 바꿔도 예전 녹음은 그때 저장한 글자로 읽힌다.
RECORD_SENTENCE = "오늘 친구랑 카페에 갔는데, 주문을 잘못해서 둘이 한참 웃었어요. 그래도 덕분에 즐거운 하루였어요."
# 미리 듣기 문장. 녹음 문장과 겹치지 않게 해서 "내 목소리로 다른 말을 한다"는 걸 들을 수 있게 한다
PREVIEW_TEXT = "안녕하세요! 이제 제 목소리로 우리 이야기를 들려드릴게요."

# 검사 기준. 앱이 5초 미만 녹음을 막으니, 워커는 앞뒤 무음을 자른 "말한 길이"로 본다
MIN_VOICED_SEC = 3.0   # 이보다 짧으면 VOICE_TOO_SHORT
MAX_SEC = 20.0         # 이보다 길면 VOICE_TOO_LONG(참조가 길수록 대사마다 느려짐. 권장 5~10초)
QUIET_PEAK_DB = -35.0  # 가장 큰 소리가 이보다 작으면 VOICE_TOO_QUIET
QUIET_RMS_DB = -45.0   # 말한 구간의 평균 크기가 이보다 작아도 VOICE_TOO_QUIET
TRIM_DB = -40.0        # 가장 큰 소리보다 이만큼 작은 곳까지를 무음으로 보고 앞뒤를 자른다


class VoiceSampleInvalid(Exception):
    """녹음이 검사를 통과하지 못함. 다시 해도 같으니 바로 fail(reason = code)."""

    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code} {detail}".strip())
        self.code = code


def _db(x: float) -> float:
    import numpy as np

    return 20 * np.log10(max(x, 1e-9))


def _resample(wav: np.ndarray, sr: int, target: int) -> np.ndarray:
    import numpy as np

    if sr == target:
        return wav
    try:
        import librosa  # qwen-tts가 이미 깔아 둠

        return librosa.resample(wav, orig_sr=sr, target_sr=target).astype(np.float32)
    except ImportError:
        n = round(len(wav) * target / sr)
        return np.interp(np.linspace(0, len(wav) - 1, n), np.arange(len(wav)), wav).astype(np.float32)


def _ffmpeg_decode(data: bytes) -> np.ndarray:
    """m4a 등을 ffmpeg로 푼다. 폰이 녹음한 m4a는 목차(moov)가 파일 끝에 있어 파이프(pipe:0)로 넣으면
    읽지 못하고 빈 소리를 돌려준다(종료 코드는 0). 그래서 임시 파일에 써서 넘긴다."""
    import numpy as np

    exe = os.environ.get("FFMPEG") or shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("m4a 녹음을 풀려면 ffmpeg가 필요해요(3호에 apt install ffmpeg, 또는 환경 변수 FFMPEG)")
    with tempfile.NamedTemporaryFile(prefix="lolog-voice-") as f:
        f.write(data)
        f.flush()
        p = subprocess.run([exe, "-nostdin", "-loglevel", "error", "-i", f.name, "-ac", "1", "-ar", str(SR),
                            "-f", "f32le", "pipe:1"], check=False, capture_output=True, timeout=60)
    out = np.frombuffer(p.stdout, dtype=np.float32).copy()
    if p.returncode != 0 or not len(out):
        raise VoiceSampleInvalid("VOICE_UNREADABLE", p.stderr.decode(errors="replace")[-200:])
    return out


def load_sample(data: bytes) -> np.ndarray:
    """녹음 파일(wav·m4a 바이트) → 24kHz 모노 float32. wav는 soundfile로, 나머지(m4a)는 ffmpeg로 푼다."""
    if not data:
        raise VoiceSampleInvalid("VOICE_UNREADABLE", "빈 파일")
    if data[:4] == b"RIFF":
        import soundfile as sf

        try:
            wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        except Exception as e:  # noqa: BLE001 - 깨진 wav
            raise VoiceSampleInvalid("VOICE_UNREADABLE", type(e).__name__) from None
        return _resample(wav.mean(axis=1), int(sr), SR)
    return _ffmpeg_decode(data)


def check_sample(wav: np.ndarray, sr: int = SR) -> np.ndarray:
    """길이·소리 크기를 검사하고, 앞뒤 무음을 잘라(앞 0.1초·뒤 0.2초 남김) 가장 큰 소리를 -1dBFS로 맞춘다.

    기본 화자 클립(experiments/261111-tts-default-voices make_voices.py의 norm)과 같은 모양으로 맞춰 참조 음성으로 쓴다.
    """
    import numpy as np

    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if len(wav) / sr > MAX_SEC:
        raise VoiceSampleInvalid("VOICE_TOO_LONG", f"{len(wav) / sr:.1f}s")
    peak = float(np.abs(wav).max()) if len(wav) else 0.0
    if _db(peak) < QUIET_PEAK_DB:
        raise VoiceSampleInvalid("VOICE_TOO_QUIET", f"peak {_db(peak):.0f}dB")
    idx = np.where(np.abs(wav) > peak * 10 ** (TRIM_DB / 20))[0]
    voiced = wav[idx[0]: idx[-1] + 1]
    if len(voiced) / sr < MIN_VOICED_SEC:
        raise VoiceSampleInvalid("VOICE_TOO_SHORT", f"{len(voiced) / sr:.1f}s")
    rms = float(np.sqrt(np.mean(voiced ** 2)))
    if _db(rms) < QUIET_RMS_DB:
        raise VoiceSampleInvalid("VOICE_TOO_QUIET", f"rms {_db(rms):.0f}dB")
    out = wav[max(idx[0] - int(0.1 * sr), 0): idx[-1] + int(0.2 * sr)]
    return (out * (10 ** (-1 / 20) / peak)).astype(np.float32)


def prepare(data: bytes) -> np.ndarray:
    """녹음 바이트 → 검사를 통과한 참조 음성(24kHz 모노)."""
    return check_sample(load_sample(data), SR)


def voice_clone(msg: dict, api, tts) -> None:
    """미리 듣기를 만들어 올린다. 녹음이 이상하면 VoiceSampleInvalid(워커가 바로 fail)."""
    import numpy as np
    import soundfile as sf

    user_id = int(msg["userId"])
    text = (msg.get("text") or "").strip()
    if not text:
        raise VoiceSampleInvalid("VOICE_TEXT_MISSING")
    t0 = time.perf_counter()
    ref = prepare(api.get_voice_sample(user_id))
    name = tts.user_voice(user_id, ref, SR, text)
    t1 = time.perf_counter()
    res = tts.say(name, PREVIEW_TEXT, seed=user_id)
    t2 = time.perf_counter()
    wav, sr = res
    buf = io.BytesIO()
    sf.write(buf, np.asarray(wav, dtype=np.float32), sr, format="WAV", subtype="PCM_16")
    api.put_voice_preview(user_id, buf.getvalue())
    log.info("user %s 내 목소리 미리 듣기: 녹음 %.1f초 → %.1f초 | 준비 %.1f초, 읽기 %.1f초",
             user_id, len(ref) / SR, len(wav) / sr, t1 - t0, t2 - t1)
