"""대사 TTS (#33). 화자의 인사 클립을 참조 음성으로 넣어 같은 목소리로 대사를 읽는다(Qwen3-TTS Base, 클론).

기본 화자 24명의 인사 클립은 #27(experiments/261111-tts-default-voices)에서 만든 `<id>.wav`이고,
모두 GREETING을 읽었다. 프로필 캐릭터는 voice_for()로 가까운 기본 화자 목소리를 붙인다.

    tts = Tts(voices_dir)                 # 모델은 처음 읽을 때 올린다
    wav, sr = tts.say("young-f-02", "아니 그걸 왜 지금 말해?")

환경 변수
  TTS_MODEL   기본 Qwen/Qwen3-TTS-12Hz-0.6B-Base
"""

from __future__ import annotations

import os
import re
from pathlib import Path

MODEL = os.environ.get("TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
GREETING = "안녕하세요! 만나서 반가워요."
DEFAULT_VOICE = "young-f-01"
# 대사 한 줄(자막 한 장)이 이보다 길게 읽히면 뜻 없는 말이 이어진 것으로 본다(#27에서 20초짜리가 나왔다)
MAX_SECONDS_PER_CHAR = 0.45
MIN_SECONDS = 4.0
TOKENS_PER_SECOND = 12.5  # 12Hz 코덱. 읽기를 여기서 끊지 않으면 "ㅋㅋㅋ" 같은 글에서 끝없이 이어진다(3090에서 9분 넘게 멈춤)
_UNSPEAKABLE = re.compile(r"[ㄱ-ㅎㅏ-ㅣ]+")  # ㅋㅋㅋ, ㅎㅎ, ㅠㅠ: 자막에는 두고 읽을 때만 뺀다


def speakable(text: str) -> str:
    """TTS에 넣을 글. 낱자모를 빼고 남는 글자가 없으면 빈 문자열(읽지 않는다)."""
    out = re.sub(r"\s+", " ", _UNSPEAKABLE.sub(" ", text)).strip()
    return out if re.search(r"[가-힣A-Za-z0-9]", out) else ""

_AGE = [  # appearance(영어 외모 글) → 나이대. 위에서부터 먼저 맞는 것
    ("kid", r"\b(child|kid|boy|girl|toddler|elementary|\d\s*-?\s*year-?\s*old)\b"),
    ("middle", r"\b(middle-?\s*aged|forties|fifties|sixties|elderly|old (man|woman)|grand(ma|pa|mother|father)|"
               r"gray hair|grey hair|mother|father|mom|dad|uncle|aunt)\b"),
]
_FEMALE = r"\b(woman|girl|female|lady|she|her|mother|mom|aunt|grandma|grandmother)\b"
_MALE = r"\b(man|boy|male|guy|he|his|father|dad|uncle|grandpa|grandfather)\b"


def voice_for(appearance: str, seed: int = 0, available: set[str] | None = None) -> str:
    """영어 외모 글에서 나이대·성별을 읽어 기본 화자 목소리 id를 고른다(프로필 캐릭터용).

    같은 (외모, seed)면 늘 같은 목소리다. seed는 userId처럼 사람마다 다른 값을 넣어 4개 중 하나로 흩는다.
    성별을 알 수 없으면 seed로 정한다.
    """
    text = appearance.lower()
    age = next((a for a, pat in _AGE if re.search(pat, text)), "young")
    f, m = re.search(_FEMALE, text), re.search(_MALE, text)
    if f and m:  # "a woman ... her" 와 "man"이 같이 나오면 먼저 나온 쪽
        gender = "f" if f.start() < m.start() else "m"
    elif f or m:
        gender = "f" if f else "m"
    else:
        gender = "fm"[seed % 2]
    for k in range(4):
        vid = f"{age}-{gender}-{(seed + k) % 4 + 1:02d}"
        if available is None or vid in available:
            return vid
    return DEFAULT_VOICE


def _local(model_id: str) -> str:
    """받아 둔 모델이면 그 폴더 경로. 모델 이름으로 읽으면 transformers 4.57이 토크나이저를 읽다가 허브에
    모델 정보를 물어봐서 HF_HUB_OFFLINE=1(인터넷 없는 젯슨)에서 죽는다. 폴더 경로로 주면 묻지 않는다."""
    if Path(model_id).exists():
        return model_id
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(model_id, local_files_only=True)
    except Exception:  # 아직 안 받았으면 이름 그대로(받으면서 읽는다)
        return model_id


class Tts:
    def __init__(self, voices_dir: str | Path, model: str = MODEL, device: str | None = None,
                 xvec_only: bool = False, gen_kwargs: dict | None = None):
        """xvec_only: 참조 음성의 화자 벡터만 쓴다(참조 음성의 내용·말투는 넘기지 않음).
        gen_kwargs: 생성 옵션(temperature, top_k 등). 없으면 모델 기본값(temperature 0.9)."""
        self.gen_kwargs = gen_kwargs or {}
        self.xvec_only = xvec_only
        self.last_tries = 0
        self.voices_dir = Path(voices_dir)
        self.model_id = model
        self.device = device
        self._model = None
        self._prompts: dict[str, object] = {}

    def voices(self) -> set[str]:
        return {p.stem for p in self.voices_dir.glob("*.wav")}

    def load(self):
        if self._model is None:
            import torch
            from qwen_tts import Qwen3TTSModel

            dev = self.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
            dtype = torch.bfloat16 if dev.startswith("cuda") else torch.float32
            self._model = Qwen3TTSModel.from_pretrained(_local(self.model_id), device_map=dev, dtype=dtype,
                                                        attn_implementation="sdpa")
        return self._model

    def unload(self) -> None:
        import gc

        import torch
        self._model = None
        self._prompts.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _prompt(self, voice: str):
        if voice not in self._prompts:
            wav = self.voices_dir / f"{voice}.wav"
            if not wav.exists():
                raise FileNotFoundError(f"목소리 클립이 없음: {wav}")
            self._prompts[voice] = self.load().create_voice_clone_prompt(
                ref_audio=str(wav), ref_text=GREETING, x_vector_only_mode=self.xvec_only)
        return self._prompts[voice]

    def say(self, voice: str, text: str, seed: int = 0, tries: int = 3):
        """(소리 float32 1차원, 샘플레이트). 읽을 글자가 없으면 None.

        길이 한도의 1.5배에서 읽기를 끊고, 한도를 넘게 나오면 시드를 바꿔 다시 읽어 가장 짧은 것을 쓴다.
        """
        import torch

        text = speakable(text)
        if not text:
            return None
        model, prompt = self.load(), self._prompt(voice)
        limit = max(MIN_SECONDS, len(text) * MAX_SECONDS_PER_CHAR)
        best = None
        for k in range(tries):
            torch.manual_seed(seed + k)
            wavs, sr = model.generate_voice_clone(text=text, language="Korean", voice_clone_prompt=prompt,
                                                  max_new_tokens=int(limit * 1.5 * TOKENS_PER_SECOND),
                                                  **self.gen_kwargs)
            wav = wavs[0]
            self.last_tries = k + 1
            if best is None or len(wav) < len(best[0]):
                best = (wav, sr)
            if len(wav) / sr <= limit:
                break
        return best


def main() -> None:
    """설치 확인용: python tts.py VOICES_DIR [--voice young-f-01] [--text ...] [--out hello.wav]"""
    import argparse
    import time

    import soundfile as sf

    ap = argparse.ArgumentParser(description="인사 클립 목소리로 한 줄 읽어 본다")
    ap.add_argument("voices")
    ap.add_argument("--voice", default=DEFAULT_VOICE)
    ap.add_argument("--text", default="설치가 끝났어요. 이제 대사를 읽을 수 있어요.")
    ap.add_argument("--out", default="hello.wav")
    a = ap.parse_args()

    tts = Tts(a.voices)
    if a.voice not in tts.voices():
        raise SystemExit(f"{a.voices}에 {a.voice}.wav가 없어요 (있는 목소리 {len(tts.voices())}개)")
    t0 = time.perf_counter()
    tts.load()
    t1 = time.perf_counter()
    wav, sr = tts.say(a.voice, a.text)
    sec = time.perf_counter() - t1
    sf.write(a.out, wav, sr)
    print(f"[ok] 모델 올림 {t1 - t0:.1f}초, 읽기 {sec:.1f}초, 소리 {len(wav) / sr:.1f}초 → {a.out}")


if __name__ == "__main__":
    main()
