"""평가에 쓰는 TTS 엔진들 (#33). 모두 say(voice, text, seed) -> (float32 소리, 샘플레이트)로 맞춘다.

엔진마다 venv가 다르다(패키지 버전이 서로 맞지 않는다). eval_accuracy.py gen을 그 venv의 python으로 돌린다.
  qwen        qwen-tts          Qwen3-TTS Base 0.6B·1.7B (src/lolog_ai/tts.py 그대로)
  chatterbox  chatterbox-tts    Chatterbox Multilingual (Resemble AI, MIT)
  xtts        coqui-tts         XTTS-v2 (Coqui Public Model License, 비상업)
읽을 글은 모두 tts.speakable()을 거친다(낱자모 빼기).
"""

from __future__ import annotations

from pathlib import Path

import tts as T

END_MARKS = ".!?…~"


def punct(text: str) -> str:
    """끝에 문장 부호가 없으면 마침표를 붙인다(이야기 대사는 대부분 부호 없이 온다)."""
    return text if text and text[-1] in END_MARKS else text + "."


class Qwen:
    def __init__(self, voices, model=T.MODEL, xvec=False, temp=None, add_punct=False, **_):
        kw = {"temperature": temp, "subtalker_temperature": temp} if temp else {}
        self.tts = T.Tts(voices, model=model, xvec_only=xvec, gen_kwargs=kw)
        self.add_punct = add_punct
        self.tts.load()
        self.name = f"qwen:{model.split('/')[-1]}"

    def say(self, voice, text, seed):
        return self.tts.say(voice, punct(text) if self.add_punct else text, seed=seed)


class Chatterbox:
    def __init__(self, voices, add_punct=False, **_):
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        self.voices, self.add_punct = Path(voices), add_punct
        self.m = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
        self.name = "chatterbox-multilingual"

    def say(self, voice, text, seed):
        import torch
        text = T.speakable(text)
        torch.manual_seed(seed)
        wav = self.m.generate(punct(text) if self.add_punct else text, language_id="ko",
                              audio_prompt_path=str(self.voices / f"{voice}.wav"))
        return wav.squeeze().float().cpu().numpy(), self.m.sr


class Xtts:
    def __init__(self, voices, add_punct=False, **_):
        from TTS.api import TTS
        self.voices, self.add_punct = Path(voices), add_punct
        self.m = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
        self.name = "xtts-v2"

    def say(self, voice, text, seed):
        import numpy as np
        import torch
        text = T.speakable(text)
        torch.manual_seed(seed)
        wav = self.m.tts(text=punct(text) if self.add_punct else text, language="ko",
                         speaker_wav=str(self.voices / f"{voice}.wav"))
        return np.asarray(wav, dtype=np.float32), 24000


ENGINES = {"qwen": Qwen, "chatterbox": Chatterbox, "xtts": Xtts}
