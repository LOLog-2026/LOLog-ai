"""카드 4개의 대사를 Qwen3-TTS 0.6B(워커와 같은 모델·목소리)로 읽어 tts/<카드>/l<번호>.wav에 둔다(맥 CPU)."""
import json, sys, time
from pathlib import Path
import soundfile as sf
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from lolog_ai import tts as T
from cards import CARDS
t = T.Tts("voices", device="cpu")
for name, sb in CARDS.items():
    out = Path("tts") / name; out.mkdir(parents=True, exist_ok=True)
    voice = {c["key"]: c["voice"] for c in sb["characters"]}
    lines = [d for s in sb["scenes"] for d in s["lines"]]
    for i, d in enumerate(lines):
        p = out / f"l{i}.wav"
        if p.exists(): continue
        t0 = time.perf_counter(); wav, sr = t.say(voice[d["speaker"]], d["text"], seed=1000 + i)
        sf.write(p, wav, sr, subtype="PCM_16")
        print(name, i, f"{len(wav)/sr:.2f}s took {time.perf_counter()-t0:.0f}s", d["text"], flush=True)
print("DONE")
