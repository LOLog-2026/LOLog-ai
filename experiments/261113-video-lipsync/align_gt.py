"""카드 음성(tts/<카드>/l<i>.wav)을 음성 인식으로 받아써 낱말별 시작·끝 시각을 gt/<카드>.json에 둔다(입 싱크 정답).
    .tts-venv/bin/python align_gt.py A B C ...
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, soundfile as sf
from faster_whisper import WhisperModel


def load16k(path):
    a, sr = sf.read(path, dtype="float32")
    n = int(len(a) * 16000 / sr)
    return np.interp(np.arange(n) * sr / 16000, np.arange(len(a)), a).astype(np.float32)

from cards import CARDS
model = WhisperModel(os.environ.get("WHISPER", "small"), device="cpu", compute_type="int8")
os.makedirs("gt", exist_ok=True)
for card in sys.argv[1:]:
    if os.path.exists(f"gt/{card}.json"): continue
    lines = [d for s in CARDS[card]["scenes"] for d in s["lines"]]
    out = []
    for i, d in enumerate(lines):
        segs, _ = model.transcribe(load16k(f"tts/{card}/l{i}.wav"), language="ko", word_timestamps=True, initial_prompt=d["text"],
                                   beam_size=5, condition_on_previous_text=False, vad_filter=False)
        words = [(w.word.strip(), round(w.start, 3), round(w.end, 3)) for s in segs for w in (s.words or [])]
        out.append({"text": d["text"], "words": words})
        print(card, i, d["text"], "|", " ".join(w[0] for w in words), flush=True)
    json.dump(out, open(f"gt/{card}.json", "w"), ensure_ascii=False, indent=1)
