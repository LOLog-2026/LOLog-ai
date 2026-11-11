"""카드 영상 평가 (#33, VIDEO_EVAL_PLAN.md). 자동 항목을 재고, 눈으로 센 체크 파일이 있으면 합쳐서 표를 낸다.

    python -I eval_video.py BATCH_DIR CARDS_DIR [--manual manual.json] [--out result.json]

BATCH_DIR: run_batch.sh 결과(카드마다 폴더 + batch.jsonl). CARDS_DIR: batch_card_*.json이 있는 폴더.
A3 받아쓰기는 qwen-tts venv(transformers, librosa)로 돌린다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_accuracy import CER_PASS, _asr, cer  # noqa: E402

_SINO = "영일이삼사오육칠팔구"
_NATIVE = {1: "한", 2: "두", 3: "세", 4: "네", 5: "다섯", 6: "여섯", 7: "일곱", 8: "여덟", 9: "아홉", 10: "열",
           20: "스물", 30: "서른", 40: "마흔", 50: "쉰"}


def _sino(n: int) -> str:
    out = ""
    for unit, name in ((1000, "천"), (100, "백"), (10, "십")):
        q, n = divmod(n, unit)
        if q:
            out += ("" if q == 1 else _SINO[q]) + name
    return out + (_SINO[n] if n else "") or "영"


def _native(n: int) -> str | None:
    if n in _NATIVE:
        return _NATIVE[n].replace("스물", "스무") if n == 20 else _NATIVE[n]
    t, o = divmod(n, 10)
    return (_NATIVE[t * 10] + _NATIVE[o]) if t * 10 in _NATIVE and o else None


def number_variants(text: str) -> list[str]:
    """Whisper는 "이십 분"을 "20분", "세 개"를 "3개"로 받아쓴다. 받아쓴 글의 숫자를 한자어·고유어 읽기로 바꾼
    후보를 만든다(자막은 숫자를 한글로 쓰게 했다)."""
    import re
    nums = re.findall(r"\d+", text)
    if not nums:
        return [text]
    sino = re.sub(r"\d+", lambda m: _sino(int(m[0])), text)
    native = re.sub(r"\d+", lambda m: _native(int(m[0])) or _sino(int(m[0])), text)
    return [text, sino, native]


def read_cer(ref: str, heard: str) -> float:
    return min(cer(ref, h) for h in number_variants(heard))


MAX_TEXT = 40
DUR_MIN, DUR_MAX = 15_000, 40_000


def auto(batch: Path, cards: Path) -> dict:
    import soundfile as sf

    runs = [json.loads(line) for line in (batch / "batch.jsonl").read_text().splitlines() if line.strip()]
    asr = _asr("openai/whisper-large-v3-turbo")
    videos = []
    for r in runs:
        v = {"id": r["id"], "done": r["render"] == 0, "tries": r["tries"], "sec": r["plan_s"] + r["rest_s"]}
        sb_path = batch / r["id"] / "storyboard.json"
        if v["done"] and sb_path.exists():
            sb = json.loads(sb_path.read_text(encoding="utf-8"))
            card = json.loads((cards / f"batch_card_{r['id']}.json").read_text(encoding="utf-8"))
            wav, sr = sf.read(batch / r["id"] / "audio.wav", dtype="float32")
            lines = [d for s in sb["scenes"] for d in s["lines"]]
            reads = []
            for d in lines:
                if not d.get("spoken", True):
                    continue
                seg = wav[int(d["startMs"] / 1000 * sr):int(d["endMs"] / 1000 * sr)]
                heard = asr(seg, sr)
                reads.append({"text": d["text"], "heard": heard, "cer": round(read_cer(d["text"], heard), 3)})
            ends = [(d["startMs"], d["endMs"]) for d in lines]
            v.update(
                scenes=len(sb["scenes"]), lines=len(lines), card_lines=len(card["lines"]),
                duration_ms=sb["durationMs"],
                read_ok=sum(x["cer"] <= CER_PASS for x in reads), read_n=len(reads), reads=reads,
                gender_ok=sum(c.get("voice", "x-x").split("-")[1] == c.get("gender") for c in sb["characters"]),
                chars=len(sb["characters"]),
                dur_ok=DUR_MIN <= sb["durationMs"] <= DUR_MAX,
                sub_ok=sum(len(d["text"]) <= MAX_TEXT for d in lines),
                timing_ok=all(a[1] <= b[0] for a, b in zip(ends, ends[1:]))
                and all(e <= len(wav) / sr * 1000 + 50 for _, e in ends),
            )
        videos.append(v)
    done = [v for v in videos if v["done"]]

    def pct(a, b):
        return round(a / b, 3) if b else None

    summary = {
        "A1_done": pct(len(done), len(videos)),
        "A2_first_try": pct(sum(v["tries"] == 1 for v in done), len(done)),
        "A3_read": pct(sum(v["read_ok"] for v in done), sum(v["read_n"] for v in done)),
        "A4_voice_gender": pct(sum(v["gender_ok"] for v in done), sum(v["chars"] for v in done)),
        "A5_duration": pct(sum(v["dur_ok"] for v in done), len(done)),
        "A6_subtitle": pct(sum(v["sub_ok"] for v in done), sum(v["lines"] for v in done)),
        "A7_timing": pct(sum(v["timing_ok"] for v in done), len(done)),
        "A8_lines_kept": pct(sum(v["lines"] for v in done), sum(v["card_lines"] for v in done)),
        "A9_sec_mean": round(sum(v["sec"] for v in done) / max(1, len(done)), 1),
        "duration_s": [round(min(v["duration_ms"] for v in done) / 1000, 1),
                       round(max(v["duration_ms"] for v in done) / 1000, 1)] if done else None,
    }
    return {"summary": summary, "videos": videos}


def manual(path: Path) -> dict:
    """체크 파일: {"checker", "date", "videos": {id: {"B4": bool, "B7": bool|null, "scenes": [{"B1","B2","B3","B5","B6"(회상만), "note"}]}}}"""
    m = json.loads(path.read_text(encoding="utf-8"))
    scenes = [s for v in m["videos"].values() for s in v["scenes"]]
    fb = [s for s in scenes if s.get("B6") is not None]
    vids = list(m["videos"].values())
    refs = [v for v in vids if v.get("B7") is not None]

    def rate(xs, k):
        return round(sum(bool(x[k]) for x in xs) / len(xs), 3) if xs else None

    return {"checker": m.get("checker"), "date": m.get("date"), "scenes": len(scenes),
            "B1_no_text": rate(scenes, "B1"), "B2_single": rate(scenes, "B2"), "B3_people": rate(scenes, "B3"),
            "B4_consistent": rate(vids, "B4"), "B5_content": rate(scenes, "B5"),
            "B6_flashback": rate(fb, "B6"), "B6_n": len(fb), "B7_ref_face": rate(refs, "B7"), "B7_n": len(refs)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("batch")
    ap.add_argument("cards")
    ap.add_argument("--manual", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = auto(Path(a.batch), Path(a.cards))
    if a.manual:
        res["manual"] = manual(Path(a.manual))
    print(json.dumps(res["summary"], ensure_ascii=False))
    if "manual" in res:
        print(json.dumps(res["manual"], ensure_ascii=False))
    for v in res["videos"]:
        bad = [x for x in v.get("reads", []) if x["cer"] > CER_PASS]
        if bad:
            print(v["id"], "읽기 실패:", "; ".join(f"{x['text']} -> {x['heard']}" for x in bad))
    if a.out:
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
