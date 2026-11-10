"""컨텐츠포인트 감지 평가셋 만들기. 표준 라이브러리만 쓴다(젯슨에서 그대로 돌아감).

evalset/stories*.json(정답이 붙은 이야기 100개)과 evalset/filler*.json(이야기가 아닌 대화 90묶음)을 이어 붙여
앱이 서버로 보내는 DetectMessage 모양의 녹음을 만들고, 정답(gold)을 같이 저장한다.

세트
  short-clean / short-stt   녹음 25개, 각 이야기 4개 + 잡담. 3~5분짜리
  long-clean  / long-stt    녹음 4개, 각 이야기 25개 + 잡담 22~23묶음 + (있으면) 방송 소리 1군데. 실제 1시간 묶음에 가까운 긴 입력
  nostory-stt               잡담만 (정답: 카드 0개)
  longstory-clean / -stt    (evalset/longstories*.json이 있으면) 녹음 20개, 각 긴 이야기(15~36줄) 2개 + 잡담
  long2-clean / long2-stt   (같은 조건) 녹음 12개, 각 짧은 이야기 15 + 긴 이야기 4 + 잡담 18 + 방송 1군데. long보다 녹음이 많아 덜 흔들림
  tv                        --tv 로 준 방송 녹음 STT 결과 그대로 (정답: 카드 0~1개)

-clean은 문장부호만 뺀 글, -stt는 폰 STT처럼 망가뜨린 글이다.
  - 이어진 말을 한 줄로 합침(VAD가 쉼 없이 이어진 말을 한 구간으로 자름. 화자가 섞임). 화제 경계는 넘지 않음
  - 글자 오류: 받침 빠짐·더해짐, 비슷한 모음·자음 바뀜, 글자 빠짐 (음절의 약 7%)
  - 띄어쓰기 흔들림, 짧은 잡음 줄("음", "あの")
  - 시간은 글자당 약 0.17초(sample_tv_37min.json 중앙값)로 잡는다
같은 --seed면 언제나 같은 결과가 나온다.

예)
  python3 make_evalset.py --out sets --tv ~/llm-bench/sample_tv_37min.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ---------- 한글 자모 ----------
CHO = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
JUNG = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
JONG = [""] + list("ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")
SIM_JUNG = {"ㅐ": "ㅔ", "ㅔ": "ㅐ", "ㅓ": "ㅗ", "ㅗ": "ㅓ", "ㅚ": "ㅙ", "ㅙ": "ㅞ", "ㅞ": "ㅚ",
            "ㅕ": "ㅛ", "ㅡ": "ㅜ", "ㅜ": "ㅡ", "ㅢ": "ㅣ", "ㅒ": "ㅖ", "ㅖ": "ㅒ"}
SIM_CHO = {"ㄱ": "ㄲ", "ㄷ": "ㄸ", "ㅂ": "ㅃ", "ㅅ": "ㅆ", "ㅈ": "ㅉ", "ㄲ": "ㄱ", "ㄸ": "ㄷ",
           "ㅃ": "ㅂ", "ㅆ": "ㅅ", "ㅉ": "ㅈ", "ㄴ": "ㄹ", "ㄹ": "ㄴ", "ㅊ": "ㅈ", "ㅋ": "ㄱ", "ㅌ": "ㄷ", "ㅍ": "ㅂ"}
ADD_JONG = ["ㄴ", "ㄹ", "ㅇ", "ㅁ", "ㄱ", "ㅅ"]
JUNK = ["음", "아", "어", "응", "あの", "yeah", "ok", "흠", "the"]


def decompose(ch: str):
    code = ord(ch) - 0xAC00
    if not 0 <= code < 11172:
        return None
    return CHO[code // 588], JUNG[(code % 588) // 28], JONG[code % 28]


def compose(c: str, v: str, t: str) -> str:
    return chr(0xAC00 + CHO.index(c) * 588 + JUNG.index(v) * 28 + JONG.index(t))


def corrupt_syllable(ch: str, rng: random.Random) -> str:
    c, v, t = decompose(ch)
    op = rng.random()
    if op < 0.3 and t:
        return compose(c, v, "")                      # 받침 빠짐
    if op < 0.45 and not t:
        return compose(c, v, rng.choice(ADD_JONG))    # 받침 더해짐
    if op < 0.7 and v in SIM_JUNG:
        return compose(c, SIM_JUNG[v], t)             # 비슷한 모음
    if op < 0.88 and c in SIM_CHO:
        return compose(SIM_CHO[c], v, t)              # 비슷한 자음
    if op < 0.95:
        return ""                                     # 글자 빠짐
    return ch + compose("ㅇ", rng.choice(["ㅓ", "ㅏ", "ㅡ"]), "")  # 군소리 더해짐


def stt_text(text: str, rng: random.Random, rate: float) -> str:
    out = []
    for ch in text:
        if decompose(ch) and rng.random() < rate:
            out.append(corrupt_syllable(ch, rng))
        elif ch == " " and rng.random() < 0.12:
            continue                                  # 띄어쓰기 빠짐
        else:
            out.append(ch)
        if ch != " " and decompose(ch) and rng.random() < 0.015:
            out.append(" ")                           # 띄어쓰기 더해짐
    return re.sub(r"\s+", " ", "".join(out)).strip()


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[.,!?~\"'…]", "", text)).strip()


# ---------- 녹음 만들기 ----------

def build(items: list[dict], rng: random.Random, stt: bool, rate: float) -> tuple[dict, dict]:
    """items: {"type": "story"|"filler"|"tv", ...}. 줄을 시간 순으로 놓고 구간(seg)으로 묶는다."""
    lines = []  # (speaker, text, item_index, line_index)
    for ii, it in enumerate(items):
        for li, (spk, text) in enumerate(it["lines"]):
            lines.append({"spk": spk, "text": text, "item": ii, "li": li, "tv": it["type"] == "tv"})

    segs: list[dict] = []
    t = rng.randint(1000, 5000)
    i = 0
    while i < len(lines):
        group = [lines[i]]
        i += 1
        # VAD가 쉼 없이 이어진 말을 한 구간으로: 35% 확률로 다음 줄까지 이어 붙임(최대 4줄, 120자).
        # 화제가 바뀌는 곳(이야기·잡담 경계)은 붙이지 않는다. 붙이면 정답 이야기끼리 구간을 나눠 가져 채점이 흐려짐
        while (stt and i < len(lines) and lines[i]["item"] == group[0]["item"]
               and rng.random() < 0.35 and len(group) < 4
               and sum(len(g["text"]) for g in group) + len(lines[i]["text"]) < 120):
            group.append(lines[i])
            i += 1
        if stt and not group[0]["tv"] and rng.random() < 0.04:
            junk = rng.choice(JUNK)
            dur = 400 + rng.randint(0, 400)
            segs.append({"startMs": t, "endMs": t + dur, "text": junk, "members": []})
            t += dur + rng.randint(300, 900)
        raw = " ".join(g["text"] for g in group)
        text = stt_text(clean_text(raw), rng, rate) if stt and not group[0]["tv"] else clean_text(raw)
        if not text:
            text = clean_text(raw)
        dur = int(len(raw) * 170 + rng.randint(200, 600))
        segs.append({"startMs": t, "endMs": t + dur, "text": text, "members": group})
        t += dur + rng.randint(300, 1500)

    # 정답: 이야기마다 첫 줄·끝 줄·펀치라인이 들어간 구간 번호, 구간별 화자(한 사람만 말한 구간만)
    gold_stories = []
    for ii, it in enumerate(items):
        if it["type"] != "story":
            continue
        idx = [k for k, s in enumerate(segs) for m in s["members"] if m["item"] == ii]
        punch = sorted({k for k, s in enumerate(segs) for m in s["members"]
                        if m["item"] == ii and m["li"] in it["punch"]})
        speakers = {}
        for k in sorted(set(idx)):
            spk = {m["spk"] for m in segs[k]["members"] if m["item"] == ii}
            others = [m for m in segs[k]["members"] if m["item"] != ii]
            if len(spk) == 1 and not others:
                speakers[str(k)] = spk.pop()
        gold_stories.append({"id": it["id"], "kind": it["kind"], "from": min(idx), "to": max(idx),
                             "punch": punch, "speakers": speakers})
    sentences = [{"startMs": s["startMs"], "endMs": s["endMs"], "text": s["text"]} for s in segs]
    return sentences, {"stories": gold_stories,
                       "tvSegs": [k for k, s in enumerate(segs) if any(m["tv"] for m in s["members"])]}


def tv_excerpts(path: Path | None, n: int, size: int, rng: random.Random) -> list[dict]:
    if not path:
        return []
    sents = json.loads(path.read_text())["sentences"]
    out = []
    for _ in range(n):
        a = rng.randint(0, len(sents) - size)
        out.append({"type": "tv", "id": f"tv{a}", "lines": [["TV", s["text"]] for s in sents[a:a + size]]})
    return out


def message(rec_id: str, sentences: list[dict]) -> dict:
    dur = (sentences[-1]["endMs"] - sentences[0]["startMs"]) // 1000 if sentences else 0
    return {"jobId": rec_id, "partNo": 1, "isLast": True, "durationSec": dur, "sentences": sentences}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "sets"))
    ap.add_argument("--tv", help="방송 녹음 STT 결과(sample_tv_37min.json). 없으면 tv 세트와 long의 방송 부분을 뺌")
    ap.add_argument("--seed", type=int, default=20261110)
    ap.add_argument("--rate", type=float, default=0.07, help="-stt 세트의 음절 오류 확률")
    args = ap.parse_args()

    stories = [x for f in sorted((HERE / "evalset").glob("stories*.json"))
               for x in json.loads(f.read_text())["stories"]]
    fillers = [x for f in sorted((HERE / "evalset").glob("filler*.json"))
               for x in json.loads(f.read_text())["blocks"]]
    for s in stories:
        s["type"] = "story"
    for f in fillers:
        f["type"] = "filler"
    tv_path = Path(args.tv).expanduser() if args.tv else None
    out = Path(args.out)

    def write(set_name: str, recs: list[tuple[str, list[dict]]], stt: bool) -> None:
        d = out / set_name
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("*.json"):
            old.unlink()
        for rec_id, items in recs:
            rng = random.Random(f"{args.seed}-{set_name.split('-')[0]}-{rec_id}-{int(stt)}")
            sents, gold = build(items, rng, stt, args.rate)
            (d / f"{rec_id}.json").write_text(json.dumps(message(rec_id, sents), ensure_ascii=False, indent=1))
            (d / f"{rec_id}.gold.json").write_text(json.dumps(gold, ensure_ascii=False, indent=1))
        n_st = sum(1 for _, items in recs for it in items if it["type"] == "story")
        print(f"{set_name}: 녹음 {len(recs)}개, 정답 이야기 {n_st}개")

    rng = random.Random(args.seed)
    ns, nf = len(stories), len(fillers)
    # short: 녹음마다 이야기 4개, 이야기 앞뒤로 잡담 1~2묶음
    order = stories[:]
    rng.shuffle(order)
    fpool = fillers[:]
    rng.shuffle(fpool)
    short, fi = [], 0
    for r in range(ns // 4):
        items = []
        for st in order[r * 4:(r + 1) * 4]:
            for _ in range(rng.choice([1, 1, 2])):
                items.append(fpool[fi % nf])
                fi += 1
            items.append(st)
        items.append(fpool[fi % nf])
        fi += 1
        short.append((f"short{r + 1:02d}", items))
    # long: 이야기 25개씩 녹음 4개(각 30분 남짓). 잡담을 고루 나누고 방송 소리를 한 군데씩 넣음
    order = stories[:]
    rng.shuffle(order)
    fpool = fillers[:]
    rng.shuffle(fpool)
    n_long = max(1, ns // 25)
    tvs = tv_excerpts(tv_path, n_long, 25, rng)
    long = []
    for r in range(n_long):
        lst = order[r::n_long] + fpool[r::n_long]
        rng.shuffle(lst)  # 이야기끼리 바로 붙는 경우도 생김(실제 대화처럼)
        if tvs:
            lst.insert(rng.randint(len(lst) // 4, 3 * len(lst) // 4), tvs[r])
        long.append((f"long{r + 1}", lst))
    noitems = fillers[:]
    rng.shuffle(noitems)
    nostory = [("nostory1", noitems[: nf // 2]), ("nostory2", noitems[nf // 2:])]

    # ---- 2차(긴 이야기). 기존 세트가 바뀌지 않게 따로 만든 rng를 쓴다 ----
    lstories = [x for f in sorted((HERE / "evalset").glob("longstories*.json"))
                for x in json.loads(f.read_text())["stories"]]
    for s in lstories:
        s["type"] = "story"
    rng2 = random.Random(f"{args.seed}-v2")
    longstory, long2 = [], []
    if lstories:
        # longstory: 녹음마다 긴 이야기 2개 + 잡담. 이야기 경계·빌드업 측정용
        order = lstories[:]
        rng2.shuffle(order)
        fpool = fillers[:]
        rng2.shuffle(fpool)
        fi = 0
        for r in range(len(order) // 2):
            items = []
            for st in order[r * 2:(r + 1) * 2]:
                for _ in range(rng2.choice([1, 2])):
                    items.append(fpool[fi % nf])
                    fi += 1
                items.append(st)
            items.append(fpool[fi % nf])
            fi += 1
            longstory.append((f"lstory{r + 1:02d}", items))
        # long2: 긴 입력 녹음 12개(각 약 25~30분). 짧은 이야기 15 + 긴 이야기 4 + 잡담 18 + 방송 1군데.
        # 이야기는 녹음끼리 겹쳐 쓰되(풀을 돌려 씀) 한 녹음 안에서는 겹치지 않는다. long(4개)보다 녹음 수를 늘려 흔들림을 줄인다
        n_long2 = 12
        spool, lpool, fpool = stories[:], lstories[:], fillers[:]
        for p in (spool, lpool, fpool):
            rng2.shuffle(p)
        si = li = fi = 0
        tvs2 = tv_excerpts(tv_path, n_long2, 25, rng2)
        for r in range(n_long2):
            lst = []
            for _ in range(15):
                lst.append(spool[si % ns]); si += 1
            for _ in range(4):
                lst.append(lpool[li % len(lpool)]); li += 1
            for _ in range(18):
                lst.append(fpool[fi % nf]); fi += 1
            rng2.shuffle(lst)
            if tvs2:
                lst.insert(rng2.randint(len(lst) // 4, 3 * len(lst) // 4), tvs2[r])
            long2.append((f"long2_{r + 1:02d}", lst))

    for stt in (False, True):
        suf = "stt" if stt else "clean"
        write(f"short-{suf}", short, stt)
        write(f"long-{suf}", long, stt)
        if lstories:
            write(f"longstory-{suf}", longstory, stt)
            write(f"long2-{suf}", long2, stt)
    write("nostory-stt", nostory, True)
    if tv_path:
        d = out / "tv"
        d.mkdir(parents=True, exist_ok=True)
        m = json.loads(tv_path.read_text())
        (d / "tv37.json").write_text(json.dumps(message("tv37", m["sentences"]), ensure_ascii=False))
        (d / "tv37.gold.json").write_text(json.dumps({"stories": [], "tvSegs": list(range(len(m["sentences"])))}))
        print("tv: 녹음 1개, 정답 이야기 0개")


if __name__ == "__main__":
    main()
