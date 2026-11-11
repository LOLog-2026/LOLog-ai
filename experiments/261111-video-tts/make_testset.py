"""TTS 정확도 평가용 대사 144개 (#33). 감지 평가셋(#20)의 이야기 대사 120개 + 어려운 대사 24개.

    python make_testset.py EVALSET_DIR testset.json   # EVALSET_DIR: #20 브랜치의 experiments/261110-detect-eval/evalset

길이별로 고르게 뽑는다(짧음 2~6자 30개, 중간 7~20자 50개, 긺 21~40자 40개). 목소리는 24명을 돌아가며 붙여
화자마다 6개씩 읽는다.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

VOICES = [f"{a}-{g}-{n:02d}" for a in ("kid", "young", "middle") for g in ("f", "m") for n in range(1, 5)]
BUCKETS = [("short", 2, 6, 30), ("medium", 7, 20, 50), ("long", 21, 40, 40)]
# (종류, TTS에 넣는 글, 들려야 하는 말). 숫자·영어·낱자모·기호처럼 읽기 어려운 것
HARD = [
    ("jamo", "ㅋㅋㅋㅋ 아 진짜 웃겨", "아 진짜 웃겨"),
    ("jamo", "ㅠㅠ 나 어떡해", "나 어떡해"),
    ("jamo", "헐 대박ㅋㅋ", "헐 대박"),
    ("jamo", "아니 ㅎㅎ 그게 아니고", "아니 그게 아니고"),
    ("digit", "3시에 만나기로 했잖아", "세 시에 만나기로 했잖아"),
    ("digit", "그거 5만 원이나 해?", "그거 오만 원이나 해"),
    ("digit", "나 어제 2시간밖에 못 잤어", "나 어제 두 시간밖에 못 잤어"),
    ("digit", "버스가 20분째 안 와", "버스가 이십 분째 안 와"),
    ("english", "그 카페 와이파이 비번이 뭐였지", "그 카페 와이파이 비번이 뭐였지"),
    ("english", "나 오늘 PT 받고 왔어", "나 오늘 피티 받고 왔어"),
    ("english", "그거 OK라고 했잖아", "그거 오케이라고 했잖아"),
    ("english", "유튜브에서 봤는데 진짜래", "유튜브에서 봤는데 진짜래"),
    ("short", "어?", "어"),
    ("short", "응.", "응"),
    ("short", "뭐?", "뭐"),
    ("short", "아니.", "아니"),
    ("exclaim", "야!!! 그걸 왜 먹어!!!", "야 그걸 왜 먹어"),
    ("exclaim", "와... 진짜 대단하다...", "와 진짜 대단하다"),
    ("exclaim", "아 맞다!", "아 맞다"),
    ("exclaim", "잠깐만, 잠깐만, 잠깐만!", "잠깐만 잠깐만 잠깐만"),
    ("repeat", "아니 아니 아니 그게 아니라", "아니 아니 아니 그게 아니라"),
    ("repeat", "하하하하 진짜 웃기다", "하하하하 진짜 웃기다"),
    ("long", "내가 분명히 냉장고에 넣어 놨는데 아침에 보니까 식탁 위에 그대로 있는 거야", "내가 분명히 냉장고에 넣어 놨는데 아침에 보니까 식탁 위에 그대로 있는 거야"),
    ("long", "그래서 선생님이 누구냐고 물어봤는데 아무도 대답을 안 해서 결국 다 같이 남았어", "그래서 선생님이 누구냐고 물어봤는데 아무도 대답을 안 해서 결국 다 같이 남았어"),
]


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    seen, pool = set(), []
    for f in sorted(src.glob("*stories*.json")):
        for st in json.loads(f.read_text(encoding="utf-8"))["stories"]:
            for _, text in st["lines"]:
                text = text.strip()
                if text not in seen:
                    seen.add(text)
                    pool.append(text)
    rng = random.Random(33)
    rng.shuffle(pool)
    items = []
    for name, lo, hi, n in BUCKETS:
        picked = [t for t in pool if lo <= len(t) <= hi][:n]
        if len(picked) < n:
            raise SystemExit(f"{name}: {len(picked)}개뿐 (필요 {n})")
        items += [{"cat": name, "text": t, "expect": t} for t in picked]
    items += [{"cat": "hard-" + c, "text": t, "expect": e} for c, t, e in HARD]
    rng.shuffle(items)
    for i, it in enumerate(items):
        it.update(id=i, voice=VOICES[i % len(VOICES)])
    out.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    print(len(items), "개 →", out)


if __name__ == "__main__":
    main()
