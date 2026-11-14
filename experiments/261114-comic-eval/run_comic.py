"""4컷만화 평가 실행: 감지 평가셋 이야기로 카드를 만들어 콘티 → (그림)까지 돌리고 결과를 남긴다.

plan   콘티만(1호 LLM만 있으면 어디서나). 만화·영상 콘티의 한 번에 통과율, 다시 물은 횟수, 검사 오류, 시간
draw   콘티 + 그림(2호, comic 워커를 멈추고). 만화 PNG, 컷 프롬프트, 단계별 시간

    # 1호 LLM 주소·키는 /etc/lolog/worker.env와 같은 값
    export LLM_BASE_URL=http://100.98.23.5:8000/v1 LLM_API_KEY=<키> LLM_MODEL=local
    python3 run_comic.py plan out/plan --video                 # 할 일 4: Qwen3.5 콘티 확인
    python3 run_comic.py draw out/r1 --chars <default-characters 폴더>   # 할 일 2: 만화 12편

카드는 감지 평가셋(실제 사람·녹음과 관계없음)의 이야기를 그대로 카드로 만든 것이다. 영상 개발용 카드 devD와 같은 12개.
도는 동안 코드를 고쳐도 섞이지 않게, 시작할 때 src를 결과 폴더에 복사해 그것으로 돌린다(#33 3차의 교훈).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
log = logging.getLogger("run_comic")
REPO = HERE.parents[1]
EVALSET = REPO / "experiments/261110-detect-eval/evalset"
CARDS = ["L04", "L07", "L14", "L18", "L28", "L38", "s09", "s22", "s33", "s47", "s67", "s96"]
# 첫 화자에 기본 화자 그림을 붙이는 카드(참조 그림 경로 확인용)
REFS = {"s09": "young-f-02", "s33": "middle-m-01", "L28": "young-m-01"}


def stories() -> dict[str, dict]:
    out = {}
    for f in sorted(EVALSET.glob("*stories*.json")):
        for s in json.loads(f.read_text(encoding="utf-8"))["stories"]:
            out[s["id"]] = s
    return out


def card(story: dict, n: int) -> dict:
    """감지 평가셋 이야기 → ComicSource 모양 카드. 화자는 처음 말한 순서대로 화자1, 화자2…"""
    order: list[str] = []
    for who, _ in story["lines"]:
        if who not in order:
            order.append(who)
    key = {w: f"화자{i + 1}" for i, w in enumerate(order[:4])}
    lines = [{"id": i + 1, "speaker": key[w], "text": t} for i, (w, t) in enumerate(story["lines"]) if w in key]
    punch = story["lines"][story["punch"][0]][1]
    return {"comicId": n, "videoId": n, "cardId": n, "kind": story["kind"], "summary": punch[:30], "place": None,
            "speakers": [{"key": k, "name": None} for k in key.values()], "lines": lines}


def attach_ref(src: dict, cid: str, chars: Path | None) -> None:
    if not chars:
        return
    c = {x["id"]: x for x in json.loads((chars / "characters.json").read_text())["characters"]}[cid]
    src["speakers"][0]["character"] = {"ref": f"default:{cid}", "appearance": c["appearance"],
                                       "imageUrl": (chars / c["image"]).resolve().as_uri()}


class Tries(logging.Handler):
    """콘티를 몇 번 물었는지와 마지막 검사 오류 수를 로그에서 센다."""

    def __init__(self):
        super().__init__(logging.INFO)
        self.tries = 0

    def emit(self, record):
        if "콘티" in record.getMessage() and "번째" in record.getMessage():
            self.tries += 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["plan", "draw"])
    ap.add_argument("out", type=Path)
    ap.add_argument("--chars", type=Path, help="default-characters 폴더(characters.json, PNG)")
    ap.add_argument("--video", action="store_true", help="plan: 영상 콘티도 만든다")
    ap.add_argument("--seed", type=int, default=33)
    ap.add_argument("--only", nargs="*", help="카드 id 일부만")
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    src_copy = a.out / "src"
    if not src_copy.exists():
        shutil.copytree(REPO / "src", src_copy)
        commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, check=False).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", "--", "src"], check=False).returncode
        (a.out / "COMMIT").write_text(commit + (" dirty" if dirty else "") + "\n")
    sys.path.insert(0, str(src_copy))

    from lolog_ai.worker.llm import Llm
    from lolog_ai.worker.storyboard_gen import StoryboardInvalid, fixed_looks, make_storyboard

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    tries = Tries()
    logging.getLogger().addHandler(tries)
    llm = Llm(os.environ["LLM_BASE_URL"], os.environ.get("LLM_MODEL", "local"), os.environ.get("LLM_API_KEY"))
    all_st = stories()
    res_f = (a.out / "results.jsonl").open("a", encoding="utf-8")
    for n, cid in enumerate(a.only or CARDS, 1):
        o = a.out / cid
        o.mkdir(exist_ok=True)
        src = card(all_st[cid], n)
        if cid in REFS:
            attach_ref(src, REFS[cid], a.chars)
        (o / "card.json").write_text(json.dumps(src, ensure_ascii=False, indent=1), encoding="utf-8")
        r = {"id": cid, "mode": a.mode, "seed": a.seed, "ref": REFS.get(cid) if a.chars else None}
        tries.tries = 0
        t0 = time.perf_counter()
        try:
            sb = make_storyboard(src, llm)
            r.update(comic_ok=True)
            (o / "storyboard.json").write_text(json.dumps(sb, ensure_ascii=False, indent=1), encoding="utf-8")
        except StoryboardInvalid as e:
            sb = None
            r.update(comic_ok=False, comic_err=str(e)[:500])
        r.update(comic_tries=tries.tries, comic_plan_s=round(time.perf_counter() - t0, 1))
        if a.video:
            from lolog_ai.worker.video_gen import make_video_storyboard
            tries.tries = 0
            t0 = time.perf_counter()
            try:
                vsb = make_video_storyboard(src, llm)
                r.update(video_ok=True, video_scenes=len(vsb["scenes"]))
                (o / "video_storyboard.json").write_text(json.dumps(vsb, ensure_ascii=False, indent=1),
                                                         encoding="utf-8")
            except StoryboardInvalid as e:
                r.update(video_ok=False, video_err=str(e)[:500])
            r.update(video_tries=tries.tries, video_plan_s=round(time.perf_counter() - t0, 1))
        if a.mode == "draw" and sb:
            from lolog_ai.render import prompt
            from lolog_ai.worker.comic import _image
            from lolog_ai.worker.draw import draw
            refs = {}
            for k, c in fixed_looks(src).items():
                with open(c["imageUrl"].removeprefix("file://"), "rb") as f:
                    refs[k] = _image(f.read())
            (o / "prompts.txt").write_text("\n\n".join(prompt(sb, p, set(refs)) for p in sb["panels"]),
                                           encoding="utf-8")
            t0 = time.perf_counter()
            (o / "comic.png").write_bytes(draw(sb, seed=a.seed, refs=refs))
            r.update(draw_s=round(time.perf_counter() - t0, 1))
        print(json.dumps(r, ensure_ascii=False), file=res_f, flush=True)
        log.info("%s 끝: %s", cid, r)


if __name__ == "__main__":
    main()
