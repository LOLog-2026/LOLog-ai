"""서버 없이 2호에서 참조 그림을 넣은 만화 한 편 그려 보기 (comic_dryrun.py와 같은 이야기).

화자1에 기본 화자 young-f-02, 화자2에 young-m-01을 붙인다. 그림은 default-characters 폴더에서 file:// 주소로 받는다.
사용(2호): sudo -u lolog env PYTHONPATH=/opt/lolog/LOLog-ai/src HF_HOME=/opt/lolog/hf-cache HF_HUB_OFFLINE=1 \
  LLM_BASE_URL=http://100.98.23.5:8000/v1 LLM_API_KEY=<키> LLM_MODEL=qwen3-4b-q4 \
  /opt/lolog/venv/bin/python scripts/comic_refs_dryrun.py /opt/lolog/chars/default-characters /opt/lolog/dryrun-refs
"""
import json
import logging
import os
import sys
import time
from pathlib import Path

from lolog_ai.worker.comic import comic
from lolog_ai.worker.llm import Llm

SRC = {
    "comicId": 1, "cardId": 1, "kind": "웃김",
    "summary": "편의점에서 지갑도 없고 폰도 꺼져 곤란했는데, 알바생이 그냥 가져가라며 보내 줬다. 알고 보니 그 알바생이 우리 과 조교님이었다.",
    "place": None,
    "speakers": [{"key": "화자1", "name": None}, {"key": "화자2", "name": None}],
    "lines": [
        {"id": 1, "speaker": "화자1", "text": "야 나 어제 진짜 웃긴 일 있었어"},
        {"id": 2, "speaker": "화자2", "text": "뭔데 또"},
        {"id": 3, "speaker": "화자1", "text": "편의점에서 삼각김밥 사고 계산하려는데 지갑을 안 가져온 거야"},
        {"id": 4, "speaker": "화자1", "text": "폰도 배터리가 나가서 꺼져 있는 거야"},
        {"id": 5, "speaker": "화자2", "text": "아 대박 그래서 어떻게 했어"},
        {"id": 6, "speaker": "화자1", "text": "알바생이 나를 엄청 불쌍하게 보더니 그냥 가져가라고 하더라"},
        {"id": 7, "speaker": "화자1", "text": "근데 그 알바생이 우리 과 조교님이었어"},
        {"id": 8, "speaker": "화자2", "text": "아 진짜 미쳤다 그럼 내일 수업 어떡해"},
    ],
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
chars_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
by_id = {c["id"]: c for c in json.loads((chars_dir / "characters.json").read_text())["characters"]}
src = json.loads(json.dumps(SRC))
for sp, cid in zip(src["speakers"], ["young-f-02", "young-m-01"]):
    c = by_id[cid]
    sp["character"] = {"ref": f"default:{cid}", "appearance": c["appearance"],
                       "imageUrl": (chars_dir / c["image"]).resolve().as_uri()}


class FakeApi:
    """백엔드 대신 결과를 파일로 남긴다. fetch는 실제 워커 코드(urllib)를 그대로 쓴다."""
    from lolog_ai.worker.api import InternalApi as _A
    fetch = _A.fetch
    timeout = 60

    def get_comic_source(self, _):
        return src

    def post_storyboard(self, _, sb):
        (out / "storyboard.json").write_text(json.dumps(sb, ensure_ascii=False, indent=1))

    def post_image(self, _, png):
        (out / "comic.png").write_bytes(png)


llm = Llm(os.environ["LLM_BASE_URL"], os.environ.get("LLM_MODEL", "local"), os.environ.get("LLM_API_KEY"))
t0 = time.perf_counter()
comic({"comicId": 1}, FakeApi(), llm)
print(f"합계 {time.perf_counter() - t0:.0f}초 → {out}/comic.png")
