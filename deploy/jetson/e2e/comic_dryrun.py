"""서버 없이 2호에서 만화 한 편 그려 보기 (콘티는 1호 LLM, 그림은 2호 FLUX).

첫 실행은 nf4 모델을 저장하느라 몇 분 더 걸린다. 두 번째 실행 시간이 실제 시간.
사용(2호): sudo -u lolog env PYTHONPATH=/opt/lolog/LOLog-ai/src HF_HOME=/opt/lolog/hf-cache \
  HF_HUB_OFFLINE=1 LLM_BASE_URL=http://100.98.23.5:8000/v1 LLM_API_KEY=<키> LLM_MODEL=qwen3-4b-q4 \
  /opt/lolog/venv/bin/python comic_dryrun.py /opt/lolog/dryrun
"""
import json
import logging
import os
import sys
import time
from pathlib import Path

from lolog_ai.worker.draw import draw
from lolog_ai.worker.llm import Llm
from lolog_ai.worker.storyboard_gen import make_storyboard

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

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

out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
out.mkdir(parents=True, exist_ok=True)
llm = Llm(os.environ["LLM_BASE_URL"], os.environ.get("LLM_MODEL", "local"), os.environ.get("LLM_API_KEY"))
t0 = time.perf_counter()
sb = make_storyboard(SRC, llm)
t1 = time.perf_counter()
(out / "storyboard.json").write_text(json.dumps(sb, ensure_ascii=False, indent=1))
png = draw(sb, seed=1)
t2 = time.perf_counter()
(out / "comic.png").write_bytes(png)
print(f"콘티 {t1 - t0:.0f}초, 그림 {t2 - t1:.0f}초, 합계 {t2 - t0:.0f}초 → {out}/comic.png")
