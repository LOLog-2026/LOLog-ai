# LOLog AI

일상 대화에서 웃긴 이야기를 찾아 카드로 모으고 4컷만화로 그려 주는 서비스 LOLog의 AI(모델·전처리·추론) 레포입니다.

## 환경 세팅

[uv](https://docs.astral.sh/uv/)를 씁니다.

```bash
# uv 설치 (처음 한 번)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 의존성 설치 (.venv 생성)
uv sync

# 실행
uv run python scripts/<스크립트>.py
```

패키지 추가는 `uv add <패키지>`, 개발용 도구는 `uv add --dev <패키지>`로 합니다. `pyproject.toml`과 `uv.lock`을 같이 커밋합니다.

노트북에서 쓰려면 VS Code나 Jupyter에서 커널로 `.venv`를 고릅니다.

## 폴더 구조

```
configs/       학습·추론 설정(yaml)
data/          데이터 (git 제외, 출처만 data/README.md에 기록)
experiments/   실험 기록, 실험 하나에 폴더 하나 (experiments/README.md 참고)
notebooks/     탐색용 노트북
scripts/       학습·평가·추론 실행 스크립트
src/lolog_ai/  공통 코드 (데이터 로딩, 전처리, 모델, 평가)
```

## 작업 규칙

브랜치·커밋·이슈·PR 규칙은 `CLAUDE.md`를 따릅니다. 실험은 [experiments/README.md](experiments/README.md)의 순서대로 기록합니다.
