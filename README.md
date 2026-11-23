# LOLog AI

일상 대화에서 웃긴 이야기를 찾아 카드로 모으고 4컷만화로 그려 주는 서비스 LOLog의 AI(모델·전처리·추론) 레포입니다.

## 환경 세팅

[uv](https://docs.astral.sh/uv/)를 씁니다.

```bash
# uv 설치 (처음 한 번)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 의존성 설치 (.venv 생성)
uv sync

# 테스트
uv run pytest

# 실행
uv run python scripts/<스크립트>.py
```

`uv sync`는 개발·테스트에 쓰는 것만 깝니다. 모델을 돌리는 패키지(torch, diffusers, qwen-tts 등)는 젯슨에만 깔고, 역할별 목록은 [deploy/](deploy/README.md)에 있습니다.

패키지 추가는 `uv add <패키지>`, 개발용 도구는 `uv add --dev <패키지>`로 합니다. `pyproject.toml`과 `uv.lock`을 같이 커밋합니다. 워커 코드가 쓰는 패키지는 `deploy/requirements-<역할>.txt`에도 적습니다.

PR을 올리면 GitHub Actions가 `uv sync --frozen` 뒤 `pytest`를 돌립니다.

노트북에서 쓰려면 VS Code나 Jupyter에서 커널로 `.venv`를 고릅니다.

## 폴더 구조

```
configs/       학습·추론 설정(yaml)
deploy/        젯슨 역할별 패키지 목록
data/          데이터 (git 제외, 출처만 data/README.md에 기록)
experiments/   실험 기록, 실험 하나에 폴더 하나 (experiments/README.md 참고)
notebooks/     탐색용 노트북
scripts/       학습·평가·추론 실행 스크립트
src/lolog_ai/  공통 코드 (데이터 로딩, 전처리, 모델, 평가)
```

## 작업 규칙

브랜치·커밋·이슈·PR 규칙은 `CLAUDE.md`를 따릅니다. 실험은 [experiments/README.md](experiments/README.md)의 순서대로 기록합니다.
