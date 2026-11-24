# 젯슨 역할별 패키지 목록

젯슨은 `uv sync`로 깔지 않는다. 역할마다 필요한 패키지가 다르고, 3호는 `transformers` 버전이 달라
venv도 따로 쓴다. 그래서 `pyproject.toml`에는 개발·테스트에 쓰는 것만 두고, 젯슨에 까는 것은 여기에 둔다.

| 파일 | 젯슨 | 역할 | venv |
|---|---|---|---|
| `requirements-detect.txt` | 1호 | `--role detect` | `/opt/lolog/venv` |
| `requirements-comic.txt` | 2호 | `--role comic` | `/opt/lolog/venv` |
| `requirements-video.txt` | 3호 | `--role video` | `/opt/lolog/tts-venv` |

```bash
uv pip install -p <venv> -r deploy/requirements-<역할>.txt
```

3호는 `scripts/setup_tts_jetson.sh`가 이 파일을 읽어 깐다.

## 규칙

- 워커 코드에 새 패키지를 쓰면 그 역할의 파일에 적는다. 테스트도 쓰면 `pyproject.toml`의 dev 그룹에도 넣는다.
- 실제로 깔린 버전은 젯슨에서 받아 여기에 `freeze-jetson-<번호>.txt`로 둔다. 다시 깔 때 같은 버전을 되살리는 기준이다.
  패키지를 새로 깔거나 올리면 다시 받는다. 지금 파일은 2026-11-23에 받았다.

```bash
/opt/lolog/venv/bin/python -m pip freeze > freeze-jetson-1.txt      # 1호
uv pip freeze -p /home/minju/LOLog-ai/.venv > freeze-jetson-2.txt   # 2호. pydeps-lipsync는 아래 참고
uv pip freeze -p /opt/lolog/tts-venv > freeze-jetson-3.txt          # 3호
```

2·3호 venv에는 pip이 없어서 `python -m pip freeze`가 안 된다. `uv pip freeze -p <venv>`로 받는다.
`uv pip freeze`는 로컬 wheel로 깐 패키지를 파일 경로로 적는다(3호 `av`). 파일에는 `이름==버전`으로 바꿔 적는다.

## 2호 venv 구조

2호 워커(`/opt/lolog/venv/bin/python`)가 쓰는 패키지는 세 곳에 나뉘어 있다.

| 곳 | 든 것 | 워커가 보는 방법 |
|---|---|---|
| `/opt/lolog/venv` | 없음 | 실행 파일만 쓴다 |
| `/home/minju/LOLog-ai/.venv` | `torch`, `diffusers` 등 114개 | `/opt/lolog/venv`의 `site-packages/flux-venv.pth` |
| `/opt/lolog/pydeps-lipsync` | `mediapipe`, `opencv` 등 18개 | `lolog-worker@comic.service.d/lipsync.conf`의 `PYTHONPATH` |

- **2호의 `~/LOLog-ai`에서 `uv sync`를 돌리지 않는다.** `uv sync`는 `pyproject.toml`에 없는 패키지를 지운다.
  워커가 쓰는 `diffusers`·`bitsandbytes`·`onnxruntime`은 `pyproject.toml`에 없고, #66 뒤로는 `torch`·`transformers`도 없다.
  그 폴더는 FLUX 실험 때 만든 체크아웃이고, 워커 코드는 `/opt/lolog/LOLog-ai/src`에 따로 있다.
- `pydeps-lipsync`의 버전은 `uv pip freeze`로 안 나온다. `*.dist-info/METADATA`의 `Name`·`Version`에서 읽는다.
- `freeze-jetson-2.txt`는 두 곳을 합친 목록이다. 버전이 겹치지 않아서 한 venv에 통째로 깔 수 있다
  (linux aarch64·Python 3.12에서 풀리는 것까지 확인했고, 실제로 깔아 보지는 않았다).
