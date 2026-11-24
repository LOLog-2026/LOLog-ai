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
/opt/lolog/venv/bin/python -m pip freeze > freeze-jetson-1.txt   # 1호
uv pip freeze -p /opt/lolog/venv > freeze-jetson-2.txt           # 2호. pydeps-lipsync는 아래 참고
uv pip freeze -p /opt/lolog/tts-venv > freeze-jetson-3.txt       # 3호
```

2·3호 venv에는 pip이 없어서 `python -m pip freeze`가 안 된다. `uv pip freeze -p <venv>`로 받는다.
`uv pip freeze`는 로컬 wheel로 깐 패키지를 파일 경로로 적는다(3호 `av`). 파일에는 `이름==버전`으로 바꿔 적는다.

## 2호 venv 구조

2호 워커(`/opt/lolog/venv/bin/python`)가 쓰는 패키지는 두 곳에 있다. 둘 다 `/opt/lolog` 안이고 `lolog` 소유다.

| 곳 | 든 것 | 워커가 보는 방법 |
|---|---|---|
| `/opt/lolog/venv` | `torch`, `diffusers` 등 114개 | venv 그대로 |
| `/opt/lolog/pydeps-lipsync` | `mediapipe`, `opencv` 등 18개 | `lolog-worker@comic.service.d/lipsync.conf`의 `PYTHONPATH` |

- `pydeps-lipsync`의 버전은 `uv pip freeze`로 안 나온다. `*.dist-info/METADATA`의 `Name`·`Version`에서 읽는다.
- `freeze-jetson-2.txt`는 두 곳을 합친 목록이다. 버전이 겹치지 않아서 한 venv에 통째로 깔 수 있다
  (linux aarch64·Python 3.12에서 풀리는 것까지 확인했고, 실제로 깔아 보지는 않았다).

### 실험용 venv에서 옮긴 기록 (#69)

2026-11-24까지 `/opt/lolog/venv`는 비어 있었고, `flux-venv.pth` 한 줄로 FLUX 실험용 체크아웃의
`/home/minju/LOLog-ai/.venv`를 빌려 썼다. 그 폴더에서 `uv sync`를 돌리면 워커가 쓰는 패키지가 지워지는 구조였다.

- `scripts/setup_comic_venv_jetson.sh`로 패키지 파일을 그대로 복사해 옮겼다. 새로 내려받지 않아서 버전이 같다
  (옮긴 뒤 freeze가 옮기기 전과 같은 것을 확인했다).
- 되돌리기: `sudo bash setup_comic_venv_jetson.sh rollback`. 바꾸기 전 폴더가
  `/opt/lolog/venv/lib/python3.12/site-packages.pth-only`에 남아 있어야 한다.
- 워커는 이제 `/home/minju/LOLog-ai/.venv`를 보지 않는다. 그 폴더와 `site-packages.pth-only`는 며칠 써 본 뒤 지워도 된다.
  둘 중 하나라도 지우면 되돌리기는 못 쓴다.
