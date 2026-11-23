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

```bash
/opt/lolog/venv/bin/python -m pip freeze > freeze-jetson-1.txt       # 1호, 2호(freeze-jetson-2.txt)
/opt/lolog/tts-venv/bin/python -m pip freeze > freeze-jetson-3.txt   # 3호
```

## 아직 확인하지 않은 것 (#64)

- `requirements-comic.txt`는 코드의 import에서 뽑았고 2호의 실제 venv와 맞춰 보지 않았다. 버전이 없다.
- 세 대의 `freeze-jetson-*.txt`가 아직 없다.
