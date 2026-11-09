# 실험: 폰 안 한국어 STT 성능 올리기

- 날짜: 2026-11-09
- 담당: 민주
- 브랜치: `claude/project-thread-unkspz`

## 목적

앱이 쓰는 폰 안 STT(sherpa-onnx SenseVoice-Small int8 + silero VAD, 구간 앞 0.3초 붙이기)의
지금 성능을 오픈소스 데이터로 재고, 폰 제약(CPU 2~4스레드) 안에서 설정·모델을 하나씩 바꿔 가며 올린다.
SenseVoice는 프롬프트를 받지 않으므로 바꿀 수 있는 것은 언어 고정, ITN, VAD 값, 입력 크기 맞춤, 모델, 후처리다.

## 데이터 (오픈소스만)

| 세트 | 내용 | 용도 |
|---|---|---|
| `fleurs-dev` | FLEURS ko_kr dev 226발화 | 설정 고르기(튜닝) |
| `fleurs-clean` | FLEURS ko_kr test 382발화 | 확인용 |
| `fleurs-babble10` / `babble5` | test + 다른 화자 3명 웅성거림 SNR 10/5dB | 시끄러운 자리 |
| `fleurs-quiet` | test를 -20dB로 | 폰이 멀거나 주머니 속 |
| `fleurs-longform` | test 발화를 쉼과 함께 이어 붙인 5분 녹음 18개 | 앱처럼 VAD로 잘라 인식 |
| `zeroth-*` | Zeroth-Korean test 457발화 (`--zeroth`) | 다른 화자·녹음 조건 |

FLEURS는 CC-BY-4.0, `storage.googleapis.com`에서 받는다. Zeroth-Korean은 CC-BY-4.0, HF `kresnik/zeroth_korean`.

채점: 주 지표는 띄어쓰기를 뺀 CER(%). 문장부호·따옴표는 빼고, 기준 글의 괄호 속 원어 표기(`비슈케크(Bishkek)`)는 읽지 않으므로 뺀다.
FLEURS 기준 글은 숫자를 아라비아 숫자로 적는다(test 382개 중 86개에 숫자). 그래서 ITN을 끄면 "백 퍼센트" vs "100%"가 틀린 글자로 잡힌다.

속도: RTF = 처리 시간 / 소리 길이. `--threads`로 스레드 수를 묶는다(리눅스는 코어도 고정).
맥 M 계열은 중급 안드로이드보다 2~3배 빠르므로 RTF 절댓값보다 설정 간 비교로 본다.

## 돌리는 법

```bash
pip install sherpa-onnx soundfile numpy   # Zeroth까지: huggingface_hub pyarrow
bash fetch_models.sh                       # models/ 에 모델 받기 (github.com)
python -I prepare_data.py --out data/stt   # FLEURS 받아 세트 만들기 (약 1.8GB)
python -I run_bench.py --set fleurs-dev --model sensevoice-int8 --threads 2 --tag baseline
```

결과는 `results.jsonl`에 쌓이고, 아래 표에 옮긴다.

## 결과

| # | 바꾼 것 | 세트 | CER | RTF(2스레드) | 비고 |
|---|---|---|---|---|---|
| | | | | | |

## 결론과 다음 할 일

-
