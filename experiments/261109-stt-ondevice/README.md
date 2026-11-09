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

## 결과 (진행 중, 2026-11-09 16:02 기준 32개 중 22개)

환경: MacBook Air M2 16GB, 전원 연결, sherpa-onnx 2.0.1, `--threads 2`. 실행 중 RTF가 0.017~0.023으로 고르게 나와 발열로 느려진 흔적은 없다.
기준 설정: `sensevoice-int8 --lang ko --itn 1 --vad silero --vad-threshold 0.5 --min-silence 0.5 --max-speech 20 --pad 0.3 --gain none`.
CER은 띄어쓰기를 뺀 값(%), RTF는 처리 시간/소리 길이.

### A. 기준 성능

| 세트 | 발화 | CER | 띄어쓰기 포함 | RTF | 구간 수 | 빈 결과 |
|---|---|---|---|---|---|---|
| fleurs-dev | 226 | 6.92 | 7.84 | 0.0185 | 292 | 0 |
| fleurs-clean (test) | 382 | 6.16 | 7.17 | 0.0181 | 498 | 0 |
| fleurs-babble10 | 382 | 15.42 | 16.10 | 0.0230 | 411 | 3 |
| fleurs-babble5 | 382 | 23.92 | 24.29 | 0.0233 | 403 | 2 |
| fleurs-quiet (-20dB) | 382 | 13.36 | 14.30 | 0.0176 | 419 | 9 |
| fleurs-longform (5분×18, 89분) | 18 | 8.07 | 8.83 | 0.0172 | 453 | 0 |

- 깨끗한 소리는 6% 안팎이지만 웅성거림·작은 소리에서 2~4배 나빠진다.
- 같은 test 발화를 긴 녹음으로 이어 붙이면 6.16 → 8.07. VAD로 자르는 과정의 손해로 보인다.
- `--max-speech 20`인데 최대 구간이 21~24초로 나온다. silero가 최대 길이에서 바로 끊지 않고 조금 넘겨 자르고, 앞 0.3초가 더 붙기 때문으로 보인다(확인 필요).

### B. 한 번에 하나만 바꾸기 (fleurs-dev, 기준 6.92)

| 바꾼 것 | CER | 차이 | 비고 |
|---|---|---|---|
| `--itn 0` | 6.95 | +0.03 | 한국어에서는 ITN이 숫자를 거의 안 바꾸는 듯(확인 예정) |
| `--lang auto` | 7.01 | +0.09 | 한국어 고정 유지 |
| `--vad none` (통째로) | 6.73 | -0.19 | 짧은 발화라 가능한 상한. RTF 0.0218 |
| `--vad-threshold 0.3` | **6.61** | **-0.31** | 구간 292→265 |
| `--vad-threshold 0.4` | 6.90 | -0.02 | |
| `--vad-threshold 0.6` | 7.53 | +0.61 | 빈 결과 1 |
| `--min-silence 0.25` | 7.55 | +0.63 | 구간 481개로 잘게 쪼개짐 |
| `--min-silence 0.8` | 6.68 | -0.24 | |
| `--pad 0` | 8.24 | +1.32 | 앞 여유가 없으면 첫 글자가 잘림 |
| `--pad 0.5` | 7.07 | +0.15 | 0.3이 맞다 |
| `--tail-pad 0.2` | 6.94 | +0.02 | 효과 없음 |
| `--gain rms` (VAD 뒤, 구간마다) | 6.82 | -0.10 | |
| `--model zipformer-ko` | 49.2 | — | 빈 결과 75개. 설정 오류 의심, 원인 확인 예정 |

### C. 작은 소리·시끄러운 소리 (기준 대비)

| 세트 | 바꾼 것 | CER | 기준 | 빈 결과 |
|---|---|---|---|---|
| fleurs-quiet | `--gain rms` (VAD 뒤) | 13.28 | 13.36 | — |
| fleurs-quiet | `--gain rms --gain-scope file` (VAD 앞) | **6.02** | 13.36 | 9→0 |
| fleurs-quiet | `--vad-threshold 0.3` | 9.15 | 13.36 | |

- 소리가 작을 때 성능이 떨어지는 원인은 인식 모델이 아니라 VAD가 말소리를 놓치는 것이다. VAD 앞에서 키우면 깨끗한 소리 수준(6.16)으로 돌아온다.
- 단 `--gain-scope file`은 파일 전체 크기를 미리 알아야 해서 폰 실시간 녹음에 그대로 못 쓴다. 그래서 앞만 보고 크기를 따라가는 실시간 자동 크기 맞춤 `--gain agc`를 추가했다(다음 실험).

### 남은 측정

C의 babble5 2개, D(긴 녹음: `--max-speech 10/30`, `--min-silence 0.25/0.8`), E(4스레드 속도), SenseVoice fp32, Whisper small.

## 다음 할 일

1. `--gain agc`를 quiet·babble5·longform·dev에서 확인 (폰에서 쓸 수 있는 형태)
2. 좋았던 것 묶기: `--vad-threshold 0.3 --min-silence 0.8 --gain agc` 를 dev에서 확인한 뒤 test 세트 전체로 확인
3. Zipformer 한국어 결과가 비정상인 원인 확인 후 다시 측정
4. 숫자 변환(ITN) 동작 확인: 덤프에서 숫자가 들어간 문장만 비교

## 결론

(측정이 끝나면 적는다)
