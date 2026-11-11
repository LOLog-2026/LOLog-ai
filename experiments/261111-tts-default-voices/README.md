# 실험: 기본 화자 24명 TTS 목소리

- 날짜: 2026-11-11
- 담당: 민주
- 이슈: #27

## 목적

기본 화자 24명(`LOLog-docs/assets/default-characters`)에게 그림의 나이대·성별·인상에 맞는 한국어 목소리를 정한다.
모든 화자가 같은 인사 "안녕하세요! 만나서 반가워요."를 읽고 목소리만 다르다.
앱 콘티 수정 화면에서 프로필 옆 미리 듣기 버튼으로 재생한다.

## 설정

- 모델: `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` (Apache-2.0, 한국어 지원). 글로 쓴 목소리 설명대로 새 목소리를 만든다
- 목소리 설명: 24명 그림을 보고 쓴 영어 `instruct` (`voice_design.json`)
- 화자당 후보 3개(시드만 다름) 중 1개를 귀로 고른다
- 나중에 영상 대사를 같은 목소리로 읽힐 때는 고른 `<id>.wav`를 참조 음성으로 Qwen3-TTS Base(클론) 모델에 넣는다

## 돌리는 법

```bash
pip install qwen-tts soundfile
python -I make_voices.py design out --n 3          # 24명 × 후보 3개, out/candidates/index.html 로 듣기
# picks.txt 에 `kid-f-01 1` 처럼 고른 후보를 적는다 (없으면 0번)
python -I make_voices.py finalize out --picks picks.txt
```

결과 `out/default-voices/`: `<id>.wav` 24개(24kHz 모노 16bit), `voices.json`(id, file, seconds, instruct, seed)

## 결과

2026-11-11, 연구실 서버(RTX 3090, torch 2.15.0+cu132, qwen-tts 0.1.1, sdpa)에서 돌렸다.

- 후보 72개를 423초에 만들었다. rtf 평균 1.83(1.58~2.68), GPU 메모리 약 5.3GB
- 후보 길이 평균 3.27초. 5초를 넘은 후보 4개는 같은 시드로 다시 만들어도 똑같이 나왔다

| 후보 | 길이 | 내용 |
|---|---|---|
| `kid-f-01_c1` | 20.50초 | 인사 뒤에 뜻 없는 말이 이어짐 |
| `young-f-04_c2` | 9.06초 | 받아쓰기는 정상 |
| `middle-m-01_c2` | 5.55초 | 앞에 웃음소리가 들어감 |
| `middle-m-04_c2` | 5.00초 | 받아쓰기는 정상 |

- 고르기(`picks.txt`): 귀로 듣지 않고 Whisper(large-v3-turbo) 받아쓰기와 음높이 중앙값(pyin)으로 골랐다. 0번 후보에 문제가 있을 때만 1번으로 바꿨다

| 화자 | 선택 | 바꾼 이유 |
|---|---|---|
| `middle-m-01` | 1 | 0번에 군소리("아")가 들어감 |
| `kid-m-04` | 1 | 0번 168Hz로 아이치고 낮음 (1번 263Hz) |
| `middle-m-04` | 1 | 0번 237Hz로 높음 (1번 135Hz) |
| `young-m-04` | 1 | 0번 248Hz로 높음 (1번 150Hz) |
| `young-m-03` | 1 | 0번 435Hz로 높음 (1번 355Hz) |
| `middle-f-04` | 1 | 0번 322Hz로 높음 (1번 250Hz) |

- 고른 24개는 모두 "안녕하세요 만나서 반가워요"로 받아쓰기가 맞았다. 길이 평균 2.76초(1.92~3.44초), 합계 3.1MB
- wav는 이 레포에 올리지 않는다. `picks.txt`와 같은 시드로 다시 만들 수 있다 (GPU·torch 버전이 다르면 달라질 수 있다)

## 결론과 다음 할 일

- 24명 목소리를 측정값으로 정했다. 사람이 24개를 들어 보고 어색하면 `picks.txt`를 고쳐 `finalize`를 다시 돌린다
- 소리치는 목소리는 음높이 측정이 부정확할 수 있다. `middle-m-03`(세 후보 모두 368~415Hz)과 `young-m-03`은 꼭 들어 본다
- `python3 -m venv`가 안 되는 서버에서는 `uv venv && uv pip install qwen-tts soundfile`을 쓴다
