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

(서버 실행 뒤 채움)

## 결론과 다음 할 일

-
