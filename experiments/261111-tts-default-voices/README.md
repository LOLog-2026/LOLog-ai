# 실험: 기본 화자 24명 TTS 목소리

- 날짜: 2026-11-11
- 담당: 민주
- 이슈: #27

## 목적

기본 화자 24명(`LOLog-docs/assets/default-characters`)에게 그림의 나이대·성별·인상에 맞는 한국어 목소리를 정한다.
앱 콘티 수정 화면의 프로필 미리 듣기 버튼(`<id>_preview.wav`)과 이후 영상 대사 읽기에 쓴다.

## 설정

- 목소리 설계: `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` (Apache-2.0, 한국어 지원). 화자별 영어 설명(`instruct`) + 한국어 기준 문장(`refText`)은 `voice_design.json`
- 클론: `Qwen/Qwen3-TTS-12Hz-0.6B-Base`. 고른 후보를 참조 음성으로 넣어 같은 목소리로 다른 문장을 읽는다 (젯슨에 올리기 가벼운 쪽)
- 미리 듣기 문장: "안녕하세요! 만나서 반가워요."

## 돌리는 법

```bash
pip install qwen-tts soundfile
python -I make_voices.py design out --n 3          # 24명 × 후보 3개, out/candidates/index.html 로 듣기
# picks.txt 에 `kid-f-01 1` 처럼 고른 후보를 적는다 (없으면 0번)
python -I make_voices.py finalize out --picks picks.txt
```

결과 `out/default-voices/`: `<id>.wav`(참조 음성), `<id>_preview.wav`(미리 듣기), `<id>_sample1·2.wav`, `voices.json`

## 결과

(서버 실행 뒤 채움)

## 결론과 다음 할 일

-
