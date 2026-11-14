# 실험 기록

실험 하나에 폴더 하나를 만듭니다. 폴더 이름은 `YYMMDD-짧은-설명`입니다(예: `261105-klue-roberta-baseline`).

## 새 실험 시작하기

1. 이슈를 만듭니다: `[Feature] ○○ 모델 파인튜닝 실험` (레이블 `ai`)
2. `develop`에서 브랜치를 만듭니다: `feature/<이슈번호>-exp-<설명>`
3. `_template/`을 복사해 실험 폴더를 만듭니다.
   ```bash
   cp -r experiments/_template experiments/261105-klue-roberta-baseline
   ```
4. 실험하면서 실험 폴더의 `README.md`를 채우고 커밋합니다.
5. 실험이 끝나면(실패해도) 결론을 적고 `develop`으로 PR을 올립니다.

## 실험 폴더에 두는 것

- `README.md`: 목적, 설정, 결과, 결론
- `config.yaml`: 그대로 다시 돌릴 수 있는 설정
- 작은 결과 파일(지표 csv, 그래프 png 등)

모델 가중치·체크포인트는 git에 올리지 않습니다. 저장 위치(Hugging Face Hub, 공유 드라이브 등)를 실험 README에 적습니다.

## 실험 목록

| 날짜 | 실험 | 모델 | 핵심 지표 | 결론 | 이슈 |
|---|---|---|---|---|---|
| 2026-11-10~ | [261110-detect-eval](261110-detect-eval/) | Qwen3-4B → Qwen3.5-4B Q4_K_M (llama.cpp) | 이야기 F1, 펀치라인 포함률 | 젯슨 모델 비교로 Qwen3.5-4B 채택(short F1 0.844, long2 0.69) | #20, #36 |
| 2026-11-11~12 | [261111-video-tts](261111-video-tts/) 카드 영상: 대사 TTS, 영상 만들기 | Qwen3-TTS 0.6B Base, Qwen3-4B, FLUX.2 klein 4B | TTS 읽기 94.2%·성별 95.8%(Jetson 3호 rtf 5.35), 영상 12/12 완성·사람 수 65.8% | TTS는 0.6B로 확정. 영상은 시연 가능, 그림 사람 수 보정 필요 | #33 |
| 2026-11-13~14 | [261113-video-lipsync](261113-video-lipsync/) 카드 영상 립싱크: 입모양, 눈 깜빡임, 고개 움직임 | FLUX.2 klein 4B(얼굴 384 편집), MediaPipe 얼굴 점, Whisper small(채점용) | 처음 보는 카드 8개 립싱크 적용률 47~61% → 96%, 낱말 시각 80ms 안 70% → 80% | FLUX 입·눈 붙이기 방식으로 확정, 2호에 반영(11-14). 수염 난 얼굴이 남음 | #38 |
| 2026-11-14~ | [261114-comic-eval](261114-comic-eval/) 만화 그림 보정, Qwen3.5 콘티 | Qwen3.5-4B, FLUX.2 klein 4B | 컷 사람 수·글자 없음, 콘티 한 번에 통과율 | 측정 전 | #41 |
