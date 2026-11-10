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
| 2026-11-10~ | [261110-detect-eval](261110-detect-eval/) | Qwen3-4B Q4_K_M (llama.cpp) | 이야기 F1, 펀치라인 포함률 | 기준선 측정 전 | #20 |
