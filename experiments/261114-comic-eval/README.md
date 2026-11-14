# 실험: 4컷만화 그림 보정과 Qwen3.5 콘티 확인

- 날짜: 2026-11-14~
- 담당: 민주
- 이슈: #41
- 브랜치: `claude/project-thread-t2w0a0`
- 커밋: 결과 폴더의 `COMMIT` 파일

## 목적

1. 영상에서 사람 수를 65.8%까지 올린 보정(#33 4차: 컷에 없는 사람 문장 빼기, 옷으로 부른 인물 넣기, 성별·윗옷 색 고정, 글자 그리라는 말 빼기, 프롬프트 앞에 "exactly N people")을 만화 콘티·컷 프롬프트에도 넣었다. 만화 그림이 실제로 나아졌는지 같은 카드 12개로 본다
2. 감지 모델이 Qwen3-4B → Qwen3.5-4B로 바뀌었는데 같은 서버를 쓰는 만화·영상 콘티는 다시 재지 않았다. 한 번에 통과하는지, 몇 번 다시 묻는지, 시간이 얼마인지 본다

## 설정

- 카드: 감지 평가셋 이야기 12개(영상 개발용 devD와 같은 L04 L07 L14 L18 L28 L38 s09 s22 s33 s47 s67 s96). 실제 녹음이 아니라 평가셋 대본이다
- 참조 그림: s09 young-f-02, s33 middle-m-01, L28 young-m-01 (`--chars`로 기본 화자 그림 폴더를 주면 붙임)
- 콘티 LLM: 1호 llama-server(Qwen3.5-4B Q4_K_M). 그림: 2호 FLUX.2 klein nf4, seed 33
- 시작할 때 `src`를 결과 폴더에 복사해 그 코드로 돌린다. 반영 전과 비교하려면 반영 전 커밋을 체크아웃해 한 번 더 돌린다

```bash
export LLM_BASE_URL=http://<1호>:8000/v1 LLM_API_KEY=<키> LLM_MODEL=local
# 4번: 콘티만 (어디서나)
python3 run_comic.py plan out/q35-plan --video
# 2번: 2호에서 comic 워커를 멈추고
sudo systemctl stop lolog-worker@comic
python3 run_comic.py draw out/after --chars <default-characters 폴더>
git checkout <반영 전 커밋> -- ../../src && python3 run_comic.py draw out/before --chars <같은 폴더>; git checkout HEAD -- ../../src
sudo systemctl start lolog-worker@comic
```

결과: `out/<이름>/results.jsonl`(카드마다 통과 여부, 다시 물은 횟수, 시간), 카드 폴더마다 `card.json` `storyboard.json` `prompts.txt` `comic.png`. `out/`은 git에 올리지 않는다.

## 눈으로 볼 것 (컷마다, 48컷)

| 항목 | 맞음의 기준 |
|---|---|
| 사람 수 | 컷에 나오는 사람 수가 콘티 cast 수와 같다(같은 사람이 두 번 나오지 않음) |
| 글자 | 간판·쪽지·화면에 글자처럼 보이는 것이 없다 |
| 옷 | 같은 인물의 윗옷 색·머리가 4컷 내내 같다 |
| 말풍선 가림 | 말풍선이 얼굴을 가리지 않는다 |
| 말한 사람 | 말풍선 꼬리가 말한 사람을 가리킨다 |

## 결과

(측정 뒤 채움)

## 결론과 다음 할 일

-
