# 실험: 컨텐츠포인트(이야기) 감지 성능 올리기

- 날짜: 2026-11-10 ~
- 담당: 민주
- 이슈: #20
- 브랜치: `claude/project-thread-3peq77`
- 대상 코드: `src/lolog_ai/worker/detect.py` (이야기 찾기 + 화자 나누기), LLM은 젯슨 1호 llama-server

## 목적

워커가 1시간 묶음에서 "카드로 만들 이야기"를 얼마나 맞게 찾는지 숫자로 재고, 프롬프트·온도·모델을
한 번에 하나씩 바꿔 가며 올린다. 지표는 평가 기준 문서 2절을 따른다.
지금까지 llm-bench는 속도·메모리·형식만 쟀고 정확도는 잰 적이 없다.

## 평가셋

이야기 구간이 표시된 한국어 대화 공개 데이터가 없어서 정답이 붙은 대화를 새로 썼다(실제 사람·녹음과 관계없음).
실제 친구 대화 녹음이 모이면 같은 형식(`<녹음>.json` + `<녹음>.gold.json`)으로 넣어 같은 채점기로 잰다.

| 파일 | 내용 |
|---|---|
| `evalset/stories.json` | 이야기 20개(웃김 9, 황당 6, 추억 5). 줄마다 화자, 펀치라인 줄 번호, 종류 |
| `evalset/filler.json` | 이야기가 아닌 대화 30묶음(약속·날씨·주문 등). 6묶음은 "ㅋㅋ", "하하"가 섞였지만 이야기는 아닌 함정 |

`make_evalset.py`가 둘을 섞어 앱이 보내는 DetectMessage 모양으로 만든다(`--seed`가 같으면 늘 같은 결과).

| 세트 | 녹음 | 정답 이야기 | 길이 | 용도 |
|---|---|---|---|---|
| `short-stt` | 5개 | 20 | 각 3~4분 | 주 지표. 짧은 입력 |
| `long-stt` | 1개 | 20 | 26분, 300구간 | 실제 1시간 묶음처럼 긴 입력. 방송 소리 25구간 × 2 섞임 |
| `short-clean`, `long-clean` | 위와 같음 | 20 | | 문장부호만 뺀 글. STT 오류가 얼마나 끌어내리는지 비교 |
| `nostory-stt` | 1개 | 0 | 7분 | 평범한 대화 오탐 |
| `tv` | 1개 | 0 | 37분 | 방송 소리 오탐 (`sample_tv_37min.json` 그대로) |

`-stt` 세트의 STT 흉내: 이어진 말 한 줄로 합치기(35%, 화자 섞임), 음절 약 7%에 받침 빠짐·비슷한 모음/자음·글자 빠짐,
띄어쓰기 흔들림, 짧은 잡음 줄. 실제 SenseVoice 결과(`sample_tv_37min.json`)처럼 보이게 맞췄다.

예) `short-stt/short1`
```
[5] 퇴근하고 너 피건해서 버스 타짜마자 졸았거든
[8] 또 치길래 눈 떴더니 기삭림이야 손님 종점이에요 이러시는거야 근데 던 웃긴 건 그 벙스가차고지로 들어가 있었 어
```

## 채점 (`run_eval.py`)

워커의 `detect()`를 그대로 불러 돌린다(프롬프트·`tidy`·형식 검사 모두 실서비스와 같음). `isLast=true`로 넣어 다음 묶음으로 넘기기는 끈다.

| 지표 | 재는 법 | 1차 목표 |
|---|---|---|
| F1 (P, R) | 카드와 정답 이야기 짝짓기: 구간 번호 IoU 0.3 이상이거나 정답 펀치라인 구간을 모두 품으면 맞음 | 0.6 |
| punch | 맞은 카드 중 펀치라인이 모두 안에 든 비율 | 90% |
| start / end | 맞은 카드의 시작·끝 시간 차이 중앙값(초) | 15초 |
| late | 시작이 3초 넘게 늦은(빌드업 빠짐) 비율 | |
| kind | 종류 일치 | 70% |
| spk | 한 사람만 말한 구간의 화자 정확도(화자 이름 바뀜 허용, 못 붙인 구간은 틀림) | 75% |
| fp / fp_per_h | 정답과 짝이 없는 카드 수, 1시간당 | 평범한 대화·방송 각 1개 이하 |
| fmt | JSON·길이 오류로 녹음이 통째로 실패 + 형식 검사로 버려진 카드 | 0 |
| time | 처리 시간 합(초) | 1시간 묶음 5분 |

LLM은 같은 입력에도 흔들리므로 설정마다 3번 돌려 평균(최소~최대)을 적는다. 차이가 범위 안이면 "차이 없음".
`--mock gold`(정답을 그대로 돌려주는 가짜 LLM)로 돌리면 모든 세트가 F1 1.0이 나오는 것을 확인했다.

## 젯슨 1호에서 돌리는 법

```bash
# 0) 코드 받기 (서비스용 /opt/lolog/LOLog-ai는 건드리지 않는다)
cd ~ && git clone https://github.com/LOLog-2026/LOLog-ai LOLog-ai-eval 2>/dev/null; cd ~/LOLog-ai-eval
git fetch origin && git checkout claude/project-thread-3peq77 && git pull
cd experiments/261110-detect-eval

# 1) 평가셋 만들기 (몇 초)
python3 make_evalset.py --out sets --tv ~/jetson-bench/llm-bench/sample_tv_37min.json
python3 run_eval.py --sets sets/short-stt --mock gold --runs 1 --tag selftest   # 모두 1이 나와야 함

# 2) 측정 중에는 감지 워커를 멈춘다(같은 LLM을 같이 쓰면 시간이 섞임). LLM 서버는 그대로 둔다
sudo systemctl stop lolog-worker@detect
export LLM_BASE_URL=http://$(sudo grep ^LLM_HOST /etc/lolog/llm.env | cut -d= -f2):8000/v1
export LLM_API_KEY=$(sudo grep ^LLM_API_KEY /etc/lolog/llm.env | cut -d= -f2)
curl -s ${LLM_BASE_URL%/v1}/health                      # {"status":"ok"}

# 3) 기준선 (tmux 안에서. 약 1~1.5시간)
python3 run_eval.py --tag base --runs 3 \
  --sets sets/short-stt sets/long-stt sets/short-clean sets/long-clean sets/nostory-stt sets/tv \
  2>&1 | tee results-base.log

# 4) 끝나면 워커 다시 켜기
sudo systemctl start lolog-worker@detect
```

마지막에 나오는 표와 `results/` 폴더를 보내 주면 아래에 옮긴다.
다른 설정은 `--variant <이름> --tag <이름>`만 바꿔 같은 명령으로 돌린다(`variants.py`).

## 결과

### A. 기준선 (Qwen3-4B-Instruct-2507 Q4_K_M, 온도 0.2, 지금 프롬프트)

(측정 전)

## 다음에 바꿔 볼 것 (한 번에 하나씩)

1. 온도 0.2 → 0 (`temp0`): 흔들림이 줄어드는지
2. 프롬프트: 펀치라인과 그 뒤 반응까지 넣으라는 규칙, "ㅋㅋ"만 있는 대화는 이야기가 아님 (기준선의 틀린 예를 보고 정함)
3. 프롬프트: 짧은 예시 한 개(few-shot)
4. 긴 입력 나누기: 1시간을 15분씩 겹쳐 잘라 따로 찾기 (long 세트가 short보다 많이 떨어질 때)
5. 모델: EXAONE 4.0 1.2B, Qwen3-8B (llm-bench README 후보)
