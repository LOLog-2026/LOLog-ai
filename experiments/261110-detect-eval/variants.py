"""한 번에 하나씩 바꿀 설정. base가 지금 워커(detect.py) 그대로다.

키
  temperature     detect·speakers 호출 온도 (워커 기본 0.2)
  detect_system   DETECT_SYSTEM 전체를 바꿈
  detect_append   DETECT_SYSTEM 끝에 규칙을 덧붙임
  detect_replace  [(원래 문장, 새 문장), ...] DETECT_SYSTEM의 규칙 한 줄만 바꿈(없으면 오류)
  speaker_system  SPEAKER_SYSTEM 전체를 바꿈
새 실험은 여기에 이름을 하나 더하고 README 표에 줄을 더한다.
"""

from __future__ import annotations

OLD_STORY_RULE = "- 평범한 안부, 일정 얘기, 짧은 맞장구만 있는 부분은 이야기가 아니다. 찾을 게 없으면 빈 목록을 돌려준다"

# 이야기 판단 기준(오탐 줄이기). 요구사항 FR-CARD-01 "웃음소리가 없어도 이야기를 찾는다"에 맞춰 웃음은 조건으로 넣지 않는다.
# 평가셋 문구는 쓰지 않고 일반적인 범주만 적는다
NEW_STORY_RULE = """- 이야기는 누군가 겪었거나 벌어진 일(사건)이 있고, 그 일에 웃음·놀람·되묻기·놀림처럼 맞장구보다 큰 반응이 따르는 부분이다. 웃음소리가 없어도 사건과 반응이 있으면 이야기다
- 다음은 이야기가 아니다: 무엇을 먹을지·언제 어디서 만날지·무엇을 살지 정하는 대화, 물건이나 정보를 묻고 답하는 대화, 지금 하는 일이나 할 일을 알리는 말, 안부·날씨, 짧은 맞장구. 웃음소리(ㅋㅋ, 하하)나 감탄이 섞여 있어도 사건이 없으면 이야기가 아니다
- 찾을 게 없으면 빈 목록을 돌려준다"""

OLD_KINDS = """이야기 종류
- 웃김: 농담, 장난, 다 같이 웃을 만한 일
- 황당: 어이없거나 뜻밖인 일
- 추억: 지난 일을 떠올리며 나눈 얘기"""

# 종류 정의가 겹쳐(웃김 "웃을 만한 일" vs 황당 "뜻밖인 일") 웃김↔황당 혼동이 큼. 고르는 순서와 기준을 나눈다
NEW_KINDS = """이야기 종류 (하나만 고른다. 위에서부터 맞는 것을 고른다)
- 추억: 몇 년 전 일(어린 시절, 학창 시절, 군대, 예전에 함께 한 일)을 함께 떠올리는 이야기. 웃기거나 뜻밖이어도 오래전 일을 돌아보는 게 중심이면 추억
- 황당: 최근에 겪은 일이 예상과 다르게 흘러가 어이없거나 당황스러운 이야기. 듣는 사람 반응이 "말이 돼", "어이없다" 쪽이면 황당
- 웃김: 최근 일 중에서 누군가의 말·행동·실수가 우스워서 다 같이 웃는 이야기. 반응이 웃음 쪽이면 웃김"""

OLD_BOUND = "- from, to는 이야기의 첫 줄과 마지막 줄 번호다. 웃음 포인트 앞의 상황 설명(빌드업)부터 넣는다"

# 맞은 카드의 36~42%가 정답보다 2줄쯤 늦게 시작함(빌드업 잘림)
NEW_BOUND = ("- from, to는 이야기의 첫 줄과 마지막 줄 번호다. from은 누군가 그 이야기를 꺼내는 줄이다. 웃음 포인트를 찾았으면 "
             "앞으로 거슬러 올라가 그 화제가 시작된 줄을 from으로 한다. 상황 설명(빌드업)을 빠뜨리지 않는다\n"
             "- to는 웃음 포인트 뒤에 이어지는 반응과 뒷얘기가 끝나는 줄이다")

OLD_MULTI = "- 한 녹음에 이야기가 여럿이면 따로 나눈다"

# 긴 입력(20분 넘음)에서 정답 25개에 카드 10~15개만 냄. 놓침은 앞·중간·뒤 고르게 퍼짐
NEW_MULTI = ("- 한 녹음에 이야기가 여럿이면 따로 나눈다. 긴 녹음에는 이야기가 수십 개 있을 수 있다. "
             "처음부터 끝까지 순서대로 훑으며 하나도 빠뜨리지 않고 모두 찾는다")

VARIANTS: dict[str, dict] = {
    "base": {},
    "temp0": {"temperature": 0.0},
    "p-story": {"detect_replace": [(OLD_STORY_RULE, NEW_STORY_RULE)]},
    "p-kind": {"detect_replace": [(OLD_KINDS, NEW_KINDS)]},
    "p-bound": {"detect_replace": [(OLD_BOUND, NEW_BOUND)]},
    "p-all": {"detect_replace": [(OLD_MULTI, NEW_MULTI)]},
    # 쌓기: 3번 확인에서 이긴 p-kind 위에 하나씩 얹는다
    "kind+all": {"detect_replace": [(OLD_KINDS, NEW_KINDS), (OLD_MULTI, NEW_MULTI)]},
}


def apply(det_module, v: dict) -> None:
    for old, new in v.get("detect_replace", []):
        if old not in det_module.DETECT_SYSTEM:
            raise ValueError(f"DETECT_SYSTEM에 바꿀 문장이 없음: {old[:40]}")
        det_module.DETECT_SYSTEM = det_module.DETECT_SYSTEM.replace(old, new)
    if "detect_system" in v:
        det_module.DETECT_SYSTEM = v["detect_system"]
    if "detect_append" in v:
        det_module.DETECT_SYSTEM = det_module.DETECT_SYSTEM + "\n" + v["detect_append"]
    if "speaker_system" in v:
        det_module.SPEAKER_SYSTEM = v["speaker_system"]
