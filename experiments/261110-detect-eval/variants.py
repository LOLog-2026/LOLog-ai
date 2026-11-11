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
    # 긴 입력 나누기(run_eval.RecLlm에서 처리): 시간 창마다 따로 찾고 합침. long2(24분)에서 가운데 재현율이 0.31까지 떨어짐
    # 창 15분이면 1시간 묶음은 5번, 8분이면 9번 부름. 겹침은 이야기 최대 길이(약 3분)보다 길게 둔다
    "chunk15": {"detect_chunk": {"min": 15, "overlap": 4}},
    "chunk8": {"detect_chunk": {"min": 8, "overlap": 3}},
    "kind+chunk15": {"detect_replace": [(OLD_KINDS, NEW_KINDS)], "detect_chunk": {"min": 15, "overlap": 4}},
    "kind+chunk8": {"detect_replace": [(OLD_KINDS, NEW_KINDS)], "detect_chunk": {"min": 8, "overlap": 3}},
}

# ---- 확인 단계(run_eval.RecLlm에서 처리): 찾은 후보마다 그 구간만 보여 주고 1~5점을 매긴다 ----
# 측정 때는 다 남기고(min_score 1) 점수만 기록한 뒤, 몇 점 이상 남길지는 sweep_verify.py로 고른다.
# 평가셋 문구는 쓰지 않고 일반 범주만 적는다
VERIFY_SYSTEM = """다음은 친구들 대화를 폰 음성 인식으로 받아 적은 기록 중 한 부분이다. 문장부호가 없고 글자가 자주 틀리니 소리가 비슷한 말로 짐작해서 읽는다.
이 부분이 나중에 4컷 만화 카드로 다시 보고 싶을 만한 "이야기"인지 판단한다.

이야기
- 누군가 겪었거나 벌어진 일이 있고, 웃기거나 어이없거나 오래 기억할 만한 지점이 있다
- 듣는 사람이 웃거나 놀라거나 되묻는 등 반응한다(웃음소리가 없어도 된다)

이야기가 아님
- 무엇을 먹을지, 언제 어디서 만날지, 무엇을 살지 정하는 대화
- 물건이나 정보를 묻고 답하는 대화, 지금 하는 일이나 할 일을 알리는 말, 안부, 날씨
- 맞장구나 감탄, 웃음만 있고 일어난 일이 없는 대화
- TV·유튜브·라디오처럼 틀어 둔 방송 소리(진행자 멘트, 광고, 노래 가사)

score: 1 확실히 이야기가 아님, 2 아마 아님, 3 애매함, 4 아마 이야기, 5 확실한 이야기"""

VERIFY_REASON = {"system": VERIFY_SYSTEM + "\nreason에 판단 이유를 한 문장(40자 이내)으로 먼저 쓰고 score를 매긴다", "reason": True}
VERIFY_SCORE = {"system": VERIFY_SYSTEM, "reason": False}

VARIANTS.update({
    "kc8+verify": {"detect_replace": [(OLD_KINDS, NEW_KINDS)], "detect_chunk": {"min": 8, "overlap": 3},
                   "verify": VERIFY_REASON},
    # 점수 토큰의 확률로 기대 점수를 받아 4점 안에서 더 잘게 거른다(출력 토큰은 그대로)
    # 방송 오탐(1시간당 10개)이 확인 점수로는 안 걸러져, 방송 소리인지를 점수와 따로 묻는다
    "kc8+verify-bc": {"detect_replace": [(OLD_KINDS, NEW_KINDS)], "detect_chunk": {"min": 8, "overlap": 3},
                      "verify": {"system": VERIFY_SYSTEM + """

broadcast: 이 부분이 사람들이 직접 나눈 대화가 아니라 틀어 둔 방송 소리(TV·유튜브·라디오)면 true, 직접 나눈 대화면 false.
방송의 단서: 진행자가 시청자나 게스트에게 하는 존댓말 멘트, 출연자·작품 소개, 코너 진행, 광고·제품 설명, 노래 가사, 여러 사람 말이 한 줄에 길게 이어짐""",
                                 "reason": False, "logprobs": True, "broadcast": True}},
    "kc8+verify-lp": {"detect_replace": [(OLD_KINDS, NEW_KINDS)], "detect_chunk": {"min": 8, "overlap": 3},
                      "verify": {**VERIFY_SCORE, "logprobs": True}},
    "kc8+verify-s": {"detect_replace": [(OLD_KINDS, NEW_KINDS)], "detect_chunk": {"min": 8, "overlap": 3},
                     "verify": VERIFY_SCORE},
})


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

# ---- 화자 나누기 출력 줄이기(run_eval.RecLlm._speakers_compact) ----
# 젯슨은 출력 약 11토큰/초라 카드마다 대사를 다시 쓰는 화자 나누기(카드당 약 500토큰)가 시간의 75~80%를 쓴다.
# 줄 번호와 화자만 받고 대사는 입력 글을 그대로 쓴다(오타 고치기는 지금도 거의 안 되고 있음)
SPEAKER_COMPACT = """다음은 대화 녹음을 글로 바꾼 기록 중 한 이야기 부분이다. 문맥으로 줄마다 누가 말했는지 나눈다.

- 줄 번호마다 말한 사람을 적는다. 대사는 다시 쓰지 않는다
- 한 줄에 여러 사람의 말이 섞여 있으면 말한 순서대로 그 줄 번호를 여러 번 적는다
- 화자는 등장 순서대로 화자1, 화자2…로 붙이고 최대 4명이다. 같은 사람은 끝까지 같은 이름을 쓴다
- 한두 글자뿐인 잡음 줄은 뺀다"""

VARIANTS.update({
    # 내용은 지금과 같고(대사 다시 씀) JSON만 문법으로 공백 없이. 품질은 그대로, 토큰만 줄이는 게 목표
    "spk-json": {"speaker": {"system": None, "mode": "json"}},
    "spk-notext": {"speaker": {"system": SPEAKER_COMPACT, "mode": "notext"}},
    "spk-pairs": {"speaker": {"system": SPEAKER_COMPACT + "\n\n출력은 [[줄 번호, 화자 번호], ...] 모양의 JSON 배열 하나다. 예: [[12,1],[13,2],[14,1]]",
                              "mode": "pairs"}},
})
