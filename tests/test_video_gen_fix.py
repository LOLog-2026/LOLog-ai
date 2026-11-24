"""영상 콘티 보정 함수(#72): LLM이 쓴 영상 콘티를 코드가 고치는 함수들의 지금 동작.

assemble()이 콘티마다 부르는 함수들이다(clean_scene, _extras, fix_appearance, _fix_flashback, fix_scene,
add_dressed, drop_phantoms, trim_lines). fix_sounds는 test_sounds.py에 있다.
xfail(strict)은 알려진 버그다. 고치면 통과로 바뀌어 테스트가 실패하니, 그때 xfail을 지운다.
"""

from __future__ import annotations

import pytest

from lolog_ai.video_storyboard import MAX_LINES, MAX_TEXT, MIN_SCENES
from lolog_ai.worker.video_gen import (
    TARGET_MAX_S,
    TOP_COLORS,
    _extras,
    _fix_flashback,
    _head,
    add_dressed,
    clean_scene,
    drop_phantoms,
    est_seconds,
    fix_appearance,
    fix_scene,
    split_line,
    trim_lines,
)

CHARS = [
    {"key": "화자1", "appearance": "young woman, long hair, navy hoodie", "role": "teller"},
    {"key": "화자2", "appearance": "young man, short hair, purple cardigan", "role": "listener"},
]
EXTRAS = [
    {"id": "grandma", "appearance": "an old woman, white hair"},
    {"id": "driver", "appearance": "middle-aged man, cap"},
    {"id": "kid", "appearance": "a 7-year-old boy"},
]
SB = {"characters": CHARS, "extras": EXTRAS}
HALF = "가" * 20  # 둘을 띄어 이으면 41자: MAX_TEXT(40)를 넘고, 띄어쓰기에서 나누면 20자씩


def line(text: str, speaker: str = "화자1") -> dict:
    return {"speaker": speaker, "text": text}


def scene(text: str = "", cast=(), extras=(), flashback: bool = False, lines=()) -> dict:
    return {"scene": text, "flashback": flashback,
            "cast": [{"key": k, "position": p} for k, p in cast],
            "extras": [{"id": i, "position": p} for i, p in extras],
            "lines": list(lines)}


# ---------- clean_scene: 그림에 글자·말풍선이 생기는 말 지우기 ----------

@pytest.mark.parametrize("text, want", [
    ("The woman (Jiyeon) laughs.", "The woman laughs."),                              # 괄호(이름)
    ('The man says "hello" to the woman.', "The man."),                               # 따옴표 말 + 말하는 동사
    ("The man shouts loudly to the woman.", "The man."),                              # 말하는 동사 뒤까지
    ("A cafe with a large menu board, the woman sits.", "A cafe, the woman sits."),   # 간판·메뉴판
    ("The woman looks at her phone showing text messages.", "The woman looks at her phone."),  # 채팅 화면
    ("The man holds a note that reads hello.", "The man holds a note."),             # 글자를 그리라는 말
    ("The man opens the door, then the dog runs out.", "The man opens the door."),   # then 뒤 다음 순간
    ("The man's bag falls. Later, the woman arrives.", "The man's bag falls."),      # 장면이 바뀌는 문장
    ("The boy smiles.", "The boy smiles."),                                           # 고칠 것 없음
])
def test_clean_scene(text, want):
    assert clean_scene(text) == want


# ---------- split_line: 자막 한도를 넘는 대사 나누기 ----------

def test_split_line_at_space_nearest_middle():
    d = {**line(HALF + " " + HALF), "sourceLineIds": [3]}
    out = split_line(d, MAX_TEXT)
    assert [x["text"] for x in out] == [HALF, HALF]
    assert all(x["speaker"] == "화자1" and x["sourceLineIds"] == [3] for x in out)  # 다른 칸은 그대로


@pytest.mark.parametrize("text", [
    "짧은 말",          # 한도 안
    "가" * 45,          # 띄어쓰기 없음
    "가 " + "나" * 44,  # 나눠도 한쪽이 한도를 넘음
])
def test_split_line_keeps_when_it_cannot(text):
    d = line(text)
    assert split_line(d, MAX_TEXT) == [d]


# ---------- fix_scene: 긴 대사 나누기, 말하는 화자 넣기, 자리 겹침 풀기 ----------

def test_fix_scene_adds_speaker_missing_from_cast():
    sc = scene(cast=[("화자1", "left")], lines=[line("안녕", "화자2")])
    fix_scene(sc, MAX_TEXT, MAX_LINES)
    assert sc["cast"] == [{"key": "화자1", "position": "left"}, {"key": "화자2", "position": "center"}]


def test_fix_scene_swaps_extra_for_speaker_when_full():
    sc = scene(cast=[("화자1", "left"), ("화자2", "right")], extras=[("grandma", "center")],
               lines=[line("나도", "화자3")])
    fix_scene(sc, MAX_TEXT, MAX_LINES)
    assert sc["extras"] == []
    assert [c["key"] for c in sc["cast"]] == ["화자1", "화자2", "화자3"]


def test_fix_scene_flashback_keeps_cast():
    # 회상 장면의 대사는 들려주는 사람의 목소리라 cast에 없어도 된다
    sc = scene(flashback=True, lines=[line("옛날에")])
    fix_scene(sc, MAX_TEXT, MAX_LINES)
    assert sc["cast"] == []


def test_fix_scene_reassigns_overlapping_positions():
    sc = scene(cast=[("화자1", "left"), ("화자2", "left")])
    fix_scene(sc, MAX_TEXT, MAX_LINES)
    assert [c["position"] for c in sc["cast"]] == ["left", "right"]


def test_fix_scene_splits_long_line_when_there_is_room():
    sc = scene(cast=[("화자1", "left")], lines=[line(HALF + " " + HALF), line("짧")])
    fix_scene(sc, MAX_TEXT, MAX_LINES)
    assert [d["text"] for d in sc["lines"]] == [HALF, HALF, "짧"]


@pytest.mark.parametrize("texts", [
    [HALF + " " + HALF, "둘째", "마지막"],              # 첫 대사가 길고 자리가 없음(#74: 마지막 대사가 사라졌다)
    ["첫째", HALF + " " + HALF, "마지막"],
    [HALF + " " + HALF, HALF + " " + HALF, "마지막"],
])
def test_fix_scene_never_drops_a_line(texts):
    # 자리가 없으면 나누지 않고 그대로 둔다. 긴 대사는 검사에 걸려 LLM에 다시 묻는다
    sc = scene(cast=[("화자1", "left")], lines=[line(t) for t in texts])
    fix_scene(sc, MAX_TEXT, MAX_LINES)
    assert [d["text"] for d in sc["lines"]] == texts


def test_fix_scene_splits_only_up_to_max_lines():
    sc = scene(cast=[("화자1", "left")], lines=[line(HALF + " " + HALF), line(HALF + " " + HALF)])
    fix_scene(sc, MAX_TEXT, MAX_LINES)
    assert [d["text"] for d in sc["lines"]] == [HALF, HALF, HALF + " " + HALF]  # 첫 대사만 나눌 자리가 있음


def test_fix_scene_keeps_lines_over_max_as_is():
    # LLM이 MAX_LINES보다 많이 쓰면(스키마가 막지만) 나누지도 자르지도 않는다. 검사가 잡는다
    texts = ["하나", "둘", "셋", HALF + " " + HALF]
    sc = scene(cast=[("화자1", "left")], lines=[line(t) for t in texts])
    fix_scene(sc, MAX_TEXT, MAX_LINES)
    assert [d["text"] for d in sc["lines"]] == texts


# ---------- est_seconds·trim_lines: 영상이 길면 가운데 장면 대사 빼기 ----------

def test_est_seconds_counts_chars_without_spaces_and_lines():
    assert est_seconds({"scenes": [{"lines": []}]}) == pytest.approx(2.9)
    assert est_seconds({"scenes": [{"lines": [line("가 나")]}]}) == pytest.approx(2.9 + 0.109 * 2 + 1.44)


def video(n_scenes: int, lines_per: int, text: str = "가나다라마바사아자차") -> dict:
    return {"scenes": [{"index": i + 1, "transition": "cut", "lines": [line(text) for _ in range(lines_per)]}
                       for i in range(n_scenes)]}


def test_trim_lines_keeps_short_video():
    sb = video(4, 1)
    trim_lines(sb)
    assert [len(s["lines"]) for s in sb["scenes"]] == [1, 1, 1, 1]
    assert sb["scenes"][0]["transition"] == "fade"  # 첫 장면 전환은 늘 fade


def test_trim_lines_cuts_middle_only():
    sb = video(6, 3)
    assert est_seconds(sb) > TARGET_MAX_S
    trim_lines(sb)
    assert est_seconds(sb) <= TARGET_MAX_S
    n = [len(s["lines"]) for s in sb["scenes"]]
    assert n[0] == 3 and n[-2:] == [3, 3]  # 상황 소개와 웃음 포인트·반응은 그대로
    assert [s["index"] for s in sb["scenes"]] == list(range(1, len(sb["scenes"]) + 1))


def test_trim_lines_removes_empty_scenes_down_to_minimum():
    sb = video(8, 3, "가" * 20)
    trim_lines(sb)
    assert len(sb["scenes"]) == MIN_SCENES
    assert est_seconds(sb) <= TARGET_MAX_S


def test_trim_lines_stops_when_nothing_left_to_cut():
    # 장면이 최소 개수이고 가운데 대사를 다 빼도 길면, 빈 장면을 남기고 멈춘다(무한 반복하지 않음)
    sb = video(MIN_SCENES, 3, "가" * 30)
    trim_lines(sb)
    assert [len(s["lines"]) for s in sb["scenes"]] == [3, 0, 3, 3]
    assert est_seconds(sb) > TARGET_MAX_S


# ---------- fix_appearance: 장면마다 같은 사람으로 그려지게 ----------

@pytest.mark.parametrize("c, i, want", [
    ({"appearance": "short black hair, hoodie", "gender": "m", "age": "young"}, 0,
     f"young man, short black hair, {TOP_COLORS[0]} hoodie"),                       # 성별 말과 윗옷 색을 붙임
    ({"appearance": "woman, long hair, red hoodie", "gender": "f"}, 1,
     "woman, long hair, red hoodie"),                                                # 이미 있으면 그대로
    ({"appearance": "woman, blue jeans and a hoodie", "gender": "f"}, 2,
     f"woman, blue jeans and a {TOP_COLORS[2]} hoodie"),                             # 바지 색은 윗옷 색이 아님
    ({"appearance": "boy, casual clothes", "gender": "m", "age": "kid"}, 3,
     f"boy, wearing a {TOP_COLORS[3]} sweatshirt"),                                  # 윗옷이 없으면 더함
    ({"appearance": "curly hair", "gender": "f", "age": "middle"}, 6,
     f"middle-aged woman, curly hair, wearing a {TOP_COLORS[0]} sweatshirt"),       # 색은 i로 돌아가며
    ({"appearance": "curly hair, shirt"}, 0, f"curly hair, {TOP_COLORS[0]} shirt"),  # 성별을 모르면 안 붙임
])
def test_fix_appearance(c, i, want):
    assert fix_appearance(c, i) == want


# ---------- _extras: 장면 extras 정리 ----------

def test_extras_drops_undefined_and_taken_positions():
    p = {"extras": [{"id": "grandma", "position": "left", "expression": ""},
                    {"id": "ghost", "position": "right"},                      # 정의 안 됨
                    {"id": "driver", "position": "center", "expression": "angry"}]}  # 화자와 자리 겹침
    assert _extras(p, EXTRAS, [{"key": "화자1", "position": "center"}]) == [{"id": "grandma", "position": "left"}]


def test_extras_keeps_three_people_per_scene():
    p = {"extras": [{"id": "grandma", "position": "left"}, {"id": "driver", "position": "right"}]}
    cast = [{"key": "화자1", "position": "center"}, {"key": "화자2", "position": "center"}]
    assert _extras(p, EXTRAS, cast) == [{"id": "grandma", "position": "left"}]


# ---------- _fix_flashback: 장면 글에만 나오는 이야기 속 인물 넣기 ----------

def test_flashback_replaces_unmentioned_speaker_with_story_person():
    sc = scene("An old woman cooks in the kitchen.", cast=[("화자1", "center")], flashback=True)
    _fix_flashback(sc, SB)
    assert sc["cast"] == []
    assert sc["extras"] == [{"id": "grandma", "position": "center"}]


def test_flashback_keeps_speaker_mentioned_in_scene():
    sc = scene("An old woman scolds the young woman.", cast=[("화자1", "center")], flashback=True)
    _fix_flashback(sc, SB)
    assert sc["cast"] == [{"key": "화자1", "position": "center"}]
    assert sc["extras"] == [{"id": "grandma", "position": "left"}]


def test_flashback_role_id_adds_person_and_keeps_speaker():
    sc = scene("The driver honks at the car.", cast=[("화자1", "center")], flashback=True)
    _fix_flashback(sc, SB)
    assert sc["cast"] == [{"key": "화자1", "position": "center"}]
    assert sc["extras"] == [{"id": "driver", "position": "left"}]


def test_present_scene_never_drops_speaker():
    sc = scene("An old woman cooks.", cast=[("화자1", "center")])
    _fix_flashback(sc, SB, drop_cast=False)
    assert sc["cast"] == [{"key": "화자1", "position": "center"}]
    assert sc["extras"] == [{"id": "grandma", "position": "left"}]


def test_flashback_skips_person_already_in_scene():
    sc = scene("An old woman cooks.", cast=[("화자1", "center")], extras=[("grandma", "left")], flashback=True)
    _fix_flashback(sc, SB)
    assert sc["cast"] == [{"key": "화자1", "position": "center"}]
    assert sc["extras"] == [{"id": "grandma", "position": "left"}]


@pytest.mark.parametrize("appearance, want", [
    ("an old woman, white hair, wearing a cardigan", "old woman"),
    ("young man with glasses", "young man"),
])
def test_head(appearance, want):
    assert _head(appearance) == want


@pytest.mark.xfail(strict=True, reason='#72 버그: "a 7-year-old boy"의 머리말이 "year-old boy"가 되어 "The boy"로 부르면 못 찾음')
def test_flashback_finds_person_with_age_in_appearance():
    sc = scene("The boy runs to the park.", cast=[("화자1", "center")], flashback=True)
    _fix_flashback(sc, SB)
    assert {"id": "kid", "position": "center"} in sc["extras"]


# ---------- drop_phantoms: 장면에 없는 사람을 부르는 문장 빼기 ----------

@pytest.mark.parametrize("text, cast, want", [
    ("The young woman sits. A man waves from the door.", [("화자1", "center")], "The young woman sits."),
    ("A man waves at her.", [("화자1", "center")], "A person waves at her."),            # 다 지우면 person으로
    ("The young woman talks to a cashier.", [("화자1", "center")], "The young woman talks to a person."),
    ("A crowd of men and women.", [("화자1", "center")], "A crowd of men and women."),   # 군중은 보지 않음
    ("The woman and the man laugh.", [("화자1", "left"), ("화자2", "right")], "The woman and the man laugh."),
])
def test_drop_phantoms(text, cast, want):
    sc = scene(text, cast=cast)
    drop_phantoms(sc, SB)
    assert sc["scene"] == want


# ---------- add_dressed: 옷으로 부른 인물 넣기 ----------

def test_add_dressed_adds_owner_of_outfit():
    sc = scene("The woman in a navy hoodie laughs at the man in a purple cardigan.", cast=[("화자1", "center")])
    add_dressed(sc, SB)
    assert sc["cast"] == [{"key": "화자1", "position": "center"}, {"key": "화자2", "position": "left"}]


def test_add_dressed_adds_story_person_to_extras():
    sc = scene("The man in a cap drives.", cast=[("화자1", "center")], flashback=True)
    add_dressed(sc, SB)
    assert sc["extras"] == [{"id": "driver", "position": "left"}]


def test_add_dressed_skips_listener_in_flashback():
    sc = scene("The man in a purple cardigan nods.", cast=[("화자1", "center")], flashback=True)
    add_dressed(sc, SB)
    assert sc["cast"] == [{"key": "화자1", "position": "center"}]


def test_add_dressed_skips_when_outfit_is_shared():
    sb = {"characters": CHARS + [{"key": "화자3", "appearance": "young man, purple cardigan", "role": "teller"}],
          "extras": []}
    sc = scene("The man in a purple cardigan nods.", cast=[("화자1", "center")])
    add_dressed(sc, sb)
    assert sc["cast"] == [{"key": "화자1", "position": "center"}]
