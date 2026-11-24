"""효과음·배경음(#45): 섞기·덕킹, 콘티의 sfx·bgm 맞추기와 검사, 워커가 소리를 받아 섞기."""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from lolog_ai.sounds import (
    BGM_DB,
    BGM_DUCK_DB,
    BGM_EXAMPLES,
    DESC_PATTERN,
    MAX_SFX,
    SEARCH_LIMIT,
    SFX_EXAMPLES,
    SFX_OFFSET_MS,
    SFX_PEAK_DB,
    clean_desc,
    duck_envelope,
    is_desc,
    mix,
    pick,
)
from lolog_ai.video_storyboard import validate
from lolog_ai.worker.api import ApiError
from lolog_ai.worker.video import add_sounds
from lolog_ai.worker.video_gen import _schema, fix_sounds

SR = 24000


def db(x: float) -> float:
    return 20 * np.log10(max(abs(x), 1e-9))


def sb_two_scenes() -> dict:
    """장면 2개(0~3초, 3~6초), 대사는 1~2초에 하나."""
    return {"scenes": [
        {"index": 1, "startMs": 0, "endMs": 3000, "sfx": None,
         "lines": [{"startMs": 1000, "endMs": 2000, "spoken": True}]},
        {"index": 2, "startMs": 3000, "endMs": 6000, "sfx": None, "lines": []},
    ]}


def test_examples_follow_rules():
    import re

    # 프롬프트 예시가 규칙(검사·스키마)을 어기면 LLM이 따라 써서 지워진다. 프롬프트대로 3~10단어
    for t in [*SFX_EXAMPLES, *BGM_EXAMPLES]:
        assert is_desc(t) and re.fullmatch(DESC_PATTERN, t) and 3 <= len(t.split()) <= 10, t
        assert " and " not in f" {t} ", t  # 한 줄에 소리 하나
    assert all(t.endswith("music") or "music with" in t for t in BGM_EXAMPLES)  # 배경음은 음악
    assert not set(SFX_EXAMPLES) & set(BGM_EXAMPLES)


@pytest.mark.parametrize("raw, want", [
    ("a door slamming shut", "a door slamming shut"),
    ("  A Dog   Barking. ", "a dog barking"),            # 소문자, 공백 하나, 끝 문장 부호
    ("none", None), ("None", None), ("", None), (None, None), (3, None),
    ("ab", None),                                         # 너무 짧음
    ("a " + "very " * 20 + "long sound", None),           # 60자 넘음
    (" ".join(["loud"] * 13), None),                       # 단어가 너무 많음
    ("문이 쾅 닫힘", None), ("door slam 2 times", None),  # 한글·숫자
    ("calm music without drums", None), ("no music", None),  # 부정은 CLAP이 못 알아들음
    ("rain", "rain"),
    # 1호 시험(11-24)에서 LLM이 실제로 소리를 이어 쓴 모양: 첫 소리만 남긴다
    ("a phone vibrating on a desk, then a loud alarm blaring suddenly, a teacher sighs", "a phone vibrating on a desk"),
    ("tense suspense music with ticking, then upbeat cheerful pop music", "tense suspense music with ticking"),
    ("a heavy thud then a record scratch", "a heavy thud"),
    ("a dog barking and a cat meowing", "a dog barking"),
    ("rain followed by thunder", "rain"),
    ("a door slamming shut while people laugh", "a door slamming shut"),
    ("cups clinking and chatting", "cups clinking and chatting"),  # 앞이 짧으면 "and"를 둔다
    ("heavy rain falling on a roof and pavement wet splashing on a", "heavy rain falling on a roof"),  # 60자에서 끊김
    ("a loud wind howling over", "a loud wind howling"),          # 끊겨 남은 말
])
def test_clean_desc(raw, want):
    assert clean_desc(raw) == want


def test_sfx_lands_at_scene_start_and_is_quieter():
    sb = sb_two_scenes()
    out = mix(np.zeros(SR * 6, dtype=np.float32), SR, sb, sfx={2: np.ones(SR * 5, dtype=np.float32)})
    start = (3000 + SFX_OFFSET_MS) * SR // 1000
    assert np.abs(out[:start]).max() == 0
    assert abs(db(np.abs(out).max()) - SFX_PEAK_DB) < 0.1
    assert np.abs(out[start + int(2.5 * SR):]).max() == 0  # 2.5초에서 자름
    assert np.abs(out[start + int(2.4 * SR)]) > 0


def test_bgm_ducks_under_dialogue():
    sb = sb_two_scenes()
    n = SR * 6
    out = mix(np.zeros(n, dtype=np.float32), SR, sb, bgm=np.ones(SR * 2, dtype=np.float32))
    assert abs(db(out[int(1.5 * SR)]) - BGM_DUCK_DB) < 0.5   # 대사 한가운데
    assert abs(db(out[int(4.0 * SR)]) - BGM_DB) < 0.5         # 대사 없는 장면
    assert out[0] == 0 and abs(out[-1]) < 1e-3                # 페이드 인·아웃


def test_unspoken_lines_do_not_duck():
    sb = sb_two_scenes()
    sb["scenes"][0]["lines"][0]["spoken"] = False
    env = duck_envelope(sb, SR * 6, SR)
    assert abs(db(env[int(1.5 * SR)]) - BGM_DB) < 0.5


def test_mix_never_clips_and_keeps_dialogue():
    sb = sb_two_scenes()
    speech = np.full(SR * 6, 0.98, dtype=np.float32)
    out = mix(speech, SR, sb, sfx={1: np.ones(SR, dtype=np.float32)}, bgm=np.ones(SR, dtype=np.float32))
    assert np.abs(out).max() <= 0.99 + 1e-6
    assert len(out) == len(speech)
    assert np.array_equal(mix(speech, SR, sb), speech)


def test_pick_is_stable():
    items = [{"id": 1}, {"id": 2}, {"id": 3}]
    assert pick(items, 7) == pick(items, 7) == {"id": 2}
    assert pick([], 7) is None


# ---------- 콘티 ----------

def test_schema_has_sfx_and_bgm_as_descriptions():
    sc = _schema(["화자1"])
    scene = sc["properties"]["scenes"]["items"]
    assert "sfx" in scene["required"] and "bgm" in sc["required"]
    for field in (sc["properties"]["bgm"], scene["properties"]["sfx"]):
        assert field == {"type": "string", "pattern": DESC_PATTERN}  # 값 목록(enum)이 아니라 설명 글
        assert "enum" not in field


def test_fix_sounds_limits_and_cleans():
    raw = ["a heavy thud", "A heavy thud.", "none", "문 닫힘", "a cartoon pop", "a door slamming shut",
           "a bright bell ding", None]
    sb = {"scenes": [{"sfx": v} for v in raw]}
    fix_sounds(sb, "Calm acoustic guitar music")
    got = [s["sfx"] for s in sb["scenes"]]
    # 다듬은 뒤 연달아 같은 thud, "none", 규칙에 안 맞는 한글, 넘친 ding(4번째)은 지운다
    assert got == ["a heavy thud", None, None, None, "a cartoon pop", "a door slamming shut", None, None]
    assert sum(v is not None for v in got) <= MAX_SFX
    assert sb["bgm"] == "calm acoustic guitar music"
    for bad in ("none", "no music", "", None):
        fix_sounds(sb, bad)
        assert sb["bgm"] is None


def test_validate_checks_sound_descriptions():
    sb = {"version": 1, "scenes": [{"index": 1, "sfx": "door_slam"}], "bgm": "Jazz Music"}  # 옛 값·대문자
    errs = validate(sb)
    assert any("sfx" in e for e in errs) and any("bgm" in e for e in errs)
    for bgm in (None, "sad slow piano music"):
        ok = {"version": 1, "scenes": [{"index": 1, "sfx": "a cartoon pop"}, {"index": 2, "sfx": None}], "bgm": bgm}
        assert not any("sfx" in e or "bgm" in e for e in validate(ok))


# ---------- 워커 ----------

def ogg_bytes(sec: float, sr: int = 44100) -> bytes:
    buf = io.BytesIO()
    t = np.arange(int(sec * sr)) / sr
    sf.write(buf, (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), sr, format="OGG", subtype="VORBIS")
    return buf.getvalue()


class FakeApi:
    """search가 설명 글마다 정해 둔 소리 목록을 가까운 순으로 limit개 돌려준다."""

    def __init__(self, sounds: dict | None = None, status: int | None = None):
        self.sounds = sounds or {}
        self.status = status
        self.asked = []

    def search_sounds(self, kind, text, limit):
        self.asked.append((kind, text, limit))
        if self.status:
            raise ApiError(self.status, '{"code": "EMBEDDING_UNAVAILABLE"}' if self.status == 503 else "{}")
        return [{"id": i, "kind": kind, "durationMs": 1000, "distance": 0.2 + i / 10, "url": f"https://s/{text}/{i}"}
                for i in range(len(self.sounds.get(text, [])))][:limit]

    def fetch(self, url):
        t, i = url.split("/")[-2:]
        return self.sounds[t][int(i)]


def test_add_sounds_mixes_what_it_can():
    sb = sb_two_scenes()
    sb["scenes"][0]["sfx"], sb["scenes"][1]["sfx"] = "a heavy thud", "a cartoon pop"
    sb["bgm"] = "calm acoustic guitar music"
    api = FakeApi({"a heavy thud": [ogg_bytes(0.5)],
                   "calm acoustic guitar music": [ogg_bytes(2.0), ogg_bytes(1.0)]})  # pop은 맞는 소리 없음
    track = np.zeros(SR * 6, dtype=np.float32)
    out, used = add_sounds(sb, track, SR, api, seed=5)
    assert used == ["a heavy thud", "calm acoustic guitar music"]
    assert np.abs(out).max() > 0 and len(out) == len(track)
    assert {a[2] for a in api.asked} == {SEARCH_LIMIT}  # 후보 SEARCH_LIMIT개 중에서 고른다
    assert ("SFX", "a heavy thud", SEARCH_LIMIT) in api.asked and ("BGM", "calm acoustic guitar music", 3) in api.asked


@pytest.mark.parametrize("status", [404, 500, 503])  # search 없음, 서버 오류, 글 벡터 모델 없음
def test_add_sounds_without_api_keeps_track(status):
    sb = sb_two_scenes()
    sb["scenes"][0]["sfx"], sb["bgm"] = "a heavy thud", "calm acoustic guitar music"
    track = np.full(SR * 6, 0.1, dtype=np.float32)
    out, used = add_sounds(sb, track, SR, FakeApi(status=status), seed=1)
    assert used == [] and out is track


def test_add_sounds_asks_each_description_once():
    sb = {"scenes": [{"index": i, "startMs": (i - 1) * 1000, "endMs": i * 1000, "sfx": "a cartoon pop", "lines": []}
                     for i in (1, 2, 3)]}
    api = FakeApi({"a cartoon pop": [ogg_bytes(0.3)]})
    add_sounds(sb, np.zeros(SR * 3, dtype=np.float32), SR, api, seed=0)
    assert api.asked == [("SFX", "a cartoon pop", SEARCH_LIMIT)]


def test_add_sounds_same_video_same_sound():
    # 같은 videoId(seed)면 늘 같은 후보를 고르고, 다른 영상은 다른 후보도 고른다
    sb = sb_two_scenes()
    sb["scenes"][0]["sfx"] = "a dog barking"
    clips = [ogg_bytes(0.4 + i / 10) for i in range(3)]
    picked = []
    for seed in (0, 1, 2, 0):
        api = FakeApi({"a dog barking": clips})
        fetched = []
        api.fetch = lambda url, f=fetched: f.append(url) or clips[int(url.rsplit("/", 1)[1])]
        add_sounds(sb, np.zeros(SR * 6, dtype=np.float32), SR, api, seed=seed)
        picked.append(fetched[0])
    assert picked[0] == picked[3] and len(set(picked)) == 3
