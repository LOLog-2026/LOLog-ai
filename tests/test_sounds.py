"""효과음·배경음(#45): 섞기·덕킹, 콘티의 sfx·bgm 맞추기와 검사, 워커가 소리를 받아 섞기."""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from lolog_ai.sounds import (
    BGM,
    BGM_DB,
    BGM_DUCK_DB,
    MAX_SFX,
    NONE,
    SFX,
    SFX_OFFSET_MS,
    SFX_PEAK_DB,
    duck_envelope,
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


def test_queries_are_api_safe():
    import re

    for q in [*SFX, *BGM]:
        assert re.fullmatch(r"[a-z0-9_]{1,30}", q)  # sounds.md: 영어 소문자·숫자·_ 1~30자
    assert NONE not in SFX and NONE not in BGM and not set(SFX) & set(BGM)


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

def test_schema_has_sfx_and_bgm():
    sc = _schema(["화자1"])
    assert set(sc["properties"]["bgm"]["enum"]) == {*BGM, NONE}
    scene = sc["properties"]["scenes"]["items"]
    assert "sfx" in scene["required"] and set(scene["properties"]["sfx"]["enum"]) == {*SFX, NONE}


def test_fix_sounds_limits_and_cleans():
    sb = {"scenes": [{"sfx": v} for v in ["thud", "thud", "none", "pop", "door_slam", "ding", "boom", None]]}
    fix_sounds(sb, "bgm_comic")
    got = [s["sfx"] for s in sb["scenes"]]
    # 연달아 같은 thud, "none", 넘친 ding(4번째), 목록에 없는 boom은 지운다
    assert got == ["thud", None, None, "pop", "door_slam", None, None, None]
    assert sum(v is not None for v in got) <= MAX_SFX
    assert sb["bgm"] == "bgm_comic"
    fix_sounds(sb, "none")
    assert sb["bgm"] is None
    fix_sounds(sb, "jazz")
    assert sb["bgm"] is None


def test_validate_rejects_unknown_sound_values():
    sb = {"version": 1, "scenes": [{"index": 1, "sfx": "boom"}], "bgm": "jazz"}
    errs = validate(sb)
    assert any("sfx" in e for e in errs) and any("bgm" in e for e in errs)
    ok = {"version": 1, "scenes": [{"index": 1, "sfx": "pop"}], "bgm": None}
    assert not any("sfx" in e or "bgm" in e for e in validate(ok))


# ---------- 워커 ----------

def ogg_bytes(sec: float, sr: int = 44100) -> bytes:
    buf = io.BytesIO()
    t = np.arange(int(sec * sr)) / sr
    sf.write(buf, (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), sr, format="OGG", subtype="VORBIS")
    return buf.getvalue()


class FakeApi:
    def __init__(self, sounds: dict | None = None, status: int | None = None):
        self.sounds = sounds or {}
        self.status = status
        self.asked = []

    def get_sounds(self, kind, query):
        self.asked.append((kind, query))
        if self.status:
            raise ApiError(self.status, "{}")
        return [{"id": i, "kind": kind, "durationMs": 1000, "url": f"https://s/{query}/{i}"}
                for i in range(len(self.sounds.get(query, [])))]

    def fetch(self, url):
        q, i = url.split("/")[-2:]
        return self.sounds[q][int(i)]


def test_add_sounds_mixes_what_it_can():
    sb = sb_two_scenes()
    sb["scenes"][0]["sfx"], sb["scenes"][1]["sfx"], sb["bgm"] = "thud", "pop", "bgm_calm"
    api = FakeApi({"thud": [ogg_bytes(0.5)], "bgm_calm": [ogg_bytes(2.0), ogg_bytes(1.0)]})  # pop은 빈 목록
    track = np.zeros(SR * 6, dtype=np.float32)
    out, used = add_sounds(sb, track, SR, api, seed=5)
    assert used == ["thud", "bgm_calm"]
    assert np.abs(out).max() > 0 and len(out) == len(track)


@pytest.mark.parametrize("status", [404, 500])
def test_add_sounds_without_api_keeps_track(status):
    sb = sb_two_scenes()
    sb["scenes"][0]["sfx"], sb["bgm"] = "thud", "bgm_calm"
    track = np.full(SR * 6, 0.1, dtype=np.float32)
    out, used = add_sounds(sb, track, SR, FakeApi(status=status), seed=1)
    assert used == [] and out is track


def test_add_sounds_asks_each_query_once():
    sb = {"scenes": [{"index": i, "startMs": (i - 1) * 1000, "endMs": i * 1000, "sfx": "pop", "lines": []}
                     for i in (1, 2, 3)]}
    api = FakeApi({"pop": [ogg_bytes(0.3)]})
    add_sounds(sb, np.zeros(SR * 3, dtype=np.float32), SR, api, seed=0)
    assert api.asked == [("SFX", "pop")]
