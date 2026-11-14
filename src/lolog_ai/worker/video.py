"""카드 영상 작업 (#33, LOLog-docs spec/video.md). 한 영상이 Jetson 두 대를 차례로 거친다.

3호 voice(msg)  lolog:video      source → 콘티(1호 LLM) → 대사 TTS → 시간표 → storyboard·audio 보내기
2호 video(msg)  lolog:comic      storyboard·audio 받기 → 장면 그림(FLUX) → 움직임·전환·자막 → mp4 보내기
                (kind "video")

3호는 TTS 모델을 워커가 켜져 있는 동안 올려 둔다(올리는 데 약 20초, 메모리 약 3GB).
로그에 대사를 남기지 않는다(TR-12). 임시 파일은 작업이 끝나면 지운다.
"""

from __future__ import annotations

import io
import logging
import tempfile
import time
from pathlib import Path

from PIL import Image

from .api import InternalApi
from .llm import Llm
from .video_gen import make_video_storyboard

log = logging.getLogger(__name__)


def voice(msg: dict, api: InternalApi, llm: Llm, tts) -> None:
    import numpy as np
    import soundfile as sf

    from ..tts import pick_voice
    from ..video_storyboard import all_lines, timeline

    vid = int(msg["videoId"])
    src = api.get_video_source(vid)
    t0 = time.perf_counter()
    sb = make_video_storyboard(src, llm)
    t1 = time.perf_counter()

    have = tts.voices()
    src_chars = {sp["key"]: sp.get("character") for sp in src["speakers"]}
    who = {}
    for i, c in enumerate(sb["characters"]):
        src_char = src_chars.get(c["key"])
        mine = _self_voice(src_char, api, tts)  # 내 목소리("self")면 그 사람 녹음, 못 받으면 None
        c["voice"] = "self" if mine else pick_voice(c, src_char, have, seed=int(src["cardId"]) + i)
        who[c["key"]] = mine or c["voice"]

    clips, sr = [], 24000
    for i, d in enumerate(all_lines(sb)):
        res = tts.say(who[d["speaker"]], d["text"], seed=vid * 100 + i)
        clips.append(None if res is None else res[0])
        if res is not None:
            sr = res[1]
    t2 = time.perf_counter()
    total = timeline(sb, [0.0 if c is None else len(c) / sr for c in clips])
    track = np.zeros(int(total / 1000 * sr) + 1, dtype=np.float32)
    for d, c in zip(all_lines(sb), clips):
        if c is not None:
            s = int(d["startMs"] / 1000 * sr)
            track[s:s + len(c)] = c[:len(track) - s]
    t3 = time.perf_counter()
    track, used = add_sounds(sb, track, sr, api, seed=vid)
    buf = io.BytesIO()
    sf.write(buf, track, sr, format="WAV", subtype="PCM_16")

    api.post_video_storyboard(vid, sb)
    api.post_video_audio(vid, buf.getvalue())
    log.info("video %s 목소리: 장면 %d개, 대사 %d개, %.1f초 | 콘티 %.0f초, TTS %.0f초, 소리 %.1f초 | %s | 소리 %s",
             vid, len(sb["scenes"]), len(clips), total / 1000, t1 - t0, t2 - t1, time.perf_counter() - t3,
             sorted({v.rsplit(":", 1)[0] for v in who.values()}), used)  # 내 목소리는 지문을 빼고 self:회원ID만


def _self_voice(src_char: dict | None, api: InternalApi, tts) -> str | None:
    """source character의 voice가 "self"면 그 사람 녹음을 받아 TTS에 올리고 목소리 이름을 돌려준다.

    백엔드는 내 목소리가 READY이고(친구 쇼츠면 동의 ③도) 그럴 때만 "self"와 voiceText를 보낸다(spec/video.md 6장).
    그 사이 지웠거나(404 VOICE_CLONE_NOT_FOUND) 녹음이 검사를 못 넘으면 None → 기본 화자 목소리로 읽는다.
    """
    from .api import ApiError
    from .voice_clone import SR, VoiceSampleInvalid, prepare

    src_char = src_char or {}
    if src_char.get("voice") != "self":
        return None
    kind, _, rid = (src_char.get("ref") or "").partition(":")
    text = (src_char.get("voiceText") or "").strip()
    if kind != "profile" or not rid.isdigit() or not text:
        log.warning("내 목소리인데 ref·voiceText가 없어 기본 목소리로: ref=%s", src_char.get("ref"))
        return None
    try:
        ref = prepare(api.get_voice_sample(int(rid)))
    except ApiError as e:
        if e.status == 404:
            log.info("user %s 내 목소리가 없어져 기본 목소리로: %s", rid, e.code)
            return None
        raise
    except VoiceSampleInvalid as e:
        log.warning("user %s 내 목소리 녹음이 검사를 못 넘어 기본 목소리로: %s", rid, e.code)
        return None
    return tts.user_voice(int(rid), ref, SR, text)


def add_sounds(sb: dict, track, sr: int, api: InternalApi, seed: int):
    """콘티의 sfx·bgm 검색어로 소리를 받아 섞는다(spec/sounds.md). (섞은 track, 쓴 검색어 목록).

    효과음·배경음은 있으면 좋은 것(P2)이라, 목록 API가 아직 없거나(404) 비었거나 받기·풀기에 실패하면
    그 소리만 빼고 영상은 그대로 만든다. 같은 영상은 늘 같은 소리를 고른다(seed = videoId).
    """
    from ..sounds import mix, pick

    want = [("SFX", sc["index"], sc["sfx"]) for sc in sb["scenes"] if sc.get("sfx")]
    if sb.get("bgm"):
        want.append(("BGM", 0, sb["bgm"]))
    sfx, bgm, used, cache = {}, None, [], {}
    for kind, idx, query in want:
        try:
            if (kind, query) not in cache:
                item = pick(api.get_sounds(kind, query), seed + len(cache))
                cache[(kind, query)] = None if item is None else _load_sound(api.fetch(item["url"]), sr)
            wav = cache[(kind, query)]
        except Exception as e:  # noqa: BLE001 - 소리 하나 때문에 영상을 실패시키지 않는다
            log.warning("소리 %s %s 못 받음, 빼고 만듦: %s", kind, query, type(e).__name__ + ": " + str(e)[:120])
            cache[(kind, query)] = wav = None
        if wav is None:
            continue
        used.append(query)
        if kind == "BGM":
            bgm = wav
        else:
            sfx[idx] = wav
    return (mix(track, sr, sb, sfx, bgm) if used else track), used


def _load_sound(data: bytes, sr: int):
    """ogg(또는 wav) 바이트 → sr 모노 float32."""
    import soundfile as sf

    from ..sounds import resample

    wav, s = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return resample(wav.mean(axis=1), int(s), sr)


def video(msg: dict, api: InternalApi) -> None:
    from ..video_render import render
    from .draw import draw_scenes

    vid = int(msg["videoId"])
    sb = api.get_video_storyboard(vid)
    wav = api.get_video_audio(vid)
    # 화자에 붙은 그림(기본 화자·프로필 캐릭터)은 참조로 넣어 만화처럼 같은 얼굴로 그린다. 백엔드가 storyboard를
    # 돌려줄 때 그림이 있는 인물에 새로 서명한 imageUrl을 붙여 준다(spec/video.md 3장). 없으면 외모 글로만 그린다
    refs = {}
    for c in sb["characters"]:
        url = c.pop("imageUrl", None)
        if url:
            refs[c["key"]] = Image.open(io.BytesIO(api.fetch(url))).convert("RGB")
    t0 = time.perf_counter()
    images = draw_scenes(sb, seed=vid, refs=refs)
    t1 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="lolog-video-") as d:
        d = Path(d)
        (d / "audio.wav").write_bytes(wav)
        res = render(sb, images, d / "audio.wav", d / "video.mp4", d / "thumb.jpg")
        mp4, thumb = (d / "video.mp4").read_bytes(), (d / "thumb.jpg").read_bytes()
    t2 = time.perf_counter()
    api.post_video(vid, mp4, thumb, int(res["durationMs"]))
    log.info("video %s 완성: 장면 %d장, %.1f초, %d바이트 | 그림 %.0f초, 합치기 %.0f초",
             vid, len(images), res["durationMs"] / 1000, len(mp4), t1 - t0, t2 - t1)

