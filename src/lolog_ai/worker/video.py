"""카드 영상 작업 (#33, LOLog-docs spec/video.md). 한 영상이 Jetson 두 대를 차례로 거친다.

3호 voice(msg)  lolog:video      source → 콘티(1호 LLM) → 대사 TTS → 시간표 → storyboard·audio 보내기
2호 video(msg)  lolog:comic      storyboard·audio 받기 → 장면 그림(FLUX) → 말하는 입모양(FLUX, lipsync.py)
                                 → 움직임·전환·자막·립싱크 → mp4 보내기
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
    for i, c in enumerate(sb["characters"]):
        c["voice"] = pick_voice(c, src_chars.get(c["key"]), have, seed=int(src["cardId"]) + i)
    who = {c["key"]: c["voice"] for c in sb["characters"]}

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
    buf = io.BytesIO()
    sf.write(buf, track, sr, format="WAV", subtype="PCM_16")

    api.post_video_storyboard(vid, sb)
    api.post_video_audio(vid, buf.getvalue())
    log.info("video %s 목소리: 장면 %d개, 대사 %d개, %.1f초 | 콘티 %.0f초, TTS %.0f초 | %s",
             vid, len(sb["scenes"]), len(clips), total / 1000, t1 - t0, t2 - t1, sorted(set(who.values())))


def video(msg: dict, api: InternalApi) -> None:
    from .. import lipsync
    from ..video_render import render
    from .draw import draw_scenes_lips

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
    images, mouths = draw_scenes_lips(sb, seed=vid, refs=refs)
    lips = None
    if mouths is not None:
        try:
            lips = lipsync.plan(sb, wav, mouths)
        except Exception:  # 립싱크는 덤이다. 실패해도 영상은 만든다
            log.exception("video %s 립싱크 계획 실패, 립싱크 없이 진행", vid)
    t1 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="lolog-video-") as d:
        d = Path(d)
        (d / "audio.wav").write_bytes(wav)
        res = render(sb, images, d / "audio.wav", d / "video.mp4", d / "thumb.jpg", lips=lips)
        mp4, thumb = (d / "video.mp4").read_bytes(), (d / "thumb.jpg").read_bytes()
    t2 = time.perf_counter()
    api.post_video(vid, mp4, thumb, int(res["durationMs"]))
    log.info("video %s 완성: 장면 %d장, %.1f초, %d바이트 | 그림 %.0f초, 합치기 %.0f초",
             vid, len(images), res["durationMs"] / 1000, len(mp4), t1 - t0, t2 - t1)

