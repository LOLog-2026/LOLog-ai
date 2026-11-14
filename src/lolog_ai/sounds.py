"""쇼츠 효과음·배경음 (#45, LOLog-docs spec/sounds.md, FR-EXT-03).

백엔드가 Freesound CC0 소리를 검색어 값마다 미리 모아 둔다. 영상 콘티가 장면마다 `sfx`, 영상 하나에 `bgm`을
아래 검색어 값 중에서 고르면(없으면 null), 3호 워커가 그 검색어 소리 목록을 받아 하나를 골라 대사 wav에 섞는다.

    mix(track, sr, sb, sfx={scene index: wav}, bgm=wav)   # 대사 위에 효과음을 얹고 배경음은 대사 동안 줄인다

검색어 목록은 api(sound_library.query)와 같아야 한다. 바꿀 때는 docs sounds.md 3장도 같이 바꾼다.
numpy만 쓴다.
"""

from __future__ import annotations

import numpy as np

# 효과음 검색어 → 콘티 LLM에게 보여 주는 뜻(언제 쓰는지)
SFX = {
    "door_slam": "문이 쾅 닫힘",
    "footsteps": "발소리, 걸어오거나 뛰어감",
    "laugh": "여럿이 웃는 소리",
    "applause": "박수",
    "phone_notify": "휴대폰 알림·진동",
    "keyboard": "키보드 타자",
    "dishes": "식기·컵 부딪힘, 식당·부엌",
    "car_horn": "자동차 경적",
    "rain": "비 오는 소리",
    "wind": "바람 소리",
    "dog": "개 짖는 소리",
    "cat": "고양이 울음",
    "pop": "뿅, 무언가 튀어나옴",
    "fail": "실패·허탈(뿌우우)",
    "success": "짠, 성공·뿌듯",
    "rimshot": "바둠츠, 썰렁한 농담 뒤",
    "whoosh": "휙, 빠르게 지나감·장면 전환",
    "thud": "쿵, 넘어지거나 부딪힘",
    "sigh": "한숨",
    "gasp": "헉, 놀람",
    "ding": "띵, 깨달음·생각남",
    "cafe": "카페 웅성거림",
    "classroom": "교실 웅성거림",
    "crowd": "사람 많은 거리·행사장 소음",
}
# 배경음 검색어 → 분위기
BGM = {
    "bgm_upbeat": "신나고 밝음",
    "bgm_calm": "잔잔함, 일상",
    "bgm_tense": "긴장, 조마조마",
    "bgm_sad": "슬픔, 짠함",
    "bgm_comic": "코믹, 엉뚱함",
}
NONE = "none"  # 콘티 스키마에서 "고르지 않음". 콘티에는 null로 들어간다

MAX_SFX = 3            # 영상 하나에 효과음은 이만큼까지(많으면 시끄럽다)
SFX_OFFSET_MS = 120    # 장면이 바뀌고 이만큼 뒤에 효과음
SFX_MAX_MS = 2500      # 효과음은 이보다 길면 자른다(배경 소음 같은 긴 소리)
SFX_PEAK_DB = -9.0     # 효과음 가장 큰 소리. 대사(-1dBFS 안팎)보다 작게
BGM_DB = -22.0         # 배경음 크기(가장 큰 소리 기준). 대사 없을 때
BGM_DUCK_DB = -32.0    # 대사가 나올 때 배경음
DUCK_PAD_MS = (150, 300)  # 대사 앞뒤로 이만큼 미리 줄이고 늦게 되돌린다
RAMP_MS = 200          # 줄이고 되돌리는 데 걸리는 시간
FADE_MS = (500, 1200)  # 배경음 처음·끝 페이드


def _db(x: float) -> float:
    return 10 ** (x / 20)


def to_null(v: str | None) -> str | None:
    return None if v in (None, "", NONE) else v


def resample(wav: np.ndarray, sr: int, target: int) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if sr == target or len(wav) == 0:
        return wav
    try:
        import librosa  # 3호에는 qwen-tts가 깔아 둠

        return librosa.resample(wav, orig_sr=sr, target_sr=target).astype(np.float32)
    except ImportError:
        n = round(len(wav) * target / sr)
        return np.interp(np.linspace(0, len(wav) - 1, n), np.arange(len(wav)), wav).astype(np.float32)


def _peak_to(wav: np.ndarray, db: float) -> np.ndarray:
    peak = float(np.abs(wav).max()) if len(wav) else 0.0
    return wav if peak <= 0 else wav * (_db(db) / peak)


def _fade(wav: np.ndarray, sr: int, in_ms: int, out_ms: int) -> np.ndarray:
    wav = wav.copy()
    a, b = min(len(wav), sr * in_ms // 1000), min(len(wav), sr * out_ms // 1000)
    if a:
        wav[:a] *= np.linspace(0, 1, a, dtype=np.float32)
    if b:
        wav[-b:] *= np.linspace(1, 0, b, dtype=np.float32)
    return wav


def duck_envelope(sb: dict, n: int, sr: int) -> np.ndarray:
    """배경음에 곱할 크기(샘플마다). 대사 구간(앞뒤 여유 포함)은 BGM_DUCK_DB, 나머지는 BGM_DB, 사이는 부드럽게."""
    step = sr // 100  # 10ms 단위로 만들고 늘린다
    m = n // step + 2
    hi, lo = _db(BGM_DB), _db(BGM_DUCK_DB)
    env = np.full(m, hi, dtype=np.float32)
    for s in sb.get("scenes") or []:
        for d in s.get("lines") or []:
            if d.get("spoken") is False:
                continue
            a = max((d["startMs"] - DUCK_PAD_MS[0]) // 10, 0)
            b = min((d["endMs"] + DUCK_PAD_MS[1]) // 10 + 1, m)
            env[a:b] = lo
    k = max(RAMP_MS // 10, 1)
    env = np.convolve(np.pad(env, (k, k), mode="edge"), np.ones(k, dtype=np.float32) / k, mode="same")[k:-k]
    return np.interp(np.arange(n), np.arange(m) * step, env).astype(np.float32)


def mix(track: np.ndarray, sr: int, sb: dict, sfx: dict[int, np.ndarray] | None = None,
        bgm: np.ndarray | None = None) -> np.ndarray:
    """대사 wav(track)에 효과음·배경음을 섞는다. sfx는 {장면 index: 소리}, 소리는 모두 sr로 맞춘 것.

    효과음은 그 장면 시작 SFX_OFFSET_MS 뒤에 SFX_PEAK_DB로, 장면이 끝나기 전에 끝나게 자른다.
    배경음은 영상 길이만큼 이어 붙여 처음·끝을 페이드하고, 대사가 나오는 동안 줄인다.
    다 섞은 뒤 넘치면(peak > 0.99) 전체를 줄인다.
    """
    out = np.asarray(track, dtype=np.float32).copy()
    n = len(out)
    scenes = {s["index"]: s for s in sb.get("scenes") or []}
    for idx, wav in (sfx or {}).items():
        s = scenes.get(idx)
        if s is None or wav is None or not len(wav):
            continue
        start = int((s["startMs"] + SFX_OFFSET_MS) * sr / 1000)
        room = int((s["endMs"] - s["startMs"] - SFX_OFFSET_MS) * sr / 1000)
        clip = wav[:max(min(room, SFX_MAX_MS * sr // 1000), 0)]
        if start >= n or not len(clip):
            continue
        clip = _fade(_peak_to(clip, SFX_PEAK_DB), sr, 10, 150)[:n - start]
        out[start:start + len(clip)] += clip
    if bgm is not None and len(bgm):
        loop = np.tile(bgm, n // len(bgm) + 1)[:n]
        loop = _fade(_peak_to(loop, 0.0), sr, *FADE_MS)
        out += loop * duck_envelope(sb, n, sr)
    peak = float(np.abs(out).max()) if n else 0.0
    if peak > 0.99:
        out *= 0.99 / peak
    return out


def pick(sounds: list[dict], seed: int) -> dict | None:
    """같은 검색어 소리 중 하나. seed(videoId 등)로 정해서 같은 영상은 늘 같은 소리."""
    return sounds[seed % len(sounds)] if sounds else None
