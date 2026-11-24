"""쇼츠 효과음·배경음 (#45, #76, LOLog-docs spec/sounds.md, FR-EXT-03).

백엔드가 Freesound CC0 소리를 모아 CLAP 소리 벡터와 함께 둔다. 영상 콘티가 장면마다 `sfx`, 영상 하나에 `bgm`을
들리는 소리를 설명하는 영어 한 줄로 쓰면(없으면 null), 3호 워커가 그 글로 api 벡터 검색(POST /internal/sounds/search)을
불러 가까운 소리 몇 개 중 하나를 골라 대사 wav에 섞는다.

    mix(track, sr, sb, sfx={scene index: wav}, bgm=wav)   # 대사 위에 효과음을 얹고 배경음은 대사 동안 줄인다

설명 글 규칙(아래 예시와 video_gen 프롬프트): CLAP은 소리를 사람이 설명한 영어 캡션("a door slams shut")으로 배워서,
장면이 아니라 귀에 들리는 소리를 짧게 써야 가까운 소리가 나온다. 한 줄에 소리 하나, 부정("no music")은 모델이 못 알아듣고,
이름·대사·숫자는 서버로 나가면 안 되고 검색에도 도움이 안 된다. 배경음은 모아 둔 30~60초 음악이라 음악으로 쓴다.
numpy만 쓴다.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # numpy는 소리를 다룰 때만 불러, numpy가 없는 1·2호에서도 이 모듈을 import할 수 있게 한다
    import numpy as np

# 콘티 LLM에게 보여 주는 설명 글 예시. 지원이 모으는 갈래(생활 소리, 사람 반응, 동물, 탈것, 날씨, 장소 분위기,
# 만화 효과음 / 음악 분위기, sounds.md 3장, lolog-api scripts/sound-queries.tsv)를 고루 담는다.
# 이 중에서 고르라는 게 아니라 이런 모양으로 쓰라는 것
SFX_EXAMPLES = [
    "a door slamming shut", "footsteps running on a wooden floor", "a phone vibrating on a table",
    "typing on a keyboard", "cups clinking in a kitchen", "a group of people laughing", "an audience clapping",
    "a woman sighing", "a man gasping in surprise", "a dog barking", "a cat meowing", "a car horn honking",
    "heavy rain falling", "strong wind blowing", "busy cafe chatter", "a noisy school classroom",
    "a cartoon pop", "a cartoon boing", "a record scratch", "a sad trombone fail sound",
    "a cheerful success jingle", "a rimshot drum hit", "a quick whoosh", "a heavy thud of something falling", "a bright bell ding",
]
BGM_EXAMPLES = [
    "upbeat cheerful pop music", "calm acoustic guitar music", "tense suspense music with ticking",
    "sad slow piano music", "playful comedy music with pizzicato strings",
]
NONE = "none"  # 콘티 스키마에서 "소리 없음". 콘티에는 null로 들어간다
# 응답 형식(JSON 스키마)에 거는 모양. 한글·숫자·대문자 이름을 출력 단계에서 막는다. 1호 시험(11-24)에서
# - 길이 제한이 없으면 "none, none, none, …"을 끝없이 이어 써 max_tokens에서 잘렸다 → 길이를 묶는다
# - 쉼표·마침표를 허용하면 "a phone vibrating, then an alarm, a teacher sighs"처럼 소리를 나열했다 → 막는다
DESC_PATTERN = r"^[a-z][a-z '-]{2,59}$"
DESC_MAX = 60          # 설명 글 최대 글자 수(api는 200자까지 받지만 길면 검색이 흐려진다)
DESC_MAX_WORDS = 12    # 프롬프트는 3~10단어. 조금 넘는 것은 받는다
_DESC_RE = re.compile(DESC_PATTERN)
_NEGATION = re.compile(r"\b(no|not|without|never)\b")
# 소리를 이어 쓴 곳. 첫 소리만 남긴다("tense music then upbeat music" → "tense music")
_NEXT_SOUND = re.compile(r",|;|\.|\s(?:and\s)?then\s|\sfollowed by\s|\swhile\s|\sand\s(?=(?:a|an|the|some)\s)")
# 60자 제한에 걸려 끊긴 끝("... splashing on a")에 남는 말
_DANGLING = re.compile(r"(?:\s(?:a|an|the|on|in|at|of|to|with|and|from|by|into|over))+$")
SEARCH_LIMIT = 3       # search로 받는 후보 수. 그중 하나를 videoId로 고른다(같은 영상은 늘 같은 소리)

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


def clean_desc(v: str | None) -> str | None:
    """설명 글을 다듬는다(소문자, 공백 하나, 끝 문장 부호 빼기). 규칙에 안 맞으면 None(그 소리는 빼고 만든다)."""
    if not isinstance(v, str):
        return None
    v = re.sub(r"\s+", " ", v.strip().lower())
    v = _NEXT_SOUND.split(v, maxsplit=1)[0].strip(" .,")  # 한 줄에 소리 하나
    head, sep, _ = v.partition(" and ")
    if sep and len(head.split()) >= 3:  # 소리가 이미 3단어 넘게 적혔으면 "and" 뒤는 덧붙인 소리다(1호 시험 11-24)
        v = head
    v = _DANGLING.sub("", v)
    return v if v != NONE and is_desc(v) else None


def is_desc(v: str) -> bool:
    """검사(video_storyboard.validate)와 clean_desc가 같이 쓴다."""
    return (3 <= len(v) <= DESC_MAX and bool(_DESC_RE.match(v)) and len(v.split()) <= DESC_MAX_WORDS
            and not _NEGATION.search(v))


def resample(wav: np.ndarray, sr: int, target: int) -> np.ndarray:
    import numpy as np

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
    import numpy as np

    peak = float(np.abs(wav).max()) if len(wav) else 0.0
    return wav if peak <= 0 else wav * (_db(db) / peak)


def _fade(wav: np.ndarray, sr: int, in_ms: int, out_ms: int) -> np.ndarray:
    import numpy as np

    wav = wav.copy()
    a, b = min(len(wav), sr * in_ms // 1000), min(len(wav), sr * out_ms // 1000)
    if a:
        wav[:a] *= np.linspace(0, 1, a, dtype=np.float32)
    if b:
        wav[-b:] *= np.linspace(1, 0, b, dtype=np.float32)
    return wav


def duck_envelope(sb: dict, n: int, sr: int) -> np.ndarray:
    """배경음에 곱할 크기(샘플마다). 대사 구간(앞뒤 여유 포함)은 BGM_DUCK_DB, 나머지는 BGM_DB, 사이는 부드럽게."""
    import numpy as np

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
    import numpy as np

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
    """search가 돌려준 후보 중 하나. seed(videoId 등)로 정해서 같은 영상은 늘 같은 소리."""
    return sounds[seed % len(sounds)] if sounds else None
