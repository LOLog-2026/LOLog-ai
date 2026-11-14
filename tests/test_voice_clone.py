"""내 목소리(#43): 녹음 풀기·검사, 미리 듣기 작업, 쇼츠의 "self", TTS 사용자 목소리 이름, 워커 분기."""

from __future__ import annotations

import io
import re
import shutil

import numpy as np
import pytest
import soundfile as sf

from lolog_ai.tts import Tts, pick_voice
from lolog_ai.worker.api import ApiError
from lolog_ai.worker.voice_clone import (
    PREVIEW_TEXT,
    RECORD_SENTENCE,
    SR,
    VoiceSampleInvalid,
    check_sample,
    load_sample,
    prepare,
    voice_clone,
)


def speech(sec: float = 6.0, sr: int = SR, amp: float = 0.3, lead: float = 0.5) -> np.ndarray:
    """말소리 대신: 앞뒤 무음 + 크기가 출렁이는 220Hz."""
    t = np.arange(int(sec * sr)) / sr
    body = amp * np.sin(2 * np.pi * 220 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
    pad = np.zeros(int(lead * sr), dtype=np.float32)
    return np.concatenate([pad, body.astype(np.float32), pad])


def wav_bytes(wav: np.ndarray, sr: int, channels: int = 1) -> bytes:
    buf = io.BytesIO()
    data = np.stack([wav] * channels, axis=1) if channels > 1 else wav
    sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ---------- 풀기·검사 ----------

def test_load_wav_48k_stereo_to_24k_mono():
    out = load_sample(wav_bytes(speech(2.0, sr=48000), 48000, channels=2))
    assert out.dtype == np.float32 and out.ndim == 1
    assert abs(len(out) / SR - 3.0) < 0.01


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg 없음")
def test_load_m4a_with_ffmpeg(tmp_path):
    import subprocess

    src = tmp_path / "a.wav"
    src.write_bytes(wav_bytes(speech(4.0, sr=44100), 44100))
    m4a = tmp_path / "a.m4a"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(src), "-c:a", "aac", str(m4a)], check=True)
    out = load_sample(m4a.read_bytes())
    assert abs(len(out) / SR - 5.0) < 0.1


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg 없음")
def test_load_phone_m4a_with_moov_at_end(tmp_path):
    """폰 녹음처럼 목차(moov)가 소리(mdat) 뒤에 있고 소리가 64KB를 넘는 m4a.
    파이프로 넘기면 ffmpeg가 앞으로 되돌아가지 못해 빈 소리가 나왔다(종료 코드 0, #47)."""
    import subprocess

    m4a = tmp_path / "phone.m4a"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "anoisesrc=d=7:a=0.3", "-ar", "48000",
                    "-ac", "1", "-c:a", "aac", "-b:a", "96k", str(m4a)], check=True)
    data = m4a.read_bytes()
    assert data.find(b"moov") > data.find(b"mdat") > 0 and len(data) > 65536
    out = load_sample(data)
    assert abs(len(out) / SR - 7.0) < 0.1
    assert len(prepare(data)) / SR > 6.5  # 검사도 통과


def test_ffmpeg_empty_output_is_unreadable():
    with pytest.raises(VoiceSampleInvalid) as e:
        load_sample(b"\x00\x00\x00\x18ftypM4A not really audio")
    assert e.value.code == "VOICE_UNREADABLE"


def test_broken_file_is_unreadable():
    with pytest.raises(VoiceSampleInvalid) as e:
        load_sample(b"RIFF....not a wav")
    assert e.value.code == "VOICE_UNREADABLE"
    with pytest.raises(VoiceSampleInvalid):
        load_sample(b"")


def test_check_trims_silence_and_normalizes():
    out = check_sample(speech(6.0, lead=1.0))
    assert abs(len(out) / SR - 6.3) < 0.05  # 앞 0.1초 + 6초 + 뒤 0.2초
    assert abs(20 * np.log10(np.abs(out).max()) - (-1.0)) < 0.05


@pytest.mark.parametrize("wav, code", [
    (speech(2.0), "VOICE_TOO_SHORT"),
    (speech(25.0), "VOICE_TOO_LONG"),
    (speech(6.0, amp=0.005), "VOICE_TOO_QUIET"),
    (np.zeros(SR * 6, dtype=np.float32), "VOICE_TOO_QUIET"),
])
def test_check_rejects(wav, code):
    with pytest.raises(VoiceSampleInvalid) as e:
        check_sample(wav)
    assert e.value.code == code


def test_sentences():
    assert 35 <= len(re.findall(r"[가-힣]", RECORD_SENTENCE)) <= 45  # 초당 5~7글자로 5~10초 안
    assert RECORD_SENTENCE != PREVIEW_TEXT


# ---------- 가짜 TTS·api ----------

class FakeModel:
    def __init__(self):
        self.prompts = []

    def create_voice_clone_prompt(self, ref_audio, ref_text, x_vector_only_mode):
        self.prompts.append((ref_audio, ref_text))
        return {"n": len(self.prompts), "text": ref_text}

    def generate_voice_clone(self, text, language, voice_clone_prompt, max_new_tokens, **kw):
        self.last_prompt = voice_clone_prompt
        return [np.full(SR, 0.1, dtype=np.float32)], SR


@pytest.fixture
def tts(tmp_path, monkeypatch):
    for vid in ("young-f-01", "young-m-01"):
        (tmp_path / f"{vid}.wav").write_bytes(wav_bytes(speech(2.0), SR))
    t = Tts(tmp_path)
    t._model = FakeModel()
    try:
        import torch  # noqa: F401
    except ImportError:  # 모델 없이 돌리는 테스트라 torch는 시드만 쓴다
        import sys
        import types
        monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(manual_seed=lambda s: None))
    return t


class FakeApi:
    def __init__(self, samples: dict[int, bytes]):
        self.samples = samples
        self.previews: dict[int, bytes] = {}

    def get_voice_sample(self, user_id):
        if user_id not in self.samples:
            raise ApiError(404, '{"code":"VOICE_CLONE_NOT_FOUND"}')
        return self.samples[user_id]

    def put_voice_preview(self, user_id, wav):
        self.previews[user_id] = wav


def test_user_voice_names_are_per_user_and_recording(tts):
    a = tts.user_voice(1, speech(5.0), SR, RECORD_SENTENCE)
    assert tts.user_voice(1, speech(5.0), SR, RECORD_SENTENCE) == a  # 같은 녹음이면 다시 만들지 않음
    assert len(tts._model.prompts) == 1
    b = tts.user_voice(2, speech(5.0), SR, RECORD_SENTENCE)
    c = tts.user_voice(1, speech(5.0, amp=0.2), SR, RECORD_SENTENCE)  # 다시 녹음
    assert len({a, b, c}) == 3 and a.startswith("self:1:") and b.startswith("self:2:")
    assert "self" not in tts.voices() and not any(v.startswith("self") for v in tts.voices())


def test_say_unknown_user_voice_fails(tts):
    with pytest.raises(KeyError):
        tts.say("self:9:abc", "안녕")


def test_voice_clone_job_uploads_preview(tts):
    api = FakeApi({42: wav_bytes(speech(6.0, sr=48000), 48000)})
    voice_clone({"jobId": 1, "kind": "voice_clone", "userId": 42, "text": RECORD_SENTENCE}, api, tts)
    out, sr = sf.read(io.BytesIO(api.previews[42]))
    assert sr == SR and len(out) == SR
    assert tts._model.prompts[0][1] == RECORD_SENTENCE  # 읽은 문장이 ref_text
    ref, ref_sr = tts._model.prompts[0][0]
    assert ref_sr == SR and abs(len(ref) / SR - 6.3) < 0.05


def test_voice_clone_job_rejects_quiet_and_missing_text(tts):
    api = FakeApi({42: wav_bytes(speech(6.0, amp=0.003), SR)})
    with pytest.raises(VoiceSampleInvalid) as e:
        voice_clone({"jobId": 1, "kind": "voice_clone", "userId": 42, "text": "문장"}, api, tts)
    assert e.value.code == "VOICE_TOO_QUIET" and not api.previews
    with pytest.raises(VoiceSampleInvalid) as e:
        voice_clone({"jobId": 1, "kind": "voice_clone", "userId": 42, "text": " "}, api, tts)
    assert e.value.code == "VOICE_TEXT_MISSING"


# ---------- 쇼츠의 "self" ----------

def test_self_voice_in_shorts(tts):
    from lolog_ai.worker.video import _self_voice

    api = FakeApi({42: wav_bytes(speech(6.0), SR)})
    ch = {"ref": "profile:42", "voice": "self", "voiceText": RECORD_SENTENCE}
    name = _self_voice(ch, api, tts)
    assert name.startswith("self:42:")
    assert tts.say(name, "아니 그걸 왜 지금 말해?") is not None
    assert tts._model.last_prompt["text"] == RECORD_SENTENCE


@pytest.mark.parametrize("ch, samples", [
    ({"ref": "profile:42", "voice": "self", "voiceText": "문장"}, {}),                      # 그사이 지움(404)
    ({"ref": "profile:42", "voice": "self", "voiceText": "문장"}, {42: b"RIFFbroken"}),     # 녹음이 깨짐
    ({"ref": "profile:42", "voice": "self", "voiceText": None}, {42: b""}),                 # voiceText 없음
    ({"ref": "default:young-f-01", "voice": "self", "voiceText": "문장"}, {}),              # 기본 화자에 self
    ({"ref": "profile:42", "voice": "young-m-01"}, {}),                                     # 고른 기본 목소리
    (None, {}),
])
def test_self_voice_falls_back(tts, ch, samples):
    from lolog_ai.worker.video import _self_voice

    assert _self_voice(ch, FakeApi(samples), tts) is None


def test_pick_voice_ignores_self():
    have = {"young-f-01", "young-f-02", "young-m-01"}
    v = pick_voice({"key": "A", "gender": "f", "age": "young"}, {"ref": "profile:42", "voice": "self"}, have, 0)
    assert v in have


def test_api_server_error_is_not_swallowed(tts):
    from lolog_ai.worker.video import _self_voice

    class Down(FakeApi):
        def get_voice_sample(self, user_id):
            raise ApiError(500, "{}")

    with pytest.raises(ApiError):
        _self_voice({"ref": "profile:42", "voice": "self", "voiceText": "문장"}, Down({}), tts)


# ---------- 워커 분기 ----------

def test_worker_routes_voice_clone(monkeypatch):
    pytest.importorskip("redis")
    from lolog_ai.worker import __main__ as wm
    from lolog_ai.worker import video
    from lolog_ai.worker import voice_clone as vc

    calls = []
    monkeypatch.setattr(vc, "voice_clone", lambda msg, api, tts: calls.append(("clone", msg["userId"])))
    monkeypatch.setattr(video, "voice", lambda msg, api, llm, tts: calls.append(("shorts", msg["videoId"])))
    w = wm.Worker.__new__(wm.Worker)
    w.cfg = type("C", (), {"role": "video"})()
    w.api = w.llm = w.tts = None
    w.run_job({"jobId": 1, "kind": "voice_clone", "userId": 42, "text": "문장"})
    w.run_job({"jobId": 2, "videoId": 7})
    assert calls == [("clone", 42), ("shorts", 7)]
