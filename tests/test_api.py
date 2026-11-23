"""메인 백엔드 /internal 호출(worker/api.py, #65): 토큰, ApiError, multipart 모양.

urlopen을 바꿔 워커가 보내는 요청을 붙잡아 본다. 서버는 띄우지 않는다.
"""

from __future__ import annotations

import email
import io
import json
import urllib.error
import urllib.request

import pytest

from lolog_ai.worker.api import ApiError, InternalApi

BASE, TOKEN = "http://127.0.0.1:18080", "secret-token"


class Reply:
    def __init__(self, body: bytes = b""):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self) -> bytes:
        return self.body


class Server:
    """urlopen 자리. 받은 요청을 모으고, 정해 둔 답이나 HTTP 오류를 준다."""

    def __init__(self, body: bytes = b"", error: tuple[int, bytes] | None = None):
        self.body, self.error = body, error
        self.requests: list = []
        self.timeouts: list = []

    def __call__(self, req, timeout=None):
        self.requests.append(req)
        self.timeouts.append(timeout)
        if self.error:
            url = req if isinstance(req, str) else req.full_url
            raise urllib.error.HTTPError(url, self.error[0], "err", {}, io.BytesIO(self.error[1]))
        return Reply(self.body)

    @property
    def last(self) -> urllib.request.Request:
        return self.requests[-1]


@pytest.fixture
def server(monkeypatch) -> Server:
    s = Server()
    monkeypatch.setattr(urllib.request, "urlopen", s)
    return s


def headers(req: urllib.request.Request) -> dict[str, str]:
    return {k.lower(): v for k, v in req.header_items()}


def parts(req: urllib.request.Request) -> dict[str, dict]:
    """multipart 본문을 풀어 {파트 이름: {filename, type, data}}로 돌려준다."""
    ctype = headers(req)["content-type"]
    assert ctype.startswith("multipart/form-data; boundary=")
    msg = email.message_from_bytes(b"Content-Type: " + ctype.encode() + b"\r\n\r\n" + req.data)
    assert msg.is_multipart()
    out = {}
    for p in msg.get_payload():
        name = p.get_param("name", header="content-disposition")
        out[name] = {"filename": p.get_filename(), "type": p.get_content_type() if p.get("Content-Type") else None,
                     "data": p.get_payload(decode=True)}
    return out


# ---------- JSON 호출 ----------

def test_json_call_sends_token_and_utf8_body(server):
    api = InternalApi(BASE, TOKEN, timeout=7)
    api.post_cards(12, {"jobId": 3, "cards": [{"summary": "웃긴 이야기"}]})
    req = server.last
    assert req.get_method() == "POST" and req.full_url == f"{BASE}/internal/recordings/12/cards"
    assert headers(req)["x-internal-token"] == TOKEN
    assert headers(req)["content-type"] == "application/json"
    assert "웃긴 이야기".encode() in req.data                      # \uXXXX로 바꾸지 않고 UTF-8 그대로
    assert json.loads(req.data) == {"jobId": 3, "cards": [{"summary": "웃긴 이야기"}]}
    assert server.timeouts == [7]


def test_get_has_no_body_and_parses_json(server):
    server.body = json.dumps({"comicId": 7, "lines": []}).encode()
    src = InternalApi(BASE, TOKEN).get_comic_source(7)
    assert src == {"comicId": 7, "lines": []}
    req = server.last
    assert req.get_method() == "GET" and req.data is None and "content-type" not in headers(req)
    assert headers(req)["x-internal-token"] == TOKEN


def test_empty_response_is_none(server):
    assert InternalApi(BASE, TOKEN).post_storyboard(7, {"version": 1}) is None


@pytest.mark.parametrize("call, method, path", [
    (lambda a: a.post_text_cards(5, {}), "POST", "/internal/texts/5/cards"),
    (lambda a: a.get_comic_source(7), "GET", "/internal/comics/7/source"),
    (lambda a: a.post_storyboard(7, {}), "POST", "/internal/comics/7/storyboard"),
    (lambda a: a.get_video_source(9), "GET", "/internal/videos/9/source"),
    (lambda a: a.post_video_storyboard(9, {}), "POST", "/internal/videos/9/storyboard"),
    (lambda a: a.get_video_storyboard(9), "GET", "/internal/videos/9/storyboard"),
    (lambda a: a.get_video_audio(9), "GET", "/internal/videos/9/audio"),
    (lambda a: a.get_character_photo(42), "GET", "/internal/users/42/character/photo"),
    (lambda a: a.get_voice_sample(42), "GET", "/internal/users/42/voice-clone/sample"),
    (lambda a: a.fail_job(41, "x"), "POST", "/internal/jobs/41/fail"),
])
def test_paths(call, method, path, server):
    server.body = b"{}"
    call(InternalApi(BASE, TOKEN))
    assert (server.last.get_method(), server.last.full_url) == (method, BASE + path)


def test_binary_get_returns_bytes(server):
    server.body = b"RIFF\x00\x01\xff"
    assert InternalApi(BASE, TOKEN).get_video_audio(9) == b"RIFF\x00\x01\xff"


# ---------- 오류 ----------

def test_http_error_becomes_api_error_with_code(monkeypatch):
    body = json.dumps({"code": "COMIC_NOT_FOUND", "message": "없음"}).encode()
    monkeypatch.setattr(urllib.request, "urlopen", Server(error=(404, body)))
    with pytest.raises(ApiError) as e:
        InternalApi(BASE, TOKEN).get_comic_source(7)
    assert e.value.status == 404 and e.value.code == "COMIC_NOT_FOUND"   # 워커가 이 둘로 작업이 지워졌는지 본다
    assert str(e.value).startswith("HTTP 404: ")


@pytest.mark.parametrize("body", [b"<html>502 Bad Gateway</html>", b"", b'["list"]', b"\xff\xfe"])
def test_api_error_code_is_none_when_body_is_not_error_response(body, monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", Server(error=(502, body)))
    with pytest.raises(ApiError) as e:
        InternalApi(BASE, TOKEN).post_storyboard(7, {})
    assert e.value.status == 502 and e.value.code is None


def test_api_error_message_keeps_only_300_chars_of_body():
    e = ApiError(500, "가" * 1000)
    assert str(e) == "HTTP 500: " + "가" * 300


# ---------- multipart ----------

def test_post_image_multipart(server):
    png = b"\x89PNG\r\n\x1a\n" + bytes(range(256))
    InternalApi(BASE, TOKEN).post_image(7, png)
    req = server.last
    assert req.get_method() == "POST" and req.full_url == f"{BASE}/internal/comics/7/image"
    assert headers(req)["x-internal-token"] == TOKEN
    assert parts(req) == {"file": {"filename": "comic.png", "type": "image/png", "data": png}}


def test_post_character_avatar_sends_appearance_field(server):
    InternalApi(BASE, TOKEN).post_character_avatar(42, b"PNGDATA", "young woman, 짧은 머리")
    got = parts(server.last)
    assert server.last.full_url == f"{BASE}/internal/users/42/character/avatar"
    assert got["file"] == {"filename": "avatar.png", "type": "image/png", "data": b"PNGDATA"}
    assert got["appearance"]["filename"] is None
    assert got["appearance"]["data"].decode() == "young woman, 짧은 머리"


def test_post_video_sends_two_files_and_duration(server):
    InternalApi(BASE, TOKEN).post_video(9, b"MP4DATA", b"JPGDATA", 31500)
    got = parts(server.last)
    assert server.last.full_url == f"{BASE}/internal/videos/9/video"
    assert got["file"] == {"filename": "video.mp4", "type": "video/mp4", "data": b"MP4DATA"}
    assert got["thumbnail"] == {"filename": "thumb.jpg", "type": "image/jpeg", "data": b"JPGDATA"}
    assert got["durationMs"]["data"] == b"31500"


def test_post_video_audio(server):
    InternalApi(BASE, TOKEN).post_video_audio(9, b"RIFFWAVE")
    assert server.last.get_method() == "POST"
    assert parts(server.last) == {"file": {"filename": "audio.wav", "type": "audio/wav", "data": b"RIFFWAVE"}}


def test_voice_preview_uses_put(server):
    InternalApi(BASE, TOKEN).put_voice_preview(42, b"RIFFWAVE")
    assert server.last.get_method() == "PUT"
    assert server.last.full_url == f"{BASE}/internal/users/42/voice-clone/preview"
    assert parts(server.last) == {"file": {"filename": "preview.wav", "type": "audio/wav", "data": b"RIFFWAVE"}}


# ---------- 그 밖 ----------

def test_fetch_does_not_send_internal_token(server):
    """서명 주소(저장소)로 가는 요청이다. 내부 토큰이 밖으로 나가면 안 된다."""
    server.body = b"IMG"
    url = "https://storage.example/ref.png?sig=abc"
    assert InternalApi(BASE, TOKEN, timeout=7).fetch(url) == b"IMG"
    assert server.requests == [url] and server.timeouts == [7]   # Request가 아니라 주소만 넘긴다(헤더 없음)


def test_fetch_http_error_becomes_api_error(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", Server(error=(403, b"expired")))
    with pytest.raises(ApiError) as e:
        InternalApi(BASE, TOKEN).fetch("https://storage.example/ref.png?sig=abc")
    assert e.value.status == 403 and e.value.code is None


def test_get_sounds_encodes_query_and_defaults_to_empty(server):
    server.body = json.dumps({"sounds": [{"id": 1, "kind": "sfx", "durationMs": 800, "url": "u"}]}).encode()
    api = InternalApi(BASE, TOKEN)
    assert api.get_sounds("sfx", "door slam&x")[0]["id"] == 1
    assert server.last.full_url == f"{BASE}/internal/sounds?kind=sfx&query=door+slam%26x"
    for body in (b"", b"{}", b'{"sounds": null}'):
        server.body = body
        assert api.get_sounds("bgm", "bgm_comic") == []


def test_fail_job_reason_is_cut_and_never_empty(server):
    api = InternalApi(BASE, TOKEN)
    api.fail_job(41, "가" * 600)
    assert json.loads(server.last.data) == {"reason": "가" * 500}
    api.fail_job(41, "")
    assert json.loads(server.last.data) == {"reason": "unknown"}
