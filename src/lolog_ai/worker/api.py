"""메인 백엔드 /internal 호출 (spec/api-spec.md "워커 전용"). 표준 라이브러리만 쓴다."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
import uuid


class ApiError(Exception):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:300]}")
        self.status = status
        try:
            self.code = json.loads(body).get("code")  # ErrorResponse.code (api-dto.md)
        except (ValueError, AttributeError):
            self.code = None


class InternalApi:
    def __init__(self, base_url: str, token: str, timeout: float = 60):
        self.base = base_url
        self.token = token
        self.timeout = timeout

    def _send(self, method: str, path: str, data: bytes | None = None,
              content_type: str = "application/json") -> bytes:
        req = urllib.request.Request(self.base + path, data=data, method=method)
        req.add_header("X-Internal-Token", self.token)
        if data is not None:
            req.add_header("Content-Type", content_type)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            raise ApiError(e.code, e.read().decode(errors="replace")) from None

    def _json(self, method: str, path: str, body: dict | None = None):
        data = None if body is None else json.dumps(body, ensure_ascii=False).encode()
        out = self._send(method, path, data)
        return json.loads(out) if out else None

    def post_cards(self, recording_id: int, result: dict) -> None:
        self._json("POST", f"/internal/recordings/{recording_id}/cards", result)

    def post_text_cards(self, text_id: int, result: dict) -> None:
        """글 결과(카드 1장). 그사이 글·카드가 지워졌으면 404 TEXT_NOT_FOUND (texts.md 5장)."""
        self._json("POST", f"/internal/texts/{text_id}/cards", result)

    def get_comic_source(self, comic_id: int) -> dict:
        return self._json("GET", f"/internal/comics/{comic_id}/source")

    def post_storyboard(self, comic_id: int, storyboard: dict) -> None:
        self._json("POST", f"/internal/comics/{comic_id}/storyboard", storyboard)

    def _post_png(self, path: str, png: bytes, filename: str, fields: dict[str, str] | None = None) -> None:
        boundary = uuid.uuid4().hex
        body = b"".join(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
            for k, v in (fields or {}).items()
        ) + (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n"
            f"Content-Type: image/png\r\n\r\n"
        ).encode() + png + f"\r\n--{boundary}--\r\n".encode()
        self._send("POST", path, body, f"multipart/form-data; boundary={boundary}")

    def _post_files(self, path: str, files: list[tuple[str, str, str, bytes]],
                    fields: dict[str, str] | None = None, method: str = "POST") -> None:
        """multipart. files: (파트 이름, 파일 이름, Content-Type, 바이트)"""
        boundary = uuid.uuid4().hex
        parts = [f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
                 for k, v in (fields or {}).items()]
        for name, filename, ctype, data in files:
            parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\n"
                         f"Content-Type: {ctype}\r\n\r\n".encode() + data + b"\r\n")
        self._send(method, path, b"".join(parts) + f"--{boundary}--\r\n".encode(),
                   f"multipart/form-data; boundary={boundary}")

    # ---------- 카드 영상 (LOLog-docs spec/video.md 3장) ----------

    def get_video_source(self, video_id: int) -> dict:
        """3호. 부르면 영상이 VOICE가 된다. 없거나 끝났으면 404 VIDEO_NOT_FOUND."""
        return self._json("GET", f"/internal/videos/{video_id}/source")

    def post_video_storyboard(self, video_id: int, storyboard: dict) -> None:
        self._json("POST", f"/internal/videos/{video_id}/storyboard", storyboard)

    def post_video_audio(self, video_id: int, wav: bytes) -> None:
        """3호. 받으면 DRAWING이 되고 백엔드가 2호 큐(lolog:comic, kind video)에 넣는다."""
        self._post_files(f"/internal/videos/{video_id}/audio", [("file", "audio.wav", "audio/wav", wav)])

    def get_video_storyboard(self, video_id: int) -> dict:
        return self._json("GET", f"/internal/videos/{video_id}/storyboard")

    def get_video_audio(self, video_id: int) -> bytes:
        return self._send("GET", f"/internal/videos/{video_id}/audio")

    def post_video(self, video_id: int, mp4: bytes, thumbnail_jpg: bytes, duration_ms: int) -> None:
        """2호. 받으면 DONE과 푸시."""
        self._post_files(f"/internal/videos/{video_id}/video",
                         [("file", "video.mp4", "video/mp4", mp4), ("thumbnail", "thumb.jpg", "image/jpeg", thumbnail_jpg)],
                         {"durationMs": str(duration_ms)})

    def post_image(self, comic_id: int, png: bytes) -> None:
        self._post_png(f"/internal/comics/{comic_id}/image", png, "comic.png")

    def get_character_photo(self, user_id: int) -> bytes:
        """프로필 캐릭터 원본 사진(characters.md /internal). 이미 지워졌으면 404 CHARACTER_NOT_FOUND."""
        return self._send("GET", f"/internal/users/{user_id}/character/photo")

    def post_character_avatar(self, user_id: int, png: bytes, appearance: str) -> None:
        """appearance는 백엔드가 저장해 두었다가 만화 source의 character.appearance로 돌려준다."""
        self._post_png(f"/internal/users/{user_id}/character/avatar", png, "avatar.png", {"appearance": appearance})

    # ---------- 내 목소리 (LOLog-docs spec/characters.md "내 목소리") ----------

    def get_voice_sample(self, user_id: int) -> bytes:
        """원본 녹음(wav 또는 m4a). 만드는 중이거나 READY가 아니면 404 VOICE_CLONE_NOT_FOUND.
        사용자 원본이라 서명 주소가 아니라 api가 직접 준다."""
        return self._send("GET", f"/internal/users/{user_id}/voice-clone/sample")

    def put_voice_preview(self, user_id: int, wav: bytes) -> None:
        """미리 듣기. 받으면 READY와 푸시. 만드는 중이 아니면 404 VOICE_CLONE_NOT_FOUND."""
        self._post_files(f"/internal/users/{user_id}/voice-clone/preview",
                         [("file", "preview.wav", "audio/wav", wav)], method="PUT")

    def fetch(self, url: str) -> bytes:
        """source의 imageUrl(서명 주소) 받기. 서명 주소라 내부 토큰을 붙이지 않는다."""
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            raise ApiError(e.code, e.read().decode(errors="replace")) from None

    def fail_job(self, job_id: int, reason: str) -> None:
        self._json("POST", f"/internal/jobs/{job_id}/fail", {"reason": reason[:500] or "unknown"})
