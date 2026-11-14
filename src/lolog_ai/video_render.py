"""영상 콘티 + 장면 그림 + 대사 음성 → mp4 (#33).

프레임을 코드로 한 장씩 그려 ffmpeg(libx264 + AAC)에 넘긴다. 움직임·전환·자막을 모두 여기서 그리므로
같은 입력이면 같은 영상이 나온다. 필요한 것은 Pillow, numpy, ffmpeg 실행 파일뿐이다.

    render(storyboard, images, "audio.wav", "out.mp4")      # images: 장면 순서의 PIL 그림(정사각형)
    render(..., lips=lipsync.plan(...))                     # 말하는 입모양·눈 깜빡임을 프레임마다 바꿔 넣는다

화면(세로 720×1280): 장면 그림을 흐리게 키운 배경 위에, 그림(720×720)을 위쪽에 두고 아래에 자막을 쓴다.
시간표(startMs·endMs)는 video_storyboard.timeline()이 채운 값을 쓴다.

환경 변수 FFMPEG: ffmpeg 경로(없으면 PATH에서 찾는다)
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from .video_storyboard import TRANSITION_MS

FONT = Path(os.environ.get("LOLOG_FONT", Path.home() / ".cache/lolog/fonts/Jua-Regular.ttf"))
ART_TOP = 170          # 장면 그림 위쪽 여백
SUB_TOP_GAP = 46       # 그림 아래에서 자막까지
FONT_SIZE, NAME_SIZE = 46, 30
NAME_COLORS = ["#FFD166", "#7BDFF2", "#F7AEF8", "#B8F2A6"]  # 화자1~4
FADE_IN_MS = 300
ZOOM = 1.12            # 확대·이동 움직임의 최대 배율


def _ffmpeg() -> str:
    exe = os.environ.get("FFMPEG") or shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg를 찾지 못했어요 (환경 변수 FFMPEG에 경로를 주세요)")
    return exe


def _ease(p: float) -> float:
    return 0.5 - 0.5 * math.cos(math.pi * min(1.0, max(0.0, p)))


def _crop_box(motion: str, p: float, t_ms: float, size: int) -> tuple[float, float, float, float]:
    """움직임 → 원본 그림에서 잘라 낼 네모(왼쪽, 위, 오른쪽, 아래). p는 장면 진행도 0~1."""
    e = _ease(p)
    scale, cx, cy = 1.0, 0.5, 0.5
    slack = (1 - 1 / ZOOM) / 2  # 확대했을 때 가운데에서 움직일 수 있는 폭
    if motion == "zoom_in":
        scale = 1 + (ZOOM - 1) * e
    elif motion == "zoom_out":
        scale = ZOOM - (ZOOM - 1) * e
    elif motion == "pan_left":
        scale, cx = ZOOM, 0.5 + slack * (1 - 2 * e)
    elif motion == "pan_right":
        scale, cx = ZOOM, 0.5 - slack * (1 - 2 * e)
    elif motion == "pan_up":
        scale, cy = ZOOM, 0.5 + slack * (1 - 2 * e)
    elif motion == "shake":
        scale = 1.06
        k = max(0.0, 1 - t_ms / 700)  # 처음 0.7초만 흔들고 잦아든다
        cx += 0.012 * k * math.sin(t_ms / 28)
        cy += 0.009 * k * math.cos(t_ms / 23)
    w = size / scale
    return (cx * size - w / 2, cy * size - w / 2, cx * size + w / 2, cy * size + w / 2)


def _wrap(text: str, font: ImageFont.FreeTypeFont, max_w: int) -> list[str]:
    lines, cur = [], ""
    for word in text.split(" "):
        trial = f"{cur} {word}".strip()
        if font.getlength(trial) <= max_w:
            cur = trial
            continue
        if cur:
            lines.append(cur)
        cur = ""
        for ch in word:  # 한 낱말이 한 줄보다 길면 글자 단위로
            if font.getlength(cur + ch) > max_w:
                lines.append(cur)
                cur = ""
            cur += ch
    if cur:
        lines.append(cur)
    if len(lines) == 2 and " " in text:  # 두 줄이면 길이가 비슷하게 다시 나눈다("…치는 / 거야"처럼 한 낱말만 남지 않게)
        words = text.split(" ")
        cuts = [(" ".join(words[:k]), " ".join(words[k:])) for k in range(1, len(words))]
        fit = [c for c in cuts if max(font.getlength(c[0]), font.getlength(c[1])) <= max_w]
        if fit:
            lines = list(min(fit, key=lambda c: abs(font.getlength(c[0]) - font.getlength(c[1]))))
    return lines


class _Subtitles:
    """대사마다 자막 그림(RGBA)을 미리 만들어 두고, 시각에 맞는 것을 프레임에 얹는다."""

    def __init__(self, sb: dict, W: int, H: int):
        self.font = ImageFont.truetype(str(FONT), FONT_SIZE)
        self.name_font = ImageFont.truetype(str(FONT), NAME_SIZE)
        names = {c["key"]: c.get("name") for c in sb["characters"]}
        order = {c["key"]: i for i, c in enumerate(sb["characters"])}
        top = ART_TOP + W + SUB_TOP_GAP
        self.items = []
        for s in sb["scenes"]:
            for d in s.get("lines") or []:
                color = NAME_COLORS[order.get(d["speaker"], 0) % len(NAME_COLORS)]
                img = self._draw(d["text"], names.get(d["speaker"]), color, W, H - top)
                self.items.append((d["startMs"], d["endMs"], img, top))

    def _draw(self, text: str, name: str | None, color: str, W: int, H: int) -> Image.Image:
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        y = 0
        if name:
            d.text((W / 2, y), name, font=self.name_font, fill=color, anchor="ma",
                   stroke_width=4, stroke_fill="black")
            y += NAME_SIZE + 14
        for line in _wrap(text, self.font, W - 80)[:3]:
            d.text((W / 2, y), line, font=self.font, fill="white", anchor="ma",
                   stroke_width=5, stroke_fill="black")
            y += FONT_SIZE + 12
        return img

    def put(self, frame: Image.Image, t_ms: float) -> None:
        for start, end, img, top in self.items:
            if start - 80 <= t_ms < end + 150:  # 소리보다 살짝 먼저 뜨고 조금 더 남는다
                frame.paste(img, (0, top), img)
                return


class _Scene:
    def __init__(self, scene: dict, image: Image.Image, W: int, H: int, lip: dict | None = None, fps: int = 30):
        self.s, self.W = scene, W
        self.lip, self.fps, self._cache = lip, fps, {}
        self.img = image.convert("RGB")
        if self.img.width != self.img.height:
            side = min(self.img.size)
            self.img = self.img.crop(((self.img.width - side) // 2, (self.img.height - side) // 2,
                                      (self.img.width + side) // 2, (self.img.height + side) // 2))
        bg = self.img.resize((H, H), Image.BILINEAR).crop(((H - W) // 2, 0, (H + W) // 2, H))
        bg = bg.filter(ImageFilter.GaussianBlur(24))
        self.bg = Image.eval(bg, lambda v: int(v * 0.42))

    def _state(self, n: int) -> tuple:
        mouths = self.lip["frames"].get(n)
        eyes = (self.lip.get("eye_frames") or {}).get(n)
        blink = self.lip["blink"] is not None and n in self.lip["blinks"]
        return (blink, tuple(sorted(mouths.items())) if mouths else None, tuple(sorted(eyes.items())) if eyes else None)

    def _patch(self, kind: str, face: int, name: str):
        """바꿔 넣을 조각과, 원래 그림과 달라진 자리만 덮는 마스크(눈 조각과 입 조각의 상자가 겹쳐도 서로 지우지 않게)."""
        key = ("patch", kind, face, name)
        if key not in self._cache:
            lp = self.lip
            box = (lp["eye_boxes"] if kind == "eye" else lp["boxes"])[face]
            crop = (lp["eye_variants"] if kind == "eye" else lp["variants"])[face][name].crop(box)
            diff = ImageChops.difference(crop, self.img.crop(box)).convert("L").point(lambda v: 255 if v > 3 else 0)
            self._cache[key] = (crop, diff.filter(ImageFilter.MaxFilter(5)), box)
        return self._cache[key]

    def _compose(self, key: tuple) -> Image.Image:
        """(눈 감음, 얼굴별 입모양, 얼굴별 눈) 조합의 장면 그림. 조합마다 한 번만 만든다."""
        if key not in self._cache:
            blink, mouths, eyes = key
            img = self.lip["blink"] if blink else self.img
            if mouths or eyes:
                img = img.copy()
                for kind, items in (("eye", eyes), ("mouth", mouths)):
                    for face, name in items or ():
                        crop, mask, box = self._patch(kind, face, name)
                        img.paste(crop, box[:2], mask)
            self._cache[key] = img
        return self._cache[key]

    def _art(self, t_ms: float) -> Image.Image:
        """이 시각의 장면 그림. 립싱크가 있으면 말하는 얼굴의 입 둘레와 감은 눈을 바꿔 넣고, 고개를 조금 움직인다."""
        if not self.lip:
            return self.img
        n = round(t_ms / 1000 * self.fps)
        key = self._state(n)
        img = self._compose(key)
        if self.lip.get("blend"):  # 입모양이 바뀌는 첫 프레임은 앞 모양과 반씩 섞는다(딱딱 끊기지 않게)
            prev = self._state(n - 1)
            if prev != key:
                bk = ("blend", prev, key)
                if bk not in self._cache:
                    self._cache[bk] = Image.blend(self._compose(prev), img, 0.5)
                img = self._cache[bk]
        moves = (self.lip.get("moves") or {}).get(n)
        if moves:
            from .lipsync import move_heads
            img = move_heads(img, self.lip["heads"], moves)
        return img

    def frame(self, t_ms: float) -> Image.Image:
        s = self.s
        p = (t_ms - s["startMs"]) / max(1, s["endMs"] - s["startMs"])
        box = _crop_box(s["motion"], p, t_ms - s["startMs"], self.img.width)
        art = self._art(t_ms).resize((self.W, self.W), Image.BILINEAR, box=box)
        out = self.bg.copy()
        out.paste(art, (0, ART_TOP))
        return out


def _blend(kind: str, a: np.ndarray, b: np.ndarray, q: float) -> np.ndarray:
    """전환 효과. a는 앞 장면, b는 뒤 장면 프레임(uint8 HxWx3), q는 0→1."""
    H, W, _ = a.shape
    e = _ease(q)
    if kind == "slide_left":
        x = int(W * e)
        return np.concatenate([a[:, x:], b[:, :x]], axis=1) if x else a
    if kind == "slide_up":
        y = int(H * e)
        return np.concatenate([a[y:], b[:y]], axis=0) if y else a
    if kind == "flash":
        white = np.full_like(a, 255)
        src, k = (a, q * 2) if q < 0.5 else (b, (1 - q) * 2)
        return (src * (1 - k) + white * k).astype(np.uint8)
    if kind == "zoom":  # 앞 장면이 커지면서 뒤 장면으로 녹아든다
        z = 1 + 0.35 * e
        w, h = int(W / z), int(H / z)
        x0, y0 = (W - w) // 2, (H - h) // 2
        a = np.asarray(Image.fromarray(a[y0:y0 + h, x0:x0 + w]).resize((W, H), Image.BILINEAR))
    return (a * (1 - e) + b * e).astype(np.uint8)  # fade, zoom


def render(sb: dict, images: list[Image.Image], audio_path: str | Path, out_path: str | Path,
           thumbnail_path: str | Path | None = None, lips: list[dict] | None = None) -> dict:
    """mp4를 만들고 {durationMs, frames, bytes}를 돌려준다. lips는 lipsync.plan()의 결과(장면 순서)."""
    scenes = sb["scenes"]
    if len(images) != len(scenes):
        raise ValueError(f"장면 {len(scenes)}개인데 그림은 {len(images)}장")
    W, H, fps = sb["size"]["width"], sb["size"]["height"], sb["size"]["fps"]
    if lips is not None and len(lips) != len(scenes):
        raise ValueError(f"장면 {len(scenes)}개인데 립싱크 계획은 {len(lips)}개")
    views = [_Scene(s, im, W, H, lips[i] if lips else None, fps) for i, (s, im) in enumerate(zip(scenes, images))]
    subs = _Subtitles(sb, W, H)
    total = sb["durationMs"]
    n_frames = math.ceil(total / 1000 * fps)
    half = TRANSITION_MS / 2

    cmd = [_ffmpeg(), "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
           "-i", str(audio_path),
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", "-t", f"{total / 1000:.3f}", str(out_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        idx = 0
        for f in range(n_frames):
            t = f * 1000 / fps
            while idx < len(scenes) - 1 and t >= scenes[idx]["endMs"]:
                idx += 1
            frame = views[idx].frame(t)
            # 장면 경계 앞뒤 half 동안 이웃 장면과 섞는다. 효과는 뒤 장면의 transition
            if idx > 0 and t - scenes[idx]["startMs"] < half and scenes[idx]["transition"] != "cut":
                q = 0.5 + (t - scenes[idx]["startMs"]) / TRANSITION_MS
                frame = Image.fromarray(_blend(scenes[idx]["transition"], np.asarray(views[idx - 1].frame(t)),
                                               np.asarray(frame), q))
            elif idx < len(scenes) - 1 and scenes[idx]["endMs"] - t <= half \
                    and scenes[idx + 1]["transition"] != "cut":
                q = 0.5 - (scenes[idx]["endMs"] - t) / TRANSITION_MS
                frame = Image.fromarray(_blend(scenes[idx + 1]["transition"], np.asarray(frame),
                                               np.asarray(views[idx + 1].frame(t)), q))
            if t < FADE_IN_MS:
                frame = Image.eval(frame, lambda v, k=t / FADE_IN_MS: int(v * k))
            subs.put(frame, t)
            if f == int(fps * 0.5) and thumbnail_path:
                frame.save(thumbnail_path, quality=85)
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        err = proc.stderr.read().decode(errors="replace")
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg 실패: {err[-500:]}")
    finally:
        if proc.poll() is None:
            proc.kill()
    return {"durationMs": total, "frames": n_frames, "bytes": Path(out_path).stat().st_size}
