"""빈 말풍선 찾기 + 한글 대사 넣기.

FLUX가 그린 컷 이미지에서 빈 말풍선(검은 테두리로 둘러싸인 흰 영역)을 찾고,
대사를 한글 폰트로 말풍선 안에 맞춰 넣는다. numpy와 Pillow만 쓴다(Jetson에서도 동작).

흐름:
    find_bubbles(image) -> [Bubble, ...]  (왼쪽→오른쪽, 위→아래 순)
    draw_dialogue(image, bubbles, ["대사1", "대사2"], font_path) -> 이미지
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

DEFAULT_FONT = Path.home() / ".cache/lolog/fonts/Jua-Regular.ttf"


@dataclass
class Bubble:
    x0: int
    y0: int
    x1: int
    y1: int
    area: int  # 원본 해상도 기준 흰 픽셀 수
    fill: float  # area / bbox 넓이. 타원이면 약 0.785

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    def text_box(self, margin: float = 0.72) -> tuple[int, int, int, int]:
        """타원 안에 들어가는 사각형. 내접 사각형은 bbox의 1/√2(≈0.707)배."""
        w, h = (self.x1 - self.x0) * margin, (self.y1 - self.y0) * margin
        return int(self.cx - w / 2), int(self.cy - h / 2), int(self.cx + w / 2), int(self.cy + h / 2)


def _label(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """4-연결 연결 요소 라벨링 (scipy 없이)."""
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    n = 0
    for sy, sx in zip(*np.nonzero(mask)):
        if labels[sy, sx]:
            continue
        n += 1
        labels[sy, sx] = n
        q = deque([(sy, sx)])
        while q:
            y, x = q.popleft()
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not labels[ny, nx]:
                    labels[ny, nx] = n
                    q.append((ny, nx))
    return labels, n


def _ellipse_iou(region: np.ndarray) -> float:
    """영역과, 그 bbox에 내접하는 타원의 IoU. 말풍선(꼬리 포함)은 높고 사각형·불규칙 영역은 낮다."""
    h, w = region.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2, (w - 1) / 2
    ellipse = ((yy - cy) / (h / 2)) ** 2 + ((xx - cx) / (w / 2)) ** 2 <= 1
    inter = np.logical_and(region, ellipse).sum()
    union = np.logical_or(region, ellipse).sum()
    return float(inter / union) if union else 0.0


def find_bubbles(
    image: Image.Image,
    *,
    work: int = 256,
    white: int = 235,
    min_area_ratio: float = 0.015,
    min_fill: float = 0.6,
    max_border_touch: float = 0.25,
    min_ellipse_iou: float = 0.85,
) -> list[Bubble]:
    """빈 말풍선 후보를 찾는다.

    - 이미지를 work×work로 줄여 밝은 픽셀(white 이상)을 연결 요소로 묶는다.
    - 너무 작은 것(얼굴·옷의 흰 부분), 타원답지 않은 것(fill이 낮음),
      이미지 가장자리에 많이 닿는 것(바깥 흰 배경)은 버린다.
    """
    W, H = image.size
    small = np.asarray(image.convert("L").resize((work, work), Image.BILINEAR))
    mask = small >= white
    labels, n = _label(mask)

    sx, sy = W / work, H / work
    out: list[Bubble] = []
    for k in range(1, n + 1):
        ys, xs = np.nonzero(labels == k)
        area = len(xs)
        if area < min_area_ratio * work * work:
            continue
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        bw, bh = x1 - x0 + 1, y1 - y0 + 1
        fill = area / (bw * bh)
        if fill < min_fill or bw / bh > 3.5 or bh / bw > 3.5:
            continue
        if _ellipse_iou(labels[y0 : y1 + 1, x0 : x1 + 1] == k) < min_ellipse_iou:
            continue  # 옷·냉장고 문처럼 타원이 아닌 흰 영역 제외
        perim = 2 * (bw + bh)
        on_edge = (xs == 0).sum() + (xs == work - 1).sum() + (ys == 0).sum() + (ys == work - 1).sum()
        if on_edge / perim > max_border_touch:
            continue
        out.append(Bubble(int(x0 * sx), int(y0 * sy), int((x1 + 1) * sx), int((y1 + 1) * sy),
                          int(area * sx * sy), float(fill)))
    out.sort(key=lambda b: (round(b.cy / (H / 4)), b.cx))  # 대략 같은 줄이면 왼쪽부터
    return out


def _wrap(text: str, font: ImageFont.FreeTypeFont, max_w: int) -> list[str]:
    """띄어쓰기 기준으로 줄바꿈하고, 한 단어가 너무 길면 글자 단위로 자른다."""
    lines: list[str] = []
    cur = ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if font.getlength(trial) <= max_w:
            cur = trial
            continue
        if cur:
            lines.append(cur)
        cur = ""
        for ch in word:
            if font.getlength(cur + ch) <= max_w:
                cur += ch
            else:
                lines.append(cur)
                cur = ch
    if cur:
        lines.append(cur)
    return lines


def fit_text(text: str, box: tuple[int, int, int, int], font_path: Path,
             max_size: int = 72, min_size: int = 14, spacing: float = 1.15):
    """box 안에 들어가는 가장 큰 글자 크기와 줄 목록을 찾는다."""
    bw, bh = box[2] - box[0], box[3] - box[1]
    for size in range(max_size, min_size - 1, -2):
        font = ImageFont.truetype(str(font_path), size)
        lines = _wrap(text, font, bw)
        line_h = size * spacing
        if len(lines) * line_h <= bh and all(font.getlength(l) <= bw for l in lines):
            return font, lines, line_h
    font = ImageFont.truetype(str(font_path), min_size)
    return font, _wrap(text, font, bw), min_size * spacing


def draw_dialogue(image: Image.Image, bubbles: list[Bubble], lines_per_bubble: list[str],
                  font_path: Path = DEFAULT_FONT, color: str = "black") -> Image.Image:
    """말풍선마다 대사를 가운데 정렬로 넣는다. 말풍선 수와 대사 수가 다르면 앞에서부터 짝짓는다."""
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for bubble, text in zip(bubbles, lines_per_bubble):
        box = bubble.text_box()
        font, lines, line_h = fit_text(text, box, font_path)
        total_h = len(lines) * line_h
        y = bubble.cy - total_h / 2
        for line in lines:
            draw.text((bubble.cx, y + line_h / 2), line, font=font, fill=color, anchor="mm")
            y += line_h
    return out
