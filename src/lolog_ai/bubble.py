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


# ── 말풍선을 코드로 그리기 (팀 명세 TR-07: 그림 모델은 글자·말풍선 없이 그림) ──

POSITION_X = {"left": 0.25, "center": 0.5, "right": 0.75}


@dataclass
class Placed:
    """코드로 그린 말풍선 하나의 자리."""
    x0: int
    y0: int
    x1: int
    y1: int
    speaker_x: int
    cost: float
    target: tuple[int, int] | None = None  # 꼬리가 향할 곳(화자 얼굴). 없으면 화자 위치 비율로


def _ambiguity(box: tuple, face: tuple | None, others: list[tuple]) -> float:
    """말풍선이 화자보다 다른 인물 얼굴에 더 가까우면 벌점. 거리는 말풍선 아래 가운데 → 얼굴 가운데."""
    if not face or not others:
        return 0.0
    bx, by = (box[0] + box[2]) / 2, box[3]

    def dist(f: tuple) -> float:
        return ((bx - (f[0] + f[2]) / 2) ** 2 + (by - (f[1] + f[3]) / 2) ** 2) ** 0.5

    return 2.0 if min(dist(f) for f in others) < dist(face) else 0.0


def _overlap(a: tuple, b: tuple) -> float:
    """a와 b가 겹치는 넓이 / b의 넓이."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(w, 0) * max(h, 0) / max(1, (b[2] - b[0]) * (b[3] - b[1]))


def _busy_map(image: Image.Image) -> np.ndarray:
    """그림이 복잡한 곳(선·경계가 많은 곳)일수록 큰 값. 말풍선이 인물을 가리지 않게 피할 때 쓴다."""
    from PIL import ImageFilter
    g = image.convert("L").filter(ImageFilter.FIND_EDGES).filter(ImageFilter.BoxBlur(6))
    return np.asarray(g, dtype=np.float32) / 255.0


def layout_text(text: str, font: ImageFont.FreeTypeFont, max_w: int, spacing: float = 1.2,
                target_ratio: float = 1.8):
    """고정 글자 크기로 줄바꿈한다. 줄 너비를 바꿔 가며 글 상자가 target_ratio(가로/세로)에
    가장 가까운 것을 골라 말풍선이 너무 납작하거나 길쭉하지 않게 한다."""
    best = None
    for w_try in range(max_w, font.size * 3 - 1, -font.size // 2):
        lines = _wrap(text, font, w_try)
        w = int(max(font.getlength(l) for l in lines))
        h = int(len(lines) * font.size * spacing)
        score = abs(w / h - target_ratio)
        if best is None or score < best[0]:
            best = (score, lines, w, h)
    return best[1], best[2], best[3]


def bubble_size(text: str, font: ImageFont.FreeTypeFont, max_w: int) -> tuple[int, int]:
    _, tw, th = layout_text(text, font, max_w)
    return int(tw * 1.42 + 36), int(th * 1.42 + 30)  # 글 상자를 감싸는 타원


def place_bubbles(image: Image.Image, dialogue: list[dict], positions: dict[str, str],
                  font_path: Path = DEFAULT_FONT, font_size: int = 32,
                  max_text_w: float = 0.42, margin: int = 12,
                  faces: list[tuple[int, int, int, int]] | None = None,
                  speaker_faces: dict[str, tuple[int, int, int, int]] | None = None) -> list[Placed]:
    """대사마다 말풍선 자리를 고른다.

    - faces(얼굴 상자)가 있으면 얼굴을 가리는 자리는 피한다(조금 넓혀서 본다).
    - speaker_faces(화자 key → 얼굴)가 있으면 화자 얼굴 가까이에 두고 꼬리를 얼굴로 향하게 한다.

    - 말풍선 크기는 대사 길이로 정한다(모든 컷에 같은 글자 크기).
    - 화자 위치(left/center/right) 위쪽에서, 그림이 덜 복잡하고 위쪽인 곳을 고른다.
    - 읽는 순서(왼쪽→오른쪽, 위→아래)를 지킨다: 다음 말풍선은 앞 말풍선보다 오른쪽이거나 아래.
    - 말풍선끼리 겹치지 않는다.
    """
    W, H = image.size
    busy = _busy_map(image)
    font = ImageFont.truetype(str(font_path), font_size)
    placed: list[Placed] = []
    pad = [(f, (f[2] - f[0]) * 0.15, (f[3] - f[1]) * 0.15) for f in faces or []]
    keep_out = [(f[0] - px, f[1] - py, f[2] + px, f[3] + py) for f, px, py in pad]
    for d in dialogue:
        bw, bh = bubble_size(d["text"], font, int(W * max_text_w))
        face = (speaker_faces or {}).get(d["speaker"])
        others = [f for k, f in (speaker_faces or {}).items() if k != d["speaker"]]
        # 말하지 않은 인물의 몸: 얼굴 아래로 얼굴 너비의 2배 정도
        other_bodies = [(f[0] - (f[2] - f[0]) // 2, f[3], f[2] + (f[2] - f[0]) // 2, H) for f in others]
        sx = (face[0] + face[2]) // 2 if face else int(POSITION_X[positions[d["speaker"]]] * W)
        best = None
        for y0 in range(margin, int(H * 0.7) - bh + 1, 8):
            for x0 in range(margin, W - bw - margin + 1, 8):
                x1, y1 = x0 + bw, y0 + bh
                if any(x0 < p.x1 + 6 and p.x0 - 6 < x1 and y0 < p.y1 + 6 and p.y0 - 6 < y1 for p in placed):
                    continue
                if placed:
                    prev = placed[-1]
                    if not (x0 >= (prev.x0 + prev.x1) / 2 and y0 >= prev.y0 - 20) and y0 < prev.y1 - 10:
                        continue  # 읽는 순서가 뒤집히는 자리
                cost = (busy[y0:y1, x0:x1].mean() * 4  # 인물·물건을 덜 가리게
                        + abs((x0 + x1) / 2 - sx) / W  # 화자 머리 위에 가깝게
                        + y0 / H * 0.5  # 만화처럼 위쪽에
                        + sum(_overlap((x0, y0, x1, y1), k) for k in keep_out) * 20  # 얼굴 가림
                        + _ambiguity((x0, y0, x1, y1), face, others)  # 다른 사람 말처럼 보임
                        + sum(_overlap(b, (x0, y0, x1, y1)) for b in other_bodies) * 4)  # 남의 몸 위
                if best is None or cost < best.cost:
                    best = Placed(x0, y0, x1, y1, sx, cost)
        if best is None:  # 자리가 없으면 맨 위 가운데에 겹쳐서라도 넣는다
            x0 = max(margin, min(W - bw - margin, sx - bw // 2))
            best = Placed(x0, margin, x0 + bw, margin + bh, sx, float("inf"))
        if face:
            fcx, fcy = (face[0] + face[2]) // 2, (face[1] + face[3]) // 2
            best.target = (fcx, face[1]) if best.y1 <= face[1] else (fcx, fcy)
        placed.append(best)
    return placed


def draw_bubbles(image: Image.Image, dialogue: list[dict], placed: list[Placed],
                 font_path: Path = DEFAULT_FONT, font_size: int = 32, max_text_w: float = 0.42,
                 stroke: int = 3) -> Image.Image:
    """흰 타원 말풍선과 화자 쪽 꼬리를 그리고 대사를 넣는다.
    타원과 꼬리를 한 마스크로 합친 뒤 바깥 테두리만 검게 칠해 이음매 없이 그린다."""
    from PIL import ImageFilter
    out = image.convert("RGB").copy()
    W, H = out.size
    font = ImageFont.truetype(str(font_path), font_size)
    for d, p in zip(dialogue, placed):
        bw, bh = p.x1 - p.x0, p.y1 - p.y0
        cx, cy = (p.x0 + p.x1) / 2, (p.y0 + p.y1) / 2
        base_x = min(max(p.speaker_x, p.x0 + bw * 0.3), p.x0 + bw * 0.7)
        base_y = cy + bh * 0.3
        if p.target:  # 화자 얼굴 쪽으로, 얼굴에 닿기 전에 멈춤
            tx, ty = p.target
            dx, dy = tx - base_x, ty - base_y
            dist = max(1.0, (dx * dx + dy * dy) ** 0.5)
            reach = min(dist - 12, (p.y1 - base_y) + max(36, H * 0.08))
            tip = (base_x + dx / dist * reach, base_y + dy / dist * reach)
        else:
            tip = (base_x + (p.speaker_x - base_x) * 0.5, min(H - 6, p.y1 + max(28, H * 0.06)))
        half = max(10, bw * 0.06)
        mask = Image.new("L", (W, H), 0)
        md = ImageDraw.Draw(mask)
        md.ellipse((p.x0, p.y0, p.x1, p.y1), fill=255)
        md.polygon([(base_x - half, base_y), (base_x + half, base_y), tip], fill=255)
        ring = mask.filter(ImageFilter.MaxFilter(stroke * 2 + 1))
        out.paste((0, 0, 0), mask=ring)
        out.paste((255, 255, 255), mask=mask)
        draw = ImageDraw.Draw(out)
        lines, _, th = layout_text(d["text"], font, int(W * max_text_w))
        line_h = th / len(lines)
        y = cy - th / 2
        for line in lines:
            draw.text((cx, y + line_h / 2), line, font=font, fill="black", anchor="mm")
            y += line_h
    return out
