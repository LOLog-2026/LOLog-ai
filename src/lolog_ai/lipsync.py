"""카드 영상 립싱크 (#33 후속). 말하는 사람의 입모양을 대사 소리에 맞춰 바꾸고 눈을 깜빡인다.

입모양 5종(X 다문 입, M 살짝, A 크게, E 옆으로, O 동그랗게)은 FLUX가 그린다. 장면 그림에서 말하는 얼굴 둘레를
정사각형으로 잘라 384로 넣고 "입만 바꿔 다시 그려라"고 시킨다. FLUX는 그래도 얼굴 위치·안경 색·배경을 조금씩
바꾸므로, 두 그림의 선(눈·안경·머리카락)을 맞춰 옮긴 뒤 입 둘레만 원래 장면에 부드럽게 섞는다(paste_mouth).
붙인 결과가 어긋나면(옆얼굴에서 머리카락이 붙는 등) 그 모양은 코드로 그린 입(draw_mouth)으로 대신한다.

    mouths = make_mouths(sb, images, edit)          # 2호 draw.py가 FLUX를 올린 동안 부른다. edit(crop, 모양) → 그림
    lips = plan(sb, images, wav_bytes, mouths)      # video_render.render(..., lips=lips)

실험 기록(2호, 2026-11-13): 입모양 한 장에 얼굴 256은 8초, 384는 12초, 512는 20초, 장면 전체(768)는 42초.
얼굴 찾기는 MediaPipe FaceLandmarker(CPU). 만화 그림에서도 대부분 잡히고, 못 찾은 장면은 지금처럼 움직임만 나온다.

환경 변수
  LIPSYNC             0이면 끈다(기본 1)
  LIPSYNC_FACE_MODEL  face_landmarker.task 경로(기본 ~/.cache/lolog/models/mediapipe/face_landmarker.task)
  LIPSYNC_MAX_FACES   영상 하나에서 FLUX로 입을 그릴 얼굴 수 한도(기본 8, 얼굴 하나에 약 1분)
  LIPSYNC_BLINK       1이면 코드로 그린 눈 깜빡임을 넣는다(기본 0. 안경·앞머리가 있는 얼굴에서 자국이 남는다)
"""

from __future__ import annotations

import io
import logging
import math
import os
import re
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

try:  # 2호(comic) 워커에만 깐다. 없으면 립싱크 없이 영상을 만든다
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

log = logging.getLogger(__name__)

ENABLED = os.environ.get("LIPSYNC", "1") != "0"
FACE_MODEL = Path(os.environ.get("LIPSYNC_FACE_MODEL",
                                 Path.home() / ".cache/lolog/models/mediapipe/face_landmarker.task"))
MAX_FACES = int(os.environ.get("LIPSYNC_MAX_FACES", "8"))
BLINK = os.environ.get("LIPSYNC_BLINK", "0") == "1"
BLEND = os.environ.get("LIPSYNC_BLEND", "1") == "1"          # 입모양이 바뀌는 첫 프레임을 앞 모양과 섞기
HEAD = float(os.environ.get("LIPSYNC_HEAD", "1"))            # 말할 때 고개 움직임 세기. 0이면 끔
NOD = float(os.environ.get("LIPSYNC_NOD", "1"))              # 듣는 사람 끄덕임 세기. 0이면 끔
DEBUG_DIR = os.environ.get("LIPSYNC_DEBUG_DIR")  # 시험할 때만. 서비스에서는 그림을 디스크에 남기지 않는다
MOUTH_SIZE = 384       # FLUX에 넣는 얼굴 조각 크기
CROP_FACES = 2.1       # 얼굴 조각 한 변 = 얼굴 너비의 몇 배

FLUX_SHAPES = {
    "X": "mouth closed, lips together in a small neutral line",
    "M": "mouth slightly open, as if speaking softly",
    "A": "mouth wide open, as if saying 'ah' loudly",
    "O": "lips rounded into a small 'o' shape, as if saying 'oh'",
    "E": "mouth open wide sideways showing upper teeth, as if saying 'ee'",
}


def mouth_prompts(sb: dict | None = None) -> dict[str, str]:
    """모양 → FLUX 편집 글. 얼굴 조각(image 1)에는 한 사람만 있으므로 누구인지는 쓰지 않는다.

    수염 난 인물이 콘티에 있으면 "X+"처럼 +가 붙은 글도 함께 준다(수염을 지키라는 말을 덧붙인 것).
    수염 말은 수염 난 얼굴에만 쓴다. 모든 얼굴에 쓰면 수염 없는 얼굴에 수염을 그린다(7회차).
    """
    from .render import VIDEO_STYLE

    def text(d: str, keep: str) -> str:
        return (f"{VIDEO_STYLE}. Redraw image 1 exactly as it is: same person, same face, same eyes, same hair, "
                f"same clothes, same background, same framing.{KEEP_LIGHT} Change only the mouth of the person: {d}.{keep} "
                f"No text.")

    out = {k: text(d, "") for k, d in FLUX_SHAPES.items()}
    looks = [c.get("appearance", "") for c in (sb or {}).get("characters") or []] + \
            [e.get("appearance", "") for e in (sb or {}).get("extras") or []]
    for hair in sorted({hair_of(a) for a in looks} - {""}) or ([] if sb is not None else ["moustache"]):
        out.update({f"{k}+{hair}": text(d, KEEP_HAIR.format(hair=hair)) for k, d in FLUX_SHAPES.items()})
    return out


def hair_of(appearance: str) -> str:
    """외모 글에 나온 수염 낱말("moustache", "beard", "moustache and beard"). 없으면 빈 문자열."""
    found = {HAIR_WORD[m.group(1).lower()] for m in HAIR_RE.finditer(appearance or "")}
    return " and ".join(sorted(found))


def eye_prompts() -> dict[str, str]:
    """눈 편집 글. B: 감은 눈(깜빡임)."""
    from .render import VIDEO_STYLE
    return {"B": (f"{VIDEO_STYLE}. Redraw image 1 exactly as it is: same person, same face, same hair, same clothes, "
                  f"same background, same framing.{KEEP_LIGHT} Change only the eyes of the person: both eyes gently "
                  f"closed, as if blinking. Keep the mouth the same. No text.")}


# 밤·실내 장면을 FLUX가 밝은 낮 색으로 다시 그리지 않게(카드 D). 10회차에서 견줘 보는 중이라 환경 변수로 끌 수 있다
KEEP_LIGHT = os.environ.get("LIPSYNC_KEEP_LIGHT", "")   # 넣어 봤지만 FLUX는 그래도 색을 바꿨다(10회차). 기본은 끔
# 콧수염·턱수염이 있는 얼굴: FLUX가 입을 다시 그리며 수염을 지운다(테스트 카드 T). 외모 글에 수염이 있을 때만 덧붙인다
HAIR_WORD = {"moustache": "moustache", "mustache": "moustache", "whiskers": "moustache", "beard": "beard",
             "bearded": "beard", "goatee": "beard", "stubble": "stubble", "sideburns": "sideburns"}
HAIR_RE = re.compile(r"\b(" + "|".join(HAIR_WORD) + r")\b", re.IGNORECASE)
# 그 인물에게 있는 수염만 말한다. "콧수염·턱수염·주름"을 한꺼번에 말하면 콧수염만 있는 얼굴에 잔수염을 그렸다(테스트 U)
KEEP_HAIR = " Keep the {hair} exactly as in image 1."


# ---------------------------------------------------------------- 얼굴 찾기와 코드로 그리는 입·감은 눈

OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
SHAPES = ["X", "M", "A", "E", "O"]
# (가로 배율, 세로 열림 배율, 윗니 보임, 혀 보임)
SHAPE_PARAM = {
    "X": (0.85, 0.00, False, False),
    "M": (0.80, 0.22, False, False),
    "A": (0.95, 0.62, True, True),
    "E": (1.10, 0.30, True, False),
    "O": (0.58, 0.52, False, True),
}


@dataclass
class Face:
    pts: np.ndarray        # (478, 2) 픽셀 좌표
    box: tuple             # 얼굴 네모 x0, y0, x1, y1
    mouth_c: np.ndarray    # 입 가운데
    mouth_w: float         # 입 가로 길이
    angle: float           # 입꼬리 두 점의 기울기(라디안)
    face_w: float
    approx: bool = False   # 얼굴 점을 못 찾아 검출 상자에 평균 얼굴 틀을 맞춘 것(입·눈 자리는 대략만 안다)


_det = None


def _detector():
    global _det
    if _det is None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        opt = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(FACE_MODEL)), num_faces=4,
                                           min_face_detection_confidence=0.2, min_face_presence_confidence=0.2)
        _det = (mp, vision.FaceLandmarker.create_from_options(opt))
    return _det


def _detect_once(a: np.ndarray):
    mp, det = _detector()
    r = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(a)))
    return [np.array([(p.x * a.shape[1], p.y * a.shape[0]) for p in lm]) for lm in r.face_landmarks]


def _anime_boxes(img: Image.Image) -> list[tuple]:
    try:
        from .faces import DEFAULT_MODEL, detect_faces
        model = Path(os.environ.get("ANIMEFACE_MODEL", DEFAULT_MODEL))
        return detect_faces(img, model=model) if model.exists() else []
    except Exception:  # onnxruntime이 없거나 모델이 없으면 MediaPipe만 쓴다
        log.debug("만화 얼굴 검출기를 못 씀", exc_info=True)
        return []


ANIME_GROW = 1.7   # 검출 상자를 몇 배로 넓혀 얼굴 점을 찾을지(머리카락·턱까지 들어가게)


def find_faces(img: Image.Image, tiles: bool = True) -> list[Face]:
    """얼굴 찾기. 통째로 못 찾은 자리는 겹치게 자른 조각(2배 확대)에서 다시 찾는다."""
    a = np.asarray(img.convert("RGB"))
    H, W = a.shape[:2]
    found = _detect_once(a)
    # 조각 탐색: 작은 얼굴이나 처음에 놓친 얼굴
    step = W // 2
    for y0 in range(0, max(1, H - step) + 1, step // 2) if tiles else ():
        for x0 in range(0, max(1, W - step) + 1, step // 2):
            crop = a[y0:y0 + step, x0:x0 + step]
            big = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            for p in _detect_once(big):
                found.append(p / 2 + [x0, y0])
    # 만화 얼굴 검출기(faces.py, 말풍선에 쓰는 것)가 찾은 상자마다 그 둘레를 키워서 다시 찾는다. MediaPipe는 작은
    # 얼굴·아이 얼굴·웃느라 감은 눈을 자주 놓친다(카드 B: 얼굴 7개 중 4개만 찾음)
    boxes = _anime_boxes(img) if tiles else []
    if tiles:
        for x0, y0, x1, y1 in boxes:
            side = max(x1 - x0, y1 - y0) * ANIME_GROW
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            bx0, by0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
            bx1, by1 = int(min(W, cx + side / 2)), int(min(H, cy + side / 2))
            if bx1 - bx0 < 24 or by1 - by0 < 24:
                continue
            k = 512 / max(bx1 - bx0, by1 - by0)
            big = cv2.resize(a[by0:by1, bx0:bx1], None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
            for p in _detect_once(big):
                found.append(p / k + [bx0, by0])
    faces: list[Face] = []
    for p in found:
        c = p[OUTER].mean(axis=0)
        if any(f.box[0] < c[0] < f.box[2] and f.box[1] < c[1] < f.box[3] for f in faces):
            continue  # 이미 찾은 얼굴(또는 그 얼굴 안에서 잘못 잡힌 것)
        l, r = p[61], p[291]
        fw = float(np.linalg.norm(p[234] - p[454]))
        if fw < 40:
            continue
        x0, y0 = p.min(axis=0)
        x1, y1 = p.max(axis=0)
        new = Face(p, (x0, y0, x1, y1), c, float(np.linalg.norm(r - l)), math.atan2(r[1] - l[1], r[0] - l[0]), fw)
        # 새 얼굴이 더 크고 앞서 찾은 얼굴의 입을 품으면 앞의 것이 잘못 잡힌 것
        faces = [f for f in faces if not (x0 < f.mouth_c[0] < x1 and y0 < f.mouth_c[1] < y1)]
        faces.append(new)
    if boxes:
        def inside(pt, b, grow=0.15):
            gx, gy = (b[2] - b[0]) * grow, (b[3] - b[1]) * grow
            return b[0] - gx < pt[0] < b[2] + gx and b[1] - gy < pt[1] < b[3] + gy
        # 검출기가 얼굴이라고 보지 않은 자리의 얼굴 점은 버린다(배경 무늬를 얼굴로 잡는 일이 있다)
        faces = [f for f in faces if any(inside(f.pts[1], b) for b in boxes)]
        # 검출기는 찾았는데 얼굴 점을 못 찾은 얼굴: 평균 얼굴 틀을 상자에 맞춰 둔다
        for b in boxes:
            if any(inside(f.pts[1], b, 0.0) for f in faces) or (b[2] - b[0]) < 40:
                continue
            side = max(b[2] - b[0], b[3] - b[1])
            p = _template() * side + [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]
            faces.append(Face(p, tuple(float(v) for v in (*p.min(axis=0), *p.max(axis=0))), p[OUTER].mean(axis=0),
                              float(np.linalg.norm(p[291] - p[61])), 0.0, float(np.linalg.norm(p[234] - p[454])),
                              approx=True))
    return sorted(faces, key=lambda f: f.mouth_c[0])


_tmpl = None


def _template() -> np.ndarray:
    """정면 얼굴 세 개의 얼굴 점 평균(검출 상자의 가운데·한 변 기준)."""
    global _tmpl
    if _tmpl is None:
        _tmpl = np.load(Path(__file__).with_name("face_template.npy")).astype(np.float64)
    return _tmpl


def _line_color(a: np.ndarray, f: Face) -> tuple:
    """원래 입 둘레에서 가장 어두운 쪽 색 → 새 입의 선 색."""
    poly = f.pts[OUTER].astype(np.int32)
    m = np.zeros(a.shape[:2], np.uint8)
    cv2.fillPoly(m, [poly], 255)
    px = a[m > 0]
    if len(px) < 10:
        return (40, 25, 25)
    lum = px.mean(axis=1)
    dark = px[lum <= np.percentile(lum, 8)]
    return tuple(int(v) for v in dark.mean(axis=0))


def erase_mouth(a: np.ndarray, f: Face) -> np.ndarray:
    poly = f.pts[OUTER].astype(np.int32)
    m = np.zeros(a.shape[:2], np.uint8)
    cv2.fillPoly(m, [poly], 255)
    k = max(3, int(f.mouth_w * 0.28)) | 1
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    # 얼굴 윤곽 밖(머리카락·배경)으로 번지지 않게 얼굴 안쪽으로만
    hull = np.zeros(a.shape[:2], np.uint8)
    cv2.fillConvexPoly(hull, cv2.convexHull(f.pts.astype(np.int32)), 255)
    e = max(3, int(f.face_w * 0.05)) | 1
    m &= cv2.erode(hull, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (e, e)))
    return cv2.inpaint(a, m, max(3, int(f.mouth_w * 0.15)), cv2.INPAINT_TELEA)


def mouth_curve(f: Face) -> float:
    """원래 입의 표정. 음수면 입꼬리가 올라간 웃는 입, 양수면 내려간 입(입 너비 대비)."""
    c, s = math.cos(-f.angle), math.sin(-f.angle)
    def ry(i):  # 입 기울기를 편 뒤의 세로 좌표
        x, y = f.pts[i] - f.mouth_c
        return x * s + y * c
    corners = (ry(61) + ry(291)) / 2
    mid = (ry(13) + ry(14)) / 2
    return float(np.clip((corners - mid) / max(1.0, f.mouth_w) * 0.9, -0.14, 0.14))


def draw_mouth(base: Image.Image, f: Face, shape: str, line_rgb: tuple, ss: int = 4, curve: float = -0.10) -> Image.Image:
    """base(입을 지운 그림)에 모양 shape의 입을 그린다. ss배로 그려 줄여서 매끄럽게."""
    sw, so, teeth, tongue = SHAPE_PARAM[shape]
    w = f.mouth_w * sw
    h = f.mouth_w * so
    lw = max(1.5, f.face_w * 0.012)
    pad = int(f.mouth_w * 1.2)
    cx, cy = f.mouth_c
    # 위아래 입술 가운데(13, 14) 사이보다 살짝 위를 윗입술 선으로
    top = f.pts[13][1] - (f.pts[13][1] - f.pts[0][1]) * 0.2 if shape != "X" else cy
    S = pad * 2 * ss
    layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    ox, oy = S / 2, S / 2 - (cy - top) * ss  # 윗입술이 top에 오게
    line = (*line_rgb, 255)
    if shape == "X":
        # 살짝 웃는 곡선 한 줄
        xs = np.linspace(-w / 2, w / 2, 24)
        ys = (xs / (w / 2)) ** 2 * (curve * w)
        d.line([(ox + x * ss, S / 2 + y * ss) for x, y in zip(xs, ys)], fill=line, width=int(lw * ss), joint="curve")
    else:
        # 위는 거의 평평, 아래는 둥근 입
        xs = np.linspace(-1, 1, 40)
        if shape == "O":
            upper = [(-math.sqrt(max(0, 1 - x * x)) * 0.45 * h) for x in xs]
            lower = [(math.sqrt(max(0, 1 - x * x)) * 0.55 * h) for x in xs]
        else:
            # 입꼬리 높이를 원래 표정(curve)에 맞춘다: 웃는 입은 꼬리가 올라가고, 시무룩한 입은 내려간다
            bend = (curve + 0.10) * w * 0.9   # 웃는 기본값(-0.10)에서 얼마나 내려갔는지
            upper = [(-0.12 * h * math.sqrt(max(0, 1 - x * x)) + 0.10 * h * x * x + bend * x * x) for x in xs]
            lower = [(h * (1 - x * x) ** 0.75 + bend * x * x) for x in xs]
        poly = [(ox + x * w / 2 * ss, oy + y * ss) for x, y in zip(xs, upper)] + \
               [(ox + x * w / 2 * ss, oy + y * ss) for x, y in zip(xs[::-1], lower[::-1])]
        inside = (95, 30, 38, 255)
        d.polygon(poly, fill=inside)
        if tongue:
            tw, th = w * 0.55, h * 0.45
            ty = oy + (max(lower) - th * 0.55) * ss
            d.ellipse([ox - tw / 2 * ss, ty, ox + tw / 2 * ss, ty + th * ss], fill=(230, 120, 125, 255))
        if teeth:
            th = h * 0.22
            tmask = Image.new("L", (S, S), 0)
            ImageDraw.Draw(tmask).rectangle([0, oy - h * ss, S, oy + th * ss], fill=255)
            pm = Image.new("L", (S, S), 0)
            ImageDraw.Draw(pm).polygon(poly, fill=255)
            white = Image.new("RGBA", (S, S), (250, 250, 250, 255))
            layer.paste(white, (0, 0), Image.fromarray(np.minimum(np.asarray(tmask), np.asarray(pm))))
        # 그 위에 선
        d.line(poly + [poly[0]], fill=line, width=int(lw * ss), joint="curve")
    layer = layer.rotate(-math.degrees(f.angle), resample=Image.BICUBIC, center=(S / 2, S / 2))
    layer = layer.resize((pad * 2, pad * 2), Image.LANCZOS)
    out = base.copy()
    out.paste(layer, (int(cx - pad), int(cy - pad)), layer)
    return out


def drawn_mouths(img: Image.Image, f: Face) -> dict[str, Image.Image]:
    """코드로 그린 입 5종(그 얼굴 입만 바꾼 장면 그림). FLUX 입을 못 쓸 때의 대비책."""
    a = np.asarray(img.convert("RGB"))
    col, cv = _line_color(a, f), mouth_curve(f)
    base = Image.fromarray(erase_mouth(a, f))
    return {s: draw_mouth(base, f, s, col, curve=cv) for s in SHAPES}


R_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
L_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]


def blink_frame(img: Image.Image, faces: list[Face]) -> Image.Image | None:
    """모든 얼굴의 눈을 감긴 그림. 눈을 피부색으로 덮고 감은 눈 선을 긋는다. 눈을 못 그릴 얼굴은 그대로."""
    a = np.asarray(img.convert("RGB")).copy()
    H, W = a.shape[:2]
    ss = 4
    layer = Image.new("RGBA", (W * ss, H * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    any_eye = False
    for f in faces:
        col = _line_color(a, f)
        skin = np.median(a[int(np.clip(f.pts[205][1], 0, H - 1)) - 2:int(np.clip(f.pts[205][1], 0, H - 1)) + 3,
                           int(np.clip(f.pts[205][0], 0, W - 1)) - 2:int(np.clip(f.pts[205][0], 0, W - 1)) + 3]
                         .reshape(-1, 3), axis=0)
        for idx in (R_EYE, L_EYE):
            p = f.pts[idx]
            c = p.mean(axis=0)
            ew = float(np.linalg.norm(p[0] - p[8]))
            if ew < 6:
                continue
            # 만화 눈은 랜드마크보다 크다: 눈 자리에서 시작해 이어진 어두운 덩어리(눈동자·속눈썹)와 흰자만 덮는다.
            # 안경테·앞머리처럼 눈 밖으로 길게 이어진 덩어리는 눈 가까운 부분만 지운다
            r = ew * 0.8
            x0, y0, x1, y1 = max(0, int(c[0] - r)), max(0, int(c[1] - r)), min(W, int(c[0] + r)), min(H, int(c[1] + r))
            patch = a[y0:y1, x0:x1]
            if patch.size == 0:
                continue
            cand = ((patch.mean(axis=2) < 110) | (patch.min(axis=2) > 235)).astype(np.uint8)
            seed = np.zeros(cand.shape, np.uint8)
            cv2.fillPoly(seed, [(p - [x0, y0]).astype(np.int32)], 1)
            seed = cv2.dilate(seed, np.ones((3, 3), np.uint8), iterations=max(1, int(ew * 0.12)))
            n, lab, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
            m = np.zeros(cand.shape, np.uint8)
            for k in range(1, n):
                comp = lab == k
                if not (comp & (seed > 0)).any():
                    continue
                bw, bh = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
                touches = stats[k, cv2.CC_STAT_LEFT] == 0 or stats[k, cv2.CC_STAT_TOP] == 0 or \
                    stats[k, cv2.CC_STAT_LEFT] + bw >= cand.shape[1] or stats[k, cv2.CC_STAT_TOP] + bh >= cand.shape[0]
                m |= ((comp & (seed > 0)) if touches else comp).astype(np.uint8)
            ys_ = np.flatnonzero(m.any(axis=1))
            if m.sum() < 8 or (ys_[-1] - ys_[0]) < ew * 0.28:
                continue  # 이미 감은 눈(가는 선)이거나 눈을 못 찾음
            m = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=2)
            patch[m > 0] = skin
            any_eye = True
            l, rgt = p[0], p[8]
            xs = np.linspace(0, 1, 20)
            pts = [((l[0] + (rgt[0] - l[0]) * t) * ss, (l[1] + (rgt[1] - l[1]) * t + ew * 0.22 * (1 - (2 * t - 1) ** 2)) * ss)
                   for t in xs]
            d.line(pts, fill=(*col, 255), width=int(max(2.0, f.face_w * 0.016) * ss), joint="curve")
    if not any_eye:
        return None
    out = Image.fromarray(a)
    layer = layer.resize((W, H), Image.LANCZOS)
    out.paste(layer, (0, 0), layer)
    return out


# ---------------------------------------------------------------- 대사 글 + 소리 → 프레임별 입모양

V = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
ONSET = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
CODA = " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"
# 모음 → 모양들(이중모음은 두 모양을 차례로)
VOWEL = {"ㅏ": "A", "ㅑ": "A", "ㅐ": "E", "ㅒ": "E", "ㅓ": "A", "ㅕ": "A", "ㅔ": "E", "ㅖ": "E", "ㅗ": "O", "ㅛ": "O",
         "ㅜ": "O", "ㅠ": "O", "ㅡ": "M", "ㅣ": "E", "ㅘ": "OA", "ㅝ": "OA", "ㅙ": "OE", "ㅞ": "OE", "ㅚ": "OE",
         "ㅟ": "OE", "ㅢ": "ME"}
LIP = set("ㅁㅂㅃㅍ")
DIGIT = dict(zip("0123456789", "영일이삼사오육칠팔구"))


def syllables(text: str) -> list[tuple[bool, str, bool]]:
    """(첫소리가 입술소리?, 모음 모양, 받침이 입술소리?) 목록."""
    out = []
    for ch in text:
        ch = DIGIT.get(ch, ch)
        o = ord(ch) - 0xAC00
        if 0 <= o < 11172:
            on, v, co = ONSET[o // 588], V[(o % 588) // 28], CODA[o % 28]
            out.append((on in LIP, VOWEL[v], co in ("ㅁ", "ㅂ", "ㅍ", "ㅄ", "ㄻ", "ㄿ")))
    return out


def envelope(wav: np.ndarray, sr: int, fps: int, n: int) -> np.ndarray:
    """프레임별 소리 크기(0~1, 그 대사 안에서 정규화)."""
    hop = sr / fps
    e = np.array([np.sqrt(np.mean(wav[int(i * hop):int((i + 1) * hop) + 1] ** 2) + 1e-12) if int(i * hop) < len(wav)
                  else 0.0 for i in range(n)])
    return e / (np.percentile(e, 95) + 1e-9)


def _hold(seq: list[str], k: int) -> list[str]:
    """한 모양을 적어도 k프레임 유지(너무 빨리 바뀌면 떨려 보인다)."""
    out, i = list(seq), 0
    while i < len(out):
        j = i
        while j < len(out) and out[j] == out[i]:
            j += 1
        if j - i < k and i > 0:
            for t in range(i, j):
                out[t] = out[i - 1]
        i = j
    return out


def shapes(text: str, wav: np.ndarray, sr: int, fps: int, method: str = "text", hold: int = 2,
           silence: float = 0.12) -> list[str]:
    n = int(np.ceil(len(wav) / sr * fps))
    env = envelope(wav, sr, fps, n)
    voiced = env > silence
    if method == "amp":
        seq = ["X" if e <= silence else "M" if e < 0.45 else "A" for e in env]
        return _hold(seq, hold)
    if method == "flap":
        seq, k = [], 0
        for v in voiced:
            seq.append(("A" if (k // 3) % 2 == 0 else "M") if v else "X")
            k = k + 1 if v else 0
        return seq
    syl = syllables(text)
    if not syl:
        return shapes(text, wav, sr, fps, "amp", hold, silence)
    idx = np.flatnonzero(voiced)
    seq = ["X"] * n
    if len(idx) == 0:
        return seq
    per = len(idx) / len(syl)
    for k, f in enumerate(idx):
        s = min(len(syl) - 1, int(k / per))
        p = k / per - s                      # 음절 안 진행도 0~1
        lip_on, v, lip_co = syl[s]
        if lip_on and p < 0.25 and per >= 3:
            seq[f] = "X"
        elif lip_co and p > 0.75 and per >= 3:
            seq[f] = "X"
        else:
            shape = v[0] if (len(v) == 1 or p < 0.4) else v[1]
            if shape == "A" and env[f] < 0.35:   # 작게 말하면 덜 벌린다
                shape = "M"
            seq[f] = shape
    return _hold(seq, hold)


OPEN = {"X": 0.0, "M": 0.3, "E": 0.5, "O": 0.6, "A": 1.0}


def _runs(voiced: np.ndarray, min_gap: int) -> list[tuple[int, int]]:
    """소리 나는 구간들(프레임). min_gap보다 짧은 끊김은 이어 붙인다."""
    runs, s = [], None
    for i, v in enumerate(list(voiced) + [False]):
        if v and s is None:
            s = i
        if not v and s is not None:
            if runs and s - runs[-1][1] < min_gap:
                runs[-1] = (runs[-1][0], i)
            else:
                runs.append((s, i))
            s = None
    return runs


def _word_sizes(text: str) -> list[tuple[int, bool]]:
    """낱말(띄어쓰기 단위)마다 (음절 수, 쉼표·마침표 바로 뒤에서 시작하는가)."""
    out = []
    for part in re.split(r"(?<=[,.!?…~])", text):
        for j, w in enumerate(n for n in (len(syllables(w)) for w in part.split()) if n):
            out.append((w, j == 0))
    return out


SYNC_GAP = 0.20      # 이보다 긴 쉼에는 낱말 경계가 온다(초)
SYNC_LEAD = 0.04     # 입모양을 소리보다 이만큼 먼저(초). 자음이 모음 소리보다 앞서 시작한다
SYNC_FINAL = 1.5     # 쉼 앞 마지막 음절은 길게 읽는다


def syllable_spans(text: str, wav: np.ndarray, sr: int, fps: int, silence: float = 0.12) -> list[tuple[int, int]]:
    """음절마다 (시작 프레임, 끝 프레임). 소리의 쉼(SYNC_GAP 이상)마다 낱말 경계가 오게 놓는다.

    쉼 앞까지의 소리 길이 비율만큼 음절을 세고 가장 가까운 낱말 경계로 맞춘다(쉼표·마침표 경계를 조금 더 좋아한다).
    쉼 사이 구간 안에서는 고르게, 구간의 끝 음절만 길게. 예전에는 쉼표·마침표 수와 쉼 수가 다르면 줄 전체를 고르게
    나눠서 낱말이 0.5초 넘게 어긋나는 줄이 있었다(15회차: 음성 인식으로 뽑은 낱말 시각과 견줌).
    """
    n = int(np.ceil(len(wav) / sr * fps))
    voiced = envelope(wav, sr, fps, n) > silence
    words = _word_sizes(text)
    N = sum(w for w, _ in words)
    runs = _runs(voiced, max(2, int(fps * SYNC_GAP)))
    if N == 0 or not runs:
        return []
    ends = np.cumsum([w for w, _ in words])
    bounds = [int(b) for b in ends[:-1]]
    punct = {int(b) for b, (_, p) in zip(ends[:-1], words[1:]) if p}
    lens = np.array([b - a for a, b in runs], float)
    parts, prev = [], 0                      # (구간, 그 구간에 놓을 음절 수)
    for r, run in enumerate(runs):
        if r == len(runs) - 1:
            cut = N
        else:
            target = N * lens[:r + 1].sum() / lens.sum()
            cand = [b for b in bounds if b > prev]
            cut = min(cand, key=lambda b: abs(b - target) - (0.75 if b in punct else 0)) if cand else None
        if cut is None or cut <= prev:       # 낱말 경계가 모자라면 이 쉼은 낱말 안의 쉼으로 보고 다음 구간과 잇는다
            if parts:
                parts[-1] = ((parts[-1][0][0], run[1]), parts[-1][1])
            else:
                parts.append((run, 0))
            continue
        if parts and parts[-1][1] == 0:
            run = (parts.pop()[0][0], run[1])
        parts.append((run, cut - prev))
        prev = cut
    lead = int(round(SYNC_LEAD * fps))
    spans = []
    for (a, b), k in parts:
        if k <= 0:
            continue
        w = np.ones(k)
        w[-1] = SYNC_FINAL
        edge = a + np.concatenate([[0], np.cumsum(w) / w.sum()]) * (b - a)
        for j in range(k):
            f0 = int(round(edge[j]))
            spans.append((max(0, f0 - lead), max(1, max(int(round(edge[j + 1])), f0 + 1) - lead)))
    return spans if len(spans) == N else []


def shapes2(text: str, wav: np.ndarray, sr: int, fps: int, hold: int = 2, silence: float = 0.12) -> list[str]:
    """방법 "text2": 음절을 syllable_spans로 놓고 모음에 맞는 모양을 넣는다. ㅁㅂㅍ에서는 입을 다문다."""
    n = int(np.ceil(len(wav) / sr * fps))
    env = envelope(wav, sr, fps, n)
    syl = syllables(text)
    spans = syllable_spans(text, wav, sr, fps, silence)
    if not syl or len(spans) != len(syl):
        return shapes(text, wav, sr, fps, "amp", hold, silence)
    lead = int(round(SYNC_LEAD * fps))
    seq = ["X"] * n
    for (f0, f1), (lip_on, v, lip_co) in zip(spans, syl):
        for f in range(f0, min(f1, n)):
            p = (f - f0) / max(1, f1 - f0)
            if env[min(n - 1, f + lead)] <= silence:      # 소리는 입보다 lead만큼 늦다
                continue
            if (lip_on and p < 0.25 and f1 - f0 >= 3) or (lip_co and p > 0.75 and f1 - f0 >= 3):
                seq[f] = "X"
            else:
                sh = v[0] if (len(v) == 1 or p < 0.4) else v[1]
                seq[f] = "M" if (sh == "A" and env[min(n - 1, f + lead)] < 0.35) else sh
    return _hold(seq, hold)


def openness_corr(seq: list[str], wav: np.ndarray, sr: int, fps: int) -> float:
    """입 벌림 정도와 소리 크기의 상관(프레임 단위). 높을수록 소리와 입이 같이 움직인다."""
    env = envelope(wav, sr, fps, len(seq))
    o = np.array([OPEN[s] for s in seq])
    return float(np.corrcoef(o, env)[0, 1]) if o.std() > 0 else 0.0


def shapes_mix(text: str, wav: np.ndarray, sr: int, fps: int, hold: int = 2, silence: float = 0.12) -> list[str]:
    """방법 "mix": 벌리는 정도는 소리 크기로, 모양(A/E/O)은 그 시점 음절의 모음으로."""
    n = int(np.ceil(len(wav) / sr * fps))
    env = envelope(wav, sr, fps, n)
    vow = shapes2(text, wav, sr, fps, hold=1, silence=silence)
    seq = []
    for e, v in zip(env, vow):
        if e <= silence:
            seq.append("X")
        elif v == "X":                       # ㅁㅂㅍ: 소리가 작아질 때만 다문다
            seq.append("X" if e < 0.5 else "M")
        elif e < 0.3:
            seq.append("M")
        else:
            seq.append(v)
    return _hold(seq, hold)


# ---------------------------------------------------------------- FLUX가 그린 입을 원래 장면에 붙이기

def square(img: Image.Image) -> Image.Image:
    """video_render._Scene과 같은 가운데 정사각형."""
    img = img.convert("RGB")
    side = min(img.size)
    return img.crop(((img.width - side) // 2, (img.height - side) // 2,
                     (img.width + side) // 2, (img.height + side) // 2))


def face_box(f: Face, W: int, H: int) -> tuple[int, int, int, int]:
    """FLUX에 넣을 얼굴 둘레 정사각형. 그림 밖으로 나가면 안으로 민다."""
    side = int(min(W, H, f.face_w * CROP_FACES))
    cx, cy = f.pts[1]
    x0 = int(min(max(0, cx - side / 2), W - side))
    y0 = int(min(max(0, cy - side / 2), H - side))
    return (x0, y0, x0 + side, y0 + side)


def _register(crop: np.ndarray, redrawn: np.ndarray, keep: np.ndarray) -> tuple[np.ndarray | None, int]:
    """redrawn → crop 좌표로 옮기는 변환(이동·회전·크기)과 맞은 점 수. 얼굴의 선(눈·안경·머리카락·윤곽)에서
    특징점을 찾아 맞춘다. keep은 crop에서 특징점을 쓸 자리(얼굴 둘레, 입은 뺀다).

    처음에는 다시 그린 그림에서 얼굴 점(MediaPipe)을 찾아 맞췄는데, 2호 시험에서 25장 중 15장이 얼굴을 못 찾았다.
    FLUX는 384 얼굴 조각의 구도를 거의 그대로 두므로(이동 0~4px) 선의 특징점으로 맞추는 쪽이 안정적이다.
    """
    g0, g1 = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY), cv2.cvtColor(redrawn, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(2000, fastThreshold=10)
    k0, d0 = orb.detectAndCompute(g0, keep)
    k1, d1 = orb.detectAndCompute(g1, None)
    if d0 is None or d1 is None:
        return None, 0
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d0, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.8 * n.distance]
    if len(good) < MIN_MATCHES:
        return None, len(good)
    src = np.float32([k1[m.queryIdx].pt for m in good])
    dst = np.float32([k0[m.trainIdx].pt for m in good])
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3)
    return M, int(inl.sum()) if inl is not None else 0


BROW_R, BROW_L = [70, 63, 105, 66, 107, 46, 53, 52, 65, 55], [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]


def _local_colors(src: np.ndarray, ref: np.ndarray, hole: np.ndarray, sigma: float) -> np.ndarray:
    """src에 (ref − src)의 부드러운 색 차이를 더한다. 차이는 hole 밖에서만 재고 hole 안으로는 둘레 값이 번져 들어간다
    (가중 흐림). 선이 1~2px 밀려 크게 다른 화소는 재는 데서 뺀다.

    색 차이는 재료마다 다르다(피부는 조금, 머리카락·배경은 많이). 한데 섞어 번지게 하면 턱 둘레 피부에 머리카락의
    차이가 묻어 회색이 된다(카드 E). 그래서 src 색을 몇 묶음으로 나눠 같은 묶음끼리만 번지게 하고, 둘레에 같은 색이
    거의 없는 묶음(입 안 색)은 전체 평균 차이를 쓴다.
    """
    # 차이는 부드러운 값이라 절반 크기에서 계산해 키운다(같은 결과에 시간은 1/4)
    H, W = src.shape[:2]
    small = lambda a, interp: cv2.resize(a, (W // 2, H // 2), interpolation=interp)  # noqa: E731
    s_src = small(src, cv2.INTER_AREA)
    diff = small(ref, cv2.INTER_AREA).astype(np.float32) - s_src.astype(np.float32)
    w = ((small(hole, cv2.INTER_NEAREST) == 0) & (np.abs(diff).max(axis=2) < 110)).astype(np.float32)
    sg = sigma / 2

    def spread(weight):
        num = cv2.GaussianBlur(diff * weight[..., None], (0, 0), sg)
        den = cv2.GaussianBlur(weight, (0, 0), sg)
        return num / np.maximum(den, 1e-3)[..., None], den

    field, den = spread(w)
    field = np.where(den[..., None] > 1e-3, field, 0)
    px = s_src.reshape(-1, 3).astype(np.float32)
    try:
        rng = np.random.default_rng(0)
        sample = px[rng.choice(len(px), size=min(len(px), 4000), replace=False)]
        _, _, centers = cv2.kmeans(sample, COLOR_GROUPS, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
                                   2, cv2.KMEANS_PP_CENTERS)
        lab = np.argmin(((px[:, None, :] - centers[None]) ** 2).sum(axis=2), axis=1).reshape(s_src.shape[:2])
        for k in range(len(centers)):
            mk = (lab == k)
            fk, dk = spread(w * mk)
            use = mk & (dk > 0.04)          # 둘레에 같은 색이 충분히 있는 화소만
            field[use] = fk[use]
    except cv2.error:
        pass
    field = cv2.resize(field, (W, H), interpolation=cv2.INTER_LINEAR)
    return np.clip(src.astype(np.float32) + field, 0, 255).astype(np.uint8)


COLOR_GROUPS = 6


def paste_mouth(scene: Image.Image, face: Face, redrawn: Image.Image, box: tuple,
                region: str = "mouth") -> tuple[Image.Image | None, str]:
    """redrawn(FLUX가 scene의 box 자리를 다시 그린 얼굴 조각)에서 region(mouth 입 둘레, eyes 눈·눈썹 둘레)만
    scene에 붙인다. (그림 또는 None, 실패 이유)"""
    a = np.asarray(scene.convert("RGB"))
    H, W = a.shape[:2]
    side = box[2] - box[0]
    k = redrawn.width / side                      # scene → 조각 배율
    crop = np.asarray(scene.crop(box).resize(redrawn.size, Image.LANCZOS))
    fa = np.asarray(redrawn.convert("RGB"))
    pts = (face.pts - [box[0], box[1]]) * k       # 조각 좌표의 얼굴 점
    mw = face.mouth_w * k

    def fill(poly, shape):
        m = np.zeros(shape, np.uint8)
        cv2.fillConvexPoly(m, cv2.convexHull(poly.astype(np.int32)), 255)
        return m

    def ell(size):
        size = max(3, int(size)) | 1
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

    hull = fill(pts, crop.shape[:2])
    lips = cv2.dilate(fill(pts[OUTER], crop.shape[:2]), ell(mw * 0.5))
    eyes = np.zeros(crop.shape[:2], np.uint8)
    ew = float(np.linalg.norm(pts[33] - pts[133]) + np.linalg.norm(pts[362] - pts[263])) / 2
    for eye, brow in ((R_EYE, BROW_R), (L_EYE, BROW_L)):
        # 눈만(눈썹·앞머리는 넣지 않는다: FLUX가 머리 색을 바꾸는 일이 있어 같이 붙이면 얼룩이 된다)
        eyes |= cv2.dilate(fill(pts[eye], crop.shape[:2]), ell(ew * EYE_GROW))
    # 특징점은 얼굴과 그 둘레(머리카락·안경)에서 찾되, 바뀌는 자리(입 또는 눈)는 뺀다
    keep = cv2.dilate(hull, ell(face.face_w * k * 0.5)) & ~(lips if region == "mouth" else eyes)
    M, n = _register(crop, fa, keep)
    if M is None or n < MIN_MATCHES:
        return None, f"선이 맞는 점이 적음({n})"
    scale = float(np.hypot(M[0, 0], M[1, 0]))
    if not 0.9 < scale < 1.1 or max(abs(M[0, 2]), abs(M[1, 2])) > 0.15 * redrawn.width:
        return None, f"구도가 크게 바뀜(크기 {scale:.2f})"
    warped = cv2.warpAffine(fa, M, (crop.shape[1], crop.shape[0]), flags=cv2.INTER_LANCZOS4,
                            borderMode=cv2.BORDER_REFLECT)
    # 섞을 자리: 원래 입 둘레를 넉넉히, 벌린 입이 내려올 턱 쪽으로 더. 얼굴 윤곽 안쪽으로만
    c, s = math.cos(face.angle), math.sin(face.angle)
    down = np.array([-s, c]) * mw * 0.3
    fw = face.face_w * k
    u = max(mw, 0.22 * fw) if region == "mouth" else ew    # 다문 입은 입 너비가 아주 작게 잡힌다(아이 얼굴). 얼굴 너비로 받친다
    if region == "mouth":
        m = cv2.dilate(fill(np.vstack([pts[OUTER], pts[OUTER] + down]), crop.shape[:2]), ell(u * 0.55))
        m &= cv2.erode(hull, ell(fw * 0.04))
    else:  # 눈과 눈썹. 만화 눈은 얼굴 점보다 커서 넉넉히 잡는다(안경·앞머리는 선을 맞췄으니 같이 붙어도 된다)
        m = eyes & cv2.dilate(hull, ell(face.face_w * k * 0.06))
    # FLUX가 실제로 바꾼 자리(원래 조각과 달라진 곳) 가운데 입(눈) 자리에 닿아 있는 덩어리를 붙일 자리로 쓴다.
    # 크게 벌린 입은 다문 입보다 훨씬 아래까지 내려오고(아이 얼굴), 틀 얼굴은 자리를 대략만 알고, 만화 눈은 크기가
    # 제각각이라 미리 잡은 자리만으로는 잘리거나 머리카락까지 들어간다
    # FLUX는 조각 전체의 색도 바꾼다(밤 장면을 낮처럼 밝게 그린 카드 D: 수염·피부·배경이 모두 달라 전부 걸러졌다).
    # 바꾸지 않는 자리(입·눈 둘레 밖)에서 잰 색 차이를 부드럽게 퍼뜨려 조각에 더해 준다(자리마다 다른 색 차이를 따라간다)
    warped = _local_colors(warped, crop, cv2.dilate(lips | eyes, ell(u * 0.5)), sigma=max(3.0, u * 0.9))
    d = np.abs(cv2.GaussianBlur(crop, (0, 0), 1.5).astype(int) - cv2.GaussianBlur(warped, (0, 0), 1.5)).max(axis=2)
    if region == "mouth":
        seed = cv2.dilate(m, ell(u * 0.6)) if face.approx else m
        reach = cv2.dilate(m, ell(u * (2.0 if face.approx else 1.2))) & cv2.erode(hull, ell(fw * 0.03))
        reach = reach & ~cv2.dilate(eyes, ell(ew * 0.3))   # 입을 찾는 중에는 눈 자리를 뺀다(웃는 눈을 뜨게 바꾼 것이 딸려 왔다)
    else:
        tight = np.zeros_like(m)
        for eye in (R_EYE, L_EYE):
            tight |= cv2.dilate(fill(pts[eye], crop.shape[:2]), ell(ew * (0.9 if face.approx else 0.35)))
        seed, reach = tight, cv2.dilate(tight, ell(ew * 0.9))
    cand = ((d > 28) & (reach > 0)).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, ell(u * 0.2))
    n, lab = cv2.connectedComponents(cand, connectivity=8)
    hit = np.unique(lab[(seed > 0) & (cand > 0)])
    extra = (np.isin(lab, hit[hit > 0]).astype(np.uint8) * 255) if len(hit) else np.zeros_like(m)
    extra = cv2.dilate(extra, ell(u * 0.25))
    if face.approx or region != "mouth":
        if extra.sum() // 255 < 30:
            return scene.convert("RGB"), ""           # 바뀐 게 없으면 원래 그림 그대로(이미 감은 눈 등)
        m = extra
    else:
        m |= extra
    # 붙일 자리의 테두리에서 두 그림이 같은 색이어야 한다(피부). 다르면 선이 어긋났거나 다른 것이 딸려 온 것
    area = float(m.sum() / 255 / max(1, (hull > 0).sum()))   # 붙일 자리 넓이 ÷ 얼굴 넓이
    if os.environ.get("LIPSYNC_TRACE"):
        print(f"TRACE {region} area={area:.3f}", flush=True)
    if region == "mouth" and not face.approx and area > MAX_MOUTH_AREA:
        return None, f"바뀐 자리가 너무 넓음({area:.2f})"   # 수염·뺨까지 바뀐 것으로 잡혀 얼룩이 된다(카드 D)
    ring = (cv2.dilate(m, ell(u * 0.3)) > 0) & (m == 0)
    if ring.sum() < 20:
        return None, "붙일 자리가 너무 작음"
    # 눈 둘레는 안경테·속눈썹 같은 선이 테두리를 지나가서, 1~2px 어긋난 선이 걸리지 않게 흐리게 한 뒤 견준다
    # 눈 둘레와 틀 얼굴은 안경테·속눈썹 같은 선이 테두리를 지나가서, 1~2px 어긋난 선이 걸리지 않게 흐리게 한 뒤 견준다
    c0, c1 = cv2.GaussianBlur(crop, (0, 0), 2.5), cv2.GaussianBlur(warped, (0, 0), 2.5)
    bad = float((np.abs(c0[ring].astype(int) - c1[ring].astype(int)).mean(axis=1) > 40).mean())
    if bad > (MAX_RING_BAD if region == "mouth" else MAX_EYE_RING_BAD):
        return None, f"테두리 색이 다름({bad:.2f})"
    # FLUX는 피부색도 조금 바꾼다. 테두리 띠에서 잰 색 차이(가운데값)만큼 붙일 조각을 옮겨 밝은 자국이 남지 않게 한다
    shift = np.median(crop[ring].astype(np.float32) - warped[ring].astype(np.float32), axis=0)
    if np.abs(shift).max() <= 30:
        warped = np.clip(warped.astype(np.float32) + shift, 0, 255).astype(np.uint8)
    alpha = cv2.GaussianBlur(m, (max(3, int(u * 0.2)) | 1,) * 2, 0).astype(np.float32)[..., None] / 255
    mixed = (crop * (1 - alpha) + warped * alpha).astype(np.uint8)
    # 조각을 장면 크기로 되돌려, 바뀐 자리만 장면에 넣는다(나머지 화소는 원래 그대로)
    back = np.asarray(Image.fromarray(mixed).resize((side, side), Image.LANCZOS))
    aback = np.asarray(Image.fromarray((alpha[..., 0] * 255).astype(np.uint8)).resize((side, side), Image.BILINEAR))
    out = a.copy()
    region = out[box[1]:box[3], box[0]:box[2]]
    w = (aback.astype(np.float32) / 255)[..., None]
    region[:] = (region * (1 - w) + back * w).astype(np.uint8)
    return Image.fromarray(out), ""


MIN_MATCHES = 25        # 얼굴 둘레에서 맞아야 하는 특징점 수
MAX_EYE_RING_BAD = 0.15  # 눈 둘레 바깥 띠에서(흐리게 한 뒤) 색이 크게 다른 점의 비율
MAX_AREA = 0.45         # 틀 얼굴: 바뀐 자리가 찾는 범위의 이 비율을 넘으면 FLUX가 입·눈 말고 다른 것까지 바꾼 것
MAX_MOUTH_AREA = float(os.environ.get("LIPSYNC_MAX_MOUTH_AREA", "0.65"))   # 입을 붙일 자리가 얼굴 넓이의 이 비율을 넘으면 입 말고 다른 것까지 바뀐 것. 0.45는 멀쩡한 얼굴도 걸렀다(E·U·G2, 13회차)
MAX_RING_BAD = 0.12     # 테두리에서 색이 크게 다른 점의 비율
EYE_GROW = 1.0          # 눈 둘레를 눈 너비의 몇 배만큼 넓혀 붙일지
EYE_KINDS = ["B"]        # B 감은 눈(깜빡임). R 올린 눈썹은 FLUX가 눈 모양까지 바꿔서 버렸다(2회차)

SUBSTITUTE = {"X": ["M"], "M": ["O", "E", "X"], "A": ["E", "O", "M"], "E": ["A", "M", "O"], "O": ["M", "A"]}  # 빠진 모양 → 대신 쓸 모양


ORIGINAL_AS_OPEN = os.environ.get("LIPSYNC_ORIGINAL_AS_OPEN", "1") != "0"


def mouth_open(f: Face) -> bool:
    """원래 그림의 입이 크게 벌어져 있는가(안쪽 입술 사이가 입 너비의 30% 이상). 틀 얼굴은 알 수 없어 False."""
    return not f.approx and float(np.linalg.norm(f.pts[13] - f.pts[14])) > 0.30 * f.mouth_w


def mouth_closed(f: Face) -> bool:
    """원래 그림의 입이 다물려 있는가(안쪽 입술 사이가 입 너비의 8% 미만)."""
    return not f.approx and float(np.linalg.norm(f.pts[13] - f.pts[14])) < 0.08 * f.mouth_w



# ---------------------------------------------------------------- 누가 말하는지, 언제 어떤 입인지

POS_X = {"left": 0.25, "center": 0.5, "right": 0.75}


def face_of(scene: dict, faces: list[Face], speaker: str) -> int | None:
    """말하는 화자의 얼굴 번호. cast·extras를 자리 순으로, 얼굴도 왼쪽부터 세워 맞춘다. 못 맞추면 None."""
    people = sorted([(POS_X[c["position"]], c.get("key")) for c in scene.get("cast") or []] +
                    [(POS_X[e["position"]], None) for e in scene.get("extras") or []], key=lambda p: p[0])
    keys = [k for _, k in people]
    if speaker not in keys or not faces:
        return None
    if len(faces) < len(people):
        return None  # 얼굴을 덜 찾았으면 건드리지 않는다(엉뚱한 사람 입이 움직이는 것보다 낫다)
    # 얼굴이 더 많으면(배경 사람) 큰 얼굴부터 사람 수만큼만 쓴다. faces.match_speakers와 같은 규칙
    big = sorted(sorted(range(len(faces)), key=lambda i: -faces[i].face_w)[:len(people)],
                 key=lambda i: faces[i].mouth_c[0])
    return big[keys.index(speaker)]


# 입을 손으로 가린 장면: FLUX가 손 위에 입을 그리거나 손을 지워 버린다(카드 C의 웃음 참는 장면). 콘티 글로 걸러 낸다
COVERED_RE = re.compile(r"\b(cover\w*|hid\w*|hand\w*\s+(over|on|in front of))\b[^.]{0,25}\bmouth\b|\bmouth\b[^.]{0,15}\bcovered\b",
                        re.IGNORECASE)


def mouth_hidden(scene: dict, speaker: str) -> bool:
    """콘티가 그 화자의 입이 가려졌다고 쓴 장면인가. 표정 글에 있으면 그 사람만, 장면 글에 있으면 장면 전체."""
    for c in scene.get("cast") or []:
        if c.get("key") == speaker and COVERED_RE.search(c.get("expression") or ""):
            return True
    return bool(COVERED_RE.search(scene.get("scene") or ""))


def speaking_faces(scene: dict, faces: list[Face]) -> dict[int, int]:
    """이 장면에서 말하는 얼굴 번호 → 말하는 글자 수(많이 말하는 얼굴부터 FLUX를 쓴다)."""
    out: dict[int, int] = {}
    for d in scene.get("lines") or []:
        fi = face_of(scene, faces, d["speaker"]) if d.get("spoken", True) and not mouth_hidden(scene, d["speaker"]) else None
        if fi is not None:
            out[fi] = out.get(fi, 0) + len(d["text"])
    return out


def make_mouths(sb: dict, images: list[Image.Image], edit) -> list[dict]:
    """장면마다 {faces, variants: {얼굴 번호: {모양: 그 얼굴 입만 바꾼 장면 그림}}, source: {얼굴: {모양: flux|drawn}}}.

    edit(crop, shape) → FLUX가 다시 그린 얼굴 조각(PIL). None을 주면 FLUX 없이 코드로 그린 입만 쓴다.
    얼굴 수가 MAX_FACES를 넘으면 많이 말하는 얼굴부터 FLUX를 쓰고 나머지는 코드로 그린다.
    """
    scenes = []
    for s, im in zip(sb["scenes"], images):
        im = square(im)
        faces = find_faces(im)   # 대사가 없는 장면도 눈은 깜빡인다
        scenes.append((s, im, faces, speaking_faces(s, faces)))
    budget = sorted(((n, i, fi) for i, (_, _, _, sp) in enumerate(scenes) for fi, n in sp.items()), reverse=True)
    use_flux = {(i, fi) for _, i, fi in budget[:MAX_FACES]} if edit is not None else set()
    out = []
    for i, (s, im, faces, sp) in enumerate(scenes):
        variants, source = {}, {}
        looks = {c.get("key"): c.get("appearance", "") for c in sb.get("characters") or []}
        hairy = {face_of(s, faces, c["key"]): hair_of(looks.get(c["key"], "")) for c in s.get("cast") or []
                 if hair_of(looks.get(c["key"], ""))}
        for fi in sp:
            f = faces[fi]
            # 얼굴 점이 대략이면 코드로는 못 그리고, 수염 난 얼굴에 코드로 그리면 수염이 지워진다 → FLUX가 안 되면 그대로 둔다
            drawn = {} if (f.approx or fi in hairy) else drawn_mouths(im, f)
            variants[fi], source[fi] = dict(drawn), {k: "drawn" for k in drawn}
            if (i, fi) not in use_flux:
                continue
            box = face_box(f, *im.size)
            crop = im.crop(box).resize((MOUTH_SIZE, MOUTH_SIZE), Image.LANCZOS)
            flux = {}
            for k in FLUX_SHAPES:
                redrawn = edit(crop, f"{k}+{hairy[fi]}" if fi in hairy else k)
                if DEBUG_DIR:  # 시험용: FLUX에 넣은 조각과 다시 그린 그림을 남긴다
                    Path(DEBUG_DIR).mkdir(parents=True, exist_ok=True)
                    crop.save(Path(DEBUG_DIR) / f"s{s['index']}_f{fi}_in.png")
                    redrawn.save(Path(DEBUG_DIR) / f"s{s['index']}_f{fi}_{k}.png")
                got, why = paste_mouth(im, f, redrawn, box)
                if got is None:
                    log.info("장면 %d 얼굴 %d 입모양 %s: FLUX 그림 못 씀 (%s)", s["index"], fi, k, why)
                else:
                    flux[k] = got
            # 한 얼굴에서 FLUX 입과 코드로 그린 입이 섞이면 그림체가 프레임마다 바뀐다. 벌린 입이 하나라도 붙고
            # 세 모양 이상 붙었을 때만 FLUX를 쓰고, 빠진 모양은 가까운 FLUX 모양으로 대신한다
            if len(flux) >= 3 and any(k in flux for k in "AEO"):
                for k in FLUX_SHAPES:
                    pick = next((c for c in [k] + SUBSTITUTE[k] if c in flux), None)
                    if pick:
                        variants[fi][k], source[fi][k] = flux[pick], "flux"
                    elif k == "X" and (mouth_closed(f) or f.approx):
                        variants[fi][k], source[fi][k] = im, "original"
            # 원래 그림의 입이 크게 벌어져 있으면(웃거나 외치는 표정) 가장 크게 벌리는 모양은 원래 그림을 쓴다.
            # 그린 표정이 대사 중에 그대로 보이고, 원래 그림이라 자국도 없다
            if ORIGINAL_AS_OPEN and fi in variants and mouth_open(f) and source[fi].get("A") == "flux":
                variants[fi]["A"], source[fi]["A"] = im, "original"
            if len(variants[fi]) < len(FLUX_SHAPES):   # 쓸 입모양이 모자란 얼굴(틀 얼굴인데 FLUX도 실패)은 건드리지 않는다
                del variants[fi], source[fi]
        # 눈(B 감은 눈, R 올린 눈썹): 말하지 않는 사람도 깜빡이므로 이 장면의 사람 수만큼 큰 얼굴 모두
        eyes = {}
        people = len(s.get("cast") or []) + len(s.get("extras") or [])
        for fi in sorted(sorted(range(len(faces)), key=lambda j: -faces[j].face_w)[:people]) if edit is not None else ():
            f = faces[fi]
            box = face_box(f, *im.size)
            crop = im.crop(box).resize((MOUTH_SIZE, MOUTH_SIZE), Image.LANCZOS)
            for k in EYE_KINDS:
                redrawn = edit(crop, k)
                if redrawn is None:
                    continue
                got, why = paste_mouth(im, f, redrawn, box, region="eyes")
                if got is None:
                    log.info("장면 %d 얼굴 %d 눈 %s: FLUX 그림 못 씀 (%s)", s["index"], fi, k, why)
                else:
                    eyes.setdefault(fi, {})[k] = got
        out.append({"image": im, "faces": faces, "variants": variants, "source": source, "eyes": eyes})
    return out


def read_wav(data: bytes) -> tuple[np.ndarray, int]:
    """PCM16 WAV 바이트 → (모노 float32, sr). worker.video.voice가 만든 형식."""
    with wave.open(io.BytesIO(data)) as w:
        sr, ch = w.getframerate(), w.getnchannels()
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    return (a.reshape(-1, ch).mean(axis=1) if ch > 1 else a), sr


def plan(sb: dict, wav: bytes, mouths: list[dict], blink: bool | None = None, blend: bool | None = None,
         head: float | None = None, eyes: bool = True, brows: bool = False, nod: float | None = None) -> list[dict]:
    """장면마다 {frames: {프레임 번호: {얼굴: 모양}}, variants, boxes, blink, blinks}. video_render.render(lips=)에 넣는다."""
    track, sr = read_wav(wav)
    fps = sb["size"]["fps"]
    blink = BLINK if blink is None else blink
    blend = BLEND if blend is None else blend
    head = HEAD if head is None else head
    nod = NOD if nod is None else nod
    out = []
    for s, mo in zip(sb["scenes"], mouths):
        faces, frames, spoke, moves, spoke_nomouth = mo["faces"], {}, [], {}, []
        for d in s.get("lines") or []:
            fi = face_of(s, faces, d["speaker"]) if d.get("spoken", True) else None
            if fi is not None and fi not in mo["variants"] and head > 0:
                # 입모양이 없는 화자(손으로 입을 가림, FLUX 실패)도 말하는 동안 고개는 움직인다
                clip = track[int(d["startMs"] / 1000 * sr):int(d["endMs"] / 1000 * sr)]
                f0 = round(d["startMs"] / 1000 * fps)
                n = int(np.ceil(len(clip) / sr * fps))
                for k, mv in enumerate(head_moves(envelope(clip, sr, fps, n), faces[fi].face_w, fps, amp=head)):
                    moves.setdefault(f0 + k, {})[fi] = mv
                spoke_nomouth.append((fi, f0, f0 + n))
            if fi is None or fi not in mo["variants"]:
                continue
            clip = track[int(d["startMs"] / 1000 * sr):int(d["endMs"] / 1000 * sr)]
            f0 = round(d["startMs"] / 1000 * fps)
            seq = shapes_mix(d["text"], clip, sr, fps)
            for k, sh in enumerate(seq):
                frames.setdefault(f0 + k, {})[fi] = sh
            spoke.append((fi, f0, f0 + len(seq)))
            if head > 0:
                env = envelope(clip, sr, fps, len(seq))
                for k, mv in enumerate(head_moves(env, faces[fi].face_w, fps, phase=0.9 * len(spoke), amp=head)):
                    moves.setdefault(f0 + k, {})[fi] = mv
        # 이 장면에서 말하는 사람은, 다른 사람이 말하는 동안 입을 다문다(원래 그림의 벌린 입으로 돌아가지 않게)
        for fi, _, _ in spoke:
            for fj, a0, a1 in spoke:
                if fj != fi:
                    for n in range(a0, a1):
                        frames.setdefault(n, {}).setdefault(fi, "X")
        blinks, bimg = set(), None
        if blink and faces:
            bimg = blink_frame(mo["image"], faces)
        if bimg is not None:  # 약 3초에 한 번, 4프레임(0.13초)
            f0, f1 = round(s["startMs"] / 1000 * fps), round(s["endMs"] / 1000 * fps)
            rng = np.random.default_rng(s["index"])
            t = f0 + int(fps * rng.uniform(0.8, 1.6))
            while t + 4 < f1 - 6:
                blinks |= set(range(t, t + 4))
                t += int(fps * rng.uniform(2.4, 3.6))
        Wd, Hd = mo["image"].size
        boxes = {}
        for fi, f in enumerate(faces):
            r = f.mouth_w * (2.6 if f.approx else 1.2)   # 틀 얼굴은 입 자리가 대략이라 넉넉히(달라진 화소만 덮는다)
            boxes[fi] = (max(0, int(f.mouth_c[0] - r)), max(0, int(f.mouth_c[1] - r)),
                         min(Wd, int(f.mouth_c[0] + r)), min(Hd, int(f.mouth_c[1] + r)))
        # FLUX가 그린 눈: 얼굴마다 2.5~4.5초에 한 번 깜빡이고(반쯤 → 감음 → 반쯤), 말하는 사람은 소리가 가장 클 때
        # 눈썹을 한 번 올린다
        eye_frames, eye_var = {}, {}
        s0, s1 = round(s["startMs"] / 1000 * fps), round(s["endMs"] / 1000 * fps)
        for fi, var in (mo.get("eyes") or {}).items() if eyes else ():
            eye_var[fi] = dict(var)
            if "B" in var:
                eye_var[fi]["b"] = Image.blend(mo["image"], var["B"], 0.5)
                rng = np.random.default_rng(1000 * s["index"] + fi)
                t = s0 + int(fps * rng.uniform(0.5, 2.0))
                while t + 5 < s1 - 4:
                    for k, st in enumerate("bBBb"):
                        eye_frames.setdefault(t + k, {})[fi] = st
                    t += int(fps * rng.uniform(2.5, 4.5))
        if eyes and brows:
            for d in s.get("lines") or []:
                fi = face_of(s, faces, d["speaker"]) if d.get("spoken", True) else None
                if fi is None or "R" not in eye_var.get(fi, {}):
                    continue
                clip = track[int(d["startMs"] / 1000 * sr):int(d["endMs"] / 1000 * sr)]
                n = int(np.ceil(len(clip) / sr * fps))
                if n < fps:      # 1초가 안 되는 대사는 건너뛴다
                    continue
                env = envelope(clip, sr, fps, n)
                k = max(3, int(fps * 0.3)) | 1
                sm = np.convolve(np.pad(env, k // 2, mode="edge"), np.ones(k) / k, mode="valid")[:n]
                peak = int(np.argmax(sm))
                f0 = round(d["startMs"] / 1000 * fps)
                for j in range(max(0, peak - 4), min(n, peak + 10)):
                    eye_frames.setdefault(f0 + j, {}).setdefault(fi, "R")   # 깜빡임과 겹치면 깜빡임이 먼저
        eye_boxes = {}
        for fi in eye_var:
            f = faces[fi]
            p = np.vstack([f.pts[R_EYE], f.pts[L_EYE], f.pts[BROW_R], f.pts[BROW_L]])
            ew = float(np.linalg.norm(f.pts[33] - f.pts[133]))
            g = ew * (2.5 if f.approx else 1.2)
            eye_boxes[fi] = (max(0, int(p[:, 0].min() - g)), max(0, int(p[:, 1].min() - g)),
                             min(Wd, int(p[:, 0].max() + g)), min(Hd, int(p[:, 1].max() + g)))
        # 듣는 사람 끄덕임: 다른 사람의 대사가 끝날 때 한 번
        if nod > 0:
            cast_faces = {face_of(s, faces, c["key"]) for c in s.get("cast") or []} - {None}
            for fi, a0, a1 in spoke + spoke_nomouth:
                for fj in cast_faces - {fi}:
                    L_ = int(fps * 0.5)
                    for j in range(L_):
                        dy = nod * NOD_DEPTH * faces[fj].face_w * math.sin(math.pi * j / (L_ - 1)) ** 2
                        cur = moves.setdefault(a1 - 4 + j, {}).get(fj, (0.0, 0.0, 0.0))
                        moves[a1 - 4 + j][fj] = (cur[0], cur[1] + dy, cur[2])
        W, H = mo["image"].size
        heads = {fi: head_field(faces[fi], W, H) for fi in {fi for mv in moves.values() for fi in mv}}
        out.append({"frames": frames, "variants": mo["variants"], "boxes": boxes, "blink": bimg, "blinks": blinks,
                    "blend": blend, "moves": moves, "heads": heads,
                    "eye_frames": eye_frames, "eye_variants": eye_var, "eye_boxes": eye_boxes})
    return out


# ---------------------------------------------------------------- 말할 때 고개 움직임

def head_field(f: Face, W: int, H: int) -> dict:
    """얼굴 f의 머리 둘레를 부드럽게 움직이기 위한 가중치(가운데 1 → 가장자리 0)와 회전 중심(턱 아래 목)."""
    cx, cy = f.pts[1][0], f.pts[1][1] - 0.25 * f.face_w          # 머리카락까지 들어가게 코보다 위를 가운데로
    rx, ry = 1.05 * f.face_w, 1.25 * f.face_w
    x0, y0, x1, y1 = (max(0, int(cx - rx)), max(0, int(cy - ry)), min(W, int(cx + rx) + 1), min(H, int(cy + ry) + 1))
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    r = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
    w = np.clip((1.0 - r) / 0.45, 0, 1)                           # 반지름 55% 안은 통째로, 밖으로 갈수록 덜 움직인다
    w = w * w * (3 - 2 * w)
    chin = f.pts[152]
    return {"box": (x0, y0, x1, y1), "w": w, "xx": xx, "yy": yy, "pivot": (float(chin[0]), float(chin[1] + 0.25 * f.face_w))}


def move_heads(img: Image.Image, heads: dict, moves: dict) -> Image.Image:
    """moves {얼굴: (dx, dy, 도)}만큼 머리 둘레를 살짝 옮기고 기울인다. 가장자리로 갈수록 0이라 이음매가 없다."""
    a = np.asarray(img).copy()
    for fi, (dx, dy, deg) in moves.items():
        h = heads.get(fi)
        if h is None or (abs(dx) < 0.05 and abs(dy) < 0.05 and abs(deg) < 0.02):
            continue
        x0, y0, x1, y1 = h["box"]
        px, py = h["pivot"]
        c, s = math.cos(math.radians(-deg)), math.sin(math.radians(-deg))   # 거꾸로 변환(결과 화소 → 원래 자리)
        sx = c * (h["xx"] - px - dx) - s * (h["yy"] - py - dy) + px
        sy = s * (h["xx"] - px - dx) + c * (h["yy"] - py - dy) + py
        mx = (h["xx"] + (sx - h["xx"]) * h["w"] - x0).astype(np.float32)
        my = (h["yy"] + (sy - h["yy"]) * h["w"] - y0).astype(np.float32)
        a[y0:y1, x0:x1] = cv2.remap(a[y0:y1, x0:x1], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(a)


def head_moves(seq_env: np.ndarray, fw: float, fps: int, phase: float = 0.0, amp: float = 1.0) -> list[tuple]:
    """대사 한 줄의 프레임별 (dx, dy, 도). 소리가 커질 때 고개가 조금 내려가고, 느리게 좌우로 기운다.
    줄의 처음과 끝 0.2초는 0에서 시작해 0으로 돌아온다(대사가 끝나면 원래 그림과 똑같아야 한다)."""
    n = len(seq_env)
    if n < 4:
        return [(0.0, 0.0, 0.0)] * n
    k = max(3, int(fps * 0.2)) | 1
    sm = np.convolve(np.pad(seq_env, k // 2, mode="edge"), np.ones(k) / k, mode="valid")[:n]
    ramp = max(2, int(fps * 0.2))
    gate = np.minimum(1.0, np.minimum(np.arange(n), np.arange(n)[::-1]) / ramp)
    t = np.arange(n) / fps
    dy = amp * HEAD_NOD * fw * (np.clip(sm, 0, 1) - 0.35) * gate
    deg = amp * HEAD_TILT * np.sin(2 * math.pi * t / 1.9 + phase) * gate
    dx = amp * HEAD_NOD * 0.4 * fw * np.sin(2 * math.pi * t / 2.7 + phase * 1.7) * gate
    return [(float(a), float(b), float(c)) for a, b, c in zip(dx, dy, deg)]


NOD_DEPTH = 0.05    # 듣는 사람이 끄덕이는 폭(얼굴 너비 대비)
HEAD_NOD = 0.03     # 고개가 내려가는 폭(얼굴 너비 대비)
HEAD_TILT = 1.2     # 좌우로 기우는 각도(도)
