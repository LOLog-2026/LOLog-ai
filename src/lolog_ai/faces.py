"""만화 얼굴 검출. 말풍선이 얼굴을 가리지 않게 하고, 꼬리를 화자 얼굴 쪽으로 향하게 할 때 쓴다.

모델: deepghs/anime_face_detection face_detect_v1.4_s (YOLOv8, MIT)
  https://huggingface.co/deepghs/anime_face_detection
  model.onnx를 DEFAULT_MODEL 경로에 둔다. onnxruntime(CPU)으로 돌린다(Jetson aarch64도 PyPI 패키지 있음).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_MODEL = Path.home() / ".cache/lolog/models/animeface/model.onnx"
THRESHOLD = 0.307  # 모델 저장소 threshold.json (F1 최대)

Box = tuple[int, int, int, int]  # x0, y0, x1, y1


@lru_cache(maxsize=2)
def _session(path: str):
    import onnxruntime as ort
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def _nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> list[int]:
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx0 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy0 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx1 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy1 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx1 - xx0, 0, None) * np.clip(yy1 - yy0, 0, None)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        order = order[1:][inter / (area[i] + area[order[1:]] - inter) < iou]
    return keep


def detect_faces(image: Image.Image, model: Path = DEFAULT_MODEL, threshold: float = THRESHOLD,
                 size: int = 640, nms_iou: float = 0.5) -> list[Box]:
    """얼굴 상자 목록(큰 것부터)."""
    W, H = image.size
    x = np.asarray(image.convert("RGB").resize((size, size)), dtype=np.float32)
    x = x.transpose(2, 0, 1)[None] / 255.0
    out = _session(str(model)).run(None, {"images": x})[0][0]  # (5, N): cx, cy, w, h, score
    out = out.T if out.shape[0] < out.shape[1] else out
    out = out[out[:, 4] >= threshold]
    if not len(out):
        return []
    sx, sy = W / size, H / size
    boxes = np.stack([(out[:, 0] - out[:, 2] / 2) * sx, (out[:, 1] - out[:, 3] / 2) * sy,
                      (out[:, 0] + out[:, 2] / 2) * sx, (out[:, 1] + out[:, 3] / 2) * sy], axis=1)
    keep = _nms(boxes, out[:, 4], nms_iou)
    faces = [tuple(int(v) for v in boxes[i]) for i in keep]
    return sorted(faces, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))


def match_speakers(faces: list[Box], positions: dict[str, str]) -> dict[str, Box]:
    """큰 얼굴부터 인물 수만큼 골라 왼쪽→오른쪽 순서로 인물(position 순)과 짝짓는다.
    얼굴 수가 인물 수보다 적으면 짝짓지 않는다(위치 비율로 대신함)."""
    if len(faces) < len(positions):
        return {}
    order = {"left": 0, "center": 1, "right": 2}
    keys = sorted(positions, key=lambda k: order[positions[k]])
    chosen = sorted(faces[: len(keys)], key=lambda b: b[0] + b[2])
    return dict(zip(keys, chosen))
