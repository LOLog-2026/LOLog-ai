"""한 번에 하나씩 바꿀 설정. base가 지금 워커(detect.py) 그대로다.

키
  temperature     detect·speakers 호출 온도 (워커 기본 0.2)
  detect_system   DETECT_SYSTEM 전체를 바꿈
  detect_append   DETECT_SYSTEM 끝에 규칙을 덧붙임
  speaker_system  SPEAKER_SYSTEM 전체를 바꿈
새 실험은 여기에 이름을 하나 더하고 README 표에 줄을 더한다.
"""

from __future__ import annotations

VARIANTS: dict[str, dict] = {
    "base": {},
    "temp0": {"temperature": 0.0},
}


def apply(det_module, v: dict) -> None:
    if "detect_system" in v:
        det_module.DETECT_SYSTEM = v["detect_system"]
    if "detect_append" in v:
        det_module.DETECT_SYSTEM = det_module.DETECT_SYSTEM + "\n" + v["detect_append"]
    if "speaker_system" in v:
        det_module.SPEAKER_SYSTEM = v["speaker_system"]
