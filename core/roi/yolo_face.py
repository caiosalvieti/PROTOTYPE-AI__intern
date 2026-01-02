# core/roi/yolo_face.py
from __future__ import annotations

from typing import Optional, Any
import numpy as np

from core.schemas import ROIResult, BBoxXYWH, WarningItem

def detect_face_yolo(rgb: np.ndarray) -> Optional[ROIResult]:
    """
    YOLO-first hook.
    Tries your existing YOLO entrypoints, but never hard-crashes if unavailable.
    """
    # 1) Try your project-level wrapper if it exists
    try:
        from core.yolo_roi import skinaizer_model_core  # your existing hook
        out = skinaizer_model_core(rgb)  # may be bbox-or-none / dict / etc
        bbox = _parse_bbox(out, rgb)
        if bbox:
            return ROIResult(bbox=bbox, method="yolo", confidence=_parse_conf(out))
        return None
    except Exception:
        pass

    # 2) Try ultralytics model_core if present (optional)
    try:
        from core.model_core import YOLO_AVAILABLE  # type: ignore
        if not YOLO_AVAILABLE:
            return None
    except Exception:
        return None

    return None

def _parse_conf(out: Any) -> float:
    # best-effort
    if isinstance(out, dict) and "confidence" in out:
        try: return float(out["confidence"])
        except Exception: return 0.0
    return 0.0

def _parse_bbox(out: Any, rgb: np.ndarray) -> Optional[BBoxXYWH]:
    H, W = rgb.shape[:2]

    # dict style
    if isinstance(out, dict):
        b = out.get("bbox") or out.get("box")
        if isinstance(b, (list, tuple)) and len(b) == 4:
            return _bbox_from_4(b, W, H)

    # tuple/list style
    if isinstance(out, (list, tuple)) and len(out) == 4:
        return _bbox_from_4(out, W, H)

    return None

def _bbox_from_4(b, W: int, H: int) -> Optional[BBoxXYWH]:
    x1, y1, a, d = [float(x) for x in b]
    # heuristic: either (x,y,w,h) or (x1,y1,x2,y2)
    if a > 0 and d > 0 and (x1 + a <= W + 1) and (y1 + d <= H + 1):
        # treat as xywh
        return BBoxXYWH(x=int(x1), y=int(y1), w=int(a), h=int(d))
    # treat as x2,y2
    x2, y2 = a, d
    w, h = int(x2 - x1), int(y2 - y1)
    if w <= 0 or h <= 0:
        return None
    return BBoxXYWH(x=int(x1), y=int(y1), w=w, h=h)
