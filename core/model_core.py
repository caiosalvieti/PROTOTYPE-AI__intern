# core/model_core.py
from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
from pathlib import Path  # <-- IMPORTAR Path
import time
import numpy as np
import os
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
# Flags so we can run without YOLO if import fails
YOLO_AVAILABLE: bool = False
YOLO_IMPORT_ERROR: Optional[str] = None

try:
    import torch  # type: ignore
    from ultralytics import YOLO  # type: ignore

    _DEVICE = 0 if torch.cuda.is_available() else "cpu"
    _YOLO_MODEL = YOLO("yolov8n.pt")  # swap to custom weights later if you want
    YOLO_AVAILABLE = True

except Exception as e:
    # On Streamlit Cloud this will trigger because cv2 wants libGL.so.1
    _YOLO_MODEL = None
    _DEVICE = "cpu"
    YOLO_AVAILABLE = False
    YOLO_IMPORT_ERROR = repr(e)  # string for debugging, but we DON'T crash


def _detect_largest_box_xyxy(img_rgb: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Run YOLO on an RGB image and return the largest bbox as (x_min, y_min, x_max, y_max).
    If YOLO is not available, returns None.
    """
    if not YOLO_AVAILABLE or _YOLO_MODEL is None:
        return None

    results = _YOLO_MODEL.predict(
        img_rgb,
        conf=0.25,
        iou=0.45,
        device=_DEVICE,
        verbose=False,
    )

    res = results[0]
    boxes = res.boxes
    if boxes is None or len(boxes) == 0:
        return None

    xyxy = boxes.xyxy.cpu().numpy()
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    idx = int(areas.argmax())
    x_min, y_min, x_max, y_max = xyxy[idx]
    return int(x_min), int(y_min), int(x_max), int(y_max)


def skinaizer_model_core(img_rgb: np.ndarray) -> Dict[str, Any]:
    """
    Core interface for the UI.

    On machines where YOLO imports correctly:
        - returns YOLO bbox + timings

    On machines where YOLO / cv2 cannot import (e.g. Streamlit Cloud without libGL):
        - returns bbox=None, timings={}, yolo_available=False
        - the caller is expected to fall back to the old detector.
    """
    # If YOLO can't be used in this environment, just say so and let the caller fallback
    if not YOLO_AVAILABLE or _YOLO_MODEL is None:
        return {
            "bbox": None,
            "timings": {},
            "yolo_available": False,
            "yolo_error": YOLO_IMPORT_ERROR,
        }

    t0 = time.perf_counter()
    bbox = _detect_largest_box_xyxy(img_rgb)
    t1 = time.perf_counter()

    timings = {
        "yolo_ms": (t1 - t0) * 1000.0,
        "total_ms": (t1 - t0) * 1000.0,
    }

    return {
        "bbox": bbox,
        "timings": timings,
        "yolo_available": True,
        "yolo_error": None,
    }
