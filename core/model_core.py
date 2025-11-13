# core/model_core.py
from __future__ import annotations
from typing import Dict, Any, Optional, Tuple

import time
import numpy as np
import torch
from ultralytics import YOLO
import cv2


# Load YOLO once, globally
_DEVICE = 0 if torch.cuda.is_available() else "cpu"
_YOLO_MODEL = YOLO("yolov8n.pt")   # later you can swap to a face/skin model


def _detect_largest_box_xyxy(img_rgb: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Run YOLO on an RGB image and return the largest bbox as (x_min, y_min, x_max, y_max)
    in the ORIGINAL image coordinate system.
    """
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
    # (optional) filter by class here if you want (e.g. person=0, face in custom model, etc.)

    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    idx = int(areas.argmax())
    x_min, y_min, x_max, y_max = xyxy[idx]
    return int(x_min), int(y_min), int(x_max), int(y_max)


def skinaizer_model_core(img_rgb: np.ndarray) -> Dict[str, Any]:
    """
    End-to-end core for now = YOLO-based ROI detection.

    Input:
        img_rgb: numpy array HxWx3, dtype uint8, RGB (this matches M.imread_rgb output)

    Output dict:
        - bbox: (x_min, y_min, x_max, y_max) or None
        - timings: {"yolo_ms": ..., "total_ms": ...}
    """
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
    }
