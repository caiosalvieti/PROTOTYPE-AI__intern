# core/roi/detect.py
from __future__ import annotations

from typing import List
import numpy as np

from core.schemas import ROIResult, BBoxXYWH, WarningItem
from core.roi.yolo_face import detect_face_yolo
from core.roi.opencv_faces import detect_face_mediapipe, detect_face_haar, detect_face_dnn

def detect_roi(rgb: np.ndarray, max_dim: int = 800, min_side: int = 120) -> ROIResult:
    warnings: List[WarningItem] = []
    H, W = rgb.shape[:2]

    # 1) YOLO
    try:
        r = detect_face_yolo(rgb)
        if r and r.bbox:
            r.warnings = warnings + (r.warnings or [])
            return r
    except Exception as e:
        warnings.append(WarningItem(code="roi_yolo_failed", message=repr(e)))

    # 2) MediaPipe
    try:
        r = detect_face_mediapipe(rgb, min_side=min_side, conf=0.5)
        if r and r.bbox:
            r.warnings = warnings + (r.warnings or [])
            return r
    except Exception as e:
        warnings.append(WarningItem(code="roi_mediapipe_failed", message=repr(e)))

    # 3) Haar
    try:
        r = detect_face_haar(rgb, max_dim=max_dim, min_side=min_side)
        if r and r.bbox:
            r.warnings = warnings + (r.warnings or [])
            return r
    except Exception as e:
        warnings.append(WarningItem(code="roi_haar_failed", message=repr(e)))

    # 4) DNN
    try:
        r = detect_face_dnn(rgb, min_side=min_side, conf_thresh=0.5)
        if r and r.bbox:
            r.warnings = warnings + (r.warnings or [])
            return r
    except Exception as e:
        warnings.append(WarningItem(code="roi_dnn_failed", message=repr(e)))

    # 5) center fallback
    cw, ch = int(W * 0.70), int(H * 0.70)
    x, y = (W - cw) // 2, (H - ch) // 2
    warnings.append(WarningItem(code="roi_fallback", message="Used center fallback ROI"))
    return ROIResult(bbox=BBoxXYWH(x=x, y=y, w=cw, h=ch), method="fallback_center", confidence=0.0, warnings=warnings)
