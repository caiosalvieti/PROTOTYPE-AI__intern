# core/roi/opencv_faces.py
from __future__ import annotations

from typing import Optional
import numpy as np

try:
    import cv2
except Exception as e:
    cv2 = None  # type: ignore

from core.schemas import BBoxXYWH, ROIResult

if cv2 is None:
    raise RuntimeError("OpenCV (cv2) missing. Use opencv-python-headless in Streamlit deployments.")

CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# MediaPipe optional
try:
    import mediapipe as mp
    mp_fd = mp.solutions.face_detection
    MP_OK = True
except Exception:
    mp_fd = None
    MP_OK = False

# DNN optional (only if model files exist)
_DNN_NET = None

def init_dnn(proto_path: str, weights_path: str) -> None:
    global _DNN_NET
    try:
        if proto_path and weights_path:
            _DNN_NET = cv2.dnn.readNetFromCaffe(proto_path, weights_path)
    except Exception:
        _DNN_NET = None

def detect_face_mediapipe(rgb: np.ndarray, min_side: int = 120, conf: float = 0.5) -> Optional[ROIResult]:
    if not MP_OK:
        return None
    H, W = rgb.shape[:2]
    with mp_fd.FaceDetection(model_selection=0, min_detection_confidence=conf) as det:
        res = det.process(rgb)
    if not res.detections:
        return None

    best = None
    best_area = -1
    best_conf = 0.0

    for d in res.detections:
        bb = d.location_data.relative_bounding_box
        x1 = max(0, int(bb.xmin * W))
        y1 = max(0, int(bb.ymin * H))
        x2 = min(W - 1, int((bb.xmin + bb.width) * W))
        y2 = min(H - 1, int((bb.ymin + bb.height) * H))
        w, h = x2 - x1, y2 - y1
        if min(w, h) < min_side:
            continue
        area = w * h
        if area > best_area:
            best_area = area
            best = BBoxXYWH(x=x1, y=y1, w=w, h=h)
            try:
                best_conf = float(d.score[0])
            except Exception:
                best_conf = 0.0

    if best is None:
        return None
    return ROIResult(bbox=best, method="mediapipe", confidence=best_conf)

def detect_face_haar(rgb: np.ndarray, max_dim: int = 800, min_side: int = 120) -> Optional[ROIResult]:
    h, w = rgb.shape[:2]
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        small = cv2.resize(rgb, (int(w * scale), int(h * scale)), cv2.INTER_AREA)
    else:
        small = rgb

    gray = cv2.cvtColor(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
    faces = CASCADE.detectMultiScale(gray, 1.1, 5)
    if len(faces) == 0:
        return None

    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    if min(fw, fh) < 40:
        return None

    inv = 1.0 / scale
    rx, ry, rw, rh = int(x * inv), int(y * inv), int(fw * inv), int(fh * inv)
    if min(rw, rh) < min_side:
        return None

    return ROIResult(bbox=BBoxXYWH(x=rx, y=ry, w=rw, h=rh), method="haar", confidence=0.55)

def detect_face_dnn(rgb: np.ndarray, min_side: int = 120, conf_thresh: float = 0.5) -> Optional[ROIResult]:
    if _DNN_NET is None:
        return None
    H, W = rgb.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        scalefactor=1.0,
        size=(300, 300),
        mean=(104.0, 177.0, 123.0),
        swapRB=False,
        crop=False,
    )
    _DNN_NET.setInput(blob)
    dets = _DNN_NET.forward()

    best = None
    best_conf = 0.0
    for i in range(dets.shape[2]):
        conf = float(dets[0, 0, i, 2])
        if conf < conf_thresh:
            continue
        x1, y1, x2, y2 = (dets[0, 0, i, 3:7] * np.array([W, H, W, H])).astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W - 1, x2), min(H - 1, y2)
        w, h = x2 - x1, y2 - y1
        if min(w, h) < min_side:
            continue
        if conf > best_conf:
            best_conf = conf
            best = BBoxXYWH(x=x1, y=y1, w=w, h=h)

    if best is None:
        return None
    return ROIResult(bbox=best, method="dnn", confidence=best_conf)
