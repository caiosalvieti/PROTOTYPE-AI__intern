# yolo_roi.py
from typing import Optional, Tuple
from pathlib import Path  # <-- Novo Importe
import numpy as np
from ultralytics import YOLO
import torch

DEFAULT_YOLO_WEIGHTS = "yolov8n.pt"  


def _get_device():
    if torch.cuda.is_available():
        return 0  # GPU 0
    return "cpu"


class YoloRoiDetector:
 
    def __init__(
        self,
        # Use o caminho absoluto como default
        weights: str = DEFAULT_YOLO_WEIGHTS_PATH, # <-- CORREÇÃO AQUI
        conf: float = 0.25,
        iou: float = 0.45,
        device: Optional[str] = None,
    ):
        self.device = device if device is not None else _get_device()
        
        # O self.model agora carrega o modelo do caminho absoluto
        self.model = YOLO(weights) 
        self.conf = conf
        self.iou = iou

    def detect_roi_bbox(self, img_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Returns (x_min, y_min, x_max, y_max) in image coordinates, or None if no detection.
        """
        # Ultralytics handles numpy BGR images directly
        results = self.model.predict(
            img_bgr,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )

        res = results[0]
        boxes = res.boxes

        if boxes is None or len(boxes) == 0:
            return None

        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)

        # TODO 
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        idx = int(areas.argmax())
        x_min, y_min, x_max, y_max = xyxy[idx]

        return int(x_min), int(y_min), int(x_max), int(y_max)


def crop_to_bbox(img_bgr: np.ndarray,
                 bbox: Tuple[int, int, int, int],
                 padding: float = 0.05) -> np.ndarray:
    """
    Crops image to bbox with optional padding (percentage of bbox size).
    """
    h, w = img_bgr.shape[:2]
    x_min, y_min, x_max, y_max = bbox

    pad_x = int((x_max - x_min) * padding)
    pad_y = int((y_max - y_min) * padding)

    x_min = max(0, x_min - pad_x)
    y_min = max(0, y_min - pad_y)
    x_max = min(w, x_max + pad_x)
    y_max = min(h, y_max + pad_y)

    return img_bgr[y_min:y_max, x_min:x_max]
