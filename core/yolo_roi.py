# core/yolo_roi.py
from __future__ import annotations
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import numpy as np
from ultralytics import YOLO
import torch
import time

# --- INTEGRAÇÃO CRÍTICA DE CAMINHO ---
# Sobe dois níveis (sai do .py, sai da pasta core/) para chegar na raiz do projeto
ROOT = Path(__file__).resolve().parent.parent 
DEFAULT_YOLO_WEIGHTS_PATH = str(ROOT / "yolov8n.pt") 

# --- Importa Pre-Processamento ---
# Importação relativa, ajustada se necessário, dependendo de como você estrutura.
# Se preprocessing.py está na raiz, use 'import preprocessing as pp' e ajuste o path.
# Se estiver na pasta core/, use 'from . import preprocessing as pp' (mais complexo para rodar direto)
# Para fins de demonstração, assumimos que está importando o módulo raiz.
# Se 'preprocessing' não for reconhecido, ajuste para o seu caminho real.
import preprocessing as pp # <-- ASSUMIMOS QUE ESTE IMPORT FUNCIONA NO SEU AMBIENTE

DEFAULT_YOLO_WEIGHTS = DEFAULT_YOLO_WEIGHTS_PATH  

# --- Lógica de Disponibilidade (mantida) ---
YOLO_AVAILABLE: bool = False
YOLO_IMPORT_ERROR: Optional[str] = None
_YOLO_MODEL: Optional[YOLO] = None
_DEVICE = "cpu"

def _get_device():
    if torch.cuda.is_available():
        return 0  # GPU 0
    return "cpu"

try:
    _DEVICE = _get_device()
    # Tenta carregar o modelo APENAS UMA VEZ na inicialização do módulo
    _YOLO_MODEL = YOLO(DEFAULT_YOLO_WEIGHTS)
    YOLO_AVAILABLE = True

except Exception as e:
    _YOLO_MODEL = None
    _DEVICE = "cpu"
    YOLO_AVAILABLE = False
    YOLO_IMPORT_ERROR = repr(e)

# --- CLASSE DE DETECÇÃO (usará o modelo carregado acima) ---
class YoloRoiDetector:
 
    def __init__(
        self,
        conf: float = 0.25,
        iou: float = 0.45,
        # O modelo é carregado globalmente acima, a instância da classe é leve
        # mas mantemos a assinatura para consistência.
    ):
        self.device = _DEVICE
        self.model = _YOLO_MODEL
        self.conf = conf
        self.iou = iou

    def detect_roi_bbox(self, img_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Runs full pre-processing, YOLO detection, and returns the largest bbox.
        """
        if not YOLO_AVAILABLE or self.model is None:
            return None

        # --- INTEGRAÇÃO DO PRÉ-PROCESSAMENTO ---
        # 1. Aplica CLAHE + Redimensionamento para 640x640
        img_preprocessed = pp.preprocess_image(img_bgr) 
        # --- FIM DA INTEGRAÇÃO ---

        results = self.model.predict(
            img_preprocessed, # Roda a inferência no array 640x640
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )

        res = results[0]
        boxes = res.boxes

        if boxes is None or len(boxes) == 0:
            return None

        # Retorna coordenadas (xyxy) do array 640x640
        xyxy = boxes.xyxy.cpu().numpy()
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        idx = int(areas.argmax())
        x_min, y_min, x_max, y_max = xyxy[idx]

        # IMPORTANTE: Se o recorte final for na imagem original,
        # você deve escalar estas coordenadas de volta aqui. 
        # Assumindo que o pipeline subsequente lida com o escalonamento, 
        # retornamos a bbox do array 640x640.
        return int(x_min), int(y_min), int(x_max), int(y_max)


def skinaizer_model_core(img_rgb: np.ndarray) -> Dict[str, Any]:
    """ Wrapper que o ui_app.py chama. """
    detector = YoloRoiDetector()
    if not YOLO_AVAILABLE or detector.model is None:
        return {
            "bbox": None,
            "timings": {},
            "yolo_available": False,
            "yolo_error": YOLO_IMPORT_ERROR,
        }

    t0 = time.perf_counter()
    # Chama o detector que agora aplica o pré-processamento internamente
    bbox = detector.detect_roi_bbox(img_rgb)
    t1 = time.perf_counter()

    timings = {
        "yolo_ms": (t1 - t0) * 1000.0,
        "total_ms": (t1 - t0) * 1000.0,
    }
    
    # Se o bbox for None, retorna bbox=None, o ui_app.py usará o fallback.

    return {
        "bbox": bbox,
        "timings": timings,
        "yolo_available": True,
        "yolo_error": None,
    }