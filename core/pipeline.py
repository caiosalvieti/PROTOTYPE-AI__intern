# core/pipeline.py
from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from joblib import load

from core.schemas import AnalysisResult, ROIResult
from core.roi.detect import detect_roi
from core.vision.features import imread_rgb, gray_world, extract_features, save_debug_panel
from core.coach.profile import infer_skin_profile
from core.coach.rules import build_coach_payload

# keep your RecEngine as-is (from rec_engine.py)
from rec_engine import RecEngine

ROOT = Path(".").resolve()
DATA = ROOT / "DATA"
INTERIM = DATA / "interim"
INTERIM.mkdir(parents=True, exist_ok=True)

def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

@lru_cache(maxsize=1)
def get_rec_engine() -> RecEngine:
    return RecEngine(str(DATA / "products_kb.csv"))

def _crop_face(rgb: np.ndarray, roi: ROIResult) -> Tuple[np.ndarray, Tuple[int,int,int,int]]:
    if not roi.bbox:
        raise ValueError("ROI missing bbox")
    x, y, w, h = roi.bbox.x, roi.bbox.y, roi.bbox.w, roi.bbox.h
    face = rgb[y:y+h, x:x+w]
    if face.size == 0:
        raise ValueError("Empty face crop")
    return face, (x, y, w, h)

def analyze_image_path(
    image_path: str,
    model_path: Optional[str] = None,
    save_debug: bool = True,
    max_dim: int = 800,
    min_side: int = 120,
) -> AnalysisResult:
    try:
        rgb = gray_world(imread_rgb(image_path))
        roi = detect_roi(rgb, max_dim=max_dim, min_side=min_side)

        face, box_xywh = _crop_face(rgb, roi)
        feats, _zones = extract_features(face)

        debug_path = None
        if save_debug:
            dbg = INTERIM / f"debug_{_ts()}.jpg"
            save_debug_panel(rgb, box_xywh, str(dbg))
            debug_path = str(dbg)

        # optional model inference
        model_pred = None
        if model_path and Path(model_path).is_file():
            bundle = load(model_path)
            model, cols = bundle["model"], bundle["features"]
            X = pd.DataFrame([{k: feats.get(k, 0.0) for k in cols}])
            model_pred = model.predict(X)[0]

        profile = infer_skin_profile(feats)
        plan = get_rec_engine().recommend(feats, profile, tier="Core", include_device=True)
        coach = build_coach_payload(feats, profile, plan)

        return AnalysisResult(
            ok=True,
            roi=roi,
            feats=feats,
            profile=profile,
            plan=plan,
            coach=coach,
            model_pred=model_pred,
            debug_path=debug_path,
        )

    except Exception as e:
        return AnalysisResult(ok=False, roi=ROIResult(bbox=None, method="none"), error=repr(e))
