# scores.py
from __future__ import annotations
import numpy as np

# Tunable thresholds
THRESH = dict(
    oil_hi=0.58,   # global_shn ~ 0..1
    dry_hi=0.62,   # derived from low shine + higher texture
    red_hi=0.58,   # global_red ~ 0.3..0.7 typical
    texture_hi=0.50,  # normalized texture score
)

def _nz(v, default=0.0):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return default
        return float(v)
    except Exception:
        return default

def _norm_texture(global_txt: float) -> float:
    """
    Heuristic normalization for Laplacian variance.
    ~0 at very smooth, approaches 1 for high values.
    """
    x = _nz(global_txt, 0.0)
    return float(np.tanh(x / 500.0))

def infer_skin_profile(feats: dict) -> dict:
    """
    Input: feats from extract_features(...)
    Output: dict with skin_type guess and concern scores 0..1
    """
    red = _nz(feats.get("global_red"), 0.35)     # 0..1
    shn = _nz(feats.get("global_shn"), 0.50)     # 0..1
    txt = _nz(feats.get("global_txt"), 150.0)    # Laplacian var
    texture_score = _norm_texture(txt)

    oiliness = np.clip(shn, 0, 1)
    dryness  = np.clip(0.7 - shn + 0.3*texture_score, 0, 1)
    redness  = np.clip((red - 0.45) / 0.25, 0, 1)
    sensitivity = np.clip(redness * 0.9, 0, 1)

    if oiliness >= THRESH["oil_hi"] and dryness < THRESH["dry_hi"]:
        skin_type = "oily"
    elif dryness >= THRESH["dry_hi"] and oiliness < THRESH["oil_hi"]:
        skin_type = "dry"
    else:
        skin_type = "combo"

    concerns = {
        "hydration": float(dryness),
        "barrier":   float(np.clip(dryness*0.7 + sensitivity*0.3, 0, 1)),
        "oil_control": float(oiliness),
        "texture":   float(texture_score),
        "redness":   float(redness),
        "sensitivity": float(sensitivity),
    }

    prioritized = sorted(concerns.items(), key=lambda kv: kv[1], reverse=True)

    return dict(
        skin_type=skin_type,
        scores=dict(
            oiliness=float(oiliness),
            dryness=float(dryness),
            redness=float(redness),
            texture=float(texture_score),
            sensitivity=float(sensitivity),
        ),
        prioritized_concerns=prioritized,
    )
