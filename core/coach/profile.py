# core/coach/profile.py
from __future__ import annotations

import os, json
from functools import lru_cache
from typing import Any, Dict, Tuple

import numpy as np

try:
    import pandas as pd
    _PANDAS_OK = True
except Exception:
    _PANDAS_OK = False

FEATURES_CSV = "DATA/metadata/features.csv"
Q_JSON       = "DATA/metadata/feature_quantiles.json"

DEFAULT_Q = {
    "shn": {"low": 0.44, "high": 0.58},
    "txt": {"med": 180.0, "high": 230.0},
    "red": {"base": 0.45, "high": 0.62},
}

@lru_cache(maxsize=1)
def load_quantiles() -> Dict[str, Dict[str, float]]:
    if os.path.isfile(Q_JSON):
        try:
            with open(Q_JSON, "r") as f:
                return json.load(f)
        except Exception:
            pass

    if _PANDAS_OK and os.path.isfile(FEATURES_CSV):
        try:
            df = pd.read_csv(FEATURES_CSV)
            if "error" in df.columns:
                df = df[df["error"].fillna("") == ""]
            shn = df.get("global_shn")
            txt = df.get("global_txt")
            red = df.get("global_red")
            if shn is not None and txt is not None and red is not None:
                q = {
                    "shn": {"low": float(shn.quantile(0.35)), "high": float(shn.quantile(0.75))},
                    "txt": {"med": float(txt.quantile(0.60)), "high": float(txt.quantile(0.80))},
                    "red": {"base": float(red.quantile(0.45)), "high": float(red.quantile(0.75))},
                }
                os.makedirs(os.path.dirname(Q_JSON), exist_ok=True)
                with open(Q_JSON, "w") as f:
                    json.dump(q, f)
                return q
        except Exception:
            pass

    return DEFAULT_Q

def _nz(v, default=0.0) -> float:
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return default
        return float(v)
    except Exception:
        return default

def _rescale(v: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))

def infer_skin_profile(feats: Dict[str, Any], q: Dict[str, Dict[str, float]] | None = None) -> Dict[str, Any]:
    q = q or load_quantiles()

    shn = _nz(feats.get("global_shn"), 0.50)
    txt = _nz(feats.get("global_txt"), 150.0)
    red = _nz(feats.get("global_red"), 0.35)

    oiliness = _rescale(shn, q["shn"]["low"], q["shn"]["high"])
    texture  = _rescale(txt, q["txt"]["med"], q["txt"]["high"])
    redness  = _rescale(red, q["red"]["base"], q["red"]["high"])

    dryness     = float(np.clip(0.7 * (1.0 - oiliness) + 0.3 * texture, 0, 1))
    sensitivity = float(np.clip(redness, 0, 1))
    acne        = float(np.clip(0.5 * oiliness + 0.5 * texture, 0, 1))

    if oiliness >= 0.6 and dryness < 0.6:
        skin_type = "oily"
    elif dryness >= 0.6 and oiliness < 0.6:
        skin_type = "dry"
    else:
        skin_type = "combo"

    scores = {
        "oiliness": oiliness,
        "dryness": dryness,
        "hydration": dryness,   # alias to keep old code working
        "texture": texture,
        "redness": redness,
        "sensitivity": sensitivity,
        "barrier": float(np.clip(0.5 * dryness + 0.5 * sensitivity, 0, 1)),
        "acne": acne,
    }

    prioritized = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    concerns = [k for k, v in prioritized if v >= 0.55][:4]
    flags = ["sensitive"] if sensitivity >= 0.6 else []
    acne_prone = acne >= 0.6

    return {
        "skin_type": skin_type,
        "scores": {k: float(v) for k, v in scores.items()},
        "prioritized_concerns": [(k, float(v)) for k, v in prioritized],
        "concerns": concerns,
        "flags": flags,
        "acne_prone": bool(acne_prone),
    }

def refresh_quantiles(features_csv: str = FEATURES_CSV, out_json: str = Q_JSON) -> Tuple[bool, Dict[str, Dict[str, float]]]:
    if not _PANDAS_OK or not os.path.isfile(features_csv):
        return False, DEFAULT_Q
    import pandas as pd
    df = pd.read_csv(features_csv)
    if "error" in df.columns:
        df = df[df["error"].fillna("") == ""]
    shn = df.get("global_shn"); txt = df.get("global_txt"); red = df.get("global_red")
    if shn is None or txt is None or red is None:
        return False, DEFAULT_Q

    q = {
        "shn": {"low": float(shn.quantile(0.35)), "high": float(shn.quantile(0.75))},
        "txt": {"med": float(txt.quantile(0.60)), "high": float(txt.quantile(0.80))},
        "red": {"base": float(red.quantile(0.45)), "high": float(red.quantile(0.75))},
    }
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(q, f)
    load_quantiles.cache_clear()
    return True, q
