from __future__ import annotations
import json, os
from typing import Dict, Any, Tuple
import numpy as np

# Optional, only used for auto-calibration
try:
    import pandas as pd
    _PANDAS_OK = True
except Exception:
    _PANDAS_OK = False

#  paths for auto-calibration 
FEATURES_CSV = "DATA/metadata/features.csv"
Q_JSON       = "DATA/metadata/feature_quantiles.json"

# sensible fallbacks if no dataset yet
DEFAULT_Q = {
    "shn": {"low": 0.44, "high": 0.58},           # shine (global_shn) 0..1
    "txt": {"med": 180.0, "high": 230.0},         # texture (global_txt) Laplacian var
    "red": {"base": 0.45, "high": 0.62},          # redness (global_red) 0..1
}

# calibration utils 
def _load_quantiles() -> Dict[str, Dict[str, float]]:
    """Try JSON -> CSV -> fallback defaults."""
    # 1) cached JSON
    if os.path.isfile(Q_JSON):
        try:
            with open(Q_JSON, "r") as f:
                q = json.load(f)
            return q
        except Exception:
            pass

    # compute from CSV
    if _PANDAS_OK and os.path.isfile(FEATURES_CSV):
        try:
            df = pd.read_csv(FEATURES_CSV)
            # keep rows where no error
            if "error" in df.columns:
                df = df[df["error"].fillna("") == ""]
            shn = df.get("global_shn")
            txt = df.get("global_txt")
            red = df.get("global_red")
            if shn is not None and txt is not None and red is not None:
                q = {
                    "shn": {
                        "low":  float(shn.quantile(0.35)),
                        "high": float(shn.quantile(0.75)),
                    },
                    "txt": {
                        "med":  float(txt.quantile(0.60)),
                        "high": float(txt.quantile(0.80)),
                    },
                    "red": {
                        "base": float(red.quantile(0.45)),
                        "high": float(red.quantile(0.75)),
                    },
                }
                # cache for next runs
                os.makedirs(os.path.dirname(Q_JSON), exist_ok=True)
                with open(Q_JSON, "w") as f:
                    json.dump(q, f)
                return q
        except Exception:
            pass

    # fallback
    return DEFAULT_Q

_Q = _load_quantiles()  # loaded at import

# math 
def _nz(v, default=0.0) -> float:
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return default
        return float(v)
    except Exception:
        return default

def _rescale(v: float, lo: float, hi: float) -> float:
    """Map v in [lo,hi] to [0,1] with clipping."""
    if hi <= lo:
        return 0.0
    return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))

# main API 
def infer_skin_profile(feats: Dict[str, Any], q: Dict[str, Dict[str, float]] | None = None) -> Dict[str, Any]:
    """
    Input: feats from extract_features(...)
    Output:
      {
        skin_type: "oily"|"dry"|"combo",
        scores: {oiliness, dryness, redness, texture, sensitivity, acne},
        prioritized_concerns: List[(concern, score)],
        concerns: List[str],           # names with score >= 0.55
        flags: List[str],              # e.g. ["sensitive"]
        acne_prone: bool
      }
    """
    q = q or _Q

    shn = _nz(feats.get("global_shn"), 0.50)   # 0.1
    txt = _nz(feats.get("global_txt"), 150.0)  # Laplacian var
    red = _nz(feats.get("global_red"), 0.35)   # 0.1

    # Quantile-aware rescaling
    oiliness = _rescale(shn, q["shn"]["low"], q["shn"]["high"])
    texture_norm = _rescale(txt, q["txt"]["med"], q["txt"]["high"])
    redness = _rescale(red, q["red"]["base"], q["red"]["high"])

    # Derived signals
    dryness = float(np.clip(0.7 * (1.0 - oiliness) + 0.3 * texture_norm, 0, 1))
    sensitivity = float(np.clip(redness, 0, 1))
    acne = float(np.clip(0.5 * oiliness + 0.5 * texture_norm, 0, 1))

    # Skin type decision
    if oiliness >= 0.6 and dryness < 0.6:
        skin_type = "oily"
    elif dryness >= 0.6 and oiliness < 0.6:
        skin_type = "dry"
    else:
        skin_type = "combo"

    # Concern scores (0..1)
    concerns_scores = {
        "oiliness":  oiliness,
        "hydration": dryness,
        "barrier":   float(np.clip(0.5 * dryness + 0.5 * sensitivity, 0, 1)),
        "texture":   texture_norm,
        "redness":   redness,
        "sensitivity": sensitivity,
        "acne":      acne,
    }

    # Priority list + binary flags
    prioritized = sorted(concerns_scores.items(), key=lambda kv: kv[1], reverse=True)
    concerns = [k for k, v in prioritized if v >= 0.55][:4]  # top hiT
    flags = []
    if sensitivity >= 0.6: flags.append("sensitive")
    acne_prone = acne >= 0.6

    return {
        "skin_type": skin_type,
        "scores": {k: float(v) for k, v in concerns_scores.items()},
        "prioritized_concerns": prioritized,
        "concerns": concerns,
        "flags": flags,
        "acne_prone": bool(acne_prone),
    }

#  recompute & write quantiles manually
def refresh_quantiles(features_csv: str = FEATURES_CSV, out_json: str = Q_JSON) -> Tuple[bool, Dict[str, Dict[str, float]]]:
    """Recompute dataset quantiles and save JSON. Returns (ok, quantiles)."""
    if not _PANDAS_OK or not os.path.isfile(features_csv):
        return False, DEFAULT_Q
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
    return True, q

