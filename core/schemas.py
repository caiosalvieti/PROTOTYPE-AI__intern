# core/schemas.py
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Literal, Optional

@dataclass
class WarningItem:
    code: str
    message: str

@dataclass
class BBoxXYWH:
    x: int
    y: int
    w: int
    h: int

@dataclass
class ROIResult:
    bbox: Optional[BBoxXYWH]
    method: Literal["yolo", "mediapipe", "haar", "dnn", "fallback_center", "none"]
    confidence: float = 0.0
    warnings: List[WarningItem] = field(default_factory=list)

@dataclass
class AnalysisResult:
    ok: bool
    roi: ROIResult
    feats: Dict[str, Any] = field(default_factory=dict)
    profile: Dict[str, Any] = field(default_factory=dict)
    plan: Dict[str, Any] = field(default_factory=dict)
    coach: Dict[str, Any] = field(default_factory=dict)
    model_pred: Any = None
    debug_path: Optional[str] = None
    error: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        return _json_sanitize(asdict(self))

def _json_sanitize(x: Any) -> Any:
    # converts numpy scalars, Path, etc into JSON-safe primitives
    try:
        import numpy as np  # local import
        if isinstance(x, (np.integer,)): return int(x)
        if isinstance(x, (np.floating,)): return float(x)
        if isinstance(x, (np.bool_,)): return bool(x)
    except Exception:
        pass

    if isinstance(x, dict):
        return {str(k): _json_sanitize(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_json_sanitize(v) for v in x]
    if isinstance(x, tuple):
        return [_json_sanitize(v) for v in x]
    return x
