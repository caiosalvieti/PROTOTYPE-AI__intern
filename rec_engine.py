from __future__ import annotations
import os
import logging
from typing import List, Dict, Any, Set, Optional
import pandas as pd
import numpy as np

DEFAULT_KB = "DATA/products_kb.csv"

# Daily usage estimates by product "form" (ml/day)
DAILY_USAGE = {
    "cleanser": 1.5,
    "serum": 0.6,
    "moisturizer": 1.0,
    "exfoliant": 0.3,
    "mask": 0.3,
    "spf": 1.2,
    "device": 0.0,
}

PLAN_TIERS = {
    "Starter": dict(weeks=2,  upsell=False, need={"cleanser": 1, "moisturizer": 1, "spf": 1}),
    "Core":    dict(weeks=6,  upsell=True,  need={"cleanser": 1, "moisturizer": 1, "spf": 1}),
    "Intense": dict(weeks=10, upsell=True,  need={"cleanser": 1, "moisturizer": 1, "spf": 1, "serum": 1}),
}

FORMS_ORDER = ["cleanser", "serum", "moisturizer", "spf", "exfoliant", "mask", "device"]

# photo severity + safety knobs
THRESH = dict(
    need_mild=0.50,   # include when ≥ 0.50
    need_strong=0.62, # stronger push
    sensitive=0.60,   # sensitivity gate
)
SAFE_COMEDO_MAX = 2  # acne-prone: max allowed comedogenicity (0–5)

# photo-driven thresholds (aligned with main.py)
RED_HIGH = 0.62
SHINE_HIGH = 0.58
TEXTURE_HIGH = 250.0
TEXTURE_MED  = 180.0

# ----------------- small utils -----------------
def _safe_list(cell: Any) -> List[str]:
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    s = str(cell).strip()
    if not s:
        return []
    s = s.replace("|", ",").replace(";", ",")
    return [x.strip().lower() for x in s.split(",") if x.strip()]

def _safe_set(cell: Any) -> Set[str]:
    return set(_safe_list(cell))

def _covers_skin_type(row_types: Set[str], user_skin_type: str) -> bool:
    if not row_types or "all" in row_types:
        return True
    if user_skin_type == "combo":
        return ("combo" in row_types) or ("normal" in row_types)
    return user_skin_type in row_types

def _pick_size(available_ml: List[float], target_weeks: int, form: str, prefer_upsell: bool, kb_row: pd.Series) -> float:
    daily = DAILY_USAGE.get(form, 0.6)
    need_ml = daily * 7 * target_weeks
    avail_sorted = sorted([float(x) for x in available_ml if pd.notna(x)])
    chosen = None
    for ml in avail_sorted:
        if ml >= need_ml:
            chosen = ml
            break
    if chosen is None:
        chosen = avail_sorted[-1] if avail_sorted else 0.0
    if prefer_upsell and avail_sorted:
        try:
            idx = avail_sorted.index(chosen)
            if idx + 1 < len(avail_sorted):
                upsell_tier = str(kb_row.get("upsell_tier", "")).upper()
                if upsell_tier in ("M", "L"):
                    chosen = next_ml
        except Exception:
            pass
    return float(chosen)

def _infer_form(row: pd.Series) -> str:
    name = str(row.get("name") or row.get("product_name") or "").lower()
    category = str(row.get("category") or "").lower()
    is_device = bool(row.get("device")) or str(row.get("form", "")).lower() == "device"
    hay = f"{name} {category}"
    if is_device or "luna" in hay or "ufo" in hay or "bear" in hay or "device" in hay: return "device"
    if "cleanser" in hay or "wash" in hay: return "cleanser"
    if "serum" in hay or ("treatment" in hay and "serum" in hay): return "serum"
    if "moisturizer" in hay or "cream" in hay or "gel-cream" in hay: return "moisturizer"
    if "spf" in hay or "sunscreen" in hay: return "spf"
    if "exfol" in hay or "aha" in hay or "bha" in hay or "pha" in hay: return "exfoliant"
    if "mask" in hay: return "mask"
    return ""

# RecEngine 
class RecEngine:
    def __init__(self, kb_path: str = DEFAULT_KB):
        self.logger = logging.getLogger("RecEngine")
        
        # Fallback to Mock KB if file missing
        if not os.path.isfile(kb_path):
            self.logger.warning(f"KB not found at {kb_path}. Using MOCK KB.")
            self.kb = self._create_mock_kb()
        else:
            kb = pd.read_csv(kb_path)
            self.kb = self._normalize_kb(kb)

    def _create_mock_kb(self) -> pd.DataFrame:
        data = [
            {"name": "LUNA 4 Sensitive", "form": "device", "skin_types": "sensitive,dry", "price_usd": 199},
            {"name": "Micro-Foam Cleanser", "form": "cleanser", "skin_types": "all", "price_usd": 49},
            {"name": "Serum Serum Serum", "form": "serum", "skin_types": "all", "actives": "hyaluronic_acid", "price_usd": 59},
            {"name": "Daily Sunscreen", "form": "spf", "skin_types": "all", "price_usd": 35}
        ]
        return self._normalize_kb(pd.DataFrame(data))

    def _normalize_kb(self, kb: pd.DataFrame) -> pd.DataFrame:
        if "product_name" not in kb.columns and "name" in kb.columns:
            kb["product_name"] = kb["name"]
        if "name" not in kb.columns and "product_name" in kb.columns:
            kb["name"] = kb["product_name"]
        if "product_name" not in kb.columns: kb["product_name"] = ""

        if "sku" not in kb.columns and "id" in kb.columns: kb["sku"] = kb["id"]
        if "sku" not in kb.columns: kb["sku"] = ""

        for c in ["form","usage","upsell_tier","skin_types","concerns","actives",
            "brand","tier","category","link","fragrance_free","comedogenicity","contra",
            "size_ml","price_usd"]:
            if c not in kb.columns: kb[c] = ""

        for c in ["product_name","name","form","tier","brand","sku","category","usage","upsell_tier"]:
            kb[c] = kb[c].astype(str).str.strip().str.lower()
        for c in ["size_ml","price_usd","comedogenicity"]:
            kb[c] = pd.to_numeric(kb[c], errors="coerce")
        
        kb["fragrance_free"] = (
            kb["fragrance_free"].fillna(0).astype(str).str.strip().str.lower()
            .isin(["1","true","yes","y"])
        )

        kb["skin_types"] = kb["skin_types"].map(_safe_set)
        kb["concerns"]   = kb["concerns"].map(_safe_set)
        kb["actives"]    = kb["actives"].map(_safe_set)
        kb["contra"]     = kb["contra"].map(_safe_set)

        mask_missing_form = kb["form"].astype(str).str.strip().eq("")
        if mask_missing_form.any():
            kb.loc[mask_missing_form, "form"] = kb[mask_missing_form].apply(_infer_form, axis=1)

        return kb

    def _to_float(self, v) -> float:
        try:
            return float(v)
        except Exception:
            try:
                arr = np.asarray(v, dtype=float).ravel()
                return float(arr[0]) if arr.size else 0.0
            except Exception:
                return 0.0

    def _concern_weights(self, profile: Dict[str, Any], feats: Dict[str, float]) -> Dict[str, float]:
        s = {k: float(v) for k, v in (profile.get("scores") or {}).items()}
        w: Dict[str, float] = {}
        
        w["oil"] = w["shine"] = s.get("oiliness", 0.0)
        w["texture"] = s.get("texture", 0.0)
        w["redness"] = s.get("redness", 0.0)
        w["sensitivity"] = s.get("sensitivity", 0.0)
        
        # --- ARCHITECT FIX: Handle "hydration" vs "dryness" mismatch ---
        # scores.py might output 'hydration' (meaning dryness score) or 'dryness'
        dry_val = s.get("dryness", s.get("hydration", 0.0))
        w["hydration"] = dry_val
        w["barrier"] = max(dry_val * 0.7, s.get("sensitivity", 0.0) * 0.3)
        
        w["clogged_pores"] = max(s.get("oiliness", 0.0), s.get("texture", 0.0) * 0.6)
        w["uv"] = 0.5

        for item in (profile.get("prioritized_concerns") or []):
            if isinstance(item, (list, tuple)) and item:
                c = item[0]
                if c == "oil_control": c = "oil"
                val = float(item[1]) if len(item) > 1 else 1.2
                w[c] = max(w.get(c, 0.0), val)
        
        # Override with CV flags
        if feats.get("global_shn", 0) > SHINE_HIGH:
            w["acne"] = max(w.get("acne", 0.0), 0.9)
        if feats.get("global_txt", 0) > TEXTURE_HIGH:
            w["texture"] = max(w.get("texture", 0.0), 1.0)
        if feats.get("global_red", 0) > RED_HIGH:
            w["redness"] = max(w.get("redness", 0.0), 0.9)
            w["barrier"] = max(w.get("barrier", 0.0), 0.6)
            
        return w

    def _severity_for_row(self, row: pd.Series, weights: Dict[str, float]) -> float:
        row_concerns = row.get("concerns") or set()
        if not row_concerns: return 0.25
        return max((weights.get(c, 0.0) for c in row_concerns), default=0.0)

    def _score_row(self, row: pd.Series, profile: Dict[str, Any], feats: Dict[str, float]) -> float:
        st = profile.get("skin_type", "")
        weights = self._concern_weights(profile, feats)

        base_by_form = {
            "serum": 0.90, "moisturizer": 0.85, "spf": 0.80, "cleanser": 0.60,
            "exfoliant": 0.55, "mask": 0.30, "device": 0.25, "treatment": 0.90
        }
        form = str(row.get("form", "")).lower()
        score = base_by_form.get(form, 0.40)

        if _covers_skin_type(set(row["skin_types"]), st):
            score += 0.40
        elif row["skin_types"]:
            score -= 0.10

        sev = self._severity_for_row(row, weights)
        score += 0.70 * sev

        # --- ARCHITECT FIX: Stronger Sensitivity Penalty ---
        sensitivity = weights.get("sensitivity", 0.0)
        if sensitivity >= THRESH["sensitive"] and not bool(row.get("fragrance_free", False)):
            score -= 1.0  # Harder penalty (was 0.5)

        acne_prone = bool(profile.get("acne_prone")) or (weights.get("oil", 0) >= 0.65)
        try:
            if acne_prone and float(row.get("comedogenicity") or 0) > SAFE_COMEDO_MAX:
                score -= 0.50
        except Exception:
            pass

        if profile.get("pregnant", False) and ("retinoid_pregnancy" in (row.get("contra") or set())):
            score -= 9.0

        return float(score)

    def _pick_exfoliant_candidates(self, kb: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
        oil = weights.get("oil", 0.0)
        dry = weights.get("hydration", 0.0)
        if oil >= dry:
            return kb[(kb["form"] == "exfoliant") & kb["actives"].fillna("").astype(str).str.contains(r"\bbha\b|salicylic", regex=True)]
        else:
            return kb[(kb["form"] == "exfoliant") & kb["actives"].fillna("").astype(str).str.contains(r"\baha\b|\bpha\b|lactic|mandelic", regex=True)]

    # --- ARCHITECT ADDITION: Device Customization ---
    def _customize_device(self, item: Dict, weights: Dict[str, float]) -> Dict:
        """Injects usage instructions for Foreo devices based on user metrics."""
        name = item.get("name", "").lower()
        sens = weights.get("sensitivity", 0.0)
        oil = weights.get("oil", 0.0)
        
        usage = item.get("usage", "")
        
        if "luna" in name:
            if sens > 0.6:
                usage = "Gentle Mode (Intensity 3-4)"
                item["reason"] += "; Adjusted for sensitivity"
            elif oil > 0.7:
                usage = "Deep Cleanse (Intensity 10)"
            else:
                usage = "Daily Mode (Intensity 8)"
        
        elif "ufo" in name:
            if sens > 0.5:
                usage = "Cryo + Green LED"
            else:
                usage = "Thermo + Red LED"
                
        item["usage"] = usage
        return item

    def recommend(self, feats: dict, profile: dict, tier: str = "Core", include_device: bool = True, top_k_per_type: int = 1) -> dict:
        tier = tier if tier in PLAN_TIERS else "Core"
        cfg = PLAN_TIERS[tier]
        target_weeks = cfg["weeks"]; prefer_upsell = cfg["upsell"]; need = dict(cfg["need"])

        if int(feats.get("qa_fail", 0)) == 1:
            return dict(plan="QA only", target_weeks=0, items=[], reasons={"qa": ["Photo QA fail"]})

        kb = self.kb.copy()
        
        # Defensive set casting
        for col in ("skin_types", "concerns", "actives", "contra"):
            if not kb[col].map(lambda x: isinstance(x, set)).all():
                kb[col] = kb[col].map(_safe_set)

        stype = profile.get("skin_type", "")
        if stype:
            kb = kb[kb["skin_types"].map(lambda s: (len(s) == 0) or (stype in s) or ("all" in s) or (stype == "combo" and ("normal" in s or "combo" in s)))]

        scores = kb.apply(lambda r: self._to_float(self._score_row(r, profile, feats)), axis=1)
        kb = kb.assign(__score=scores)

        by_type: Dict[str, List[Dict[str, Any]]] = {}
        reasons: Dict[str, List[str]] = {}
        weights = self._concern_weights(profile, feats)

        def pick_for(form: str | List[str], k: int, extra_filter=None, min_score: float = 0.0):
            forms = [form] if isinstance(form, str) else form
            cand = kb[kb["form"].isin(forms)].copy()
            if extra_filter: cand = extra_filter(cand)
            if cand.empty: return
            
            cand = cand.sort_values("__score", ascending=False)
            if min_score > 0: cand = cand[cand["__score"] >= min_score]
            if cand.empty: return

            chosen_rows = cand.head(max(1, k))
            chosen = []
            for _, row in chosen_rows.iterrows():
                # Size logic (simplified)
                size_ml = float(row.get("size_ml") or 0.0)
                
                item = dict(
                    sku=row.get("sku", ""),
                    name=row.get("product_name", ""),
                    brand=row.get("brand", ""),
                    form=str(row.get("form", "")).lower(),
                    size_ml=size_ml,
                    usage=row.get("usage", ""),
                    concerns=list(row.get("concerns") or []),
                    price_usd=row.get("price_usd", None),
                    fragrance_free=bool(row.get("fragrance_free", False)),
                    reason=f"Matches {row.get('skin_types', {'all'})} skin",
                )
                
                # --- ARCHITECT INJECTION ---
                if "device" in forms:
                    item = self._customize_device(item, weights)
                
                chosen.append(item)

            by_type[forms[0] if isinstance(form, str) else forms[0]] = chosen[:max(1, top_k_per_type)]

        # Run logic
        for f, k in need.items(): pick_for(f, k)
        
        if max(weights.get("texture", 0), weights.get("oil", 0)) >= THRESH["need_mild"]:
            pick_for("exfoliant", 1, extra_filter=lambda df: self._pick_exfoliant_candidates(df, weights))

        if include_device:
            pick_for("device", 1)

        items: List[Dict[str, Any]] = []
        for f in FORMS_ORDER:
            items += by_type.get(f, [])

        return dict(
            plan=tier,
            target_weeks=target_weeks,
            skin_type=stype or profile.get("skin_type", "combo"),
            items=items,
            by_type=by_type,
            reasons=reasons,
        )