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
                # Simple upsell logic: if tier is High/Luxury, push bigger size
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
        
        # --- Mock Data Fallback ---
        # If the CSV doesn't exist (likely in a new clone), create a mock KB in memory
        # so the internship demo doesn't crash.
        if not os.path.isfile(kb_path):
            self.logger.warning(f"KB not found at {kb_path}. Using IN-MEMORY MOCK KB.")
            self.kb = self._create_mock_kb()
            return

        kb = pd.read_csv(kb_path)
        self.kb = self._normalize_kb(kb)

    def _create_mock_kb(self) -> pd.DataFrame:
        """Creates a minimal Foreo-aligned product list for demo purposes."""
        data = [
            {"name": "LUNA 4 Sensitive", "form": "device", "skin_types": "sensitive,dry", "price_usd": 199},
            {"name": "LUNA 4 Combination", "form": "device", "skin_types": "combo,oily", "price_usd": 199},
            {"name": "Micro-Foam Cleanser", "form": "cleanser", "skin_types": "all", "price_usd": 49},
            {"name": "Serum Serum Serum", "form": "serum", "skin_types": "all", "actives": "hyaluronic_acid", "price_usd": 59},
            {"name": "Supercharged Ha+PGA", "form": "moisturizer", "skin_types": "dry,normal", "price_usd": 69},
            {"name": "UFO 2", "form": "device", "skin_types": "all", "price_usd": 279}
        ]
        return self._normalize_kb(pd.DataFrame(data))

    def _normalize_kb(self, kb: pd.DataFrame) -> pd.DataFrame:
        # Aliases & required defaults
        if "product_name" not in kb.columns and "name" in kb.columns:
            kb["product_name"] = kb["name"]
        if "name" not in kb.columns and "product_name" in kb.columns:
            kb["name"] = kb["product_name"]
        if "product_name" not in kb.columns: kb["product_name"] = ""
        
        # Ensure ID/SKU
        if "sku" not in kb.columns and "id" in kb.columns: kb["sku"] = kb["id"]
        if "sku" not in kb.columns: kb["sku"] = kb["product_name"].apply(lambda x: str(x)[:5].upper())

        # Ensure standard cols
        for c in ["form","usage","upsell_tier","skin_types","concerns","actives",
                  "brand","tier","category","link","fragrance_free","comedogenicity","contra",
                  "size_ml","price_usd"]:
            if c not in kb.columns: kb[c] = ""

        # Normalize strings
        for c in ["product_name","form","tier","brand","sku","category","usage","upsell_tier"]:
            kb[c] = kb[c].astype(str).str.strip().str.lower()
        
        # Normalize numbers
        for c in ["size_ml","price_usd","comedogenicity"]:
            kb[c] = pd.to_numeric(kb[c], errors="coerce")
            
        kb["fragrance_free"] = (
            kb["fragrance_free"]
            .fillna(0).astype(str).str.strip().str.lower()
            .isin(["1","true","yes","y"])
        )

        # Parse list-like into sets
        kb["skin_types"] = kb["skin_types"].map(_safe_set)
        kb["concerns"]   = kb["concerns"].map(_safe_set)
        kb["actives"]    = kb["actives"].map(_safe_set)
        kb["contra"]     = kb["contra"].map(_safe_set)

        # Infer missing form
        mask_missing_form = kb["form"].astype(str).str.strip().eq("")
        if mask_missing_form.any():
            kb.loc[mask_missing_form, "form"] = kb[mask_missing_form].apply(_infer_form, axis=1)
            
        return kb

    def _to_float(self, v) -> float:
        try:
            return float(v)
        except Exception:
            return 0.0

    def _concern_weights(self, profile: Dict[str, Any], feats: Dict[str, float]) -> Dict[str, float]:
        # Extract base scores
        s = {k: float(v) for k, v in (profile.get("scores") or {}).items()}
        w = {}
        
        # Map raw scores to concern weights
        w["oil"] = s.get("oiliness", 0.0)
        w["texture"] = s.get("texture", 0.0)
        w["redness"] = s.get("redness", 0.0)
        w["hydration"] = s.get("dryness", 0.0) # Assume dryness score exists
        w["sensitivity"] = s.get("sensitivity", 0.0)
        
        # Augment with Feature thresholds (Computer Vision overrides)
        if feats.get("global_shn", 0) > SHINE_HIGH:
            w["oil"] = max(w.get("oil", 0), 0.8)
            w["acne"] = 0.7
        if feats.get("global_txt", 0) > TEXTURE_HIGH:
            w["texture"] = max(w.get("texture", 0), 0.9)
        if feats.get("global_red", 0) > RED_HIGH:
            w["redness"] = max(w.get("redness", 0), 0.85)
            w["sensitivity"] = max(w.get("sensitivity", 0), 0.8)

        return w

    def _score_row(self, row: pd.Series, profile: Dict[str, Any], weights: Dict[str, float]) -> float:
        score = 0.5 # Base score
        
        # 1. Skin Type Match
        st = profile.get("skin_type", "normal")
        if _covers_skin_type(row["skin_types"], st):
            score += 0.3
        elif row["skin_types"]: # If specific types listed but don't match
            score -= 0.2

        # 2. Sensitivity Check (Standard)
        if weights.get("sensitivity", 0) > THRESH["sensitive"]:
            if not row["fragrance_free"]: score -= 0.5
            if "sensitive" in row["skin_types"]: score += 0.4

        # 3. Concern Matching
        row_concerns = row["concerns"]
        for concern, weight in weights.items():
            if concern in row_concerns:
                score += (weight * 0.5)

        return float(score)

    def _generate_reason(self, row: pd.Series, weights: Dict[str, float]) -> str:
        """
        Architectural Helper: Reverse-engineers the score to explain WHY it was picked.
        """
        reasons = []
        
        # 1. Did we pick it for sensitivity?
        if weights.get("sensitivity", 0) > THRESH["sensitive"]:
            if row.get("fragrance_free"):
                reasons.append("Fragrance-free for sensitivity")
        
        # 2. Did we pick it for a specific concern?
        # Check the top 2 concerns
        for concern, score in sorted(weights.items(), key=lambda x: x[1], reverse=True)[:2]:
            if score > 0.5 and concern in row["concerns"]:
                reasons.append(f"Targets {concern}")
        
        # 3. Fallback
        if not reasons:
            return f"Matches {row.get('skin_types', 'your')} skin type"
            
        return " & ".join(reasons)

    def _get_device_settings(self, device_name: str, weights: Dict[str, float]) -> Dict[str, str]:
        """
        FOREO-SPECIFIC LOGIC:
        Generates custom usage instructions based on skin analysis.
        """
        setting = "Standard Mode"
        reason = "Balanced routine."
        
        sens = weights.get("sensitivity", 0.0)
        oil = weights.get("oil", 0.0)
        text = weights.get("texture", 0.0)

        if "luna" in device_name.lower():
            if sens > THRESH["sensitive"]:
                setting = "Gentle Mode (Intensity 3-4)"
                reason = f"High sensitivity detected ({sens*100:.0f}%). Reduced intensity to protect barrier."
            elif oil > 0.7:
                setting = "Deep Cleansing (Intensity 10-12)"
                reason = "Higher oiliness detected. Increased intensity for T-zone breakdown."
            else:
                setting = "Daily Cleanse (Intensity 8)"
                reason = "Optimal setting for daily maintenance."
        
        elif "ufo" in device_name.lower():
            if sens > 0.5:
                setting = "Green LED + Cryotherapy"
                reason = "Cooling Cryo-mode to reduce detected redness."
            elif text > 0.6:
                setting = "Red LED + Thermotherapy"
                reason = "Heat + Red Light to stimulate collagen and smooth texture."
        
        elif "bear" in device_name.lower():
            setting = "Contour Mode"
            reason = "Microcurrent toning for definition."

        return {"setting": setting, "reason": reason}

    def recommend(self, feats: dict, profile: dict, tier: str = "Core", include_device: bool = True) -> dict:
        tier = tier if tier in PLAN_TIERS else "Core"
        cfg = PLAN_TIERS[tier]
        
        kb = self.kb.copy()
        weights = self._concern_weights(profile, feats)
        
        # --- ARCHITECT FIX: Stronger Sensitivity Penalty Wrapper ---
        def safe_score(r):
            s = self._score_row(r, profile, weights)
            # Hard penalize fragrance if sensitive (The "Firewall")
            if weights.get("sensitivity", 0) > THRESH["sensitive"] and not r["fragrance_free"]:
                s -= 2.0 
            return s

        # Calculate scores using the safe wrapper
        kb["__score"] = kb.apply(safe_score, axis=1)
        
        # Sort by score
        kb = kb.sort_values("__score", ascending=False)

        plan_items = []
        devices = []
        by_type = {}
        
        # Select products based on Tier requirements
        for form, count in cfg["need"].items():
            candidates = kb[kb["form"] == form]
            if not candidates.empty:
                selected = candidates.head(count)
                for _, row in selected.iterrows():
                    item = row.to_dict()
                    
                    # Generate dynamic reasoning
                    reason_text = self._generate_reason(row, weights)
                    
                    # Clean up for UI
                    ui_item = {
                        "name": item.get("product_name"),
                        "form": form,
                        "brand": item.get("brand"),
                        "price": item.get("price_usd"),
                        "reason": reason_text, # Dynamic reason
                        "usage": item.get("usage", "Use as directed")
                    }
                    plan_items.append(ui_item)
                    by_type.setdefault(form, []).append(ui_item)

        # Device Selection (Critical for Foreo)
        if include_device:
            dev_candidates = kb[kb["form"] == "device"]
            if dev_candidates.empty:
                # Fallback if no device in CSV
                best_device = "LUNA 4"
                dev_row = None
            else:
                # Re-sort devices specifically to ensure best match
                dev_candidates = dev_candidates.sort_values("__score", ascending=False)
                best_device = dev_candidates.iloc[0]["product_name"]
                dev_row = dev_candidates.iloc[0]
            
            # Generate Custom Settings
            dev_config = self._get_device_settings(best_device, weights)
            
            device_obj = {
                "name": best_device,
                "category": "Device",
                "setting": dev_config["setting"],
                "reason": dev_config["reason"],
                "price_usd": dev_row["price_usd"] if dev_row is not None else 0,
                "form": "device",
                "usage": dev_config["setting"]
            }
            plan_items.append(device_obj)
            devices.append(device_obj)

        return {
            "plan_tier": tier,
            "skin_type": profile.get("skin_type", "unknown"),
            "concerns": list(weights.keys()),
            "items": plan_items,  # Flat list for shopping cart
            "devices": devices,   # Specific list for "My Routine" UI
            "by_type": by_type
        }