# rec_engine.py
from __future__ import annotations
import os
from typing import List, Dict, Any
import pandas as pd

DEFAULT_KB = "DATA/products_kb.csv"

# Daily usage estimates by product "form" (ml/day)
DAILY_USAGE = {
    "cleanser": 1.5,     # AM + PM ~ 0.75 ml each
    "serum":    0.6,     # 2-3 pumps total
    "moisturizer": 1.0,  # pea/bean size
    "exfoliant": 0.3,    # 2-3x/week ~ avg per day
    "mask":     0.3,     # 2-4x/week ~ avg per day
    "spf":      1.2,     # 2mg/cm2 equivalent ~ face/neck heuristic
    "device":   0.0,     # no ml
}

PLAN_TIERS = {
    "Starter": dict(weeks=2,  upsell=False),
    "Core":    dict(weeks=6,  upsell=True),
    "Intense": dict(weeks=10, upsell=True),
}

FORMS_ORDER = ["cleanser", "serum", "moisturizer", "spf", "exfoliant", "mask", "device"]

def _safe_list(cell: Any) -> List[str]:
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    s = str(cell).strip()
    if not s: return []
    if ";" in s: return [x.strip().lower() for x in s.split(";") if x.strip()]
    return [x.strip().lower() for x in s.split(",") if x.strip()]

def _covers_skin_type(row_types: List[str], user_skin_type: str) -> bool:
    if not row_types: return True
    if "all" in row_types: return True
    if user_skin_type == "combo" and ("combo" in row_types or "normal" in row_types):
        return True
    return user_skin_type in row_types

def _matches_concern(row_concerns: List[str], prioritized: List[tuple]) -> float:
    if not row_concerns: return 0.25
    want = [c for (c, _) in prioritized[:3]]
    hits = sum(1 for c in want if c in row_concerns)
    return 0.25 + 0.25 * hits  # 0.25..1.0

def _pick_size(available_ml: List[float], target_weeks: int, form: str, prefer_upsell: bool, kb_row: pd.Series) -> float:
    daily = DAILY_USAGE.get(form, 0.6)
    need_ml = daily * 7 * target_weeks

    avail_sorted = sorted(available_ml)
    chosen = None
    for ml in avail_sorted:
        if ml >= need_ml:
            chosen = ml
            break
    if chosen is None:
        chosen = avail_sorted[-1] if avail_sorted else 0.0

    # Gentle upsell: only for products tagged upsell-y
    if prefer_upsell and avail_sorted:
        try:
            idx = avail_sorted.index(chosen)
            if idx + 1 < len(avail_sorted):
                next_ml = avail_sorted[idx + 1]
                upsell_tier = str(kb_row.get("upsell_tier", "")).upper()
                if upsell_tier in ("M", "L"):
                    chosen = next_ml
        except Exception:
            pass

    return float(chosen)

def _infer_form(row: pd.Series) -> str:
    """Heuristic to infer form if column missing in KB."""
    name = str(row.get("name") or row.get("product_name") or "").lower()
    category = str(row.get("category") or "").lower()
    is_device = bool(row.get("device")) or str(row.get("form","")).lower() == "device"
    hay = f"{name} {category}"
    if is_device or "luna" in hay or "device" in hay:
        return "device"
    if "cleanser" in hay or "wash" in hay:
        return "cleanser"
    if "serum" in hay or "treatment" in hay and "serum" in hay:
        return "serum"
    if "moisturizer" in hay or "cream" in hay or "gel-cream" in hay:
        return "moisturizer"
    if "spf" in hay or "sunscreen" in hay:
        return "spf"
    if "exfol" in hay or "aha" in hay or "bha" in hay or "pha" in hay:
        return "exfoliant"
    if "mask" in hay:
        return "mask"
    return ""  # unknown -> will be ranked but rarely selected

class RecEngine:
    def __init__(self, kb_path: str = DEFAULT_KB):
        if not os.path.isfile(kb_path):
            raise FileNotFoundError(f"Knowledge base not found: {kb_path}")

        # Load raw CSV
        self.kb = pd.read_csv(kb_path)
        cols = set(self.kb.columns)

        # Aliases for common names
        if "product_name" not in cols and "name" in cols:
            self.kb["product_name"] = self.kb["name"]
        if "sku" not in cols and "id" in cols:
            self.kb["sku"] = self.kb["id"]

        # Ensure required columns exist (with defaults)
        if "form" not in cols:
            self.kb["form"] = ""
        if "usage" not in cols:
            self.kb["usage"] = ""
        if "upsell_tier" not in cols:
            self.kb["upsell_tier"] = ""
        for c in ["skin_types","concerns","actives","brand","tier","category","link"]:
            if c not in self.kb.columns:
                self.kb[c] = ""

        # Normalize types
        for c in ["size_ml", "price_usd"]:
            if c in self.kb.columns:
                self.kb[c] = pd.to_numeric(self.kb[c], errors="coerce")

        if "include_device" in self.kb.columns:
            self.kb["include_device"] = self.kb["include_device"].fillna(False).astype(bool)
        else:
            self.kb["include_device"] = False

        # Text normalization
        for c in ["product_name","name","form","tier","skin_types","concerns","actives","brand","sku","category"]:
            if c in self.kb.columns:
                self.kb[c] = self.kb[c].astype(str).str.strip().str.lower()

        # If form is empty, try inferring it
        mask_missing_form = (self.kb["form"].astype(str).str.strip() == "")
        if mask_missing_form.any():
            self.kb.loc[mask_missing_form, "form"] = self.kb[mask_missing_form].apply(_infer_form, axis=1)

        # Final coercion
        self.kb["size_ml"] = pd.to_numeric(self.kb["size_ml"], errors="coerce")

    def _score_row(self, row: pd.Series, skin_type: str, prioritized: List[tuple]) -> float:
        stypes = _safe_list(row.get("skin_types"))
        concerns = _safe_list(row.get("concerns"))
        if not _covers_skin_type(stypes, skin_type):
            return 0.0
        base = 0.4 if row.get("form","") in ("serum","moisturizer","spf") else 0.3
        return base + _matches_concern(concerns, prioritized)

    def recommend(self, feats: dict, profile: dict, tier: str = "Core", include_device: bool = True) -> dict:
        tier = tier if tier in PLAN_TIERS else "Core"
        target_weeks = PLAN_TIERS[tier]["weeks"]
        prefer_upsell = PLAN_TIERS[tier]["upsell"]

        skin_type = profile.get("skin_type", "combo")
        prioritized = profile.get("prioritized_concerns", [])
        kb = self.kb.copy()

        # Rank products by relevance
        kb["relevance"] = kb.apply(lambda r: self._score_row(r, skin_type, prioritized), axis=1)
        kb = kb.sort_values(["relevance"], ascending=False)

        # Pick one per key form
        picks: List[pd.Series] = []
        chosen_forms = set()

        def pick_form(form_name: str, min_rel: float = 0.5, allow_lower=False):
            nonlocal picks, chosen_forms
            if form_name in chosen_forms: 
                return
            cands = kb[kb["form"] == form_name]
            cands = cands[cands["relevance"] >= (min_rel if not allow_lower else 0.0)]
            if len(cands) == 0 and allow_lower:
                cands = kb[kb["form"] == form_name]
            if len(cands) > 0:
                picks.append(cands.iloc[0])
                chosen_forms.add(form_name)

        # Core routine
        pick_form("cleanser", min_rel=0.3, allow_lower=True)
        pick_form("serum",    min_rel=0.5, allow_lower=True)
        pick_form("moisturizer", min_rel=0.5, allow_lower=True)
        pick_form("spf",      min_rel=0.3, allow_lower=True)

        # Extras
        top_concerns = [c for (c, _) in prioritized[:2]]
        if "texture" in top_concerns:
            pick_form("exfoliant", min_rel=0.5, allow_lower=True)
        if include_device:
            pick_form("device", min_rel=0.3, allow_lower=True)

        # Build sized items
        items = []
        for row in picks:
            form = str(row.get("form","")).lower()
            # Prefer exact matches by product_name; else fall back to sku-family
            same_name = self.kb["product_name"] == row.get("product_name", "")
            sizes_by_name = [float(x) for x in self.kb.loc[same_name, "size_ml"].dropna().tolist()]
            if sizes_by_name:
                sizes = sorted(set(sizes_by_name))
            else:
                sku = str(row.get("sku",""))
                prefix = sku.split("-")[0] if sku else ""
                sizes = sorted(set([float(x) for x in self.kb.loc[self.kb["sku"].str.startswith(prefix, na=False), "size_ml"].dropna().tolist()]))

            size_ml = _pick_size(
                sizes or [float(row.get("size_ml") or 0.0)],
                target_weeks=target_weeks,
                form=form,
                prefer_upsell=prefer_upsell,
                kb_row=row
            )

            items.append(dict(
                sku=row.get("sku",""),
                name=row.get("product_name",""),
                brand=row.get("brand",""),
                form=form,
                size_ml=size_ml,
                usage=row.get("usage",""),
                concerns=_safe_list(row.get("concerns")),
                reason=self._why_line(form, prioritized, row),
                price_usd=row.get("price_usd", None),
                upsell_tier=row.get("upsell_tier",""),
            ))

        return dict(
            plan=tier,
            target_weeks=target_weeks,
            skin_type=skin_type,
            top_concerns=top_concerns,
            items=items,
        )

    @staticmethod
    def _why_line(form: str, prioritized: List[tuple], row: pd.Series) -> str:
        top = [c for (c, _) in prioritized[:2]]
        if form == "serum":
            return f"Targets {', '.join(top) or 'daily balance'} with actives: {row.get('actives','')}."
        if form == "moisturizer":
            return "Locks in hydration and supports barrier overnight."
        if form == "cleanser":
            return "Gentle cleanse to prep skin without stripping."
        if form == "spf":
            return "Daily UV protection to prevent new damage."
        if form == "exfoliant":
            return "Smooths texture; use 2–3×/week PM."
        if form == "mask":
            return "Occasional boost aligned to your concerns."
        if form == "device":
            return "Optional device to enhance routine adherence/results."
        return "Complements routine."
