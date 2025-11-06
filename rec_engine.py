from __future__ import annotations
import os
from typing import List, Dict, Any, Set
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
    "Starter": dict(weeks=2,  upsell=False, need={"cleanser":1, "moisturizer":1, "spf":1}),
    "Core":    dict(weeks=6,  upsell=True,  need={"cleanser":1, "moisturizer":1, "spf":1}),
    "Intense": dict(weeks=10, upsell=True,  need={"cleanser":1, "moisturizer":1, "spf":1, "serum":1}),
}

FORMS_ORDER = ["cleanser", "serum", "moisturizer", "spf", "exfoliant", "mask", "device"]

# photo-driven thresholds (same as main.py) 
RED_HIGH = 0.62
SHINE_HIGH = 0.58
TEXTURE_HIGH = 250.0
TEXTURE_MED = 180.0

def _safe_list(cell: Any) -> List[str]:
    if cell is None or (isinstance(cell, float) and pd.isna(cell)): return []
    s = str(cell).strip()
    if not s: return []
    s = s.replace("|", ",").replace(";", ",")
    return [x.strip().lower() for x in s.split(",") if x.strip()]

def _safe_set(cell: Any) -> Set[str]:
    return set(_safe_list(cell))

def _covers_skin_type(row_types: Set[str], user_skin_type: str) -> bool:
    if not row_types or "all" in row_types: return True
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
            chosen = ml; break
    if chosen is None:
        chosen = avail_sorted[-1] if avail_sorted else 0.0
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
    name = str(row.get("name") or row.get("product_name") or "").lower()
    category = str(row.get("category") or "").lower()
    is_device = bool(row.get("device")) or str(row.get("form","")).lower() == "device"
    hay = f"{name} {category}"
    if is_device or "luna" in hay or "device" in hay: return "device"
    if "cleanser" in hay or "wash" in hay: return "cleanser"
    if "serum" in hay or ("treatment" in hay and "serum" in hay): return "serum"
    if "moisturizer" in hay or "cream" in hay or "gel-cream" in hay: return "moisturizer"
    if "spf" in hay or "sunscreen" in hay: return "spf"
    if "exfol" in hay or "aha" in hay or "bha" in hay or "pha" in hay: return "exfoliant"
    if "mask" in hay: return "mask"
    return ""

class RecEngine:
    def __init__(self, kb_path: str = DEFAULT_KB):
        if not os.path.isfile(kb_path):
            raise FileNotFoundError(f"Knowledge base not found: {kb_path}")
        self.kb = pd.read_csv(kb_path)

        # Aliases & defaults
        cols = set(self.kb.columns)
        if "product_name" not in cols and "name" in cols:
            self.kb["product_name"] = self.kb["name"]
        if "sku" not in cols and "id" in cols:
            self.kb["sku"] = self.kb["id"]
        for c in ["form","usage","upsell_tier","skin_types","concerns","actives","brand","tier","category","link",
                  "fragrance_free","comedogenicity","contra"]:
            if c not in self.kb.columns: self.kb[c] = ""

        # Normalize
        for c in ["product_name","name","form","tier","brand","sku","category","usage","upsell_tier"]:
            self.kb[c] = self.kb[c].astype(str).str.strip().str.lower()
        for c in ["size_ml","price_usd","comedogenicity"]:
            self.kb[c] = pd.to_numeric(self.kb[c], errors="coerce")
        self.kb["fragrance_free"] = self.kb["fragrance_free"].fillna(0).astype(str).str.strip().str.lower().isin(["1","true","yes"])

        # Parse list-like columns into sets
        self.kb["skin_types"] = self.kb["skin_types"].map(_safe_set)
        self.kb["concerns"] = self.kb["concerns"].map(_safe_set)
        self.kb["actives"] = self.kb["actives"].map(_safe_set)
        self.kb["contra"] = self.kb["contra"].map(_safe_set)

        # Infer missing form
        mask_missing_form = self.kb["form"].astype(str).str.strip().eq("")
        if mask_missing_form.any():
            self.kb.loc[mask_missing_form, "form"] = self.kb[mask_missing_form].apply(_infer_form, axis=1)

    # concern weights derived from profile + feats 
    def _concern_weights(self, profile: Dict[str, Any], feats: Dict[str, float]) -> Dict[str, float]:
        w: Dict[str, float] = {}

        # From profile
        for c in profile.get("concerns", []):
            w[c] = max(w.get(c, 0.0), 1.0)
        for item in profile.get("prioritized_concerns", []):
            if isinstance(item, (list, tuple)) and len(item) >= 1:
                c, wt = item[0], (float(item[1]) if len(item) > 1 else 1.2)
                w[c] = max(w.get(c, 0.0), wt)

        # From photo features (feats)
        if feats.get("global_shn", 0) > SHINE_HIGH:
            w["acne"] = max(w.get("acne", 0.0), 0.9)
            w["oiliness"] = max(w.get("oiliness", 0.0), 0.8)
        if feats.get("global_txt", 0) > TEXTURE_HIGH:
            w["texture"] = max(w.get("texture", 0.0), 1.0)
        elif feats.get("global_txt", 0) > TEXTURE_MED:
            w["texture"] = max(w.get("texture", 0.0), 0.6)
        if feats.get("global_red", 0) > RED_HIGH:
            w["redness"] = max(w.get("redness", 0.0), 0.9)
            w["barrier"] = max(w.get("barrier", 0.0), 0.6)

        return w

    def _score_row(self, row: pd.Series, profile: Dict[str, Any], feats: Dict[str, float]) -> float:
        st = profile.get("skin_type", "")
        weights = self._concern_weights(profile, feats)

        # Form baseline (helps separate roles)
        base_by_form = {"cleanser":0.6,"serum":1.0,"moisturizer":0.9,"spf":0.8,"exfoliant":0.5,"mask":0.2,"device":0.3}
        score = base_by_form.get(str(row.get("form","")).lower(), 0.3)

        # Fit by skin type
        if _covers_skin_type(set(row["skin_types"]), st): score += 3.0
        elif row["skin_types"]: score -= 1.0  # explicit mismatch

        # Concern match
        if weights:
            score += sum(weights.get(c, 0.0) for c in row["concerns"])

        # Safety/contra
        flags = set(profile.get("flags", []))
        if ("sensitive" in flags) and (not bool(row.get("fragrance_free", False))):
            score -= 2.0
        if profile.get("acne_prone") and float(row.get("comedogenicity") or 0) > 2.0:
            score -= 3.0
        if profile.get("pregnant", False) and ("retinoid_pregnancy" in row["contra"]):
            score -= 9.0

        # Synergy with photo signals
        acts = set(row["actives"])
        if feats.get("global_red",0) > RED_HIGH and ({"niacinamide","azelaic_acid"} & acts):
            score += 1.5
        if feats.get("global_shn",0) > SHINE_HIGH and ("bha" in acts):
            score += 1.5
        if feats.get("global_txt",0) > TEXTURE_HIGH and ({"aha","pha"} & acts):
            score += 1.0

        return float(score)

    def recommend(self, feats: dict, profile: dict, tier: str = "Core", include_device: bool = True, top_k_per_type: int = 1) -> dict:
        tier = tier if tier in PLAN_TIERS else "Core"
        cfg = PLAN_TIERS[tier]
        target_weeks = cfg["weeks"]; prefer_upsell = cfg["upsell"]; need = dict(cfg["need"])

        kb = self.kb.copy()
        # Optionally soft-filter by user skin type
        st = profile.get("skin_type","")
        if st:
            kb = kb[ kb["skin_types"].map(lambda s: (len(s)==0) or (st in s) or ("all" in s) or (st=="combo" and ("normal" in s or "combo" in s))) ]

        # Score
        kb["__score"] = kb.apply(lambda r: self._score_row(r, profile, feats), axis=1)

        # Build plan per form
        by_type: Dict[str, List[Dict[str, Any]]] = {}
        reasons: Dict[str, List[str]] = {}

        def pick_for(form: str, k: int):
            cand = kb[kb["form"] == form].copy()
            if cand.empty: return
            cand = cand.sort_values("__score", ascending=False).head(max(1, k))
            chosen = []
            why = []
            for _, row in cand.iterrows():
                # sizes to choose from: same product name (preferred) then same SKU family
                same_name = self.kb["product_name"] == row.get("product_name", "")
                sizes_by_name = [float(x) for x in self.kb.loc[same_name, "size_ml"].dropna().tolist()]
                if sizes_by_name:
                    sizes = sorted(set(sizes_by_name))
                else:
                    sku = str(row.get("sku",""))
                    prefix = sku.split("-")[0] if sku else ""
                    sizes = sorted(set([float(x) for x in self.kb.loc[self.kb["sku"].str.startswith(prefix, na=False), "size_ml"].dropna().tolist()]))

                size_ml = _pick_size(sizes or [float(row.get("size_ml") or 0.0)],
                                     target_weeks=target_weeks,
                                     form=form,
                                     prefer_upsell=prefer_upsell,
                                     kb_row=row)

                chosen.append(dict(
                    sku=row.get("sku",""),
                    name=row.get("product_name",""),
                    brand=row.get("brand",""),
                    form=form,
                    size_ml=size_ml,
                    usage=row.get("usage",""),
                    concerns=list(row["concerns"]),
                    price_usd=row.get("price_usd", None),
                    upsell_tier=row.get("upsell_tier",""),
                    fragrance_free=bool(row.get("fragrance_free", False)),
                    comedogenicity=float(row.get("comedogenicity") or 0.0),
                    actives=list(row["actives"]),
                ))

                why_bits = []
                if _covers_skin_type(set(row["skin_types"]), st): why_bits.append(f"fits {st}")
                hit_cons = [c for c in row["concerns"] if c in self._concern_weights(profile, feats)]
                if hit_cons: why_bits.append("concerns: " + ", ".join(sorted(hit_cons)[:3]))
                if bool(row.get("fragrance_free", False)): why_bits.append("fragrance-free")
                if float(row.get("comedogenicity") or 0) <= 2 and profile.get("acne_prone"): why_bits.append("low-comedogenic")
                why_bits.append(f"score {row['__score']:.1f}")
                why.append(f"{row.get('product_name','')} → " + "; ".join(why_bits))

            by_type[form] = chosen
            reasons[form] = why

        # required forms
        for f, k in need.items():
            pick_for(f, k)
        # optional exfoliant based on texture concern
        cw = self._concern_weights(profile, feats)
        if cw.get("texture", 0) >= 0.6:
            pick_for("exfoliant", 1)
        if include_device:
            pick_for("device", 1)

        # Flatten items in a stable order
        items = []
        for f in FORMS_ORDER:
            items += by_type.get(f, [])

        top_concerns = sorted(cw.keys(), key=lambda c: cw[c], reverse=True)[:3]

        return dict(
            plan=tier,
            target_weeks=target_weeks,
            skin_type=st or profile.get("skin_type","combo"),
            top_concerns=top_concerns,
            items=items,
            # extras (optional, for debugging/UX)
            by_type=by_type,
            reasons=reasons,
        )
