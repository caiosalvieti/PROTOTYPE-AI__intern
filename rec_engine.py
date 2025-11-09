# rec_engine.py
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

# --- photo-severity + safety knobs ---
THRESH = dict(
    need_mild=0.50,      # include when concern ≥ 0.50
    need_strong=0.62,    # stronger push (e.g., retinoid)
    sensitive=0.60,      # sensitivity gate (fragrance-free)
)
SAFE_COMEDO_MAX = 2      # acne-prone: max allowed comedogenicity (0–5)

def _as_bool(x):
    s = str(x).strip().lower()
    return s in ("1","true","yes","y")

# photo-driven thresholds (same as main.py)
RED_HIGH = 0.62
SHINE_HIGH = 0.58
TEXTURE_HIGH = 250.0
TEXTURE_MED = 180.0

# ----------------- small utils 
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

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#                   RecEngine
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
class RecEngine:

    def __init__(self, kb_path: str = DEFAULT_KB):
        if not os.path.isfile(kb_path):
            raise FileNotFoundError(f"Knowledge base not found: {kb_path}")
            self.kb = pd.read_csv(kb_path)
    # Aliases & defaults (make BOTH sides exist)
        cols = set(self.kb.columns)

            # product_name / name — ensure both exist
        if "product_name" not in cols and "name" in cols:
            self.kb["product_name"] = self.kb["name"]
        if "name" not in cols and "product_name" in cols:
            self.kb["name"] = self.kb["product_name"]
        if "product_name" not in self.kb.columns:
            self.kb["product_name"] = ""
        if "name" not in self.kb.columns:
            self.kb["name"] = ""

        # sku / id — ensure both exist
        if "sku" not in cols and "id" in cols:
            self.kb["sku"] = self.kb["id"]
        if "id" not in cols and "sku" in cols:
            self.kb["id"] = self.kb["sku"]
        if "sku" not in self.kb.columns:
            self.kb["sku"] = ""
        if "id" not in self.kb.columns:
            self.kb["id"] = ""

    # Ensure required-but-missing columns exist with sane defaults
        for c in [
        "form","usage","upsell_tier","skin_types","concerns","actives",
        "brand","tier","category","link","fragrance_free","comedogenicity","contra"]: 
            if c not in self.kb.columns: self.kb[c] = ""

    # Normalize (now safe because all columns exist)
            for c in ["product_name","name","form","tier","brand","sku","category","usage","upsell_tier"]:
             self.kb[c] = self.kb[c].astype(str).str.strip().str.lower()

            for c in ["size_ml","price_usd","comedogenicity"]:
             self.kb[c] = pd.to_numeric(self.kb[c], errors="coerce")

            self.kb["fragrance_free"] = (
        self.kb["fragrance_free"]
        .fillna(0)
        .astype(str).str.strip().str.lower()
        .isin(["1","true","yes","y"])
    )

    # Parse list-like columns into sets
            self.kb["skin_types"] = self.kb["skin_types"].map(_safe_set)
            self.kb["concerns"]    = self.kb["concerns"].map(_safe_set)
            self.kb["actives"]     = self.kb["actives"].map(_safe_set)
            self.kb["contra"]      = self.kb["contra"].map(_safe_set)

    # Infer missing form
            mask_missing_form = self.kb["form"].astype(str).str.strip().eq("")
            if mask_missing_form.any():
             self.kb.loc[mask_missing_form, "form"] = self.kb[mask_missing_form].apply(_infer_form, axis=1)


            # Normalize
            for c in ["product_name","name","form","tier","brand","sku","category","usage","upsell_tier"]:
                self.kb[c] = self.kb[c].astype(str).str.strip().str.lower()
            for c in ["size_ml","price_usd","comedogenicity"]:
                self.kb[c] = pd.to_numeric(self.kb[c], errors="coerce")
            self.kb["fragrance_free"] = self.kb["fragrance_free"].fillna(0).astype(str).str.strip().str.lower().isin(["1","true","yes"])

            # Parse list-like columns into sets
            self.kb["skin_types"] = self.kb["skin_types"].map(_safe_set)
            self.kb["concerns"]    = self.kb["concerns"].map(_safe_set)
            self.kb["actives"]     = self.kb["actives"].map(_safe_set)
            self.kb["contra"]      = self.kb["contra"].map(_safe_set)

            # Infer missing form
            mask_missing_form = self.kb["form"].astype(str).str.strip().eq("")
            if mask_missing_form.any():
                self.kb.loc[mask_missing_form, "form"] = self.kb[mask_missing_form].apply(_infer_form, axis=1)
    
        # ---------- severity weights from profile.scores (+ mapping to KB tokens) ----------
    def _weights(self, profile: Dict[str, Any]) -> Dict[str, float]:
            # profile['scores'] keys come from scores.py: oiliness, dryness, redness, texture, sensitivity
            s = {k: float(v) for k, v in (profile.get("scores") or {}).items()}
            w: Dict[str, float] = {}

            # Map to KB concern tokens
            w["oil"] = w["shine"] = s.get("oiliness", 0.0)
            w["clogged_pores"] = max(s.get("oiliness", 0.0), s.get("texture", 0.0) * 0.6)
            w["texture"] = s.get("texture", 0.0)
            w["redness"] = s.get("redness", 0.0)
            w["hydration"] = s.get("dryness", 0.0)
            w["barrier"] = max(s.get("dryness", 0.0) * 0.7, s.get("sensitivity", 0.0) * 0.3)
            w["sensitivity"] = s.get("sensitivity", 0.0)
            w["uv"] = 0.5  # always useful baseline

            # Also honor prioritized_concerns from profile (list of tuples)
            for c, val in (profile.get("prioritized_concerns") or []):
                # map 'oil_control' -> 'oil'
                if c == "oil_control":
                    c = "oil"
                w[c] = max(w.get(c, 0.0), float(val))

            return w

    def _concern_weights(self, profile: Dict[str, Any], feats: Dict[str, float]) -> Dict[str, float]:
        """Compatibility wrapper: keep your original name, but base it on _weights() and add a few direct photo nudges."""
        w = self._weights(profile)

        # Direct nudges from raw feats (keeps your earlier behavior)
        if feats.get("global_shn", 0) > SHINE_HIGH:
            w["acne"] = max(w.get("acne", 0.0), 0.9)
        if feats.get("global_txt", 0) > TEXTURE_HIGH:
            w["texture"] = max(w.get("texture", 0.0), 1.0)
        elif feats.get("global_txt", 0) > TEXTURE_MED:
            w["texture"] = max(w.get("texture", 0.0), 0.6)
        if feats.get("global_red", 0) > RED_HIGH:
            w["redness"] = max(w.get("redness", 0.0), 0.9)
            w["barrier"] = max(w.get("barrier", 0.0), 0.6)

        return w

    def _severity_for_row(self, row: pd.Series, weights: Dict[str, float]) -> float:
        """How much this product addresses the user's top concerns (0..1+)."""
        row_concerns = row.get("concerns") or set()
        if not row_concerns:
            return 0.25
        return max((weights.get(c, 0.0) for c in row_concerns), default=0.0)

    def _score_row(self, row: pd.Series, profile: Dict[str, Any], feats: Dict[str, float]) -> float:
        st = profile.get("skin_type", "")
        weights = self._concern_weights(profile, feats)

        # Form baseline to separate roles
        base_by_form = {"serum":0.90,"moisturizer":0.85,"spf":0.80,"cleanser":0.60,"exfoliant":0.55,"mask":0.30,"device":0.25,"treatment":0.90}
        form = str(row.get("form","")).lower()
        score = base_by_form.get(form, 0.40)

        # Skin-type coverage
        if _covers_skin_type(set(row["skin_types"]), st):
            score += 0.40
        elif row["skin_types"]:
            score -= 0.10  # explicit mismatch

        # Severity from image/profile
        sev = self._severity_for_row(row, weights)  # 0..1
        score += 0.70 * sev

        # Safety / contra
        sensitivity = weights.get("sensitivity", 0.0)
        if sensitivity >= THRESH["sensitive"] and not bool(row.get("fragrance_free", False)):
            score -= 0.50

        # derive acne-prone from oiliness/texture if user didn't set a flag
        acne_prone = bool(profile.get("acne_prone")) or (weights.get("oil",0) >= 0.65) or (weights.get("clogged_pores",0) >= 0.65)
        try:
            if acne_prone and float(row.get("comedogenicity") or 0) > SAFE_COMEDO_MAX:
                score -= 0.50
        except Exception:
            pass

        if profile.get("pregnant", False) and ("retinoid_pregnancy" in (row.get("contra") or set())):
            score -= 9.0  # hard block

        # Synergy with actives and photo signals
        acts = set(row.get("actives") or set())
        if feats.get("global_red",0) > RED_HIGH and ({"niacinamide","azelaic_acid"} & acts):
            score += 0.20
        if feats.get("global_shn",0) > SHINE_HIGH and ("bha" in acts or "salicylic_acid" in acts):
            score += 0.25
        if feats.get("global_txt",0) > TEXTURE_HIGH and ({"aha","pha","lactic_acid","mandelic_acid"} & acts):
            score += 0.20

        return float(score)

    def _pick_exfoliant_candidates(self, kb: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
        """Prefer BHA for oil/clogged; AHA/PHAs for dry/texture."""
        oil = weights.get("oil", 0.0)
        dry = weights.get("hydration", 0.0)
        if oil >= dry:
            # BHA lane
            return kb[(kb["form"] == "exfoliant") & kb["actives"].fillna("").astype(str).str.contains(r"\bbha\b|salicylic", regex=True)]
        else:
            # AHA/PHAs lane
            return kb[(kb["form"] == "exfoliant") & kb["actives"].fillna("").astype(str).str.contains(r"\baha\b|\bpha\b|lactic|mandelic", regex=True)]
    def _to_float(self, v) -> float:
        """Coerce anything to a single float (guards against arrays/Series)."""
        try:
            return float(v)
        except Exception:
            try:
                arr = np.asarray(v, dtype=float).ravel()
                return float(arr[0]) if arr.size else 0.0
            except Exception:
                return 0.0

    # --------------------------- main API ---------------------------
    def recommend(self, feats: dict, profile: dict, tier: str = "Core", include_device: bool = True, top_k_per_type: int = 1) -> dict:
        tier = tier if tier in PLAN_TIERS else "Core"
        cfg = PLAN_TIERS[tier]
        target_weeks = cfg["weeks"]; prefer_upsell = cfg["upsell"]; need = dict(cfg["need"])

        # If photo QA failed, do not prescribe (UI will show reasons)
        if int(feats.get("qa_fail", 0)) == 1:
            issues = feats.get("qa_issues", "")
            return dict(plan="QA only", target_weeks=0, skin_type=None, top_concerns=[],
                        items=[], reasons={"qa":[f"Photo QA fail ({issues}). Retake suggested."]}, by_type={})

        kb = self.kb.copy()
        stype = profile.get("skin_type","")

        # Soft filter by skin type
        if stype:
            kb = kb[kb["skin_types"].map(lambda s: (len(s)==0) or (stype in s) or ("all" in s) or (stype=="combo" and ("normal" in s or "combo" in s)))]

        # Score per-row
        kb["__score"] = kb.apply(lambda r: self._score_row(r, profile, feats), axis=1)

        by_type: Dict[str, List[Dict[str, Any]]] = {}
        reasons: Dict[str, List[str]] = {}
        weights = self._concern_weights(profile, feats)

        def pick_for(form: str | List[str], k: int, extra_filter=None, min_score: float = 0.0):
            forms = [form] if isinstance(form, str) else form
            cand = kb[kb["form"].isin(forms)].copy()
            if extra_filter is not None:
                cand = extra_filter(cand)
            if cand.empty:
                return
            cand = cand.sort_values("__score", ascending=False)
            if min_score > 0:
                cand = cand[cand["__score"] >= min_score]
            if cand.empty:
                return

            chosen_rows = cand.head(max(1, k))
            chosen, why = [], []
            for _, row in chosen_rows.iterrows():
                # sizes to choose from: same product name (preferred) then same SKU family
                same_name = self.kb["product_name"] == row.get("product_name", "")
                sizes_by_name = [float(x) for x in self.kb.loc[same_name, "size_ml"].dropna().tolist()]
                if sizes_by_name:
                    sizes = sorted(set(sizes_by_name))
                else:
                    sku = str(row.get("sku",""))
                    prefix = sku.split("-")[0] if sku else ""
                    sizes = sorted(set([float(x) for x in self.kb.loc[self.kb["sku"].str.startswith(prefix, na=False), "size_ml"].dropna().tolist()]))

                fform = str(row.get("form","")).lower()
                size_ml = _pick_size(sizes or [float(row.get("size_ml") or 0.0)],
                                     target_weeks=target_weeks,
                                     form=fform,
                                     prefer_upsell=prefer_upsell,
                                     kb_row=row)

                chosen.append(dict(
                    sku=row.get("sku",""),
                    name=row.get("product_name",""),
                    brand=row.get("brand",""),
                    form=fform,
                    size_ml=size_ml,
                    usage=row.get("usage",""),
                    concerns=list(row.get("concerns") or []),
                    price_usd=row.get("price_usd", None),
                    upsell_tier=row.get("upsell_tier",""),
                    fragrance_free=bool(row.get("fragrance_free", False)),
                    comedogenicity=float(row.get("comedogenicity") or 0.0),
                    actives=list(row.get("actives") or []),
                    reason="",  # filled by human-readable lines below (for old-UI compatibility)
                ))

                # human reason lines that reference the photo severity
                rlines = []
                top = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)[:2]
                if fform in ("serum","treatment"):
                    rlines.append("targets " + ", ".join([f"{k} ({v:.2f})" for k,v in top]))
                if fform == "exfoliant":
                    if weights.get("oil",0) >= weights.get("hydration",0):
                        rlines.append(f"bha lane for oil/clogging (oil {weights.get('oil',0):.2f})")
                    else:
                        rlines.append(f"aha/pha lane for texture/dryness (texture {weights.get('texture',0):.2f})")
                if fform == "moisturizer":
                    rlines.append(f"barrier/hydration (dry {weights.get('hydration',0):.2f}, sens {weights.get('sensitivity',0):.2f})")
                if fform == "spf":
                    rlines.append("daily uv prevention")
                if rlines:
                    reasons.setdefault(fform, []).extend(rlines)
                    chosen[-1]["reason"] = "; ".join(rlines)

                # always include numeric score for debugging
                reasons.setdefault(fform, []).append(f"{row.get('product_name','')} → score {row['__score']:.2f}")

            # keep only top_k_per_type if requested
            by_type[forms[0] if isinstance(form, str) else forms[0]] = chosen[:max(1, top_k_per_type)]

        # Required forms
        for f, k in need.items():
            pick_for(f, k, min_score=0.0)

        # Extras based on photo severity (oil OR texture)
        if max(weights.get("texture",0), weights.get("oil",0)) >= THRESH["need_mild"]:
            pick_for("exfoliant", 1, extra_filter=lambda df: self._pick_exfoliant_candidates(df, weights))

        if include_device:
            pick_for("device", 1)

        # Flatten items in a stable order
        items: List[Dict[str, Any]] = []
        for f in FORMS_ORDER:
            items += by_type.get(f, [])

        # Top concerns for header
        top_concerns = sorted(weights.keys(), key=lambda c: weights[c], reverse=True)[:3]

        return dict(
            plan=tier,
            target_weeks=target_weeks,
            skin_type=stype or profile.get("skin_type","combo"),
            top_concerns=top_concerns,
            items=items,
            by_type=by_type,
            reasons=reasons,
        )
