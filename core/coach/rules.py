# core/coach/rules.py
from __future__ import annotations
from typing import Dict, List

RED_HIGH = 0.62
SHINE_HIGH = 0.58
TEXTURE_HIGH = 250.0
TEXTURE_MED  = 180.0

def recommend_from_features(feats: Dict[str, float]) -> List[str]:
    recs: List[str] = []
    red, txt, shn = feats.get("global_red", 0.0), feats.get("global_txt", 0.0), feats.get("global_shn", 0.0)
    dark = feats.get("qa_mean_gray", 128) < 45
    bright = feats.get("qa_mean_gray", 128) > 210
    blur = feats.get("qa_blur", 200) < 100
    glare = feats.get("qa_specular", 0.0) > 0.04
    wb = feats.get("qa_wb_shift", 0.0) > 18

    if any([dark, bright, blur, glare, wb]):
        if dark:   recs.append("Retake: brighter room.")
        if bright: recs.append("Retake: avoid overexposed light.")
        if blur:   recs.append("Retake: hold still / focus.")
        if glare:  recs.append("Retake: pat skin dry / avoid direct lamp.")
        if wb:     recs.append("Retake: use neutral white light.")
        return recs

    if red > RED_HIGH:
        recs += ["Barrier repair: ceramides+cholesterol+FFA.", "Niacinamide 2–5%.", "Daily SPF 30+."]
    if shn > SHINE_HIGH and txt > TEXTURE_MED:
        recs += ["BHA 0.5–2% (3x/week).", "Light gel moisturizer.", "Adapalene 0.1% (2x/week, buffer)."]
    if txt > TEXTURE_HIGH:
        recs += ["Gentle PHA/AHA (1–2x/week).", "Humectant + occlusive at night."]
    if not recs:
        recs.append("Gentle cleanser + ceramide moisturizer + SPF.")
    return recs

def build_coach_payload(feats: Dict[str, float], profile: Dict, plan: Dict) -> Dict:
    tips = recommend_from_features(feats)
    return {
        "summary": f"Skin type: {profile.get('skin_type', 'unknown')} | Top concerns: {profile.get('concerns', [])}",
        "tips": tips,
        "notes": plan.get("top_concerns", []),
    }
