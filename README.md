# SkinAizer (Internship Prototype) — Streamlit Skin Analysis + Recommendations

This repository is an **internship-ready prototype** of **SkinAizer**: a lightweight, privacy-first skin analysis demo built with **Streamlit**.  
It takes a face photo, detects a **face ROI (region of interest)** (YOLO/OpenCV), computes **basic skin signals/scores**, and generates **product recommendations** + a simple “skin coach” chat flow.

> Goal: demonstrate end-to-end thinking (UI → CV inference → scoring → recs → explanation) in a clean, testable, deployable repo.

---

## What it does (user flow)

1. **Upload an image** (or use a sample from the dataset folder).
2. **Detect face ROI** (best-effort):
   - If YOLO is available: use YOLO-based detection to find the face region.
   - If YOLO isn’t available: fallback logic prevents the app from crashing and still runs (reduced features).
3. **Compute scores** (example: hydration/oiliness/texture proxies depending on your current implementation).
4. **Recommend products** based on scores + rules/weights.
5. **Explain results** in a simple “coach” UI (structured guidance, not medical advice).

---

## Architecture (high level)

**Streamlit UI** → **ROI detector** → **Scoring** → **Recommendation engine** → **Coach UX**

- **Probabilistic space** (detector outputs) gets converted into **deterministic pixel space** (cropping/ROI).
- All core steps return **structured dict outputs** to keep the pipeline auditable and easy to debug.

---

## Repository layout (what each file is for)

> Names may differ slightly depending on your branch; this is the intended contract.

