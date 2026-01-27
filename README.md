
# SkinAizer (Internship Prototype) - Streamlit Skin Analysis + Recommendations

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

```

.
├── streamlit_app.py          # Main Streamlit entrypoint (UI + session state + orchestration)
├── main.py                   # Pipeline glue / app-level helpers (non-UI)
├── scores.py                 # Feature extraction + scoring logic (deterministic, testable)
├── rec_engine.py             # Recommendation engine (rules/weights → ranked products)
├── core/
│   ├── yolo_roi.py           # ROI detection + cropping utilities (YOLO/OpenCV boundary)
│   └── model_core.py         # Model loading/runtime guards, device selection, fallback behavior
├── data/
│   └── ...                   # Optional sample images (NOT required for production)
├── requirements.txt          # Runtime dependencies (Streamlit, numpy, opencv, ultralytics, etc.)
└── README.md                 # You are here

````

**Key design rules**
- `core/*` must be **safe to import** even when optional ML deps fail (Streamlit Cloud-friendly).
- `scores.py` and `rec_engine.py` should be **pure-ish**: deterministic inputs → deterministic outputs.
- UI (`streamlit_app.py`) owns **session state** and presentation only.

---

## Quickstart (local)

### 1) Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows
````

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Run the app

```bash
streamlit run streamlit_app.py
```

Open the URL Streamlit prints in your terminal.

---

## Optional: YOLO / Ultralytics notes (Streamlit Cloud-friendly)

This project is designed to **not crash** when YOLO is unavailable.

* If `ultralytics` + `torch` load successfully, ROI detection runs normally.
* If they fail (common on restricted runtimes), the app should:

  * keep running,
  * show a warning,
  * use fallback logic (or skip ROI-based steps).

A common setup pattern used in the code:

* Setting `YOLO_CONFIG_DIR=/tmp/Ultralytics` to avoid permission issues on hosted environments.

---

## Deploying to Streamlit Cloud (recommended for internship demo)

1. Push this repo to GitHub.
2. Create a new Streamlit Cloud app → select your repo/branch.
3. Set the entrypoint to:

   * `streamlit_app.py`
4. Ensure `requirements.txt` is present and correct.

**Tip:** If YOLO is heavy for your deployment tier, keep the fallback path robust so your demo always runs.

---

## How to test (minimal)

If you have `pytest` in your dependencies:

```bash
pytest -q
```

Recommended test targets:

* `scores.py`: given a fixed ROI image array → stable numeric outputs
* `rec_engine.py`: given fixed scores → stable ranked recommendations
* `core/yolo_roi.py`: ROI normalization/cropping doesn’t crash on edge cases (no face, tiny face, rotated image)

---

## Security & privacy stance (internship-friendly)

* **No external image upload** is required by design (process in-memory).
* Avoid logging raw image bytes or saving user uploads to disk.
* Keep error messages user-safe: no stack traces in the UI; show a short failure reason + recovery hint.

---

## Troubleshooting

**“App runs but doesn’t detect face”**

* Verify YOLO dependencies are installed.
* Try a clearer front-facing portrait.
* Confirm the ROI module is pointing to the correct weights/model.

**“Streamlit Cloud fails to build due to torch/ultralytics”**

* Keep fallback mode working (app should still run).
* Consider pinning compatible versions in `requirements.txt`.

**“Recommendations look too generic / low quality”**

* Improve `scores.py` signal quality (normalize, calibrate thresholds).
* Add rule-based guardrails in `rec_engine.py` (e.g., sensitivity-first, acne-safe, fragrance-free buckets).
* Make the coach explain *why* a product was selected (feature → rule → product tag).

---

## Disclaimer

This is a **prototype for internship/demo purposes**.
It is **not** a medical device and does not provide medical diagnosis or treatment advice.

---

## Contact / Ownership

Maintainer: **Caio Salvieti Da Silva**
Project: **SkinAizer** — internship prototype branch / repo

```
```
