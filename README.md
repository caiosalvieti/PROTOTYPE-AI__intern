# Skinaizer – Privacy-First Skin Analysis

> Real-time skin assessment without storing faces. Streamlit app + ML models + optional Raspberry Pi camera kit.

---

## ✨ Features

* **On-device inference** (no cloud upload) with opt-in telemetry
* **Face/skin region detection** (no image persistence; RAM-only pipeline)
* **Condition scoring** (e.g., hydration, redness, blemishes) with model ensembles
* **Explainable outputs** (saliency/SHAP-like overlays)
* **Recommendation engine** (rules + ML) with a small product KB
* **Raspberry Pi capture module** (Pi 5 + Cam Module 3 + LED ring)
* **API layer** for mobile apps (FastAPI, JSON)
* **Reproducible ML training** (Poetry + Makefile)

---

## 📦 Stack

* **App:** Streamlit
* **API:** FastAPI (uvicorn)
* **ML:** scikit-learn, XGBoost/LightGBM (optional), OpenCV
* **Explainers:** shap (optional)
* **Packaging:** Poetry, pyproject.toml
* **CI (optional):** GitHub Actions
* **Container:** Docker

---

## 📁 Repository layout

```
skinaizer/
├─ apps/
│  ├─ web/                 # Streamlit UI
│  │  ├─ app.py
│  │  └─ pages/
│  ├─ api/                 # FastAPI service
│  │  ├─ main.py
│  │  └─ routers/
│  └─ tools/               # CLI tools (e.g., batch scoring)
│     └─ score.py
├─ models/
│  ├─ current/             # *.pkl / *.onnx
│  └─ experiments/
├─ DATA/
│  ├─ raw/
│  ├─ interim/
│  └─ products_kb.csv
├─ src/
│  ├─ skinaizer/           # Python package (preprocess, infer, explain, privacy)
│  └─ pipeline.py
├─ notebooks/              # EDA + experiments
├─ configs/
│  ├─ app.toml
│  └─ model.yaml
├─ tests/
├─ .env.example
├─ Makefile
├─ pyproject.toml
├─ README.md
└─ LICENSE
```

---

## 🚀 Quickstart (local)

**Requirements:** Python 3.11+, Poetry, Git, macOS/Linux/WSL.

```bash
# 1) Clone
git clone https://github.com/your-org/skinaizer.git
cd skinaizer

# 2) Create env
poetry install --with dev

# 3) Configure
cp .env.example .env                       # edit secrets & toggles
mkdir -p models/current DATA/interim

# 4) (Optional) fetch a demo model
# put model files into models/current/ e.g., skin_gbrt.pkl

# 5) Run web app
poetry run streamlit run apps/web/app.py

# 6) Run API (in another shell)
poetry run uvicorn apps.api.main:app --reload --port 8000
```

Open:

* UI: [http://localhost:8501](http://localhost:8501)
* API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## ⚙️ Configuration

Edit `.env` (use `.env.example` as a guide):

```
# Runtime
ENV=dev
LOG_LEVEL=INFO

# Privacy
DISABLE_IMAGE_PERSISTENCE=true
ANONYMIZE_SESSION_IDS=true
TELEMETRY_OPTOUT=true    # set false to allow minimal, aggregated metrics

# Models
MODEL_DIR=./models/current
PRODUCT_KB=./DATA/products_kb.csv

# API
API_HOST=0.0.0.0
API_PORT=8000
```

Fine-tune behavior in `configs/app.toml` and `configs/model.yaml` (thresholds, preprocessing, ensemble weights, recommendation rules).

---

## 🔒 Privacy by design

* **No writes of user images to disk** by default (`DISABLE_IMAGE_PERSISTENCE=true`).
* All frames processed in memory; optional **ephemeral cache** with secure wipe on shutdown.
* Face boxes are not stored; **only aggregate scores** can be logged if telemetry is enabled.
* “Screenshot” and “export” buttons are **off** by default for compliance.

---

## 📸 Raspberry Pi capture (optional)

**Hardware:** Raspberry Pi 5, Camera Module 3 (12 MP, AF), LED ring/soft light, short USB-C PD.

1. Flash **Raspberry Pi OS** (64-bit).
2. Enable camera: `sudo raspi-config` → Interface Options.
3. Install deps:

```bash
sudo apt update && sudo apt install -y python3-pip libatlas-base-dev
pip3 install opencv-python-headless fastapi uvicorn pydantic
```

4. Launch a small capture server (example in `apps/tools/capture_server.py`) or stream frames via RTSP/WebRTC to the main app.
5. In Streamlit, set `CAPTURE_ENDPOINT` in `.env`.

> Tip: keep **consistent lighting**; enable the LED at a fixed intensity to reduce variance.

---

## 🧪 Testing

```bash
# Unit tests + lint
poetry run pytest -q
poetry run ruff check .
poetry run mypy src

# E2E (headless)
poetry run pytest -q -m "e2e"
```

---

## 🧠 Training & experiments

1. Drop curated data into `DATA/raw/` (no faces; numeric features or anonymized descriptors).
2. Use notebooks in `notebooks/` for EDA and feature selection.
3. Train via Makefile recipes:

```bash
# Reproduce a compact GBRT baseline
poetry run python src/pipeline.py train --config configs/model.yaml

# Evaluate + export model
poetry run python src/pipeline.py eval --split groupkfold
poetry run python src/pipeline.py export --out models/current/skin_gbrt.pkl
```

4. (Optional) Export to ONNX for mobile:

```bash
poetry run python src/pipeline.py export-onnx --out models/current/skin.onnx
```

**Algorithms included:** Gradient Boosting (baseline), Random Forest (robust baseline), XGBoost/LightGBM (optional), Logistic/Linear models for ablations. SHAP summaries for model choice justification.

---

## 📱 Mobile integration

* **REST API** (`/v1/score`) accepts a cropped skin patch or precomputed features; returns scores + explanations.
* iOS/Swift client can call the API or run **ONNX Runtime** on-device for fully offline mode.
* See `apps/api/routers/score.py` for payload schemas.

---

## 🐳 Docker

```bash
docker build -t skinaizer:latest .
docker run --rm -p 8501:8501 -p 8000:8000 --env-file .env skinaizer:latest
```

---

## 🔧 Makefile cheatsheet

```bash
make setup          # bootstrap (poetry, pre-commit)
make run-web        # streamlit
make run-api        # uvicorn
make train          # ML training
make test           # tests + lint
make fmt            # ruff format
```

---

## 🧭 Roadmap

* [ ] Improved skin-region segmentation (light-invariant)
* [ ] Model cards & datasheets for transparency
* [ ] On-device iOS prototype (Core ML / ORT)
* [ ] Calibration with color checker & auto-white balance
* [ ] Multi-illumination consensus scoring
* [ ] Differential privacy for telemetry

---

## 🤝 Contributing

1. Create a feature branch: `git checkout -b feat/your-thing`
2. Run `make test` before pushing
3. Open a PR with screenshots and a short demo clip (synthetic data only)

Code style: PEP 8, type hints, small pure functions, docstrings.

---

## 📄 License

MIT (see `LICENSE`). If you integrate clinical features, ensure regulatory compliance for your jurisdiction before distribution.

---

## 📚 Citation

If this tool helps your research, please cite:

```
K. Nimz et al. Skinaizer: Privacy-First Skin Analysis with On-Device ML, 2025. GitHub repository.
```

---

## 🆘 Troubleshooting

* **Every photo gives the same result** → clear model cache, verify preprocessing pipeline isn’t normalizing to a constant; run `make test` and check `DATA/products_kb.csv` isn’t being used as a fallback.
* **Camera not detected (Pi)** → `libcamera-hello` test, check ribbon seating, enable camera in raspi-config.
* **Performance drops on M-series Macs** → prefer `opencv-python-headless`, set `OMP_NUM_THREADS=1` for reproducibility.

---

## 📝 Acknowledgements

Thanks to the open-source community and academic collaborators in computational toxicology and ML for reproducible research practices.

---
