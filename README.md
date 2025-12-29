Skinaizer - Privacy-First Skin Analysis for People Who Actually Ship

Real-time skin assessment that doesn’t hoard faces. Streamlit front end, FastAPI backend, reproducible ML pipeline, optional Raspberry Pi capture. Built for local runs first; cloud is an opt-in, not a requirement.

What this project is (and isn’t)

Is: A modular, on-device skin analysis stack. You feed images or features; it detects skin regions, scores conditions (hydration, redness, blemishes), explains outputs, and returns actionable recommendations.
Isn’t: A data-grab. Images never touch disk by default. Telemetry is off unless you explicitly enable minimal, aggregated metrics.
Audience: Engineers, applied ML folks, and product teams who need a demoable prototype that doesn’t collapse under real-world lighting and device variability.
Feature set (clear and blunt)

On-device inference: CPU-friendly models with optional ONNX export. Runs on macOS/Linux/WSL and Raspberry Pi 5.
Skin region detection: OpenCV-based pipelines; configurable thresholds for different cameras/lighting. No persistent storage of frames.
Condition scoring: Ensemble of classical ML (Gradient Boosting, Random Forest). Calibrated outputs; predictable and debuggable.
Explainability: Saliency/SHAP-style overlays and per-feature attributions, so decisions aren’t black boxes.
Recommendations: Rule system + light ML re-ranking tied to a small product knowledge base you can swap out.
API surface: FastAPI endpoints for batch scoring and mobile integration. Clean JSON, typed schemas.
Reproducible training: Poetry + Makefile + pinned deps. Notebooks for EDA; CLI for train/eval/export.
Raspberry Pi capture (optional): Camera Module 3 + LED ring setup. Deterministic capture lighting to reduce score variance.
Tech stack and why

Streamlit UI: Fast iteration for UX flows and operator tooling.
FastAPI: Strong typing, auto-docs, simple deploy story.
scikit-learn (+ LightGBM/XGBoost optional): Baselines that are stable, explainable, and fast on CPU.
OpenCV: Composable image ops, zero drama.
Poetry: Environment hygiene and reproducible builds.
Docker: One-liner runtime parity across machines.
Repository layout

skinaizer/
├─ apps/
│  ├─ web/                 # Streamlit UI (quick demos + operator view)
│  │  ├─ app.py
│  │  └─ pages/
│  ├─ api/                 # FastAPI service (clean contracts)
│  │  ├─ main.py
│  │  └─ routers/
│  └─ tools/               # CLI utilities (batch scoring, capture helpers)
│     └─ score.py
├─ models/
│  ├─ current/             # Deployed *.pkl / *.onnx
│  └─ experiments/         # Checkpoints and trials
├─ DATA/
│  ├─ raw/                 # Source data (no faces or anonymized descriptors)
│  ├─ interim/             # Preprocessed artifacts
│  └─ products_kb.csv      # Lightweight recommender KB
├─ src/
│  └─ skinaizer/           # Core package: preprocess, infer, explain, privacy
├─ configs/
│  ├─ app.toml             # UI and runtime toggles
│  └─ model.yaml           # Features, thresholds, ensembles, calibration
├─ notebooks/              # EDA + experiments (keep deterministic)
├─ tests/                  # Unit + E2E
├─ .env.example
├─ Makefile
├─ pyproject.toml
├─ README.md
└─ LICENSE
Quickstart

Requirements: Python 3.11+, Poetry, Git. Linux/macOS/WSL tested.

git clone https://github.com/your-org/skinaizer.git
cd skinaizer

poetry install --with dev

cp .env.example .env
mkdir -p models/current DATA/interim

# Drop a demo model in models/current/, e.g. skin_gbrt.pkl

# UI
poetry run streamlit run apps/web/app.py

# API (separate shell)
poetry run uvicorn apps.api.main:app --reload --port 8000
Open:

UI: http://localhost:8501
API docs: http://localhost:8000/docs
Configuration you’ll actually touch

.env (see .env.example):

ENV=dev
LOG_LEVEL=INFO

DISABLE_IMAGE_PERSISTENCE=true
ANONYMIZE_SESSION_IDS=true
TELEMETRY_OPTOUT=true

MODEL_DIR=./models/current
PRODUCT_KB=./DATA/products_kb.csv

API_HOST=0.0.0.0
API_PORT=8000
configs/app.toml: UI toggles, capture endpoints, export permissions. configs/model.yaml: feature pipeline, condition thresholds, ensemble weights, calibration mode.

Privacy model (zero hand-waving)

Images are processed in memory. Disk writes are blocked by default.
Any optional cache is ephemeral and wiped on shutdown.
Telemetry is opt-in and aggregates only non-identifying stats (counts, model latency, score distributions).
Export functions are off by default and must be enabled explicitly.
Architecture (how pieces talk)

UI → API: Streamlit calls FastAPI with either an image patch or precomputed features.
API → Core: skinaizer package handles preprocessing, inference, calibration, explanation, and recommendation generation.
Storage: Only models and configs live on disk. No user frames unless you toggle exports on.
Extensibility: Replace products_kb.csv, swap models in models/current/, adjust thresholds in configs/model.yaml without code changes.
Raspberry Pi capture (optional, but practical)

Hardware: Raspberry Pi 5, Camera Module 3, constant-intensity LED ring.

Setup:

# On Pi
sudo raspi-config   # enable camera
sudo apt update && sudo apt install -y python3-pip libatlas-base-dev
pip3 install opencv-python-headless fastapi uvicorn pydantic
Run the capture server from apps/tools/capture_server.py (HTTP or RTSP). Point the UI to it by setting CAPTURE_ENDPOINT in .env.

Notes:

Fix the camera distance and lighting angle once. Consistency > perfection.
Use a color checker in calibration mode to anchor white balance.
API contracts

POST /v1/score

Input: base64 image patch or feature vector.
Output: condition scores, confidence, optional explanation vectors, and recommendation entries.
GET /healthz

Input: none.
Output: build info and model availability.
POST /v1/explain (optional)

Input: same payload as /v1/score.
Output: saliency map metadata or SHAP values.
Schemas live under apps/api/routers/ and are typed with Pydantic.

Training and evaluation

Use the CLI to keep runs reproducible:

# Train baseline GBRT
poetry run python src/skinaizer/pipeline.py train --config configs/model.yaml

# Evaluate with GroupKFold and save metrics
poetry run python src/skinaizer/pipeline.py eval --split groupkfold

# Export model (pickle) and optionally ONNX
poetry run python src/skinaizer/pipeline.py export --out models/current/skin_gbrt.pkl
poetry run python src/skinaizer/pipeline.py export-onnx --out models/current/skin.onnx
Method choices:

Start with Gradient Boosting and Random Forest for stability.
Add LightGBM/XGBoost when you need speed on larger feature sets.
Keep calibration on (Platt/Isotonic) if scores drive user-facing messaging.
Testing

poetry run pytest -q            # unit tests
poetry run ruff check .         # lint
poetry run mypy src             # types
poetry run pytest -q -m "e2e"   # headless E2E
What to test:

Deterministic preprocessing for the same input.
Score monotonicity across controlled lighting sets.
No disk writes when DISABLE_IMAGE_PERSISTENCE=true.
Docker

docker build -t skinaizer:latest .
docker run --rm -p 8501:8501 -p 8000:8000 --env-file .env skinaizer:latest
Use this when you need parity across dev machines or a quick server deploy behind a reverse proxy.

Makefile shortcuts

make setup          # poetry, pre-commit
make run-web        # streamlit
make run-api        # uvicorn
make train          # model training
make test           # tests + lint
make fmt            # auto-format
Roadmap (practical, not wishful)

Improved segmentation robust to mixed indoor lighting.
Model cards and datasheets shipped with each release.
iOS on-device prototype via Core ML / ONNX Runtime.
Calibration with color checker and ambient-light compensation.
Multi-illumination consensus to reduce false positives.
Differential privacy options for telemetry when enabled.
Contributing

Fork and branch: feat/<topic> or fix/<topic>.
Keep PRs tight and test-backed. Attach screenshots for UI changes.
Follow PEP 8, type hints everywhere, and prefer small pure functions over clever blobs.
License

MIT. If you aim for medical use cases, handle regulatory compliance in your jurisdiction before distribution. This repo does not constitute a medical device.

Citation

Caio Salvieti Da Silva et al. Skinaizer: Privacy-First Skin Analysis with On-Device ML, 2025. GitHub repository.
Troubleshooting that saves time

Identical scores for all images: check preprocessing; a faulty normalization can flatline features. Run make test and verify the feature variance.
Pi camera not detected: run libcamera-hello, reseat the ribbon cable, re-enable in raspi-config.
M-series Macs acting slow: use opencv-python-headless, set OMP_NUM_THREADS=1, and avoid massive SHAP runs on CPU in dev.
