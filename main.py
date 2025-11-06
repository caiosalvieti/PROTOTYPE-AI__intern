#  imports & config UPDATE 6.11.25  #
import os, sys, glob, json, argparse, logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import cv2

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load

from scores import infer_skin_profile
from rec_engine import RecEngine

#  paths 
ROOT = Path(".").resolve()
DATA = ROOT / "DATA"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
META = DATA / "metadata"
OUT = ROOT / "experiments"
for d in [DATA, RAW, INTERIM, PROCESSED, META, OUT]:
    d.mkdir(parents=True, exist_ok=True)

#  logging 
LOG = logging.getLogger("skinaizer")
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
LOG.addHandler(handler)
LOG.setLevel(logging.INFO)

#  perf CV
cv2.setNumThreads(2)

# recommendation engine 
REC_ENGINE = RecEngine(str(DATA / "products_kb.csv"))

# constants 
QA_BLUR_MIN = 15.0
QA_SPECULAR_MAX_FRAC = 0.20
QA_WB_SHIFT_MAX = 50.0
QA_DARK_GRAY = 20.0
QA_BRIGHT_GRAY = 240.0

RED_HIGH = 0.62
SHINE_HIGH = 0.58
TEXTURE_MED = 180.0
TEXTURE_HIGH = 250.0

# FACE DETECTION: Haar + MediaPipe + DNN 
CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_face_haar(rgb: np.ndarray, max_dim: int = 800, min_side: int = 120) -> Optional[Tuple[int,int,int,int]]:
    h, w = rgb.shape[:2]
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        small = cv2.resize(rgb, (int(w * scale), int(h * scale)), cv2.INTER_AREA)
    else:
        small = rgb

    gray = cv2.cvtColor(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
    faces = CASCADE.detectMultiScale(gray, 1.1, 5)
    if len(faces) == 0:
        return None

    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    if min(fw, fh) < 40:
        return None
    inv = 1.0 / scale
    rx, ry, rw, rh = int(x * inv), int(y * inv), int(fw * inv), int(fh * inv)
    if min(rw, rh) < min_side:
        return None
    return (rx, ry, rw, rh)

#  MediaPipe
try:
    import mediapipe as mp
    mp_fd = mp.solutions.face_detection
    MP_OK = True
except Exception:
    mp_fd = None
    MP_OK = False

def detect_face_mediapipe(rgb: np.ndarray, min_side: int = 120, conf: float = 0.5):
    if not MP_OK:
        return None
    H, W = rgb.shape[:2]
    with mp_fd.FaceDetection(model_selection=0, min_detection_confidence=conf) as det:
        res = det.process(rgb)
    if not res.detections:
        return None
    best, best_area = None, -1
    for d in res.detections:
        bb = d.location_data.relative_bounding_box
        x1 = max(0, int(bb.xmin * W))
        y1 = max(0, int(bb.ymin * H))
        x2 = min(W - 1, int((bb.xmin + bb.width) * W))
        y2 = min(H - 1, int((bb.ymin + bb.height) * H))
        w = x2 - x1
        h = y2 - y1
        if min(w, h) < min_side:
            continue
        a = w * h
        if a > best_area:
            best_area, best = a, (x1, y1, w, h)
    return best

# OpenCV DNN SSD 
DNN_PROTO   = ROOT / "tools" / "models" / "deploy.prototxt.txt"
DNN_WEIGHTS = ROOT / "tools" / "models" / "res10_300x300_ssd_iter_140000.caffemodel"
try:
    DNN_NET = cv2.dnn.readNetFromCaffe(str(DNN_PROTO), str(DNN_WEIGHTS)) if DNN_PROTO.is_file() and DNN_WEIGHTS.is_file() else None
except Exception:
    DNN_NET = None

def detect_face_dnn(rgb: np.ndarray, min_side: int = 120, conf_thresh: float = 0.5):
    if DNN_NET is None:
        return None
    H, W = rgb.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                 scalefactor=1.0, size=(300, 300),
                                 mean=(104.0, 177.0, 123.0), swapRB=False, crop=False)
    DNN_NET.setInput(blob)
    dets = DNN_NET.forward()
    best, best_conf = None, 0.0
    for i in range(dets.shape[2]):
        conf = float(dets[0, 0, i, 2])
        if conf < conf_thresh:
            continue
        x1, y1, x2, y2 = (dets[0, 0, i, 3:7] * np.array([W, H, W, H])).astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W - 1, x2), min(H - 1, y2)
        w, h = x2 - x1, y2 - y1
        if min(w, h) < min_side:
            continue
        if conf > best_conf:
            best_conf, best = conf, (x1, y1, w, h)
    return best

def detect_face_with_fallback(rgb: np.ndarray, max_dim: int = 800, min_side: int = 120):
    #  MediaPipe
    box = detect_face_mediapipe(rgb, min_side=min_side, conf=0.5)
    if box is not None:
        return box
    #  Haar
    box = detect_face_haar(rgb, max_dim=max_dim, min_side=min_side)
    if box is not None:
        return box
    #  DNN
    return detect_face_dnn(rgb, min_side=min_side, conf_thresh=0.5)

# utils 
def ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def imread_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def imwrite_rgb(path: str, img_rgb: np.ndarray) -> None:
    cv2.imwrite(path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

def gray_world(img_rgb: np.ndarray) -> np.ndarray:
    img = img_rgb.astype(np.float32)
    mean = np.mean(img, axis=(0, 1)) + 1e-6
    scale = np.mean(mean) / mean
    img *= scale
    return np.clip(img, 0, 255).astype(np.uint8)

# QA & features 
def qa_checks(face_rgb: np.ndarray):
    gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
    mean_gray = float(np.mean(gray))
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    hsv = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    specular = float(((v > 240) & (s < 30)).sum()) / v.size
    r, g, b = [np.mean(c) for c in cv2.split(face_rgb)]
    wb_shift = float(max(abs(r - b), abs(g - b)))

    issues = []
    if mean_gray < QA_DARK_GRAY: issues.append("too_dark")
    if mean_gray > QA_BRIGHT_GRAY: issues.append("too_bright")
    if blur < QA_BLUR_MIN: issues.append("blurry")
    if specular > QA_SPECULAR_MAX_FRAC: issues.append("glare_wet")
    if wb_shift > QA_WB_SHIFT_MAX: issues.append("colored_light")

    fail = ("too_dark" in issues) or ("too_bright" in issues)
    qa = dict(mean_gray=mean_gray, blur=blur, specular=specular, wb_shift=wb_shift)
    return issues, fail, qa

def crop_zones(face_rgb: np.ndarray) -> Dict[str, np.ndarray]:
    h, w, _ = face_rgb.shape
    z = {}
    fw, fh = int(0.6 * w), int(0.25 * h); fx, fy = (w - fw) // 2, 0
    z["forehead"] = face_rgb[fy:fy + fh, fx:fx + fw]
    y1, y2 = int(0.30 * h), int(0.65 * h); xL2, xR1 = int(0.40 * w), int(0.60 * w)
    z["cheek_left"]  = face_rgb[y1:y2, 0:xL2]
    z["cheek_right"] = face_rgb[y1:y2, xR1:w]
    cy1, cy2 = int(0.65 * h), int(0.95 * h)
    z["chin"] = face_rgb[cy1:cy2, fx:fx + fw]
    return z

def redness_index(rgb: np.ndarray) -> float:
    r, g, b = cv2.split(rgb.astype(np.float32))
    s = r + g + b + 1e-6
    return float(np.mean(r / s))

def texture_index(rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def shine_index(rgb: np.ndarray) -> float:
    gray = cv2.medianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), 3)
    return float(np.mean(gray)) / 255.0

def extract_features(face_rgb: np.ndarray):
    issues, fail, qa = qa_checks(face_rgb)
    zones = crop_zones(face_rgb)
    feats = {}
    for name, z in zones.items():
        feats[f"{name}_red"] = redness_index(z)
        feats[f"{name}_txt"] = texture_index(z)
        feats[f"{name}_shn"] = shine_index(z)
    feats.update({
        "global_red": redness_index(face_rgb),
        "global_txt": texture_index(face_rgb),
        "global_shn": shine_index(face_rgb),
        **{f"qa_{k}": v for k, v in qa.items()},
        "qa_fail": int(fail),
        "qa_issues": "|".join(issues),
    })
    return feats, zones

def save_debug_panel(rgb: np.ndarray, face_box: Tuple[int,int,int,int], zones: Dict[str, np.ndarray], save_path: str):
    x, y, w, h = face_box
    dbg = rgb.copy()
    cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 0), 2)
    face = rgb[y:y + h, x:x + w].copy()
    h2, w2, _ = face.shape
    fw, fh = int(0.6 * w2), int(0.25 * h2); fx, fy = (w2 - fw) // 2, 0
    cv2.rectangle(face, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)
    y1, y2 = int(0.30 * h2), int(0.65 * h2); xL2, xR1 = int(0.40 * w2), int(0.60 * w2)
    cv2.rectangle(face, (0, y1), (xL2, y2), (255, 0, 0), 2)
    cv2.rectangle(face, (xR1, y1), (w2, y2), (255, 0, 0), 2)
    cy1, cy2 = int(0.65 * h2), int(0.95 * h2)
    cv2.rectangle(face, (fx, cy1), (fx + fw, cy2), (255, 0, 0), 2)
    panel = np.hstack([
        cv2.resize(dbg, (rgb.shape[1], rgb.shape[0])),
        cv2.resize(face, (rgb.shape[1], rgb.shape[0]))
    ])
    imwrite_rgb(save_path, panel)

# dataset (images -> features.csv) 
def build_dataset(input_glob: str, save_csv: str, save_debug_imgs: bool = True,
                  max_files: Optional[int] = None, log_every: int = 50, checkpoint_every: int = 200,
                  max_dim: int = 800, min_side: int = 120, debug_dir: Path = INTERIM):
    rows: List[Dict] = []
    paths = sorted(glob.glob(input_glob))
    if max_files:
        paths = paths[:max_files]
    total = len(paths)
    LOG.info(f"Building dataset from {total} files")

    for i, p in enumerate(paths, 1):
        try:
            img = gray_world(imread_rgb(p))
            box = detect_face_with_fallback(img, max_dim=max_dim, min_side=min_side)
            if box is None:
                rows.append(dict(image_id=os.path.basename(p), error="no_face"))
            else:
                x, y, w, h = box
                if min(w, h) < 100:
                    rows.append(dict(image_id=os.path.basename(p), error="face_too_small"))
                else:
                    face = img[y:y + h, x:x + w]
                    feats, zones = extract_features(face)
                    feats.update(dict(image_id=os.path.basename(p), error=""))
                    rows.append(feats)
                    if save_debug_imgs:
                        dbg_path = debug_dir / f"debug_{Path(p).stem}.jpg"
                        save_debug_panel(img, box, zones, str(dbg_path))
        except Exception as e:
            rows.append(dict(image_id=os.path.basename(p), error=str(e)))

        if (i % log_every == 0) or (i == total):
            LOG.info(f"[build] {i}/{total} processed")
        if (i % checkpoint_every == 0) or (i == total):
            pd.DataFrame(rows).to_csv(save_csv, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(save_csv, index=False)
    LOG.info(f"[dataset] saved {save_csv} with {len(df)} rows")
    return df

#  train 
def train_baseline(features_csv: str, labels_csv: str, model_path: str):
    feats = pd.read_csv(features_csv)
    if not Path(labels_csv).is_file():
        LOG.warning("[train] labels.csv not found; skipping.")
        return
    labels = pd.read_csv(labels_csv)
    df = feats.merge(labels, on="image_id")
    df = df[df["error"] == ""]
    X = df.select_dtypes(include=[np.number]).copy()
    y = df["target"]

    # RF is scale-invariant; drop QA flag if present
    X = X.drop(columns=[c for c in ["qa_fail"] if c in X.columns], errors="ignore")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1s, bals, aucs = [], [], []
    for tr, te in skf.split(X, y):
        clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
        clf.fit(X.iloc[tr], y.iloc[tr])
        pred = clf.predict(X.iloc[te])
        f1s.append(f1_score(y.iloc[te], pred, average="macro"))
        bals.append(balanced_accuracy_score(y.iloc[te], pred))
        if len(np.unique(y)) == 2:
            proba = clf.predict_proba(X.iloc[te])[:, 1]
            try:
                aucs.append(roc_auc_score(y.iloc[te], proba))
            except Exception:
                pass
    msg = f"[train] F1(macro) {np.mean(f1s):.3f}±{np.std(f1s):.3f} | BalAcc {np.mean(bals):.3f}±{np.std(bals):.3f}"
    if aucs:
        msg += f" | ROC-AUC {np.mean(aucs):.3f}±{np.std(aucs):.3f}"
    LOG.info(msg)

    clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    clf.fit(X, y)
    dump(dict(model=clf, features=list(X.columns)), model_path)
    LOG.info(f"[train] saved model -> {model_path}")

# recs 
def recommend_from_features(r: Dict[str, float]) -> List[str]:
    recs: List[str] = []
    red, txt, shn = r.get("global_red", 0.0), r.get("global_txt", 0.0), r.get("global_shn", 0.0)
    dark = r.get("qa_mean_gray", 128) < 45
    bright = r.get("qa_mean_gray", 128) > 210
    blur = r.get("qa_blur", 200) < 100
    glare = r.get("qa_specular", 0.0) > 0.04
    wb = r.get("qa_wb_shift", 0.0) > 18

    if any([dark, bright, blur, glare, wb]):
        if dark:   recs.append("Retake: brighter room.")
        if bright: recs.append("Retake: avoid overexposed light.")
        if blur:   recs.append("Retake: hold still / focus.")
        if glare:  recs.append("Retake: pat skin dry / avoid direct lamp.")
        if wb:     recs.append("Retake: use neutral white light.")
        return recs

    if red > RED_HIGH:     recs += ["Barrier repair: ceramides+cholesterol+FFA.", "Niacinamide 2–5%.", "Daily SPF 30+."]
    if shn > SHINE_HIGH and txt > TEXTURE_MED:
        recs += ["BHA 0.5–2% (3x/week).", "Light gel moisturizer.", "Adapalene 0.1% (2x/week, buffer)."]
    if txt > TEXTURE_HIGH: recs += ["Gentle PHA/AHA (1–2x/week).", "Humectant + occlusive at night."]
    if not recs:           recs.append("Gentle cleanser + ceramide moisturizer + SPF.")
    return recs

#  infer 
def infer(image_path: str, model_path: str, save_debug: bool = True,
          max_dim: int = 800, min_side: int = 120):
    img = gray_world(imread_rgb(image_path))
    box = detect_face_with_fallback(img, max_dim=max_dim, min_side=min_side)
    if box is None:
        LOG.error("[infer] no face detected")
        return

    x, y, w, h = box
    face = img[y:y + h, x:x + w]
    feats, zones = extract_features(face)

    if save_debug:
        dbg = INTERIM / f"debug_{ts()}.jpg"
        save_debug_panel(img, box, zones, str(dbg))
        LOG.info(f"[infer] debug -> {dbg}")

    LOG.info(json.dumps(feats, indent=2))

    #  model inference
    if Path(model_path).is_file():
        bundle = load(model_path); model, cols = bundle["model"], bundle["features"]
        X = pd.DataFrame([{k: feats.get(k, 0.0) for k in cols}])
        pred = model.predict(X)[0]
        LOG.info(f"[infer] model -> {pred}")

    # heuristic recs + profile/plan
    recs = recommend_from_features(feats)
    [LOG.info(" - " + r) for r in recs]

    profile = infer_skin_profile(feats)
    plan = REC_ENGINE.recommend(feats, profile, tier="Core", include_device=True)
    print("\n[profile]", json.dumps(profile, indent=2))
    print("\n[plan]", json.dumps(plan, indent=2))

# CLI 
def main():
    ap = argparse.ArgumentParser(description="SkinAizer – Skin AI prototype")
    ap.add_argument("-v", "--verbose", action="store_true", help="verbose logs")

    sp = ap.add_subparsers(dest="cmd")

    b = sp.add_parser("build-dataset")
    b.add_argument("--glob", default=str(RAW / "*.*"))
    b.add_argument("--max", type=int, default=None)
    b.add_argument("--log-every", type=int, default=50)
    b.add_argument("--checkpoint-every", type=int, default=200)
    b.add_argument("--max-dim", type=int, default=800)
    b.add_argument("--min-side", type=int, default=120)

    t = sp.add_parser("train")
    t.add_argument("--features", default=str(META / "features.csv"))
    t.add_argument("--labels", default=str(META / "labels.csv"))
    t.add_argument("--model", default=str(OUT / "baseline_rf.joblib"))

    i = sp.add_parser("infer")
    i.add_argument("--image", required=True)
    i.add_argument("--model", default=str(OUT / "baseline_rf.joblib"))
    i.add_argument("--no-debug", action="store_true")
    i.add_argument("--max-dim", type=int, default=800)
    i.add_argument("--min-side", type=int, default=120)

    args = ap.parse_args()
    if args.verbose:
        LOG.setLevel(logging.DEBUG)

    if args.cmd == "build-dataset":
        build_dataset(input_glob=args.glob,
                      save_csv=str(META / "features.csv"),
                      max_files=args.max,
                      log_every=args.log_every,
                      checkpoint_every=args.checkpoint_every,
                      max_dim=args.max_dim,
                      min_side=args.min_side)
    elif args.cmd == "train":
        train_baseline(args.features, args.labels, args.model)
    elif args.cmd == "infer":
        infer(args.image, args.model, save_debug=not args.no_debug,
              max_dim=args.max_dim, min_side=args.min_side)
    else:
        print("Usage:")
        print("  python main.py build-dataset --glob 'DATA/raw/*.*' --max 200")
        print("  python main.py train --features DATA/metadata/features.csv --labels DATA/metadata/labels.csv")
        print("  python main.py infer --image path.jpg")

if __name__ == "__main__":
    main()
