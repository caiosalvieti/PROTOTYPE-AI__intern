# streamlit_app.py
import os, io, json, tempfile, importlib
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from PIL import Image
import streamlit as st

# --- import project modules (hot-reloadable) ---
M       = importlib.import_module("main")
SC      = importlib.import_module("scores")
REMOD   = importlib.import_module("rec_engine")

# ---------- helpers ----------
def _load_rec_engine():
    # Prefer the shared instance from main.py if it exists and is of correct type
    rec = getattr(M, "REC_ENGINE", None)
    if rec is not None and isinstance(rec, REMOD.RecEngine):
        return rec

    # Otherwise try common KB locations
    for p in [
        "DATA/products_kb.csv",
        "data/interim/products_kb.csv",
        "products_kb.csv",
    ]:
        if os.path.isfile(p):
            try:
                return REMOD.RecEngine(p)
            except Exception:
                pass
    return None

def _list_images(root: str) -> List[str]:
    res = []
    for base, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                res.append(os.path.join(base, f))
    return sorted(res)

def _save_uploaded(tmp_dir: str, uf) -> str:
    ext = os.path.splitext(uf.name)[1].lower() or ".jpg"
    out = os.path.join(tmp_dir, f"upload{ext}")
    with open(out, "wb") as f:
        f.write(uf.getbuffer())
    return out

def _draw_overlay(rgb: np.ndarray, box) -> np.ndarray:
    import cv2
    x, y, w, h = box
    dbg = rgb.copy()
    cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return dbg

def _run_pipeline(image_path: str, tier: str = "Core", include_device: bool = True,
                  max_dim: int = 900, min_side: int = 120, rec=None) -> Dict[str, Any]:
    rgb = M.imread_rgb(image_path)
    rgb = M.gray_world(rgb)

    # face detect
    box = M.detect_face_with_fallback(rgb, max_dim=max_dim, min_side=min_side)
    if box is None:
        return {"error": "no_face_detected"}

    x, y, w, h = box
    face = rgb[y:y + h, x:x + w]
    feats, zones = M.extract_features(face)
    profile = SC.infer_skin_profile(feats)

    # plan (RecEngine may be None)
    plan = None
    if rec is not None:
        try:
            plan = rec.recommend(feats, profile, tier=tier, include_device=include_device)
        except Exception as e:
            plan = {"error": f"rec_engine_error: {e}"}

    # debug panel bytes
    dbg_bytes = None
    try:
        with tempfile.TemporaryDirectory() as td_dbg:
            dbg_path = os.path.join(td_dbg, "debug.jpg")
            M.save_debug_panel(rgb, box, zones, dbg_path)
            with open(dbg_path, "rb") as f:
                dbg_bytes = f.read()
    except Exception:
        pass

    qa = {
        "fail": bool(feats.get("qa_fail", 0)),
        "issues": feats.get("qa_issues", ""),
        "mean_gray": float(feats.get("qa_mean_gray", 0)),
    }

    return {
        "box": [int(x), int(y), int(w), int(h)],
        "features": feats,
        "profile": profile,
        "plan": plan,
        "debug_bytes": dbg_bytes,
        "qa": qa,
        "rgb": rgb,
    }
def _render_results(src_label: str, img_source, out: Dict[str, Any]) -> None:
    st.subheader(f"Results — {src_label}")

    # -------- image(s) --------
    col_img, col_overlay = st.columns(2)

    # original (dataset path vs UploadedFile)
    if isinstance(img_source, str):
        col_img.image(img_source, caption=os.path.basename(img_source), use_container_width=True)
    else:
        # pass UploadedFile directly to avoid PIL issues with some JPEGs
        col_img.image(img_source, caption="Uploaded", use_container_width=True)

    # overlay (from debug panel if present, else draw rectangle)
    if out.get("debug_bytes"):
        col_overlay.image(out["debug_bytes"], caption="Debug panel (face + zones)", use_container_width=True)
    else:
        try:
            dbg = _draw_overlay(out["rgb"], out["box"])
            col_overlay.image(dbg, caption="Detection overlay", use_container_width=True)
        except Exception:
            pass

    # -------- QA badge --------
    qa = out.get("qa", {}) or {}
    if qa.get("fail"):
        issues = qa.get("issues") or "too_dark/too_bright"
        st.warning(f"⚠️ Retake suggested — {issues}")
    else:
        st.success("✅ PASS — good photo quality")

    # -------- profile metrics --------
    st.markdown("### Skin profile")
    prof = out.get("profile", {}) or {}
    scores = prof.get("scores", {}) or {}
    cols = st.columns(5)
    cols[0].metric("Skin type", prof.get("skin_type", "?"))
    cols[1].metric("Oiliness", f"{scores.get('oiliness', 0):.2f}")
    cols[2].metric("Dryness",  f"{scores.get('hydration', 0):.2f}")
    cols[3].metric("Redness",  f"{scores.get('redness', 0):.2f}")
    cols[4].metric("Texture",  f"{scores.get('texture', 0):.2f}")

    with st.expander("Raw JSON (profile & features)"):
        c1, c2 = st.columns(2)
        with c1:
            st.json(prof)
        with c2:
            st.json(out.get("features", {}))

    # -------- plan & reasons --------
    plan = out.get("plan")
    if not plan:
        st.info("No plan (RecEngine not loaded).")
        return
    if "error" in plan:
        st.error(plan["error"])
        return

    st.markdown(f"### Suggested routine — {plan.get('plan', 'Core')}")
    reasons = plan.get("reasons") or {}
    items   = plan.get("items", [])

    for it in items:
        name = f"{(it.get('brand') or '').title()} {it.get('name','')}".strip()
        form = it.get("form", "")
        size = it.get("size_ml")
        line = f"- **{name}** — *{form}*"
        if size:
            try:
                line += f" · {float(size):.0f} ml"
            except Exception:
                line += f" · {size} ml"
        st.markdown(line)
        if it.get("reason"):  # old-engine support
            st.caption(it["reason"])

    if reasons:  # new-engine rationale per form
        st.caption("Why these picks:")
        for f, lines in reasons.items():
            if not lines:
                continue
            with st.expander(f"{f} rationale"):
                for ln in lines:
                    st.write("• " + ln)

# ---------- UI ----------
st.set_page_config(page_title="SkinAizer — Live Analysis", page_icon="🧴", layout="wide")
st.title("SkinAizer — Analyze a selfie and generate a routine")

# Sidebar controls
with st.sidebar:
    st.header("Plan options")
    tier = st.selectbox("Plan tier", ["Starter", "Core", "Intense"], index=1)
    include_device = st.checkbox("Include device", value=True)

    st.header("Detector")
    max_dim  = st.slider("Max image dimension (px)", 600, 1600, 900, step=50)
    min_side = st.slider("Min face side (px)", 80, 240, 120, step=10)

    st.header("Dev")
    if st.button("Reload modules"):
        try:
            importlib.reload(M); importlib.reload(SC); importlib.reload(REMOD)
            st.success("Modules reloaded.")
        except Exception as e:
            st.error(f"Reload failed: {e}")

# Rec engine instance (load once per run)
REC = _load_rec_engine()
if REC is None:
    st.warning("RecEngine not loaded. Make sure DATA/products_kb.csv exists (or update the path).")

# Data roots you want to expose in the UI
DATA_DIRS = [
    "DATA/raw",
    "DATA/interim",
    "data/raw",
    "data/interim/processed",
    "data/interim/raw_sample",
]
DATA_DIRS = [p for p in DATA_DIRS if os.path.isdir(p)]

# Main interaction: Upload vs Dataset
tab1, tab2 = st.tabs(["Upload", "Pick from dataset"])

with tab1:
    left, right = st.columns([3, 1])
    with left:
        uploaded = st.file_uploader("Upload a selfie (JPG/PNG)", type=["jpg", "jpeg", "png"])
    with right:
        go_upload = st.button("Analyze uploaded", type="primary", use_container_width=True)

    if go_upload and uploaded:
        with tempfile.TemporaryDirectory() as td:
            img_path = _save_uploaded(td, uploaded)
            with st.spinner("Analyzing…"):
                out = _run_pipeline(img_path, tier=tier, include_device=include_device,
                                    max_dim=max_dim, min_side=min_side, rec=REC)
        if out.get("error"):
            st.error(out["error"])
        else:
            _render_results("Uploaded", uploaded, out)

with tab2:
    if not DATA_DIRS:
        st.info("No dataset folders found. Create e.g. `DATA/raw` and place images there.")
    else:
        root = st.selectbox("Dataset folder", DATA_DIRS)
        imgs = _list_images(root) if root else []
        if not imgs:
            st.info("No images found in the selected folder.")
        else:
            picked_img = st.selectbox("Choose an image", imgs, index=0)
            go_dataset = st.button("Analyze selected", use_container_width=True)

            if go_dataset and picked_img:
                with st.spinner("Analyzing…"):
                    out = _run_pipeline(picked_img, tier=tier, include_device=include_device,
                                        max_dim=max_dim, min_side=min_side, rec=REC)
                if out.get("error"):
                    st.error(out["error"])
                else:
                    _render_results("Dataset", picked_img, out)
