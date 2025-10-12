# ui_app.py
import os, io, json, tempfile, importlib
from pathlib import Path
from PIL import Image
import streamlit as st

# --- Import your modules ---
M  = importlib.import_module("main")       # your pipeline functions live here
SC = importlib.import_module("scores")     # infer_skin_profile
REMOD = importlib.import_module("rec_engine")  # RecEngine class

# --- KB / RecEngine (use main.REC_ENGINE if already created) ---
def _load_rec_engine():
    if hasattr(M, "REC_ENGINE") and M.REC_ENGINE is not None:
        return M.REC_ENGINE
    # fallback search
    for p in ["DATA/products_kb.csv", "data/interim/products_kb.csv", "products_kb.csv"]:
        if os.path.isfile(p):
            return REMOD.RecEngine(p)
    return None

REC = _load_rec_engine()

# --- Dataset roots for quick-pick (adjust if needed) ---
DATA_DIRS = ["DATA/raw", "DATA/interim", "data/raw", "data/interim/processed", "data/interim/raw_sample"]

st.set_page_config(page_title="SkinAizer — Live Analysis", page_icon="🌿", layout="wide")
st.title("🌿 SkinAizer — Analyze any face with your model")
st.caption("Upload a selfie (or pick from your dataset). We run your face→features pipeline, build a skin profile, and generate a plan.")

# ---------- helpers ----------
def _save_uploaded(tmp_dir: str, uf) -> str:
    ext = os.path.splitext(uf.name)[1].lower() or ".jpg"
    out = os.path.join(tmp_dir, f"upload{ext}")
    with open(out, "wb") as f:
        f.write(uf.getbuffer())
    return out

def _list_images(root):
    res = []
    for base, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith((".jpg",".jpeg",".png")):
                res.append(os.path.join(base, f))
    return res

def run_pipeline(image_path: str):
    """
    Uses your real pipeline from main.py:
      - imread_rgb -> gray_world -> detect_face_with_fallback
      - extract_features(face)
      - scores.infer_skin_profile
      - RecEngine.recommend
      - debug panel image
    Returns a dict with everything Streamlit needs.
    """
    # read + WB
    rgb = M.imread_rgb(image_path)
    rgb = M.gray_world(rgb)

    # face box
    box = M.detect_face_with_fallback(rgb)
    if box is None:
        return {"error": "no_face_detected"}

    x, y, w, h = box
    face = rgb[y:y+h, x:x+w]

    # features + zones
    feats, zones = M.extract_features(face)

    # profile
    profile = SC.infer_skin_profile(feats)

    # plan (if KB loaded)
    plan = None
    if REC is not None:
        plan = REC.recommend(feats, profile, tier="Core", include_device=True)

    # debug panel (saved to temp to display)
    dbg_path = None
    try:
        with tempfile.TemporaryDirectory() as td_dbg:
            dbg_path = os.path.join(td_dbg, "debug.jpg")
            M.save_debug_panel(rgb, box, zones, dbg_path)
            with open(dbg_path, "rb") as f:
                dbg_bytes = f.read()
    except Exception:
        dbg_bytes = None

    return {
        "box": [int(x), int(y), int(w), int(h)],
        "features": feats,
        "profile": profile,
        "plan": plan,
        "debug_bytes": dbg_bytes
    }

# ---------- UI ----------
left, right = st.columns([1,1])

with left:
    uploaded = st.file_uploader("Upload a selfie (JPG/PNG)", type=["jpg","jpeg","png"])
    go_upload = st.button("Analyze Uploaded", type="primary", use_container_width=True)

with right:
    candidates = [p for p in DATA_DIRS if os.path.isdir(p)]
    picked_root = st.selectbox("Or pick from dataset folder", candidates) if candidates else None
    picked_img = None
    if picked_root:
        imgs = _list_images(picked_root)
        picked_img = st.selectbox("Choose an image", imgs) if imgs else None
    go_dataset = st.button("Analyze Selected", use_container_width=True)

def _render_results(src_label, image_path_or_file, out):
    st.subheader(f"🧠 Results — {src_label}")

    # Show image
    if isinstance(image_path_or_file, str):
        st.image(Image.open(image_path_or_file), caption=os.path.basename(image_path_or_file), use_column_width=True)
    else:
        st.image(Image.open(io.BytesIO(image_path_or_file.getvalue())), caption="Uploaded", use_column_width=True)

    # Debug panel (if available)
    if out.get("debug_bytes"):
        st.image(out["debug_bytes"], caption="Debug panel (face + zones)", use_column_width=True)

    # Profile / features / plan
    c1, c2 = st.columns([1,1])
    with c1:
        st.markdown("**Skin profile**")
        st.json(out.get("profile", {}))
    with c2:
        st.markdown("**Features**")
        st.json(out.get("features", {}))

    plan = out.get("plan")
    if plan:
        st.markdown("**Suggested routine (Core)**")
        items = plan.get("items", [])
        for it in items:
            st.write(f"- **{it.get('form','')}** — {it.get('name','')} ({it.get('size_ml','?')} ml)")
            if it.get("reason"):
                st.caption(it["reason"])

# Run (uploaded)
if go_upload and uploaded:
    with tempfile.TemporaryDirectory() as td:
        img_path = _save_uploaded(td, uploaded)
        with st.spinner("Analyzing…"):
            out = run_pipeline(img_path)
    if out.get("error"):
        st.error(out["error"])
    else:
        _render_results("Uploaded", uploaded, out)

# Run (dataset)
if go_dataset and picked_img:
    with st.spinner("Analyzing…"):
        out = run_pipeline(picked_img)
    if out.get("error"):
        st.error(out["error"])
    else:
        _render_results("Dataset", picked_img, out)
