# ui_app.py
import os, io, json, tempfile, importlib, hashlib, base64
from pathlib import Path
from PIL import Image
import streamlit as st

# ---------- Imports from your project ----------
M  = importlib.import_module("main")         # your pipeline
SC = importlib.import_module("scores")       # infer_skin_profile
REMOD = importlib.import_module("rec_engine")# RecEngine

# ---------- App config ----------
st.set_page_config(page_title="SkinAizer — Live Analysis", page_icon="🌿", layout="wide")

# ---------- Minimal CSS (cards / pills / header) ----------
st.markdown("""
<style>
/* Centered, clean hero */
.hero {
  padding: 1.0rem 0 0.5rem 0;
}
h1 { letter-spacing: -0.02em; }
.small { opacity:.7; font-size:.9rem; }

/* Card */
.card {
  border: 1px solid rgba(255,255,255,0.08);
  background: var(--secondary-background-color);
  border-radius: 16px; padding: 16px; margin: 6px 0;
  box-shadow: 0 1px 2px rgba(0,0,0,0.25);
}
.card .title { font-weight: 600; font-size: 1.0rem; margin-bottom: 4px; }
.card .desc { opacity:.8; font-size:.9rem; }
.pill {
  display:inline-block; padding: 2px 10px; border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.12); font-size:.75rem; opacity:.9; margin-right:6px;
}
.badge {
  display:inline-block; padding: 2px 8px; border-radius: 8px; font-size:.75rem;
  background: rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.1); margin:2px 6px 2px 0;
}
code { background: rgba(255,255,255,0.06); padding: 2px 6px; border-radius: 6px; }
hr { border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 10px 0; }
a.btn {
  text-decoration:none; font-weight:600; border:1px solid rgba(255,255,255,0.18);
  padding:8px 12px; border-radius:12px; display:inline-block; margin-top:6px;
}
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown(
    "<div class='hero'><h1>🌿 SkinAizer</h1><div class='small'>Analyze a face • build a profile • suggest a routine</div></div>",
    unsafe_allow_html=True
)

# ---------- Sidebar controls ----------
st.sidebar.markdown("### Controls")
tier = st.sidebar.radio("Plan tier", ["Starter", "Core", "Intense"], index=1, horizontal=True)
include_device = st.sidebar.toggle("Include device", value=True)
show_debug = st.sidebar.toggle("Show debug panel (zones)", value=True)
st.sidebar.divider()
st.sidebar.markdown("**Tip**: Press **R** to rerun after changing sidebar settings.", help=None)

# ---------- RecEngine loader (cached) ----------
@st.cache_resource
def _load_rec_engine():
    # Prefer instance from main if present
    if hasattr(M, "REC_ENGINE") and getattr(M, "REC_ENGINE") is not None:
        return getattr(M, "REC_ENGINE")
    for p in ["DATA/products_kb.csv", "data/interim/products_kb.csv", "data/products_kb.csv", "products_kb.csv"]:
        if os.path.isfile(p):
            return REMOD.RecEngine(p)
    return None

REC = _load_rec_engine()

# ---------- Utils ----------
DATA_DIRS = ["DATA/raw", "DATA/interim", "data/raw", "data/interim/processed", "data/interim/raw_sample"]

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

def _hash_file(path_or_bytes) -> str:
    h = hashlib.md5()
    if isinstance(path_or_bytes, (bytes, bytearray)):
        h.update(path_or_bytes)
    elif isinstance(path_or_bytes, str):
        with open(path_or_bytes, "rb") as f:
            h.update(f.read())
    return h.hexdigest()

# ---------- Core pipeline (uses your code) ----------
def run_pipeline(image_path: str):
    rgb = M.imread_rgb(image_path)
    rgb = M.gray_world(rgb)
    box = M.detect_face_with_fallback(rgb)
    if box is None:
        return {"error": "No face detected. Try a clearer, front-facing image in neutral light."}

    x, y, w, h = box
    face = rgb[y:y+h, x:x+w]
    feats, zones = M.extract_features(face)
    profile = SC.infer_skin_profile(feats)

    plan = REC.recommend(feats, profile, tier=tier, include_device=include_device) if REC else None

    dbg_bytes = None
    if show_debug:
        try:
            with tempfile.TemporaryDirectory() as td_dbg:
                dbg_path = os.path.join(td_dbg, "debug.jpg")
                M.save_debug_panel(rgb, box, zones, dbg_path)
                with open(dbg_path, "rb") as f: dbg_bytes = f.read()
        except Exception:
            pass

    return dict(features=feats, profile=profile, plan=plan, debug_bytes=dbg_bytes, face_box=[int(v) for v in [x,y,w,h]])

# ---------- Layout: tabs ----------
tab_analyze, tab_results, tab_plan = st.tabs(["🔎 Analyze", "🧠 Profile & Features", "✨ Suggested Routine"])

with tab_analyze:
    left, right = st.columns([1,1], vertical_alignment="top")

    with left:
        up = st.file_uploader("Upload a selfie", type=["jpg","jpeg","png"])
        go_up = st.button("Analyze Uploaded", type="primary", use_container_width=True)

    with right:
        roots = [p for p in DATA_DIRS if os.path.isdir(p)]
        root = st.selectbox("…or choose a dataset folder", roots) if roots else None
        pick = st.selectbox("Pick an image", _list_images(root)) if root else None
        go_ds = st.button("Analyze Selected", use_container_width=True)

    st.markdown("<div class='small'>We don’t store uploads. Images are processed in-session only.</div>", unsafe_allow_html=True)

# Use session_state to carry results across tabs
if "last_out" not in st.session_state: st.session_state.last_out = None
if "last_src" not in st.session_state: st.session_state.last_src = None
if "last_img" not in st.session_state: st.session_state.last_img = None

def _render_image_inline(src_label, src_file_or_path):
    if isinstance(src_file_or_path, str):
        st.image(Image.open(src_file_or_path), use_container_width=True, caption=src_label)
    else:
        st.image(Image.open(io.BytesIO(src_file_or_path.getvalue())), use_container_width=True, caption=src_label)

# Trigger analysis
if tab_analyze:
    if go_up and up:
        with tempfile.TemporaryDirectory() as td:
            path = _save_uploaded(td, up)
            with st.spinner("Analyzing…"):
                out = run_pipeline(path)
        st.session_state.last_out = out
        st.session_state.last_src = "Uploaded"
        st.session_state.last_img = up

    if go_ds and pick:
        with st.spinner("Analyzing…"):
            out = run_pipeline(pick)
        st.session_state.last_out = out
        st.session_state.last_src = "Dataset"
        st.session_state.last_img = pick

# ---------- Results tab ----------
with tab_results:
    out = st.session_state.last_out
    if not out:
        st.info("Run an analysis in the **Analyze** tab first.")
    elif out.get("error"):
        st.error(out["error"])
    else:
        # Left: image + debug
        c1, c2 = st.columns([1,1])
        with c1:
            _render_image_inline(st.session_state.last_src, st.session_state.last_img)
            if out.get("debug_bytes"):
                st.image(out["debug_bytes"], caption="Debug panel (face + zones)", use_container_width=True)

        # Right: profile metrics
        prof = out.get("profile", {})
        scores = (prof.get("scores") or {})
        st.markdown("##### Profile")
        k1,k2,k3 = st.columns(3)
        k1.metric("Skin type", prof.get("skin_type","—"))
        k2.metric("Oiliness", f"{scores.get('oiliness',0):.2f}")
        k3.metric("Dryness", f"{scores.get('dryness',0):.2f}")
        k4,k5,k6 = st.columns(3)
        k4.metric("Redness",  f"{scores.get('redness',0):.2f}")
        k5.metric("Texture",  f"{scores.get('texture',0):.2f}")
        k6.metric("Sensitivity", f"{scores.get('sensitivity',0):.2f}")

        st.markdown("##### Prioritized concerns")
        concerns = prof.get("prioritized_concerns", [])
        if concerns:
            st.markdown(" ".join([f"<span class='badge'>{c}: {v:.2f}</span>" for c,v in concerns[:6]]), unsafe_allow_html=True)
        else:
            st.write("—")

        st.markdown("##### Raw features")
        st.json(out.get("features", {}))

        # Download button
        payload = dict(source=st.session_state.last_src, profile=prof, features=out.get("features"), face_box=out.get("face_box"))
        st.download_button("Download JSON", data=json.dumps(payload, indent=2), file_name="skin_profile.json")

# ---------- Plan tab ----------
with tab_plan:
    out = st.session_state.last_out
    if not out:
        st.info("Run an analysis in the **Analyze** tab first.")
    elif out.get("error"):
        st.error(out["error"])
    else:
        plan = out.get("plan") or {}
        if not plan:
            st.warning("No plan available (missing KB?).")
        else:
            st.markdown(f"**Tier:** `{plan.get('plan','—')}` • **Target weeks:** `{plan.get('target_weeks','—')}` • **Skin type:** `{plan.get('skin_type','—')}`")
            st.divider()
            items = plan.get("items", [])
            if not items:
                st.write("—")
            for it in items:
                name = it.get("name","")
                form = it.get("form","")
                size_ml = it.get("size_ml","")
                reason = it.get("reason","")
                link = it.get("link","") or "#"
                brand = it.get("brand","")
                st.markdown(f"""
                <div class='card'>
                  <div class='pill'>{form or 'product'}</div>
                  <div class='title'>{name}</div>
                  <div class='desc'>{brand} • {size_ml} ml</div>
                  <div class='small' style='margin-top:6px;'>{reason}</div>
                  <a class='btn' href='{link}' target='_blank' rel='noopener'>View</a>
                </div>
                """, unsafe_allow_html=True)

# ---------- Footer ----------
st.write("")
st.markdown("<div class='small'>© SkinAizer prototype • locally processed • made with ❤️</div>", unsafe_allow_html=True)
