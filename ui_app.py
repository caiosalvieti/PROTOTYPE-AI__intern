# streamlit_app.py
import os, io, json, tempfile, importlib
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
from PIL import Image
import streamlit as st

# A função skinaizer_model_core é importada e usada.
# Garanta que o core/model_core.py ou core/yolo_roi.py foi corrigido 
# para carregar o modelo YOLO no caminho absoluto!
from core.yolo_roi import skinaizer_model_core #  yolo_roi.py

# page config 
st.set_page_config(page_title="SkinAizer", page_icon="🧴", layout="wide")

# import project modules (hot-reloadable)
M     = importlib.import_module("main")
SC    = importlib.import_module("scores")
REMOD = importlib.import_module("rec_engine")


# --- FUNÇÕES DE CACHE (OBJETOS PESADOS) ---

@st.cache_resource
def _load_rec_engine():
    """
    Carrega e inicializa o RecEngine. Usamos st.cache_resource
    para garantir que a leitura e parsing do products_kb.csv ocorra apenas uma vez.
    """
    # Prefer the shared instance from main.py if it exists and is of correct type
    rec = getattr(M, "REC_ENGINE", None)
    if rec is not None and isinstance(rec, REMOD.RecEngine):
        return rec

    # Otherwise try common KB locations
    for p in ["DATA/products_kb.csv", "data/interim/products_kb.csv", "products_kb.csv"]:
        if os.path.isfile(p):
            try:
                # Note: REMOD is the imported rec_engine module
                return REMOD.RecEngine(p)
            except Exception:
                pass
    return None


@st.cache_resource
def _load_skinaizer_core_model():
    """
    Garante que a inicialização do YOLO (via skinaizer_model_core)
    ocorra apenas na primeira execução do Streamlit, retornando a função.
    """
    # Retorna a função de inferência que usa o modelo YOLO cacheado internamente
    return skinaizer_model_core


# --- HELPERS (LOGIC) ---

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


def _run_pipeline(
    image_path: str,
    tier: str = "Core",
    include_device: bool = True,
    max_dim: int = 900,
    min_side: int = 120,
    rec=None,
    flags: Dict[str, bool] | None = None
) -> Dict[str, Any]:
    import time
    
    # Carrega a função de inferência do Core (agora cacheada)
    core_model_func = _load_skinaizer_core_model()

    #  total timing start 
    t_total0 = time.perf_counter()

    # 1) load + base preproc (existing logic)
    rgb = M.imread_rgb(image_path)
    rgb = M.gray_world(rgb)

    # 2) YOLO-based detection via função cacheada
    core_out = core_model_func(rgb)
    yolo_bbox = core_out.get("bbox")
    timings = core_out.get("timings", {}).copy()

    # 3) Convert YOLO bbox -> (x, y, w, h) or fallback to old detector
    if yolo_bbox is not None:
        x_min, y_min, x_max, y_max = yolo_bbox
        x = x_min
        y = y_min
        w = x_max - x_min
        h = y_max - y_min
        box = (x, y, w, h)
    else:
        # fallback to original detector if YOLO fails
        box = M.detect_face_with_fallback(rgb, max_dim=max_dim, min_side=min_side)

    if box is None:
        return {"error": "no_face_detected"}

    x, y, w, h = box

    # 4) crop face ROI (as before)
    face = rgb[y:y + h, x:x + w]

    # 5) feature extraction + profile (existing logic)
    feats, zones = M.extract_features(face)
    profile = SC.infer_skin_profile(feats)

    # merge UI flags into profile for RecEngine 
    flags = flags or {}
    if flags.get("sensitive"):
        # bump sensitivity so RecEngine prefers fragrance-free
        profile.setdefault("scores", {})
        profile["scores"]["sensitivity"] = max(profile["scores"].get("sensitivity", 0.0), 0.8)
        profile.setdefault("flags", []).append("sensitive")
    if flags.get("acne_prone"):
        profile["acne_prone"] = True
    if flags.get("pregnant"):
        profile["pregnant"] = True

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

    # total timing end 
    t_total1 = time.perf_counter()
    timings["total_pipeline_ms"] = (t_total1 - t_total0) * 1000.0

    return {
        "box": [int(x), int(y), int(w), int(h)],
        "features": feats,
        "profile": profile,
        "plan": plan,
        "debug_bytes": dbg_bytes,
        "qa": qa,
        "rgb": rgb,
        "timings": timings,
    }


def _render_results(src_label: str, img_source, out: Dict[str, Any]) -> None:
    st.subheader(f"Results — {src_label}")

    # images
    col_img, col_overlay = st.columns(2)

    # show original image (path vs UploadedFile)
    if isinstance(img_source, str):
        col_img.image(img_source, caption=os.path.basename(img_source), width="stretch")
    else:
        col_img.image(img_source, caption="Uploaded", width="stretch")
    if out.get("debug_bytes"):
        col_overlay.image(out["debug_bytes"], caption="Debug panel (face + zones)", width="stretch")
    else:
        try:
            dbg = _draw_overlay(out["rgb"], out["box"])
            col_overlay.image(dbg, caption="Detection overlay", width="stretch")
        except Exception:
            pass


    # QA badge 
    qa = out.get("qa", {}) or {}
    if qa.get("fail"):
        issues = qa.get("issues") or "too_dark/too_bright"
        st.warning(f"⚠️ Retake suggested — {issues}")
    else:
        st.success("✅ PASS - good photo quality")

    # timings
    timings = out.get("timings") or {}
    if timings:
        pretty = ", ".join(f"{k}={v:.1f} ms" for k, v in timings.items())
        st.caption(f"Model timings: {pretty}")

    # profile metrics 
    st.markdown("### Skin profile")
    prof = out.get("profile", {}) or {}
    scores = prof.get("scores", {}) or {}
    cols = st.columns(5)
    cols[0].metric("Skin type", prof.get("skin_type", "?"))
    cols[1].metric("Oiliness", f"{scores.get('oiliness', 0):.2f}")
    cols[2].metric("Dryness",  f"{scores.get('hydration', 0):.2f}")
    cols[3].metric("Redness",  f"{scores.get('redness', 0):.2f}")
    cols[4].metric("Texture",  f"{scores.get('texture', 0):.2f}")

    # show flags
    badges = []
    if prof.get("acne_prone"):
        badges.append("acne-prone")
    if prof.get("pregnant"):
        badges.append("pregnant")
    if "sensitive" in (prof.get("flags") or []) or (scores.get("sensitivity", 0) >= 0.6):
        badges.append("sensitive")
    if badges:
        st.caption("Flags: " + ", ".join(badges))

    with st.expander("Raw JSON (profile & features)"):
        c1, c2 = st.columns(2)
        with c1:
            st.json(prof)
        with c2:
            st.json(out.get("features", {}))

    # plan & reasons 
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

        tags = []
        if it.get("fragrance_free"):
            tags.append("fragrance-free")
        try:
            if float(it.get("comedogenicity") or 9) <= 2:
                tags.append("low-comedogenic")
        except Exception:
            pass
        if tags:
            st.caption(" · ".join(tags))

        if it.get("reason"):
            st.caption(it["reason"])

    if reasons:
        st.caption("Why these picks:")
        for f, lines in reasons.items():
            if not lines:
                continue
            with st.expander(f"{f} rationale"):
                for ln in lines:
                    st.write("• " + ln)


# UI 
st.title("SkinAizer: From a Selfie to your daily SkinCare")

# Sidebar controls (single consolidated block)
with st.sidebar:
    st.header("Plan options")
    tier = st.selectbox("Plan tier", ["Starter", "Core", "Intense"], index=1)
    include_device = st.checkbox("Include device", value=True)

    st.header("Profile flags")
    flag_sensitive = st.checkbox("Sensitive", value=False, help="Prefers fragrance-free options")
    flag_acne = st.checkbox("Acne-prone", value=False, help="Avoids high comedogenicity")
    flag_preg = st.checkbox("Pregnant (avoid retinoids)", value=False)

    st.header("Detector")
    max_dim  = st.slider("Max image dimension (px)", 600, 1600, 900, step=50)
    min_side = st.slider("Min face side (px)", 80, 240, 120, step=10)

    st.header("Dev")
    if st.button("Reload modules"):
        try:
            # Você pode precisar recarregar o módulo core também se ele for modificado
            # importlib.reload(core.yolo_roi)
            importlib.reload(M)
            importlib.reload(SC)
            importlib.reload(REMOD)
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
        uploaded = st.file_uploader("Upload your selfie (JPG/PNG)", type=["jpg", "jpeg", "png"])
        # (Optional) webcam capture
        # cam_img = st.camera_input("Or use your webcam")

    with right:
        go_upload = st.button("Analyze uploaded", type="primary", use_container_width=True)

    # Prefer file_uploader; you can add a webcam branch if you enabled it
    if go_upload and uploaded:
        with tempfile.TemporaryDirectory() as td:
            img_path = _save_uploaded(td, uploaded)
            with st.spinner("Analyzing…"):
                out = _run_pipeline(
                    img_path,
                    tier=tier,
                    include_device=include_device,
                    max_dim=max_dim,
                    min_side=min_side,
                    rec=REC,
                    flags={"sensitive": flag_sensitive, "acne_prone": flag_acne, "pregnant": flag_preg},
                )
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
                    out = _run_pipeline(
                        picked_img,
                        tier=tier,
                        include_device=include_device,
                        max_dim=max_dim,
                        min_side=min_side,
                        rec=REC,
                        flags={"sensitive": flag_sensitive, "acne_prone": flag_acne, "pregnant": flag_preg},
                    )
                if out.get("error"):
                    st.error(out["error"])
                else:
                    _render_results("Dataset", picked_img, out)