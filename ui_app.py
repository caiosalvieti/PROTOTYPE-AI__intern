# streamlit_app.py — FULL VERSION (with persistence + Bridge chat)

import os, json, tempfile, importlib
from typing import Dict, Any, List, Optional

import numpy as np
import streamlit as st

# YOLO core (cached)
from core.yolo_roi import skinaizer_model_core

# page config
st.set_page_config(page_title="SkinAizer", page_icon="🧴", layout="wide")

# import project modules (hot-reloadable)
M     = importlib.import_module("main")
SC    = importlib.import_module("scores")
REMOD = importlib.import_module("rec_engine")

# =========================
# GLOBAL PERSISTENCE (KEY FIX)
# =========================
if "pipeline_out" not in st.session_state:
    st.session_state["pipeline_out"] = None
if "img_source" not in st.session_state:
    st.session_state["img_source"] = None
if "img_label" not in st.session_state:
    st.session_state["img_label"] = ""

# =========================
# BRIDGE CHAT (4–5 questions)
# =========================
BRIDGE_FUNNEL = [
    ("goal", "What do you most want to improve?", ["Gloss control", "Redness relief", "Deep hydration", "Blemish control", "Spot fading"]),
    ("sensitive", "Does your skin react easily to new products?", ["Yes", "No"]),
    ("avoid", "Any ingredients you want to avoid?", ["Avoid alcohol", "Avoid fragrance", "No restrictions"]),
    ("sun", "What is your daily sun exposure level?", ["Low", "Medium", "High"]),
    ("plan", "Routine length?", ["3 weeks", "1 month"]),
]

def bridge_init():
    st.session_state.setdefault("bridge_step", 0)
    st.session_state.setdefault("bridge_messages", [])
    st.session_state.setdefault("bridge_answers", {})

def bridge_render() -> tuple[bool, Dict[str, str]]:
    bridge_init()

    # history
    for m in st.session_state["bridge_messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    step = st.session_state["bridge_step"]
    if step >= len(BRIDGE_FUNNEL):
        return True, st.session_state["bridge_answers"]

    key, q, options = BRIDGE_FUNNEL[step]
    with st.chat_message("assistant"):
        st.markdown(q)
        cols = st.columns(len(options))
        for i, opt in enumerate(options):
            if cols[i].button(opt, key=f"bridge_{step}_{i}"):
                st.session_state["bridge_messages"].append({"role": "assistant", "content": q})
                st.session_state["bridge_messages"].append({"role": "user", "content": opt})
                st.session_state["bridge_answers"][key] = opt
                st.session_state["bridge_step"] += 1
                st.rerun()

    return False, st.session_state["bridge_answers"]

def bridge_weights_and_rules(feats: Dict[str, Any], answers: Dict[str, str]) -> tuple[Dict[str, float], Dict[str, Any]]:
    oiliness = float(feats.get("global_shn", 0.0))
    redness  = float(feats.get("global_red", 0.0))
    texture  = float(feats.get("global_txt", 0.0))

    weights = {
        "sebum_control": oiliness,
        "soothing": redness,
        "texture": texture,
        "spf": 0.3,
    }

    goal = answers.get("goal")
    if goal == "Gloss control":
        weights["sebum_control"] += 0.7
    elif goal == "Redness relief":
        weights["soothing"] += 0.7
    elif goal == "Deep hydration":
        weights["soothing"] += 0.2
    elif goal == "Blemish control":
        weights["sebum_control"] += 0.4
    elif goal == "Spot fading":
        weights["texture"] += 0.3

    sun = answers.get("sun")
    if sun == "High":
        weights["spf"] += 0.7
    elif sun == "Medium":
        weights["spf"] += 0.4

    s = sum(weights.values()) or 1.0
    weights = {k: v / s for k, v in weights.items()}

    rules = {
        "exclude_alcohol": (answers.get("avoid") == "Avoid alcohol"),
        "exclude_fragrance": (answers.get("avoid") == "Avoid fragrance"),
        "sensitive": (answers.get("sensitive") == "Yes"),
        "plan_length": "3_weeks" if answers.get("plan") == "3 weeks" else "1_month",
        "goal": answers.get("goal"),
        "sun_exposure": answers.get("sun"),
    }
    return weights, rules

def bridge_merge_into_profile(base_profile: Dict[str, Any], rules: Dict[str, Any], weights: Dict[str, float]) -> Dict[str, Any]:
    """
    Merge chat-derived context into profile in a way that doesn't break RecEngine.
    """
    prof = json.loads(json.dumps(base_profile))  # quick deep copy

    prof["bridge"] = {"rules": rules, "weights": weights}

    # make sensitive compatible with your existing scoring
    if rules.get("sensitive"):
        prof.setdefault("scores", {})
        prof["scores"]["sensitivity"] = max(float(prof["scores"].get("sensitivity", 0.0)), 0.8)
        prof.setdefault("flags", [])
        if "sensitive" not in prof["flags"]:
            prof["flags"].append("sensitive")

    return prof

def reset_chat_only():
    for k in ["bridge_step", "bridge_messages", "bridge_answers"]:
        if k in st.session_state:
            del st.session_state[k]

# =========================
# CACHE RESOURCES
# =========================
@st.cache_resource
def _load_rec_engine():
    rec = getattr(M, "REC_ENGINE", None)
    if rec is not None and isinstance(rec, REMOD.RecEngine):
        return rec

    for p in ["DATA/products_kb.csv", "data/interim/products_kb.csv", "products_kb.csv"]:
        if os.path.isfile(p):
            try:
                return REMOD.RecEngine(p)
            except Exception:
                pass
    return None

@st.cache_resource
def _load_skinaizer_core_model():
    return skinaizer_model_core

# =========================
# HELPERS
# =========================
def _save_uploaded(tmp_dir: str, uf) -> str:
    ext = os.path.splitext(uf.name)[1].lower() or ".jpg"
    out = os.path.join(tmp_dir, f"upload{ext}")
    with open(out, "wb") as f:
        f.write(uf.getbuffer())
    return out

def _list_images(root: str) -> List[str]:
    res = []
    for base, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                res.append(os.path.join(base, f))
    return sorted(res)

def _draw_overlay(rgb: np.ndarray, box) -> np.ndarray:
    import cv2
    x, y, w, h = box
    dbg = rgb.copy()
    cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return dbg

# =========================
# PIPELINE (YOUR FULL VERSION)
# =========================
def _run_pipeline(
    image_path: str,
    tier: str = "Core",
    include_device: bool = True,
    max_dim: int = 900,
    min_side: int = 120,
    rec=None,
    flags: Optional[Dict[str, bool]] = None
) -> Dict[str, Any]:
    import time

    W_proc, H_proc = 640, 640

    core_model_func = _load_skinaizer_core_model()

    t_total0 = time.perf_counter()

    # 1) load + base preproc
    rgb = M.imread_rgb(image_path)
    H_orig, W_orig = rgb.shape[:2]
    rgb = M.gray_world(rgb)

    # 2) YOLO-based detection
    core_out = core_model_func(rgb)
    yolo_bbox = core_out.get("bbox")
    timings = core_out.get("timings", {}).copy()

    # 3) Convert YOLO bbox -> (x, y, w, h) and RESCALE, or fallback
    if yolo_bbox is not None:
        x_min_proc, y_min_proc, x_max_proc, y_max_proc = yolo_bbox

        scale_w = W_orig / W_proc
        scale_h = H_orig / H_proc

        x_min = int(x_min_proc * scale_w)
        y_min = int(y_min_proc * scale_h)
        x_max = int(x_max_proc * scale_w)
        y_max = int(y_max_proc * scale_h)

        x = x_min
        y = y_min
        w = x_max - x_min
        h = y_max - y_min
        box = (x, y, w, h)
    else:
        box = M.detect_face_with_fallback(rgb, max_dim=max_dim, min_side=min_side)

    if box is None:
        return {"error": "no_face_detected"}

    x, y, w, h = box

    # 4) crop face ROI
    face = rgb[y:y + h, x:x + w]

    # 5) feature extraction + profile
    feats, zones = M.extract_features(face)
    profile = SC.infer_skin_profile(feats)

    # merge UI flags into profile for RecEngine
    flags = flags or {}
    if flags.get("sensitive"):
        profile.setdefault("scores", {})
        profile["scores"]["sensitivity"] = max(profile["scores"].get("sensitivity", 0.0), 0.8)
        profile.setdefault("flags", []).append("sensitive")
    if flags.get("acne_prone"):
        profile["acne_prone"] = True
    if flags.get("pregnant"):
        profile["pregnant"] = True

    # plan
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

# =========================
# RENDER
# =========================
def _render_results(out: Dict[str, Any], img_source, img_label: str, rec, tier: str, include_device: bool):
    st.subheader(f"Results — {img_label}")

    # images
    col_img, col_overlay = st.columns(2)
    col_img.image(img_source, caption="Original", width="stretch")

    if out.get("debug_bytes"):
        col_overlay.image(out["debug_bytes"], caption="Debug panel (face + zones)", width="stretch")
    else:
        try:
            dbg = _draw_overlay(out["rgb"], out["box"])
            col_overlay.image(dbg, caption="Detection overlay", width="stretch")
        except Exception:
            pass

    # QA
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

    # profile
    st.markdown("### Skin profile")
    prof = out.get("profile", {}) or {}
    scores = prof.get("scores", {}) or {}
    cols = st.columns(5)
    cols[0].metric("Skin type", prof.get("skin_type", "?"))
    cols[1].metric("Oiliness", f"{scores.get('oiliness', 0):.2f}")
    cols[2].metric("Dryness",  f"{scores.get('hydration', 0):.2f}")
    cols[3].metric("Redness",  f"{scores.get('redness', 0):.2f}")
    cols[4].metric("Texture",  f"{scores.get('texture', 0):.2f}")

    # base plan
    plan = out.get("plan")
    if not plan:
        st.info("No plan (RecEngine not loaded).")
    elif isinstance(plan, dict) and "error" in plan:
        st.error(plan["error"])
    else:
        st.markdown(f"### Suggested routine — {plan.get('plan', 'Core')}")
        items = plan.get("items", [])
        for it in items:
            name = f"{(it.get('brand') or '').title()} {it.get('name','')}".strip()
            st.write(f"- **{name}**")

    # BRIDGE
    st.divider()
    st.subheader("Personalize your routine (chat)")

    ready, answers = bridge_render()
    if ready:
        feats = out.get("features", {}) or {}
        weights, rules = bridge_weights_and_rules(feats, answers)
        merged_profile = bridge_merge_into_profile(prof, rules, weights)

        st.success("Chat complete ✅ Routine updated with your preferences.")
        with st.expander("Bridge details"):
            st.write("Answers:", answers)
            st.write("Weights (w_k):", weights)
            st.write("Rules:", rules)

        if rec is not None:
            try:
                updated_plan = rec.recommend(feats, merged_profile, tier=tier, include_device=include_device)
            except Exception as e:
                updated_plan = {"error": f"rec_engine_error: {e}"}

            st.markdown(f"### Updated routine — {rules['plan_length']}")
            if isinstance(updated_plan, dict) and "error" in updated_plan:
                st.error(updated_plan["error"])
            else:
                for it in updated_plan.get("items", []):
                    name = f"{(it.get('brand') or '').title()} {it.get('name','')}".strip()
                    st.write(f"- **{name}**")
        else:
            st.warning("RecEngine not loaded, cannot refresh routine.")

# =========================
# MAIN UI
# =========================
st.title("SkinAizer: From a Selfie to your daily SkinCare")

REC = _load_rec_engine()
if REC is None:
    st.warning("RecEngine not loaded. Make sure DATA/products_kb.csv exists (or update the path).")

with st.sidebar:
    st.header("Settings")
    tier = st.selectbox("Plan tier", ["Starter", "Core", "Intense"], index=1)
    include_device = st.checkbox("Include device", value=True)

    st.header("Profile flags")
    flag_sensitive = st.checkbox("Sensitive", value=False, help="Prefers fragrance-free options")
    flag_acne = st.checkbox("Acne-prone", value=False, help="Avoids high comedogenicity")
    flag_preg = st.checkbox("Pregnant (avoid retinoids)", value=False)

    st.header("Detector")
    max_dim  = st.slider("Max image dimension (px)", 600, 1600, 900, step=50)
    min_side = st.slider("Min face side (px)", 80, 240, 120, step=10)

    st.header("Actions")
    if st.button("New Analysis / Reset", use_container_width=True):
        st.session_state["pipeline_out"] = None
        st.session_state["img_source"] = None
        st.session_state["img_label"] = ""
        reset_chat_only()
        st.rerun()

# If we already have results, show them (persistence)
if st.session_state["pipeline_out"] is not None:
    _render_results(
        st.session_state["pipeline_out"],
        st.session_state["img_source"],
        st.session_state["img_label"] or "Last Analysis",
        REC,
        tier,
        include_device
    )
else:
    # Otherwise show upload/dataset UI
    DATA_DIRS = ["DATA/raw", "DATA/interim", "data/raw", "data/interim/processed", "data/interim/raw_sample"]
    DATA_DIRS = [p for p in DATA_DIRS if os.path.isdir(p)]

    tab1, tab2 = st.tabs(["Upload", "Pick from dataset"])

    with tab1:
        uploaded = st.file_uploader("Upload your selfie (JPG/PNG)", type=["jpg", "jpeg", "png"])
        go_upload = st.button("Analyze uploaded", type="primary", use_container_width=True)

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
                st.session_state["pipeline_out"] = out
                st.session_state["img_source"] = uploaded
                st.session_state["img_label"] = "Uploaded"
                st.rerun()

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
                        st.session_state["pipeline_out"] = out
                        st.session_state["img_source"] = picked_img  # path ok
                        st.session_state["img_label"] = "Dataset"
                        st.rerun()
