import os, json, tempfile, importlib
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import streamlit as st

# YOLO core (cached)
from core.yolo_roi import skinaizer_model_core

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="SkinAizer", page_icon="🧴", layout="wide")

# -----------------------------
# Custom CSS (Modern "Foreo" Style + Chat Polish)
# -----------------------------
def load_custom_css():
    st.markdown("""
        <style>
        /* Import a modern font (Poppins is great for tech/beauty) */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Poppins', sans-serif;
        }

        /* --- METRIC CARDS --- */
        div[data-testid="stMetric"] {
            background-color: #ffffff;
            border: 1px solid #f0f0f0;
            padding: 15px;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            transition: transform 0.2s;
            text-align: center;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.1);
        }
        label[data-testid="stMetricLabel"] {
            font-weight: 600;
            color: #888;
            font-size: 0.9rem;
        }
        div[data-testid="stMetricValue"] {
            color: #E5007D; /* Foreo Pink */
            font-weight: 600;
        }

        /* --- BUTTONS (Global) --- */
        div.stButton > button {
            border-radius: 25px;
            font-weight: 600;
            border: none;
            padding: 10px 24px;
            transition: all 0.3s ease;
        }
        div.stButton > button:hover {
            transform: scale(1.02);
            box-shadow: 0 4px 10px rgba(229, 0, 125, 0.4);
        }

        /* --- CHAT STYLING (The "Apple/Messenger" Look) --- */
        
        /* 1. Chat Bubbles */
        div[data-testid="stChatMessage"] {
            border: none;
            padding: 1rem;
            margin-bottom: 0.5rem;
            background-color: transparent; 
        }

        /* AI Message (Left, Grey) */
        div[data-testid="stChatMessage"]:nth-child(odd) {
            background-color: #F0F2F6; 
            border-radius: 20px 20px 20px 5px;
            margin-right: 15%; /* Keep it to the left */
        }

        /* User Message (Right, Pink) */
        div[data-testid="stChatMessage"]:nth-child(even) {
            background-color: #E5007D; 
            color: white;
            border-radius: 20px 20px 5px 20px;
            margin-left: 15%; /* Push it to the right */
        }
        
        /* Fix text color inside pink user bubbles */
        div[data-testid="stChatMessage"]:nth-child(even) p {
            color: white !important;
        }

        /* 2. Option Buttons ("Pills") inside Chat */
        div[data-testid="column"] button {
            border-radius: 50px !important;
            border: 1px solid #E5007D !important;
            background-color: white !important;
            color: #E5007D !important;
            font-size: 0.85rem !important;
            padding: 5px 15px !important;
            margin: 2px !important;
            box-shadow: none !important;
        }
        div[data-testid="column"] button:hover {
            background-color: #E5007D !important;
            color: white !important;
        }

        /* --- SIDEBAR --- */
        section[data-testid="stSidebar"] {
            background-color: #F8F9FA;
            border-right: 1px solid #EAEAEA;
        }

        /* --- IMAGES --- */
        img {
            border-radius: 12px;
        }
        </style>
    """, unsafe_allow_html=True)

# Load the styles immediately
load_custom_css()

# Hot-reloadable project modules
M     = importlib.import_module("main")
SC    = importlib.import_module("scores")
REMOD = importlib.import_module("rec_engine")

# -----------------------------
# Session state persistence
# -----------------------------
def _ss_init():
    st.session_state.setdefault("pipeline_out", None)     # stores last analysis output dict
    st.session_state.setdefault("img_source", None)       # stores UploadedFile or path (for display)
    st.session_state.setdefault("img_source_label", "")   # "Uploaded" / "Dataset"

    # Bridge chat state
    st.session_state.setdefault("bridge_step", 0)
    st.session_state.setdefault("bridge_messages", [])
    st.session_state.setdefault("bridge_answers", {})
    st.session_state.setdefault("bridge_done", False)

_ss_init()

def reset_all():
    st.session_state["pipeline_out"] = None
    st.session_state["img_source"] = None
    st.session_state["img_source_label"] = ""

    st.session_state["bridge_step"] = 0
    st.session_state["bridge_messages"] = []
    st.session_state["bridge_answers"] = {}
    st.session_state["bridge_done"] = False


# -----------------------------
# Bridge chat (diagnostic funnel)
# -----------------------------
BRIDGE_FUNNEL = [
    ("goal", "What do you most want to improve?",
     ["Gloss control", "Redness relief", "Deep hydration", "Blemish control", "Spot fading"]),
    ("sensitive", "Does your skin react easily to new products?",
     ["Yes", "No"]),
    ("avoid", "Any ingredients you want to avoid?",
     ["Avoid alcohol", "Avoid fragrance", "No restrictions"]),
    ("sun", "What is your daily sun exposure level?",
     ["Low", "Medium", "High"]),
    ("plan", "Routine length?",
     ["3 weeks", "1 month"]),
]

def bridge_render() -> Tuple[bool, Dict[str, str]]:
    """Renders the chat and returns (ready, answers)."""
    # FOREO Avatar URL (Official logo or icon)
    FOREO_AVATAR = "https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/Foreo_logo.svg/1024px-Foreo_logo.svg.png"
    
    # Show history
    for m in st.session_state["bridge_messages"]:
        role = m["role"]
        # Use avatar only for assistant to look professional
        if role == "assistant":
            with st.chat_message(role, avatar=FOREO_AVATAR):
                st.markdown(m["content"])
        else:
            with st.chat_message(role): # Default user icon
                st.markdown(m["content"])

    step = int(st.session_state["bridge_step"])
    if step >= len(BRIDGE_FUNNEL):
        st.session_state["bridge_done"] = True
        return True, dict(st.session_state["bridge_answers"])

    key, q, options = BRIDGE_FUNNEL[step]

    # Render current question
    with st.chat_message("assistant", avatar=FOREO_AVATAR):
        st.markdown(f"**{q}**")
        # Use columns to make buttons look like "Pills" / "Chips"
        # We wrap them in a container to apply our specific CSS
        cols = st.columns(len(options))
        for i, opt in enumerate(options):
            if cols[i].button(opt, key=f"bridge_{step}_{i}"):
                st.session_state["bridge_messages"].append({"role": "assistant", "content": q})
                st.session_state["bridge_messages"].append({"role": "user", "content": opt})
                st.session_state["bridge_answers"][key] = opt
                st.session_state["bridge_step"] += 1
                st.rerun()

    return False, dict(st.session_state["bridge_answers"])

def bridge_weights_and_rules(feats: Dict[str, Any], answers: Dict[str, str]) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Create weights + restriction rules from chat answers + image features."""
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
    weights = {k: float(v) / s for k, v in weights.items()}

    rules = {
        "exclude_alcohol": (answers.get("avoid") == "Avoid alcohol"),
        "exclude_fragrance": (answers.get("avoid") == "Avoid fragrance"),
        "sensitive": (answers.get("sensitive") == "Yes"),
        "plan_length": "3_weeks" if answers.get("plan") == "3 weeks" else "1_month",
        "goal": goal or "",
        "sun": sun or "",
    }
    return weights, rules

def merge_rules_into_profile(profile: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
    """Blend bridge rules into your existing profile so RecEngine can react."""
    prof = dict(profile or {})
    prof.setdefault("scores", {})
    prof.setdefault("flags", [])

    if rules.get("sensitive"):
        # force high sensitivity
        prof["scores"]["sensitivity"] = max(float(prof["scores"].get("sensitivity", 0.0)), 0.8)
        if "sensitive" not in prof["flags"]:
            prof["flags"].append("sensitive")

    # store context (optional for later use inside RecEngine)
    prof["bridge"] = {"rules": rules}
    return prof


# -----------------------------
# Caching heavy objects
# -----------------------------
@st.cache_resource
def _load_rec_engine():
    # Prefer shared instance from main.py
    rec = getattr(M, "REC_ENGINE", None)
    if rec is not None and isinstance(rec, REMOD.RecEngine):
        return rec

    # Otherwise try common KB locations
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


# -----------------------------
# Helpers
# -----------------------------
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


# -----------------------------
# Pipeline (your YOLO-rescale version)
# -----------------------------
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

    dbg_bytes = None
    t_total0 = time.perf_counter()

    # 1) load + preproc
    rgb = M.imread_rgb(image_path)
    H_orig, W_orig = rgb.shape[:2]
    rgb = M.gray_world(rgb)

    # 2) YOLO detection
    core_model_func = _load_skinaizer_core_model()
    core_out = core_model_func(rgb)
    yolo_bbox = core_out.get("bbox")
    timings = (core_out.get("timings") or {}).copy()

    # 3) bbox rescale or fallback
    if yolo_bbox is not None:
        x_min_proc, y_min_proc, x_max_proc, y_max_proc = yolo_bbox
        scale_w = W_orig / W_proc
        scale_h = H_orig / H_proc
        x_min = int(x_min_proc * scale_w)
        y_min = int(y_min_proc * scale_h)
        x_max = int(x_max_proc * scale_w)
        y_max = int(y_max_proc * scale_h)
        box = (x_min, y_min, x_max - x_min, y_max - y_min)
    else:
        box = M.detect_face_with_fallback(rgb, max_dim=max_dim, min_side=min_side)

    if box is None:
        return {"error": "no_face_detected"}

    x, y, w, h = box
    face = rgb[y:y + h, x:x + w]

    feats, zones = M.extract_features(face)
    profile = SC.infer_skin_profile(feats)

    # merge sidebar flags (optional)
    flags = flags or {}
    if flags.get("sensitive"):
        profile.setdefault("scores", {})
        profile["scores"]["sensitivity"] = max(profile["scores"].get("sensitivity", 0.0), 0.8)
        profile.setdefault("flags", []).append("sensitive")
    if flags.get("acne_prone"):
        profile["acne_prone"] = True
    if flags.get("pregnant"):
        profile["pregnant"] = True

    # rec engine
    plan = None
    if rec is not None:
        try:
            plan = rec.recommend(feats, profile, tier=tier, include_device=include_device)
        except Exception as e:
            plan = {"error": f"rec_engine_error: {e}"}

    # debug image bytes
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

    timings["total_pipeline_ms"] = (time.perf_counter() - t_total0) * 1000.0

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


# -----------------------------
# Routine schedule builder (3 weeks vs 1 month)
# -----------------------------
def _pick_item(items: List[Dict[str, Any]], form: str) -> Optional[Dict[str, Any]]:
    for it in items:
        if (it.get("form") or "").lower() == form.lower():
            return it
    return None

def _fmt_item(it: Optional[Dict[str, Any]]) -> str:
    if not it:
        return "—"
    name = f"{(it.get('brand') or '').title()} {it.get('name','')}".strip()
    usage = (it.get("usage") or "").strip()
    return f"**{name}**" + (f"  \n_{usage}_" if usage else "")

def build_schedule(plan: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]:
    """Return schedule grouped by week label."""
    items = plan.get("items") or []

    spf   = _pick_item(items, "spf")
    mois  = _pick_item(items, "moisturizer")
    serum = _pick_item(items, "serum")
    exfol = _pick_item(items, "exfoliant")
    dev   = _pick_item(items, "device")

    sensitive = bool(rules.get("sensitive"))
    plan_len = rules.get("plan_length", "3_weeks")

    # frequencies
    if sensitive:
        exfol_freq = "1×/week (night)"
    else:
        exfol_freq = "2–3×/week (night)"

    # Helper for week block
    def week_block(title: str, am_steps: List[str], pm_steps: List[str], notes: List[str]):
        return {
            "title": title,
            "am": " → ".join(am_steps),
            "pm": " → ".join(pm_steps),
            "notes": " • ".join([n for n in notes if n]),
        }

    # Define AM/PM basics
    am_standard = ["Cleanse", _fmt_item(serum), _fmt_item(mois), _fmt_item(spf)]
    pm_basic = ["Cleanse", _fmt_item(mois)]
    
    if exfol:
        pm_exfol = ["Cleanse", f"{_fmt_item(exfol)}  \n_{exfol_freq}_", _fmt_item(mois)]
    else:
        pm_exfol = pm_basic

    # --- Weeks ---
    w1 = week_block(
        "Week 1 — Balance & barrier",
        am_steps=am_standard,
        pm_steps=pm_basic,
        notes=[
            "Keep it boring. Photo-based baseline first.",
            f"Device: {_fmt_item(dev) if dev else 'optional'} (gentle, 2–3×/week)."
        ]
    )

    w2 = week_block(
        "Week 2 — Active treatment",
        am_steps=am_standard,
        pm_steps=pm_exfol, 
        notes=[
            "Introduce only 1 strong active at a time.",
            "If irritation: stop actives 72h, focus moisturizer + SPF."
        ]
    )

    if sensitive:
        pm_w3 = pm_basic
    else:
        pm_w3 = pm_exfol

    w3 = week_block(
        "Week 3 — Consolidation & protection",
        am_steps=am_standard,
        pm_steps=pm_w3,
        notes=["Stabilize. Small increases only if skin is calm."]
    )

    if plan_len == "3_weeks":
        return {"3-week plan": [w1, w2, w3]}

    # --- 1-Month Logic ---
    if exfol:
        pm_w2_slow = ["Cleanse", f"{_fmt_item(exfol)}  \n_1×/week (night)_", _fmt_item(mois)]
    else:
        pm_w2_slow = pm_basic

    w2_slow = week_block(
        "Week 2 — Gentle active intro",
        am_steps=am_standard,
        pm_steps=pm_w2_slow,
        notes=["Start low frequency. Track redness/itching."]
    )

    freq_w3 = '1–2×/week (night)' if sensitive else '2×/week (night)'
    if exfol:
        pm_w3_slow = ["Cleanse", f"{_fmt_item(exfol)}  \n_{freq_w3}_", _fmt_item(mois)]
    else:
        pm_w3_slow = pm_basic

    w3_slow = week_block(
        "Week 3 — Build tolerance",
        am_steps=am_standard,
        pm_steps=pm_w3_slow,
        notes=["If stable, increase one step."]
    )

    if sensitive:
        pm_w4 = pm_basic
    else:
        pm_w4 = pm_exfol

    w4 = week_block(
        "Week 4 — Consolidate",
        am_steps=am_standard,
        pm_steps=pm_w4,
        notes=["Maintain. Re-take photo at end of week 4."]
    )

    return {"1-month plan": [w1, w2_slow, w3_slow, w4]}


# -----------------------------
# Rendering
# -----------------------------
def _render_results(src_label: str, img_source, out: Dict[str, Any], rec, tier: str, include_device: bool):
    st.subheader(f"Results — {src_label}")

    # images
    col_img, col_overlay = st.columns(2)
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

    # QA
    qa = out.get("qa", {}) or {}
    if qa.get("fail"):
        st.warning(f"⚠️ Retake suggested — {qa.get('issues') or 'quality issue'}")
    else:
        st.success("✅ PASS - good photo quality")

    # timings
    timings = out.get("timings") or {}
    if timings:
        pretty = ", ".join(f"{k}={v:.1f} ms" for k, v in timings.items())
        st.caption(f"Timings: {pretty}")

    # profile metrics
    st.markdown("### Skin profile")
    prof = out.get("profile", {}) or {}
    scores = prof.get("scores", {}) or {}
    cols = st.columns(5)
    cols[0].metric("Skin type", prof.get("skin_type", "?"))
    cols[1].metric("Oiliness", f"{scores.get('oiliness', 0):.2f}")
    cols[2].metric("Hydration", f"{scores.get('hydration', 0):.2f}")
    cols[3].metric("Redness", f"{scores.get('redness', 0):.2f}")
    cols[4].metric("Texture", f"{scores.get('texture', 0):.2f}")

    with st.expander("Raw JSON (profile & features)"):
        c1, c2 = st.columns(2)
        with c1:
            st.json(prof)
        with c2:
            st.json(out.get("features", {}))

    # base plan
    plan = out.get("plan")
    if not plan:
        st.info("No plan (RecEngine not loaded).")
        return
    if isinstance(plan, dict) and "error" in plan:
        st.error(plan["error"])
        return

    st.markdown(f"### Suggested routine — {plan.get('plan', tier)}")
    for it in (plan.get("items") or []):
        name = f"{(it.get('brand') or '').title()} {it.get('name','')}".strip()
        st.markdown(f"- **{name}** — *{it.get('form','')}*")

    # Bridge chat under results
    st.divider()
    st.subheader("Bridge: personalize your routine (4–5 quick questions)")
    ready, answers = bridge_render()

    if not ready:
        st.caption("Answer the quick prompts above to tailor the plan.")
        return

    # apply bridge context
    weights, rules = bridge_weights_and_rules(out["features"], answers)
    updated_profile = merge_rules_into_profile(prof, rules)

    st.success("✅ Chat complete — applying your context to the routine.")

    # recompute plan (best-effort)
    updated_plan = plan
    if rec is not None:
        try:
            updated_plan = rec.recommend(out["features"], updated_profile, tier=tier, include_device=include_device)
        except Exception as e:
            st.warning(f"Could not recompute plan with bridge context: {e}")

    # show rules/weights
    with st.expander("What changed (rules & weights)"):
        st.json({"weights": weights, "rules": rules})

    # schedule output
    if not updated_plan or (isinstance(updated_plan, dict) and "error" in updated_plan):
        st.info("No updated plan available to build a schedule.")
        return

    schedule = build_schedule(updated_plan, rules)
    for block_name, weeks in schedule.items():
        st.markdown(f"## {block_name}")
        for w in weeks:
            with st.expander(w["title"], expanded=True):
                st.markdown("**AM**")
                st.markdown(w["am"])
                st.markdown("**PM**")
                st.markdown(w["pm"])
                if w.get("notes"):
                    st.caption(w["notes"])


# -----------------------------
# Main UI
# -----------------------------
st.title("SkinAizer: From a Selfie to your daily SkinCare")

REC = _load_rec_engine()
if REC is None:
    st.warning("RecEngine not loaded. Make sure `DATA/products_kb.csv` exists (or update the path).")

with st.sidebar:
    st.header("Plan options")
    tier = st.selectbox("Plan tier", ["Starter", "Core", "Intense"], index=1)
    include_device = st.checkbox("Include device", value=True)

    st.header("Optional quick flags (pre-chat)")
    flag_sensitive = st.checkbox("Sensitive", value=False)
    flag_acne      = st.checkbox("Acne-prone", value=False)
    flag_preg      = st.checkbox("Pregnant (avoid retinoids)", value=False)

    st.header("Detector")
    max_dim  = st.slider("Max image dimension (px)", 600, 1600, 900, step=50)
    min_side = st.slider("Min face side (px)", 80, 240, 120, step=10)

    st.header("Session")
    if st.button("New analysis / reset"):
        reset_all()
        st.rerun()

    st.header("Dev")
    if st.button("Reload modules"):
        try:
            importlib.reload(M)
            importlib.reload(SC)
            importlib.reload(REMOD)
            st.success("Modules reloaded.")
        except Exception as e:
            st.error(f"Reload failed: {e}")

# If we already analyzed something, show it (persistent)
if st.session_state["pipeline_out"] is not None:
    _render_results(
        st.session_state["img_source_label"],
        st.session_state["img_source"],
        st.session_state["pipeline_out"],
        REC,
        tier,
        include_device,
    )
    st.stop()

# Otherwise show input tabs
DATA_DIRS = [
    "DATA/raw",
    "DATA/interim",
    "data/raw",
    "data/interim/processed",
    "data/interim/raw_sample",
]
DATA_DIRS = [p for p in DATA_DIRS if os.path.isdir(p)]

tab1, tab2 = st.tabs(["Upload", "Pick from dataset"])

with tab1:
    left, right = st.columns([3, 1])
    with left:
        uploaded = st.file_uploader("Upload your selfie (JPG/PNG)", type=["jpg", "jpeg", "png"])
    with right:
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
            # persist results so chat survives reruns
            st.session_state["pipeline_out"] = out
            st.session_state["img_source"] = uploaded
            st.session_state["img_source_label"] = "Uploaded"
            st.rerun()

with tab2:
    if not DATA_DIRS:
        st.info("No dataset folders found. Create `DATA/raw` and place images there.")
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
                    st.session_state["img_source"] = picked_img
                    st.session_state["img_source_label"] = "Dataset"
                    st.rerun()