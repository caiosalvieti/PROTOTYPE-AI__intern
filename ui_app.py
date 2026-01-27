import os, json, tempfile, importlib, time
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import streamlit as st
import cv2

# YOLO core (cached)
from core.yolo_roi import skinaizer_model_core

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="SkinAizer", page_icon="🧴", layout="wide")

# -----------------------------
# 1. VISUALS: Mobile App CSS
# -----------------------------
def load_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Poppins', sans-serif;
        }

        /* --- CHAT CONTAINER --- */
        /* Targets the scrollable container for the chat */
        [data-testid="stVerticalBlockBorderWrapper"] > div > [data-testid="stVerticalBlock"] {
            gap: 0.5rem;
        }

        /* --- BUBBLES --- */
        /* AI (Left) */
        div[data-testid="stChatMessage"]:nth-child(odd) {
            background-color: #F4F4F8;
            border-radius: 18px 18px 18px 4px;
            padding: 5px 10px;
            margin-right: 15%;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
            border: none;
        }
        /* User (Right) */
        div[data-testid="stChatMessage"]:nth-child(even) {
            background-color: #E5007D;
            color: white;
            border-radius: 18px 18px 4px 18px;
            padding: 5px 10px;
            margin-left: 15%;
            box-shadow: 0 4px 10px rgba(229, 0, 125, 0.2);
            border: none;
        }
        div[data-testid="stChatMessage"]:nth-child(even) p {
            color: white !important;
        }

        /* --- ACTION DOCK (Buttons) --- */
        /* Make the button container look like a mobile menu */
        div.stButton > button {
            width: 100%;
            border-radius: 12px;
            border: 1px solid #E5007D;
            background-color: white;
            color: #E5007D;
            font-weight: 600;
            padding: 12px 10px;
            transition: all 0.2s;
            height: auto;
            white-space: normal;
        }
        div.stButton > button:hover {
            background-color: #E5007D;
            color: white;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        /* --- METRIC CARDS --- */
        div[data-testid="stMetric"] {
            background-color: white;
            border: 1px solid #eee;
            border-radius: 12px;
            padding: 10px;
            text-align: center;
            box-shadow: 0 2px 5px rgba(0,0,0,0.02);
        }
        div[data-testid="stMetricValue"] {
            color: #E5007D;
            font-size: 1.4rem;
        }
        
        /* --- PRODUCT CARDS (HTML) --- */
        .product-card {
            background: white;
            border: 1px solid #eee;
            border-radius: 12px;
            padding: 15px;
            margin-bottom: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        .product-name { font-weight: bold; color: #333; font-size: 1.1rem; }
        .product-meta { color: #888; font-size: 0.9rem; margin-bottom: 5px; }
        .product-reason { background: #FFF0F7; color: #E5007D; padding: 5px 10px; border-radius: 8px; font-size: 0.85rem; display: inline-block; margin-top: 5px;}
        </style>
    """, unsafe_allow_html=True)

load_custom_css()

# --- 2. BACKEND SETUP ---
# Hot-reloadable project modules
M     = importlib.import_module("main")
SC    = importlib.import_module("scores")
REMOD = importlib.import_module("rec_engine")

# Helper for Foreo Avatar
FOREO_AVATAR = "https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/Foreo_logo.svg/1024px-Foreo_logo.svg.png"

# -----------------------------
# 3. LOGIC: State & Helpers
# -----------------------------
def _ss_init():
    st.session_state.setdefault("pipeline_out", None)
    st.session_state.setdefault("img_source", None)
    st.session_state.setdefault("img_source_label", "")
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
# 4. LOGIC: Chat & Bridge
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
    """Renders the chat in a fixed container with action dock."""
    
    # 1. Scrollable History Area
    with st.container(height=350, border=True):
        if not st.session_state["bridge_messages"]:
             with st.chat_message("assistant", avatar=FOREO_AVATAR):
                st.write("I've analyzed your skin profile. Let's customize your routine with a few quick questions.")

        for m in st.session_state["bridge_messages"]:
            role = m["role"]
            if role == "assistant":
                with st.chat_message("assistant", avatar=FOREO_AVATAR):
                    st.write(m["content"])
            else:
                with st.chat_message("user"):
                    st.write(m["content"])
        
        # Show current question inside chat too (for flow)
        step = int(st.session_state["bridge_step"])
        if step < len(BRIDGE_FUNNEL):
            key, q, options = BRIDGE_FUNNEL[step]
            # Check last message to avoid dupes
            last_msg = st.session_state["bridge_messages"][-1]["content"] if st.session_state["bridge_messages"] else ""
            if last_msg != q:
                with st.chat_message("assistant", avatar=FOREO_AVATAR):
                    st.write(q)

    # 2. Action Dock (Bottom Buttons)
    step = int(st.session_state["bridge_step"])
    if step < len(BRIDGE_FUNNEL):
        key, q, options = BRIDGE_FUNNEL[step]
        st.write("---")
        st.caption("Select an option:")
        
        cols = st.columns(2)
        for i, opt in enumerate(options):
            col = cols[i % 2]
            if col.button(opt, key=f"bridge_{step}_{i}", use_container_width=True):
                st.session_state["bridge_messages"].append({"role": "assistant", "content": q})
                st.session_state["bridge_messages"].append({"role": "user", "content": opt})
                st.session_state["bridge_answers"][key] = opt
                st.session_state["bridge_step"] += 1
                st.rerun()
        
        return False, dict(st.session_state["bridge_answers"])

    # Done
    st.session_state["bridge_done"] = True
    return True, dict(st.session_state["bridge_answers"])

# --- Bridge Logic ---
def bridge_weights_and_rules(feats: Dict[str, Any], answers: Dict[str, str]) -> Tuple[Dict[str, float], Dict[str, Any]]:
    oiliness = float(feats.get("global_shn", 0.0))
    redness  = float(feats.get("global_red", 0.0))
    texture  = float(feats.get("global_txt", 0.0))

    weights = {"sebum_control": oiliness, "soothing": redness, "texture": texture, "spf": 0.3}

    goal = answers.get("goal")
    if goal == "Gloss control": weights["sebum_control"] += 0.7
    elif goal == "Redness relief": weights["soothing"] += 0.7
    elif goal == "Deep hydration": weights["soothing"] += 0.2
    elif goal == "Blemish control": weights["sebum_control"] += 0.4
    elif goal == "Spot fading": weights["texture"] += 0.3

    sun = answers.get("sun")
    if sun == "High": weights["spf"] += 0.7
    elif sun == "Medium": weights["spf"] += 0.4

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
    prof = dict(profile or {})
    prof.setdefault("scores", {})
    prof.setdefault("flags", [])

    if rules.get("sensitive"):
        prof["scores"]["sensitivity"] = max(float(prof["scores"].get("sensitivity", 0.0)), 0.8)
        if "sensitive" not in prof["flags"]:
            prof["flags"].append("sensitive")

    prof["bridge"] = {"rules": rules}
    return prof

# -----------------------------
# 5. LOGIC: Pipeline & Caching
# -----------------------------
@st.cache_resource
def _load_rec_engine():
    rec = getattr(M, "REC_ENGINE", None)
    if rec is not None and isinstance(rec, REMOD.RecEngine): return rec
    for p in ["DATA/products_kb.csv", "products_kb.csv"]:
        if os.path.isfile(p):
            try: return REMOD.RecEngine(p)
            except: pass
    return None

@st.cache_resource
def _load_skinaizer_core_model():
    return skinaizer_model_core

def _save_uploaded(tmp_dir: str, uf) -> str:
    ext = os.path.splitext(uf.name)[1].lower() or ".jpg"
    out = os.path.join(tmp_dir, f"upload{ext}")
    with open(out, "wb") as f: f.write(uf.getbuffer())
    return out

def _draw_overlay(rgb: np.ndarray, box) -> np.ndarray:
    import cv2
    x, y, w, h = box
    dbg = rgb.copy()
    cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return dbg

def _run_pipeline(image_path, tier="Core", include_device=True, max_dim=900, min_side=120, rec=None, flags=None):
    import time
    t_total0 = time.perf_counter()
    rgb = M.imread_rgb(image_path)
    H_orig, W_orig = rgb.shape[:2]
    rgb = M.gray_world(rgb)

    core_model_func = _load_skinaizer_core_model()
    core_out = core_model_func(rgb)
    yolo_bbox = core_out.get("bbox")
    timings = (core_out.get("timings") or {}).copy()

    if yolo_bbox is not None:
        x_min_proc, y_min_proc, x_max_proc, y_max_proc = yolo_bbox
        # Simplified scaling logic for demo
        scale_w = W_orig / 640
        scale_h = H_orig / 640
        x_min = int(x_min_proc * scale_w)
        y_min = int(y_min_proc * scale_h)
        x_max = int(x_max_proc * scale_w)
        y_max = int(y_max_proc * scale_h)
        box = (x_min, y_min, x_max - x_min, y_max - y_min)
    else:
        box = M.detect_face_with_fallback(rgb, max_dim=max_dim, min_side=min_side)

    if box is None: return {"error": "no_face_detected"}

    x, y, w, h = box
    face = rgb[y:y + h, x:x + w]
    feats, zones = M.extract_features(face)
    profile = SC.infer_skin_profile(feats)

    # Flags
    flags = flags or {}
    if flags.get("sensitive"):
        profile.setdefault("scores", {})["sensitivity"] = 0.8
        profile.setdefault("flags", []).append("sensitive")

    plan = None
    if rec is not None:
        try: plan = rec.recommend(feats, profile, tier=tier, include_device=include_device)
        except Exception as e: plan = {"error": f"rec_engine_error: {e}"}

    dbg_bytes = None
    try:
        with tempfile.TemporaryDirectory() as td_dbg:
            dbg_path = os.path.join(td_dbg, "debug.jpg")
            M.save_debug_panel(rgb, box, zones, dbg_path)
            with open(dbg_path, "rb") as f: dbg_bytes = f.read()
    except: pass

    qa = {"fail": bool(feats.get("qa_fail", 0)), "issues": feats.get("qa_issues", "")}
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
# 6. LOGIC: Schedule Builder
# -----------------------------
def _pick_item(items: List[Dict[str, Any]], form: str) -> Optional[Dict[str, Any]]:
    for it in items:
        if (it.get("form") or "").lower() == form.lower(): return it
    return None

def _fmt_item(it: Optional[Dict[str, Any]]) -> str:
    if not it: return "—"
    name = f"{(it.get('brand') or '').title()} {it.get('name','')}".strip()
    usage = (it.get("usage") or "").strip()
    return f"**{name}**" + (f"  \n_{usage}_" if usage else "")

def build_schedule(plan: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]:
    items = plan.get("items") or []
    spf   = _pick_item(items, "spf")
    mois  = _pick_item(items, "moisturizer")
    serum = _pick_item(items, "serum")
    exfol = _pick_item(items, "exfoliant")
    dev   = _pick_item(items, "device")

    sensitive = bool(rules.get("sensitive"))
    plan_len = rules.get("plan_length", "3_weeks")
    
    exfol_freq = "1×/week (night)" if sensitive else "2–3×/week (night)"

    def week_block(title, am_steps, pm_steps, notes):
        return {"title": title, "am": " → ".join(am_steps), "pm": " → ".join(pm_steps), "notes": " • ".join([n for n in notes if n])}

    am_standard = ["Cleanse", _fmt_item(serum), _fmt_item(mois), _fmt_item(spf)]
    pm_basic = ["Cleanse", _fmt_item(mois)]
    pm_exfol = ["Cleanse", f"{_fmt_item(exfol)}  \n_{exfol_freq}_", _fmt_item(mois)] if exfol else pm_basic

    w1 = week_block("Week 1 — Balance", am_standard, pm_basic, ["Baseline first.", f"Device: {_fmt_item(dev) if dev else 'optional'}"])
    w2 = week_block("Week 2 — Treatment", am_standard, pm_exfol, ["Introduce actives."])
    
    pm_w3 = pm_basic if sensitive else pm_exfol
    w3 = week_block("Week 3 — Protect", am_standard, pm_w3, ["Stabilize."])

    if plan_len == "3_weeks": return {"3-week plan": [w1, w2, w3]}

    # 1 Month Logic
    pm_w2_slow = ["Cleanse", f"{_fmt_item(exfol)}  \n_1×/week_", _fmt_item(mois)] if exfol else pm_basic
    w2_slow = week_block("Week 2 — Intro", am_standard, pm_w2_slow, ["Start low frequency."])
    
    freq_w3 = '1–2×/week' if sensitive else '2×/week'
    pm_w3_slow = ["Cleanse", f"{_fmt_item(exfol)}  \n_{freq_w3}_", _fmt_item(mois)] if exfol else pm_basic
    w3_slow = week_block("Week 3 — Build", am_standard, pm_w3_slow, ["Increase step."])
    
    pm_w4 = pm_basic if sensitive else pm_exfol
    w4 = week_block("Week 4 — Maintain", am_standard, pm_w4, ["Consolidate."])

    return {"1-month plan": [w1, w2_slow, w3_slow, w4]}

# -----------------------------
# 7. UI: Main Render
# -----------------------------
def _render_results(src_label: str, img_source, out: Dict[str, Any], rec, tier: str, include_device: bool):
    # Header & Metrics
    c1, c2 = st.columns([1, 1.5])
    with c1:
        if isinstance(img_source, str):
            st.image(img_source, caption=os.path.basename(img_source), width=300)
        else:
            st.image(img_source, caption="Uploaded", width=300)
    with c2:
        st.subheader("Skin Profile")
        scores = out.get("profile", {}).get("scores", {})
        m1, m2 = st.columns(2)
        m3, m4 = st.columns(2)
        m1.metric("Oiliness", f"{scores.get('oiliness', 0):.2f}")
        m2.metric("Redness", f"{scores.get('redness', 0):.2f}")
        m3.metric("Texture", f"{scores.get('texture', 0):.2f}")
        m4.metric("Hydration", f"{scores.get('hydration', 0):.2f}")
        
        # Debug toggle
        with st.expander("Debug View"):
            if out.get("debug_bytes"):
                st.image(out["debug_bytes"], caption="Debug panel", use_container_width=True)

    st.divider()

    # Chat Bridge
    st.subheader("💬 Skin Coach")
    ready, answers = bridge_render()

    if not ready:
        return

    # FINAL RESULTS
    weights, rules = bridge_weights_and_rules(out["features"], answers)
    updated_profile = merge_rules_into_profile(out["profile"], rules)
    
    updated_plan = out["plan"]
    if rec is not None:
        try: updated_plan = rec.recommend(out["features"], updated_profile, tier=tier, include_device=include_device)
        except Exception as e: st.warning(f"Rec Engine Error: {e}")

    # Tabs for Plan
    t1, t2, t3 = st.tabs(["📅 Routine Schedule", "🛍️ Shopping List", "🤖 AI Reasoning"])
    
    with t1:
        schedule = build_schedule(updated_plan, rules)
        for block_name, weeks in schedule.items():
            st.markdown(f"#### {block_name}")
            for w in weeks:
                with st.expander(w["title"]):
                    st.markdown(f"**AM:** {w['am']}")
                    st.markdown(f"**PM:** {w['pm']}")
                    st.caption(w["notes"])

    with t2:
        for item in updated_plan.get("items", []):
            # Render HTML Card for items
            price = item.get('price_usd', 0)
            name = item.get('name')
            form = item.get('form', 'Product').upper()
            html = f"""
            <div class="product-card">
                <div style="display:flex; justify-content:space-between;">
                    <div class="product-name">{name}</div>
                    <div style="font-weight:bold;">${price}</div>
                </div>
                <div class="product-meta">{form}</div>
            </div>
            """
            st.markdown(html, unsafe_allow_html=True)

    with t3:
        for item in updated_plan.get("items", []):
            st.markdown(f"**{item.get('name')}**")
            st.caption(f"Reason: {item.get('reason', 'Matches skin profile')}")

# -----------------------------
# 8. MAIN ENTRY
# -----------------------------
st.title("SkinAizer x FOREO")

REC = _load_rec_engine()
if REC is None: st.warning("RecEngine not loaded. Check `DATA/products_kb.csv`.")

with st.sidebar:
    st.header("Settings")
    tier = st.selectbox("Plan Tier", ["Starter", "Core", "Intense"], index=1)
    include_device = st.checkbox("Include Device", value=True)
    
    st.markdown("---")
    if st.button("New Analysis"):
        reset_all()
        st.rerun()

# Router
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

# Upload View
DATA_DIRS = ["DATA/raw", "data/raw"]
DATA_DIRS = [p for p in DATA_DIRS if os.path.isdir(p)]

tab1, tab2 = st.tabs(["Upload", "Pick from Dataset"])

with tab1:
    uploaded = st.file_uploader("Upload Selfie", type=["jpg", "png", "jpeg"])
    if uploaded and st.button("Analyze Upload", type="primary"):
        with st.spinner("Processing..."):
            with tempfile.TemporaryDirectory() as td:
                img_path = _save_uploaded(td, uploaded)
                out = _run_pipeline(img_path, tier=tier, include_device=include_device, rec=REC)
        
        if out.get("error"): st.error(out["error"])
        else:
            st.session_state["pipeline_out"] = out
            st.session_state["img_source"] = uploaded
            st.session_state["img_source_label"] = "Uploaded"
            st.rerun()

with tab2:
    if not DATA_DIRS: st.info("No dataset folders found.")
    else:
        root = st.selectbox("Dataset Folder", DATA_DIRS)
        imgs = _list_images(root)
        if not imgs: st.info("Folder empty.")
        else:
            picked = st.selectbox("Select Image", imgs)
            if st.button("Analyze Selected", type="primary"):
                with st.spinner("Processing..."):
                    out = _run_pipeline(picked, tier=tier, include_device=include_device, rec=REC)
                
                if out.get("error"): st.error(out["error"])
                else:
                    st.session_state["pipeline_out"] = out
                    st.session_state["img_source"] = picked
                    st.session_state["img_source_label"] = "Dataset"
                    st.rerun()