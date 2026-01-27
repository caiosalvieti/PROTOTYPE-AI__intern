import os, json, tempfile, importlib, time
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
# Custom CSS (Mobile App / Action Dock Style)
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
        
        /* --- METRICS --- */
        div[data-testid="stMetric"] {
            background-color: white;
            border: 1px solid #eee;
            border-radius: 12px;
            padding: 10px;
            text-align: center;
        }
        div[data-testid="stMetricValue"] {
            color: #E5007D;
            font-size: 1.4rem;
        }
        </style>
    """, unsafe_allow_html=True)

load_custom_css()

# Hot-reloadable project modules
M     = importlib.import_module("main")
SC    = importlib.import_module("scores")
REMOD = importlib.import_module("rec_engine")

# -----------------------------
# Session & Helpers
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

# FOREO Avatar
FOREO_AVATAR = "https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/Foreo_logo.svg/1024px-Foreo_logo.svg.png"

# -----------------------------
# 2.0 The "App-Like" Bridge
# -----------------------------
BRIDGE_FUNNEL = [
    ("goal", "What is your main goal today?", ["✨ Gloss Control", "🔴 Redness Relief", "💧 Deep Hydration", "🎯 Spot Fading"]),
    ("sensitive", "Does your skin react easily?", ["Yes, it's sensitive", "No, it's resilient"]),
    ("sun", "Daily sun exposure?", ["☁️ Low / Indoor", "☀️ High / Outdoor"]),
    ("plan", "Routine commitment?", ["⚡ 3 Weeks (Fast)", "📅 1 Month (Steady)"]),
]

def bridge_render() -> Tuple[bool, Dict[str, str]]:
    
    # 1. Fixed Height Chat Window (Scrollable)
    # This keeps the UI stable!
    with st.container(height=400, border=True):
        # Greeting if empty
        if not st.session_state["bridge_messages"]:
            with st.chat_message("assistant", avatar=FOREO_AVATAR):
                st.write("Hello! I've analyzed your skin. Let's fine-tune your routine with 4 quick questions.")

        # History
        for m in st.session_state["bridge_messages"]:
            role = m["role"]
            if role == "assistant":
                with st.chat_message("assistant", avatar=FOREO_AVATAR):
                    st.write(m["content"])
            else:
                with st.chat_message("user"):
                    st.write(m["content"])

        # Current Question (Preview inside chat)
        step = int(st.session_state["bridge_step"])
        if step < len(BRIDGE_FUNNEL):
            key, q, options = BRIDGE_FUNNEL[step]
            
            # Logic to avoid showing duplicate questions
            last_msg = st.session_state["bridge_messages"][-1]["content"] if st.session_state["bridge_messages"] else ""
            if last_msg != q:
                with st.chat_message("assistant", avatar=FOREO_AVATAR):
                    st.write(q)

    # 2. The "Action Dock" (Fixed at bottom of chat area)
    if step < len(BRIDGE_FUNNEL):
        key, q, options = BRIDGE_FUNNEL[step]
        st.write("---") # Visual separator
        st.caption("Select an option:")
        
        # Grid layout for buttons (2 per row looks like a mobile menu)
        cols = st.columns(2)
        for i, opt in enumerate(options):
            # Modulo logic to place buttons in the 2 columns
            col = cols[i % 2]
            if col.button(opt, key=f"btn_{step}_{i}", use_container_width=True):
                # Update State
                st.session_state["bridge_messages"].append({"role": "assistant", "content": q})
                st.session_state["bridge_messages"].append({"role": "user", "content": opt})
                st.session_state["bridge_answers"][key] = opt
                st.session_state["bridge_step"] += 1
                st.rerun()

        return False, dict(st.session_state["bridge_answers"])

    # If done
    st.session_state["bridge_done"] = True
    return True, dict(st.session_state["bridge_answers"])

# -----------------------------
# Logic Connectors
# -----------------------------
def bridge_weights_and_rules(feats, answers):
    oil = feats.get("global_shn", 0)
    red = feats.get("global_red", 0)
    weights = {"sebum_control": oil, "soothing": red, "spf": 0.3}

    # Map fancy labels back to logic keys
    g = answers.get("goal", "")
    if "Gloss" in g: weights["sebum_control"] += 0.8
    if "Redness" in g: weights["soothing"] += 0.8
    if "Hydration" in g: weights["soothing"] += 0.3
    
    if "High" in answers.get("sun", ""): weights["spf"] += 0.8

    rules = {
        "sensitive": "sensitive" in answers.get("sensitive", "").lower(),
        "plan_length": "1_month" if "1 Month" in answers.get("plan", "") else "3_weeks"
    }
    return weights, rules

def merge_prof(prof, rules):
    p = dict(prof)
    if rules["sensitive"]:
        p.setdefault("scores", {})["sensitivity"] = 0.9
        p.setdefault("flags", []).append("sensitive")
    return p

# -----------------------------
# Caching
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
            try: return REMOD.RecEngine(p)
            except: pass
    return None

@st.cache_resource
def _load_skinaizer_core_model():
    return skinaizer_model_core

# -----------------------------
# Helpers
# -----------------------------
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

# -----------------------------
# Pipeline
# -----------------------------
def _run_pipeline(image_path, tier="Core", include_device=True, max_dim=900, min_side=120, rec=None):
    import time
    t0 = time.perf_counter()
    
    # 1. Load
    rgb = M.imread_rgb(image_path)
    rgb = M.gray_world(rgb) # Color correction

    # 2. Detect
    core_model = _load_skinaizer_core_model()
    core_out = core_model(rgb)
    
    # Fallback logic simplified for demo
    box = core_out.get("bbox")
    if box is None:
        box = M.detect_face_with_fallback(rgb, max_dim=max_dim, min_side=min_side)

    if box is None: return {"error": "No face detected"}

    # 3. Analyze
    x, y, w, h = [int(v) for v in box] # Ensure ints
    face = rgb[y:y+h, x:x+w]
    feats, zones = M.extract_features(face)
    profile = SC.infer_skin_profile(feats)

    # 4. Recommend (Initial Pass)
    plan = None
    if rec:
        plan = rec.recommend(feats, profile, tier=tier, include_device=include_device)

    # 5. Debug Image
    dbg_bytes = None
    try:
        with tempfile.TemporaryDirectory() as td:
            dbg_path = os.path.join(td, "debug.jpg")
            M.save_debug_panel(rgb, box, zones, dbg_path)
            with open(dbg_path, "rb") as f: dbg_bytes = f.read()
    except: pass

    return {
        "box": box,
        "features": feats,
        "profile": profile,
        "plan": plan,
        "debug_bytes": dbg_bytes,
        "rgb": rgb,
        "qa": {"fail": False} # Simplified QA
    }

# -----------------------------
# Main Rendering
# -----------------------------
st.title("SkinAizer x FOREO")

REC = _load_rec_engine()

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    if st.button("New Analysis"):
        reset_all()
        st.rerun()

# --- Main Logic ---
if st.session_state["pipeline_out"]:
    out = st.session_state["pipeline_out"]
    
    # 1. Results Header
    c1, c2 = st.columns([1, 1.5])
    with c1:
        if st.session_state["img_source"]:
             st.image(st.session_state["img_source"], caption="Input", use_container_width=True)
    with c2:
        st.subheader("Skin Analysis")
        scores = out["profile"]["scores"]
        
        # Metric Cards
        m1, m2 = st.columns(2)
        m3, m4 = st.columns(2)
        m1.metric("Oiliness", f"{scores.get('oiliness',0):.2f}")
        m2.metric("Redness", f"{scores.get('redness',0):.2f}")
        m3.metric("Texture", f"{scores.get('texture',0):.2f}")
        m4.metric("Hydration", f"{scores.get('hydration',0):.2f}")

    st.divider()

    # 2. APP-LIKE BRIDGE CHAT
    st.subheader("💬 Skin Coach")
    ready, answers = bridge_render()

    # 3. Final Output (Only shows when chat is done)
    if ready:
        # Re-run recommendation with new rules
        weights, rules = bridge_weights_and_rules(out["features"], answers)
        updated_prof = merge_prof(out["profile"], rules)
        
        final_plan = REC.recommend(out["features"], updated_prof, tier="Core", include_device=True) if REC else {}
        
        st.success("✨ Routine Tailored!")
        
        # Tabs for better organization
        tabs = st.tabs(["📅 Routine", "🛍️ List", "🤖 AI Reasoning"])
        
        with tabs[0]:
            st.markdown(f"**Duration:** {rules['plan_length'].replace('_', ' ').title()}")
            for item in final_plan.get("items", []):
                st.info(f"**{item.get('name')}**\n\n_{item.get('usage')}_")

        with tabs[1]:
             st.dataframe([{"Product": i["name"], "Price": f"${i.get('price_usd',0)}"} for i in final_plan.get("items", [])])
             
        with tabs[2]:
            st.markdown("### Why these products?")
            for item in final_plan.get("items", []):
                name = item.get('name')
                reason = item.get('reason', 'Matches your skin profile.')
                st.markdown(f"**{name}**")
                st.caption(f"Reason: {reason}")

else:
    # --- Upload Screen ---
    st.markdown("### Upload a selfie to begin")
    uploaded = st.file_uploader("", type=["jpg", "png"])
    if uploaded and st.button("Analyze Skin", type="primary"):
        with st.spinner("Processing..."):
            with tempfile.TemporaryDirectory() as td:
                path = _save_uploaded(td, uploaded)
                res = _run_pipeline(path, rec=REC)
                
            st.session_state["pipeline_out"] = res
            st.session_state["img_source"] = uploaded
            st.rerun()