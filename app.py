"""
ChurnGuard AI – Premium Fintech SaaS Dashboard
================================================
Banking Intelligence Platform · Churn Prediction & Retention Engine
Run:  streamlit run app.py
"""

from __future__ import annotations
import os, sys, json, pickle, datetime, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from utils.data_utils       import preprocess_single_customer, get_feature_display_names
from utils.model_utils      import (models_exist, load_artifacts, load_training_results,
                                     predict_churn, get_feature_importances, generate_sample_customer)
from utils.retention_engine import get_risk_category, generate_retention_strategies, get_action_plan
from utils.auth             import (verify_credentials, register_user, get_demo_hint,
                                     list_users, delete_user, change_password)
from utils.report_generator import generate_pdf_report
from utils.segmentation     import (segmentation_exists, train_segmentation, load_segmentation,
                                     predict_segment, get_segment_summary, SEGMENT_META, SEG_FEATURES)

try:
    import anthropic as _anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════════════
#  CSS – Premium Fintech Dark Theme
# ═══════════════════════════════════════════════════════════════════════════════
def inject_css():
    st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;600;700;800;900&display=swap');

/* ── Design Tokens ── */
:root {
  --bg:      #080d1a;
  --bg2:     #0e1628;
  --bg3:     #121e35;
  --card:    rgba(14,22,44,0.92);
  --card2:   rgba(18,30,53,0.85);
  --bdr:     rgba(0,212,255,0.14);
  --bdr2:    rgba(0,212,255,0.32);
  --bdr3:    rgba(124,58,237,0.28);
  --a:       #00d4ff;
  --a2:      #7c3aed;
  --a3:      #00ffaa;
  --danger:  #ef4444;
  --warn:    #f59e0b;
  --ok:      #10b981;
  --fg:      #e8f0ff;
  --fg2:     #b0c4de;
  --muted:   #6b80a0;
  --font-h:  'Outfit', sans-serif;
  --font-b:  'Inter', sans-serif;
}

/* ── Global Reset ── */
html, body, .stApp {
  background: var(--bg) !important;
  font-family: var(--font-b) !important;
  color: var(--fg) !important;
}
.main .block-container {
  padding: 1.6rem 2.2rem 4rem !important;
  max-width: 1480px !important;
}

/* ── Hide Streamlit Chrome ── */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg2); }
::-webkit-scrollbar-thumb { background: rgba(0,212,255,0.28); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,212,255,0.5); }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #060b16 0%, #0a1220 100%) !important;
  border-right: 1px solid var(--bdr) !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }
[data-testid="stSidebar"] .stRadio > div { gap: 0.08rem !important; }
[data-testid="stSidebar"] .stRadio label {
  font-family: var(--font-b) !important;
  font-size: 0.84rem !important;
  font-weight: 500 !important;
  color: var(--muted) !important;
  padding: 0.5rem 0.9rem !important;
  border-radius: 10px !important;
  transition: all 0.2s ease !important;
  cursor: pointer !important;
  display: flex !important;
  align-items: center !important;
  gap: 0.5rem !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
  color: var(--fg) !important;
  background: rgba(0,212,255,0.07) !important;
}
[data-testid="stSidebar"] .stRadio [aria-checked="true"] + label,
[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"]:has(input:checked) {
  color: var(--a) !important;
  background: rgba(0,212,255,0.10) !important;
}

/* ── Typography ── */
h1, h2, h3 { font-family: var(--font-h) !important; }
h1 {
  font-size: 2.4rem !important; font-weight: 800 !important;
  background: linear-gradient(135deg, var(--a), var(--a2));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  line-height: 1.15 !important;
}
h2 { font-size: 1.3rem !important; font-weight: 700 !important; color: var(--fg) !important; }
h3 { font-size: 1rem !important; font-weight: 600 !important; color: var(--fg) !important; }

/* ── Cards ── */
.cg-card {
  background: var(--card);
  backdrop-filter: blur(18px);
  -webkit-backdrop-filter: blur(18px);
  border: 1px solid var(--bdr);
  border-radius: 16px;
  padding: 1.3rem 1.5rem;
  margin-bottom: 0.8rem;
  box-shadow: 0 4px 32px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.04);
  transition: border-color 0.25s, box-shadow 0.25s, transform 0.2s;
}
.cg-card:hover {
  border-color: var(--bdr2);
  box-shadow: 0 8px 48px rgba(0,212,255,0.1), inset 0 1px 0 rgba(255,255,255,0.06);
  transform: translateY(-1px);
}
.cg-card-title {
  font-family: var(--font-b); font-size: 0.68rem; font-weight: 600;
  letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted);
  margin-bottom: 0.45rem;
}
.cg-card-value {
  font-family: var(--font-h); font-size: 2rem; font-weight: 800; line-height: 1.1;
}
.cg-card-delta { font-size: 0.77rem; color: var(--muted); margin-top: 0.25rem; }

/* ── KPI Metric Card ── */
.kpi-card {
  background: var(--card2);
  border: 1px solid var(--bdr);
  border-radius: 14px;
  padding: 1.1rem 1.2rem;
  text-align: center;
  transition: all 0.22s;
  position: relative;
  overflow: hidden;
}
.kpi-card::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, transparent, var(--accent-clr, #00d4ff), transparent);
}
.kpi-card:hover {
  border-color: var(--bdr2);
  transform: translateY(-2px);
  box-shadow: 0 8px 32px rgba(0,212,255,0.08);
}
.kpi-icon { font-size: 1.4rem; margin-bottom: 0.35rem; }
.kpi-value {
  font-family: var(--font-h); font-size: 1.8rem; font-weight: 800;
  line-height: 1.1; color: var(--accent-clr, #00d4ff);
}
.kpi-label { font-size: 0.72rem; color: var(--muted); margin-top: 0.2rem; font-weight: 500; }
.kpi-sub { font-size: 0.68rem; color: var(--muted); margin-top: 0.1rem; }

/* ── Risk Badges ── */
.risk-badge {
  display: inline-flex; align-items: center; gap: 0.4rem;
  padding: 0.38rem 1rem; border-radius: 99px;
  font-family: var(--font-h); font-size: 0.84rem; font-weight: 700; letter-spacing: 0.04em;
}
.risk-high   { background: rgba(239,68,68,.15);  border: 1px solid rgba(239,68,68,.5);  color: #ef4444; }
.risk-medium { background: rgba(245,158,11,.15); border: 1px solid rgba(245,158,11,.5); color: #f59e0b; }
.risk-low    { background: rgba(16,185,129,.15); border: 1px solid rgba(16,185,129,.5); color: #10b981; }

/* ── Strategy Cards ── */
.strat-card {
  background: rgba(14,22,44,0.9);
  border-left: 3px solid var(--a);
  border-radius: 0 12px 12px 0;
  padding: 0.85rem 1.1rem;
  margin-bottom: 0.6rem;
  transition: background 0.2s;
}
.strat-card:hover { background: rgba(20,32,58,0.9); }
.strat-title { font-family: var(--font-h); font-weight: 700; font-size: 0.9rem; margin-bottom: 0.15rem; }
.strat-desc  { font-size: 0.79rem; color: var(--fg2); line-height: 1.55; }
.prio-badge  {
  display: inline-block; padding: 2px 7px; border-radius: 4px;
  font-size: 0.64rem; font-weight: 700; letter-spacing: 0.05em; margin-left: 0.4rem;
}

/* ── Mini Metric Pills ── */
.mp { background: rgba(0,212,255,0.07); border: 1px solid rgba(0,212,255,0.18); border-radius: 8px; padding: 0.5rem 0.7rem; text-align: center; }
.mp .mv { font-family: var(--font-h); font-size: 1.2rem; font-weight: 700; color: var(--a); }
.mp .ml { font-size: 0.65rem; color: var(--muted); }

/* ── Segment Cards ── */
.seg-card { background: var(--card); border: 1px solid var(--bdr); border-radius: 14px; padding: 1.1rem 1.2rem; height: 100%; }
.seg-name { font-family: var(--font-h); font-size: 0.92rem; font-weight: 700; margin-bottom: 0.3rem; }
.seg-desc { font-size: 0.77rem; color: var(--fg2); line-height: 1.5; margin-bottom: 0.5rem; }
.seg-action { font-size: 0.74rem; color: var(--a3); font-style: italic; }

/* ── Chat Bubbles ── */
.chat-user {
  background: rgba(124,58,237,0.18); border: 1px solid rgba(124,58,237,0.3);
  border-radius: 16px 16px 4px 16px; padding: 0.7rem 0.95rem; margin: 0.35rem 0;
  max-width: 80%; margin-left: auto; font-size: 0.86rem;
}
.chat-ai {
  background: rgba(0,212,255,0.07); border: 1px solid rgba(0,212,255,0.14);
  border-radius: 16px 16px 16px 4px; padding: 0.7rem 0.95rem; margin: 0.35rem 0;
  max-width: 86%; font-size: 0.86rem; line-height: 1.6;
}
.chat-lbl { font-size: 0.66rem; color: var(--muted); margin-bottom: 0.15rem; }

/* ── Buttons ── */
.stButton > button {
  background: linear-gradient(135deg, var(--a), var(--a2)) !important;
  color: #fff !important; border: none !important; border-radius: 10px !important;
  font-family: var(--font-h) !important; font-weight: 600 !important;
  font-size: 0.85rem !important; padding: 0.5rem 1.3rem !important;
  transition: opacity 0.2s, transform 0.15s !important;
  letter-spacing: 0.02em !important;
}
.stButton > button:hover { opacity: 0.84 !important; transform: translateY(-1px) !important; }
.stButton > button:active { transform: translateY(0) !important; }

/* ── Danger Button ── */
.danger-btn > button {
  background: rgba(239,68,68,0.15) !important;
  border: 1px solid rgba(239,68,68,0.4) !important;
  color: #ef4444 !important;
}
.danger-btn > button:hover { background: rgba(239,68,68,0.25) !important; }

/* ── Form Inputs ── */
.stSelectbox label, .stSlider label, .stNumberInput label,
.stTextInput label, .stTextArea label, .stMultiSelect label {
  color: var(--muted) !important; font-size: 0.78rem !important; font-weight: 500 !important;
}
.stTextInput input, .stNumberInput input, .stTextArea textarea {
  background: var(--bg2) !important; border: 1px solid var(--bdr) !important;
  border-radius: 8px !important; color: var(--fg) !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
  border-color: var(--a) !important; box-shadow: 0 0 0 2px rgba(0,212,255,0.13) !important;
}
.stSelectbox > div > div {
  background: var(--bg2) !important; border-color: var(--bdr) !important; color: var(--fg) !important;
}
div[data-baseweb="select"] > div {
  background: var(--bg2) !important; border-color: var(--bdr) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--bg2) !important; border-radius: 10px !important;
  border: 1px solid var(--bdr) !important; padding: 3px !important; gap: 2px !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important; color: var(--muted) !important;
  border-radius: 8px !important; font-family: var(--font-b) !important;
  font-size: 0.83rem !important; padding: 0.4rem 1rem !important; border: none !important;
  transition: all 0.2s !important;
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, var(--a), var(--a2)) !important;
  color: #fff !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.2rem !important; }

/* ── Dataframe / Table ── */
.stDataFrame { border-radius: 12px !important; overflow: hidden; }
[data-testid="stDataFrameResizable"] { border: 1px solid var(--bdr) !important; border-radius: 12px !important; }
.stDataFrame thead th {
  background: var(--bg3) !important; color: var(--a) !important;
  font-family: var(--font-b) !important; font-size: 0.76rem !important;
  font-weight: 600 !important; border-bottom: 1px solid var(--bdr) !important;
}

/* ── Expander ── */
details > summary {
  background: var(--bg2) !important; border: 1px solid var(--bdr) !important;
  border-radius: 10px !important; padding: 0.65rem 1rem !important;
  font-family: var(--font-b) !important; font-size: 0.83rem !important;
  color: var(--fg2) !important; cursor: pointer !important;
}

/* ── Alerts / Info ── */
.stAlert { border-radius: 12px !important; }

/* ── Login Card ── */
.login-wrap {
  max-width: 460px; margin: 2.5rem auto;
  background: var(--card); border: 1px solid var(--bdr);
  border-radius: 24px; padding: 2.8rem 2.4rem; text-align: center;
  box-shadow: 0 20px 80px rgba(0,0,0,0.5), 0 0 0 1px rgba(0,212,255,0.05);
}

/* ── Page Section Headers ── */
.sh { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.9rem; }
.sh-ico {
  width: 32px; height: 32px;
  background: linear-gradient(135deg, rgba(0,212,255,0.15), rgba(124,58,237,0.15));
  border: 1px solid var(--bdr); border-radius: 9px;
  display: flex; align-items: center; justify-content: center;
  font-size: 0.9rem; flex-shrink: 0;
}
.sh-ttl { font-family: var(--font-h); font-size: 1.15rem; font-weight: 700; color: var(--fg); }

/* ── Alert Card ── */
.alert-card {
  border-radius: 12px; padding: 0.85rem 1.1rem; margin-bottom: 0.6rem;
  display: flex; align-items: flex-start; gap: 0.8rem;
}
.alert-high   { background: rgba(239,68,68,.1);  border: 1px solid rgba(239,68,68,.3); }
.alert-medium { background: rgba(245,158,11,.1); border: 1px solid rgba(245,158,11,.3); }
.alert-info   { background: rgba(0,212,255,.07); border: 1px solid rgba(0,212,255,.2); }

/* ── Dividers ── */
.cg-div { border: none; border-top: 1px solid var(--bdr); margin: 1rem 0; }

/* ── Animations ── */
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }
@keyframes fadeIn { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }
@keyframes shimmer { 0%{background-position:-200% 0} 100%{background-position:200% 0} }

.live-dot {
  display: inline-block; width: 7px; height: 7px;
  background: var(--ok); border-radius: 50%;
  animation: pulse 1.6s ease-in-out infinite; margin-right: 4px;
}
.page-content { animation: fadeIn 0.3s ease-out; }

/* ── Footer ── */
.cg-footer {
  text-align: center; font-size: 0.7rem; color: var(--muted);
  padding: 2rem 0 0.5rem; border-top: 1px solid var(--bdr); margin-top: 2.5rem;
  font-family: var(--font-b);
}

/* ── Sidebar Active Nav ── */
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
  width: 100% !important; box-sizing: border-box !important;
}

/* ── Progress/Spinner ── */
.stSpinner > div { border-top-color: var(--a) !important; }

/* ── Number Input Arrows ── */
.stNumberInput button { background: var(--bg3) !important; border-color: var(--bdr) !important; }

/* ── Metric ── */
[data-testid="metric-container"] {
  background: var(--card) !important; border: 1px solid var(--bdr) !important;
  border-radius: 14px !important; padding: 1rem !important;
}
[data-testid="metric-container"] label {
  color: var(--muted) !important; font-size: 0.76rem !important; font-weight: 600 !important;
  text-transform: uppercase !important; letter-spacing: 0.08em !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
  font-family: var(--font-h) !important; font-weight: 800 !important; color: var(--fg) !important;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  State & Model Bootstrap
# ═══════════════════════════════════════════════════════════════════════════════
def init_state():
    defaults = {
        "authenticated": False, "user_role": None, "username": None,
        "prediction_history": [], "chat_history": [], "last_prediction": None,
        "_alerts_dismissed": set(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


@st.cache_resource(show_spinner=False)
def _load_model_artifacts(sig):
    return load_artifacts()

def boot_models():
    if not models_exist():
        from train_model import run_training
        run_training(n_samples=10_000)
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    artifact_names = ("best_model.pkl", "scaler.pkl", "feature_names.pkl", "best_model_name.pkl")
    sig = tuple(
        (n, os.stat(os.path.join(model_dir, n)).st_mtime_ns,
         os.path.getsize(os.path.join(model_dir, n)))
        for n in artifact_names
    )
    return _load_model_artifacts(sig)

@st.cache_data(show_spinner=False)
def _read_dataset(path, modified_ns, file_size):
    return pd.read_csv(path)

def get_dataset():
    path = os.path.join(os.path.dirname(__file__), "data", "banking_churn_data.csv")
    if not os.path.exists(path):
        from data.generate_data import generate_banking_dataset
        df = generate_banking_dataset(10_000)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
    stat = os.stat(path)
    return _read_dataset(path, stat.st_mtime_ns, stat.st_size)

@st.cache_data(show_spinner=False)
def _read_training_results(path, mtime, fsize):
    return load_training_results()

def get_training_results():
    path = os.path.join(os.path.dirname(__file__), "models", "training_results.pkl")
    if not os.path.exists(path):
        return None
    stat = os.stat(path)
    return _read_training_results(path, stat.st_mtime_ns, stat.st_size)

@st.cache_data(show_spinner=False)
def get_seg_data(df):
    if not segmentation_exists():
        train_segmentation(df, n_clusters=5)
    km, scaler, pca = load_segmentation()
    X = df[SEG_FEATURES].copy().fillna(0)
    Xs = scaler.transform(X)
    labels = km.predict(Xs)
    X_pca = pca.transform(Xs)
    summary = get_segment_summary(df, labels)
    return labels, X_pca, summary


# ═══════════════════════════════════════════════════════════════════════════════
#  Chart Helpers
# ═══════════════════════════════════════════════════════════════════════════════
_BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter,sans-serif", color="#6b80a0", size=11),
    margin=dict(l=16, r=16, t=44, b=16),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.07)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.07)"),
)
PAL = ["#00d4ff","#7c3aed","#10b981","#f59e0b","#ef4444","#f97316","#06b6d4","#8b5cf6","#ec4899","#14b8a6"]

def sf(fig, title="", h=370):
    fig.update_layout(
        **_BASE_LAYOUT, height=h,
        title=dict(text=title, font=dict(family="Outfit,sans-serif", size=13, color="#e8f0ff"))
    )
    return fig

def page_header(icon, title):
    st.markdown(
        f"<div class='sh'><div class='sh-ico'>{icon}</div>"
        f"<div class='sh-ttl'>{title}</div></div>",
        unsafe_allow_html=True
    )

def kpi_card(icon, value, label, color="#00d4ff", sub=""):
    return (
        f"<div class='kpi-card' style='--accent-clr:{color}'>"
        f"<div class='kpi-icon'>{icon}</div>"
        f"<div class='kpi-value'>{value}</div>"
        f"<div class='kpi-label'>{label}</div>"
        f"{'<div class=\"kpi-sub\">'+sub+'</div>' if sub else ''}"
        f"</div>"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  LOGIN / SIGNUP
# ═══════════════════════════════════════════════════════════════════════════════
def page_login():
    st.markdown("""
<div style="text-align:center;padding:1rem 0 0">
  <div style="font-size:3.5rem;margin-bottom:0.3rem">🛡️</div>
  <div style="font-family:'Outfit',sans-serif;font-size:2rem;font-weight:900;
       background:linear-gradient(135deg,#00d4ff,#7c3aed);
       -webkit-background-clip:text;-webkit-text-fill-color:transparent">
    ChurnGuard AI
  </div>
  <div style="color:#6b80a0;font-size:0.85rem;margin-top:0.3rem">
    Banking Intelligence Platform
  </div>
</div>""", unsafe_allow_html=True)

    _, col, _ = st.columns([1, 2, 1])
    with col:
        tab_in, tab_up = st.tabs(["🔐  Sign In", "✨  Create Account"])

        with tab_in:
            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
            u = st.text_input("Username", placeholder="Enter username", key="li_u")
            p = st.text_input("Password", type="password", placeholder="Enter password", key="li_p")
            if st.button("Sign In →", use_container_width=True, key="li_btn"):
                if u and p:
                    ok, result = verify_credentials(u, p)
                    if ok:
                        st.session_state.authenticated = True
                        st.session_state.user_role = result
                        st.session_state.username = u.strip().lower()
                        st.rerun()
                    else:
                        st.error(f"❌ {result}")
                else:
                    st.warning("Please enter both username and password.")
            st.markdown(
                f"<div style='margin-top:0.8rem;text-align:center;font-size:0.76rem;color:#6b80a0'>"
                f"ℹ️ {get_demo_hint()}</div>",
                unsafe_allow_html=True
            )

        with tab_up:
            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
            nu = st.text_input("Choose Username", placeholder="min. 3 characters", key="su_u")
            np_ = st.text_input("Password", type="password", placeholder="min. 6 characters", key="su_p")
            np2 = st.text_input("Confirm Password", type="password", placeholder="Repeat password", key="su_p2")
            nr = st.selectbox("Role", ["Analyst", "Viewer"], key="su_r")
            if st.button("Create Account →", use_container_width=True, key="su_btn"):
                if np_ != np2:
                    st.error("Passwords do not match.")
                else:
                    ok, msg = register_user(nu, np_, nr)
                    if ok:
                        st.success(f"✅ {msg} — you can now sign in.")
                    else:
                        st.error(f"❌ {msg}")


# ═══════════════════════════════════════════════════════════════════════════════
#  EXECUTIVE OVERVIEW (Home)
# ═══════════════════════════════════════════════════════════════════════════════
def page_overview(df):
    st.markdown("""
<div class='page-content' style="margin-bottom:1.4rem">
  <div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:0.3rem">
    <div style="font-size:2.2rem">🛡️</div>
    <div>
      <div style="font-family:'Outfit',sans-serif;font-size:1.6rem;font-weight:800;
           background:linear-gradient(135deg,#00d4ff,#7c3aed);
           -webkit-background-clip:text;-webkit-text-fill-color:transparent">
        ChurnGuard AI
      </div>
      <div style="font-size:0.8rem;color:#6b80a0;margin-top:1px">
        <span class='live-dot'></span>
        Live System · All Models Operational
      </div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    # ── KPI Row ───────────────────────────────────────────────────────────────
    cr      = df["Churn"].mean()
    total   = len(df)
    churned = int(df["Churn"].sum())
    retained = total - churned
    active_pct = df["IsActiveMember"].mean()
    avg_bal = df["Balance"].mean()
    high_risk_pct = (df["CreditScore"] < 500).mean()

    cols = st.columns(5)
    data = [
        ("👥", f"{total:,}",       "Total Customers",    "#00d4ff", ""),
        ("⚠️", f"{cr:.1%}",        "Churn Rate",         "#ef4444", f"{churned:,} at risk"),
        ("✅", f"{retained:,}",    "Retained Customers", "#10b981", f"{retained/total:.1%} retention"),
        ("💳", f"{active_pct:.1%}","Active Members",     "#7c3aed", ""),
        ("💰", f"${avg_bal:,.0f}", "Avg Account Balance","#f59e0b", ""),
    ]
    for col, (icon, val, lbl, clr, sub) in zip(cols, data):
        col.markdown(kpi_card(icon, val, lbl, clr, sub), unsafe_allow_html=True)

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    # ── Smart Alert Banner ────────────────────────────────────────────────────
    if cr > 0.18:
        st.markdown(f"""
<div class='alert-card alert-high'>
  <div style='font-size:1.3rem'>🚨</div>
  <div>
    <div style='font-family:"Outfit",sans-serif;font-weight:700;font-size:0.88rem;color:#ef4444'>
      High Portfolio Churn Alert
    </div>
    <div style='font-size:0.78rem;color:#b0c4de;margin-top:0.2rem'>
      Overall churn rate is {cr:.1%} — above the 18% critical threshold.
      Immediate retention action is recommended for high-risk segments.
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    # ── Charts Row 1 ─────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        vals = [int((df["Churn"] == 0).sum()), int((df["Churn"] == 1).sum())]
        fig = go.Figure(go.Pie(
            labels=["Retained", "Churned"], values=vals, hole=0.6,
            marker_colors=["#10b981", "#ef4444"],
            textinfo="percent+label", textfont_size=11,
        ))
        fig.add_annotation(text=f"{cr:.1%}<br>Churn", x=0.5, y=0.5, showarrow=False,
                           font_size=16, font_color="#e8f0ff", font_family="Outfit,sans-serif")
        st.plotly_chart(sf(fig, "📊 Churn Distribution", 310), use_container_width=True)

    with c2:
        gc = df.groupby("Geography")["Churn"].mean().reset_index()
        gc.columns = ["Geo", "Rate"]
        fig = px.bar(gc, x="Geo", y="Rate", color="Rate",
                     color_continuous_scale=["#10b981", "#f59e0b", "#ef4444"],
                     text=gc["Rate"].map("{:.1%}".format))
        fig.update_traces(textposition="outside", textfont_color="#e8f0ff", marker_line_width=0)
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(sf(fig, "🌍 Churn by Geography", 310), use_container_width=True)

    with c3:
        d2 = df.copy()
        d2["AG"] = pd.cut(d2["Age"], [18, 25, 35, 45, 55, 65, 100],
                          labels=["18-24","25-34","35-44","45-54","55-64","65+"], right=False)
        ag = d2.groupby("AG", observed=True)["Churn"].mean().reset_index()
        ag.columns = ["Group", "Rate"]
        fig = px.area(ag, x="Group", y="Rate", color_discrete_sequence=["#00d4ff"])
        fig.update_traces(line_width=2.5, fill="tozeroy",
                          fillcolor="rgba(0,212,255,0.08)")
        fig.add_hline(y=cr, line_dash="dot", line_color="#f59e0b",
                      annotation_text=f"Avg {cr:.1%}", annotation_font_color="#f59e0b")
        st.plotly_chart(sf(fig, "📈 Churn by Age Group", 310), use_container_width=True)

    # ── Charts Row 2 ─────────────────────────────────────────────────────────
    c1, c2 = st.columns([1.4, 1])
    with c1:
        pc = df.groupby("NumOfProducts")["Churn"].mean().reset_index()
        pc.columns = ["Products", "Rate"]
        tc = df.groupby("Tenure")["Churn"].mean().reset_index()
        tc.columns = ["Tenure", "Rate"]
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=("by No. of Products", "by Tenure (yrs)"))
        fig.add_trace(
            go.Bar(x=pc["Products"].astype(str), y=pc["Rate"],
                   marker_color=["#10b981","#f59e0b","#ef4444","#ef4444"][:len(pc)],
                   text=pc["Rate"].map("{:.1%}".format), textposition="outside",
                   textfont_color="#e8f0ff", showlegend=False),
            row=1, col=1)
        fig.add_trace(
            go.Scatter(x=tc["Tenure"], y=tc["Rate"], mode="lines+markers",
                       line_color="#7c3aed", line_width=2.5,
                       marker=dict(size=7, color="#7c3aed"), showlegend=False),
            row=1, col=2)
        fig.update_layout(**_BASE_LAYOUT, height=320,
                          title=dict(text="🔍 Churn Drivers Deep-Dive",
                                     font=dict(family="Outfit,sans-serif", size=13, color="#e8f0ff")))
        fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.07)")
        fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        cc = df.groupby("CardType")["Churn"].mean().reset_index().sort_values("Churn")
        cc.columns = ["Card", "Rate"]
        clrs = {"Silver":"#6b80a0","Gold":"#f59e0b","Platinum":"#7c3aed","Diamond":"#00d4ff"}
        colors_list = [clrs.get(c, "#00d4ff") for c in cc["Card"]]
        fig = go.Figure(go.Bar(
            x=cc["Rate"], y=cc["Card"], orientation="h",
            marker_color=colors_list,
            text=cc["Rate"].map("{:.1%}".format), textposition="outside",
            textfont_color="#e8f0ff",
        ))
        st.plotly_chart(sf(fig, "💳 Churn by Card Type", 320), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  CHURN PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════════
def page_predict(model, scaler, feature_names, best_name):
    page_header("🔮", "Customer Churn Prediction")
    st.caption("Fill the customer profile and click **Predict** for an AI-powered churn risk assessment.")

    col_a, col_b = st.columns([3, 1])
    with col_b:
        if st.button("🎲  Random Sample Customer", use_container_width=True):
            st.session_state["_sample"] = generate_sample_customer()
            st.rerun()

    s = st.session_state.get("_sample", {})

    with st.form("predict_form"):
        t1, t2, t3 = st.tabs(["👤 Demographics", "💰 Financials", "📱 Behaviour"])

        with t1:
            r1, r2, r3 = st.columns(3)
            with r1:
                age      = st.slider("Age", 18, 80, int(s.get("Age", 40)))
                gender   = st.selectbox("Gender", ["Male","Female"],
                               index=0 if s.get("Gender","Male")=="Male" else 1)
            with r2:
                geography = st.selectbox("Geography", ["France","Germany","Spain"],
                                index=["France","Germany","Spain"].index(s.get("Geography","France")))
                card_type = st.selectbox("Card Type", ["Silver","Gold","Platinum","Diamond"],
                                index=["Silver","Gold","Platinum","Diamond"].index(s.get("CardType","Gold")))
            with r3:
                tenure      = st.slider("Tenure (Years)", 0, 10, int(s.get("Tenure", 3)))
                num_products = st.selectbox("Number of Products", [1,2,3,4],
                                   index=int(s.get("NumOfProducts",2))-1)

        with t2:
            f1, f2, f3 = st.columns(3)
            with f1: credit_score = st.slider("Credit Score", 300, 850, int(s.get("CreditScore", 650)))
            with f2: balance = st.number_input("Account Balance ($)", 0.0, 300_000.0,
                                   float(s.get("Balance", 80_000.0)), step=1_000.0)
            with f3: salary = st.number_input("Estimated Salary ($)", 10_000.0, 200_000.0,
                                  float(s.get("EstimatedSalary", 100_000.0)), step=1_000.0)

        with t3:
            b1, b2, b3 = st.columns(3)
            with b1:
                is_active      = st.selectbox("Active Member", ["Yes","No"],
                                     index=0 if s.get("IsActiveMember",1)==1 else 1)
                online_banking = st.selectbox("Online Banking", ["Yes","No"],
                                     index=0 if s.get("OnlineBankingUsage",1)==1 else 1)
            with b2:
                tx_freq  = st.slider("Monthly Transactions", 1, 30, int(s.get("TransactionFrequency",12)))
                acct_act = st.selectbox("Account Activity", ["High","Medium","Low"],
                               index=["High","Medium","Low"].index(s.get("AccountActivity","Medium")))
            with b3:
                has_loan = st.selectbox("Has Active Loan", ["Yes","No"],
                               index=0 if s.get("HasLoan",0)==1 else 1)

        submitted = st.form_submit_button("⚡  Predict Churn Risk", use_container_width=True)

    if submitted:
        customer = {
            "Age": age, "Gender": gender, "Geography": geography,
            "CreditScore": credit_score, "Balance": balance, "EstimatedSalary": salary,
            "NumOfProducts": num_products, "Tenure": tenure,
            "IsActiveMember": 1 if is_active == "Yes" else 0,
            "TransactionFrequency": tx_freq, "AccountActivity": acct_act,
            "CardType": card_type, "OnlineBankingUsage": 1 if online_banking == "Yes" else 0,
            "HasLoan": 1 if has_loan == "Yes" else 0,
        }
        X = preprocess_single_customer(customer, scaler, feature_names)
        label, prob = predict_churn(model, X)
        risk_cat, risk_color, risk_emoji = get_risk_category(prob)
        strategies  = generate_retention_strategies(customer, prob)
        action_plan = get_action_plan(prob)
        seg_id, seg_meta = predict_segment(customer)

        entry = {
            "customer": customer, "probability": prob, "label": label,
            "risk_category": risk_cat, "segment": seg_meta["name"],
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        st.session_state.last_prediction = entry
        st.session_state.prediction_history.append(entry.copy())

        st.markdown("<hr class='cg-div'>", unsafe_allow_html=True)
        page_header("📊", "Prediction Results")

        left, right = st.columns([1, 1.6])
        with left:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=prob * 100,
                number={"suffix": "%", "font": {"size":40, "family":"Outfit,sans-serif", "color":"#e8f0ff"}},
                gauge={
                    "axis": {"range":[0,100], "tickcolor":"#6b80a0"},
                    "bar": {"color": risk_color, "thickness": 0.22},
                    "bgcolor": "rgba(0,0,0,0)",
                    "bordercolor": "rgba(255,255,255,0.1)",
                    "steps": [
                        {"range":[0,30],  "color":"rgba(16,185,129,0.12)"},
                        {"range":[30,60], "color":"rgba(245,158,11,0.12)"},
                        {"range":[60,100],"color":"rgba(239,68,68,0.12)"},
                    ],
                    "threshold": {"line":{"color":risk_color,"width":3}, "thickness":0.8, "value":prob*100},
                }
            ))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=260,
                              margin=dict(l=24,r=24,t=10,b=4), font=dict(color="#6b80a0"))
            st.plotly_chart(fig, use_container_width=True)

            rc_cls = {"High Risk":"risk-high","Medium Risk":"risk-medium","Low Risk":"risk-low"}[risk_cat]
            st.markdown(
                f"<div style='text-align:center;margin-top:-0.4rem'>"
                f"<span class='risk-badge {rc_cls}'>{risk_emoji} {risk_cat}</span></div>",
                unsafe_allow_html=True
            )
            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            m1.markdown(f"<div class='mp'><div class='mv'>{prob:.0%}</div><div class='ml'>Churn Prob</div></div>", unsafe_allow_html=True)
            m2.markdown(f"<div class='mp'><div class='mv'>{'YES' if label else 'NO'}</div><div class='ml'>Prediction</div></div>", unsafe_allow_html=True)
            m3.markdown(f"<div class='mp'><div class='mv'>{(1-prob):.0%}</div><div class='ml'>Retention</div></div>", unsafe_allow_html=True)
            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
            st.markdown(f"""
<div class="cg-card" style="text-align:center;padding:0.85rem">
  <div style="font-size:1.3rem">{seg_meta['icon']}</div>
  <div style="font-family:'Outfit',sans-serif;font-weight:700;font-size:0.85rem;
       color:{seg_meta['color']};margin:0.15rem 0">{seg_meta['name']}</div>
  <div style="font-size:0.7rem;color:#6b80a0">Customer Segment</div>
</div>""", unsafe_allow_html=True)

        with right:
            st.markdown("#### 🎯 Retention Strategies")
            pc_map = {"CRITICAL":"#ef4444","HIGH":"#f97316","MEDIUM":"#f59e0b","LOW":"#10b981"}
            for strat in strategies[:4]:
                pc = pc_map.get(strat["priority"], "#6b80a0")
                st.markdown(f"""
<div class="strat-card" style="border-left-color:{strat['badge_color']}">
  <div class="strat-title">{strat['icon']} {strat['title']}
    <span class="prio-badge" style="background:rgba(0,0,0,0.3);color:{pc};border:1px solid {pc}">
      {strat['priority']}
    </span>
  </div>
  <div class="strat-desc">{strat['description']}</div>
  <div style="font-size:0.7rem;color:#00d4aa;margin-top:0.3rem">
    Est. retention improvement: {strat['estimated_retention']}
  </div>
</div>""", unsafe_allow_html=True)

            # Feature importance mini-chart
            fi_df = get_feature_importances(model, feature_names)
            if fi_df is not None:
                display_names = get_feature_display_names()
                fi_top = fi_df.head(8).copy()
                fi_top["label"] = fi_top["feature"].map(lambda x: display_names.get(x, x))
                fig = go.Figure(go.Bar(
                    x=fi_top["importance"][::-1], y=fi_top["label"][::-1],
                    orientation="h",
                    marker=dict(
                        color=fi_top["importance"][::-1],
                        colorscale=[[0,"#7c3aed"],[1,"#00d4ff"]],
                    )
                ))
                st.plotly_chart(sf(fig, "🔍 Top Risk Factors (Feature Importance)", 290), use_container_width=True)

        # Action Plan
        ap_color  = {"High Risk":"rgba(239,68,68,0.1)","Medium Risk":"rgba(245,158,11,0.08)","Low Risk":"rgba(16,185,129,0.08)"}[risk_cat]
        ap_border = {"High Risk":"rgba(239,68,68,0.4)", "Medium Risk":"rgba(245,158,11,0.35)","Low Risk":"rgba(16,185,129,0.35)"}[risk_cat]
        st.markdown(f"""
<div style="background:{ap_color};border:1px solid {ap_border};border-radius:12px;padding:0.9rem 1.2rem;margin-top:0.6rem">
  <div style="font-family:'Outfit',sans-serif;font-weight:700;font-size:0.85rem;margin-bottom:0.35rem">
    ⚡ Recommended Next Step
  </div>
  <div style="font-size:0.82rem;color:#b0c4de">{action_plan['immediate']}</div>
  <div style="font-size:0.76rem;color:#6b80a0;margin-top:0.25rem">
    Short-term: {action_plan['short_term']}
  </div>
</div>""", unsafe_allow_html=True)

        # Downloads
        report_json = json.dumps({
            "generated": datetime.datetime.now().isoformat(), "model": best_name,
            "customer": customer, "churn_probability": f"{prob:.4f}",
            "risk_category": risk_cat, "predicted_churn": bool(label),
            "segment": seg_meta["name"],
            "strategies": [{"title":s["title"],"description":s["description"]} for s in strategies[:4]],
            "action_plan": {k:v for k,v in action_plan.items() if k not in ("color","emoji")},
        }, indent=2)

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button("⬇️  Download JSON Report", data=report_json,
                file_name=f"churnguard_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json", use_container_width=True)
        with dl2:
            try:
                pdf = generate_pdf_report(customer, prob, risk_cat, strategies, action_plan, best_name)
                st.download_button("📄  Download PDF Report", data=pdf,
                    file_name=f"churnguard_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf", use_container_width=True)
            except Exception:
                st.caption("📄 PDF export unavailable in this environment.")


# ═══════════════════════════════════════════════════════════════════════════════
#  CUSTOMER INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════════════
def page_customer_intelligence(df, model, scaler, feature_names):
    page_header("🔍", "Customer Intelligence")
    st.caption("Search, filter, and drill into individual customer profiles.")

    with st.expander("🔎  Filters & Search", expanded=True):
        fc1, fc2, fc3, fc4 = st.columns(4)
        with fc1:
            geo_f = st.multiselect("Geography", df["Geography"].unique().tolist(),
                                   default=df["Geography"].unique().tolist())
        with fc2:
            ch_f = st.multiselect("Churn Status", [0,1], default=[0,1],
                                   format_func=lambda x: "Churned" if x else "Retained")
        with fc3:
            age_r = st.slider("Age Range", int(df["Age"].min()), int(df["Age"].max()),
                              (int(df["Age"].min()), int(df["Age"].max())))
        with fc4:
            risk_f = st.multiselect("Card Type", df["CardType"].unique().tolist(),
                                    default=df["CardType"].unique().tolist())
        sid = st.text_input("Search Customer ID", placeholder="e.g. CG0000123")

    mask = (
        df["Geography"].isin(geo_f) &
        df["Churn"].isin(ch_f) &
        df["Age"].between(*age_r) &
        df["CardType"].isin(risk_f)
    )
    if sid:
        mask &= df["CustomerID"].str.contains(sid.upper(), na=False)
    filtered = df[mask]

    # KPI strip
    churn_rate = f"{filtered['Churn'].mean():.2%}" if not filtered.empty else "—"
    avg_bal    = f"${filtered['Balance'].mean():,.0f}" if not filtered.empty else "—"
    cols = st.columns(4)
    data = [
        ("Filtered Customers", f"{len(filtered):,}", "#00d4ff"),
        ("Filtered Churn Rate",  churn_rate,           "#ef4444"),
        ("Avg Balance",         avg_bal,               "#f59e0b"),
        ("Missing Values",      str(filtered.isnull().sum().sum()), "#10b981"),
    ]
    for col, (lbl, val, clr) in zip(cols, data):
        col.markdown(
            f"<div class='cg-card' style='text-align:center'>"
            f"<div class='cg-card-value' style='color:{clr}'>{val}</div>"
            f"<div class='cg-card-delta'>{lbl}</div></div>",
            unsafe_allow_html=True
        )

    st.caption(f"Showing {min(len(filtered), 300):,} of {len(filtered):,} filtered records.")
    display_cols = ["CustomerID","Age","Gender","Geography","CardType","Balance",
                    "NumOfProducts","Tenure","IsActiveMember","Churn"]
    display_df = filtered[[c for c in display_cols if c in filtered.columns]].head(300)
    st.dataframe(display_df, use_container_width=True, height=400)

    with st.expander("📈  Descriptive Statistics"):
        st.dataframe(filtered.describe().T.round(3), use_container_width=True)

    st.download_button(
        "⬇️  Download Filtered Data (CSV)",
        data=filtered.to_csv(index=False).encode(),
        file_name=f"churnguard_export_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
def page_analytics(df):
    page_header("📊", "Customer Analytics")
    st.caption("Key churn drivers across products, activity, card type, geography, and tenure.")

    t1, t2, t3 = st.tabs(["📦 Products & Activity", "🌍 Geography & Card", "📅 Trends"])

    with t1:
        c1, c2 = st.columns(2)
        with c1:
            pc = df.groupby("NumOfProducts")["Churn"].mean().reset_index()
            pc.columns = ["Products","Rate"]
            fig = px.bar(pc, x="Products", y="Rate", color="Rate",
                         color_continuous_scale=["#10b981","#f59e0b","#ef4444"],
                         text=pc["Rate"].map("{:.1%}".format))
            fig.update_traces(textposition="outside", textfont_color="#e8f0ff")
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(sf(fig,"Churn Rate by Number of Products"), use_container_width=True)
        with c2:
            ac = df.groupby("IsActiveMember")["Churn"].mean().reset_index()
            ac["Label"] = ac["IsActiveMember"].map({0:"Inactive",1:"Active"})
            fig = px.bar(ac, x="Label", y="Churn", color="Label",
                         color_discrete_map={"Active":"#10b981","Inactive":"#ef4444"},
                         text=ac["Churn"].map("{:.1%}".format))
            fig.update_traces(textposition="outside", textfont_color="#e8f0ff")
            fig.update_layout(showlegend=False)
            st.plotly_chart(sf(fig,"Active vs Inactive Churn Rate"), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            on = df.groupby("OnlineBankingUsage")["Churn"].mean().reset_index()
            on["Label"] = on["OnlineBankingUsage"].map({0:"No Online Banking",1:"Uses Online Banking"})
            fig = px.bar(on, x="Label", y="Churn", color="Label",
                         color_discrete_map={"No Online Banking":"#ef4444","Uses Online Banking":"#10b981"},
                         text=on["Churn"].map("{:.1%}".format))
            fig.update_traces(textposition="outside", textfont_color="#e8f0ff")
            fig.update_layout(showlegend=False)
            st.plotly_chart(sf(fig,"Online Banking vs Churn"), use_container_width=True)
        with c2:
            loan = df.groupby("HasLoan")["Churn"].mean().reset_index() if "HasLoan" in df.columns else None
            if loan is not None:
                loan["Label"] = loan["HasLoan"].map({0:"No Loan",1:"Has Loan"})
                fig = px.bar(loan, x="Label", y="Churn", color="Label",
                             color_discrete_map={"No Loan":"#7c3aed","Has Loan":"#f97316"},
                             text=loan["Churn"].map("{:.1%}".format))
                fig.update_traces(textposition="outside", textfont_color="#e8f0ff")
                fig.update_layout(showlegend=False)
                st.plotly_chart(sf(fig,"Loan Holders vs Churn"), use_container_width=True)

    with t2:
        c1, c2 = st.columns(2)
        with c1:
            geo = df.groupby("Geography")["Churn"].mean().reset_index()
            fig = px.bar(geo, x="Geography", y="Churn", color="Churn",
                         color_continuous_scale=["#10b981","#f59e0b","#ef4444"],
                         text=geo["Churn"].map("{:.1%}".format))
            fig.update_traces(textposition="outside", textfont_color="#e8f0ff")
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(sf(fig,"Churn Rate by Geography"), use_container_width=True)
        with c2:
            cc = df.groupby("CardType")["Churn"].mean().reset_index().sort_values("Churn")
            cc.columns = ["Card","Rate"]
            fig = px.bar(cc, x="Card", y="Rate", color_discrete_sequence=["#7c3aed"],
                         text=cc["Rate"].map("{:.1%}".format))
            fig.update_traces(textposition="outside", textfont_color="#e8f0ff")
            st.plotly_chart(sf(fig,"Churn Rate by Card Type"), use_container_width=True)

        # Balance distribution
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df[df["Churn"]==0]["Balance"], name="Retained",
                                   marker_color="rgba(16,185,129,0.7)", nbinsx=40, opacity=0.8))
        fig.add_trace(go.Histogram(x=df[df["Churn"]==1]["Balance"], name="Churned",
                                   marker_color="rgba(239,68,68,0.7)", nbinsx=40, opacity=0.8))
        fig.update_layout(**_BASE_LAYOUT, barmode="overlay", height=300,
                          title=dict(text="Balance Distribution: Retained vs Churned",
                                     font=dict(family="Outfit,sans-serif", size=13, color="#e8f0ff")))
        st.plotly_chart(fig, use_container_width=True)

    with t3:
        c1, c2 = st.columns(2)
        with c1:
            tc = df.groupby("Tenure")["Churn"].mean().reset_index()
            tc.columns = ["Tenure","Rate"]
            fig = px.bar(tc, x="Tenure", y="Rate", color="Rate",
                         color_continuous_scale=["#10b981","#f59e0b","#ef4444"],
                         text=tc["Rate"].map("{:.1%}".format))
            fig.update_traces(textposition="outside", textfont_color="#e8f0ff")
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(sf(fig,"Churn Rate by Tenure (Years)"), use_container_width=True)
        with c2:
            ac_lvl = df.groupby("AccountActivity")["Churn"].mean().reset_index() if "AccountActivity" in df.columns else None
            if ac_lvl is not None:
                ac_lvl.columns = ["Activity","Rate"]
                clr_map = {"High":"#10b981","Medium":"#f59e0b","Low":"#ef4444"}
                fig = px.bar(ac_lvl, x="Activity", y="Rate", color="Activity",
                             color_discrete_map=clr_map, text=ac_lvl["Rate"].map("{:.1%}".format))
                fig.update_traces(textposition="outside", textfont_color="#e8f0ff")
                fig.update_layout(showlegend=False)
                st.plotly_chart(sf(fig,"Churn by Account Activity Level"), use_container_width=True)

        # Credit Score vs Churn scatter
        d2 = df.copy()
        d2["AG"] = pd.cut(d2["Age"],[18,25,35,45,55,65,100],
                          labels=["18–24","25–34","35–44","45–54","55–64","65+"],right=False)
        ag = d2.groupby("AG", observed=True)["Churn"].mean().reset_index()
        ag.columns = ["Age Group","Churn Rate"]
        fig = px.line(ag, x="Age Group", y="Churn Rate", markers=True,
                      color_discrete_sequence=["#00d4ff"])
        fig.update_traces(line_width=2.5, marker_size=9)
        fig.add_hline(y=df["Churn"].mean(), line_dash="dot", line_color="#f59e0b",
                      annotation_text="Overall Avg", annotation_font_color="#f59e0b")
        st.plotly_chart(sf(fig,"Churn Rate by Age Group", 300), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════
def page_segmentation(df):
    page_header("🧩", "Customer Segmentation")
    st.caption("K-Means (k=5) clusters customers into 5 behavioural archetypes. PCA used for 2D visualisation.")

    with st.spinner("Loading segmentation model…"):
        labels, X_pca, summary = get_seg_data(df)

    # Segment summary cards
    for col, (seg_id, meta) in zip(st.columns(5), SEGMENT_META.items()):
        row  = summary[summary["Segment"] == seg_id]
        cnt  = int(row["Count"].values[0])  if not row.empty else 0
        rate = f"{row['ChurnRate'].values[0]:.1%}" if not row.empty else "–"
        col.markdown(f"""
<div class="cg-card" style="text-align:center;padding:0.95rem;border-top:2px solid {meta['color']}">
  <div style="font-size:1.5rem">{meta['icon']}</div>
  <div style="font-family:'Outfit',sans-serif;font-weight:700;font-size:0.78rem;
       color:{meta['color']};margin:0.2rem 0">{meta['name']}</div>
  <div style="font-family:'Outfit',sans-serif;font-size:1.4rem;font-weight:800">{cnt:,}</div>
  <div style="font-size:0.68rem;color:#6b80a0">customers</div>
  <div style="font-size:0.75rem;color:#ef4444;margin-top:0.15rem">{rate} churn</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # Charts
    c1, c2 = st.columns([1, 1.2])
    with c1:
        sc = summary[["SegmentName","ChurnRate","Count"]].sort_values("ChurnRate", ascending=False)
        clr_map = {m["name"]: m["color"] for m in SEGMENT_META.values()}
        fig = px.bar(sc, x="SegmentName", y="ChurnRate", color="SegmentName",
                     color_discrete_map=clr_map, text=sc["ChurnRate"].map("{:.1%}".format))
        fig.update_traces(textposition="outside", textfont_color="#e8f0ff")
        fig.update_layout(showlegend=False)
        fig.update_xaxes(tickangle=-15)
        st.plotly_chart(sf(fig,"Churn Rate by Segment", 380), use_container_width=True)

    with c2:
        # PCA scatter plot
        pca_df = pd.DataFrame(X_pca, columns=["PC1","PC2"])
        pca_df["Segment"] = [SEGMENT_META.get(l,{}).get("name",f"Cluster {l}") for l in labels]
        pca_df["Churn"] = df["Churn"].values[:len(pca_df)]
        fig = px.scatter(pca_df.sample(min(1500,len(pca_df)), random_state=42),
                         x="PC1", y="PC2", color="Segment",
                         color_discrete_map=clr_map, opacity=0.65, size_max=5,
                         hover_data={"PC1":False,"PC2":False,"Segment":True})
        fig.update_traces(marker_size=4)
        st.plotly_chart(sf(fig,"PCA Cluster Visualisation", 380), use_container_width=True)

    # Segment profile cards
    st.markdown("##### Segment Profiles")
    for col, (seg_id, meta) in zip(st.columns(5), SEGMENT_META.items()):
        col.markdown(f"""
<div class="seg-card" style="border-top:3px solid {meta['color']}">
  <div style="font-size:1.5rem;margin-bottom:0.2rem">{meta['icon']}</div>
  <div class="seg-name" style="color:{meta['color']}">{meta['name']}</div>
  <div class="seg-desc">{meta['description']}</div>
  <div class="seg-action">→ {meta['action']}</div>
</div>""", unsafe_allow_html=True)

    # Segment stats table
    with st.expander("📋 Detailed Segment Statistics"):
        disp = summary.rename(columns={
            "SegmentName":"Segment","Count":"Customers","ChurnRate":"Churn Rate",
            "AvgAge":"Avg Age","AvgBalance":"Avg Balance ($)",
            "AvgSalary":"Avg Salary ($)","AvgProducts":"Avg Products",
            "AvgTenure":"Avg Tenure","ActivePct":"Active %"
        })
        st.dataframe(disp.drop(columns=["Segment"],errors="ignore"), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
def page_model_performance(results, best_name, model, scaler, feature_names):
    page_header("🏆", "Model Performance")
    if results is None:
        st.warning("Run `python train_model.py` first.")
        return

    mres = {k:v for k,v in results.items() if not k.startswith("_")}
    br   = mres.get(best_name, list(mres.values())[0])

    # Active model banner
    st.markdown(f"""
<div style="background:rgba(0,212,255,0.06);border:1px solid rgba(0,212,255,0.22);
     border-radius:14px;padding:1.1rem 1.4rem;margin-bottom:1rem;
     display:flex;align-items:center;gap:1rem">
  <div style="font-size:2rem">🤖</div>
  <div>
    <div style="font-family:'Outfit',sans-serif;font-weight:800;font-size:1.1rem;color:#00d4ff">
      {best_name}
    </div>
    <div style="font-size:0.78rem;color:#6b80a0;margin-top:0.2rem">
      Active prediction model &nbsp;·&nbsp; Trained on banking churn dataset &nbsp;·&nbsp;
      Selected for highest AUC-ROC on held-out test set
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    # 4 Metric cards
    metrics = [
        ("Accuracy",  br["accuracy"],  "#00d4ff",
         "Correct predictions for all tested customers."),
        ("Precision", br["precision"], "#7c3aed",
         "Proportion of churn alerts that are correct."),
        ("Recall",    br["recall"],    "#f59e0b",
         "Share of actual churners correctly identified."),
        ("F1 Score",  br["f1"],        "#10b981",
         "Harmonic mean of Precision & Recall."),
    ]
    for col, (label, val, clr, desc) in zip(st.columns(4), metrics):
        col.markdown(f"""
<div class="cg-card" style="text-align:center">
  <div class="cg-card-title">{label}</div>
  <div class="cg-card-value" style="color:{clr}">{val:.2%}</div>
  <div style="font-size:0.7rem;color:#6b80a0;margin-top:0.4rem;line-height:1.45">{desc}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # All-model comparison table
    if len(mres) > 1:
        st.markdown("##### 📊 All Models Comparison")
        rows = []
        for name, res in mres.items():
            rows.append({
                "Model": f"{'⭐ ' if name == best_name else ''}{name}",
                "Accuracy":  f"{res['accuracy']:.4f}",
                "Precision": f"{res['precision']:.4f}",
                "Recall":    f"{res['recall']:.4f}",
                "F1 Score":  f"{res['f1']:.4f}",
                "AUC-ROC":   f"{res.get('roc_auc', res.get('auc',0)):.4f}" if "roc_auc" in res or "auc" in res else "—",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Feature importance chart
    fi_df = get_feature_importances(model, feature_names)
    if fi_df is not None:
        display_names = get_feature_display_names()
        fi_top = fi_df.head(12).copy()
        fi_top["label"] = fi_top["feature"].map(lambda x: display_names.get(x, x))
        fig = go.Figure(go.Bar(
            x=fi_top["importance"][::-1], y=fi_top["label"][::-1],
            orientation="h",
            marker=dict(color=fi_top["importance"][::-1],
                        colorscale=[[0,"#7c3aed"],[0.5,"#00d4ff"],[1,"#00ffaa"]]),
            text=fi_top["importance"][::-1].map("{:.4f}".format),
            textposition="outside", textfont_color="#e8f0ff",
        ))
        st.plotly_chart(sf(fig,"🔍 Feature Importances (Active Model)", 420), use_container_width=True)

    # How it works callout
    st.markdown(f"""
<div style="background:rgba(14,22,44,0.85);border:1px solid rgba(0,212,255,0.15);
     border-radius:14px;padding:1.1rem 1.4rem">
  <div style="font-family:'Outfit',sans-serif;font-weight:700;font-size:0.85rem;
       color:#e8f0ff;margin-bottom:0.5rem">How {best_name} Works</div>
  <div style="font-size:0.8rem;color:#b0c4de;line-height:1.65">
    <b style="color:#00d4ff">{best_name}</b> is a supervised classification algorithm trained on
    historical banking records. Given a customer's demographics, financial profile, and behavioural
    signals, it outputs a <b style="color:#e8f0ff">churn probability between 0% and 100%</b>.
    Customers above the 50% threshold are flagged as at-risk, enabling the retention team to intervene
    proactively. The model was evaluated on a held-out test set it has never seen during training.
  </div>
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  AI RISK ASSISTANT (rule-based + optional Anthropic)
# ═══════════════════════════════════════════════════════════════════════════════
_RULE_INSIGHTS = [
    ("🌍", "Geography Risk", "Germany consistently shows the highest churn rate in this dataset. Customers in Germany are ~2× more likely to churn than those in France or Spain. Targeted geo-specific retention campaigns are recommended."),
    ("📦", "Product Concentration", "Customers with only 1 product have significantly higher churn risk. Cross-selling a second product (e.g. savings + current account bundle) reduces churn probability by ~30%."),
    ("📱", "Digital Engagement", "Inactive members churn at nearly 2× the rate of active members. Boosting engagement through the mobile app — push notifications, personalised dashboards — is the highest-ROI retention lever."),
    ("💳", "Card Type Signal", "Silver cardholders show disproportionately high churn. A complimentary upgrade to Gold, paired with a cashback offer, can reduce churn by 15–20% for this cohort."),
    ("👴", "Senior Customer Risk", "Customers aged 55+ represent a high-risk segment. They often have single-product relationships and lower digital adoption. Dedicated relationship managers for this cohort are the recommended intervention."),
    ("💰", "Balance Paradox", "Counterintuitively, customers with very high balances (>$150k) sometimes show elevated churn — likely considering wealth management alternatives. VIP retention packages targeting this segment are warranted."),
    ("📊", "Tenure Sweet-Spot", "Churn risk is highest in the first 2 years of the customer relationship. Early-tenure onboarding programmes and 90-day check-ins dramatically reduce this early-exit risk."),
    ("🎯", "Multi-Factor Risk", "The highest-risk customers typically combine: Age 45-65 + Germany + 1 product + Inactive status. Customers matching all 4 factors should be escalated immediately to a Senior Relationship Manager."),
]

def page_ai_assistant(df, model, scaler, feature_names, best_name, results):
    page_header("🤖", "AI Risk Assistant")
    st.caption("Data-driven churn insights powered by your trained ML model and dataset analysis.")

    # Local insights — always available
    st.markdown("### 💡 Model-Driven Insights")
    cr = df["Churn"].mean()

    for i in range(0, len(_RULE_INSIGHTS), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(_RULE_INSIGHTS):
                icon, title, body = _RULE_INSIGHTS[i + j]
                col.markdown(f"""
<div class="cg-card">
  <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.5rem">
    <div style="font-size:1.2rem">{icon}</div>
    <div style="font-family:'Outfit',sans-serif;font-weight:700;font-size:0.9rem">{title}</div>
  </div>
  <div style="font-size:0.8rem;color:#b0c4de;line-height:1.6">{body}</div>
</div>""", unsafe_allow_html=True)

    # Live risk stats from dataset
    st.markdown("### 📈 Live Portfolio Risk Stats")
    high_risk_geo = df.groupby("Geography")["Churn"].mean().idxmax()
    high_risk_geo_rate = df.groupby("Geography")["Churn"].mean().max()
    single_prod_cr = df[df["NumOfProducts"]==1]["Churn"].mean()
    inactive_cr    = df[df["IsActiveMember"]==0]["Churn"].mean()
    senior_cr      = df[df["Age"]>55]["Churn"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card("🌍", f"{high_risk_geo_rate:.1%}", f"Churn in {high_risk_geo}", "#ef4444"), unsafe_allow_html=True)
    c2.markdown(kpi_card("📦", f"{single_prod_cr:.1%}",    "1-Product Churn Rate",  "#f97316"), unsafe_allow_html=True)
    c3.markdown(kpi_card("📱", f"{inactive_cr:.1%}",       "Inactive Member Churn", "#f59e0b"), unsafe_allow_html=True)
    c4.markdown(kpi_card("👴", f"{senior_cr:.1%}",         "Senior (55+) Churn",    "#7c3aed"), unsafe_allow_html=True)

    if results:
        mres = {k:v for k,v in results.items() if not k.startswith("_")}
        br   = mres.get(best_name, list(mres.values())[0])
        st.markdown("### 🤖 Active Model Stats")
        s1, s2, s3, s4 = st.columns(4)
        s1.markdown(kpi_card("🎯", f"{br['accuracy']:.2%}", "Model Accuracy", "#00d4ff"), unsafe_allow_html=True)
        s2.markdown(kpi_card("📡", f"{br['recall']:.2%}",   "Churn Recall",   "#10b981"), unsafe_allow_html=True)
        s3.markdown(kpi_card("🏅", f"{br['f1']:.2%}",       "F1 Score",       "#7c3aed"), unsafe_allow_html=True)
        s4.markdown(kpi_card("🔬", best_name,                "Active Model",   "#f59e0b"), unsafe_allow_html=True)

    # Optional Anthropic chatbot
    st.markdown("---")
    st.markdown("### 💬 AI Chat Assistant *(optional — requires Anthropic API key)*")
    with st.expander("⚙️  Connect Anthropic API", expanded=not bool(st.session_state.get("_ckey",""))):
        st.caption("Get your key at [console.anthropic.com](https://console.anthropic.com)")
        ak = st.text_input("Anthropic API Key", type="password",
                           value=st.session_state.get("_ckey",""), placeholder="sk-ant-…")
        if st.button("Save Key"):
            st.session_state["_ckey"] = ak
            st.success("Key saved for this session.")

    ak = st.session_state.get("_ckey","")
    if not ANTHROPIC_AVAILABLE:
        st.info("Install `anthropic` to enable the chat assistant: `pip install anthropic`")
        return

    if ak:
        _SYS = (
            "You are ChurnGuard AI Assistant — an expert banking analytics advisor. "
            "You specialise in churn prediction models, banking customer behaviour, "
            "retention strategy design, and interpreting ML metrics (AUC-ROC, F1, precision/recall). "
            "Be concise, professional, and data-driven. Use bullet points where appropriate. "
            "Never invent statistics. If asked about a customer's prediction, use provided context."
        )
        # Quick prompts
        qp_cols = st.columns(4)
        prompts = [
            "What factors most drive churn?",
            "Explain AUC-ROC in simple terms.",
            "Best retention for high-risk seniors?",
            "How does K-Means segment customers?",
        ]
        for col, prompt in zip(qp_cols, prompts):
            if col.button(prompt[:30]+"…", use_container_width=True):
                st.session_state.chat_history.append({"role":"user","content":prompt})
                st.rerun()

        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"<div class='chat-lbl'>👤 You</div><div class='chat-user'>{msg['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='chat-lbl'>🛡️ ChurnGuard AI</div><div class='chat-ai'>{msg['content']}</div>", unsafe_allow_html=True)

        ui = st.text_input("Message", placeholder="Ask about churn, customers, or retention…",
                           label_visibility="collapsed", key="ci")
        sc_col, cc_col = st.columns([5,1])
        with sc_col:
            send = st.button("Send ➤", use_container_width=True)
        with cc_col:
            if st.button("Clear", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

        if send and ui.strip():
            st.session_state.chat_history.append({"role":"user","content":ui.strip()})
            msgs = list(st.session_state.chat_history)
            if st.session_state.last_prediction:
                lp = st.session_state.last_prediction
                ctx = (f"[Context: Last prediction — Age {lp['customer']['Age']}, "
                       f"Geography {lp['customer']['Geography']}, "
                       f"Prob {lp['probability']:.1%}, Risk {lp['risk_category']}, "
                       f"Segment {lp.get('segment','–')}] ")
                msgs[-1] = {"role":"user","content":ctx+msgs[-1]["content"]}
            with st.spinner("Thinking…"):
                try:
                    client = _anthropic.Anthropic(api_key=ak)
                    resp   = client.messages.create(
                        model="claude-sonnet-4-20250514", max_tokens=700,
                        system=_SYS, messages=msgs)
                    reply = resp.content[0].text
                    st.session_state.chat_history[-1] = {"role":"user","content":ui.strip()}
                    st.session_state.chat_history.append({"role":"assistant","content":reply})
                    st.rerun()
                except Exception as e:
                    st.error(f"API error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PREDICTION HISTORY
# ═══════════════════════════════════════════════════════════════════════════════
def page_history():
    page_header("📋", "Prediction History")
    history = st.session_state.prediction_history
    if not history:
        st.info("No predictions yet. Go to **🔮 Predict** to get started.")
        return

    st.caption(f"{len(history)} prediction(s) this session")
    rows = [{
        "Timestamp":   h["timestamp"],
        "Age":         h["customer"]["Age"],
        "Gender":      h["customer"]["Gender"],
        "Geography":   h["customer"]["Geography"],
        "Credit Score":h["customer"]["CreditScore"],
        "Balance":     f"${h['customer']['Balance']:,.0f}",
        "Products":    h["customer"]["NumOfProducts"],
        "Segment":     h.get("segment","–"),
        "Churn Prob":  f"{h['probability']:.1%}",
        "Risk":        h["risk_category"],
        "Predicted":   "YES ⚠️" if h["label"] else "NO ✅",
    } for h in reversed(history)]

    hist_df = pd.DataFrame(rows)
    st.dataframe(hist_df, use_container_width=True, height=380)

    if len(history) >= 3:
        rc = hist_df["Risk"].value_counts().reset_index()
        rc.columns = ["Risk","Count"]
        fig = px.pie(rc, values="Count", names="Risk", color="Risk", hole=0.52,
                     color_discrete_map={"Low Risk":"#10b981","Medium Risk":"#f59e0b","High Risk":"#ef4444"})
        c1, _ = st.columns([1, 1])
        with c1:
            st.plotly_chart(sf(fig,"Session Risk Distribution", 280), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("⬇️  Export CSV", data=hist_df.to_csv(index=False).encode(),
            file_name=f"history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv", use_container_width=True)
    with c2:
        report_json = json.dumps(history, default=str, indent=2)
        st.download_button("⬇️  Export JSON", data=report_json,
            file_name=f"history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json", use_container_width=True)
    with c3:
        if st.button("🗑️  Clear History", use_container_width=True):
            st.session_state.prediction_history = []
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  ADMIN SETTINGS  (Administrator role only)
# ═══════════════════════════════════════════════════════════════════════════════
def page_admin(df, results, best_name):
    if st.session_state.user_role != "Administrator":
        st.error("🔒 This page is restricted to Administrators.")
        return

    page_header("⚙️", "Admin Settings")

    t1, t2, t3 = st.tabs(["👥 User Management", "📊 System Stats", "🔑 Change Password"])

    with t1:
        st.markdown("##### Current Users")
        users = list_users()
        user_df = pd.DataFrame(users)
        st.dataframe(user_df, use_container_width=True, hide_index=True)

        st.markdown("##### Add New User")
        na1, na2, na3 = st.columns(3)
        with na1: new_u = st.text_input("Username", key="adm_nu")
        with na2: new_p = st.text_input("Password", type="password", key="adm_np")
        with na3: new_r = st.selectbox("Role", ["Analyst","Viewer","Administrator"], key="adm_nr")
        if st.button("➕  Create User", use_container_width=True):
            ok, msg = register_user(new_u, new_p, new_r)
            st.success(msg) if ok else st.error(msg)

        st.markdown("##### Remove User")
        del1, del2 = st.columns(2)
        with del1: del_u = st.text_input("Username to remove", key="adm_du")
        with del2:
            st.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
            if st.button("🗑️  Delete User", use_container_width=True):
                ok, msg = delete_user(del_u)
                st.success(msg) if ok else st.error(msg)

    with t2:
        st.markdown("##### Dataset Statistics")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total Records", f"{len(df):,}")
        s2.metric("Total Features", f"{len(df.columns)}")
        s3.metric("Churn Rate", f"{df['Churn'].mean():.2%}")
        s4.metric("Missing Values", str(df.isnull().sum().sum()))

        st.markdown("##### Model Registry")
        if results:
            mres = {k:v for k,v in results.items() if not k.startswith("_")}
            rows = [{"Model": f"{'⭐ ' if n==best_name else ''}{n}",
                     "Accuracy": f"{r['accuracy']:.4f}",
                     "F1": f"{r['f1']:.4f}",
                     "Status": "Active" if n==best_name else "Available"}
                    for n, r in mres.items()]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with t3:
        st.markdown("##### Change Your Password")
        cp1, cp2, cp3 = st.columns(3)
        with cp1: old_pw = st.text_input("Current Password", type="password", key="cp_old")
        with cp2: new_pw = st.text_input("New Password", type="password", key="cp_new")
        with cp3: new_pw2 = st.text_input("Confirm New Password", type="password", key="cp_conf")
        if st.button("🔐  Update Password", use_container_width=True):
            if new_pw != new_pw2:
                st.error("New passwords do not match.")
            else:
                ok, msg = change_password(st.session_state.username, old_pw, new_pw)
                st.success(msg) if ok else st.error(msg)


# ═══════════════════════════════════════════════════════════════════════════════
#  SMART ALERTS
# ═══════════════════════════════════════════════════════════════════════════════
def page_alerts(df):
    page_header("🔔", "Smart Alerts")
    st.caption("Automated alerts generated from real-time dataset analysis.")

    alerts = []
    cr = df["Churn"].mean()
    if cr > 0.18:
        alerts.append(("high", "🚨 Critical Portfolio Churn Rate",
                       f"Overall churn rate is {cr:.1%} — exceeding the 18% critical threshold. Immediate action required.",
                       "portfolio_cr"))

    geo_cr = df.groupby("Geography")["Churn"].mean()
    for geo, rate in geo_cr.items():
        if rate > 0.25:
            alerts.append(("high", f"🌍 High Churn in {geo}",
                           f"{geo} shows {rate:.1%} churn rate — 25%+ threshold exceeded. Recommend geo-specific campaign.",
                           f"geo_{geo}"))

    single_cr = df[df["NumOfProducts"]==1]["Churn"].mean()
    if single_cr > 0.25:
        alerts.append(("medium", "📦 Single-Product Customers at Risk",
                       f"{single_cr:.1%} of single-product customers are at risk. Cross-sell opportunity: {int((df['NumOfProducts']==1).sum()):,} customers.",
                       "single_prod"))

    inactive_cr = df[df["IsActiveMember"]==0]["Churn"].mean()
    if inactive_cr > 0.25:
        alerts.append(("medium", "📱 Inactive Member Churn Spike",
                       f"Inactive members churning at {inactive_cr:.1%}. Recommend re-engagement campaign for {int((df['IsActiveMember']==0).sum()):,} customers.",
                       "inactive_cr"))

    senior_cr = df[df["Age"]>55]["Churn"].mean()
    if senior_cr > 0.2:
        alerts.append(("medium", "👴 Senior Customer Alert",
                       f"Customers 55+ churn at {senior_cr:.1%}. Dedicated outreach for this {int((df['Age']>55).sum()):,}-customer cohort is recommended.",
                       "senior_cr"))

    alerts.append(("info", "✅ Model Operational",
                   "All ML models loaded successfully. Predictions are live and accurate.",
                   "model_ok"))
    alerts.append(("info", "🔄 Data Freshness",
                   f"Dataset last loaded: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}. {len(df):,} customer records in memory.",
                   "data_fresh"))

    if not alerts:
        st.success("✅ No active alerts. All metrics are within acceptable thresholds.")
        return

    dismissed = st.session_state.get("_alerts_dismissed", set())
    active = [a for a in alerts if a[3] not in dismissed]

    if not active:
        st.success("✅ All alerts dismissed. No active alerts.")
        if st.button("↩️  Reset Alerts"):
            st.session_state._alerts_dismissed = set()
            st.rerun()
        return

    st.markdown(f"**{len(active)} active alert(s)**")
    for level, title, body, key in active:
        cls = f"alert-{level}"
        icon = "🔴" if level == "high" else "🟡" if level == "medium" else "🔵"
        c1, c2 = st.columns([9, 1])
        with c1:
            st.markdown(f"""
<div class='alert-card {cls}'>
  <div style='font-size:1.2rem'>{icon}</div>
  <div>
    <div style='font-family:"Outfit",sans-serif;font-weight:700;font-size:0.87rem'>{title}</div>
    <div style='font-size:0.78rem;color:#b0c4de;margin-top:0.2rem'>{body}</div>
  </div>
</div>""", unsafe_allow_html=True)
        with c2:
            if st.button("✕", key=f"dismiss_{key}"):
                st.session_state._alerts_dismissed.add(key)
                st.rerun()

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    if st.button("✕  Dismiss All Alerts"):
        st.session_state._alerts_dismissed = {a[3] for a in alerts}
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
def build_sidebar(best_name):
    with st.sidebar:
        # Logo + Brand
        st.markdown("""
<div style="text-align:center;padding:1.4rem 0 0.8rem">
  <div style="font-size:2.4rem;margin-bottom:0.25rem">🛡️</div>
  <div style="font-family:'Outfit',sans-serif;font-size:1.1rem;font-weight:800;
       background:linear-gradient(135deg,#00d4ff,#7c3aed);
       -webkit-background-clip:text;-webkit-text-fill-color:transparent">
    ChurnGuard AI
  </div>
  <div style="font-size:0.67rem;color:#6b80a0;margin-top:2px">Banking Intelligence Platform</div>
</div>
<hr style="border:none;border-top:1px solid rgba(0,212,255,0.12);margin:0.5rem 0 0.8rem">
""", unsafe_allow_html=True)

        # Role-aware navigation
        is_admin = st.session_state.user_role == "Administrator"
        nav_items = [
            "🏠  Overview",
            "🔮  Predict",
            "🔍  Customer Intelligence",
            "📊  Analytics",
            "🧩  Segmentation",
            "🏆  Model Performance",
            "🤖  AI Risk Assistant",
            "📋  Prediction History",
        ]
        if is_admin:
            nav_items += ["⚙️  Admin Settings", "🔔  Smart Alerts"]
        else:
            nav_items += ["🔔  Smart Alerts"]

        page = st.radio("nav", nav_items, label_visibility="collapsed")

        # Info widgets
        st.markdown("<hr style='border:none;border-top:1px solid rgba(0,212,255,0.12);margin:0.8rem 0'>",
                    unsafe_allow_html=True)
        st.markdown(f"""
<div style="background:rgba(0,212,255,0.05);border:1px solid rgba(0,212,255,0.14);
     border-radius:10px;padding:0.7rem;font-size:0.76rem;margin-bottom:0.5rem">
  <div style="color:#6b80a0;font-size:0.63rem;text-transform:uppercase;
       letter-spacing:0.09em;margin-bottom:3px">Active Model</div>
  <div style="color:#00d4ff;font-weight:600">{best_name}</div>
  <div style="color:#6b80a0;font-size:0.66rem;margin-top:2px">⭐ Best AUC on Test Set</div>
</div>
<div style="background:rgba(124,58,237,0.07);border:1px solid rgba(124,58,237,0.18);
     border-radius:10px;padding:0.7rem;font-size:0.76rem;margin-bottom:0.5rem">
  <div style="color:#6b80a0;font-size:0.63rem;text-transform:uppercase;
       letter-spacing:0.09em;margin-bottom:3px">Session</div>
  <div style="color:#e8f0ff">{len(st.session_state.prediction_history)} prediction(s)</div>
  <div style="color:#6b80a0;font-size:0.66rem;margin-top:2px">
    {st.session_state.get('username','guest')} · {st.session_state.get('user_role','Viewer')}
  </div>
</div>""", unsafe_allow_html=True)

        # Logout
        st.markdown("<div style='height:0.2rem'></div>", unsafe_allow_html=True)
        if st.button("🚪  Sign Out", use_container_width=True):
            for key in ["authenticated","user_role","username","prediction_history",
                        "chat_history","last_prediction","_sample","_alerts_dismissed"]:
                st.session_state.pop(key, None)
            st.rerun()

    return page


def render_footer():
    st.markdown("""
<div class="cg-footer">
  🛡️ ChurnGuard AI &nbsp;·&nbsp; AI-Powered Banking Intelligence Platform<br>
  <span style="font-size:0.66rem">
    Streamlit &nbsp;·&nbsp; Scikit-learn &nbsp;·&nbsp; XGBoost &nbsp;·&nbsp; Plotly &nbsp;·&nbsp; ReportLab
    &nbsp;·&nbsp; © 2024 · All rights reserved
  </span>
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    init_state()
    inject_css()

    if not st.session_state.authenticated:
        page_login()
        return

    with st.spinner("⚙️ Loading AI models…"):
        try:
            model, scaler, feature_names, best_name = boot_models()
        except Exception as e:
            st.error(f"Model load failed: {e}")
            st.info("Run `python train_model.py` first.")
            return

    df      = get_dataset()
    results = get_training_results()
    page    = build_sidebar(best_name)

    st.markdown("<div class='page-content'>", unsafe_allow_html=True)

    if   page.startswith("🏠"):  page_overview(df)
    elif page.startswith("🔮"):  page_predict(model, scaler, feature_names, best_name)
    elif page.startswith("🔍"):  page_customer_intelligence(df, model, scaler, feature_names)
    elif page.startswith("📊"):  page_analytics(df)
    elif page.startswith("🧩"):  page_segmentation(df)
    elif page.startswith("🏆"):  page_model_performance(results, best_name, model, scaler, feature_names)
    elif page.startswith("🤖"):  page_ai_assistant(df, model, scaler, feature_names, best_name, results)
    elif page.startswith("📋"):  page_history()
    elif page.startswith("⚙️"):  page_admin(df, results, best_name)
    elif page.startswith("🔔"):  page_alerts(df)

    st.markdown("</div>", unsafe_allow_html=True)
    render_footer()


if __name__ == "__main__":
    main()
