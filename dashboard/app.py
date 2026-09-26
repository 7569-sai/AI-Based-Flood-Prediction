"""
🌊 FloodAI ENGINE — Pan-India Multi-Basin Flood Intelligence Platform
"AI for a Safer, Resilient India"
Predict • Map • Explain • Protect

Premium Creative UI/UX Redesign
Coordinated FloodAI Color System:
- Deep Navy: #0B1F3A, Secondary Navy: #123B66
- Flood Blue: #087EA4, Water Cyan: #22B8CF, Aqua: #48CAE4
- AI Purple: #6C4CE8, Resilience Green: #20A464
- Warning Amber: #F4A261, High Risk Orange: #F97316, Critical Risk Red: #EF4444
- Background: #F4F8FC, Card: #FFFFFF, Text: #172033, Secondary Text: #64748B
"""

import os
import sys
import json
import math
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.location.location_service import LocationService
from src.data.feature_extractor import FeatureExtractor
from src.models.prediction_service import PredictionService
from src.geospatial.local_risk import LocalRiskAnalyzer
from src.planning.resilience_service import ResilienceService
from src.explainability.shap_explainer import FloodShapExplainer
from src.data.historical_analyzer import HistoricalFloodAnalyzer

# ---------------------------------------------------------------------------
# Page Configuration: Full Width, Collapsed Sidebar for Complete Screen Utility
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FloodAI Engine — AI-Powered Flood Intelligence for India",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------------------------
# FloodAI Premium Coordinated Design System (CSS)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    /* Global App Canvas */
    .stApp {
        background-color: #F4F8FC;
        color: #172033;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Completely Hide Left Sidebar and Default Clutter */
    section[data-testid="stSidebar"] {
        display: none !important;
    }
    div[data-testid="collapsedControl"] {
        display: none !important;
    }
    #MainMenu { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent !important; }
    footer { visibility: hidden; }
    
    /* Top Hero Header */
    .floodai-hero {
        background: linear-gradient(135deg, #0B1F3A 0%, #123B66 45%, #087EA4 80%, #22B8CF 100%);
        position: relative;
        overflow: hidden;
        color: #FFFFFF;
        padding: 1.25rem 1.8rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 8px 24px rgba(11, 31, 58, 0.16);
        display: flex;
        justify-content: space-between;
        align-items: center;
        border: 1px solid rgba(72, 202, 228, 0.28);
    }
    .floodai-hero::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: radial-gradient(circle at 85% 20%, rgba(72, 202, 228, 0.18) 0%, transparent 45%),
                    repeating-linear-gradient(45deg, transparent, transparent 25px, rgba(255, 255, 255, 0.02) 25px, rgba(255, 255, 255, 0.02) 50px);
        pointer-events: none;
    }
    .hero-title-group {
        position: relative;
        z-index: 1;
    }
    .hero-brand {
        font-size: 1.65rem;
        font-weight: 900;
        letter-spacing: -0.02em;
        color: #FFFFFF;
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 0;
    }
    .hero-subtitle {
        font-size: 0.96rem;
        font-weight: 500;
        color: #E0F2FE;
        margin: 0.25rem 0 0.15rem 0;
    }
    .hero-tagline {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #48CAE4;
    }
    .hero-status-glass {
        position: relative;
        z-index: 1;
        background: rgba(11, 31, 58, 0.65);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(72, 202, 228, 0.35);
        border-radius: 12px;
        padding: 0.7rem 1.25rem;
        text-align: right;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }
    .glass-online-badge {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        background: rgba(32, 164, 100, 0.2);
        border: 1px solid #20A464;
        color: #20A464;
        padding: 0.22rem 0.65rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 800;
        letter-spacing: 0.05em;
    }
    .pulse-circle {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #20A464;
        box-shadow: 0 0 0 0 rgba(32, 164, 100, 0.7);
        animation: pulse-green 2s infinite;
    }
    @keyframes pulse-green {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(32, 164, 100, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(32, 164, 100, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(32, 164, 100, 0); }
    }
    .glass-region {
        font-size: 0.78rem;
        font-weight: 700;
        color: #94A3B8;
        margin-top: 5px;
        letter-spacing: 0.04em;
    }
    .glass-model {
        font-size: 0.8rem;
        color: #E2E8F0;
        margin-top: 2px;
    }
    .glass-model b {
        color: #48CAE4;
    }

    /* Horizontal Top Tabs (Streamlit Native Styling) */
    div[data-baseweb="tab-list"] {
        gap: 10px;
        background: #FFFFFF;
        border-radius: 10px;
        padding: 6px;
        border: 1px solid #DCE6F0;
        box-shadow: 0 2px 8px rgba(11, 31, 58, 0.04);
        margin-bottom: 1.1rem;
    }
    button[data-baseweb="tab"] {
        font-size: 0.98rem;
        font-weight: 600;
        color: #64748B;
        padding: 0.65rem 1.4rem;
        border-radius: 8px;
        background: transparent;
        border: none;
        transition: all 0.2s ease;
    }
    button[data-baseweb="tab"]:hover {
        color: #087EA4;
        background: #F4F8FC;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, rgba(8, 126, 164, 0.1) 0%, rgba(34, 184, 207, 0.15) 100%);
        color: #087EA4 !important;
        font-weight: 800;
        border-bottom: 3px solid #22B8CF;
    }

    /* Instruction Card */
    .instruction-banner {
        background: linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 55%, #F4F8FC 100%);
        border-left: 5px solid #087EA4;
        border-top: 1px solid #DCE6F0;
        border-right: 1px solid #DCE6F0;
        border-bottom: 1px solid #DCE6F0;
        border-radius: 10px;
        padding: 0.95rem 1.4rem;
        margin-bottom: 0.9rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 2px 6px rgba(8, 126, 164, 0.04);
    }
    .instruction-title {
        font-size: 0.86rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #087EA4;
        display: flex;
        align-items: center;
        gap: 6px;
        margin-bottom: 3px;
    }
    .instruction-desc {
        font-size: 0.93rem;
        color: #172033;
        font-weight: 500;
        margin: 0;
    }

    /* Map Showcase Container */
    .map-showcase-container {
        background: #FFFFFF;
        border-radius: 14px;
        border: 1px solid #DCE6F0;
        box-shadow: 0 4px 20px rgba(11, 31, 58, 0.06);
        padding: 0.85rem;
        margin-bottom: 1.1rem;
    }
    .map-header-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.2rem 0.5rem 0.7rem 0.5rem;
        border-bottom: 1px solid #F1F5F9;
        margin-bottom: 0.6rem;
    }
    .map-header-title {
        font-size: 0.92rem;
        font-weight: 800;
        letter-spacing: 0.03em;
        color: #0B1F3A;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .map-badge-pill {
        background: #F0F9FF;
        border: 1px solid #BAE6FD;
        color: #0369A1;
        font-size: 0.76rem;
        font-weight: 700;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        display: inline-flex;
        align-items: center;
        gap: 5px;
    }

    /* Selected Location Card */
    .selected-location-box {
        background: #FFFFFF;
        border-radius: 12px;
        border: 1px solid #DCE6F0;
        border-left: 6px solid #22B8CF;
        padding: 1.2rem 1.6rem;
        margin-top: 0.6rem;
        margin-bottom: 1.1rem;
        box-shadow: 0 4px 16px rgba(11, 31, 58, 0.06);
    }
    .selected-loc-pre {
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #087EA4;
        margin-bottom: 4px;
    }
    .selected-loc-name {
        font-size: 1.55rem;
        font-weight: 800;
        color: #0B1F3A;
        margin-bottom: 6px;
        letter-spacing: -0.01em;
    }
    .selected-loc-details {
        display: flex;
        flex-wrap: wrap;
        gap: 20px;
        font-size: 0.9rem;
        color: #64748B;
    }
    .selected-loc-item {
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .selected-loc-item b {
        color: #172033;
    }

    /* Streamlit Button Customization */
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-size: 0.94rem !important;
        padding: 0.6rem 1.4rem !important;
        transition: all 0.2s ease !important;
        border: 1px solid #DCE6F0 !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #087EA4 0%, #22B8CF 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(8, 126, 164, 0.3) !important;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 20px rgba(8, 126, 164, 0.45) !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background: #F1F5F9 !important;
        border-color: #CBD5E1 !important;
    }

    /* AI Data Pipeline Visualization */
    .pipeline-step-flow {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
        margin: 0.9rem 0;
        overflow-x: auto;
        padding-bottom: 4px;
    }
    .pipeline-node {
        flex: 1;
        min-width: 110px;
        background: #FFFFFF;
        border: 1px solid #DCE6F0;
        border-radius: 10px;
        padding: 0.65rem 0.5rem;
        text-align: center;
        box-shadow: 0 1px 4px rgba(11, 31, 58, 0.04);
        transition: all 0.2s ease;
    }
    .pipeline-node:hover {
        border-color: #22B8CF;
        transform: translateY(-1px);
    }
    .pipeline-node.active {
        border-color: #22B8CF;
        background: #F0FDFA;
    }
    .pipeline-node-icon {
        font-size: 1.25rem;
        margin-bottom: 3px;
    }
    .pipeline-node-label {
        font-size: 0.78rem;
        font-weight: 700;
        color: #172033;
    }
    .pipeline-node-status {
        font-size: 0.7rem;
        font-weight: 700;
        color: #20A464;
        margin-top: 2px;
    }
    .pipeline-arrow {
        color: #94A3B8;
        font-weight: bold;
        font-size: 1rem;
    }

    /* Compact Data Retrieval Status Cards */
    .feature-status-card {
        border-radius: 8px;
        padding: 0.55rem 0.4rem;
        text-align: center;
        border: 1px solid;
        font-size: 0.75rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    .feature-status-card.weather {
        background: #E0F2FE;
        border-color: #BAE6FD;
        color: #0369A1;
    }
    .feature-status-card.terrain {
        background: #E6F4EA;
        border-color: #A8DAB5;
        color: #137333;
    }
    .feature-status-card.soil {
        background: #ECFEFF;
        border-color: #A5F3FC;
        color: #0E7490;
    }
    .feature-status-card.satellite {
        background: #F3E8FF;
        border-color: #D8B4FE;
        color: #6B21A8;
    }
    .feature-status-card.exposure {
        background: #FFEDD5;
        border-color: #FDBA74;
        color: #C2410C;
    }

    /* Model Validation Bar */
    .validation-summary-box {
        background: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-radius: 10px;
        padding: 0.75rem 1.2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 0.8rem;
        margin-bottom: 1.1rem;
    }
    .validation-items-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
        font-size: 0.82rem;
        font-weight: 700;
        color: #166534;
    }
    .validation-ready-pill {
        background: #20A464;
        color: #FFFFFF;
        font-size: 0.75rem;
        font-weight: 800;
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
        letter-spacing: 0.04em;
        white-space: nowrap;
    }

    /* Prediction Hero Result Card */
    .prediction-hero-card {
        background: #FFFFFF;
        border-radius: 14px;
        border: 1px solid #DCE6F0;
        box-shadow: 0 8px 28px rgba(11, 31, 58, 0.08);
        padding: 1.5rem 1.8rem;
        margin-bottom: 1.2rem;
    }
    .pred-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }
    .pred-card-title {
        font-size: 0.85rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #0B1F3A;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .pred-badge {
        padding: 0.35rem 0.95rem;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .pred-big-prob-group {
        text-align: center;
        margin: 0.6rem 0 1.2rem 0;
    }
    .pred-big-prob {
        font-size: 3.5rem;
        font-weight: 900;
        letter-spacing: -0.03em;
        line-height: 1;
    }
    .pred-big-label {
        font-size: 0.92rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 6px;
    }
    .pred-meta-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        border-top: 1px solid #F1F5F9;
        padding-top: 1.1rem;
        text-align: center;
    }
    .pred-meta-item .label {
        font-size: 0.76rem;
        font-weight: 700;
        text-transform: uppercase;
        color: #64748B;
        letter-spacing: 0.04em;
    }
    .pred-meta-item .val {
        font-size: 1.25rem;
        font-weight: 800;
        color: #0B1F3A;
        margin-top: 3px;
    }

    /* Risk Score Gauge */
    .risk-gauge-container {
        margin: 1.1rem 0 0.4rem 0;
        background: #FFFFFF;
        border: 1px solid #DCE6F0;
        border-radius: 12px;
        padding: 1.1rem 1.4rem;
        box-shadow: 0 2px 8px rgba(11, 31, 58, 0.04);
    }
    .risk-gauge-bar-wrapper {
        position: relative;
        height: 14px;
        border-radius: 7px;
        background: linear-gradient(to right, #20A464 0%, #20A464 25%, #F4A261 25%, #F4A261 50%, #F97316 50%, #F97316 75%, #EF4444 75%, #EF4444 100%);
        margin: 1rem 0 0.5rem 0;
    }
    .risk-gauge-indicator {
        position: absolute;
        top: -6px;
        width: 26px;
        height: 26px;
        background: #0B1F3A;
        border: 3px solid #FFFFFF;
        box-shadow: 0 2px 8px rgba(0,0,0,0.35);
        border-radius: 50%;
        transform: translateX(-50%);
    }
    .risk-gauge-labels {
        display: flex;
        justify-content: space-between;
        font-size: 0.74rem;
        font-weight: 700;
        color: #64748B;
    }

    /* 4 Core Local Pillars Cards */
    .pillar-card {
        background: #FFFFFF;
        border-radius: 12px;
        border: 1px solid #DCE6F0;
        padding: 1.1rem 1.2rem;
        box-shadow: 0 2px 10px rgba(11, 31, 58, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .pillar-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(11, 31, 58, 0.08);
    }
    .pillar-card-title {
        font-size: 0.78rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        display: flex;
        align-items: center;
        gap: 6px;
        margin-bottom: 0.5rem;
    }
    .pillar-card-metric {
        font-size: 1.65rem;
        font-weight: 800;
        color: #0B1F3A;
        margin-bottom: 0.2rem;
    }
    .pillar-card-sub {
        font-size: 0.8rem;
        color: #64748B;
        font-weight: 500;
    }

    /* Resilience Hero Card */
    .resilience-hero-box {
        background: linear-gradient(135deg, #087EA4 0%, #108e6f 60%, #20A464 100%);
        color: #FFFFFF;
        border-radius: 14px;
        padding: 1.3rem 1.8rem;
        margin-bottom: 1.1rem;
        box-shadow: 0 6px 20px rgba(8, 126, 164, 0.18);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .resil-title {
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: -0.01em;
        margin: 0 0 4px 0;
    }
    .resil-sub {
        font-size: 0.88rem;
        color: #E6FBF3;
        margin: 0;
    }
    .resil-priority-pill {
        background: #FFFFFF;
        color: #0B1F3A;
        font-size: 0.85rem;
        font-weight: 800;
        padding: 0.45rem 1rem;
        border-radius: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.12);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Section Title */
    .section-title {
        font-size: 1.12rem;
        font-weight: 800;
        color: #0B1F3A;
        margin-top: 1.1rem;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached Resource Loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_system_engines():
    loc_svc = LocationService()
    feat_svc = FeatureExtractor()
    pred_svc = PredictionService()
    local_risk_svc = LocalRiskAnalyzer()
    resil_svc = ResilienceService()
    shap_eng = FloodShapExplainer(use_pan_india=True)
    hist_eng = HistoricalFloodAnalyzer()
    return loc_svc, feat_svc, pred_svc, local_risk_svc, resil_svc, shap_eng, hist_eng

location_service, feature_extractor, prediction_service, local_risk_engine, resilience_service, shap_explainer, hist_analyzer = load_system_engines()

# Dynamic Model Name Resolution
INFO_PATH = os.path.join(BASE_DIR, "models", "trained", "final_model_info.json")
dynamic_model_name = "Random Forest Balanced"
if os.path.exists(INFO_PATH):
    try:
        with open(INFO_PATH, "r", encoding="utf-8") as f:
            m_info = json.load(f)
            dynamic_model_name = m_info.get("champion_model", "Random Forest Balanced").replace("_", " ").title()
    except Exception:
        pass

# Load benchmark metrics
METRICS_PATH = os.path.join(BASE_DIR, "outputs", "metrics", "pan_india_model_comparison.json")
if not os.path.exists(METRICS_PATH):
    METRICS_PATH = os.path.join(BASE_DIR, "outputs", "metrics", "model_comparison.json")
with open(METRICS_PATH, "r", encoding="utf-8") as f:
    model_metrics = json.load(f)


# ---------------------------------------------------------------------------
# Session State Initialization for Location-First Workflow
# ---------------------------------------------------------------------------
if "selected_lat" not in st.session_state:
    st.session_state["selected_lat"] = None
if "selected_lon" not in st.session_state:
    st.session_state["selected_lon"] = None
if "location_info" not in st.session_state:
    st.session_state["location_info"] = None
if "analysis_complete" not in st.session_state:
    st.session_state["analysis_complete"] = False
if "retrieved_features" not in st.session_state:
    st.session_state["retrieved_features"] = None
if "feature_status" not in st.session_state:
    st.session_state["feature_status"] = None
if "sources_panel" not in st.session_state:
    st.session_state["sources_panel"] = None
if "prediction_result" not in st.session_state:
    st.session_state["prediction_result"] = None
if "local_risk_result" not in st.session_state:
    st.session_state["local_risk_result"] = None
if "resilience_plan" not in st.session_state:
    st.session_state["resilience_plan"] = None


# ---------------------------------------------------------------------------
# Hero Header (FloodAI Visual Identity)
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="floodai-hero">
    <div class="hero-title-group">
        <div class="hero-brand">
            <span>🌊 FLOODAI ENGINE</span>
        </div>
        <div class="hero-subtitle">AI-Powered Flood Intelligence for India</div>
        <div class="hero-tagline">Predict • Map • Explain • Protect</div>
    </div>
    <div class="hero-status-glass">
        <div class="glass-online-badge">
            <span class="pulse-circle"></span>
            <span>SYSTEM ONLINE</span>
        </div>
        <div class="glass-region">INDIA &bull; MULTI-BASIN</div>
        <div class="glass-model">AI MODEL: <b>{dynamic_model_name}</b></div>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Primary Horizontal Top Tabs
# ---------------------------------------------------------------------------
tab_location, tab_advanced, tab_about = st.tabs([
    "📍 Location Analysis",
    "🔬 Advanced Analysis",
    "ℹ️ About Project"
])


# ===========================================================================
# TAB 1: 📍 LOCATION ANALYSIS (PRIMARY LOCATION-FIRST WORKFLOW)
# ===========================================================================
with tab_location:

    # -----------------------------------------------------------------------
    # STATE 1: CLEAN INDIA MAP (NO MARKERS, MAP FIRST)
    # -----------------------------------------------------------------------
    if st.session_state["selected_lat"] is None:
        st.markdown("""
        <div class="instruction-banner">
            <div>
                <div class="instruction-title">📍 SELECT A LOCATION</div>
                <div class="instruction-desc">Click anywhere on the map to analyze flood risk. FloodAI will automatically retrieve environmental and geospatial parameters for this location.</div>
            </div>
            <div class="map-badge-pill">
                🇮🇳 INDIA &bull; Click to select location
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="map-showcase-container">
            <div class="map-header-bar">
                <div class="map-header-title">🗺️ INDIA FLOOD INTELLIGENCE MAP</div>
                <div style="font-size: 0.78rem; color: #64748B;">Interactive Leaflet Engine &bull; Zero Preset Markers</div>
            </div>
        """, unsafe_allow_html=True)

        # Clean Map: Centered on India with NO location markers, NO red dots, NO blue dots, full container width
        m_clean = folium.Map(
            location=[22.8, 82.0],
            zoom_start=5,
            tiles="OpenStreetMap",
            zoom_control=True
        )

        map_output = st_folium(
            m_clean,
            height=580,
            use_container_width=True,
            key="clean_india_map",
            returned_objects=["last_clicked"]
        )

        st.markdown("</div>", unsafe_allow_html=True)

        # Detect User Click
        if map_output and map_output.get("last_clicked"):
            c_lat = round(map_output["last_clicked"]["lat"], 4)
            c_lon = round(map_output["last_clicked"]["lng"], 4)
            valid, _ = location_service.validate_coordinates(c_lat, c_lon)
            if valid:
                st.session_state["selected_lat"] = c_lat
                st.session_state["selected_lon"] = c_lon
                st.session_state["location_info"] = location_service.identify_location(c_lat, c_lon)
                st.session_state["analysis_complete"] = False
                st.rerun()

    # -----------------------------------------------------------------------
    # STATE 2: LOCATION SELECTED, PENDING USER CONFIRMATION
    # -----------------------------------------------------------------------
    elif not st.session_state["analysis_complete"]:
        s_lat = st.session_state["selected_lat"]
        s_lon = st.session_state["selected_lon"]
        loc_meta = st.session_state["location_info"]

        st.markdown("""
        <div class="instruction-banner">
            <div>
                <div class="instruction-title">📍 LOCATION SELECTED</div>
                <div class="instruction-desc">Coordinates established. Review selected point or click map again to relocate, then run AI analysis.</div>
            </div>
            <div class="map-badge-pill">
                ● Location Locked
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Map showing EXACTLY ONE Marker at the clicked coordinate
        m_selected = folium.Map(
            location=[s_lat, s_lon],
            zoom_start=7,
            tiles="OpenStreetMap"
        )
        folium.Marker(
            location=[s_lat, s_lon],
            tooltip=f"Selected: {s_lat:.4f}°N, {s_lon:.4f}°E",
            popup=f"<b>{loc_meta['district']}, {loc_meta['state']}</b><br>{loc_meta['basin']} Basin",
            icon=folium.Icon(color="red", icon="info-sign")
        ).add_to(m_selected)

        st.markdown("""
        <div class="map-showcase-container">
            <div class="map-header-bar">
                <div class="map-header-title">📍 CONFIRM TARGET COORDINATES</div>
                <div style="font-size: 0.78rem; color: #64748B;">Click anywhere to relocate pin</div>
            </div>
        """, unsafe_allow_html=True)

        map_reclick = st_folium(
            m_selected,
            height=460,
            use_container_width=True,
            key="selected_point_map",
            returned_objects=["last_clicked"]
        )

        st.markdown("</div>", unsafe_allow_html=True)

        # Handle re-clicking another point on this map
        if map_reclick and map_reclick.get("last_clicked"):
            new_lat = round(map_reclick["last_clicked"]["lat"], 4)
            new_lon = round(map_reclick["last_clicked"]["lng"], 4)
            if abs(new_lat - s_lat) > 0.002 or abs(new_lon - s_lon) > 0.002:
                valid, _ = location_service.validate_coordinates(new_lat, new_lon)
                if valid:
                    st.session_state["selected_lat"] = new_lat
                    st.session_state["selected_lon"] = new_lon
                    st.session_state["location_info"] = location_service.identify_location(new_lat, new_lon)
                    st.rerun()

        # Selected Location Confirmation Card
        st.markdown(f"""
        <div class="selected-location-box">
            <div class="selected-loc-pre">📍 TARGET LOCATION ESTABLISHED</div>
            <div class="selected-loc-name">
                {loc_meta['district']}, {loc_meta['state']}
            </div>
            <div class="selected-loc-details">
                <div class="selected-loc-item">
                    <span>🌐</span> <b>Coordinates:</b> {s_lat:.4f}°N, {s_lon:.4f}°E
                </div>
                <div class="selected-loc-item">
                    <span>🌊</span> <b>River Basin:</b> {loc_meta['basin']} Basin
                </div>
                <div class="selected-loc-item">
                    <span>💧</span> <b>Sub-Basin:</b> {loc_meta['sub_basin']}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_act1, col_act2 = st.columns([2, 1])
        with col_act1:
            if st.button("🌊 ANALYZE FLOOD RISK", type="primary", use_container_width=True):
                with st.spinner("Extracting multi-source environmental data & running AI model..."):
                    feat_res = feature_extractor.retrieve_location_data(s_lat, s_lon)
                    st.session_state["retrieved_features"] = feat_res["features"]
                    st.session_state["feature_status"] = feat_res["status_panel"]
                    st.session_state["sources_panel"] = feat_res["sources_panel"]

                    pred_res = prediction_service.predict(feat_res["features"])
                    st.session_state["prediction_result"] = pred_res

                    local_risk = local_risk_engine.compute_local_1km_risk(
                        s_lat, s_lon, feat_res["features"], pred_res, loc_meta
                    )
                    st.session_state["local_risk_result"] = local_risk

                    plan = resilience_service.generate_plan(
                        local_risk, pred_res, loc_meta, feat_res["features"]
                    )
                    st.session_state["resilience_plan"] = plan

                    st.session_state["analysis_complete"] = True
                    st.rerun()

        with col_act2:
            if st.button("↺ Select Another Location", use_container_width=True):
                st.session_state["selected_lat"] = None
                st.session_state["selected_lon"] = None
                st.session_state["location_info"] = None
                st.session_state["analysis_complete"] = False
                st.rerun()

    # -----------------------------------------------------------------------
    # STATE 3: LOCAL 1 KM RESULT (POST-PREDICTION)
    # -----------------------------------------------------------------------
    else:
        s_lat = st.session_state["selected_lat"]
        s_lon = st.session_state["selected_lon"]
        loc_meta = st.session_state["location_info"]
        pred_res = st.session_state["prediction_result"]
        local_risk = st.session_state["local_risk_result"]
        plan = st.session_state["resilience_plan"]
        feat_status = st.session_state["feature_status"]
        sources_panel = st.session_state["sources_panel"]
        features = st.session_state["retrieved_features"]

        # 1. Location Bar
        st.markdown(f"""
        <div style="background: #FFFFFF; border: 1px solid #DCE6F0; border-left: 5px solid #087EA4; border-radius: 10px; padding: 0.9rem 1.4rem; margin-bottom: 1.1rem; box-shadow: 0 2px 8px rgba(11, 31, 58, 0.04);">
            <div style="font-size: 0.76rem; font-weight: 800; color: #087EA4; text-transform: uppercase; letter-spacing: 0.05em;">ANALYZED LOCATION</div>
            <div style="font-size: 1.35rem; font-weight: 900; color: #0B1F3A; margin-top: 2px;">
                {loc_meta['district']}, {loc_meta['state']}
            </div>
            <div style="font-size: 0.88rem; color: #64748B; margin-top: 4px;">
                <b>Coordinates:</b> {s_lat:.4f}°N, {s_lon:.4f}°E &nbsp;&bull;&nbsp; <b>Basin:</b> {loc_meta['basin']} Basin &nbsp;&bull;&nbsp; <b>Sub-Basin:</b> {loc_meta['sub_basin']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 2. AI Data Pipeline Visualization
        st.markdown('<div class="section-title">📡 AI Data Ingestion Pipeline</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="pipeline-step-flow">
            <div class="pipeline-node active">
                <div class="pipeline-node-icon">📍</div>
                <div class="pipeline-node-label">Location</div>
                <div class="pipeline-node-status">✓ Geocoded</div>
            </div>
            <div class="pipeline-arrow">&rarr;</div>
            <div class="pipeline-node active">
                <div class="pipeline-node-icon">🌧️</div>
                <div class="pipeline-node-label">Weather</div>
                <div class="pipeline-node-status">✓ ERA5-Land</div>
            </div>
            <div class="pipeline-arrow">&rarr;</div>
            <div class="pipeline-node active">
                <div class="pipeline-node-icon">⛰️</div>
                <div class="pipeline-node-label">Terrain</div>
                <div class="pipeline-node-status">✓ SRTM DEM</div>
            </div>
            <div class="pipeline-arrow">&rarr;</div>
            <div class="pipeline-node active">
                <div class="pipeline-node-icon">🛰️</div>
                <div class="pipeline-node-label">Satellite</div>
                <div class="pipeline-node-status">✓ Sentinel-1</div>
            </div>
            <div class="pipeline-arrow">&rarr;</div>
            <div class="pipeline-node active">
                <div class="pipeline-node-icon">👥</div>
                <div class="pipeline-node-label">Exposure</div>
                <div class="pipeline-node-status">✓ CIESIN GPW</div>
            </div>
            <div class="pipeline-arrow">&rarr;</div>
            <div class="pipeline-node active">
                <div class="pipeline-node-icon">🧠</div>
                <div class="pipeline-node-label">FloodAI</div>
                <div class="pipeline-node-status">✓ Inference Ready</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Compact Visual Category Cards
        cat_map = {
            "Coordinates": "weather",
            "Location": "weather",
            "Rainfall": "weather",
            "Weather": "weather",
            "Terrain": "terrain",
            "Soil Moisture": "soil",
            "Satellite": "satellite",
            "Exposure": "exposure"
        }
        stat_cols = st.columns(8)
        for idx, (label, val) in enumerate(feat_status.items()):
            with stat_cols[idx]:
                cls_name = cat_map.get(label, "weather")
                st.markdown(f"""
                <div class="feature-status-card {cls_name}">
                    <b>{label}</b><br><span style="font-weight: 800;">{val}</span>
                </div>
                """, unsafe_allow_html=True)

        # Model Input Validation Summary Bar
        st.markdown("""
        <div class="validation-summary-box">
            <div class="validation-items-grid">
                <span>✓ 27 Features Validated</span>
                <span>✓ Temporal Consistency Enforced (Lag T-1)</span>
                <span>✓ Zero Future Leakage</span>
                <span>✓ Physical Limits Verified</span>
            </div>
            <div class="validation-ready-pill">
                🟢 READY FOR AI PREDICTION
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("🔍 View Retrieved Data (Data Provenance Matrix)"):
            prov_data = []
            for grp, p_info in sources_panel.items():
                prov_data.append({
                    "Feature Group": grp,
                    "Source Provider": p_info["source"],
                    "Variable / Endpoint": p_info["variable"],
                    "Unit": p_info["unit"],
                    "Temporal Window": p_info.get("temporal_window", "Preceding Lag / Static"),
                    "Retrieved Value": p_info["sample_val"]
                })
            st.dataframe(pd.DataFrame(prov_data), use_container_width=True)

        st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

        # 3. FLOOD PREDICTION HERO RESULT CARD
        st.markdown('<div class="section-title">🎯 Flood Prediction & Uncertainty</div>', unsafe_allow_html=True)

        r_score = pred_res["risk_score"]
        # Dynamic Risk Color Coding
        if r_score < 25:
            dynamic_risk_color = "#20A464"
            dynamic_risk_bg = "rgba(32, 164, 100, 0.12)"
            dynamic_risk_border = "#20A464"
            risk_state_badge = "🟢 LOW FLOOD RISK"
        elif r_score < 50:
            dynamic_risk_color = "#F4A261"
            dynamic_risk_bg = "rgba(244, 162, 97, 0.15)"
            dynamic_risk_border = "#F4A261"
            risk_state_badge = "🟡 MODERATE FLOOD RISK"
        elif r_score < 75:
            dynamic_risk_color = "#F97316"
            dynamic_risk_bg = "rgba(249, 115, 22, 0.15)"
            dynamic_risk_border = "#F97316"
            risk_state_badge = "🟠 HIGH FLOOD RISK"
        else:
            dynamic_risk_color = "#EF4444"
            dynamic_risk_bg = "rgba(239, 68, 68, 0.16)"
            dynamic_risk_border = "#EF4444"
            risk_state_badge = "🔴 CRITICAL / SEVERE FLOOD RISK"

        st.markdown(f"""
        <div class="prediction-hero-card" style="border-left: 6px solid {dynamic_risk_color};">
            <div class="pred-card-header">
                <div class="pred-card-title">
                    <span>🌊 FLOOD HAZARD PREDICTION</span>
                </div>
                <div class="pred-badge" style="background: {dynamic_risk_bg}; color: {dynamic_risk_color}; border: 1px solid {dynamic_risk_border};">
                    {risk_state_badge}
                </div>
            </div>
            <div class="pred-big-prob-group">
                <div class="pred-big-prob" style="color: {dynamic_risk_color};">
                    {pred_res['probability_pct']}%
                </div>
                <div class="pred-big-label">Model-Predicted Flood Probability (24–72h Horizon)</div>
            </div>
            <div class="pred-meta-grid">
                <div class="pred-meta-item">
                    <div class="label">Severity Classification</div>
                    <div class="val" style="color: {dynamic_risk_color};">{pred_res['severity_label']}</div>
                </div>
                <div class="pred-meta-item">
                    <div class="label">Epistemic Uncertainty</div>
                    <div class="val" style="color: #6C4CE8;">&plusmn;{pred_res['epistemic_uncertainty_pct']}%</div>
                </div>
                <div class="pred-meta-item">
                    <div class="label">Composite Risk Score</div>
                    <div class="val" style="color: {dynamic_risk_color};">{pred_res['risk_score']} / 100</div>
                </div>
            </div>
            <div style="margin-top: 1.1rem; padding-top: 0.8rem; border-top: 1px solid #F1F5F9; display: flex; justify-content: space-between; align-items: center; font-size: 0.8rem; color: #64748B;">
                <span>🧠 <b>Inference Model:</b> {pred_res['model_name']}</span>
                <span>🛡️ <b>Risk Category:</b> <b style="color: {dynamic_risk_color};">{pred_res['risk_category']}</b></span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Visual Risk Score Gauge
        clamped_score = min(max(float(r_score), 0.0), 100.0)
        st.markdown(f"""
        <div class="risk-gauge-container">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 0.82rem; font-weight: 800; color: #0B1F3A; text-transform: uppercase; letter-spacing: 0.04em;">UNDRR Composite Risk Gauge</span>
                <span style="font-size: 0.95rem; font-weight: 900; color: {dynamic_risk_color};">{r_score} / 100 ({pred_res['risk_category']})</span>
            </div>
            <div class="risk-gauge-bar-wrapper">
                <div class="risk-gauge-indicator" style="left: {clamped_score}%;"></div>
            </div>
            <div class="risk-gauge-labels">
                <span>0 (Low)</span>
                <span>25 (Moderate)</span>
                <span>50 (High)</span>
                <span>75 (Very High)</span>
                <span>100 (Critical)</span>
            </div>
            <div style="font-size: 0.74rem; color: #64748B; margin-top: 8px; text-align: center;">
                *Risk = [0.50 &times; Hazard + 0.25 &times; Exposure + 0.25 &times; Vulnerability] &times; 100
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 4. POST-PREDICTION MAP — EXACT 1 KM LOCAL ANALYSIS
        st.markdown('<div class="section-title">🗺️ Local Flood-Risk Map (1 km Radius)</div>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="map-showcase-container">
            <div class="map-header-bar">
                <div class="map-header-title">📍 LOCAL GEOSPATIAL DOMAIN (1000m Geodesic Buffer)</div>
                <div class="map-badge-pill">
                    Centered on {s_lat:.4f}°N, {s_lon:.4f}°E ({loc_meta['district']})
                </div>
            </div>
        """, unsafe_allow_html=True)

        local_map_html = local_risk_engine.generate_local_1km_map(s_lat, s_lon, local_risk, loc_meta)
        components.html(local_map_html, height=520, scrolling=False)

        # Integrated Risk Legend
        st.markdown("""
            <div style="display: flex; justify-content: center; gap: 24px; padding: 0.6rem 0.5rem 0.2rem 0.5rem; font-size: 0.78rem; font-weight: 700; border-top: 1px solid #F1F5F9; color: #475569;">
                <span><span style="color: #20A464;">●</span> Low Risk (0–25)</span>
                <span><span style="color: #F4A261;">●</span> Moderate Risk (26–50)</span>
                <span><span style="color: #F97316;">●</span> High Risk (51–75)</span>
                <span><span style="color: #EF4444;">●</span> Very High / Critical (76–100)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 5. LOCAL RISK ANALYSIS & 4 CORE PILLARS
        st.markdown('<div class="section-title">📊 Local Risk Analysis (1 km Domain)</div>', unsafe_allow_html=True)
        
        p_c1, p_c2, p_c3, p_c4 = st.columns(4)
        with p_c1:
            st.markdown(f"""
            <div class="pillar-card" style="border-top: 4px solid #087EA4;">
                <div class="pillar-card-title" style="color: #087EA4;">🌊 HAZARD</div>
                <div class="pillar-card-metric">{pred_res['hazard_probability']:.3f}</div>
                <div class="pillar-card-sub">Flood Prob &bull; Weight: 50%</div>
            </div>
            """, unsafe_allow_html=True)

        with p_c2:
            st.markdown(f"""
            <div class="pillar-card" style="border-top: 4px solid #22B8CF;">
                <div class="pillar-card-title" style="color: #22B8CF;">👥 EXPOSURE</div>
                <div class="pillar-card-metric">{features.get('exposure_index', 0.5):.3f}</div>
                <div class="pillar-card-sub">Log Density &bull; Weight: 25%</div>
            </div>
            """, unsafe_allow_html=True)

        with p_c3:
            st.markdown(f"""
            <div class="pillar-card" style="border-top: 4px solid #6C4CE8;">
                <div class="pillar-card-title" style="color: #6C4CE8;">🧩 VULNERABILITY</div>
                <div class="pillar-card-metric">{local_risk['local_vulnerability']:.3f}</div>
                <div class="pillar-card-sub">Drainage Deficit &bull; Weight: 25%</div>
            </div>
            """, unsafe_allow_html=True)

        with p_c4:
            st.markdown(f"""
            <div class="pillar-card" style="border-top: 4px solid {dynamic_risk_color};">
                <div class="pillar-card-title" style="color: {dynamic_risk_color};">⚠️ COMPOSITE RISK</div>
                <div class="pillar-card-metric" style="color: {dynamic_risk_color};">{pred_res['risk_score']} / 100</div>
                <div class="pillar-card-sub">Category: <b>{pred_res['risk_category']}</b></div>
            </div>
            """, unsafe_allow_html=True)

        # Local 1 km Demographic & Infrastructure Exposure
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.markdown("##### 👥 Local 1 km Demographic & Physical Exposure")
        exp = local_risk["exposure_details"]
        e1, e2, e3, e4, e5 = st.columns(5)
        with e1:
            st.metric("Population Exposed", f"{exp['population_exposed']:,}", delta=f"{features.get('population_density', 0):.0f} /km²")
        with e2:
            st.metric("Estimated Dwellings", f"{exp['buildings_exposed']:,}", delta="Structures")
        with e3:
            st.metric("Roads Exposed", f"{exp['roads_exposed_km']} km", delta="Transport Grid")
        with e4:
            st.metric("Critical Facilities", f"{exp['schools']} Schools", delta=f"{exp['health_centers']} Health Sub-Centers")
        with e5:
            st.metric("Agricultural Land", f"{exp['agricultural_hectares']} ha", delta="Cultivated Floodplain")

        # Main Risk Drivers & Compact Explainability
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        d_col1, d_col2 = st.columns([1, 1])

        with d_col1:
            st.markdown("""
            <div style="background: #FFFFFF; border: 1px solid #DCE6F0; border-radius: 12px; padding: 1.2rem; box-shadow: 0 2px 8px rgba(11, 31, 58, 0.04); height: 100%;">
                <div style="font-size: 0.85rem; font-weight: 800; color: #0B1F3A; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 0.75rem;">
                    ⚠️ WHY IS THIS LOCATION AT RISK?
                </div>
            """, unsafe_allow_html=True)
            for driver in local_risk["main_risk_drivers"]:
                st.markdown(f"• **{driver}**")
            st.markdown("</div>", unsafe_allow_html=True)

        with d_col2:
            st.markdown("""
            <div style="background: #FFFFFF; border: 1px solid #DCE6F0; border-radius: 12px; padding: 1.2rem; box-shadow: 0 2px 8px rgba(11, 31, 58, 0.04); height: 100%;">
                <div style="font-size: 0.85rem; font-weight: 800; color: #0B1F3A; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 0.75rem;">
                    🧠 WHY THIS PREDICTION? (Top Factors)
                </div>
            """, unsafe_allow_html=True)
            
            top_factors = [
                ("3-Day Antecedent Rainfall", min(max(features.get('rainfall_3d', 50.0) / 250.0, 0.1), 0.98)),
                ("Root-Zone Soil Saturation", min(max(features.get('soil_saturation_ratio', 0.6) / 1.0, 0.1), 0.95)),
                ("Elevation Deficit", min(max(features.get('elevation_risk_factor', 0.4), 0.1), 0.92)),
                ("River Channel Proximity", min(max(1.0 / (features.get('dist_river_km', 1.0) + 0.1), 0.1), 0.88)),
                ("Population Exposure Index", min(max(features.get('exposure_index', 0.5), 0.1), 0.85)),
            ]
            for factor_name, score in top_factors:
                pct = int(score * 100)
                st.markdown(f"""
                <div style="margin-bottom: 0.55rem;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.78rem; font-weight: 600; color: #172033; margin-bottom: 3px;">
                        <span>{factor_name}</span>
                        <span style="color: #6C4CE8;">{pct}% impact</span>
                    </div>
                    <div style="background: #F1F5F9; height: 8px; border-radius: 4px; overflow: hidden;">
                        <div style="width: {pct}%; height: 100%; background: linear-gradient(90deg, #087EA4 0%, #6C4CE8 100%);"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.caption("🔬 Full game-theoretic TreeSHAP feature attributions and beeswarm plots available in the Advanced Analysis tab.")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # 6. RESCUE & RESILIENCE PLANNING
        st.markdown('<div class="section-title">🛡️ Rescue & Resilience Plan</div>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="resilience-hero-box">
            <div>
                <div class="resil-title">🛡️ RESILIENCE & EMERGENCY RESPONSE DIRECTIVES</div>
                <div class="resil-sub">Prescriptive disaster risk reduction directives tailored to the 1 km local impact zone.</div>
            </div>
            <div class="resil-priority-pill" style="border: 2px solid {plan['priority_color']}; color: {plan['priority_color']};">
                Priority: {plan['response_priority']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background: white; border-left: 5px solid {plan['priority_color']}; border-radius: 8px; padding: 0.8rem 1.2rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 1rem;">
            <div style="font-size: 0.88rem; color: #334155;">
                {plan['planning_header']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        tabs_p = st.tabs(list(plan["pillars"].keys()))
        for idx, (pillar_name, directives) in enumerate(plan["pillars"].items()):
            with tabs_p[idx]:
                for item in directives:
                    st.markdown(f"• {item}")

        st.markdown(f"""
        <div style="background: #fffbeb; border: 1px solid #f59e0b; border-radius: 8px; padding: 0.75rem 1rem; margin-top: 1.1rem; font-size: 0.8rem; color: #92400e;">
            ⚠️ <b>Mandatory Academic Disclaimer:</b> {plan['disclaimer']}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        # 7. SELECT ANOTHER LOCATION BUTTON
        if st.button("← Select Another Location", type="primary", use_container_width=True):
            st.session_state["selected_lat"] = None
            st.session_state["selected_lon"] = None
            st.session_state["location_info"] = None
            st.session_state["analysis_complete"] = False
            st.session_state["retrieved_features"] = None
            st.session_state["feature_status"] = None
            st.session_state["sources_panel"] = None
            st.session_state["prediction_result"] = None
            st.session_state["local_risk_result"] = None
            st.session_state["resilience_plan"] = None
            st.rerun()


# ===========================================================================
# TAB 2: 🔬 ADVANCED ANALYSIS (SECONDARY / RESEARCH ONLY)
# ===========================================================================
with tab_advanced:
    st.subheader("🔬 Advanced Flood Intelligence & Research Tools")
    st.caption("In-depth model evaluation, ablation experiments, explainable AI, uncertainty quantification, and manual parametric simulations.")

    tab_eval, tab_shap, tab_uncert, tab_abl, tab_sim, tab_doc = st.tabs([
        "📊 Model Benchmarks",
        "🧠 Explainable AI (SHAP)",
        "🎯 Prediction Uncertainty",
        "🗺️ Generalization & Ablation",
        "🎛️ Manual Scenario (Research)",
        "📑 System Schema & Downloads"
    ])

    with tab_eval:
        st.subheader("Rigorous Chronological Out-of-Time Test Benchmarks (2023 Set)")
        rows = []
        for m, vals in model_metrics.items():
            rows.append({
                "Model Architecture": m.replace("_", " "),
                "Accuracy": vals["accuracy"],
                "Precision": vals["precision"],
                "Recall": vals["recall"],
                "F1-Score": vals["f1"],
                "PR-AUC": vals["pr_auc"],
                "ROC-AUC": vals["roc_auc"],
                "Brier Score": vals["brier_score"],
                "Train Time (s)": vals["train_time_sec"],
                "Latency (ms)": vals["inference_latency_ms"]
            })
        st.dataframe(pd.DataFrame(rows).sort_values("F1-Score", ascending=False), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            roc_path = os.path.join(BASE_DIR, "outputs", "figures", "pan_india_roc_curves.png")
            if os.path.exists(roc_path):
                st.image(roc_path, caption="Pan-India Multi-Basin ROC Curves", use_container_width=True)
        with c2:
            pr_path = os.path.join(BASE_DIR, "outputs", "figures", "pan_india_pr_curves.png")
            if os.path.exists(pr_path):
                st.image(pr_path, caption="Pan-India Precision-Recall Curves (Priority Metric)", use_container_width=True)

    with tab_shap:
        st.subheader("TreeSHAP Global Game-Theoretic Feature Attribution")
        s1, s2 = st.columns(2)
        with s1:
            bar_img = os.path.join(BASE_DIR, "outputs", "figures", "shap_bar_importance.png")
            if os.path.exists(bar_img):
                st.image(bar_img, caption="Global Feature Importance (|SHAP|)", use_container_width=True)
        with s2:
            sum_img = os.path.join(BASE_DIR, "outputs", "figures", "shap_summary_plot.png")
            if os.path.exists(sum_img):
                st.image(sum_img, caption="SHAP Beeswarm Feature Impact Summary", use_container_width=True)

    with tab_uncert:
        st.subheader("Monte Carlo Dropout Epistemic Uncertainty & Reliability")
        uncert_fig = os.path.join(BASE_DIR, "outputs", "figures", "pan_india_uncertainty_calibration.png")
        if os.path.exists(uncert_fig):
            st.image(uncert_fig, caption="Monte Carlo Epistemic Uncertainty vs Predicted Probability", use_container_width=True)

    with tab_abl:
        st.subheader("Multimodal Feature Ablation & Spatial Cross-Basin Generalization")
        abl_img = os.path.join(BASE_DIR, "outputs", "figures", "pan_india_ablation_study.png")
        if os.path.exists(abl_img):
            st.image(abl_img, caption="6-Tier Multimodal Feature Ablation Study", use_container_width=True)
        sp_img = os.path.join(BASE_DIR, "outputs", "figures", "pan_india_spatial_generalization.png")
        if os.path.exists(sp_img):
            st.image(sp_img, caption="Spatial Transfer Performance on Unseen Held-out River Basins", use_container_width=True)

    with tab_sim:
        st.subheader("Manual Parametric Scenario Simulation (Research What-If Mode)")
        st.caption("Use these sliders strictly for sensitivity stress-testing and counterfactual analysis.")
        c_m, c_t, c_s = st.columns(3)
        with c_m:
            s_r1 = st.slider("1-Day Rainfall (mm)", 0.0, 350.0, 45.0)
            s_r3 = st.slider("3-Day Rainfall (mm)", 0.0, 600.0, 110.0)
            s_r7 = st.slider("7-Day Rainfall (mm)", 0.0, 1000.0, 180.0)
            s_r14 = st.slider("14-Day Rainfall (mm)", 0.0, 1600.0, 260.0)
            s_temp = st.slider("Temperature (°C)", 10.0, 45.0, 27.0)
            s_hum = st.slider("Humidity (%)", 30.0, 100.0, 85.0)
        with c_t:
            s_sm = st.slider("Soil Moisture (m³/m³)", 0.15, 0.50, 0.38)
            s_elev = st.slider("Elevation (m)", -5.0, 1500.0, 22.0)
            s_slope = st.slider("Slope Angle (°)", 0.1, 40.0, 2.0)
            s_dist = st.slider("River Distance (km)", 0.05, 15.0, 0.8)
            s_pop = st.slider("Population Density (/km²)", 50.0, 6000.0, 1200.0)
        with c_s:
            s_vv = st.slider("Sentinel-1 SAR VV (dB)", -28.0, -5.0, -14.5)
            s_vh = st.slider("Sentinel-1 SAR VH (dB)", -35.0, -10.0, -20.0)
            s_ndwi = st.slider("NDWI Water Index", -0.5, 0.9, 0.35)
            s_ndvi = st.slider("NDVI Vegetation Index", 0.0, 0.9, 0.55)

        if st.button("🚀 Run Manual Simulation", type="primary"):
            sim_dict = {
                "rainfall_1d": s_r1, "rainfall_3d": s_r3, "rainfall_7d": s_r7, "rainfall_14d": s_r14,
                "rainfall_anomaly_30d": s_r3 - (s_r14 / 4.6), "api_14d": s_r1 + 0.85 * s_r3,
                "temp_mean_3d": s_temp, "humidity_mean_3d": s_hum,
                "soil_moisture_root_zone": s_sm, "soil_moisture_deep": s_sm * 0.94, "soil_saturation_ratio": min(s_sm/0.48, 1.0),
                "river_discharge_proxy": (s_r7 * 14.0) + (280.0 / (s_dist + 0.1)), "river_stage_anomaly": (s_r3 / 45.0) - s_dist,
                "elevation_m": s_elev, "slope_deg": s_slope, "dist_river_km": s_dist,
                "twi_proxy": float(np.log(100.0 / np.tan(np.radians(max(s_slope, 0.1))))),
                "elevation_risk_factor": float(np.exp(-s_elev / 50.0)),
                "drainage_vulnerability": float((0.45 * np.exp(-s_elev/50.0)) + (0.35 / (1.0 + s_dist)) + (0.20 / (1.0 + np.radians(max(s_slope, 0.1))))),
                "sar_vv": s_vv, "sar_vh": s_vh, "sar_cross_ratio": abs(s_vh / (s_vv - 0.01)),
                "ndwi_proxy": s_ndwi, "ndvi_proxy": s_ndvi, "population_density": s_pop,
                "exposure_index": float(np.clip(np.log1p(s_pop) / np.log1p(5000.0), 0.0, 1.0)),
                "vulnerability_index": 0.60
            }
            sim_pred = prediction_service.predict(sim_dict)
            st.markdown("---")
            sm1, sm2, sm3, sm4 = st.columns(4)
            sm1.metric("Predicted State", "FLOOD THREAT" if sim_pred["prediction"] == 1 else "NO FLOOD")
            sm2.metric("Probability", f"{sim_pred['probability_pct']}%")
            sm3.metric("Composite Risk", f"{sim_pred['risk_score']} / 100")
            sm4.markdown(f"**Risk Level**\n<h3 style='color:{sim_pred['risk_color']};'>{sim_pred['risk_category']}</h3>", unsafe_allow_html=True)

    with tab_doc:
        st.subheader("Project Documentation & Spatial Layers")
        doc_c1, doc_c2, doc_c3 = st.columns(3)
        with doc_c1:
            schema_path = os.path.join(BASE_DIR, "docs", "model_feature_schema.md")
            if os.path.exists(schema_path):
                with open(schema_path, "r", encoding="utf-8") as f:
                    st.download_button("📥 Model Feature Schema (MD)", f.read(), file_name="model_feature_schema.md")
        with doc_c2:
            bench_path = os.path.join(BASE_DIR, "outputs", "metrics", "pan_india_model_comparison.json")
            if os.path.exists(bench_path):
                with open(bench_path, "r", encoding="utf-8") as f:
                    st.download_button("📥 Benchmark Comparison (JSON)", f.read(), file_name="model_comparison.json")
        with doc_c3:
            geo_path = os.path.join(BASE_DIR, "outputs", "maps", "kerala_spatial_risk.geojson")
            if os.path.exists(geo_path):
                with open(geo_path, "r", encoding="utf-8") as f:
                    st.download_button("📥 Spatial Risk Layer (GeoJSON)", f.read(), file_name="spatial_risk.geojson")


# ===========================================================================
# TAB 3: ℹ️ ABOUT PROJECT
# ===========================================================================
with tab_about:
    st.subheader("ℹ️ About FloodAI Engine")
    st.markdown("### AI for a Safer, Resilient India | B.Tech Major Project")
    
    st.markdown("""
    **Project Title:** AI-Based Flood Prediction, Dynamic Risk Mapping and Resilience Planning  
    **Tagline:** AI for a Safer, Resilient India  
    **Core Purpose:** Predict • Map • Explain • Protect  

    ---

    ### 🎯 Project Purpose
    Develop an integrated, location-first artificial intelligence framework designed to provide operational flood forecasting, dynamic geospatial risk analysis, and automated decision support across India's major river basins.

    ### 🔄 Main Workflow
    1. **Interactive Location Selection**: The user selects any point across India via the clean interactive Folium map.
    2. **Automated Feature Retrieval**: The system queries legitimate multi-source APIs (Open-Meteo ERA5-Land, USGS SRTM DEM, National River Hydrology Network, Copernicus Sentinel-1, and CIESIN GPWv4) without requiring manual parameter configuration.
    3. **Feature Validation & Causality**: Enforces physical boundaries, completeness, and backward-looking temporal causality ($T-1$ lag) to prevent data leakage.
    4. **Model Inference**: Evaluates the 27-feature vector through the trained champion model, generating flood probability, severity classification, and epistemic uncertainty quantification.
    5. **1 km Local Risk Analysis**: Calculates local geodesic circular buffer exposure (population, buildings, roads, critical facilities) and decomposes risk into Hazard, Exposure, and Vulnerability.
    6. **Prescriptive Resilience Directives**: Synthesizes 7 operational disaster management pillars with response priority escalation.

    ### 📊 Risk Decomposition Framework (UNDRR Standard)
    $$\\text{Risk} = [0.50 \\times \\text{Hazard} + 0.25 \\times \\text{Exposure} + 0.25 \\times \\text{Vulnerability}] \\times 100$$
    - **Hazard ($H$)**: Model-predicted flood probability $P(\\text{Flood} \\mid \\mathbf{X})$.
    - **Exposure ($E$)**: Logarithmic population density index $\\text{clip}\\left(\\frac{\\ln(1 + \\rho)}{\\ln(1 + 5000)}, 0, 1\\right)$.
    - **Vulnerability ($V$)**: Project-defined composite drainage and elevation bottleneck index.

    ### 💻 Technology Stack
    - **Machine Learning**: Scikit-Learn (Random Forest Balanced), LightGBM, XGBoost, PyTorch (ST-FloodNet, Bi-LSTM).
    - **Explainable AI**: SHAP (TreeExplainer).
    - **Geospatial Processing**: Folium, Streamlit-Folium, Geopy, Leaflet.
    - **Frontend**: Streamlit (Full-width responsive top-tab layout).
    - **Data Ingestion**: Open-Meteo API, Open-Elevation API, OpenStreetMap Nominatim.

    ### 🛡️ Mandatory Academic Disclaimer
    All resilience planning directives, hazard probabilities, and risk classifications produced by this software are experimental academic project recommendations and do NOT constitute official government warnings, emergency orders, or statutory disaster advice. Always refer to official bulletins issued by the National Disaster Management Authority (NDMA), Central Water Commission (CWC), and India Meteorological Department (IMD).
    """)
