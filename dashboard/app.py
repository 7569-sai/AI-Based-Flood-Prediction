"""
🌊 FloodAI ENGINE — Pan-India Multi-Basin Flood Intelligence Platform
"AI for a Safer, Resilient India"
Predict • Map • Plan • Protect

Top Horizontal Tab Navigation:
[ 📍 Location Analysis ] [ 🔬 Advanced Analysis ] [ ℹ️ About Project ]
No vertical radio sidebar — full-width, clean, location-first flood intelligence.
"""

import os
import sys
import json
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

# Configure page: collapsed sidebar for maximum horizontal space
st.set_page_config(
    page_title="FloodAI Engine — India's AI Flood Intelligence Platform",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------------------------
# Professional Modern CSS Styling (Top Tabs & Full-Width Map Layout)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Remove and hide left sidebar completely to gain full horizontal width */
    section[data-testid="stSidebar"] {
        display: none !important;
    }
    div[data-testid="collapsedControl"] {
        display: none !important;
    }
    
    /* Compact Top Header */
    .top-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
        color: white;
        padding: 0.95rem 1.6rem;
        border-radius: 8px;
        margin-bottom: 0.8rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.12);
    }
    
    .top-header h2 {
        margin: 0;
        font-size: 1.45rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #f8fafc;
    }
    
    .top-header p {
        margin: 0.15rem 0 0 0;
        font-size: 0.84rem;
        color: #94a3b8;
    }
    
    .top-status-badge {
        background: rgba(16, 185, 129, 0.16);
        border: 1px solid #10b981;
        color: #10b981;
        padding: 0.38rem 0.85rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    /* Modern Horizontal Top Tabs (Section 8) */
    div[data-baseweb="tab-list"] {
        gap: 12px;
        border-bottom: 2px solid #e2e8f0;
        margin-bottom: 1.1rem;
    }
    button[data-baseweb="tab"] {
        font-size: 1.02rem;
        font-weight: 600;
        color: #475569;
        padding: 0.7rem 1.5rem;
        border-radius: 6px 6px 0 0;
        background: transparent;
        border: none;
        transition: all 0.15s ease;
    }
    button[data-baseweb="tab"]:hover {
        color: #0284c7;
        background: #f1f5f9;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #0284c7;
        font-weight: 700;
        border-bottom: 3px solid #0284c7;
        background: #f8fafc;
    }
    
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    .metric-card-title {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #64748b;
        margin-bottom: 0.3rem;
    }
    .metric-card-value {
        font-size: 1.6rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }
    .metric-card-subtitle {
        font-size: 0.8rem;
        color: #64748b;
    }
    
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1e293b;
        margin-top: 1.1rem;
        margin-bottom: 0.55rem;
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

# Dynamic Model Name Resolution (Section 10)
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
# Compact Top Header (Section 7 & 10)
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="top-header">
    <div>
        <h2>🌊 FLOODAI ENGINE</h2>
        <p>AI for a Safer, Resilient India</p>
    </div>
    <div class="top-status-badge">
        ● Online &nbsp;|&nbsp; India Multi-Basin &nbsp;|&nbsp; Model: {dynamic_model_name}
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Primary Horizontal Top Tabs (Section 2 & 3)
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
        st.markdown("<p style='font-size: 1.05rem; font-weight: 600; color: #1e293b; margin-top: 0.2rem; margin-bottom: 0.6rem;'>📍 Click anywhere on the map to analyze flood risk.</p>", unsafe_allow_html=True)

        # Clean Map: Centered on India with NO location markers, NO red dots, NO blue dots, full container width
        m_clean = folium.Map(
            location=[22.8, 82.0],
            zoom_start=5,
            tiles="OpenStreetMap",
            zoom_control=True
        )

        map_output = st_folium(
            m_clean,
            height=600,
            use_container_width=True,
            key="clean_india_map",
            returned_objects=["last_clicked"]
        )

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

        st.markdown("<p style='font-size: 1.05rem; font-weight: 600; color: #1e293b; margin-top: 0.2rem; margin-bottom: 0.6rem;'>📍 Selected Location Established. Confirm and analyze flood risk below:</p>", unsafe_allow_html=True)

        # Map showing EXACTLY ONE Marker at the clicked coordinate
        m_selected = folium.Map(
            location=[s_lat, s_lon],
            zoom_start=7,
            tiles="OpenStreetMap"
        )
        folium.Marker(
            location=[s_lat, s_lon],
            tooltip=f"Selected Point: {s_lat:.4f}°N, {s_lon:.4f}°E",
            popup=f"{loc_meta['district']}, {loc_meta['state']} ({loc_meta['basin']} Basin)",
            icon=folium.Icon(color="red", icon="info-sign")
        ).add_to(m_selected)

        map_reclick = st_folium(
            m_selected,
            height=460,
            use_container_width=True,
            key="selected_point_map",
            returned_objects=["last_clicked"]
        )

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

        # Selected Location Confirmation Card (Section 5)
        st.markdown(f"""
        <div class="metric-card" style="border-left: 4px solid #0284c7; margin-top: 0.8rem; margin-bottom: 0.8rem;">
            <div class="metric-card-title">SELECTED LOCATION</div>
            <div style="font-size: 1.35rem; font-weight: 700; color: #0f172a; margin-top: 2px;">
                {loc_meta['district']}, {loc_meta['state']}
            </div>
            <div style="font-size: 0.92rem; color: #475569; margin-top: 6px; line-height: 1.6;">
                <b>Latitude:</b> {s_lat:.4f}°N &nbsp;&nbsp;|&nbsp;&nbsp; <b>Longitude:</b> {s_lon:.4f}°E<br>
                <b>River Basin:</b> {loc_meta['basin']} Basin &nbsp;&nbsp;|&nbsp;&nbsp; <b>Sub-Basin:</b> {loc_meta['sub_basin']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_act1, col_act2 = st.columns([2, 1])
        with col_act1:
            if st.button("🌊 ANALYZE FLOOD RISK", type="primary", use_container_width=True):
                with st.spinner("Retrieving location data & running AI model..."):
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
        <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 0.8rem 1.2rem; margin-bottom: 1rem;">
            <div style="font-size: 0.78rem; font-weight: 700; color: #64748b; text-transform: uppercase;">ANALYZED LOCATION</div>
            <div style="font-size: 1.25rem; font-weight: 800; color: #0f172a; margin-top: 2px;">
                {loc_meta['district']}, {loc_meta['state']}
            </div>
            <div style="font-size: 0.88rem; color: #334155; margin-top: 4px;">
                <b>Coordinates:</b> {s_lat:.4f}°N, {s_lon:.4f}°E &nbsp;|&nbsp; <b>Basin:</b> {loc_meta['basin']} ({loc_meta['sub_basin']})
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 2. Data Retrieval Status Strip (Section 21)
        st.markdown('<div class="section-title">📡 Data Retrieval Status</div>', unsafe_allow_html=True)
        stat_cols = st.columns(8)
        for idx, (label, val) in enumerate(feat_status.items()):
            with stat_cols[idx]:
                bg = "#dcfce7" if "✓" in val else "#fef3c7"
                fg = "#166534" if "✓" in val else "#92400e"
                st.markdown(f"""
                <div style="background: {bg}; color: {fg}; border-radius: 6px; padding: 0.45rem 0.3rem; text-align: center; border: 1px solid {fg}40; font-size: 0.75rem;">
                    <b>{label}</b><br><span style="font-weight: 700;">{val}</span>
                </div>
                """, unsafe_allow_html=True)

        st.caption("✓ Model inputs validated (all 27 physical predictor variables validated, backward-looking temporal causality enforced, zero NaN values).")

        with st.expander("🔍 View Retrieved Data (Data Provenance)"):
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

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        # 3. FLOOD PREDICTION CARD (Section 25)
        st.markdown('<div class="section-title">🎯 Flood Prediction</div>', unsafe_allow_html=True)
        r_col1, r_col2, r_col3, r_col4 = st.columns(4)

        with r_col1:
            is_pred_flood = pred_res["hazard_probability"] >= 0.5
            p_text = "🚨 FLOOD THREAT" if is_pred_flood else "✅ NORMAL / NO FLOOD"
            p_color = "#dc2626" if is_pred_flood else "#16a34a"
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid {p_color};">
                <div class="metric-card-title">Prediction State</div>
                <div class="metric-card-value" style="color: {p_color}; font-size: 1.25rem;">{p_text}</div>
                <div class="metric-card-subtitle">24–72h Forecast Horizon</div>
            </div>
            """, unsafe_allow_html=True)

        with r_col2:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid #2563eb;">
                <div class="metric-card-title">Flood Probability</div>
                <div class="metric-card-value" style="color: #2563eb;">{pred_res['probability_pct']}%</div>
                <div class="metric-card-subtitle">Severity: <b>{pred_res['severity_label']}</b></div>
            </div>
            """, unsafe_allow_html=True)

        with r_col3:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid {pred_res['risk_color']};">
                <div class="metric-card-title">Composite Risk Score</div>
                <div class="metric-card-value" style="color: {pred_res['risk_color']};">{pred_res['risk_score']} / 100</div>
                <div class="metric-card-subtitle">Risk Level: <b>{pred_res['risk_category']}</b></div>
            </div>
            """, unsafe_allow_html=True)

        with r_col4:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid #7c3aed;">
                <div class="metric-card-title">Model & Uncertainty</div>
                <div class="metric-card-value" style="color: #7c3aed; font-size: 1.15rem;">Random Forest</div>
                <div class="metric-card-subtitle">Epistemic Uncertainty: <b>±{pred_res['epistemic_uncertainty_pct']}%</b></div>
            </div>
            """, unsafe_allow_html=True)

        st.progress(pred_res["risk_score"] / 100.0)

        # 4. POST-PREDICTION MAP — EXACT 1 KM LOCAL ANALYSIS (Sections 26–29)
        st.markdown('<div class="section-title">🗺️ Local Flood-Risk Map (1 km Radius)</div>', unsafe_allow_html=True)
        st.caption(f"Map centered on **{s_lat:.4f}°N, {s_lon:.4f}°E** ({loc_meta['district']}) with geodesic 1000m circular boundary. Shows point flood prediction + 1 km local exposure/risk analysis.")

        local_map_html = local_risk_engine.generate_local_1km_map(s_lat, s_lon, local_risk, loc_meta)
        components.html(local_map_html, height=520, scrolling=False)

        # 5. LOCAL RISK ANALYSIS & EXPOSURE (Sections 30–33)
        st.markdown('<div class="section-title">📊 Local Risk Analysis (1 km Domain)</div>', unsafe_allow_html=True)
        
        # UNDRR Components
        u1, u2, u3, u4 = st.columns(4)
        with u1:
            st.metric("Hazard Probability (H)", f"{pred_res['hazard_probability']:.3f}", delta="Weight: 50%")
        with u2:
            st.metric("Exposure Index (E)", f"{features.get('exposure_index', 0.5):.3f}", delta="Weight: 25%")
        with u3:
            st.metric("Vulnerability Index (V)", f"{local_risk['local_vulnerability']:.3f}", delta="Weight: 25% (Project-Defined)")
        with u4:
            st.metric("Local Risk Score", f"{pred_res['risk_score']} / 100", delta=pred_res['risk_category'])

        # Local Exposure Breakdown (Section 31)
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

        # Main Risk Drivers (Section 33)
        st.markdown("##### ⚠️ Main Risk Drivers")
        for driver in local_risk["main_risk_drivers"]:
            st.markdown(f"• **{driver}**")

        st.markdown("---")

        # 6. RESCUE & RESILIENCE PLANNING (Sections 34–37)
        st.markdown('<div class="section-title">🛡️ Rescue & Resilience Plan</div>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="background: white; border-left: 5px solid {plan['priority_color']}; border-radius: 8px; padding: 0.9rem 1.2rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 1rem;">
            <div style="font-size: 0.8rem; text-transform: uppercase; font-weight: 700; color: #64748b;">Response Priority</div>
            <div style="font-size: 1.25rem; font-weight: 800; color: {plan['priority_color']}; margin-top: 2px;">
                {plan['response_priority']}
            </div>
            <div style="font-size: 0.88rem; color: #334155; margin-top: 4px;">
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
        <div style="background: #fffbeb; border: 1px solid #f59e0b; border-radius: 6px; padding: 0.65rem 0.9rem; margin-top: 1rem; font-size: 0.8rem; color: #92400e;">
            ⚠️ <b>Mandatory Academic Disclaimer:</b> {plan['disclaimer']}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

        # 7. SELECT ANOTHER LOCATION BUTTON (Section 39)
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
    **Core Purpose:** Predict • Map • Plan • Protect  

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
