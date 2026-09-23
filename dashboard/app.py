"""
Interactive Streamlit Dashboard for AI-Based Flood Prediction & Resilience Planning.
Comprehensive 7-page interface:
1. Home / Overview
2. Flood Hazard & Risk Prediction
3. Interactive Flood Risk Map
4. Vulnerability & Exposure Analysis
5. Explainable AI (SHAP)
6. Resilience Planning Engine
7. Model Performance & Benchmarks
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.geospatial.risk_mapper import SpatialRiskEngine, get_risk_category
from src.explainability.shap_explainer import FloodShapExplainer
from src.resilience.recommendation_engine import ResilienceEngine

st.set_page_config(
    page_title="FloodAI — Flood Prediction & Resilience",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Cached Resource Loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_engines():
    risk_eng = SpatialRiskEngine()
    shap_eng = FloodShapExplainer()
    res_eng = ResilienceEngine()
    return risk_eng, shap_eng, res_eng

risk_engine, shap_explainer, resilience_engine = load_engines()

# Load benchmark metrics
METRICS_PATH = os.path.join(BASE_DIR, "outputs", "metrics", "model_comparison.json")
with open(METRICS_PATH, "r", encoding="utf-8") as f:
    model_metrics = json.load(f)

# Sidebar Navigation
st.sidebar.title("🌊 FloodAI Engine")
st.sidebar.caption("AI-Based Flood Prediction & Resilience Planning | B.Tech Major Project")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation Menu",
    [
        "🏠 1. Home / Overview",
        "🎯 2. Flood Prediction",
        "🗺️ 3. Flood Risk Map",
        "👥 4. Vulnerability & Exposure",
        "🔍 5. Explainable AI (SHAP)",
        "📋 6. Resilience Planning",
        "📊 7. Model Performance"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Study Area**: Kerala River Basins\n\n"
    "**Champion Model**: Random Forest (Balanced)\n\n"
    "**Test F1**: 0.8571 | **Recall**: 97.8%\n\n"
    "**Data Stack**: ERA5-Land, GEE SRTM, CIESIN, Sentinel-1"
)

# ===========================================================================
# PAGE 1: HOME / OVERVIEW
# ===========================================================================
if page == "🏠 1. Home / Overview":
    st.title("AI-Based Flood Prediction and Resilience Planning")
    st.subheader("An Integrated Multimodal Geospatial AI System for Disaster Risk Reduction")
    
    st.markdown("""
    This major project implements an autonomous end-to-end artificial intelligence framework that unifies 
    **meteorological reanalysis**, **satellite remote sensing (Sentinel-1 SAR)**, **topographic digital elevation models (SRTM 30m)**, 
    and **demographic exposure data** to forecast flood hazard, evaluate compound disaster risk, explain predictions with **SHAP**, 
    and synthesize context-specific **resilience planning recommendations**.
    """)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Historical Daily Records", value="35,056", delta="6 Full Years (2018–2023)")
    with col2:
        st.metric(label="Monitoring Basins", value="8 Major Basins", delta="16 Active Stations")
    with col3:
        st.metric(label="Champion Recall Rate", value="97.83%", delta="90 / 92 Flood Events Detected")
    with col4:
        st.metric(label="Champion PR-AUC", value="0.9888", delta="Out-of-Time Test Set")

    st.markdown("---")
    
    st.markdown("### 🏛️ Three Core Scientific Objectives")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("""
        **MODULE A: Flood Prediction**
        * 24- to 72-hour lead time prediction.
        * Multi-day backward-looking antecedent hydrometeorological memory.
        * Evaluates Logistic Regression, Random Forest, Gradient Boosting, XGBoost, and PyTorch Bi-LSTM.
        """)
    with c2:
        st.success("""
        **MODULE B: Spatial Risk Mapping**
        * IPCC/UNDRR standard risk framework:
          $$\\text{Risk} = \\text{Hazard} \\times \\text{Exposure} \\times \\text{Vulnerability}$$
        * Continuous 0–100 composite flood risk score.
        * Interactive multi-layer Folium GIS visualization.
        """)
    with c3:
        st.warning("""
        **MODULE C: Resilience Planning**
        * Rule-based prescriptive mitigation engine.
        * Emergency (0–48h), Structural, Policy, and Agricultural action plans.
        * Grounded in localized geomorphic bottlenecks.
        """)

    st.markdown("---")
    st.markdown("### 🗺️ Study Area: Kerala River Basins")
    st.write(
        "Kerala features steep hydraulic gradients dropping from the Western Ghats (>2,000m) to coastal wetlands (-1m at Kuttanad). "
        "The system evaluates key disaster hotspots across the **Periyar**, **Chalakkudy**, **Pamba**, **Meenachil**, and **Bharathapuzha** river basins."
    )


# ===========================================================================
# PAGE 2: FLOOD PREDICTION
# ===========================================================================
elif page == "🎯 2. Flood Prediction":
    st.title("🎯 Flood Hazard & Risk Prediction")
    st.write("Input meteorological, soil, and terrain conditions to predict 24–72h flood occurrence and risk.")

    # Preset scenarios
    preset = st.selectbox(
        "Load Preset Historical Scenario:",
        ["Custom Scenario", "Aluva Peak Flood (August 2018)", "Kuttanad Monsoon Surge (October 2021)", "Normal Dry Season"]
    )

    # Defaults
    if preset == "Aluva Peak Flood (August 2018)":
        d_r1, d_r3, d_r7, d_r14 = 95.0, 185.0, 290.0, 420.0
        d_sm, d_elev, d_slope, d_dist, d_pop = 0.44, 10.0, 4.7, 0.3, 1359.0
    elif preset == "Kuttanad Monsoon Surge (October 2021)":
        d_r1, d_r3, d_r7, d_r14 = 70.0, 130.0, 210.0, 310.0
        d_sm, d_elev, d_slope, d_dist, d_pop = 0.45, -1.0, 3.3, 0.1, 687.0
    elif preset == "Normal Dry Season":
        d_r1, d_r3, d_r7, d_r14 = 2.0, 5.0, 12.0, 20.0
        d_sm, d_elev, d_slope, d_dist, d_pop = 0.22, 25.0, 5.0, 2.5, 800.0
    else:
        d_r1, d_r3, d_r7, d_r14 = 45.0, 95.0, 150.0, 220.0
        d_sm, d_elev, d_slope, d_dist, d_pop = 0.36, 15.0, 4.0, 0.8, 1100.0

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 🌧️ Meteorological & Hydrological Inputs")
        r1 = st.slider("1-Day Prior Rainfall (mm)", 0.0, 300.0, float(d_r1))
        r3 = st.slider("3-Day Antecedent Rainfall (mm)", 0.0, 600.0, float(d_r3))
        r7 = st.slider("7-Day Cumulative Rainfall (mm)", 0.0, 1000.0, float(d_r7))
        r14 = st.slider("14-Day Cumulative Rainfall (mm)", 0.0, 1500.0, float(d_r14))
        sm = st.slider("Root-Zone Soil Moisture (m³/m³)", 0.15, 0.52, float(d_sm))

    with col_b:
        st.markdown("#### ⛰️ Terrain & Exposure Inputs")
        elev = st.slider("Elevation (m above sea level)", -5.0, 500.0, float(d_elev))
        slope = st.slider("Slope Angle (°)", 0.1, 35.0, float(d_slope))
        dist_riv = st.slider("Distance to Primary River (km)", 0.05, 10.0, float(d_dist))
        pop = st.slider("Population Density (persons / km²)", 50.0, 5000.0, float(d_pop))

    if st.button("🚀 Run Flood Prediction & Risk Analysis", type="primary"):
        # Build feature payload
        payload = {
            "rainfall_1d": r1,
            "rainfall_3d": r3,
            "rainfall_7d": r7,
            "rainfall_14d": r14,
            "rainfall_anomaly_30d": r3 - (r14 / 4.6),
            "api_14d": r1 + (0.85 * r3),
            "temp_mean_3d": 25.0,
            "humidity_mean_3d": 88.0,
            "soil_moisture_root_zone": sm,
            "soil_moisture_deep": sm * 0.95,
            "soil_saturation_ratio": min(sm / 0.48, 1.0),
            "elevation_m": elev,
            "slope_deg": slope,
            "dist_river_km": dist_riv,
            "twi_proxy": float(np.log(100.0 / np.tan(np.radians(max(slope, 0.1))))),
            "elevation_risk_factor": float(np.exp(-elev / 50.0)),
            "sar_vv": -10.5 - (6.0 if r3 > 80 else 0.0),
            "sar_vh": -17.0 - (4.0 if r3 > 80 else 0.0),
            "sar_cross_ratio": 1.45,
            "ndwi_proxy": 0.4,
            "ndvi_proxy": 0.55,
            "population_density": pop,
            "exposure_index": float(np.clip(np.log1p(pop) / np.log1p(5000.0), 0.0, 1.0)),
            "drainage_vulnerability": float(0.45 * np.exp(-elev/50.0) + 0.35 / (1.0 + dist_riv) + 0.20 / (1.0 + np.radians(slope))),
            "vulnerability_index": 0.5
        }

        res = risk_engine.compute_point_risk(payload)

        st.markdown("---")
        st.markdown("### 📊 Prediction & Risk Outputs")
        
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Flood Occurrence", "FLOOD PREDICTED" if res["prediction"] == 1 else "NO FLOOD")
        with m2:
            st.metric("Hazard Probability", f"{res['hazard_probability']*100:.1f}%")
        with m3:
            st.metric("Composite Risk Score", f"{res['risk_score']} / 100")
        with m4:
            st.markdown(f"**Risk Category**\n<h3 style='color:{res['risk_color']};'>{res['risk_category']}</h3>", unsafe_allow_html=True)

        st.progress(res["risk_score"] / 100.0)
        
        c_h, c_e, c_v = st.columns(3)
        c_h.info(f"**Hazard (50% weight)**: {res['hazard_probability']:.2f}")
        c_e.info(f"**Exposure (25% weight)**: {res['exposure_index']:.2f}")
        c_v.info(f"**Vulnerability (25% weight)**: {res['vulnerability_index']:.2f}")


# ===========================================================================
# PAGE 3: FLOOD RISK MAP
# ===========================================================================
elif page == "🗺️ 3. Flood Risk Map":
    st.title("🗺️ Geospatial Flood Risk Map — Kerala River Basins")
    st.write("Interactive multi-layer geospatial visualization displaying real-time risk scores across monitoring stations.")

    scenario = st.selectbox(
        "Select Hydraulic Scenario:",
        ["extreme_flood_2018", "monsoon_surge", "normal"],
        format_func=lambda s: {
            "extreme_flood_2018": "Catastrophic 2018 Kerala Flood Peak (Aug 16, 2018)",
            "monsoon_surge": "Monsoon Surge Event (October 2021)",
            "normal": "Normal Wet Season Baseline (August 2023)"
        }[s]
    )

    df_res, map_path = risk_engine.generate_basin_risk_map(scenario)

    if os.path.exists(map_path):
        with open(map_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        components.html(html_content, height=550, scrolling=True)

    st.markdown("#### Station Risk Table")
    st.dataframe(
        df_res[["station_name", "basin", "district", "elevation_m", "rainfall_3d", "hazard_probability", "risk_score", "risk_category"]]
        .sort_values("risk_score", ascending=False),
        use_container_width=True
    )


# ===========================================================================
# PAGE 4: VULNERABILITY & EXPOSURE
# ===========================================================================
elif page == "👥 4. Vulnerability & Exposure":
    st.title("👥 Vulnerability & Exposure Analysis")
    st.markdown("""
    In the UNDRR framework, flood risk is not merely high water (hazard), but the exposure of human populations 
    and the physical vulnerability of the drainage landscape.
    """)

    df_feat = pd.read_csv(os.path.join(PROCESSED_DIR, "full_features.csv"))
    station_summary = df_feat.groupby("station_name").agg({
        "elevation_m": "first",
        "slope_deg": "first",
        "population_density": "first",
        "dist_river_km": "first",
        "drainage_vulnerability": "first",
        "vulnerability_index": "first",
        "exposure_index": "first"
    }).reset_index()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Elevation vs. Drainage Vulnerability")
        st.scatter_chart(station_summary, x="elevation_m", y="drainage_vulnerability", color="station_name")
        st.caption("Low-lying floodplain stations (< 20m) exhibit high drainage vulnerability.")

    with c2:
        st.subheader("Population Exposure Index across Stations")
        st.bar_chart(station_summary.set_index("station_name")["population_density"])
        st.caption("High population density in coastal urban centers (Thiruvananthapuram, Aluva, Alappuzha) magnifies disaster risk.")


# ===========================================================================
# PAGE 5: EXPLAINABLE AI (SHAP)
# ===========================================================================
elif page == "🔍 5. Explainable AI (SHAP)":
    st.title("🔍 Explainable AI (TreeSHAP)")
    st.write("Deconstructing model predictions into exact game-theoretic Shapley feature contributions.")

    t1, t2 = st.columns(2)
    with t1:
        st.subheader("Global Feature Importance (|SHAP|)")
        bar_img = os.path.join(BASE_DIR, "outputs", "figures", "shap_bar_importance.png")
        if os.path.exists(bar_img):
            st.image(bar_img, use_container_width=True)
            
    with t2:
        st.subheader("SHAP Beeswarm Summary Plot")
        sum_img = os.path.join(BASE_DIR, "outputs", "figures", "shap_summary_plot.png")
        if os.path.exists(sum_img):
            st.image(sum_img, use_container_width=True)

    st.markdown("---")
    st.subheader("🔬 Local Instance Explanation")
    st.write("Demonstration of top risk-increasing and risk-mitigating factors for Aluva during monsoon flooding:")

    sample_case = {
        "rainfall_1d": 95.0, "rainfall_3d": 185.0, "rainfall_7d": 290.0, "rainfall_14d": 410.0,
        "rainfall_anomaly_30d": 120.0, "api_14d": 145.0, "temp_mean_3d": 24.5, "humidity_mean_3d": 94.0,
        "soil_moisture_root_zone": 0.44, "soil_moisture_deep": 0.41, "soil_saturation_ratio": 0.92,
        "elevation_m": 10.0, "slope_deg": 4.7, "dist_river_km": 0.3, "twi_proxy": 7.1,
        "elevation_risk_factor": 0.82, "sar_vv": -15.5, "sar_vh": -21.0, "sar_cross_ratio": 1.35,
        "ndwi_proxy": 0.65, "ndvi_proxy": 0.45, "population_density": 1359.0,
        "exposure_index": 0.85, "drainage_vulnerability": 0.78, "vulnerability_index": 0.81
    }
    exp_res = shap_explainer.explain_instance(sample_case)

    col_inc, col_dec = st.columns(2)
    with col_inc:
        st.error("#### 🔺 Top Factors INCREASING Flood Risk")
        for f in exp_res["factors_increasing_risk"]:
            st.markdown(f"**+ {f['readable_name']}** (Observed: `{f['value']}` | SHAP Impact: `+{f['shap_value']}`)")

    with col_dec:
        st.success("#### 🔻 Top Factors DECREASING Flood Risk")
        for f in exp_res["factors_decreasing_risk"]:
            st.markdown(f"**- {f['readable_name']}** (Observed: `{f['value']}` | SHAP Impact: `{f['shap_value']}`)")


# ===========================================================================
# PAGE 6: RESILIENCE PLANNING
# ===========================================================================
elif page == "📋 6. Resilience Planning":
    st.title("📋 Resilience Planning Engine")
    st.warning("⚠️ **MANDATORY DISCLAIMER:** Project-generated planning recommendation — not official government guidance.")

    col1, col2, col3 = st.columns(3)
    with col1:
        loc_name = st.selectbox("Location", ["Aluva", "Chalakkudy", "Kuttanad", "Chengannur", "Idukki", "Palakkad"])
    with col2:
        haz_val = st.slider("Hazard Probability", 0.0, 1.0, 0.85)
    with col3:
        rain_val = st.slider("3-Day Rainfall (mm)", 10.0, 400.0, 210.0)

    elev_map = {"Aluva": 10.0, "Chalakkudy": 21.0, "Kuttanad": -1.0, "Chengannur": 13.0, "Idukki": 672.0, "Palakkad": 77.0}
    basin_map = {"Aluva": "Periyar", "Chalakkudy": "Chalakkudy", "Kuttanad": "Pamba-Meenachil Delta", "Chengannur": "Pamba", "Idukki": "Upper Periyar", "Palakkad": "Bharathapuzha"}

    recs = resilience_engine.generate_recommendations(
        hazard_prob=haz_val,
        exposure_index=0.75,
        vulnerability_index=0.78,
        elevation_m=elev_map.get(loc_name, 15.0),
        dist_river_km=0.4,
        rainfall_3d=rain_val,
        basin_name=basin_map.get(loc_name, "Periyar"),
        location_name=loc_name
    )

    st.subheader(f"Urgency Level: {recs['urgency_level']}")

    tab1, tab2, tab3, tab4 = st.tabs(["🚨 Immediate Emergency (0-48h)", "🏗️ Structural Interventions", "📜 Policy & Planning", "🌾 Agricultural & Ecological"])
    with tab1:
        for a in recs["emergency_actions_immediate"]:
            st.markdown(f"**[{a['target_agency']}] {a['action']}**")
            st.write(a["detail"])
            st.markdown("---")

    with tab2:
        for s in recs["structural_mitigation"]:
            st.markdown(f"**[{s['priority']} Priority] {s['measure']}**")
            st.write(s["detail"])
            st.markdown("---")

    with tab3:
        for p in recs["policy_and_planning"]:
            st.markdown(f"**[{p['timeframe']}] {p['initiative']}**")
            st.write(p["detail"])
            st.markdown("---")

    with tab4:
        for ag in recs["agricultural_and_ecological"]:
            st.markdown(f"**[{ag['focus']}] {ag['strategy']}**")
            st.write(ag["detail"])
            st.markdown("---")


# ===========================================================================
# PAGE 7: MODEL PERFORMANCE
# ===========================================================================
elif page == "📊 7. Model Performance":
    st.title("📊 Model Performance & Benchmarks")
    st.write("Rigorous out-of-time evaluation on the 2023 chronological test set (5,840 records, 92 flood events).")

    # Table
    rows = []
    for m, vals in model_metrics.items():
        rows.append({
            "Model": m.replace("_", " "),
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
    df_metrics = pd.DataFrame(rows).sort_values("F1-Score", ascending=False)
    st.dataframe(df_metrics, use_container_width=True)

    st.markdown("---")
    st.subheader("📈 Evaluation Curves & Ablation Study")

    c1, c2 = st.columns(2)
    with c1:
        st.image(os.path.join(BASE_DIR, "outputs", "figures", "roc_curves.png"), caption="ROC Curves Comparison", use_container_width=True)
        st.image(os.path.join(BASE_DIR, "outputs", "figures", "confusion_matrices.png"), caption="Confusion Matrices", use_container_width=True)
    with c2:
        st.image(os.path.join(BASE_DIR, "outputs", "figures", "pr_curves.png"), caption="Precision-Recall Curves (Priority Metric)", use_container_width=True)
        st.image(os.path.join(BASE_DIR, "outputs", "figures", "ablation_study.png"), caption="Multimodal Feature Ablation Study", use_container_width=True)
