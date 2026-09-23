# AI-Based Flood Prediction and Resilience Planning

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141.1-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.64.0-FF4B4B.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 19 Passed](https://img.shields.io/badge/Tests-19%20Passed%20(100%25)-brightgreen.svg)]()

> **4th-Year B.Tech Major Project**  
> An integrated multimodal geospatial artificial intelligence framework combining meteorological reanalysis, satellite remote sensing (Sentinel-1 SAR), digital elevation geomorphology (SRTM 30m), machine learning, explainable AI (SHAP), and rule-based resilience planning for flood risk reduction.

---

## 📌 Table of Contents
1. [Project Overview & Objectives](#-project-overview--objectives)
2. [Study Area & Dataset](#-study-area--dataset)
3. [System Architecture](#-system-architecture)
4. [Methodology & Leakage Prevention](#-methodology--leakage-prevention)
5. [Models & Benchmark Results](#-models--benchmark-results)
6. [Multimodal Ablation Study](#-multimodal-ablation-study)
7. [Geospatial Flood Risk & Vulnerability](#-geospatial-flood-risk--vulnerability)
8. [Explainable AI (TreeSHAP)](#-explainable-ai-treeshap)
9. [Resilience Planning Engine](#-resilience-planning-engine)
10. [REST API Documentation (FastAPI)](#-rest-api-documentation-fastapi)
11. [Interactive Dashboard (Streamlit)](#-interactive-dashboard-streamlit)
12. [Installation & Quickstart](#-installation--quickstart)
13. [Limitations & Future Scope](#-limitations--future-scope)

---

## 🌊 Project Overview & Objectives

Floods represent one of the most catastrophic natural disasters globally and across India, where intense monsoon deluges interact with steep topography to trigger rapid inundation. 

This project answers four fundamental operational questions:
> **"How likely is flooding, where is the risk concentrated, who and what is vulnerable, and what resilience measures should be implemented?"**

### Core Research Objectives:
* **Objective 1 (Flood Hazard Prediction)**: Develop an AI-based flood prediction model utilizing backward-looking meteorological accumulation, soil moisture saturation, and satellite radar backscatter.
* **Objective 2 (Spatial Risk Mapping)**: Formulate a standardized UNDRR-compliant disaster risk index integrating predicted physical hazard ($H$), demographic exposure ($E$), and drainage vulnerability ($V$) into a 0–100 risk score.
* **Objective 3 (Explainable Resilience Planning)**: Implement TreeSHAP explainability to decompose risk factors and generate context-tailored resilience recommendations with agency attributions.

---

## 🗺️ Study Area & Dataset

### Study Area: Kerala River Basins, Southwestern India
Spanning $8.52^\circ\text{N} - 11.80^\circ\text{N}, 75.78^\circ\text{E} - 77.06^\circ\text{E}$, encompassing 16 monitoring stations across 8 major river basins:
* **Periyar Basin**: Aluva, Kalady, Malayattoor, Idukki
* **Chalakkudy Basin**: Chalakkudy
* **Karuvannur Basin**: Thrissur Town
* **Pamba Basin**: Chengannur, Ranni
* **Pamba-Meenachil Delta**: Kuttanad / Mankombu (below sea level at -1.0m), Alappuzha Coastal, Kottayam
* **Western Ghats / Other Basins**: Munnar, Palakkad, Kozhikode, Mananthavady (Wayanad), Thiruvananthapuram

### Multimodal Data Stack (35,056 Records, 2018–2023)
* **Meteorology & Hydrology**: Daily rainfall, temperature, relative humidity, pressure, root-zone & deep soil moisture from ECMWF ERA5-Land via Open-Meteo.
* **Terrain Geomorphology**: USGS SRTM 30m Digital Elevation Model via Google Earth Engine (`USGS/SRTMGL1_003`).
* **Demographic Exposure**: CIESIN Gridded Population of the World GPWv4.11 (`CIESIN/GPWv411/GPW_UNWPP-Adjusted`).
* **Radar Remote Sensing**: Copernicus Sentinel-1 C-band SAR GRD (`COPERNICUS/S1_GRD`).
* **Ground Truth**: Central Water Commission (CWC) and Kerala State Disaster Management Authority (KSDMA) official records.
* **Manifest & Hash**: Serialized to `data/DATA_MANIFEST.json` (SHA-256: `a7e9fc5d40372261051c98e9f2a0d17c3ebbbdfa9d1889a8786a5ee37766f60f`).

---

## 🏛️ System Architecture

```
                               ┌────────────────────────────────┐
                               │ Real-Time Meteorological & GEE │
                               │     Remote Sensing Inputs      │
                               └───────────────┬────────────────┘
                                               │
                                               ▼
                               ┌────────────────────────────────┐
                               │   Certified 7-Vector Leakage   │
                               │      Preprocessing Engine      │
                               └───────────────┬────────────────┘
                                               │
                       ┌───────────────────────┴───────────────────────┐
                       ▼                                               ▼
        ┌─────────────────────────────┐                 ┌─────────────────────────────┐
        │  MODULE A: PREDICTION       │                 │  MODULE B: GEOSPATIAL RISK  │
        │  Random Forest (F1: 0.8571) │────────────────>│  Risk = (0.50*H + 0.25*E    │
        │  Recall: 97.8% | PR: 0.9888 │                 │         + 0.25*V) * 100     │
        └──────────────┬──────────────┘                 └──────────────┬──────────────┘
                       │                                               │
                       │               ┌───────────────────────────────┘
                       ▼               ▼
        ┌─────────────────────────────────────────────────────────────┐
        │   MODULE C: EXPLAINABILITY (SHAP) & RESILIENCE PLANNING     │
        │   • TreeSHAP Risk Drivers & Mitigators                      │
        │   • Emergency (0-48h), Structural & Agricultural Actions    │
        └──────────────────────────────┬──────────────────────────────┘
                                       │
                       ┌───────────────┴───────────────┐
                       ▼                               ▼
        ┌─────────────────────────────┐ ┌─────────────────────────────┐
        │    FastAPI REST Backend     │ │ Streamlit Interactive 7-Page│
        │      (Port 8000)            │ │     Dashboard (Port 8501)   │
        └─────────────────────────────┘ └─────────────────────────────┘
```

---

## 🛡️ Methodology & Leakage Prevention

To ensure scientific defensibility, an automated **7-vector programmatic leakage audit** (`src/data/audit_leakage.py`) was executed:
1. **Temporal Leakage**: Strict chronological split: Train (2018–2021), Val (2022), Test (2023).
2. **Spatial Leakage**: 16 geographically separated station centroids evaluated across independent river basins.
3. **Target Leakage**: Linear/rank correlation check (max feature correlation: 0.5066 for `sar_vv`).
4. **Preprocessing Leakage**: `StandardScaler` fit strictly on Train and frozen for Val/Test.
5. **Duplicate Leakage**: Zero compound primary key overlaps across splits.
6. **Satellite Acquisition Leakage**: Pre-event radar baseline backscatter and antecedent dielectric proxies only.
7. **Feature-Construction Leakage**: Verified strict `shift(1)` backward-looking lag on all rolling windows.

---

## 📊 Models & Benchmark Results

Evaluated on the **2023 chronological out-of-time test set** (5,840 records, 92 flood events):

| Model Name | Accuracy | Precision | Recall | F1-Score | PR-AUC | ROC-AUC | Brier Score | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 0.9202 | 0.1637 | **0.9891** | 0.2809 | 0.6218 | 0.9934 | 0.0566 | **0.0005** |
| **Random Forest (Champion)** | **0.9949** | **0.7627** | 0.9783 | **0.8571** | 0.9888 | **0.9997** | **0.0090** | 0.0082 |
| **Gradient Boosting** | 0.9577 | 0.2700 | **0.9891** | 0.4242 | 0.9849 | 0.9992 | 0.0271 | 0.0012 |
| **XGBoost** | 0.9825 | 0.4740 | **0.9891** | 0.6408 | **0.9893** | 0.9994 | 0.0133 | 0.0012 |
| **Bi-LSTM (Sequential DL)** | 0.9330 | 0.1816 | 0.8804 | 0.3011 | 0.5097 | 0.9672 | 0.0546 | 0.0503 |

* **Champion Model Selection**: **Random Forest Classifier** achieved the highest F1-score (**0.8571**), missing only 2 flood events (97.83% Recall) with only 28 false positives (vs 101 for XGBoost and 465 for Logistic Regression) and best calibration (Brier 0.0090).

---

## 🔬 Multimodal Ablation Study

Empirical evidence demonstrating the value of multimodal geospatial data integration:

| Experiment | Feature Stack | Features | F1-Score | Recall | PR-AUC | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Exp A** | Meteorological Only | 8 | 0.5294 | 0.9783 | 0.8026 | 0.9967 |
| **Exp B** | Meteorological + Terrain DEM | 16 | 0.6040 | 0.9783 | 0.9796 | 0.9989 |
| **Exp C** | Met + Terrain + Sentinel-1 SAR | 21 | 0.6190 | **0.9891** | **0.9900** | **0.9994** |
| **Exp D** | All Features (including Exposure/Vuln) | 25 | **0.6408** | **0.9891** | 0.9893 | **0.9994** |

* **Key Takeaway**: Incorporating SRTM DEM terrain features yields a **+17.7% PR-AUC improvement** over purely meteorological models. Adding Sentinel-1 SAR radar backscatter raises Recall to 98.91% and achieves peak PR-AUC (0.9900).

---

## 🗺️ Geospatial Flood Risk & Vulnerability

Follows the UNDRR tripartite framework:
$$\text{Risk Score} = \left[ 0.50 \cdot H + 0.25 \cdot E + 0.25 \cdot V \right] \times 100$$
* **Hazard ($H \in [0, 1]$)**: Model output probability $P(\text{Flood} \mid X)$.
* **Exposure ($E \in [0, 1]$)**: Log-normalized population density ceiling at 5,000 ppl/km².
* **Vulnerability ($V \in [0, 1]$)**: Multi-criteria index composed of lowland elevation deficit, river proximity, and flat slope drainage bottlenecks.
* **Project-Defined Thresholds**:
  * $0–20$: Very Low | $21–40$: Low | $41–60$: Moderate | $61–80$: High | $81–100$: Very High
* **Artifacts**: Multi-layer interactive Folium Leaflet maps (`outputs/maps/kerala_flood_risk_map.html`) and GeoJSON (`outputs/maps/kerala_spatial_risk.geojson`).

---

## 🔍 Explainable AI (TreeSHAP)

* **Global Explainability**: Identifies top global drivers (`rainfall_3d`, `sar_cross_ratio`, `soil_saturation_ratio`).
* **Local Instance Attribution**: Computes exact Shapley values identifying the top 3 factors driving risk higher and top 3 factors mitigating risk.
* **Figures Generated**: `outputs/figures/shap_summary_plot.png` and `outputs/figures/shap_bar_importance.png`.

---

## 📋 Resilience Planning Engine

Rule-based decision support system generating context-specific actions across four tiers:
1. **Immediate Emergency (0–48 Hours)**: Siren alerts, lowland evacuation, mobile de-watering pump staging.
2. **Structural Interventions**: Bund elevation, culvert widening, transformer plinth elevation.
3. **Policy & Planning**: Floodplain setback zoning, Aapda Mitra volunteer networks.
4. **Agricultural Adaptation**: Swarna-Sub1 submergence-tolerant paddy, sluice gate de-silting.
* **Disclaimer**: Prominently tagged as *"Project-generated planning recommendation — not official government guidance."*

---

## 🌐 REST API Documentation (FastAPI)

FastAPI service running on `http://127.0.0.1:8000`:
* `GET  /health`: System health and SQLite database connectivity status.
* `POST /predict`: Point-level flood occurrence, calibrated probability, and composite risk score.
* `POST /risk-score`: Computes 0–100 risk score from Hazard, Exposure, and Vulnerability inputs.
* `POST /explain`: TreeSHAP feature decomposition for top risk drivers.
* `POST /geospatial-risk`: Returns GeoJSON basin risk features across historical scenarios.
* `GET  /model-info`: Champion model card metadata and parameters.
* `GET  /metrics`: Complete benchmark comparison and ablation metrics.
* `GET  /recommendations`: Prescriptive resilience protocols with agency attributions.

Interactive Swagger UI: `http://127.0.0.1:8000/docs`

---

## 💻 Interactive Dashboard (Streamlit)

7 interactive pages accessible on `http://localhost:8501`:
1. **Home / Overview**: Project mission, architecture diagram, and key metrics.
2. **Flood Prediction**: Custom parameter sliders and historical presets (2018 peak, 2021 surge).
3. **Flood Risk Map**: Embedded Leaflet map with scenario toggles and station risk tables.
4. **Vulnerability & Exposure**: Scatter and distribution plots of elevation vs. drainage vulnerability.
5. **Explainable AI**: Global SHAP beeswarm plots and local waterfall feature breakdowns.
6. **Resilience Planning**: Tailored emergency and structural actions by river basin.
7. **Model Performance**: Benchmark comparison tables, ROC, PR, and ablation charts.

---

## 🚀 Installation & Quickstart

```bash
# 1. Clone & Enter Directory
cd flood_prediction_resilience

# 2. Install Dependencies
pip install -r requirements.txt

# 3. Run Automated Tests (19 tests)
pytest tests -v

# 4. Run End-to-End Live Demonstration
python src/demo_end_to_end.py

# 5. Launch FastAPI Backend
uvicorn api.main:app --host 127.0.0.1 --port 8000

# 6. Launch Streamlit Dashboard
streamlit run dashboard/app.py --server.port 8501
```

---

## 📄 License
This project is open-source under the [MIT License](LICENSE).
