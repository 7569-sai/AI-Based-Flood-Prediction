# System Architecture: FloodAI Engine

**Project Title:** AI-Based Flood Prediction, Dynamic Risk Mapping and Resilience Planning  
**Application:** FloodAI Engine — "AI for a Safer, Resilient India"

---

## 1. High-Level Architectural Blueprint

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              PAN-INDIA MULTI-SOURCE DATA LAYER                         │
│  • ECMWF ERA5-Land: Daily Precipitation, Temperature, Humidity, Soil Moisture (2 layers)│
│  • Google Earth Engine: USGS SRTM 30m DEM, CIESIN Population GPWv4, Sentinel-1 SAR GRD │
│  • Central Water Commission (CWC): River Stage Records, Barrage Discharge Baselines    │
│  • Scalable Storage: Parquet (Snappy), GeoJSON, SQLite relational persistence          │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        DATA INTEGRITY & LEAKAGE AUDIT ENGINE                           │
│  • Strictly Backward-Looking Rolling Windows: Rain (1d, 3d, 7d, 14d), API, Anomalies   │
│  • Chronological Partitioning: Train (2018–2021) | Val (2022) | Test (2023)           │
│  • Spatial Holdout Segregation: Brahmaputra & Mahanadi Basins held out for transfer    │
│  • Certified 7-Vector Programmatic Audit: 100% verified clean                          │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                 ┌──────────────────────────┴──────────────────────────┐
                 ▼                                                     ▼
┌─────────────────────────────────────────────────┐   ┌──────────────────────────────────┐
│             MODULE A: PREDICTION                │   │    MODULE B: DYNAMIC RISK        │
│  • Baselines: Logistic Reg, Random Forest, GB,  │   │  • Hazard H = P(Flood | X)       │
│    XGBoost, LightGBM, Bi-LSTM                   │──>│  • Exposure E = f(Population)    │
│  • Proposed Architecture: ST-FloodNet           │   │  • Vulnerability V = f(DEM, Riv) │
│    - Multi-Scale Temporal 1D-CNN                │   │  • Dynamic Risk Score ∈ [0, 100] │
│    - Spatial Terrain & Satellite Encoders       │   │  • Multi-layer Folium GIS Map    │
│    - Gated Cross-Modal Fusion                   │   └────────────────┬─────────────────┘
│    - Multi-Task Head (Hazard + Severity)        │                    │
│    - Monte Carlo Dropout Epistemic Uncertainty  │                    │
└────────────────────────┬────────────────────────┘                    │
                         │                   ┌─────────────────────────┘
                         ▼                   ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                     MODULE C: RESILIENCE PLANNING & DECISION SUPPORT                   │
│  • Context-Aware Prescriptive Engine Grounded in Identified Geomorphic Bottlenecks     │
│  • 4 Action Domains: Immediate (0-48h), Structural, Policy/Zoning, Agricultural        │
│  • Mandatory Academic Disclaimer Tagging (Not Official Government Guidance)            │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                 ┌──────────────────────────┴──────────────────────────┐
                 ▼                                                     ▼
┌─────────────────────────────────────────────────┐   ┌──────────────────────────────────┐
│                FASTAPI REST API                 │   │     MODERN STREAMLIT DASHBOARD   │
│  • Endpoints: /predict, /risk-score, /explain,  │   │  • Structured Sidebar Hierarchy  │
│    /stations, /location-lookup, /recommendations│   │  • Dynamic Location Analysis     │
│  • Pydantic v2 Schema Validation                │   │  • Real-Time Environmental Fetch │
│  • SQLite Inference Audit Logging               │   │  • 2018–2023 Historical Deep Dive│
└─────────────────────────────────────────────────┘   └──────────────────────────────────┘
```

---

## 2. ST-FloodNet Proposed Architecture Pipeline

```
Antecedent Meteorology ────► [Multi-Scale 1D Conv] ──► [Bi-GRU Memory] ──┐
                                                                         │
Topographic DEM / Slope ───► [Spatial Terrain Encoder] ─────────────────┼──► [Gated Cross-Modal Fusion]
                                                                         │                │
Sentinel-1 SAR / Indices ──► [Satellite Radar Encoder] ─────────────────┘                ▼
                                                                                   [MC Dropout (p=0.35)]
                                                                                          │
                                                                   ┌──────────────────────┴──────────────────────┐
                                                                   ▼                                             ▼
                                                     [Hazard Logit -> Sigmoid]                     [Severity Head -> Softmax]
                                                                   │                                             │
                                                       Hazard Probability P(Flood)                   Severity Level (0, 1, 2, 3)
                                                       + Epistemic Uncertainty Variance σ²
```

---

## 3. Directory Layout

- `src/data/`: Data builders, ingestion scripts, and leakage verification (`build_pan_india_dataset.py`, `audit_pan_india_leakage.py`, `historical_analyzer.py`).
- `src/models/`: Model training, benchmarking, ablation studies, and evaluation curves (`train_pan_india.py`).
- `src/geospatial/`: Pan-India location profiles (`location_service.py`) and dynamic Folium risk maps (`risk_mapper.py`).
- `src/explainability/`: TreeSHAP global beeswarm and local attribution (`shap_explainer.py`).
- `src/resilience/`: Multi-domain resilience recommendation engine (`recommendation_engine.py`).
- `api/`: FastAPI web server, schemas, and endpoints (`main.py`, `schemas.py`).
- `dashboard/`: Streamlit interactive web platform (`app.py`).
- `tests/`: Automated pytest verification suite.
