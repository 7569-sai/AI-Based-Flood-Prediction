# System Architecture: AI-Based Flood Prediction & Resilience Planning

## 1. High-Level Architectural Overview

The framework is structured into three integrated modules operating over a centralized data, model, and persistence layer:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION & PIPELINE                       │
│  • ECMWF ERA5-Land (Daily Rain, Temp, RH, Root/Deep Soil Moisture)     │
│  • Google Earth Engine (USGS SRTM 30m DEM, CIESIN Population GPWv4)    │
│  • Sentinel-1 SAR Dual-Polarization (VV, VH) & CWC Ground Truth Records│
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                 PREPROCESSING & LEAKAGE-FREE ENGINE                    │
│  • Strictly backward-looking lags: Rain(1d, 3d, 7d, 14d), API, TWI     │
│  • Chronological Splitting: Train (2018-2021) | Val (2022) | Test (2023│
│  • Certified 7-Vector Data Leakage Audit Framework                     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
         ┌──────────────────────────┴──────────────────────────┐
         ▼                                                     ▼
┌──────────────────────────────────┐        ┌──────────────────────────────────┐
│   MODULE A: ML/DL PREDICTION     │        │   MODULE B: GEOSPATIAL RISK      │
│  • Random Forest (Champion)      │        │  • Hazard H = P(Flood | X)       │
│  • XGBoost / Gradient Boosting   │───────>│  • Exposure E = f(Population)    │
│  • Sequential Bi-LSTM (PyTorch)  │        │  • Vulnerability V = f(Elev,Riv) │
│  • Calibrated Hazard Probability │        │  • Composite Risk R ∈ [0, 100]   │
└────────────────┬─────────────────┘        └────────────────┬─────────────────┘
                 │                                           │
                 │                 ┌─────────────────────────┘
                 ▼                 ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      MODULE C: RESILIENCE PLANNING                     │
│  • Context-Aware Prescriptive Engine                                   │
│  • Immediate Emergency (0-48h), Structural, Policy, Agricultural       │
│  • Mandatory Academic Disclaimer Tagging                               │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
         ┌──────────────────────────┴──────────────────────────┐
         ▼                                                     ▼
┌──────────────────────────────────┐        ┌──────────────────────────────────┐
│        BACKEND REST API          │        │       STREAMLIT DASHBOARD        │
│  • FastAPI Application           │        │  • 7 Interactive Pages           │
│  • Schemas & Type Validation     │        │  • Folium Leaflet Maps           │
│  • Endpoints: /predict, /risk    │        │  • SHAP Waterfall & Beeswarm     │
│  • SQLite + SQLAlchemy Logging   │        │  • Scenario Exploration Tool     │
└──────────────────────────────────┘        └──────────────────────────────────┘
```

---

## 2. Component Directory Structure

* **`src/data/`**: Data acquisition from GEE and Open-Meteo (`acquire_data.py`), lag feature calculations (`preprocess.py`), and 7-vector leakage verification (`audit_leakage.py`).
* **`src/models/`**: Model training, chronological validation, ablation testing, and artifact serialization (`train.py`).
* **`src/geospatial/`**: Geospatial risk engine, Folium map builder, and GeoJSON exporter (`risk_mapper.py`).
* **`src/explainability/`**: TreeSHAP global beeswarm/bar and local instance attribution (`shap_explainer.py`).
* **`src/resilience/`**: Rule-based resilience recommendation engine (`recommendation_engine.py`).
* **`src/utils/`**: Database ORM models and inference logging (`database.py`).
* **`api/`**: FastAPI REST endpoints and Pydantic schemas (`main.py`, `schemas.py`).
* **`dashboard/`**: Streamlit 7-page interactive dashboard (`app.py`).

---

## 3. Database Schema (SQLite + SQLAlchemy)

* **`predictions`**:
  * `id` (INTEGER, PK, Auto-increment)
  * `latitude` (FLOAT), `longitude` (FLOAT)
  * `timestamp` (DATETIME, UTC)
  * `probability` (FLOAT), `risk_score` (FLOAT), `risk_category` (VARCHAR)
  * `hazard_level` (INTEGER: 0-3), `model_version` (VARCHAR), `input_payload` (TEXT)
* **`locations`**:
  * `id` (INTEGER, PK), `station_id` (VARCHAR), `location_name` (VARCHAR), `basin` (VARCHAR), `district` (VARCHAR)
  * `latitude` (FLOAT), `longitude` (FLOAT), `elevation_m` (FLOAT), `slope_deg` (FLOAT), `population_density` (FLOAT)
* **`model_metrics`**:
  * `id` (INTEGER, PK), `model` (VARCHAR), `accuracy` (FLOAT), `precision` (FLOAT), `recall` (FLOAT), `f1` (FLOAT), `auc` (FLOAT), `pr_auc` (FLOAT), `brier_score` (FLOAT)
