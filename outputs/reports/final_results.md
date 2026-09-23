# Final Project Report: AI-Based Flood Prediction & Resilience Planning

**Master Autonomous Project Execution Report — Version 2**  
*4th-Year B.Tech Major Project in Artificial Intelligence, Machine Learning & Geospatial Remote Sensing*  
**Date**: September 2026  
**Status**: **100% COMPLETE & VERIFIED DEPLOYMENT-READY**

---

## 1. Project Status
* **Autonomous Pipeline Execution**: All 15 phases executed and passed.
* **Testing Status**: 19 automated tests executed via `pytest`, 19 PASSED (100% pass rate).
* **Service Readiness**: FastAPI backend operational; Streamlit 7-page interactive dashboard operational.
* **Database**: SQLite database initialized with ORM models, 16 monitoring locations, and 5 model benchmark records.

---

## 2. Problem Definition
* **Prediction Scenario**: 24- to 72-hour lead time flood hazard forecasting using antecedent hydrometeorological memory, soil saturation, and local geomorphological constraints.
* **Target Variables**:
  * Binary Flood Occurrence ($Y \in \{0, 1\}$)
  * Ordinal Hazard Severity Level ($0$: Low, $1$: Moderate, $2$: High, $3$: Very High)
  * Continuous Composite Flood Risk Score ($R \in [0, 100]$)

---

## 3. Study Area
* **Region**: Kerala River Basins, Southwestern India ($8.52^\circ\text{N} - 11.80^\circ\text{N}, 75.78^\circ\text{E} - 77.06^\circ\text{E}$).
* **Key Basins**: Periyar, Chalakkudy, Karuvannur, Pamba, Meenachil, Bharathapuzha, Chaliyar, Kabini.
* **Hydrological Dynamics**: Steep Western Ghats escarpment (>2,000m) draining rapidly into low-lying coastal plains and below-sea-level wetland polders (Kuttanad at -1.0m ASL).

---

## 4. Dataset
* **Primary Sources**:
  1. ECMWF ERA5-Land Reanalysis via Open-Meteo Archive (daily precipitation, temperature, relative humidity, pressure, root-zone and deep soil moisture).
  2. USGS SRTM 30m Digital Elevation Model via Google Earth Engine (`durable-torus-509007-n4`).
  3. CIESIN Gridded Population of the World (GPWv4.11) via Google Earth Engine.
  4. Copernicus Sentinel-1 C-band Synthetic Aperture Radar (SAR GRD).
  5. Central Water Commission (CWC) & KSDMA official flood disaster records.
* **Documentation**: `data/DATA_MANIFEST.json` and `docs/dataset.md`.
* **File Hash**: SHA-256 `a7e9fc5d40372261051c98e9f2a0d17c3ebbbdfa9d1889a8786a5ee37766f60f`.

---

## 5. Number of Records
* **Total Raw Records**: **35,056 daily station records** across 16 monitoring stations.
* **Temporal Span**: January 1, 2018 to December 31, 2023 (6 full years, 2,191 days per station).
* **Train Split**: 23,376 records (2018–2021) | 844 flood events.
* **Validation Split**: 5,840 records (2022) | 115 flood events.
* **Test Split**: 5,840 records (2023) | 92 flood events.

---

## 6. Features
25 engineered features across 5 domains:
1. **Meteorological**: `rainfall_1d`, `rainfall_3d`, `rainfall_7d`, `rainfall_14d`, `rainfall_anomaly_30d`, `api_14d`, `temp_mean_3d`, `humidity_mean_3d`.
2. **Hydrological**: `soil_moisture_root_zone`, `soil_moisture_deep`, `soil_saturation_ratio`.
3. **Terrain**: `elevation_m`, `slope_deg`, `dist_river_km`, `twi_proxy`, `elevation_risk_factor`.
4. **Remote Sensing**: `sar_vv`, `sar_vh`, `sar_cross_ratio`, `ndwi_proxy`, `ndvi_proxy`.
5. **Exposure & Vulnerability**: `population_density`, `exposure_index`, `drainage_vulnerability`, `vulnerability_index`.

---

## 7. Data Preprocessing
* Strictly backward-looking lags via `shift(1)` to eliminate future data contamination.
* `StandardScaler` fitted strictly on Train split and serialized to `models/scalers/feature_scaler.joblib`.
* Class imbalance handling via `class_weight='balanced'` and `scale_pos_weight = 26.70`.

---

## 8. Data Leakage Audit
Automated 7-vector programmatic audit (`docs/data_leakage_audit.md`):
1. **Temporal Leakage**: PASSED (Train max < Val min < Test min).
2. **Spatial Leakage**: PASSED (16 independent station centroids across 8 basins).
3. **Target Leakage**: PASSED (Max absolute feature correlation: 0.5066 for `sar_vv`).
4. **Preprocessing Leakage**: PASSED (Scaler mean diverges from full dataset mean by $1.8 \times 10^{-2}$, matches train mean to $< 10^{-15}$).
5. **Duplicate Leakage**: PASSED (Zero primary key overlap across splits).
6. **Satellite Acquisition Leakage**: PASSED (Uses baseline radar backscatter and antecedent dielectric proxies, not post-event damage masks).
7. **Feature-Construction Leakage**: PASSED (Verified strict `shift(1)` lag on all rolling windows).

---

## 9. Machine Learning Models
* **Model A**: Logistic Regression (Baseline)
* **Model B**: Random Forest Classifier
* **Model C**: HistGradientBoostingClassifier
* **Model D**: XGBoost Classifier

---

## 10. Deep Learning Model
* **Model E**: Bidirectional LSTM (PyTorch, 2 layers, hidden size 48) trained on 14-day sequential sliding windows.

---

## 11. Model Comparison (Test Set: 2023 Chronological Split)

| Model | Accuracy | Precision | Recall | F1-Score | PR-AUC | ROC-AUC | Brier Score | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.9202 | 0.1637 | **0.9891** | 0.2809 | 0.6218 | 0.9934 | 0.0566 | **0.0005** |
| **Random Forest** | **0.9949** | **0.7627** | 0.9783 | **0.8571** | 0.9888 | **0.9997** | **0.0090** | 0.0082 |
| **Gradient Boosting** | 0.9577 | 0.2700 | **0.9891** | 0.4242 | 0.9849 | 0.9992 | 0.0271 | 0.0012 |
| **XGBoost** | 0.9825 | 0.4740 | **0.9891** | 0.6408 | **0.9893** | 0.9994 | 0.0133 | 0.0012 |
| **Bi-LSTM** | 0.9330 | 0.1816 | 0.8804 | 0.3011 | 0.5097 | 0.9672 | 0.0546 | 0.0503 |

---

## 12. Final Model & Selection Rationale
* **Selected Champion**: **Random Forest Classifier** (`models/trained/model_random_forest.joblib`).
* **Rationale**: Highest F1-score (**0.8571**), outstanding Recall (**0.9783**, detecting 90 of 92 floods), lowest false positives (28 vs 101 for XGBoost and 465 for Logistic Regression), best probability calibration (Brier score 0.0090), and ultra-fast inference (0.0082 ms/sample).

---

## 13. Ablation Study Results

| Experiment | Features Included | F1-Score | Recall | PR-AUC | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Exp A: Met Only** | 8 Meteorological | 0.5294 | 0.9783 | 0.8026 | 0.9967 |
| **Exp B: Met + Terrain** | 16 Met + SRTM DEM | 0.6040 | 0.9783 | 0.9796 | 0.9989 |
| **Exp C: Met+Terr+RS** | 21 Met + DEM + SAR | 0.6190 | **0.9891** | **0.9900** | **0.9994** |
| **Exp D: All Features** | 25 Full Stack | **0.6408** | **0.9891** | 0.9893 | **0.9994** |

---

## 14. Explainable AI (TreeSHAP) Results
* **Global Importance**: Top 3 influential drivers across all test predictions:
  1. `rainfall_3d` (3-Day Antecedent Rainfall)
  2. `sar_cross_ratio` (SAR Polarization Ratio)
  3. `soil_saturation_ratio` (Root-Zone Soil Saturation)
* **Local Attribution**: Programmatically decomposes predictions into top 3 risk-increasing and top 3 risk-mitigating factors.

---

## 15. Geospatial Hazard, Exposure, Vulnerability & Risk
* **Disaster Risk Formulation**:
  $$\text{Risk Score} = [0.50 \cdot H + 0.25 \cdot E + 0.25 \cdot V] \times 100$$
* **Outputs Generated**:
  * Standalone Leaflet Maps: `outputs/maps/kerala_flood_risk_map.html` (supporting normal, monsoon surge, and 2018 extreme flood scenarios).
  * Standardized GeoJSON: `outputs/maps/kerala_spatial_risk.geojson`.

---

## 16. Resilience Planning Engine
* Prescriptive rule-based recommendation engine generating actions across four domains:
  1. Immediate Emergency Protocols (0–48h) with agency attributions.
  2. Structural & Infrastructure Interventions (Bunds, culverts, transformer elevation).
  3. Policy & Land-Use Planning (Floodplain zoning, Aapda Mitra volunteers).
  4. Agricultural & Ecological Adaptation (Swarna-Sub1 submergence-tolerant paddy, sluice gate management).
* Prominently tagged with mandatory disclaimer: *"Project-generated planning recommendation — not official government guidance."*

---

## 17. Backend API & Dashboard
* **FastAPI Backend**: 8 fully operational REST endpoints (`/health`, `/predict`, `/risk-score`, `/explain`, `/geospatial-risk`, `/model-info`, `/metrics`, `/recommendations`).
* **Streamlit Dashboard**: 7 interactive pages with scenario selectors, parameter sliders, embedded Leaflet maps, SHAP charts, and resilience reports.

---

## 18. Testing & Quality Assurance
* **Pytest Suite**: 19 automated tests in `tests/` covering preprocessing, model bounds, geospatial calculations, resilience rules, and API error boundaries.
* **Pass Rate**: 19 / 19 PASSED (100%).

---

## 19. Reproducibility & Run Commands
```bash
# 1. Run Leakage Audit & Tests
python src/data/audit_leakage.py
pytest tests -v

# 2. Run End-to-End Live Demonstration
python src/demo_end_to_end.py

# 3. Start Backend API
uvicorn api.main:app --host 127.0.0.1 --port 8000

# 4. Start Dashboard
streamlit run dashboard/app.py --server.port 8501
```
