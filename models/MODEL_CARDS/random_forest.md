# Model Card: Random Forest Flood Hazard Classifier

## 1. Model Overview
* **Model Name**: Random Forest Flood Hazard Classifier
* **Model Type**: Bagging Decision Tree Ensemble (`sklearn.ensemble.RandomForestClassifier`)
* **Version**: 1.0.0
* **Date**: September 2026
* **Trained File**: `models/trained/model_random_forest.joblib`
* **Champion Status**: **Primary Operational Model**

---

## 2. Intended Use
* **Primary Intended Use**: Predicting 24- to 72-hour flood hazard probability ($P(\text{Flood}) \in [0, 1]$) and binary flood occurrence ($0$ or $1$) across Kerala river basins.
* **Intended Users**: Disaster management authorities (KSDMA), municipal emergency coordinators, regional planners, hydrologists.
* **Out-of-Scope Uses**: High-frequency urban sewer backflow micro-modeling; flash flood prediction in ungauged arid environments without local calibration.

---

## 3. Training Data & Preprocessing
* **Dataset**: Kerala River Basins Hydrometeorological & Geospatial Dataset (`data/raw/kerala_flood_raw.csv`)
* **Training Period**: 2018-01-01 to 2021-12-31 (23,376 daily station records, 844 positive flood events)
* **Validation Period**: 2022-01-01 to 2022-12-31 (5,840 records, 115 flood events)
* **Test Period**: 2023-01-01 to 2023-12-31 (5,840 records, 92 flood events)
* **Input Features**: 25 engineered features spanning meteorology (rainfall 1d, 3d, 7d, 14d, API), hydrology (root-zone soil moisture, saturation ratio), terrain (elevation, slope, river distance, TWI), remote sensing (SAR VV, VH, NDWI), and exposure proxies.

---

## 4. Hyperparameters & Training Configuration
* `n_estimators`: 150
* `max_depth`: 12
* `min_samples_split`: 6
* `class_weight`: "balanced" (accounts for 97:3 class imbalance)
* `criterion`: "gini"
* `random_state`: 42

---

## 5. Evaluation Metrics (Chronological Test Set: 2023)
* **Accuracy**: 0.9949
* **Precision**: 0.7627
* **Recall**: 0.9783 (90 / 92 flood events detected)
* **F1-Score**: 0.8571
* **PR-AUC**: 0.9888
* **ROC-AUC**: 0.9997
* **Brier Score**: 0.0090
* **Inference Latency**: 0.0082 ms per sample

---

## 6. Limitations & Known Failure Modes
1. **Extreme Ungauged Flash Floods**: If an unprecedented micro-scale cloudburst (>200mm in 2 hours) strikes a catchment without prior multi-day accumulation, reliance on daily lag features may slightly underestimate initial onset speed.
2. **Dam Gate Releases**: The model evaluates meteorological rainfall and soil saturation; sudden unannounced upstream dam releases that cause downstream flooding without local rainfall may register as false negatives unless river gauges reflect stage increases.
