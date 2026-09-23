# Model Card: XGBoost Flood Hazard Classifier

## 1. Model Overview
* **Model Name**: XGBoost Flood Hazard Classifier
* **Model Type**: Extreme Gradient Boosted Trees (`xgboost.XGBClassifier`)
* **Version**: 1.0.0
* **Trained File**: `models/trained/model_xgboost.joblib`

## 2. Intended Use
* Operational benchmarking, secondary verification, and feature importance comparison with SHAP.

## 3. Metrics (2023 Chronological Test Set)
* **Accuracy**: 0.9825
* **Precision**: 0.4740
* **Recall**: 0.9891 (91 / 92 flood events detected, only 1 missed)
* **F1-Score**: 0.6408
* **PR-AUC**: 0.9893
* **ROC-AUC**: 0.9994
* **Brier Score**: 0.0133
* **Inference Latency**: 0.0012 ms per sample

## 4. Hyperparameters
* `n_estimators`: 180, `max_depth`: 6, `learning_rate`: 0.06
* `scale_pos_weight`: ~20.0, `subsample`: 0.85, `colsample_bytree`: 0.85

## 5. Strengths & Limitations
* **Strength**: Highest individual PR-AUC (0.9893) and near-zero false negatives (98.91% recall).
* **Limitation**: Generates more false positives (101 vs 28 for Random Forest), leading to lower precision in operational deployment.
