"""
Dedicated Prediction Service for FloodAI Engine.
Manages trained model inference, probability calibration, severity classification,
and epistemic uncertainty estimation.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.geospatial.risk_mapper import get_risk_category

MODELS_DIR = os.path.join(BASE_DIR, "models", "trained")
META_PATH = os.path.join(BASE_DIR, "data", "processed", "india", "pan_india_metadata.json")


class PredictionService:
    """
    Executes flood prediction inference using the trained champion model,
    quantifies predictive uncertainty, and calculates the composite risk score.
    """
    def __init__(self):
        model_path = os.path.join(MODELS_DIR, "model_pan_india_random_forest.joblib")
        if not os.path.exists(model_path):
            model_path = os.path.join(MODELS_DIR, "model_random_forest.joblib")
        
        assert os.path.exists(model_path), f"Champion model not found at {model_path}!"
        self.model = joblib.load(model_path)
        self.model_name = "Random Forest (Balanced) [Champion]"

        if os.path.exists(META_PATH):
            with open(META_PATH, "r", encoding="utf-8") as f:
                self.feature_cols = json.load(f)["feature_cols"]
        else:
            self.feature_cols = [
                "rainfall_1d", "rainfall_3d", "rainfall_7d", "rainfall_14d",
                "rainfall_anomaly_30d", "api_14d", "temp_mean_3d", "humidity_mean_3d",
                "soil_moisture_root_zone", "soil_moisture_deep", "soil_saturation_ratio",
                "river_discharge_proxy", "river_stage_anomaly",
                "elevation_m", "slope_deg", "dist_river_km", "twi_proxy",
                "elevation_risk_factor", "drainage_vulnerability",
                "sar_vv", "sar_vh", "sar_cross_ratio", "ndwi_proxy", "ndvi_proxy",
                "population_density", "exposure_index", "vulnerability_index"
            ]

    def predict(self, feature_dict):
        """
        Executes inference on a validated 27-feature dictionary.
        Returns probability, severity, uncertainty, and UNDRR risk score.
        """
        row = [float(feature_dict.get(col, 0.0)) for col in self.feature_cols]
        X = pd.DataFrame([row], columns=self.feature_cols)

        # 1. Hazard Probability from Ensemble
        proba = float(self.model.predict_proba(X)[0, 1])

        # 2. Epistemic Uncertainty via Tree Estimator Variance
        tree_preds = [tree.predict_proba(X.values)[0, 1] for tree in self.model.estimators_]
        epistemic_std = float(np.std(tree_preds))
        # Ensure conservative calibrated uncertainty representation
        uncertainty_pct = max(epistemic_std * 100.0, 2.5)

        # 3. Severity Categorization
        if proba < 0.30:
            severity = 0
            severity_label = "None (Baseline)"
        elif proba < 0.50:
            severity = 1
            severity_label = "Minor (Localized Waterlogging)"
        elif proba < 0.75:
            severity = 2
            severity_label = "Moderate (Riverine Spillover)"
        else:
            severity = 3
            severity_label = "Severe / Catastrophic (Major Inundation)"

        # 4. UNDRR Risk Formulation: Risk = 0.50H + 0.25E + 0.25V
        exposure_idx = float(feature_dict.get("exposure_index", 0.5))
        vuln_idx = float(feature_dict.get("vulnerability_index", 0.5))
        risk_score = round(float((0.50 * proba + 0.25 * exposure_idx + 0.25 * vuln_idx) * 100.0), 2)
        category, color_hex = get_risk_category(risk_score)

        return {
            "hazard_probability": round(proba, 4),
            "probability_pct": round(proba * 100.0, 1),
            "severity": severity,
            "severity_label": severity_label,
            "epistemic_uncertainty_pct": round(uncertainty_pct, 1),
            "risk_score": risk_score,
            "risk_category": category,
            "risk_color": color_hex,
            "prediction": int(proba >= 0.5),
            "model_name": self.model_name,
            "hazard_component": round(proba * 0.50, 4),
            "exposure_component": round(exposure_idx * 0.25, 4),
            "vulnerability_component": round(vuln_idx * 0.25, 4)
        }
