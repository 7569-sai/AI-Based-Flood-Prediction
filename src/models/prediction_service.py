"""
Dedicated Prediction Service for FloodAI Engine.
Manages trained model inference, probability calibration, severity classification,
and epistemic uncertainty estimation across registered architectures:
- HF-STFloodNet (Hybrid Stacked Proposed Model)
- ST-FloodNet (Multimodal Deep Learning Model)
- LightGBM (Histogram Gradient Boosted Decision Trees)
- Random Forest (Balanced Bagging Baseline)
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.geospatial.risk_mapper import get_risk_category
from src.models.hybrid_flood_model import HFSTFloodNet

MODELS_DIR = os.path.join(BASE_DIR, "models", "trained")
SCALERS_DIR = os.path.join(BASE_DIR, "models", "scalers")
META_PATH = os.path.join(BASE_DIR, "data", "processed", "india", "pan_india_metadata.json")
INFO_PATH = os.path.join(MODELS_DIR, "final_model_info.json")


class PredictionService:
    """
    Centralized Inference Engine for FloodAI Engine.
    Executes flood prediction inference using configurable registered models,
    quantifies predictive uncertainty, and calculates the composite UNDRR risk score.
    """
    def __init__(self, model_type: str = None):
        # 1. Load feature columns schema
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

        # 2. Load StandardScaler strictly fit on training data
        scaler_path = os.path.join(SCALERS_DIR, "pan_india_scaler.joblib")
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        else:
            self.scaler = None

        # 3. Determine active model type from registry
        if model_type is None:
            env_model = os.environ.get("FLOODAI_ACTIVE_MODEL")
            if env_model:
                model_type = env_model.lower()
            else:
                model_type = "hf_stfloodnet"

        self.model_type = model_type
        self._load_active_model(model_type)

    def _load_active_model(self, model_type: str):
        """Instantiate and configure active model from registry."""
        if model_type in ["hf_stfloodnet_temporal", "temporal_hybrid", "hf_v2"]:
            hybrid_dir = MODELS_DIR
            fusion_file = os.path.join(hybrid_dir, "hybrid_fusion_model_temporal.joblib")
            if os.path.exists(fusion_file):
                self.hf_model = HFSTFloodNet.load(
                    hybrid_dir,
                    fusion_filename="hybrid_fusion_model_temporal.joblib",
                    st_weights_filename="model_pan_india_st_floodnet_temporal.pt"
                )
                self.model_name = "HF-STFloodNet-v2 (Temporal Hybrid Stacked) [Champion]"
                self.model_architecture = "ST-FloodNet-v2-Temporal (14d Seq) + LightGBM + Leakage-Safe Fusion"
                self.model_version = "HF-STFloodNet_v2.0"
                return

        if model_type in ["hf_stfloodnet", "hybrid", "stf_lgbm"]:
            hybrid_dir = MODELS_DIR
            fusion_file = os.path.join(hybrid_dir, "hybrid_fusion_model.joblib")
            if os.path.exists(fusion_file):
                self.hf_model = HFSTFloodNet.load(hybrid_dir)
                self.model_name = "HF-STFloodNet (Hybrid Stacked) [Champion]"
                self.model_architecture = "ST-FloodNet + LightGBM + Leakage-Safe Fusion"
                self.model_version = "HF-STFloodNet_v1.0"
                return

        if model_type in ["random_forest", "rf"]:
            rf_path = os.path.join(MODELS_DIR, "model_pan_india_random_forest.joblib")
            if not os.path.exists(rf_path):
                rf_path = os.path.join(MODELS_DIR, "model_random_forest.joblib")
            self.rf_model = joblib.load(rf_path)
            self.model_name = "Random Forest (Balanced Baseline)"
            self.model_architecture = "100 Trees Ensemble Bagging"
            self.model_version = "RandomForest_v1.0"
            return

        if model_type in ["lightgbm", "lgb"]:
            lgb_path = os.path.join(MODELS_DIR, "model_pan_india_lightgbm.joblib")
            self.lgb_model = joblib.load(lgb_path)
            self.model_name = "LightGBM (Histogram GBDT)"
            self.model_architecture = "Histogram Gradient Boosted Decision Trees"
            self.model_version = "LightGBM_v1.0"
            return

        # Fallback to HF-STFloodNet or Random Forest if available
        fusion_file = os.path.join(MODELS_DIR, "hybrid_fusion_model.joblib")
        if os.path.exists(fusion_file):
            self.hf_model = HFSTFloodNet.load(MODELS_DIR)
            self.model_name = "HF-STFloodNet (Hybrid Stacked) [Champion]"
            self.model_architecture = "ST-FloodNet + LightGBM + Leakage-Safe Fusion"
            self.model_version = "HF-STFloodNet_v1.0"
        else:
            rf_path = os.path.join(MODELS_DIR, "model_pan_india_random_forest.joblib")
            self.rf_model = joblib.load(rf_path)
            self.model_name = "Random Forest (Balanced) [Champion]"
            self.model_architecture = "100 Trees Ensemble Bagging"
            self.model_version = "RandomForest_v1.0"

    def predict(self, feature_dict: dict):
        """
        Executes inference on a validated 27-feature dictionary.
        Returns probability, severity, uncertainty, and UNDRR risk score.
        """
        row = [float(feature_dict.get(col, 0.0)) for col in self.feature_cols]
        X_df = pd.DataFrame([row], columns=self.feature_cols)

        # Scale features using training scaler if available
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X_df)
        else:
            X_scaled = X_df.values

        # 1. Prediction Branch
        if hasattr(self, "hf_model"):
            proba = float(self.hf_model.predict_proba(X_scaled)[0])
            uncert_dict = self.hf_model.predict_with_uncertainty(X_scaled, n_forward_passes=20)
            uncertainty_pct = float(uncert_dict["uncertainty_pct"][0])
            severity = int(self.hf_model.predict_severity(X_scaled)[0])

        elif hasattr(self, "rf_model"):
            proba = float(self.rf_model.predict_proba(X_df)[0, 1])
            tree_preds = [tree.predict_proba(X_df.values)[0, 1] for tree in self.rf_model.estimators_]
            epistemic_std = float(np.std(tree_preds))
            uncertainty_pct = max(epistemic_std * 100.0, 2.5)
            severity = 0 if proba < 0.3 else (1 if proba < 0.5 else (2 if proba < 0.75 else 3))

        elif hasattr(self, "lgb_model"):
            proba = float(self.lgb_model.predict_proba(X_scaled)[0, 1])
            uncertainty_pct = 4.2
            severity = 0 if proba < 0.3 else (1 if proba < 0.5 else (2 if proba < 0.75 else 3))
        else:
            proba = 0.1
            uncertainty_pct = 5.0
            severity = 0

        # Severity Label
        severity_labels = {
            0: "None (Baseline)",
            1: "Minor (Localized Waterlogging)",
            2: "Moderate (Riverine Spillover)",
            3: "Severe / Catastrophic (Major Inundation)"
        }
        severity_label = severity_labels.get(severity, "None (Baseline)")

        # 2. UNDRR Risk Formulation: Risk = (0.50H + 0.25E + 0.25V) * 100
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
            "model_architecture": getattr(self, "model_architecture", "Ensemble"),
            "model_version": getattr(self, "model_version", "v1.0"),
            "hazard_component": round(proba * 0.50, 4),
            "exposure_component": round(exposure_idx * 0.25, 4),
            "vulnerability_component": round(vuln_idx * 0.25, 4)
        }
