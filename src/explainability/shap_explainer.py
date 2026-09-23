"""
Explainable AI (XAI) Engine using TreeSHAP.
Computes exact Shapley additive explanations for global model understanding
and local instance-level risk attribution.
Identifies top factors driving risk higher and lower for any input scenario.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models", "trained")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")
METRICS_DIR = os.path.join(BASE_DIR, "outputs", "metrics")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


class FloodShapExplainer:
    def __init__(self, use_pan_india=True):
        pan_india_model = os.path.join(MODELS_DIR, "model_pan_india_random_forest.joblib")
        kerala_model = os.path.join(MODELS_DIR, "model_random_forest.joblib")
        pan_india_meta = os.path.join(BASE_DIR, "data", "processed", "india", "pan_india_metadata.json")

        if use_pan_india and os.path.exists(pan_india_model) and os.path.exists(pan_india_meta):
            model_path = pan_india_model
            meta_path = pan_india_meta
            self.is_pan_india = True
        else:
            model_path = kerala_model
            meta_path = os.path.join(PROCESSED_DIR, "feature_metadata.json")
            self.is_pan_india = False

        assert os.path.exists(model_path), f"Champion model not found at {model_path}!"
        self.model = joblib.load(model_path)

        with open(meta_path, "r") as f:
            self.metadata = json.load(f)
        self.feature_cols = self.metadata["feature_cols"]

        # Initialize TreeExplainer
        self.explainer = shap.TreeExplainer(self.model)

    def generate_global_explanations(self, sample_size=300):
        """Compute global SHAP importance and generate summary beeswarm and bar charts."""
        print(f"Generating global TreeSHAP explanations on {sample_size} test samples...")
        if self.is_pan_india:
            test_pq = os.path.join(BASE_DIR, "data", "processed", "india", "test.parquet")
            df_test = pd.read_parquet(test_pq)
        else:
            df_test = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))
        
        # Stratified or balanced sample across flood and non-flood to capture both regimes
        pos = df_test[df_test["flood_occurred"] == 1]
        neg = df_test[df_test["flood_occurred"] == 0].sample(n=min(sample_size - len(pos), len(df_test) - len(pos)), random_state=42)
        sample_df = pd.concat([pos, neg], ignore_index=True)
        X_sample = sample_df[self.feature_cols]

        shap_values = self.explainer.shap_values(X_sample)

        # For binary classification in sklearn RandomForest, shap_values can be a list of 2 arrays [class 0, class 1]
        # or a 3D array [samples, features, 2]. Select class 1 (Flood).
        if isinstance(shap_values, list):
            sv_flood = shap_values[1]
        elif len(shap_values.shape) == 3:
            sv_flood = shap_values[:, :, 1]
        else:
            sv_flood = shap_values

        # 1. Save SHAP Summary Beeswarm Plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(sv_flood, X_sample, feature_names=self.feature_cols, show=False, max_display=15)
        plt.title("SHAP Summary Plot: Impact of Environmental Features on Flood Hazard", fontsize=12)
        plt.tight_layout()
        summary_path = os.path.join(FIGURES_DIR, "shap_summary_plot.png")
        plt.savefig(summary_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"SHAP summary plot saved to: {summary_path}")

        # 2. Save SHAP Global Bar Plot
        plt.figure(figsize=(10, 7))
        mean_abs_shap = np.mean(np.abs(sv_flood), axis=0)
        shap_importance = pd.Series(mean_abs_shap, index=self.feature_cols).sort_values(ascending=True)
        shap_importance.tail(15).plot(kind="barh", color="#1f77b4")
        plt.title("Global Mean |SHAP Value| (Average Impact on Model Output Magnitude)", fontsize=12)
        plt.xlabel("Mean |SHAP Value|")
        plt.tight_layout()
        bar_path = os.path.join(FIGURES_DIR, "shap_bar_importance.png")
        plt.savefig(bar_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"SHAP bar importance saved to: {bar_path}")

        # Save feature ranking JSON
        shap_ranking = shap_importance.sort_values(ascending=False).to_dict()
        with open(os.path.join(METRICS_DIR, "shap_feature_importance.json"), "w", encoding="utf-8") as f:
            json.dump(shap_ranking, f, indent=2)

        return summary_path, bar_path

    def explain_instance(self, feature_dict, save_plot=False):
        """
        Explain a single observation with actual Shapley values.
        Identifies top 3 risk-increasing factors and top 3 risk-decreasing factors.
        """
        row = [feature_dict.get(col, 0.0) for col in self.feature_cols]
        X = pd.DataFrame([row], columns=self.feature_cols)

        # Model output
        prob = float(self.model.predict_proba(X)[0, 1])
        prediction = int(prob >= 0.5)

        # Compute SHAP values for this instance
        shap_vals = self.explainer.shap_values(X)
        if isinstance(shap_vals, list):
            sv = shap_vals[1][0]
            base_val = float(self.explainer.expected_value[1])
        elif len(shap_vals.shape) == 3:
            sv = shap_vals[0, :, 1]
            base_val = float(self.explainer.expected_value[1])
        else:
            sv = shap_vals[0]
            base_val = float(self.explainer.expected_value)

        # Map SHAP values to feature names and raw values
        contributions = []
        for name, val, phi in zip(self.feature_cols, row, sv):
            contributions.append({
                "feature": name,
                "value": round(float(val), 2),
                "shap_value": round(float(phi), 4),
                "direction": "increases_risk" if phi > 0 else "decreases_risk"
            })

        # Sort into risk increasing and risk decreasing
        df_contrib = pd.DataFrame(contributions)
        increasing = df_contrib[df_contrib["shap_value"] > 0].sort_values("shap_value", ascending=False).head(3).to_dict("records")
        decreasing = df_contrib[df_contrib["shap_value"] < 0].sort_values("shap_value", ascending=True).head(3).to_dict("records")

        # Human-readable factor descriptions
        factor_desc = {
            "rainfall_1d": "1-Day Prior Rainfall",
            "rainfall_3d": "3-Day Cumulative Rainfall",
            "rainfall_7d": "7-Day Cumulative Rainfall",
            "rainfall_14d": "14-Day Antecedent Rainfall",
            "rainfall_anomaly_30d": "30-Day Rainfall Anomaly",
            "api_14d": "Antecedent Precipitation Index (API)",
            "soil_moisture_root_zone": "Root-Zone Soil Saturation",
            "soil_saturation_ratio": "Soil Saturation Ratio",
            "elevation_m": "Terrain Elevation Above Sea Level",
            "slope_deg": "Terrain Slope Angle",
            "dist_river_km": "Proximity to Major River Channel",
            "twi_proxy": "Topographic Wetness Index (TWI)",
            "elevation_risk_factor": "Lowland Inundation Factor",
            "sar_vv": "Sentinel-1 SAR VV Radar Backscatter",
            "sar_vh": "Sentinel-1 SAR VH Radar Backscatter",
            "sar_cross_ratio": "SAR Polarization Cross-Ratio",
            "ndwi_proxy": "Surface Water Index (NDWI Proxy)",
            "population_density": "Population Density Exposure",
            "exposure_index": "Normalized Exposure Index",
            "drainage_vulnerability": "Drainage Bottleneck Vulnerability",
            "vulnerability_index": "Composite Vulnerability Index"
        }

        for item in increasing:
            item["readable_name"] = factor_desc.get(item["feature"], item["feature"])
        for item in decreasing:
            item["readable_name"] = factor_desc.get(item["feature"], item["feature"])

        # Optional local plot
        plot_path = None
        if save_plot:
            plt.figure(figsize=(8, 4.5))
            top_factors = pd.concat([
                df_contrib.sort_values("shap_value").head(4),
                df_contrib.sort_values("shap_value").tail(4)
            ]).drop_duplicates()
            colors = ["#d62728" if x > 0 else "#2ca02c" for x in top_factors["shap_value"]]
            plt.barh(top_factors["feature"], top_factors["shap_value"], color=colors)
            plt.axvline(0, color="black", linestyle="--", alpha=0.6)
            plt.title(f"Local SHAP Feature Contributions (Predicted Flood Prob: {prob:.2%})", fontsize=11)
            plt.xlabel("SHAP Value (Contribution to Log-Odds of Flood)")
            plt.tight_layout()
            plot_path = os.path.join(FIGURES_DIR, "local_shap_explanation.png")
            plt.savefig(plot_path, dpi=180, bbox_inches="tight")
            plt.close()

        return {
            "prediction": prediction,
            "probability": round(prob, 4),
            "base_value": round(base_val, 4),
            "factors_increasing_risk": increasing,
            "factors_decreasing_risk": decreasing,
            "all_contributions": contributions,
            "local_plot": plot_path
        }


if __name__ == "__main__":
    explainer = FloodShapExplainer()
    explainer.generate_global_explanations()

    # Test local explanation on a sample flood case (Aluva during heavy rain)
    sample_case = {
        "rainfall_1d": 95.0,
        "rainfall_3d": 185.0,
        "rainfall_7d": 290.0,
        "rainfall_14d": 410.0,
        "rainfall_anomaly_30d": 120.0,
        "api_14d": 145.0,
        "temp_mean_3d": 24.5,
        "humidity_mean_3d": 94.0,
        "soil_moisture_root_zone": 0.44,
        "soil_moisture_deep": 0.41,
        "soil_saturation_ratio": 0.92,
        "elevation_m": 10.0,
        "slope_deg": 4.7,
        "dist_river_km": 0.3,
        "twi_proxy": 7.1,
        "elevation_risk_factor": 0.82,
        "sar_vv": -15.5,
        "sar_vh": -21.0,
        "sar_cross_ratio": 1.35,
        "ndwi_proxy": 0.65,
        "ndvi_proxy": 0.45,
        "population_density": 1359.0,
        "exposure_index": 0.85,
        "drainage_vulnerability": 0.78,
        "vulnerability_index": 0.81
    }
    
    explanation = explainer.explain_instance(sample_case, save_plot=True)
    print("\n--- Local SHAP Explanation Demo ---")
    print(f"Predicted Probability: {explanation['probability']:.2%}")
    print("Top factors increasing risk:")
    for f in explanation["factors_increasing_risk"]:
        print(f"  + {f['readable_name']}: val={f['value']}, SHAP=+{f['shap_value']}")
    print("Top factors decreasing risk:")
    for f in explanation["factors_decreasing_risk"]:
        print(f"  - {f['readable_name']}: val={f['value']}, SHAP={f['shap_value']}")
