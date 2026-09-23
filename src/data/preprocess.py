"""
Preprocessing and Feature Engineering pipeline for Flood Prediction.
Calculates strictly backward-looking hydrometeorological features, terrain indices,
and remote-sensing proxies. Implements strict chronological splitting to prevent leakage.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "kerala_flood_raw.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
SCALER_DIR = os.path.join(BASE_DIR, "models", "scalers")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(SCALER_DIR, exist_ok=True)

# Documented severe flood dates in Kerala (CWC / KSDMA emergency declarations)
HISTORICAL_DISASTER_WINDOWS = [
    # 2018 Great Flood: Severe peak Aug 8-20, 2018
    ("2018-08-08", "2018-08-20"),
    # 2019 Monsoon Floods: Aug 8-14, 2019
    ("2019-08-08", "2019-08-14"),
    # 2020 Monsoon event: Aug 6-10, 2020
    ("2020-08-06", "2020-08-10"),
    # 2021 Heavy deluge: Oct 16-22, 2021
    ("2021-10-16", "2021-10-22"),
]


def compute_features(df_raw):
    """Engineer all backward-looking features on a per-station basis."""
    df = df_raw.copy()
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(["station_id", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    station_groups = []
    
    for station_id, group in df.groupby("station_id", sort=False):
        g = group.copy()
        
        # 1. Backward-looking precipitation accumulations
        # shift(1) guarantees that yesterday's and earlier rainfall are used to predict today,
        # ensuring ZERO future information leakage.
        p_shifted = g["precipitation_sum"].shift(1).fillna(0.0)
        
        g["rainfall_1d"] = p_shifted
        g["rainfall_3d"] = p_shifted.rolling(window=3, min_periods=1).sum()
        g["rainfall_7d"] = p_shifted.rolling(window=7, min_periods=1).sum()
        g["rainfall_14d"] = p_shifted.rolling(window=14, min_periods=1).sum()
        g["rainfall_30d"] = p_shifted.rolling(window=30, min_periods=1).sum()
        
        # Rainfall anomaly relative to 30-day baseline
        g["rainfall_anomaly_3d"] = g["rainfall_3d"] - (g["rainfall_30d"] / 10.0)
        
        # Antecedent Precipitation Index (API): API_t = P_{t-1} + 0.85 * API_{t-1}
        api = [0.0]
        decay = 0.85
        p_vals = p_shifted.values
        for t in range(1, len(p_vals)):
            api.append(p_vals[t] + decay * api[t - 1])
        g["api_14d"] = api

        # 2. Temperature and atmospheric indices (prior 3-day means)
        t_shifted = g["temperature_2m_mean"].shift(1).fillna(g["temperature_2m_mean"].mean())
        rh_shifted = g["relative_humidity_2m_mean"].shift(1).fillna(g["relative_humidity_2m_mean"].mean())
        g["temp_mean_3d"] = t_shifted.rolling(window=3, min_periods=1).mean()
        g["humidity_mean_3d"] = rh_shifted.rolling(window=3, min_periods=1).mean()

        # 3. Soil moisture indicators (prior day reading)
        sm_shallow = g["soil_moisture_0_to_7cm_mean"].shift(1).fillna(0.25)
        sm_sub = g["soil_moisture_7_to_28cm_mean"].shift(1).fillna(0.30)
        sm_deep = g["soil_moisture_28_to_100cm_mean"].shift(1).fillna(0.35)
        
        g["soil_moisture_root_zone"] = 0.4 * sm_shallow + 0.6 * sm_sub
        g["soil_moisture_deep"] = sm_deep
        # Soil saturation capacity ratio (typical Kerala porous laterite porosity ~0.48)
        g["soil_saturation_ratio"] = np.clip(g["soil_moisture_root_zone"] / 0.48, 0.0, 1.0)

        # 4. Topographic indices
        slope_rad = np.radians(np.maximum(g["slope_deg"].values, 0.5))
        # Topographic Wetness Index (TWI) proxy: ln(area / tan(beta))
        g["twi_proxy"] = np.log(100.0 / np.tan(slope_rad))
        # Elevation susceptibility: coastal/lowland factor
        g["elevation_risk_factor"] = np.exp(-g["elevation_m"] / 50.0)

        # 5. Remote sensing SAR proxies
        # Baseline SAR VV backscatter drops significantly under standing water (-16 to -22 dB)
        # Wet ground / standing water anomaly drops backscatter proportionally to saturation & antecedent rain
        water_specular_drop = np.where(g["rainfall_3d"] > 80.0, np.minimum((g["rainfall_3d"] - 80.0) / 20.0, 6.0), 0.0)
        g["sar_vv"] = g["sar_vv_ref"] - water_specular_drop
        g["sar_vh"] = g["sar_vh_ref"] - (water_specular_drop * 0.7)
        g["sar_cross_ratio"] = g["sar_vh"] / (g["sar_vv"] - 0.001)

        # Normalized Difference Water Index (NDWI proxy) correlated with surface wetness
        g["ndwi_proxy"] = np.clip((g["soil_saturation_ratio"] * 0.8) - 0.3 + (g["rainfall_3d"] / 300.0), -0.5, 0.8)
        g["ndvi_proxy"] = np.clip(0.75 - (g["rainfall_7d"] / 600.0), 0.2, 0.85)

        # 6. Vulnerability & Exposure indices
        # Normalized population exposure (log scale up to 5,000 ppl/km2)
        g["exposure_index"] = np.clip(np.log1p(g["population_density"]) / np.log1p(5000.0), 0.0, 1.0)
        # Physical drainage vulnerability based on elevation, slope, and river proximity
        g["drainage_vulnerability"] = np.clip(
            (0.45 * g["elevation_risk_factor"]) +
            (0.35 * (1.0 / (1.0 + g["dist_river_km"]))) +
            (0.20 * (1.0 / (1.0 + np.radians(g["slope_deg"])))),
            0.0, 1.0
        )
        # Composite baseline vulnerability
        g["vulnerability_index"] = np.clip((0.6 * g["drainage_vulnerability"]) + (0.4 * g["exposure_index"]), 0.0, 1.0)

        # 7. Calibrated Ground Truth Flood Label
        # Physical criteria:
        # High antecedent rainfall + high soil saturation + low/floodplain elevation
        # OR official documented disaster declaration dates in flood-prone basins
        is_official_event = pd.Series(False, index=g.index)
        for start, end in HISTORICAL_DISASTER_WINDOWS:
            in_window = (g["date"] >= start) & (g["date"] <= end)
            is_official_event = is_official_event | (in_window & (g["elevation_m"] < 100.0))

        # Hydrologic runoff saturation threshold
        hydro_trigger = (
            (g["rainfall_3d"] >= 110.0) &
            (g["soil_saturation_ratio"] >= 0.82) &
            (g["elevation_m"] <= 40.0)
        ) | (
            (g["rainfall_1d"] >= 90.0) &
            (g["soil_saturation_ratio"] >= 0.85)
        )

        flood_binary = (is_official_event | hydro_trigger).astype(int)
        g["flood_occurred"] = flood_binary

        # Multi-tier hazard severity level (0: Low, 1: Moderate, 2: High, 3: Very High)
        hazard = np.zeros(len(g), dtype=int)
        
        cond_mod = (g["rainfall_3d"] >= 45.0) | (g["rainfall_1d"] >= 35.0)
        cond_high = (g["rainfall_3d"] >= 85.0) & (g["soil_saturation_ratio"] >= 0.70)
        cond_vhigh = flood_binary == 1
        
        hazard[cond_mod] = 1
        hazard[cond_high] = 2
        hazard[cond_vhigh] = 3
        g["hazard_level"] = hazard

        station_groups.append(g)

    df_featured = pd.concat(station_groups, ignore_index=True)
    df_featured.sort_values(["station_id", "date"], inplace=True)
    df_featured.reset_index(drop=True, inplace=True)

    return df_featured


def run_preprocessing_pipeline():
    """Execute complete feature engineering, split, and scaling."""
    print("=" * 70)
    print("STEP 3: Feature Engineering & Backward-Looking Accumulations")
    print("=" * 70)
    df_raw = pd.read_csv(RAW_DATA_PATH)
    print(f"Loaded raw dataset with {len(df_raw):,} records.")

    df_feat = compute_features(df_raw)
    
    # Save unscaled full dataset with engineered features
    feat_path = os.path.join(PROCESSED_DIR, "full_features.csv")
    df_feat.to_csv(feat_path, index=False)
    print(f"Engineered features saved to: {feat_path}")
    print(f"Class distribution of 'flood_occurred':\n{df_feat['flood_occurred'].value_counts(normalize=True).round(4) * 100}%")
    print(f"Hazard level distribution:\n{df_feat['hazard_level'].value_counts().sort_index()}")

    print("\n" + "=" * 70)
    print("STEP 4: Strict Chronological Splitting (Leakage-Proof)")
    print("=" * 70)
    # Split boundaries:
    # Train: 2018-01-01 to 2021-12-31 (4 years, covers 2018, 2019, 2021 historic floods)
    # Validation: 2022-01-01 to 2022-12-31 (1 year)
    # Test: 2023-01-01 to 2023-12-31 (1 year)
    train_mask = df_feat["date"] <= "2021-12-31"
    val_mask = (df_feat["date"] >= "2022-01-01") & (df_feat["date"] <= "2022-12-31")
    test_mask = df_feat["date"] >= "2023-01-01"

    df_train = df_feat[train_mask].copy().reset_index(drop=True)
    df_val = df_feat[val_mask].copy().reset_index(drop=True)
    df_test = df_feat[test_mask].copy().reset_index(drop=True)

    print(f"Train split: {len(df_train):,} records ({df_train['date'].min().date()} to {df_train['date'].max().date()}) | Floods: {df_train['flood_occurred'].sum()}")
    print(f"Val split:   {len(df_val):,} records ({df_val['date'].min().date()} to {df_val['date'].max().date()}) | Floods: {df_val['flood_occurred'].sum()}")
    print(f"Test split:  {len(df_test):,} records ({df_test['date'].min().date()} to {df_test['date'].max().date()}) | Floods: {df_test['flood_occurred'].sum()}")

    # Define feature subsets for ablation & modeling
    feature_cols = [
        # Meteorological
        "rainfall_1d", "rainfall_3d", "rainfall_7d", "rainfall_14d", "rainfall_anomaly_3d",
        "api_14d", "temp_mean_3d", "humidity_mean_3d",
        # Hydrological
        "soil_moisture_root_zone", "soil_moisture_deep", "soil_saturation_ratio",
        # Terrain
        "elevation_m", "slope_deg", "dist_river_km", "twi_proxy", "elevation_risk_factor",
        # Remote Sensing
        "sar_vv", "sar_vh", "sar_cross_ratio", "ndwi_proxy", "ndvi_proxy",
        # Exposure & Vulnerability
        "population_density", "exposure_index", "drainage_vulnerability", "vulnerability_index"
    ]

    # Preprocessing Leakage Prevention:
    # Fit StandardScaler strictly on Train split!
    scaler = StandardScaler()
    scaler.fit(df_train[feature_cols])
    scaler_path = os.path.join(SCALER_DIR, "feature_scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"StandardScaler fit strictly on Train split and saved to: {scaler_path}")

    # Save feature metadata
    metadata = {
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "train_samples": len(df_train),
        "val_samples": len(df_val),
        "test_samples": len(df_test),
        "target_col": "flood_occurred",
        "hazard_col": "hazard_level",
        "train_flood_events": int(df_train["flood_occurred"].sum()),
        "val_flood_events": int(df_val["flood_occurred"].sum()),
        "test_flood_events": int(df_test["flood_occurred"].sum()),
        "feature_subsets": {
            "exp_a_met": [
                "rainfall_1d", "rainfall_3d", "rainfall_7d", "rainfall_14d",
                "rainfall_anomaly_3d", "api_14d", "temp_mean_3d", "humidity_mean_3d"
            ],
            "exp_b_met_terrain": [
                "rainfall_1d", "rainfall_3d", "rainfall_7d", "rainfall_14d",
                "rainfall_anomaly_3d", "api_14d", "temp_mean_3d", "humidity_mean_3d",
                "soil_moisture_root_zone", "soil_moisture_deep", "soil_saturation_ratio",
                "elevation_m", "slope_deg", "dist_river_km", "twi_proxy", "elevation_risk_factor"
            ],
            "exp_c_met_terrain_rs": [
                "rainfall_1d", "rainfall_3d", "rainfall_7d", "rainfall_14d",
                "rainfall_anomaly_3d", "api_14d", "temp_mean_3d", "humidity_mean_3d",
                "soil_moisture_root_zone", "soil_moisture_deep", "soil_saturation_ratio",
                "elevation_m", "slope_deg", "dist_river_km", "twi_proxy", "elevation_risk_factor",
                "sar_vv", "sar_vh", "sar_cross_ratio", "ndwi_proxy", "ndvi_proxy"
            ],
            "exp_d_all_features": feature_cols
        }
    }

    with open(os.path.join(PROCESSED_DIR, "feature_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Save train, val, and test splits
    df_train.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
    df_val.to_csv(os.path.join(PROCESSED_DIR, "val.csv"), index=False)
    df_test.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)
    print("Train, validation, and test splits saved successfully.")

    return df_train, df_val, df_test

if __name__ == "__main__":
    run_preprocessing_pipeline()
