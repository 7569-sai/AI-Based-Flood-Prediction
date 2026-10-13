"""
Unit and Integration Tests for HF-STFloodNet (Hybrid Spatiotemporal Flood Network).
Validates hybrid architecture loading, meta-feature construction, probability bounds,
multi-level severity classification, propagated MC Dropout uncertainty, model registry,
resilience pipeline integration, and artifact reproducibility.
"""

import os
import sys
import json
import tempfile
import pytest
import numpy as np
import pandas as pd
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.models.hybrid_flood_model import HFSTFloodNet, KEY_STRUCTURED_FEATURES
from src.models.prediction_service import PredictionService
from src.models.train_pan_india import STFloodNet

MODELS_DIR = os.path.join(BASE_DIR, "models", "trained")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed", "india")


@pytest.fixture(scope="module")
def hybrid_model():
    """Load pre-trained HF-STFloodNet hybrid model."""
    assert os.path.exists(os.path.join(MODELS_DIR, "hybrid_fusion_model.joblib")), "Hybrid model artifact not found!"
    return HFSTFloodNet.load(MODELS_DIR, device="cpu")


@pytest.fixture(scope="module")
def sample_features():
    """Create a physically-grounded test feature dictionary."""
    return {
        "rainfall_1d": 65.0,
        "rainfall_3d": 140.0,
        "rainfall_7d": 220.0,
        "rainfall_14d": 310.0,
        "rainfall_anomaly_30d": 45.0,
        "api_14d": 85.0,
        "temp_mean_3d": 26.5,
        "humidity_mean_3d": 88.0,
        "soil_moisture_root_zone": 0.44,
        "soil_moisture_deep": 0.41,
        "soil_saturation_ratio": 0.92,
        "river_discharge_proxy": 1450.0,
        "river_stage_anomaly": 2.4,
        "elevation_m": 12.0,
        "slope_deg": 2.5,
        "dist_river_km": 0.8,
        "twi_proxy": 7.5,
        "elevation_risk_factor": 0.78,
        "drainage_vulnerability": 0.72,
        "sar_vv": -15.5,
        "sar_vh": -22.0,
        "sar_cross_ratio": 1.42,
        "ndwi_proxy": 0.45,
        "ndvi_proxy": 0.52,
        "population_density": 1250.0,
        "exposure_index": 0.70,
        "vulnerability_index": 0.68
    }


def test_hybrid_artifacts_exist():
    """Verify that all core hybrid research artifacts exist on disk."""
    expected = [
        os.path.join(MODELS_DIR, "hybrid_fusion_model.joblib"),
        os.path.join(MODELS_DIR, "model_pan_india_st_floodnet.pt"),
        os.path.join(MODELS_DIR, "model_pan_india_lightgbm.joblib"),
        os.path.join(MODELS_DIR, "hybrid_model_metadata.json"),
        os.path.join(RESULTS_DIR, "final_hybrid_comparison.csv"),
        os.path.join(RESULTS_DIR, "final_hybrid_comparison.md"),
        os.path.join(RESULTS_DIR, "spatial_generalization_hybrid.csv"),
        os.path.join(RESULTS_DIR, "hybrid_ablation.csv")
    ]
    for path in expected:
        assert os.path.exists(path), f"Artifact missing: {path}"


def test_hf_stfloodnet_class_loading_and_attributes(hybrid_model):
    """Verify that HFSTFloodNet loads sub-branches and meta-learner correctly."""
    assert hybrid_model.st_model is not None
    assert hybrid_model.lgb_model is not None
    assert hybrid_model.fusion_model is not None
    assert len(hybrid_model.feature_cols) == 27
    assert hybrid_model.pca_model is not None
    assert hybrid_model.pca_model.n_components == 8
    assert "HF-STFloodNet" in hybrid_model.model_name


def test_meta_feature_generation_shape(hybrid_model):
    """Verify that meta-feature generation produces exact expected column dimensionality."""
    dummy_scaled = np.random.randn(15, 27)
    meta_X = hybrid_model.build_hybrid_features(dummy_scaled)
    # Expected meta features: p_st (1) + p_lgb (1) + diff (1) + prod (1) + mean (1) + PCA embs (8) + struct (7) = 20
    assert meta_X.shape == (15, 20), f"Meta feature shape mismatch: {meta_X.shape}"
    assert not np.isnan(meta_X).any(), "Meta features contain NaN values!"


def test_hybrid_probability_bounds_and_dtype(hybrid_model):
    """Verify that hybrid inference probabilities lie strictly in [0, 1]."""
    dummy_scaled = np.random.randn(25, 27)
    probs = hybrid_model.predict_proba(dummy_scaled)
    assert probs.shape == (25,)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert not np.isnan(probs).any()

    preds = hybrid_model.predict(dummy_scaled, threshold=0.50)
    assert np.all(np.isin(preds, [0, 1]))


def test_hybrid_severity_classification(hybrid_model):
    """Verify that multi-level severity classes lie strictly in {0, 1, 2, 3}."""
    dummy_scaled = np.random.randn(20, 27)
    sev = hybrid_model.predict_severity(dummy_scaled)
    assert sev.shape == (20,)
    assert np.all(np.isin(sev, [0, 1, 2, 3]))


def test_propagated_mc_uncertainty_quantification(hybrid_model):
    """Verify that hybrid MC Dropout uncertainty produces valid statistics."""
    dummy_scaled = np.random.randn(10, 27)
    uncert = hybrid_model.predict_with_uncertainty(dummy_scaled, n_forward_passes=15)

    assert "mean_probability" in uncert
    assert "epistemic_variance" in uncert
    assert "predictive_entropy" in uncert
    assert "uncertainty_pct" in uncert
    assert "ci_lower" in uncert
    assert "ci_upper" in uncert

    assert len(uncert["mean_probability"]) == 10
    assert np.all(uncert["epistemic_variance"] >= 0.0)
    assert np.all(uncert["predictive_entropy"] >= 0.0)
    assert np.all(uncert["uncertainty_pct"] >= 2.0) and np.all(uncert["uncertainty_pct"] <= 45.0)
    assert np.all(uncert["ci_lower"] <= uncert["ci_upper"])


def test_prediction_service_model_registry(sample_features):
    """Verify PredictionService can dynamically switch between registered models."""
    # 1. Proposed HF-STFloodNet
    ps_hyb = PredictionService(model_type="hf_stfloodnet")
    res_hyb = ps_hyb.predict(sample_features)
    assert "HF-STFloodNet" in res_hyb["model_name"]
    assert 0.0 <= res_hyb["hazard_probability"] <= 1.0
    assert 0.0 <= res_hyb["risk_score"] <= 100.0

    # 2. Random Forest Baseline
    ps_rf = PredictionService(model_type="random_forest")
    res_rf = ps_rf.predict(sample_features)
    assert "Random Forest" in res_rf["model_name"]
    assert 0.0 <= res_rf["hazard_probability"] <= 1.0

    # 3. LightGBM Baseline
    ps_lgb = PredictionService(model_type="lightgbm")
    res_lgb = ps_lgb.predict(sample_features)
    assert "LightGBM" in res_lgb["model_name"]
    assert 0.0 <= res_lgb["hazard_probability"] <= 1.0


def test_prediction_service_missing_features():
    """Verify PredictionService handles missing optional features with robust defaults."""
    ps = PredictionService(model_type="hf_stfloodnet")
    partial_dict = {
        "rainfall_1d": 50.0,
        "rainfall_3d": 120.0,
        "elevation_m": 15.0
    }
    res = ps.predict(partial_dict)
    assert "hazard_probability" in res
    assert 0.0 <= res["hazard_probability"] <= 1.0
    assert 0.0 <= res["risk_score"] <= 100.0
    assert res["severity"] in [0, 1, 2, 3]


def test_hybrid_serialization_roundtrip(hybrid_model):
    """Verify that saving and reloading the hybrid pipeline preserves exact inference."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        hybrid_model.save(tmp_dir)
        reloaded = HFSTFloodNet.load(tmp_dir, device="cpu")

        test_X = np.random.randn(8, 27)
        p_orig = hybrid_model.predict_proba(test_X)
        p_reload = reloaded.predict_proba(test_X)

        np.testing.assert_allclose(p_orig, p_reload, atol=1e-5)
