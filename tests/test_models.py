"""
Unit Tests for Machine Learning and Deep Learning Models.
Validates baseline models, LightGBM, and proposed ST-FloodNet architecture.
"""

import os
import sys
import json
import joblib
import pytest
import numpy as np
import pandas as pd
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.models.train_pan_india import STFloodNet

MODELS_DIR = os.path.join(BASE_DIR, "models", "trained")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
INDIA_DIR = os.path.join(PROCESSED_DIR, "india")


def test_model_artifacts_exist():
    """Verify all trained model artifacts exist on disk."""
    expected = [
        "model_logistic_regression.joblib",
        "model_random_forest.joblib",
        "model_gradient_boosting.joblib",
        "model_xgboost.joblib",
        "model_bilstm.pt",
        "model_pan_india_lightgbm.joblib",
        "model_pan_india_st_floodnet.pt"
    ]
    for m in expected:
        p = os.path.join(MODELS_DIR, m)
        assert os.path.exists(p), f"Artifact missing: {p}"


def test_random_forest_prediction_bounds():
    """Verify that Random Forest generates valid probabilities in [0, 1]."""
    rf = joblib.load(os.path.join(MODELS_DIR, "model_pan_india_random_forest.joblib"))
    df_test = pd.read_parquet(os.path.join(INDIA_DIR, "test.parquet"))
    
    with open(os.path.join(INDIA_DIR, "pan_india_metadata.json"), "r") as f:
        meta = json.load(f)
    
    feature_cols = meta["feature_cols"]
    sample_X = df_test[feature_cols].iloc[:20]
    
    probs = rf.predict_proba(sample_X)[:, 1]
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0), "Probabilities out of bounds [0, 1]"
    assert probs.shape == (20,), "Output shape mismatch"


def test_lightgbm_prediction_bounds():
    """Verify that LightGBM generates valid probabilities in [0, 1]."""
    lgb_model = joblib.load(os.path.join(MODELS_DIR, "model_pan_india_lightgbm.joblib"))
    df_test = pd.read_parquet(os.path.join(INDIA_DIR, "test.parquet"))
    
    with open(os.path.join(INDIA_DIR, "pan_india_metadata.json"), "r") as f:
        meta = json.load(f)
    
    feature_cols = meta["feature_cols"]
    sample_X = df_test[feature_cols].iloc[:20]
    
    probs = lgb_model.predict_proba(sample_X)[:, 1]
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert probs.shape == (20,)


def test_st_floodnet_uncertainty_quantification():
    """Verify that ST-FloodNet outputs valid probabilities and epistemic variance."""
    st_model = STFloodNet(num_features=27)
    pt_path = os.path.join(MODELS_DIR, "model_pan_india_st_floodnet.pt")
    st_model.load_state_dict(torch.load(pt_path, map_location="cpu"))
    st_model.eval()

    sample_x = torch.randn(10, 27)
    mean_prob, var_uncert, entropy = st_model.predict_with_uncertainty(sample_x, n_forward_passes=10)

    assert len(mean_prob) == 10
    assert np.all(mean_prob >= 0.0) and np.all(mean_prob <= 1.0)
    assert np.all(var_uncert >= 0.0), "Epistemic variance must be non-negative"
    assert np.all(entropy >= 0.0), "Predictive entropy must be non-negative"


def test_champion_model_recall_threshold():
    """Verify that the champion model maintains >= 90% recall on the out-of-time test set."""
    metrics_path = os.path.join(BASE_DIR, "outputs", "metrics", "pan_india_model_comparison.json")
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    rf_recall = metrics["Random_Forest"]["recall"]
    assert rf_recall >= 0.90, f"Champion recall ({rf_recall}) fell below safety threshold 0.90"
