"""
Unit Tests for Machine Learning and Deep Learning Models.
"""

import os
import sys
import joblib
import pytest
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

MODELS_DIR = os.path.join(BASE_DIR, "models", "trained")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")


def test_model_artifacts_exist():
    """Verify all trained model artifacts exist on disk."""
    expected = [
        "model_logistic_regression.joblib",
        "model_random_forest.joblib",
        "model_gradient_boosting.joblib",
        "model_xgboost.joblib",
        "model_bilstm.pt"
    ]
    for m in expected:
        p = os.path.join(MODELS_DIR, m)
        assert os.path.exists(p), f"Artifact missing: {p}"


def test_random_forest_prediction_bounds():
    """Verify that Random Forest generates valid probabilities in [0, 1]."""
    rf = joblib.load(os.path.join(MODELS_DIR, "model_random_forest.joblib"))
    df_test = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))
    
    with open(os.path.join(PROCESSED_DIR, "feature_metadata.json"), "r") as f:
        meta = json_load = pd.read_json(os.path.join(PROCESSED_DIR, "feature_metadata.json"), typ="series")
    
    feature_cols = meta["feature_cols"]
    sample_X = df_test[feature_cols].iloc[:20]
    
    probs = rf.predict_proba(sample_X)[:, 1]
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0), "Probabilities out of bounds [0, 1]"
    assert probs.shape == (20,), "Output shape mismatch"


def test_champion_model_recall_threshold():
    """Verify that the champion model maintains >= 90% recall on the out-of-time test set."""
    metrics_path = os.path.join(BASE_DIR, "outputs", "metrics", "model_comparison.json")
    import json
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    rf_recall = metrics["Random_Forest"]["recall"]
    assert rf_recall >= 0.90, f"Champion recall ({rf_recall}) fell below safety threshold 0.90"
