"""
Unit Tests for Data Preprocessing and Feature Engineering.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data.preprocess import compute_features

PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")


def test_processed_splits_exist():
    """Verify that train, val, and test split CSVs exist."""
    assert os.path.exists(os.path.join(PROCESSED_DIR, "train.csv")), "train.csv missing"
    assert os.path.exists(os.path.join(PROCESSED_DIR, "val.csv")), "val.csv missing"
    assert os.path.exists(os.path.join(PROCESSED_DIR, "test.csv")), "test.csv missing"


def test_chronological_ordering_and_no_overlap():
    """Verify strict chronological monotonicity between Train, Val, and Test."""
    df_train = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    df_val = pd.read_csv(os.path.join(PROCESSED_DIR, "val.csv"))
    df_test = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))

    max_train = pd.to_datetime(df_train["date"]).max()
    min_val = pd.to_datetime(df_val["date"]).min()
    max_val = pd.to_datetime(df_val["date"]).max()
    min_test = pd.to_datetime(df_test["date"]).min()

    assert max_train < min_val, f"Train overlap with Val: {max_train} >= {min_val}"
    assert max_val < min_test, f"Val overlap with Test: {max_val} >= {min_test}"


def test_feature_engineering_backward_looking():
    """Verify that lag features do not leak current-day precipitation."""
    sample_data = pd.DataFrame({
        "station_id": ["TEST_01"] * 5,
        "date": pd.date_range("2023-01-01", periods=5),
        "precipitation_sum": [10.0, 50.0, 100.0, 0.0, 20.0],
        "temperature_2m_mean": [25.0] * 5,
        "relative_humidity_2m_mean": [80.0] * 5,
        "soil_moisture_0_to_7cm_mean": [0.3] * 5,
        "soil_moisture_7_to_28cm_mean": [0.35] * 5,
        "soil_moisture_28_to_100cm_mean": [0.4] * 5,
        "elevation_m": [15.0] * 5,
        "slope_deg": [3.0] * 5,
        "population_density": [1000.0] * 5,
        "dist_river_km": [0.5] * 5,
        "sar_vv_ref": [-11.0] * 5,
        "sar_vh_ref": [-17.5] * 5
    })

    df_out = compute_features(sample_data)
    
    # Day 0: rainfall_1d should be 0.0 (no prior day in window)
    assert df_out.loc[0, "rainfall_1d"] == 0.0
    # Day 1: rainfall_1d should be day 0 precipitation (10.0)
    assert df_out.loc[1, "rainfall_1d"] == 10.0
    # Day 2: rainfall_1d should be day 1 precipitation (50.0)
    assert df_out.loc[2, "rainfall_1d"] == 50.0


def test_no_null_values_in_features():
    """Verify that processed features contain zero NaN values."""
    df_train = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    assert df_train.isnull().sum().sum() == 0, "Train dataset contains unhandled NaN values"
