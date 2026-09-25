"""
Comprehensive unit and end-to-end tests for the APIFeatureExtractor layer.
Verifies coordinate bounds validation, API query & offline fallback, data integrity,
strict 27-feature vector construction, physical limits, and inference compatibility.
"""

import pytest
import numpy as np
from src.data.api_feature_extractor import APIFeatureExtractor, EXPECTED_FEATURE_ORDER
from src.geospatial.risk_mapper import SpatialRiskEngine


@pytest.fixture(scope="module")
def extractor():
    return APIFeatureExtractor(timeout_sec=5)


@pytest.fixture(scope="module")
def risk_engine():
    return SpatialRiskEngine(use_pan_india=True)


def test_coordinate_bounds_validation(extractor):
    """Verify that coordinates outside global limits throw ValueError."""
    with pytest.raises(ValueError):
        extractor.extract_features_for_location(95.0, 77.0)

    with pytest.raises(ValueError):
        extractor.extract_features_for_location(-100.0, 77.0)

    with pytest.raises(ValueError):
        extractor.extract_features_for_location(28.0, 195.0)

    with pytest.raises(ValueError):
        extractor.extract_features_for_location(28.0, -200.0)


def test_delhi_feature_extraction(extractor):
    """Test full extraction pipeline on Delhi coordinates (Indo-Gangetic Basin)."""
    lat, lon = 28.6644, 77.2389
    res = extractor.extract_features_for_location(lat, lon)

    assert "location_info" in res
    assert "status_panel" in res
    assert "sources_panel" in res
    assert "features" in res
    assert "validation_passed" in res

    loc = res["location_info"]
    assert loc["latitude"] == round(lat, 4)
    assert loc["longitude"] == round(lon, 4)
    assert "basin" in loc and len(loc["basin"]) > 0

    features = res["features"]
    assert len(features) == 27
    for key in EXPECTED_FEATURE_ORDER:
        assert key in features, f"Missing expected feature {key}"
        assert features[key] is not None
        assert not np.isnan(features[key])
        assert not np.isinf(features[key])


def test_guwahati_feature_extraction(extractor):
    """Test full extraction pipeline on Guwahati coordinates (Brahmaputra Basin)."""
    lat, lon = 26.1856, 91.7490
    res = extractor.extract_features_for_location(lat, lon)

    assert res["validation_passed"] is True
    features = res["features"]
    assert features["elevation_m"] >= -10.0
    assert 0.0 <= features["soil_saturation_ratio"] <= 1.0
    assert 0.0 <= features["exposure_index"] <= 1.0


def test_physical_limits_and_temporal_consistency(extractor):
    """Verify that extracted features satisfy physical ranges and non-negative rainfall."""
    lat, lon = 9.9312, 76.2673  # Kochi (Periyar Basin)
    res = extractor.extract_features_for_location(lat, lon)
    f = res["features"]

    # Non-negative rainfall
    assert f["rainfall_1d"] >= 0.0
    assert f["rainfall_3d"] >= 0.0
    assert f["rainfall_7d"] >= 0.0
    assert f["rainfall_14d"] >= 0.0

    # Monotonic accumulation: R_14d >= R_7d >= R_3d >= R_1d
    assert f["rainfall_3d"] >= f["rainfall_1d"] - 1e-4
    assert f["rainfall_7d"] >= f["rainfall_3d"] - 1e-4
    assert f["rainfall_14d"] >= f["rainfall_7d"] - 1e-4

    # Soil moisture physical limits
    assert 0.0 <= f["soil_moisture_root_zone"] <= 1.0
    assert 0.0 <= f["soil_moisture_deep"] <= 1.0

    # Terrain physical limits
    assert -500.0 <= f["elevation_m"] <= 9000.0
    assert 0.0 <= f["slope_deg"] <= 90.0

    # Remote sensing limits
    assert -40.0 <= f["sar_vv"] <= 0.0
    assert -45.0 <= f["sar_vh"] <= 0.0
    assert -1.0 <= f["ndwi_proxy"] <= 1.0
    assert -1.0 <= f["ndvi_proxy"] <= 1.0


def test_feature_validation_catches_corrupt_data(extractor):
    """Verify that corrupt or violated feature vectors are flagged with validation errors."""
    corrupt_features = {k: 1.0 for k in EXPECTED_FEATURE_ORDER}
    
    # Intentionally corrupt rainfall
    corrupt_features["rainfall_1d"] = -5.0
    passed, errs = extractor.validate_feature_vector(corrupt_features)
    assert passed is False
    assert any("negative" in e.lower() for e in errs)

    # Intentionally violate temporal causality: R_3d < R_1d
    corrupt_features["rainfall_1d"] = 50.0
    corrupt_features["rainfall_3d"] = 10.0
    passed, errs = extractor.validate_feature_vector(corrupt_features)
    assert passed is False
    assert any("temporal" in e.lower() for e in errs)

    # Intentionally missing feature
    del corrupt_features["elevation_m"]
    passed, errs = extractor.validate_feature_vector(corrupt_features)
    assert passed is False
    assert any("missing" in e.lower() for e in errs)


def test_model_inference_compatibility(extractor, risk_engine):
    """Verify that extracted features directly feed SpatialRiskEngine without schema errors."""
    lat, lon = 25.5941, 85.1376  # Patna (Ganga Basin)
    res = extractor.extract_features_for_location(lat, lon)
    features = res["features"]

    # Compute risk with trained model
    risk_output = risk_engine.compute_point_risk(features)

    assert "hazard_probability" in risk_output
    assert 0.0 <= risk_output["hazard_probability"] <= 1.0
    assert "risk_score" in risk_output
    assert 0.0 <= risk_output["risk_score"] <= 100.0
    assert "risk_category" in risk_output
    assert risk_output["risk_category"] in ["Very Low", "Low", "Moderate", "High", "Very High"]
    assert "prediction" in risk_output
    assert risk_output["prediction"] in [0, 1]
