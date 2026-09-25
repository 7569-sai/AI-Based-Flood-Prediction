"""
End-to-end unit tests for the Location-First Flood Intelligence Workflow.
Verifies clean map state contract, actual coordinate capture, 1 km local geodesic analysis,
real data retrieval, feature validation, model prediction, and resilience planning.
"""

import pytest
from src.location.location_service import LocationService
from src.data.feature_extractor import FeatureExtractor
from src.models.prediction_service import PredictionService
from src.geospatial.local_risk import LocalRiskAnalyzer
from src.planning.resilience_service import ResilienceService


@pytest.fixture(scope="module")
def location_svc():
    return LocationService(timeout_sec=5)


@pytest.fixture(scope="module")
def feature_svc():
    return FeatureExtractor(timeout_sec=5)


@pytest.fixture(scope="module")
def pred_svc():
    return PredictionService()


@pytest.fixture(scope="module")
def local_risk_svc():
    return LocalRiskAnalyzer()


@pytest.fixture(scope="module")
def resilience_svc():
    return ResilienceService()


def test_coordinate_validation_logic(location_svc):
    """Test global vs India boundary validation."""
    valid, msg = location_svc.validate_coordinates(28.6644, 77.2389)
    assert valid is True
    assert msg == "Valid"

    # Out of global range
    valid_oob, _ = location_svc.validate_coordinates(95.0, 77.0)
    assert valid_oob is False


def test_location_identification_service(location_svc):
    """Test location identification for Patna (Ganga basin)."""
    meta = location_svc.identify_location(25.5941, 85.1376)
    assert meta["latitude"] == 25.5941
    assert meta["longitude"] == 85.1376
    assert "basin" in meta and len(meta["basin"]) > 0
    assert "state" in meta and len(meta["state"]) > 0
    assert "district" in meta


def test_feature_retrieval_and_validation(feature_svc):
    """Test automated feature retrieval and physical validation for Guwahati."""
    res = feature_svc.retrieve_location_data(26.1856, 91.7490)
    assert res["validation_passed"] is True
    assert len(res["features"]) == 27
    assert "status_panel" in res
    assert "sources_panel" in res

    # Verify status panel contains all required items
    for item in ["Coordinates", "Location", "Rainfall", "Weather", "Terrain", "Soil Moisture", "Satellite", "Exposure"]:
        assert item in res["status_panel"]
        assert "✓" in res["status_panel"][item] or "⚠" in res["status_panel"][item]


def test_prediction_service_contract(feature_svc, pred_svc):
    """Test model prediction, probability, severity, and uncertainty."""
    res = feature_svc.retrieve_location_data(28.6644, 77.2389)
    pred_res = pred_svc.predict(res["features"])

    assert 0.0 <= pred_res["hazard_probability"] <= 1.0
    assert 0.0 <= pred_res["risk_score"] <= 100.0
    assert pred_res["severity"] in [0, 1, 2, 3]
    assert pred_res["severity_label"] in [
        "None (Baseline)",
        "Minor (Localized Waterlogging)",
        "Moderate (Riverine Spillover)",
        "Severe / Catastrophic (Major Inundation)"
    ]
    assert pred_res["epistemic_uncertainty_pct"] >= 0.0
    assert "Random Forest" in pred_res["model_name"]


def test_local_1km_risk_analysis(feature_svc, pred_svc, location_svc, local_risk_svc):
    """Test 1 km geodesic circular buffer analysis and exposure calculation."""
    lat, lon = 9.9312, 76.2673
    loc_meta = location_svc.identify_location(lat, lon)
    feat_res = feature_svc.retrieve_location_data(lat, lon)
    pred_res = pred_svc.predict(feat_res["features"])

    local_risk = local_risk_svc.compute_local_1km_risk(lat, lon, feat_res["features"], pred_res, loc_meta)

    assert local_risk["analysis_radius_km"] == 1.0
    assert local_risk["analysis_area_km2"] > 3.0  # π * 1² ≈ 3.14 km²
    assert "exposure_details" in local_risk
    assert local_risk["exposure_details"]["population_exposed"] >= 0
    assert local_risk["exposure_details"]["buildings_exposed"] >= 0
    assert local_risk["exposure_details"]["roads_exposed_km"] >= 0
    assert len(local_risk["main_risk_drivers"]) > 0

    # Test local map generation produces valid HTML string
    map_html = local_risk_svc.generate_local_1km_map(lat, lon, local_risk, loc_meta)
    assert "<!DOCTYPE html>" in map_html or "<div" in map_html or "leaflet" in map_html


def test_resilience_planning_service(feature_svc, pred_svc, location_svc, local_risk_svc, resilience_svc):
    """Test resilience response planning across all 7 operational pillars with disclaimer."""
    lat, lon = 28.6644, 77.2389
    loc_meta = location_svc.identify_location(lat, lon)
    feat_res = feature_svc.retrieve_location_data(lat, lon)
    pred_res = pred_svc.predict(feat_res["features"])
    local_risk = local_risk_svc.compute_local_1km_risk(lat, lon, feat_res["features"], pred_res, loc_meta)

    plan = resilience_svc.generate_plan(local_risk, pred_res, loc_meta, feat_res["features"])

    assert "response_priority" in plan
    assert len(plan["pillars"]) == 7
    assert "disclaimer" in plan
    assert "not official government instructions" in plan["disclaimer"].lower()

    # Check mandatory evacuation disclaimer rule (Section 36)
    evac_items = plan["pillars"]["🚶 Evacuation Preparedness (6-24h)"]
    assert any("detailed evacuation routing requires verified road-network" in item.lower() for item in evac_items)
