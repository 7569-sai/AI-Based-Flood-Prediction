"""
Integration Tests for FastAPI Backend Endpoints.
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from api.main import app

client = TestClient(app)


def test_api_health_endpoint():
    """Verify /health returns HTTP 200 and healthy status."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["database_connected"] is True


def test_api_predict_valid_input():
    """Verify /predict returns valid schema with prediction, probability, risk_score."""
    payload = {
        "latitude": 10.1076,
        "longitude": 76.3516,
        "location_name": "Aluva",
        "basin": "Periyar",
        "rainfall_1d": 80.0,
        "rainfall_3d": 170.0,
        "rainfall_7d": 260.0,
        "rainfall_14d": 380.0,
        "soil_moisture_root_zone": 0.42,
        "elevation_m": 10.0,
        "slope_deg": 4.7,
        "dist_river_km": 0.3,
        "population_density": 1359.0
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "prediction" in data
    assert 0.0 <= data["probability"] <= 1.0
    assert 0.0 <= data["risk_score"] <= 100.0
    assert data["hazard_category"] in ["Low", "Moderate", "High", "Very High"]
    assert "inference_id" in data


def test_api_predict_invalid_input():
    """Verify /predict rejects out-of-bound coordinates or negative rainfall with HTTP 422."""
    # Negative rainfall
    bad_payload = {
        "latitude": 10.1076,
        "longitude": 76.3516,
        "rainfall_1d": -50.0,  # Invalid
        "rainfall_3d": 100.0,
        "rainfall_7d": 200.0,
        "rainfall_14d": 300.0,
        "elevation_m": 10.0,
        "slope_deg": 5.0,
        "dist_river_km": 1.0
    }
    resp = client.post("/predict", json=bad_payload)
    assert resp.status_code == 422

    # Latitude outside India region (bounds 6.0 to 38.0)
    bad_coords = {
        "latitude": 45.0,  # Outside India bounds [6.0, 38.0]
        "longitude": 76.3516,
        "rainfall_1d": 50.0,
        "rainfall_3d": 100.0,
        "rainfall_7d": 200.0,
        "rainfall_14d": 300.0,
        "elevation_m": 10.0,
        "slope_deg": 5.0,
        "dist_river_km": 1.0
    }
    resp = client.post("/predict", json=bad_coords)
    assert resp.status_code == 422


def test_api_stations_endpoint():
    """Verify /stations returns 36 Pan-India monitoring stations."""
    resp = client.get("/stations")
    assert resp.status_code == 200
    data = resp.json()
    assert "stations" in data
    assert len(data["stations"]) == 36
    assert data["total_stations"] == 36


def test_api_location_lookup_endpoint():
    """Verify /location-lookup returns environmental profile and simulated weather."""
    payload = {
        "latitude": 28.6644,
        "longitude": 77.2389,
        "intensity": "monsoon_surge"
    }
    resp = client.post("/location-lookup", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "profile" in data
    assert "environmental_parameters" in data
    assert "risk_result" in data
    assert data["profile"]["basin"] == "Yamuna"


def test_api_historical_timeline_endpoint():
    """Verify /historical-timeline returns multi-year records and chronic diagnostics."""
    resp = client.get("/historical-timeline?latitude=26.1856&longitude=91.7490")
    assert resp.status_code == 200
    data = resp.json()
    assert "records_count" in data
    assert data["records_count"] > 1000
    assert data["chronic_vulnerability"]["is_chronic_low_flood_prone"] is True


def test_api_risk_score_endpoint():
    """Verify /risk-score computes composite score and returns category."""
    payload = {
        "hazard_probability": 0.85,
        "population_density": 1500.0,
        "elevation_m": 8.0,
        "slope_deg": 2.0,
        "dist_river_km": 0.2
    }
    resp = client.post("/risk-score", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["risk_score"] > 0
    assert "Very High" in data["risk_category"] or "High" in data["risk_category"]


def test_api_explain_endpoint():
    """Verify /explain returns SHAP factor breakdown."""
    payload = {
        "latitude": 10.1076,
        "longitude": 76.3516,
        "rainfall_1d": 75.0,
        "rainfall_3d": 150.0,
        "rainfall_7d": 250.0,
        "rainfall_14d": 350.0,
        "elevation_m": 10.0,
        "slope_deg": 4.7,
        "dist_river_km": 0.3
    }
    resp = client.post("/explain", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "factors_increasing_risk" in data
    assert "factors_decreasing_risk" in data
    assert len(data["factors_increasing_risk"]) > 0


def test_api_recommendations_endpoint():
    """Verify /recommendations returns prioritized actions with disclaimer."""
    resp = client.get("/recommendations?hazard_prob=0.88&exposure_index=0.8&vulnerability_index=0.85")
    assert resp.status_code == 200
    data = resp.json()
    assert "not official government guidance" in data["disclaimer"]
    assert data["urgency_level"] == "CRITICAL / EMERGENCY"
    assert len(data["emergency_actions_immediate"]) > 0
