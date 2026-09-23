"""
Unit Tests for Geospatial Risk Engine and Mapping.
"""

import os
import sys
import pytest
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.geospatial.risk_mapper import SpatialRiskEngine, get_risk_category

MAPS_DIR = os.path.join(BASE_DIR, "outputs", "maps")


def test_risk_category_mapping():
    """Verify that score ranges map to the project-defined categories."""
    assert get_risk_category(10)[0] == "Very Low"
    assert get_risk_category(30)[0] == "Low"
    assert get_risk_category(50)[0] == "Moderate"
    assert get_risk_category(70)[0] == "High"
    assert get_risk_category(90)[0] == "Very High"


def test_spatial_engine_point_risk():
    """Verify point risk computation outputs strictly valid ranges."""
    engine = SpatialRiskEngine()
    sample = {
        "rainfall_1d": 100.0,
        "rainfall_3d": 200.0,
        "rainfall_7d": 300.0,
        "rainfall_14d": 400.0,
        "rainfall_anomaly_3d": 50.0,
        "api_14d": 120.0,
        "temp_mean_3d": 25.0,
        "humidity_mean_3d": 90.0,
        "soil_moisture_root_zone": 0.45,
        "soil_moisture_deep": 0.40,
        "soil_saturation_ratio": 0.94,
        "elevation_m": 8.0,
        "slope_deg": 2.5,
        "dist_river_km": 0.2,
        "twi_proxy": 7.5,
        "elevation_risk_factor": 0.85,
        "sar_vv": -16.0,
        "sar_vh": -22.0,
        "sar_cross_ratio": 1.37,
        "ndwi_proxy": 0.6,
        "ndvi_proxy": 0.4,
        "population_density": 1500.0,
        "exposure_index": 0.86,
        "drainage_vulnerability": 0.82,
        "vulnerability_index": 0.84
    }
    res = engine.compute_point_risk(sample)
    
    assert 0.0 <= res["hazard_probability"] <= 1.0
    assert 0.0 <= res["exposure_index"] <= 1.0
    assert 0.0 <= res["vulnerability_index"] <= 1.0
    assert 0.0 <= res["risk_score"] <= 100.0
    assert res["risk_category"] in ["Very Low", "Low", "Moderate", "High", "Very High"]


def test_map_generation_and_geojson_export():
    """Verify that Folium map HTML and spatial GeoJSON are created."""
    engine = SpatialRiskEngine()
    df_res, map_path = engine.generate_basin_risk_map("monsoon_surge")
    
    assert os.path.exists(map_path), "Map HTML was not generated"
    geojson_path = os.path.join(MAPS_DIR, "kerala_spatial_risk.geojson")
    assert os.path.exists(geojson_path), "GeoJSON was not generated"
    assert len(df_res) == 16, "All 16 stations must be mapped"
