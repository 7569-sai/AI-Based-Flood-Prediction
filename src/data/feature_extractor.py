"""
Dedicated Feature Extraction Service for FloodAI Engine.
Manages multi-source data extraction from Open-Meteo, Open-Elevation/SRTM,
National River Hydrology network, Sentinel-1 SAR models, and CIESIN GPWv4.
Performs unit conversion, physical limits validation, and temporal leakage checks.
"""

import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data.api_feature_extractor import APIFeatureExtractor, EXPECTED_FEATURE_ORDER


class FeatureExtractor:
    """
    Automated feature retrieval service that connects to real data APIs
    and formats the exact 27-feature vector for the FloodAI prediction models.
    """
    def __init__(self, timeout_sec=5):
        self.extractor = APIFeatureExtractor(timeout_sec=timeout_sec)
        self.expected_order = EXPECTED_FEATURE_ORDER

    def retrieve_location_data(self, lat, lon, date_str=None):
        """
        Extracts all meteorological, terrain, hydrological, remote sensing,
        and demographic features for a given geographic point in India.
        """
        raw_res = self.extractor.extract_features_for_location(lat, lon, date_str=date_str)
        
        # Format clean retrieval status panel matching Section 21 specification
        status_panel = {
            "Coordinates": "✓ Retrieved",
            "Location": "✓ Retrieved" if raw_res["location_info"].get("state") != "India" else "✓ Retrieved (Basin Index)",
            "Rainfall": raw_res["status_panel"]["🌧 Rainfall"],
            "Weather": raw_res["status_panel"]["🌡 Temperature"],
            "Terrain": raw_res["status_panel"]["⛰ Elevation"],
            "Soil Moisture": raw_res["status_panel"]["💧 Soil Moisture"],
            "Satellite": raw_res["status_panel"]["🛰 Satellite Radar"],
            "Exposure": raw_res["status_panel"]["👥 Population Exposure"]
        }

        return {
            "location_info": raw_res["location_info"],
            "status_panel": status_panel,
            "sources_panel": raw_res["sources_panel"],
            "features": raw_res["features"],
            "validation_passed": raw_res["validation_passed"],
            "validation_errors": raw_res["validation_errors"]
        }

    def validate_features(self, feature_dict):
        """
        Validates feature vector against required physical ranges and schema contracts.
        """
        return self.extractor.validate_feature_vector(feature_dict)
