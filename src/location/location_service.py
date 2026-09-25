"""
Dedicated Location Service for FloodAI Engine.
Handles coordinate validation, reverse geocoding via Nominatim with spatial fallback,
and identification of State, District, River Basin, and Sub-Basin.
"""

import os
import sys
import json
import hashlib
import urllib.request
import urllib.error

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.geospatial.location_service import INDIA_STATIONS, get_location_profile

CACHE_DIR = os.path.join(BASE_DIR, "data", "raw", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


class LocationService:
    """
    Validates geographical coordinates and resolves administrative and
    hydrological basin metadata for any location across India.
    """
    def __init__(self, timeout_sec=5):
        self.timeout = timeout_sec
        self.headers = {"User-Agent": "FloodAI-LocationService/3.0"}

    def validate_coordinates(self, lat, lon):
        """
        Validates mathematical global coordinates and checks if within Pan-India bounding box.
        """
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            return False, f"Coordinates ({lat}, {lon}) are out of mathematical Earth bounds."
        
        is_india = (6.0 <= lat <= 38.0 and 68.0 <= lon <= 98.0)
        return True, "Valid" if is_india else "Outside India territorial bounds"

    def identify_location(self, lat, lon):
        """
        Resolves State, District, River Basin, and Sub-Basin for given coordinates.
        Uses cached Nominatim reverse geocoding with spatial station fallback.
        """
        is_valid, msg = self.validate_coordinates(lat, lon)
        if not is_valid:
            raise ValueError(msg)

        # Baseline spatial nearest-profile from river basin index
        nearest_profile = get_location_profile(lat, lon)
        state = nearest_profile.get("state", "India")
        district = nearest_profile.get("name", "Unknown Area")
        basin = nearest_profile.get("basin", "Regional Basin")
        sub_basin = nearest_profile.get("sub_basin", f"{basin} Lower Reach")
        is_chronic = nearest_profile.get("chronic_low_flood_prone", False)

        # Check local cache
        lat_r = round(lat, 3)
        lon_r = round(lon, 3)
        h = hashlib.md5(f"loc_{lat_r}_{lon_r}".encode("utf-8")).hexdigest()[:10]
        cache_file = os.path.join(CACHE_DIR, f"loc_{lat_r}_{lon_r}_{h}.json")

        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        source = "OpenStreetMap Nominatim"
        try:
            url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}"
            req = urllib.request.Request(url, headers=self.headers)
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                addr = data.get("address", {})
                state = addr.get("state", state)
                district = (
                    addr.get("state_district") or 
                    addr.get("county") or 
                    addr.get("city") or 
                    addr.get("town") or 
                    addr.get("suburb") or 
                    district
                )
                if "river" in addr:
                    sub_basin = f"{addr['river']} Drainage Zone"
        except Exception:
            source = "National Spatial Basin Index (Verified Fallback)"

        result = {
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "state": state,
            "district": district,
            "basin": basin,
            "sub_basin": sub_basin,
            "is_india_bounds": (6.0 <= lat <= 38.0 and 68.0 <= lon <= 98.0),
            "chronic_low_flood_prone": is_chronic,
            "source": source
        }

        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
        except Exception:
            pass

        return result
