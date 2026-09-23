"""
Historical Flood Analyzer & Low-Flood Vulnerability Diagnostic Engine.
Generates multi-year historical time-series (2018-2023) with marked historical flood occurrences
and diagnoses chronic low-flood susceptibility (areas prone to inundation even with moderate rainfall).
"""

import os
import sys
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.geospatial.location_service import INDIA_STATIONS, get_location_profile


class HistoricalFloodAnalyzer:
    def __init__(self):
        self.stations = {st["id"]: st for st in INDIA_STATIONS}

    def get_historical_time_series(self, lat, lon, station_id=None):
        """
        Generate or retrieve multi-year daily time series (2018-2023) with historical flood events.
        """
        profile = get_location_profile(lat, lon, station_id=station_id)
        flood_years = profile.get("historical_flood_years", [2019, 2021])
        is_chronic = profile.get("chronic_low_flood_prone", False)

        # Generate continuous daily date range from 2018 to 2023
        dates = pd.date_range(start="2018-01-01", end="2023-12-31", freq="D")
        
        # Seasonal monsoon dynamics across Indian sub-regions
        # Southwest monsoon: June to September (doy 152 to 273)
        # Northeast monsoon: October to November (doy 274 to 334)
        np.random.seed(int(abs(lat * 100 + lon * 10)) % 10000)
        
        records = []
        for d in dates:
            doy = d.dayofyear
            year = d.year
            is_sw_monsoon = 152 <= doy <= 273
            is_ne_monsoon = 274 <= doy <= 334
            
            # Base rainfall
            if is_sw_monsoon:
                rain_prob = 0.55
                rain_mean = 28.0
            elif is_ne_monsoon and ("Tamil" in profile["state"] or "Kerala" in profile["state"]):
                rain_prob = 0.50
                rain_mean = 25.0
            else:
                rain_prob = 0.12
                rain_mean = 4.0

            has_rain = np.random.rand() < rain_prob
            rain = np.random.exponential(rain_mean) if has_rain else 0.0

            # Historical flood peaks during known disaster years
            is_flood_day = False
            if year in flood_years:
                # August surge (typical for Ganga, Yamuna, Periyar, Godavari)
                if (d.month == 8 and 10 <= d.day <= 18) or (d.month == 7 and 12 <= d.day <= 16 and "Delhi" in profile["name"]) or (d.month == 6 and 15 <= d.day <= 22 and "Assam" in profile["state"]):
                    rain = np.random.uniform(90.0, 180.0)
                    is_flood_day = True

            # If chronic low-flood prone, even 45mm rainfall in saturated periods triggers localized inundation
            if is_chronic and rain >= 40.0 and (is_sw_monsoon or is_ne_monsoon) and np.random.rand() < 0.40:
                is_flood_day = True

            # Soil moisture follows antecedent rainfall
            sm = min(0.18 + (rain / 250.0) + (0.15 if is_sw_monsoon else 0.0), 0.48)

            records.append({
                "date": d.strftime("%Y-%m-%d"),
                "rainfall_mm": round(float(rain), 1),
                "soil_moisture": round(float(sm), 3),
                "flood_occurred": 1 if is_flood_day else 0,
                "hazard_level": 3 if is_flood_day else (2 if rain > 60 else (1 if rain > 25 else 0))
            })

        df_hist = pd.DataFrame(records)
        df_hist["date"] = pd.to_datetime(df_hist["date"])
        
        # Calculate 3-day rolling rainfall
        df_hist["rainfall_3d"] = df_hist["rainfall_mm"].rolling(3, min_periods=1).sum()

        return profile, df_hist

    def diagnose_chronic_vulnerability(self, profile):
        """
        Evaluate and diagnose why a location suffers flood risk EVEN WITH LOW/MODERATE rainfall.
        """
        elev = profile["elevation_m"]
        slope = profile["slope_deg"]
        dist_riv = profile["dist_river_km"]
        pop = profile["population_density"]
        is_chronic = profile.get("chronic_low_flood_prone", False)
        reason = profile.get("chronic_reason", "Standard river floodplain vulnerability.")

        # Compute vulnerability components
        elev_score = float(np.exp(-elev / 50.0))
        slope_deficit = float(1.0 / (1.0 + np.radians(max(slope, 0.1))))
        river_proximity = float(1.0 / (1.0 + dist_riv))
        
        chronic_factors = []
        if elev <= 15.0:
            chronic_factors.append({
                "factor": "Severe Elevation Deficit",
                "severity": "CRITICAL" if elev <= 5.0 else "HIGH",
                "detail": f"Elevation is only {elev}m above sea level, providing virtually zero gravity drainage gradient."
            })
        if slope <= 1.2:
            chronic_factors.append({
                "factor": "Flat Runoff Bottleneck",
                "severity": "HIGH",
                "detail": f"Slope angle is merely {slope:.1f}°, causing rapid surface water accumulation rather than runoff discharge."
            })
        if dist_riv <= 0.4:
            chronic_factors.append({
                "factor": "Extreme River Channel Proximity",
                "severity": "CRITICAL",
                "detail": f"Located within {dist_riv*1000:.0f}m of major river channel; highly susceptible to bank overtopping and storm drain backflow."
            })
        if pop >= 2500:
            chronic_factors.append({
                "factor": "High Impervious Urban Exposure",
                "severity": "HIGH",
                "detail": f"Dense urban population ({pop:.0f} ppl/km²) with high paved surface fraction preventing natural rainwater infiltration."
            })

        return {
            "is_chronic_low_flood_prone": is_chronic,
            "primary_geomorphic_cause": reason,
            "elevation_m": elev,
            "slope_deg": slope,
            "dist_river_km": dist_riv,
            "population_density": pop,
            "contributing_bottlenecks": chronic_factors,
            "recommended_threshold_adjustment": "Issue advisory watch when 3-day rainfall exceeds 40mm (vs standard 80mm threshold) due to chronic terrain vulnerability."
        }


if __name__ == "__main__":
    analyzer = HistoricalFloodAnalyzer()
    prof, df = analyzer.get_historical_time_series(26.1856, 91.7490)  # Guwahati
    print(f"Historical Records for {prof['name']}: {len(df)}")
    print(f"Total flood days in 2018-2023: {df['flood_occurred'].sum()}")
    diag = analyzer.diagnose_chronic_vulnerability(prof)
    print("Chronic Vulnerability:", diag["is_chronic_low_flood_prone"])
    print("Primary Cause:", diag["primary_geomorphic_cause"])
