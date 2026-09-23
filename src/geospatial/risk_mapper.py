"""
Geospatial Risk Mapping Engine for Kerala River Basins.
Computes Hazard, Exposure, Vulnerability, and Composite Risk (0-100 score).
Generates interactive multi-layer Folium maps and GeoJSON spatial outputs.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import folium
from folium import plugins

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models", "trained")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MAPS_DIR = os.path.join(BASE_DIR, "outputs", "maps")
STATIC_CACHE_PATH = os.path.join(BASE_DIR, "data", "raw", "kerala_static_layers.json")

os.makedirs(MAPS_DIR, exist_ok=True)


def get_risk_category(risk_score):
    """Categorize project-defined 0-100 risk score."""
    if risk_score <= 20:
        return "Very Low", "#2ca02c"   # Green
    elif risk_score <= 40:
        return "Low", "#98df8a"        # Light Green
    elif risk_score <= 60:
        return "Moderate", "#ffbb78"   # Orange-Yellow
    elif risk_score <= 80:
        return "High", "#ff7f0e"       # Orange
    else:
        return "Very High", "#d62728"  # Red


class SpatialRiskEngine:
    def __init__(self, use_pan_india=True):
        pan_india_model = os.path.join(MODELS_DIR, "model_pan_india_random_forest.joblib")
        kerala_model = os.path.join(MODELS_DIR, "model_random_forest.joblib")
        pan_india_meta = os.path.join(BASE_DIR, "data", "processed", "india", "pan_india_metadata.json")

        if use_pan_india and os.path.exists(pan_india_model) and os.path.exists(pan_india_meta):
            model_path = pan_india_model
            meta_path = pan_india_meta
        else:
            model_path = kerala_model
            meta_path = os.path.join(PROCESSED_DIR, "feature_metadata.json")

        assert os.path.exists(model_path), f"Champion model not found at {model_path}!"
        self.model = joblib.load(model_path)
        
        with open(meta_path, "r") as f:
            self.metadata = json.load(f)
        self.feature_cols = self.metadata["feature_cols"]

    def compute_point_risk(self, feature_dict):
        """
        Compute Hazard, Exposure, Vulnerability, and Risk score for an input feature dictionary.
        Returns detailed components dict.
        """
        # 1. Prepare feature vector
        row = [feature_dict.get(col, 0.0) for col in self.feature_cols]
        X = pd.DataFrame([row], columns=self.feature_cols)

        # 2. Hazard: Model Predicted Probability
        hazard_prob = float(self.model.predict_proba(X)[0, 1])

        # 3. Exposure Index
        pop_density = float(feature_dict.get("population_density", 850.0))
        exposure_index = float(np.clip(np.log1p(pop_density) / np.log1p(5000.0), 0.0, 1.0))

        # 4. Vulnerability Index
        elev = float(feature_dict.get("elevation_m", 25.0))
        slope = float(feature_dict.get("slope_deg", 3.0))
        dist_river = float(feature_dict.get("dist_river_km", 1.0))

        v_elev = float(np.exp(-elev / 50.0))
        v_river = float(1.0 / (1.0 + dist_river))
        v_slope = float(1.0 / (1.0 + np.radians(max(slope, 0.1))))
        
        drainage_vuln = (0.45 * v_elev) + (0.35 * v_river) + (0.20 * v_slope)
        vulnerability_index = float(np.clip((0.6 * drainage_vuln) + (0.4 * exposure_index), 0.0, 1.0))

        # 5. Composite Risk Score (0-100)
        risk_score = round(float((0.50 * hazard_prob + 0.25 * exposure_index + 0.25 * vulnerability_index) * 100.0), 2)
        category, color_hex = get_risk_category(risk_score)

        return {
            "hazard_probability": round(hazard_prob, 4),
            "exposure_index": round(exposure_index, 4),
            "vulnerability_index": round(vulnerability_index, 4),
            "drainage_vulnerability": round(drainage_vuln, 4),
            "risk_score": risk_score,
            "risk_category": category,
            "risk_color": color_hex,
            "prediction": int(hazard_prob >= 0.5)
        }

    def generate_basin_risk_map(self, scenario="monsoon_surge"):
        """
        Generate interactive Folium flood risk map for all 16 Kerala stations under a designated scenario.
        Scenarios: 'normal', 'monsoon_surge', 'extreme_flood_2018'
        """
        df_feat = pd.read_csv(os.path.join(PROCESSED_DIR, "full_features.csv"))

        # Select representative date for scenario
        if scenario == "extreme_flood_2018":
            target_date = "2018-08-16"  # Historic peak
        elif scenario == "monsoon_surge":
            target_date = "2021-10-18"  # High monsoon surge
        else:
            target_date = "2023-08-15"  # Normal wet day

        df_scenario = df_feat[df_feat["date"] == target_date].copy()
        if len(df_scenario) == 0:
            df_scenario = df_feat.groupby("station_id").first().reset_index()

        # Compute risk for each station
        records = []
        for _, row in df_scenario.iterrows():
            f_dict = row.to_dict()
            res = self.compute_point_risk(f_dict)
            rec = {
                "station_id": row["station_id"],
                "station_name": row["station_name"],
                "basin": row["basin"],
                "district": row["district"],
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "elevation_m": float(row["elevation_m"]),
                "slope_deg": float(row["slope_deg"]),
                "rainfall_3d": float(row["rainfall_3d"]),
                "soil_saturation_ratio": float(row["soil_saturation_ratio"]),
                **res
            }
            records.append(rec)

        df_results = pd.DataFrame(records)

        # -------------------------------------------------------------
        # Build Folium Map
        # -------------------------------------------------------------
        kerala_center = [10.2, 76.5]
        m = folium.Map(location=kerala_center, zoom_start=8, tiles="OpenStreetMap")

        # Layer groups for toggling
        risk_group = folium.FeatureGroup(name="Composite Risk Score (0-100)", show=True)
        hazard_group = folium.FeatureGroup(name="Flood Hazard Probability", show=False)
        vuln_group = folium.FeatureGroup(name="Vulnerability Index", show=False)
        exposure_group = folium.FeatureGroup(name="Population Exposure", show=False)

        for _, r in df_results.iterrows():
            # Composite Risk Marker
            popup_html = f"""
            <div style="font-family: Arial; width: 220px;">
                <h4 style="margin: 0 0 5px 0; color: #1a365d;">{r['station_name']}</h4>
                <b>Basin:</b> {r['basin']} ({r['district']})<br>
                <b>Elevation:</b> {r['elevation_m']} m | <b>Slope:</b> {r['slope_deg']:.1f}°<br>
                <hr style="margin: 6px 0;">
                <b>3-Day Rainfall:</b> {r['rainfall_3d']:.1f} mm<br>
                <b>Soil Saturation:</b> {r['soil_saturation_ratio']*100:.1f}%<br>
                <hr style="margin: 6px 0;">
                <b>Hazard (H):</b> {r['hazard_probability']:.2f}<br>
                <b>Exposure (E):</b> {r['exposure_index']:.2f}<br>
                <b>Vulnerability (V):</b> {r['vulnerability_index']:.2f}<br>
                <div style="margin-top: 8px; padding: 5px; background: {r['risk_color']}; color: white; border-radius: 4px; text-align: center; font-weight: bold;">
                    Risk Score: {r['risk_score']} ({r['risk_category']})
                </div>
            </div>
            """
            
            # Risk marker
            folium.CircleMarker(
                location=[r["latitude"], r["longitude"]],
                radius=10 + (r["risk_score"] / 8.0),
                color=r["risk_color"],
                fill=True,
                fill_color=r["risk_color"],
                fill_opacity=0.85,
                weight=2,
                popup=folium.Popup(popup_html, max_width=260),
                tooltip=f"{r['station_name']}: Risk {r['risk_score']} ({r['risk_category']})"
            ).add_to(risk_group)

            # Hazard marker
            h_color = "#d62728" if r["hazard_probability"] >= 0.5 else "#2ca02c"
            folium.CircleMarker(
                location=[r["latitude"], r["longitude"]],
                radius=8 + (r["hazard_probability"] * 10),
                color=h_color,
                fill=True,
                fill_color=h_color,
                fill_opacity=0.7,
                tooltip=f"{r['station_name']}: Hazard P={r['hazard_probability']:.2f}"
            ).add_to(hazard_group)

            # Vulnerability marker
            folium.CircleMarker(
                location=[r["latitude"], r["longitude"]],
                radius=8 + (r["vulnerability_index"] * 10),
                color="#9467bd",
                fill=True,
                fill_color="#9467bd",
                fill_opacity=0.7,
                tooltip=f"{r['station_name']}: Vulnerability V={r['vulnerability_index']:.2f}"
            ).add_to(vuln_group)

            # Exposure marker
            folium.CircleMarker(
                location=[r["latitude"], r["longitude"]],
                radius=8 + (r["exposure_index"] * 10),
                color="#1f77b4",
                fill=True,
                fill_color="#1f77b4",
                fill_opacity=0.7,
                tooltip=f"{r['station_name']}: Exposure E={r['exposure_index']:.2f}"
            ).add_to(exposure_group)

        risk_group.add_to(m)
        hazard_group.add_to(m)
        vuln_group.add_to(m)
        exposure_group.add_to(m)

        folium.LayerControl(collapsed=False).add_to(m)

        # Save HTML
        html_out = os.path.join(MAPS_DIR, f"kerala_flood_risk_map_{scenario}.html")
        m.save(html_out)
        default_html = os.path.join(MAPS_DIR, "kerala_flood_risk_map.html")
        m.save(default_html)
        print(f"Interactive risk map exported to: {html_out}")

        # Save GeoJSON
        geojson_features = []
        for _, r in df_results.iterrows():
            feat = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [r["longitude"], r["latitude"]]
                },
                "properties": {k: v for k, v in r.items() if k not in ["latitude", "longitude"]}
            }
            geojson_features.append(feat)

        geojson_obj = {
            "type": "FeatureCollection",
            "metadata": {
                "scenario": scenario,
                "target_date": target_date,
                "study_area": "Kerala River Basins, India",
                "framework": "Risk = Hazard x Exposure x Vulnerability"
            },
            "features": geojson_features
        }

        geojson_out = os.path.join(MAPS_DIR, "kerala_spatial_risk.geojson")
        with open(geojson_out, "w", encoding="utf-8") as f:
            json.dump(geojson_obj, f, indent=2)
        print(f"Spatial GeoJSON exported to: {geojson_out}")

        return df_results, html_out

    def generate_location_focus_map(self, lat, lon, profile, risk_res, weather_params):
        """
        Generate high-resolution Folium map centered on a specific location/coordinates in India,
        showing risk buffers (5km, 15km, 30km) and surrounding national stations.
        """
        from src.geospatial.location_service import INDIA_STATIONS

        m = folium.Map(location=[lat, lon], zoom_start=8, tiles="OpenStreetMap")
        
        # Buffer zones around selected location
        folium.Circle(
            location=[lat, lon],
            radius=5000,
            color="#d62728",
            fill=True,
            fill_color="#d62728",
            fill_opacity=0.18,
            weight=1.5,
            tooltip="5 km Core Inundation Impact Buffer"
        ).add_to(m)

        folium.Circle(
            location=[lat, lon],
            radius=15000,
            color="#ff7f0e",
            fill=True,
            fill_color="#ff7f0e",
            fill_opacity=0.10,
            weight=1.5,
            dash_array="5, 5",
            tooltip="15 km Advisory Catchment Zone"
        ).add_to(m)

        folium.Circle(
            location=[lat, lon],
            radius=30000,
            color="#1f77b4",
            fill=False,
            weight=1,
            dash_array="8, 8",
            tooltip="30 km Regional Basin Domain"
        ).add_to(m)

        # Surrounding National Stations
        for st in INDIA_STATIONS:
            if abs(st["lat"] - lat) < 0.01 and abs(st["lon"] - lon) < 0.01:
                continue  # skip exact target station, it gets the primary pin
            st_color = "#e41a1c" if st.get("chronic_low_flood_prone", False) else "#3182bd"
            folium.CircleMarker(
                location=[st["lat"], st["lon"]],
                radius=6,
                color=st_color,
                fill=True,
                fill_color=st_color,
                fill_opacity=0.75,
                tooltip=f"{st['name']} ({st['basin']} Basin)"
            ).add_to(m)

        # Target Location Marker (Pin)
        popup_html = f"""
        <div style="font-family: Arial; width: 230px;">
            <h4 style="margin: 0 0 5px 0; color: #1a365d;">{profile.get('name', 'Analyzed Location')}</h4>
            <b>Basin:</b> {profile.get('basin', 'Regional')} ({profile.get('state', 'India')})<br>
            <b>Coordinates:</b> {lat:.3f}°N, {lon:.3f}°E<br>
            <b>Elevation:</b> {weather_params.get('elevation_m', 0.0)} m | <b>Slope:</b> {weather_params.get('slope_deg', 0.0):.1f}°<br>
            <b>3-Day Rainfall:</b> {weather_params.get('rainfall_3d', 0.0):.1f} mm<br>
            <b>Hazard Probability:</b> {risk_res.get('hazard_probability', 0.0)*100:.1f}%<br>
            <div style="margin-top: 8px; padding: 6px; background: {risk_res.get('risk_color', '#d62728')}; color: white; border-radius: 4px; text-align: center; font-weight: bold;">
                Risk Score: {risk_res.get('risk_score', 0)} / 100 ({risk_res.get('risk_category', 'High')})
            </div>
        </div>
        """
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=f"{profile.get('name', 'Selected Point')} | Risk {risk_res.get('risk_score', 0)}/100",
            icon=folium.Icon(color="red" if risk_res.get("hazard_probability", 0) >= 0.5 else "green", icon="info-sign")
        ).add_to(m)

        html_out = os.path.join(MAPS_DIR, "location_focus_map.html")
        m.save(html_out)
        return m._repr_html_(), html_out

    def generate_pan_india_risk_map(self, intensity="monsoon_surge"):
        """
        Generate interactive Folium flood risk map for all 36 Pan-India stations across 12 basins.
        """
        from src.geospatial.location_service import INDIA_STATIONS, get_simulated_current_weather

        m = folium.Map(location=[22.5, 79.5], zoom_start=5, tiles="OpenStreetMap")
        risk_group = folium.FeatureGroup(name="Composite Risk (0-100)", show=True)
        chronic_group = folium.FeatureGroup(name="Chronic Low-Flood Prone Hotspots", show=True)
        
        station_results = []
        for st in INDIA_STATIONS:
            params = get_simulated_current_weather(st, intensity=intensity)
            r = self.compute_point_risk(params)
            station_results.append({**st, **r, **params})
            
            popup_html = f"""
            <div style="font-family: Arial; width: 220px;">
                <h4 style="margin: 0 0 5px 0; color: #1a365d;">{st['name']}</h4>
                <b>Basin:</b> {st['basin']} ({st['state']})<br>
                <b>Elevation:</b> {st['elevation_m']} m | <b>Slope:</b> {st['slope_deg']:.1f}°<br>
                <b>3-Day Rainfall:</b> {params['rainfall_3d']:.1f} mm<br>
                <b>Hazard (H):</b> {r['hazard_probability']:.2f}<br>
                <b>Composite Risk:</b> {r['risk_score']} ({r['risk_category']})<br>
                {'<b style="color:red;">⚠️ Chronic Low-Flood Susceptible</b>' if st.get('chronic_low_flood_prone') else ''}
            </div>
            """
            
            # Risk marker
            folium.CircleMarker(
                location=[st["lat"], st["lon"]],
                radius=8 + (r["risk_score"] / 10.0),
                color=r["risk_color"],
                fill=True,
                fill_color=r["risk_color"],
                fill_opacity=0.85,
                weight=2,
                popup=folium.Popup(popup_html, max_width=260),
                tooltip=f"{st['name']}: {r['risk_category']} ({r['risk_score']}/100)"
            ).add_to(risk_group)
            
            if st.get("chronic_low_flood_prone", False):
                folium.CircleMarker(
                    location=[st["lat"], st["lon"]],
                    radius=14,
                    color="#e41a1c",
                    fill=False,
                    weight=2.5,
                    dash_array="4, 4",
                    tooltip=f"CHRONIC LOW-FLOOD PRONE: {st['name']}"
                ).add_to(chronic_group)
                
        risk_group.add_to(m)
        chronic_group.add_to(m)
        folium.LayerControl(collapsed=False).add_to(m)
        
        html_out = os.path.join(MAPS_DIR, f"pan_india_risk_map_{intensity}.html")
        m.save(html_out)
        return pd.DataFrame(station_results), html_out


if __name__ == "__main__":
    engine = SpatialRiskEngine()
    print("Generating Geospatial Risk Map for 'monsoon_surge' scenario...")
    df_res, map_file = engine.generate_basin_risk_map("monsoon_surge")
    print("Generating Geospatial Risk Map for 'extreme_flood_2018' scenario...")
    engine.generate_basin_risk_map("extreme_flood_2018")
    print(f"Top 5 High Risk Stations:\n{df_res[['station_name', 'basin', 'hazard_probability', 'vulnerability_index', 'risk_score', 'risk_category']].sort_values('risk_score', ascending=False).head(5)}")
