"""
FastAPI Application for AI-Based Flood Prediction and Resilience Planning.
Provides RESTful APIs for inference, risk scoring, SHAP explainability,
geospatial basin risk analysis, and resilience planning.
"""

import os
import sys
import json
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from api.schemas import (
    FloodPredictionRequest, FloodPredictionResponse,
    RiskScoreRequest, RiskScoreResponse,
    ExplainResponse, GeospatialRiskRequest,
    RecommendationsResponse,
    LocationLookupRequest, LocationLookupResponse,
    HistoricalTimelineResponse
)
from src.geospatial.risk_mapper import SpatialRiskEngine, get_risk_category
from src.explainability.shap_explainer import FloodShapExplainer
from src.resilience.recommendation_engine import ResilienceEngine
from src.geospatial.location_service import get_location_profile, get_simulated_current_weather, INDIA_STATIONS
from src.data.historical_analyzer import HistoricalFloodAnalyzer
from src.utils.database import log_prediction, SessionLocal, LocationRecord, ModelMetricRecord, init_db

app = FastAPI(
    title="AI-Based Flood Prediction & Resilience Planning API (Pan-India)",
    description="Operational 4th-Year Major Project API for hydrometeorological flood forecasting, composite risk scoring, SHAP explainability, and rule-based resilience planning across Indian river basins.",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engines
init_db()
risk_engine = SpatialRiskEngine()
explainer = FloodShapExplainer()
resilience_engine = ResilienceEngine()
historical_analyzer = HistoricalFloodAnalyzer()


def _complete_feature_dict(req: FloodPredictionRequest) -> Dict[str, float]:
    """Populate default or derived values for any missing features."""
    d = req.model_dump()
    
    # Calculate derived indices if not supplied
    if d.get("elevation_risk_factor") is None:
        import numpy as np
        d["elevation_risk_factor"] = float(np.exp(-d["elevation_m"] / 50.0))
        
    if d.get("soil_saturation_ratio") is None:
        d["soil_saturation_ratio"] = min(d["soil_moisture_root_zone"] / 0.48, 1.0)
        
    if d.get("twi_proxy") is None:
        import numpy as np
        slope_rad = np.radians(max(d["slope_deg"], 0.5))
        d["twi_proxy"] = float(np.log(100.0 / np.tan(slope_rad)))

    # Vulnerability & exposure proxies
    import numpy as np
    pop = d.get("population_density", 850.0)
    d["exposure_index"] = float(np.clip(np.log1p(pop) / np.log1p(5000.0), 0.0, 1.0))
    
    v_elev = d["elevation_risk_factor"]
    v_riv = float(1.0 / (1.0 + d["dist_river_km"]))
    v_slp = float(1.0 / (1.0 + np.radians(max(d["slope_deg"], 0.1))))
    d["drainage_vulnerability"] = float((0.45 * v_elev) + (0.35 * v_riv) + (0.20 * v_slp))
    d["vulnerability_index"] = float(np.clip((0.6 * d["drainage_vulnerability"]) + (0.4 * d["exposure_index"]), 0.0, 1.0))
    
    return d


# ---------------------------------------------------------------------------
# 1. Health Endpoint
# ---------------------------------------------------------------------------
@app.get("/health", tags=["System"])
def health_check():
    """Verify backend API readiness, model availability, and database connectivity."""
    db_ok = False
    try:
        from sqlalchemy import text
        session = SessionLocal()
        session.execute(text("SELECT 1"))
        session.close()
        db_ok = True
    except Exception as err:
        db_ok = False

    return {
        "status": "healthy",
        "service": "AI Flood Prediction & Resilience Engine",
        "database_connected": db_ok,
        "champion_model": "RandomForest_v1.0 (Balanced)",
        "features_count": len(risk_engine.feature_cols)
    }


# ---------------------------------------------------------------------------
# 2. Flood Prediction Endpoint
# ---------------------------------------------------------------------------
@app.post("/predict", response_model=FloodPredictionResponse, tags=["Inference"])
def predict_flood(payload: FloodPredictionRequest):
    """
    Predict 24- to 72-hour flood hazard occurrence, probability, and composite risk score.
    """
    try:
        full_dict = _complete_feature_dict(payload)
        risk_res = risk_engine.compute_point_risk(full_dict)
        
        # Hazard severity categorization
        prob = risk_res["hazard_probability"]
        if prob < 0.25:
            h_lvl = 0
            h_cat = "Low"
        elif prob < 0.50:
            h_lvl = 1
            h_cat = "Moderate"
        elif prob < 0.75:
            h_lvl = 2
            h_cat = "High"
        else:
            h_lvl = 3
            h_cat = "Very High"

        # Log into SQLite DB
        inf_id = log_prediction(
            latitude=payload.latitude,
            longitude=payload.longitude,
            probability=prob,
            risk_score=risk_res["risk_score"],
            risk_category=risk_res["risk_category"],
            hazard_level=h_lvl,
            input_dict=full_dict
        )

        return FloodPredictionResponse(
            prediction=risk_res["prediction"],
            probability=prob,
            hazard_level=h_lvl,
            hazard_category=h_cat,
            risk_score=risk_res["risk_score"],
            risk_category=risk_res["risk_category"],
            risk_color=risk_res["risk_color"],
            model_version="RandomForest_v1.0",
            inference_id=inf_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


# ---------------------------------------------------------------------------
# 3. Risk Score Endpoint
# ---------------------------------------------------------------------------
@app.post("/risk-score", response_model=RiskScoreResponse, tags=["Risk Analysis"])
def compute_risk_score(payload: RiskScoreRequest):
    """
    Compute composite 0-100 Flood Risk Score from Hazard, Exposure, and Vulnerability.
    """
    try:
        import numpy as np
        h = payload.hazard_probability
        
        # Exposure Index
        e = float(np.clip(np.log1p(payload.population_density) / np.log1p(5000.0), 0.0, 1.0))
        
        # Vulnerability Index
        v_elev = float(np.exp(-payload.elevation_m / 50.0))
        v_riv = float(1.0 / (1.0 + payload.dist_river_km))
        v_slp = float(1.0 / (1.0 + np.radians(max(payload.slope_deg, 0.1))))
        drainage_vuln = (0.45 * v_elev) + (0.35 * v_riv) + (0.20 * v_slp)
        v = float(np.clip((0.6 * drainage_vuln) + (0.4 * e), 0.0, 1.0))
        
        # Risk Score: 50% Hazard + 25% Exposure + 25% Vulnerability
        risk_val = round(float((0.50 * h + 0.25 * e + 0.25 * v) * 100.0), 2)
        category, color_hex = get_risk_category(risk_val)

        return RiskScoreResponse(
            hazard_probability=round(h, 4),
            exposure_index=round(e, 4),
            vulnerability_index=round(v, 4),
            drainage_vulnerability=round(drainage_vuln, 4),
            risk_score=risk_val,
            risk_category=category,
            risk_color=color_hex
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Risk calculation error: {str(e)}")


# ---------------------------------------------------------------------------
# 4. Explainability Endpoint (TreeSHAP)
# ---------------------------------------------------------------------------
@app.post("/explain", response_model=ExplainResponse, tags=["Explainability"])
def explain_prediction(payload: FloodPredictionRequest):
    """
    Decompose prediction into exact Shapley additive values (TreeSHAP).
    Returns top 3 risk-increasing and top 3 risk-mitigating factors.
    """
    try:
        full_dict = _complete_feature_dict(payload)
        exp_res = explainer.explain_instance(full_dict, save_plot=False)
        return ExplainResponse(
            prediction=exp_res["prediction"],
            probability=exp_res["probability"],
            base_value=exp_res["base_value"],
            factors_increasing_risk=exp_res["factors_increasing_risk"],
            factors_decreasing_risk=exp_res["factors_decreasing_risk"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SHAP explanation error: {str(e)}")


# ---------------------------------------------------------------------------
# 5. Geospatial Risk Endpoint
# ---------------------------------------------------------------------------
@app.post("/geospatial-risk", tags=["Geospatial"])
def get_geospatial_risk(payload: GeospatialRiskRequest):
    """
    Retrieve spatial GeoJSON risk data and interactive map reference for Kerala river basins.
    Supported scenarios: 'normal', 'monsoon_surge', 'extreme_flood_2018'
    """
    try:
        df_res, map_file = risk_engine.generate_basin_risk_map(payload.scenario)
        geojson_path = os.path.join(BASE_DIR, "outputs", "maps", "kerala_spatial_risk.geojson")
        with open(geojson_path, "r", encoding="utf-8") as f:
            geojson_data = json.load(f)
        return {
            "scenario": payload.scenario,
            "stations_evaluated": len(df_res),
            "geojson": geojson_data,
            "map_html_path": map_file
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Geospatial processing error: {str(e)}")


# ---------------------------------------------------------------------------
# 6. Model Info Endpoint
# ---------------------------------------------------------------------------
@app.get("/model-info", tags=["Models"])
def get_model_info():
    """Retrieve champion model metadata, hyperparameters, and feature roster."""
    info_path = os.path.join(BASE_DIR, "models", "trained", "final_model_info.json")
    if os.path.exists(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"model": "RandomForest", "status": "active"}


# ---------------------------------------------------------------------------
# 7. Model Metrics Endpoint
# ---------------------------------------------------------------------------
@app.get("/metrics", tags=["Models"])
def get_benchmark_metrics():
    """Retrieve out-of-time test set evaluation metrics across all trained models."""
    metrics_path = os.path.join(BASE_DIR, "outputs", "metrics", "model_comparison.json")
    ablation_path = os.path.join(BASE_DIR, "outputs", "metrics", "ablation_metrics.json")
    
    comp = {}
    ablation = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            comp = json.load(f)
    if os.path.exists(ablation_path):
        with open(ablation_path, "r", encoding="utf-8") as f:
            ablation = json.load(f)

    return {
        "benchmark_comparison": comp,
        "ablation_study": ablation
    }


# ---------------------------------------------------------------------------
# 8. Resilience Recommendations Endpoint
# ---------------------------------------------------------------------------
@app.get("/recommendations", response_model=RecommendationsResponse, tags=["Resilience Planning"])
def get_recommendations(
    hazard_prob: float = Query(..., ge=0.0, le=1.0, description="Hazard probability [0, 1]"),
    exposure_index: float = Query(0.5, ge=0.0, le=1.0, description="Exposure index [0, 1]"),
    vulnerability_index: float = Query(0.5, ge=0.0, le=1.0, description="Vulnerability index [0, 1]"),
    elevation_m: float = Query(15.0, description="Elevation in meters"),
    dist_river_km: float = Query(0.5, description="Distance to river in km"),
    rainfall_3d: float = Query(100.0, description="3-day rainfall accumulation in mm"),
    basin_name: str = Query("Periyar", description="River basin name"),
    location_name: str = Query("Aluva", description="Locality or station name")
):
    """
    Generate context-tailored resilience planning recommendations.
    Tagged with mandatory academic/planning disclaimer.
    """
    recs = resilience_engine.generate_recommendations(
        hazard_prob=hazard_prob,
        exposure_index=exposure_index,
        vulnerability_index=vulnerability_index,
        elevation_m=elevation_m,
        dist_river_km=dist_river_km,
        rainfall_3d=rainfall_3d,
        basin_name=basin_name,
        location_name=location_name
    )
    return RecommendationsResponse(**recs)


# ---------------------------------------------------------------------------
# 9. Pan-India Station Roster Endpoint
# ---------------------------------------------------------------------------
@app.get("/stations", tags=["Geospatial"])
def list_india_stations():
    """Retrieve all 36 national flood-monitoring stations across 12 major Indian river basins."""
    return {
        "total_stations": len(INDIA_STATIONS),
        "basins": list(set(s["basin"] for s in INDIA_STATIONS)),
        "stations": INDIA_STATIONS
    }


# ---------------------------------------------------------------------------
# 10. Location Lookup Endpoint
# ---------------------------------------------------------------------------
@app.post("/location-lookup", response_model=LocationLookupResponse, tags=["Geospatial"])
def lookup_location(payload: LocationLookupRequest):
    """
    Given latitude and longitude (or station_id) anywhere in India:
    1. Identifies the nearest river basin & station profile.
    2. Auto-retrieves realistic hydrometeorological parameters (rainfall, soil moisture, elevation).
    3. Runs real-time inference returning hazard probability, risk score, and category.
    """
    try:
        profile = get_location_profile(payload.latitude, payload.longitude, station_id=payload.station_id)
        weather_params = get_simulated_current_weather(profile, intensity=payload.intensity)
        
        # Compute risk
        risk_res = risk_engine.compute_point_risk(weather_params)
        
        return LocationLookupResponse(
            profile=profile,
            environmental_parameters=weather_params,
            risk_result=risk_res
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Location lookup error: {str(e)}")


# ---------------------------------------------------------------------------
# 11. Historical Timeline & Low-Flood Vulnerability Endpoint
# ---------------------------------------------------------------------------
@app.get("/historical-timeline", response_model=HistoricalTimelineResponse, tags=["Historical Analytics"])
def get_historical_timeline(
    latitude: float = Query(..., ge=6.0, le=38.0),
    longitude: float = Query(..., ge=68.0, le=98.0),
    station_id: Optional[str] = Query(None)
):
    """
    Retrieve 2018-2023 daily historical rainfall, soil moisture, and flood occurrence dates,
    plus diagnostic analysis of chronic low-flood risk vulnerability.
    """
    try:
        profile, df_hist = historical_analyzer.get_historical_time_series(latitude, longitude, station_id=station_id)
        chronic_diag = historical_analyzer.diagnose_chronic_vulnerability(profile)
        
        flood_days = int(df_hist["flood_occurred"].sum())
        
        # Sample recent records (last 60 days) for quick JSON inspection
        sample_recs = df_hist.tail(60).to_dict("records")
        for r in sample_recs:
            r["date"] = str(r["date"].date()) if hasattr(r["date"], "date") else str(r["date"])
            
        return HistoricalTimelineResponse(
            station_profile=profile,
            chronic_vulnerability=chronic_diag,
            records_count=len(df_hist),
            historical_flood_events=flood_days,
            recent_flood_years=profile.get("historical_flood_years", []),
            sample_records=sample_recs
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Historical timeline error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=False)

