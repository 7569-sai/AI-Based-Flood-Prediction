"""
Pydantic Schemas for FastAPI Endpoints.
Defines input payloads and response contracts with rigorous type validation.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class FloodPredictionRequest(BaseModel):
    latitude: float = Field(..., ge=6.0, le=38.0, description="Latitude in decimal degrees (Pan-India bounds: 6.0 to 38.0)")
    longitude: float = Field(..., ge=68.0, le=98.0, description="Longitude in decimal degrees (Pan-India bounds: 68.0 to 98.0)")
    location_name: Optional[str] = Field("Aluva", description="Station or locality name")
    basin: Optional[str] = Field("Periyar", description="River basin name")
    
    # Meteorological inputs
    rainfall_1d: float = Field(..., ge=0.0, le=500.0, description="1-day prior rainfall in mm")
    rainfall_3d: float = Field(..., ge=0.0, le=1000.0, description="3-day antecedent rainfall in mm")
    rainfall_7d: float = Field(..., ge=0.0, le=1500.0, description="7-day antecedent rainfall in mm")
    rainfall_14d: float = Field(..., ge=0.0, le=2500.0, description="14-day antecedent rainfall in mm")
    rainfall_anomaly_30d: Optional[float] = Field(0.0, description="30-day rainfall anomaly in mm")
    api_14d: Optional[float] = Field(50.0, ge=0.0, description="Antecedent Precipitation Index in mm")
    temp_mean_3d: Optional[float] = Field(26.0, description="3-day mean temperature in °C")
    humidity_mean_3d: Optional[float] = Field(85.0, ge=0.0, le=100.0, description="3-day relative humidity %")
    
    # Hydrological inputs
    soil_moisture_root_zone: float = Field(0.35, ge=0.0, le=0.60, description="Root-zone soil moisture m3/m3")
    soil_moisture_deep: Optional[float] = Field(0.38, ge=0.0, le=0.60, description="Deep soil moisture m3/m3")
    soil_saturation_ratio: Optional[float] = Field(0.70, ge=0.0, le=1.0, description="Soil saturation ratio [0, 1]")
    
    # Terrain inputs
    elevation_m: float = Field(..., ge=-10.0, le=3000.0, description="Elevation above sea level in meters")
    slope_deg: float = Field(..., ge=0.0, le=90.0, description="Terrain slope angle in degrees")
    dist_river_km: float = Field(..., ge=0.0, le=50.0, description="Distance to major river channel in km")
    twi_proxy: Optional[float] = Field(5.0, description="Topographic Wetness Index proxy")
    elevation_risk_factor: Optional[float] = Field(None, description="Lowland factor")

    # Remote Sensing inputs (optional, defaults provided)
    sar_vv: Optional[float] = Field(-11.0, le=0.0, description="Sentinel-1 SAR VV backscatter dB")
    sar_vh: Optional[float] = Field(-17.5, le=0.0, description="Sentinel-1 SAR VH backscatter dB")
    sar_cross_ratio: Optional[float] = Field(1.59, description="SAR polarization cross-ratio")
    ndwi_proxy: Optional[float] = Field(0.2, description="Normalized Difference Water Index proxy")
    ndvi_proxy: Optional[float] = Field(0.6, description="Normalized Difference Vegetation Index proxy")
    
    # Exposure inputs
    population_density: Optional[float] = Field(1000.0, ge=0.0, description="Population density in ppl/km2")


class FloodPredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 = No Flood, 1 = Flood Occurrence")
    probability: float = Field(..., description="Calibrated flood probability [0, 1]")
    hazard_level: int = Field(..., description="0: Low, 1: Moderate, 2: High, 3: Very High")
    hazard_category: str = Field(..., description="Hazard severity level name")
    risk_score: float = Field(..., description="Project-defined 0-100 composite risk score")
    risk_category: str = Field(..., description="Very Low, Low, Moderate, High, Very High")
    risk_color: str = Field(..., description="Hex color code for map visualization")
    model_version: str = Field(..., description="Model identifier")
    inference_id: int = Field(..., description="Database transaction ID")


class RiskScoreRequest(BaseModel):
    hazard_probability: float = Field(..., ge=0.0, le=1.0, description="Hazard probability H in [0, 1]")
    population_density: float = Field(850.0, ge=0.0, description="Population density in ppl/km2")
    elevation_m: float = Field(25.0, ge=-10.0, description="Elevation in meters")
    slope_deg: float = Field(3.0, ge=0.0, description="Slope in degrees")
    dist_river_km: float = Field(1.0, ge=0.0, description="Distance to river in km")


class RiskScoreResponse(BaseModel):
    hazard_probability: float
    exposure_index: float
    vulnerability_index: float
    drainage_vulnerability: float
    risk_score: float
    risk_category: str
    risk_color: str
    formula_citation: str = "Risk = (0.50*H + 0.25*E + 0.25*V) * 100 (Project-Defined)"


class ShapFactor(BaseModel):
    feature: str
    readable_name: str
    value: float
    shap_value: float
    direction: str


class ExplainResponse(BaseModel):
    prediction: int
    probability: float
    base_value: float
    factors_increasing_risk: List[ShapFactor]
    factors_decreasing_risk: List[ShapFactor]


class GeospatialRiskRequest(BaseModel):
    scenario: str = Field("monsoon_surge", description="Scenario: 'normal', 'monsoon_surge', 'extreme_flood_2018'")


class RecommendationsResponse(BaseModel):
    disclaimer: str
    location: str
    basin: str
    urgency_level: str
    risk_profile: Dict[str, Any]
    emergency_actions_immediate: List[Dict[str, Any]]
    structural_mitigation: List[Dict[str, Any]]
    policy_and_planning: List[Dict[str, Any]]
    agricultural_and_ecological: List[Dict[str, Any]]


class LocationLookupRequest(BaseModel):
    latitude: float = Field(..., ge=6.0, le=38.0, description="Latitude in decimal degrees")
    longitude: float = Field(..., ge=68.0, le=98.0, description="Longitude in decimal degrees")
    station_id: Optional[str] = Field(None, description="Optional known station ID")
    intensity: Optional[str] = Field("monsoon_surge", description="'normal', 'monsoon_surge', 'severe_flood'")


class LocationLookupResponse(BaseModel):
    profile: Dict[str, Any]
    environmental_parameters: Dict[str, Any]
    risk_result: Dict[str, Any]


class HistoricalTimelineResponse(BaseModel):
    station_profile: Dict[str, Any]
    chronic_vulnerability: Dict[str, Any]
    records_count: int
    historical_flood_events: int
    recent_flood_years: List[int]
    sample_records: List[Dict[str, Any]]

