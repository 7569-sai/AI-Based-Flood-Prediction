# Feature Dictionary & Mathematical Definitions

**Project Title:** AI-Based Flood Prediction, Dynamic Risk Mapping and Resilience Planning  
**Application:** FloodAI Engine — "AI for a Safer, Resilient India"

---

## 1. Feature Catalog (27 Engineered Predictors)

Every feature in the FloodAI Engine has an explicit scientific provenance, mathematical formulation, and temporal timestamp relationship.

| # | Feature Name | Domain | Type | Unit | Range | Mathematical / Physical Definition |
| :-: | :--- | :--- | :-: | :---: | :---: | :--- |
| **1** | `rainfall_1d` | Meteorological | float | mm | $[0, 350]$ | Total precipitation accumulation on preceding day: $P(T-1)$. |
| **2** | `rainfall_3d` | Meteorological | float | mm | $[0, 650]$ | 3-day backward rolling sum: $\sum_{i=1}^{3} P(T-i)$. |
| **3** | `rainfall_7d` | Meteorological | float | mm | $[0, 1100]$ | 7-day backward rolling sum: $\sum_{i=1}^{7} P(T-i)$. |
| **4** | `rainfall_14d` | Meteorological | float | mm | $[0, 1600]$ | 14-day backward rolling sum: $\sum_{i=1}^{14} P(T-i)$. |
| **5** | `rainfall_anomaly_30d` | Meteorological | float | mm | $[-50, 450]$ | Deviation of 3-day rainfall from 30-day baseline: $R_{3d} - (R_{30d} / 10.0)$. |
| **6** | `api_14d` | Meteorological | float | mm | $[0, 600]$ | Antecedent Precipitation Index: $R_{1d} + 0.85 \cdot R_{3d}$. |
| **7** | `temp_mean_3d` | Meteorological | float | °C | $[8, 45]$ | 3-day mean 2m air temperature. |
| **8** | `humidity_mean_3d` | Meteorological | float | % | $[30, 100]$ | 3-day mean relative humidity at 2m. |
| **9** | `soil_moisture_root_zone` | Hydrological | float | $\text{m}^3/\text{m}^3$ | $[0.15, 0.50]$ | Volumetric soil water content in upper layer (0–7 cm, ERA5-Land). |
| **10** | `soil_moisture_deep` | Hydrological | float | $\text{m}^3/\text{m}^3$ | $[0.14, 0.48]$ | Deep soil moisture content (7–28 cm layer). |
| **11** | `soil_saturation_ratio` | Hydrological | float | ratio | $[0, 1]$ | Degree of soil saturation against field capacity: $\min(\text{SM} / 0.48, 1.0)$. |
| **12** | `river_discharge_proxy` | Hydrological | float | $\text{m}^3/\text{s}$ | $[50, 18500]$ | Catchment upstream discharge proxy derived from 7-day rainfall and proximity. |
| **13** | `river_stage_anomaly` | Hydrological | float | meters | $[-2.0, 6.5]$ | Estimated river water level relative to local danger warning threshold. |
| **14** | `elevation_m` | Topographic | float | meters | $[-5, 1600]$ | Height above mean sea level from USGS SRTM 30m Digital Elevation Model. |
| **15** | `slope_deg` | Topographic | float | degrees | $[0.1, 45]$ | Terrain slope inclination angle derived from SRTM gradient. |
| **16** | `dist_river_km` | Topographic | float | km | $[0.05, 15]$ | Minimum planar Euclidean distance to primary river network. |
| **17** | `twi_proxy` | Topographic | float | index | $[2.0, 12.0]$ | Topographic Wetness Index: $\ln(100.0 / \tan(\max(\text{slope}, 0.1)^\circ))$. |
| **18** | `elevation_risk_factor` | Topographic | float | index | $[0, 1]$ | Low-elevation deficit factor: $\exp(-\text{elevation} / 50.0)$. |
| **19** | `drainage_vulnerability` | Topographic | float | index | $[0, 1]$ | Weighted geomorphic drainage bottleneck score: $0.45 V_{\text{elev}} + 0.35 V_{\text{river}} + 0.20 V_{\text{slope}}$. |
| **20** | `sar_vv` | Remote Sensing | float | dB | $[-28, -5]$ | Sentinel-1 C-band SAR VV backscatter intensity. |
| **21** | `sar_vh` | Remote Sensing | float | dB | $[-35, -10]$ | Sentinel-1 C-band SAR VH cross-polarized backscatter intensity. |
| **22** | `sar_cross_ratio` | Remote Sensing | float | ratio | $[0.8, 2.5]$ | Polarimetric cross-ratio: $|\text{VH} / (\text{VV} - 0.01)|$. |
| **23** | `ndwi_proxy` | Remote Sensing | float | index | $[-0.3, 0.85]$ | Normalized Difference Water Index surface water indicator. |
| **24** | `ndvi_proxy` | Remote Sensing | float | index | $[0.1, 0.85]$ | Normalized Difference Vegetation Index canopy vigor proxy. |
| **25** | `population_density` | Exposure | float | $\text{ppl}/\text{km}^2$ | $[50, 6000]$ | Human population density from CIESIN GPWv4. |
| **26** | `exposure_index` | Exposure | float | index | $[0, 1]$ | Logarithmic population exposure: $\text{clip}(\ln(1 + \text{pop}) / \ln(1 + 5000), 0, 1)$. |
| **27** | `vulnerability_index` | Vulnerability | float | index | $[0, 1]$ | Composite vulnerability combining drainage deficit and exposure: $0.60 V_{\text{drain}} + 0.40 E$. |

---

## 2. Target Variables

1. **`flood_occurred` (Binary Ground Truth)**:  
   $1$ if significant flood inundation occurs, $0$ otherwise.
2. **`flood_severity` (Ordinal Class)**:  
   - $0$ = None (Non-flood baseline)
   - $1$ = Minor (Isolated shallow waterlogging)
   - $2$ = Moderate (Riverine overflow, agricultural floodplain submersion)
   - $3$ = Severe / Catastrophic (Major levee breach, residential inundation, emergency evacuation)
