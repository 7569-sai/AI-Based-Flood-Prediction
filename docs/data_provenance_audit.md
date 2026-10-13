# Data Provenance & Variable Classification Audit

**Project Title:** AI-Based Flood Prediction, Dynamic Risk Mapping and Resilience Planning  
**Application:** FloodAI Engine — "AI for a Safer, Resilient India"  
**Audit Standard:** Scientific Provenance, Empirical Integrity & Traceability  

---

## 1. Executive Summary & Provenance Disclosure

In accordance with scientific integrity guidelines and operational requirements, this audit explicitly documents the origin, calculation methodology, physical grounding, and synthetic/proxy nature of all 27 input features and target labels utilized in the FloodAI Engine.

**Explicit Provenance Declaration:**  
While meteorological baselines, topographic indices, river network geometries, and demographic distributions are strictly anchored to real-world geospatial and climate datasets (IMD gridded data, ERA5-Land reanalysis, USGS SRTM DEM, HydroSHEDS river networks, and CIESIN GPWv4), certain hydrological time series (e.g. gauge discharge and stage anomaly) and high-resolution satellite radar/optical observations (Sentinel-1 SAR backscatter and Sentinel-2 spectral indices) are implemented as **physically calibrated proxies and synthetic time series**. 

Furthermore, the target labels (`flood_occurred` and `flood_severity`) represent **physically simulated ground truth** constructed via multi-criteria hydrometeorological thresholds aligned with documented historical Indian flood events (e.g., Kerala August 2018, Delhi Yamuna July 2023, Assam Brahmaputra monsoon waves). They are **not** direct telemetric gauge records or official CWC disaster registry outputs.

---

## 2. Comprehensive 27-Feature Provenance Catalog

| # | Feature Name | Primary Domain | Provenance Class | Underlying Real-World Source | Proxy Formulation / Derivation Formula |
| :-: | :--- | :--- | :--- | :--- | :--- |
| **1** | `rainfall_1d` | Meteorological | **Observational / Reanalysis** | IMD 0.25° Gridded / ERA5-Land | 24-hour total precipitation accumulation at station location: $P(T-1)$ (mm). |
| **2** | `rainfall_3d` | Meteorological | **Derived Deterministic** | Derived from `rainfall_1d` | 3-day backward rolling cumulative precipitation: $\sum_{i=1}^{3} P(T-i)$ (mm). |
| **3** | `rainfall_7d` | Meteorological | **Derived Deterministic** | Derived from `rainfall_1d` | 7-day backward rolling cumulative precipitation: $\sum_{i=1}^{7} P(T-i)$ (mm). |
| **4** | `rainfall_14d` | Meteorological | **Derived Deterministic** | Derived from `rainfall_1d` | 14-day backward rolling cumulative precipitation: $\sum_{i=1}^{14} P(T-i)$ (mm). |
| **5** | `rainfall_anomaly_30d` | Meteorological | **Derived Statistical** | Derived from `rainfall_1d` | Departure of short-term rainfall from monthly baseline: $R_{3d} - (R_{30d} / 10.0)$ (mm). |
| **6** | `api_14d` | Meteorological | **Physically Derived** | Antecedent Precipitation Model | Antecedent Precipitation Index: $R_{1d} + 0.85 \cdot R_{3d}$ representing residual catchment moisture. |
| **7** | `temp_mean_3d` | Meteorological | **Observational / Reanalysis** | IMD / ERA5-Land 2m Temperature | 3-day trailing average air temperature at 2m height (°C). |
| **8** | `humidity_mean_3d` | Meteorological | **Observational / Reanalysis** | ERA5-Land 2m Relative Humidity | 3-day trailing average relative humidity at 2m height (%). |
| **9** | `soil_moisture_root_zone`| Hydrological | **Physical Reanalysis Proxy** | ERA5-Land Volumetric Soil Water | Layer 1 (0–7 cm depth) volumetric soil water content ($\text{m}^3/\text{m}^3$). |
| **10**| `soil_moisture_deep` | Hydrological | **Physical Reanalysis Proxy** | ERA5-Land Volumetric Soil Water | Layer 2 (7–28 cm depth) volumetric soil water content ($\text{m}^3/\text{m}^3$). |
| **11**| `soil_saturation_ratio` | Hydrological | **Derived Physical Ratio** | Soil Hydraulic Capacity Model | Ratio of current root-zone moisture to field capacity: $\min(\text{SM}_{\text{root}} / 0.48, 1.0)$. |
| **12**| `river_discharge_proxy` | Hydrological | **Calibrated Physical Proxy** | Rational Runoff Formulation | Catchment discharge proxy calibrated from $R_{7d}$, basin area, and river proximity ($\text{m}^3/\text{s}$). |
| **13**| `river_stage_anomaly` | Hydrological | **Calibrated Physical Proxy** | Hydrometric Rating Curve Proxy | River elevation departure relative to local danger warning level (meters). |
| **14**| `elevation_m` | Topographic | **Observational Geospatial** | USGS SRTM 30m Digital Elevation Model | Absolute ground surface elevation above mean sea level (meters). |
| **15**| `slope_deg` | Topographic | **Observational Geospatial** | USGS SRTM 30m DEM Gradient | Topographic surface slope inclination angle (degrees). |
| **16**| `dist_river_km` | Topographic | **Observational Geospatial** | HydroSHEDS / OpenStreetMap Rivers | Geodesic / Planar Euclidean distance to primary hydrological drainage network (km). |
| **17**| `twi_proxy` | Topographic | **Derived Geomorphic** | Beven & Kirkby TWI Formulation | Topographic Wetness Index: $\ln(a / \tan \beta) \approx \ln(100.0 / \tan(\max(\text{slope}, 0.1)^\circ))$. |
| **18**| `elevation_risk_factor` | Topographic | **Derived Geomorphic** | Empirical Floodplain Hypsometry | Exponential low-elevation susceptibility: $\exp(-\text{elevation} / 50.0)$. |
| **19**| `drainage_vulnerability`| Topographic | **Derived Composite Index** | Geomorphic Multi-Criteria Model | Weighted drainage deficit: $0.45 V_{\text{elev}} + 0.35 V_{\text{river}} + 0.20 V_{\text{slope}} \in [0, 1]$. |
| **20**| `sar_vv` | Remote Sensing | **Calibrated Radar Proxy** | Sentinel-1 SAR Calibration Baseline | C-band vertical transmit/vertical receive radar backscatter intensity (dB). Calibrated to drop during specular open-water inundation. |
| **21**| `sar_vh` | Remote Sensing | **Calibrated Radar Proxy** | Sentinel-1 SAR Calibration Baseline | C-band vertical transmit/horizontal receive cross-polarized radar backscatter (dB). Sensitive to vegetation flooding and volume scattering. |
| **22**| `sar_cross_ratio` | Remote Sensing | **Derived Polarimetric** | Sentinel-1 Dual-Pol Ratio | Polarimetric backscatter ratio: $|\text{VH} / (\text{VV} - 0.01)|$. Differentiates smooth open water from inundated canopy. |
| **23**| `ndwi_proxy` | Remote Sensing | **Calibrated Optical Proxy** | Sentinel-2 Spectral Response Model | Normalized Difference Water Index proxy: $(\rho_{\text{Green}} - \rho_{\text{NIR}}) / (\rho_{\text{Green}} + \rho_{\text{NIR}})$. Strongly positive over surface water. |
| **24**| `ndvi_proxy` | Remote Sensing | **Calibrated Optical Proxy** | Sentinel-2 Spectral Response Model | Normalized Difference Vegetation Index proxy: $(\rho_{\text{NIR}} - \rho_{\text{Red}}) / (\rho_{\text{NIR}} + \rho_{\text{Red}})$. Captures agricultural canopy health. |
| **25**| `population_density` | Exposure | **Observational Demographic** | CIESIN GPWv4 (Gridded Population) | Ambient human settlement density ($\text{persons}/\text{km}^2$). |
| **26**| `exposure_index` | Exposure | **Derived Demographic Index** | UNDRR Disaster Exposure Standard | Logarithmically scaled exposure: $\text{clip}(\ln(1 + \text{pop}) / \ln(1 + 5000), 0, 1)$. |
| **27**| `vulnerability_index` | Vulnerability | **Derived Composite Index** | UNDRR Sendai Framework | Weighted systemic vulnerability: $0.60 V_{\text{drainage}} + 0.40 E_{\text{pop}} \in [0, 1]$. |

---

## 3. Summary of Provenance Categories

1. **Observational & Direct Geospatial (7 features - 26%):**
   - Direct real-world measurements or spatial models from official scientific repositories: `rainfall_1d`, `temp_mean_3d`, `humidity_mean_3d`, `elevation_m`, `slope_deg`, `dist_river_km`, `population_density`.
2. **Derived Deterministic & Mathematical Formulations (9 features - 33%):**
   - Mathematically transformed features derived strictly from observational values: `rainfall_3d`, `rainfall_7d`, `rainfall_14d`, `rainfall_anomaly_30d`, `api_14d`, `twi_proxy`, `elevation_risk_factor`, `drainage_vulnerability`, `exposure_index`.
3. **Calibrated Physical & Remote Sensing Proxies (11 features - 41%):**
   - Hydrometeorologically coupled simulations and sensor radiative transfer proxies: `soil_moisture_root_zone`, `soil_moisture_deep`, `soil_saturation_ratio`, `river_discharge_proxy`, `river_stage_anomaly`, `sar_vv`, `sar_vh`, `sar_cross_ratio`, `ndwi_proxy`, `ndvi_proxy`, `vulnerability_index`.

---

## 4. Scientific Target Labeling Disclosure

- **`flood_occurred`:** Binary indicator ($0 = \text{No Flood}, 1 = \text{Flood Inundation}$).
  - Construction: Computed using a multi-criteria physical hydrological rule based on excessive cumulative rainfall ($R_{3d} > 120\text{ mm}$ or $R_{7d} > 250\text{ mm}$), saturated soil profile ($\text{soil\_saturation\_ratio} > 0.85$), river stage exceeding danger levels ($\text{stage\_anomaly} > 1.5\text{ m}$), and radar backscatter attenuation ($\text{sar\_vv} < -18\text{ dB}$ or $\text{ndwi} > 0.35$), modulated by documented historical monsoon flood temporal windows.
- **`flood_severity`:** 4-class ordinal indicator ($0 = \text{None}, 1 = \text{Minor}, 2 = \text{Moderate}, 3 = \text{Severe/Catastrophic}$).
  - Construction: Hierarchical physical score integrating runoff depth, flood return period proxy, and infrastructure inundation thresholds.

**Operational Implication:** Because the dataset features calibrated proxies and simulated ground truth, models trained on this benchmark demonstrate theoretical validity, multimodal representation capabilities, and spatial transferability across hydrological archetypes. However, deployment in operational flood forecasting requires coupling with live Central Water Commission (CWC) telemetry and real-time Copernicus Sentinel-1 GRD imagery.
