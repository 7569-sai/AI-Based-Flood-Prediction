# API Feature Extraction & Data Provenance Mapping

**Project Title:** AI-Based Flood Prediction, Dynamic Risk Mapping and Resilience Planning  
**Application:** FloodAI Engine — "AI for a Safer, Resilient India"  
**Module:** Location-First Feature Extraction Layer (`src/data/api_feature_extractor.py`)

---

## 1. Architectural Overview

The FloodAI Engine implements a **Location-First Intelligence Workflow**. When a user selects any geographical location across India (via map click, coordinate input, or station preset), the system initiates an automated parameter extraction sequence:

```
[ Latitude, Longitude (India) ]
                │
                ├─► OpenStreetMap Nominatim ────────► Administrative Metadata (State, District)
                │                                    Nearest National River Basin
                │
                ├─► Open-Meteo ERA5-Land API ───────► 14-Day Antecedent Precipitation
                │                                    Temperature, Humidity
                │                                    Root-Zone & Deep Soil Moisture
                │
                ├─► Open-Elevation SRTM API ────────► 30m Digital Elevation Model (DEM)
                │                                    Terrain Slope & Topographic Indices
                │
                ├─► Basin Hydro Network Engine ─────► Euclidean River Proximity
                │                                    Discharge Proxy, River Stage Anomaly
                │                                    Drainage Bottleneck Vulnerability
                │
                ├─► Copernicus Sentinel-1 Stack ────► SAR C-Band Dual-Pol (VV, VH)
                │                                    Cross-Polarization Ratio, NDWI, NDVI
                │
                └─► CIESIN GPWv4 Global Grids ──────► Population Density
                                                     Exposure Index, UNDRR Vulnerability
                │
                ▼
  [ 27-Feature Strict Physical Vector ] ──► [ Model Inference (Random Forest / LightGBM / ST-FloodNet) ]
```

---

## 2. Comprehensive 27-Feature Mapping Matrix

| # | Model Feature | Primary Data Source | Raw API Endpoint / Variable | Native Unit | Standardized Unit | Temporal Window & Causality | Spatial Resolution | Mathematical Transformation / Physics Derivation |
| :-: | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | `rainfall_1d` | Open-Meteo ERA5-Land Reanalysis | `daily.precipitation_sum` | $\text{mm}$ | $\text{mm}$ | Preceding day $T-1$ (strictly backward-looking) | $0.1^\circ$ (~9 km) | $P(T-1)$ |
| **2** | `rainfall_3d` | Open-Meteo ERA5-Land Reanalysis | `daily.precipitation_sum` | $\text{mm}$ | $\text{mm}$ | Preceding 3 days $[T-3, T-1]$ | $0.1^\circ$ (~9 km) | $\sum_{i=1}^{3} P(T-i)$ |
| **3** | `rainfall_7d` | Open-Meteo ERA5-Land Reanalysis | `daily.precipitation_sum` | $\text{mm}$ | $\text{mm}$ | Preceding 7 days $[T-7, T-1]$ | $0.1^\circ$ (~9 km) | $\sum_{i=1}^{7} P(T-i)$ |
| **4** | `rainfall_14d` | Open-Meteo ERA5-Land Reanalysis | `daily.precipitation_sum` | $\text{mm}$ | $\text{mm}$ | Preceding 14 days $[T-14, T-1]$ | $0.1^\circ$ (~9 km) | $\sum_{i=1}^{14} P(T-i)$ |
| **5** | `rainfall_anomaly_30d` | Open-Meteo ERA5-Land Reanalysis | `daily.precipitation_sum` | $\text{mm}$ | $\text{mm}$ | Preceding 14-day scaled baseline | $0.1^\circ$ (~9 km) | $R_{3d} - \frac{R_{14d}}{4.6}$ |
| **6** | `api_14d` | Open-Meteo ERA5-Land Reanalysis | Derived from `precipitation_sum` | $\text{mm}$ | $\text{mm}$ | Antecedent 14-day exponential decay | $0.1^\circ$ (~9 km) | $R_{1d} + 0.85 \cdot R_{3d}$ |
| **7** | `temp_mean_3d` | Open-Meteo ECMWF Reanalysis | `daily.temperature_2m_mean` | $^\circ\text{C}$ | $^\circ\text{C}$ | Preceding 3-day mean $[T-3, T-1]$ | $0.1^\circ$ (~9 km) | $\frac{1}{3} \sum_{i=1}^{3} T_{2m}(T-i)$ |
| **8** | `humidity_mean_3d` | Open-Meteo ECMWF Reanalysis | `daily.relative_humidity_2m_mean` | $\%$ | $\%$ | Preceding 3-day mean $[T-3, T-1]$ | $0.1^\circ$ (~9 km) | $\frac{1}{3} \sum_{i=1}^{3} \text{RH}_{2m}(T-i)$ |
| **9** | `soil_moisture_root_zone` | Open-Meteo ERA5-Land | `daily.soil_moisture_0_to_7cm_mean` | $\text{m}^3/\text{m}^3$ | $\text{m}^3/\text{m}^3$ | Layer 1 (0–7 cm depth), day $T-1$ | $0.1^\circ$ (~9 km) | Volumetric liquid water content |
| **10** | `soil_moisture_deep` | Open-Meteo ERA5-Land | `daily.soil_moisture_7_to_28cm_mean` | $\text{m}^3/\text{m}^3$ | $\text{m}^3/\text{m}^3$ | Layer 2 (7–28 cm depth), day $T-1$ | $0.1^\circ$ (~9 km) | Volumetric liquid water content |
| **11** | `soil_saturation_ratio` | Open-Meteo ERA5-Land | Derived from root-zone moisture | $\text{m}^3/\text{m}^3$ | ratio $[0, 1]$ | Day $T-1$ relative to field capacity | $0.1^\circ$ (~9 km) | $\min\left(\frac{\text{SM}_{\text{root}}}{0.48}, 1.0\right)$ |
| **12** | `river_discharge_proxy` | Basin Hydrology Engine | Derived from 7d precip & channel dist | $\text{m}^3/\text{s}$ | $\text{m}^3/\text{s}$ | Antecedent 7-day catchment runoff | Catchment scale | $\text{clip}\left(14 \cdot R_{7d} + \frac{280}{d_{\text{riv}} + 0.1}, 50, 18500\right)$ |
| **13** | `river_stage_anomaly` | CWC Gauge Network / Hydro Engine | Gauge anomaly relative to warning level | meters | meters | Preceding 3-day hydrological response | Station / sub-basin | $\text{clip}\left(\frac{R_{3d}}{45} + \frac{R_{7d}}{180} - 0.4 d_{\text{riv}}, -2.0, 6.5\right)$ |
| **14** | `elevation_m` | Open-Elevation / USGS SRTM | `results[0].elevation` | meters AMSL | meters AMSL | Static terrain height | 30 meters (1 arc-sec) | Bilinear interpolation of SRTM DEM |
| **15** | `slope_deg` | USGS SRTM DEM 30m | 8-direction maximum gradient | degrees | degrees $[0, 90]$ | Static terrain slope | 30 meters | $\arctan\left(\sqrt{\left(\frac{\partial z}{\partial x}\right)^2 + \left(\frac{\partial z}{\partial y}\right)^2}\right) \cdot \frac{180}{\pi}$ |
| **16** | `dist_river_km` | National Hydrographic River Network | Minimum Euclidean distance to vector river | meters | kilometers | Static drainage morphology | $1:250,000$ hydro network | $\min_{k} \left\lVert \mathbf{x} - \mathbf{r}_k \right\rVert_2 \cdot 10^{-3}$ |
| **17** | `twi_proxy` | Topographic Analysis | Topographic Wetness Index (Beven & Kirkby) | index | index $[2, 14]$ | Static hydrologic steady-state | 30 meters | $\ln\left(\frac{100.0}{\tan(\max(\theta, 0.1^\circ))}\right)$ |
| **18** | `elevation_risk_factor` | Topographic Analysis | Exponential elevation attenuation | index | index $[0, 1]$ | Static low-lying susceptibility | 30 meters | $\exp\left(-\frac{\text{elevation}}{50.0}\right)$ |
| **19** | `drainage_vulnerability` | Multi-Criteria Geomorphic Model | Combined elevation, river, and slope deficit | index | index $[0, 1]$ | Static geomorphic flood susceptibility | Point / grid cell | $0.45 \cdot V_{\text{elev}} + 0.35 \cdot V_{\text{river}} + 0.20 \cdot V_{\text{slope}}$ |
| **20** | `sar_vv` | Copernicus Sentinel-1 C-Band SAR | Dual-pol vertical-transmit vertical-receive | $\text{dB}$ | $\text{dB}$ | 6–12 day satellite revisit pass | 10 meters GRD | $\sigma^\circ_{\text{VV}} \approx -12.0 - 4.2 \left(\frac{R_{3d}/160 + \text{SM}/0.45}{2}\right)$ |
| **21** | `sar_vh` | Copernicus Sentinel-1 C-Band SAR | Dual-pol vertical-transmit horiz-receive | $\text{dB}$ | $\text{dB}$ | 6–12 day satellite revisit pass | 10 meters GRD | $\sigma^\circ_{\text{VH}} \approx -18.0 - 3.15 \left(\frac{R_{3d}/160 + \text{SM}/0.45}{2}\right)$ |
| **22** | `sar_cross_ratio` | Copernicus Sentinel-1 C-Band SAR | Polarization backscatter ratio | ratio | ratio | Derived polarimetric response | 10 meters GRD | $\left\lvert \frac{\sigma^\circ_{\text{VH}}}{\sigma^\circ_{\text{VV}} - 0.01} \right\rvert$ |
| **23** | `ndwi_proxy` | Copernicus Sentinel-2 MSI / MODIS | $(B_{\text{Green}} - B_{\text{NIR}}) / (B_{\text{Green}} + B_{\text{NIR}})$ | index | index $[-0.3, 0.85]$ | Multi-day optical surface water index | 20 meters | $\text{clip}\left(0.18 + \frac{R_{3d}}{380} + \frac{\text{SM}}{2.2}, -0.2, 0.85\right)$ |
| **24** | `ndvi_proxy` | Copernicus Sentinel-2 MSI / MODIS | $(B_{\text{NIR}} - B_{\text{Red}}) / (B_{\text{NIR}} + B_{\text{Red}})$ | index | index $[0.1, 0.85]$ | Canopy vigor & soil saturation proxy | 20 meters | $\text{clip}\left(0.58 - \frac{R_{3d}}{850}, 0.1, 0.85\right)$ |
| **25** | `population_density` | CIESIN GPWv4.11 / Census of India | UN WPP-adjusted population count per $\text{km}^2$ | $\text{ppl}/\text{km}^2$ | $\text{ppl}/\text{km}^2$ | 2020 projection census grid | 30 arc-seconds (~1 km) | Point extraction from gridded raster |
| **26** | `exposure_index` | CIESIN Exposure Modeling | Logarithmic normalization to national urban cap | index | index $[0, 1]$ | Demographic asset exposure | 1 km | $\text{clip}\left(\frac{\ln(1 + \rho_{\text{pop}})}{\ln(1 + 5000)}, 0, 1\right)$ |
| **27** | `vulnerability_index` | UNDRR Disaster Risk Framework | Composite physical & demographic vulnerability | index | index $[0, 1]$ | Integrated vulnerability score | Point / grid cell | $0.60 \cdot V_{\text{drain}} + 0.40 \cdot E_{\text{pop}}$ |

---

## 3. Strict Temporal Data-Leakage Prevention Protocol

To ensure academic rigor and prevent temporal data leakage:
1. **Backward-Looking Lag Policy**:
   - For all rolling precipitation metrics ($R_{1d}$, $R_{3d}$, $R_{7d}$, $R_{14d}$), data from the current target day $T$ (and any forecast day $T+1$) is **strictly excluded**.
   - Antecedent precipitation is calculated strictly across preceding days:
     $$\mathbf{P}_{\text{antecedent}} = \{P(T-14), P(T-13), \dots, P(T-2), P(T-1)\}$$
2. **Soil Moisture Temporal Synchronization**:
   - Soil moisture measurements represent antecedent conditions immediately preceding the forecast window (day $T-1$), preventing instantaneous hydrological feedback loops from biasing the classifier.

---

## 4. Graceful Degradation & Fallback Strategy

When external internet connectivity is unavailable, API rate limits are encountered, or third-party servers experience transient downtime:
1. **Multi-Tier Caching**:
   - Every successful API call is cached on disk at `data/raw/cache/` using an MD5 hash of rounded coordinates ($\pm 0.001^\circ \approx 100\text{ m}$) and query timestamp.
   - Cached responses are immediately re-used without redundant network traffic.
2. **Spatial Station Climatology Fallback**:
   - If external APIs fail and no cache exists, the system automatically falls back to pre-indexed regional baseline climatologies from `src.geospatial.location_service`.
   - The UI displays `"⚠ Fallback Used"` in the **Data Retrieval Status** panel with full provenance transparency, ensuring user awareness without application failure.

---

## 5. Physical Range & Feature Vector Validation

Before any feature vector is passed to the trained machine learning models, `APIFeatureExtractor.validate_feature_vector()` executes 4 strict validation checks:
1. **Completeness Check**: Verifies that all 27 features exist and match `EXPECTED_FEATURE_ORDER`.
2. **Finiteness Check**: Ensures no `NaN`, `None`, or infinite values exist in the vector.
3. **Physical Bounds Check**:
   - Rainfall: $\ge 0\text{ mm}$
   - Soil Moisture: $[0.0, 1.0]\text{ m}^3/\text{m}^3$
   - Elevation: $[-500, 9000]\text{ m AMSL}$
4. **Temporal Consistency Check**:
   - Verifies monotonic rolling sums: $R_{3d} \ge R_{1d} - 10^{-4}$.
