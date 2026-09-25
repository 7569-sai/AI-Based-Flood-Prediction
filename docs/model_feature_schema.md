# Model Feature Schema & Preprocessing Specification

**Project Title:** AI-Based Flood Prediction, Dynamic Risk Mapping and Resilience Planning  
**Application:** FloodAI Engine — "AI for a Safer, Resilient India"  
**Model Family:** Pan-India Multi-Basin Classifiers (Champion: Random Forest Balanced)

---

## 1. Schema Definition & Ordering Contract

The machine learning models (Random Forest, LightGBM, ST-FloodNet) strictly require the following **27-dimensional feature vector** in the exact sequential order defined in `data/processed/india/pan_india_metadata.json`:

| Index | Feature Name | Domain | Data Type | Physical Unit | Valid Range | Description & Formulation |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| **0** | `rainfall_1d` | Meteorological | `float64` | mm | $[0.0, 450.0]$ | 24-hour antecedent rainfall on preceding day ($T-1$). |
| **1** | `rainfall_3d` | Meteorological | `float64` | mm | $[0.0, 750.0]$ | 3-day backward rolling cumulative rainfall ($\sum_{i=1}^{3} P(T-i)$). |
| **2** | `rainfall_7d` | Meteorological | `float64` | mm | $[0.0, 1200.0]$ | 7-day backward rolling cumulative rainfall ($\sum_{i=1}^{7} P(T-i)$). |
| **3** | `rainfall_14d` | Meteorological | `float64` | mm | $[0.0, 1800.0]$ | 14-day backward rolling cumulative rainfall ($\sum_{i=1}^{14} P(T-i)$). |
| **4** | `rainfall_anomaly_30d` | Meteorological | `float64` | mm | $[-50.0, 500.0]$ | Deviation of 3-day surge against antecedent baseline: $R_{3d} - (R_{14d} / 4.6)$. |
| **5** | `api_14d` | Meteorological | `float64` | mm | $[0.0, 800.0]$ | Antecedent Precipitation Index with retention coefficient: $R_{1d} + 0.85 \cdot R_{3d}$. |
| **6** | `temp_mean_3d` | Meteorological | `float64` | °C | $[5.0, 50.0]$ | 3-day mean 2m ambient air temperature. |
| **7** | `humidity_mean_3d` | Meteorological | `float64` | % | $[20.0, 100.0]$ | 3-day mean 2m relative humidity. |
| **8** | `soil_moisture_root_zone` | Hydrological | `float64` | $\text{m}^3/\text{m}^3$ | $[0.10, 0.55]$ | Root-zone volumetric soil moisture content (0–7 cm depth, ERA5-Land). |
| **9** | `soil_moisture_deep` | Hydrological | `float64` | $\text{m}^3/\text{m}^3$ | $[0.10, 0.52]$ | Deep-layer volumetric soil moisture content (7–28 cm depth). |
| **10** | `soil_saturation_ratio` | Hydrological | `float64` | ratio | $[0.0, 1.0]$ | Saturation relative to field capacity: $\min(\text{SM}_{\text{root}} / 0.48, 1.0)$. |
| **11** | `river_discharge_proxy` | Hydrological | `float64` | $\text{m}^3/\text{s}$ | $[50.0, 18500.0]$ | Upstream catchment runoff volume proxy: $\text{clip}(14 \cdot R_{7d} + \frac{280}{d_{\text{riv}} + 0.1}, 50, 18500)$. |
| **12** | `river_stage_anomaly` | Hydrological | `float64` | meters | $[-2.0, 6.5]$ | Gauge stage deviation relative to warning threshold: $\text{clip}(\frac{R_{3d}}{45} + \frac{R_{7d}}{180} - 0.4 d_{\text{riv}}, -2, 6.5)$. |
| **13** | `elevation_m` | Topographic | `float64` | m AMSL | $[-10.0, 8848.0]$ | Ground surface elevation above mean sea level (USGS SRTM 30m DEM). |
| **14** | `slope_deg` | Topographic | `float64` | degrees | $[0.0, 90.0]$ | Terrain topographic inclination angle derived from maximum 8-direction gradient. |
| **15** | `dist_river_km` | Topographic | `float64` | km | $[0.01, 50.0]$ | Minimum Euclidean distance to major drainage channel / river vector network. |
| **16** | `twi_proxy` | Topographic | `float64` | index | $[1.5, 15.0]$ | Topographic Wetness Index: $\ln(100.0 / \tan(\max(\theta, 0.1^\circ)))$. |
| **17** | `elevation_risk_factor` | Topographic | `float64` | index | $[0.0, 1.0]$ | Low-lying elevation vulnerability factor: $\exp(-\text{elevation} / 50.0)$. |
| **18** | `drainage_vulnerability` | Topographic | `float64` | index | $[0.0, 1.0]$ | Geomorphic drainage bottleneck score: $0.45 V_{\text{elev}} + 0.35 V_{\text{river}} + 0.20 V_{\text{slope}}$. |
| **19** | `sar_vv` | Remote Sensing | `float64` | dB | $[-35.0, 0.0]$ | Sentinel-1 C-Band SAR dual-pol vertical transmit / receive backscatter. |
| **20** | `sar_vh` | Remote Sensing | `float64` | dB | $[-40.0, -5.0]$ | Sentinel-1 C-Band SAR dual-pol vertical transmit / horizontal receive backscatter. |
| **21** | `sar_cross_ratio` | Remote Sensing | `float64` | ratio | $[0.5, 3.5]$ | Polarimetric volume scattering cross-ratio: $|\sigma^\circ_{\text{VH}} / (\sigma^\circ_{\text{VV}} - 0.01)|$. |
| **22** | `ndwi_proxy` | Remote Sensing | `float64` | index | $[-0.5, 0.90]$ | Normalized Difference Water Index surface water indicator. |
| **23** | `ndvi_proxy` | Remote Sensing | `float64` | index | $[-0.2, 0.90]$ | Normalized Difference Vegetation Index canopy vigor and vegetative attenuation. |
| **24** | `population_density` | Exposure | `float64` | $\text{ppl}/\text{km}^2$ | $[0.0, 45000.0]$ | Human settlement density from CIESIN GPWv4 / Census of India. |
| **25** | `exposure_index` | Exposure | `float64` | index | $[0.0, 1.0]$ | Logarithmic demographic exposure: $\text{clip}(\ln(1 + \rho_{\text{pop}}) / \ln(1 + 5000), 0, 1)$. |
| **26** | `vulnerability_index` | Vulnerability | `float64` | index | $[0.0, 1.0]$ | UNDRR composite vulnerability score: $0.60 \cdot V_{\text{drain}} + 0.40 \cdot E_{\text{pop}}$. |

---

## 2. Target Variables & Encodings

1. **`flood_occurred` (Binary Classification Target)**:
   - `0`: No significant flood inundation
   - `1`: Significant flood event (riverine overbank flow, flash flood, or deep waterlogging)
2. **`flood_severity` (Ordinal Multi-Class Target)**:
   - `0`: None (Normal baseline)
   - `1`: Minor (Localized shallow ponding $<0.3$ m)
   - `2`: Moderate (Agricultural inundation, roads submerged, river stage warning)
   - `3`: Severe / Catastrophic (Widespread residential submersion, embankment breach)

---

## 3. Preprocessing & Scaling Rules

- Tree-based models (Random Forest Champion, LightGBM, XGBoost) consume the raw unstandardized physical feature vector directly.
- Neural models (ST-FloodNet, Bi-LSTM) utilize Z-score standardization parameters saved in `models/trained/scaler.joblib`.
- Temporal alignment: For real-time point inference at timestamp $T$, all antecedent rainfall windows exclude $T$ and draw strictly from days $\{T-14, \dots, T-1\}$.
