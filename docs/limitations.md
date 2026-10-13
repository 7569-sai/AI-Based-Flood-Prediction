# Limitations, Known Weaknesses & Operational Caveats

**Project Title:** AI-Based Flood Prediction, Dynamic Risk Mapping and Resilience Planning  
**Application:** FloodAI Engine — "AI for a Safer, Resilient India"  
**Scientific Standard:** Complete Empirical Honesty & Ethical Technology Disclosure  

---

## 1. Data Provenance & Ground Truth Proxies

While this study constructs a pan-India dataset spanning 78,876 daily station records across 36 monitoring stations and 24 river basins (2018–2023), several data characteristics represent calibrated proxies rather than uncalibrated observational telemetry:
1. **River Gauge Telemetry:** Continuous national daily river discharge and stage anomaly feeds across all 36 stations are represented via physically-calibrated hydrological proxies derived from upstream catchment area, 7-day cumulative rainfall, and stage rating curves, rather than live unedited Central Water Commission (CWC) telemetry loggers.
2. **Sentinel-1 SAR Radar:** Synthetic aperture radar backscatter values ($\sigma^0_{\text{vv}}$, $\sigma^0_{\text{vh}}$) and polarimetric cross-ratios are generated from radiative transfer response models calibrated against historical Sentinel-1 GRD passes, rather than daily cloud-free orbital passes (which naturally have a 6–12 day revisit period).
3. **Simulated Ground Truth Labels:** Ground truth binary inundation (`flood_occurred`) and 4-tier ordinal severity (`flood_severity`) are computed via a multi-criteria physical trigger function aligned with documented disaster timelines (e.g. Kerala 2018, Assam 2020, Delhi Yamuna 2023) rather than official post-disaster insurance claims or surveyed flood extent polygons.

---

## 2. Hydrodynamic & Physics Limitations

1. **Absence of 2D Shallow Water Equations:** The system uses statistical and deep learning approximations of flood hazard rather than solving Saint-Venant or Navier-Stokes shallow water equations. Backwater hydraulics, tidal estuaries, and complex channel confluences are approximated via terrain slope and river proximity.
2. **Dam & Reservoir Operating Rules:** Extreme floods in peninsular India (e.g. Periyar, Krishna, Godavari) are heavily influenced by human operational decisions regarding dam reservoir gates (e.g. Idukki, Mullaperiyar, Srisailam). The model incorporates upstream rainfall but does not ingest real-time dam gate opening schedules.
3. **Urban Micro-Drainage & Stormwater Inundation:** In densely populated urban areas (e.g. Mumbai Mithi basin, Delhi Yamuna floodplain, Bengaluru), flash waterlogging is frequently caused by blocked stormwater drains, culvert sedimentation, and impervious surface paving. While population density and elevation risk factors are included, sub-meter stormwater network modeling requires local hydraulic hydrodynamic models (e.g. SWMM / HEC-RAS 2D).

---

## 3. Machine Learning & Stacking Behavior

1. **In-Basin Tabular Superiority:** Gradient-boosted decision trees (LightGBM, XGBoost) achieve higher pointwise precision on stationary in-basin test sets than deep learning models. While HF-STFloodNet improves probability calibration (lowest Brier score of 0.0074) and provides epistemic uncertainty estimates, practitioners must acknowledge that pure tabular tree ensembles remain strong competitors for tabular hydrological tasks.
2. **Sensor Outage Vulnerability:** Complete failure of rain gauge telemetry causes significant performance degradation across all models, proving that remote sensing and topography cannot completely substitute for in-situ precipitation measurement during the first 24 hours of flash flood events.

---

## 4. Operational Recommendations for Field Deployment

For State Disaster Management Authorities (SDMAs) or civil protection agencies seeking to deploy the FloodAI Engine in real-time operational forecasting:
1. **Direct Telemetry Integration:** Replace simulated gauge and stage proxies with automated REST/MQTT webhooks to the CWC National Hydrology Project (NHP) sensor network.
2. **Copernicus Data Space Connector:** Couple the remote sensing encoder with automated Earth Engine or Copernicus Data Space APIs to ingest live Sentinel-1 GRD imagery as new orbital tiles become available.
3. **Reservoir Dispatch Integration:** Incorporate upstream dam storage percentages and release volumes into the dynamic hydrometeorological feature vector.
