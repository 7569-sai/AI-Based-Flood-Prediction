# Feature Dictionary: AI-Based Flood Prediction & Resilience Planning

This document provides a formal definition, mathematical derivation, physical unit, and data source for every variable in the processed feature matrix (`data/processed/full_features.csv`).

---

## 1. Meteorological Features

| Feature Name | Description & Formula | Temporal Window | Unit | Source |
| :--- | :--- | :---: | :---: | :--- |
| `rainfall_1d` | Prior-day total precipitation: $P_{t-1}$ | Lag 1 day ($T-1$) | mm | ECMWF ERA5-Land |
| `rainfall_3d` | 3-day antecedent cumulative rainfall: $\sum_{i=1}^3 P_{t-i}$ | Rolling 3 days | mm | ECMWF ERA5-Land |
| `rainfall_7d` | 7-day antecedent cumulative rainfall: $\sum_{i=1}^7 P_{t-i}$ | Rolling 7 days | mm | ECMWF ERA5-Land |
| `rainfall_14d` | 14-day antecedent cumulative rainfall: $\sum_{i=1}^{14} P_{t-i}$ | Rolling 14 days | mm | ECMWF ERA5-Land |
| `rainfall_30d` | 30-day antecedent cumulative rainfall: $\sum_{i=1}^{30} P_{t-i}$ | Rolling 30 days | mm | ECMWF ERA5-Land |
| `rainfall_anomaly_30d` | Deviation of 3-day rainfall from 30-day baseline: $\text{Rain}_{3d} - (\text{Rain}_{30d}/10)$ | Rolling 30 days | mm | Derived |
| `api_14d` | Antecedent Precipitation Index: $API_t = P_{t-1} + 0.85 \cdot API_{t-1}$ | Exponential decay ($k=0.85$) | mm | Hydrological Formula |
| `temp_mean_3d` | 3-day mean 2-meter air temperature | Rolling 3 days | °C | ECMWF ERA5-Land |
| `humidity_mean_3d` | 3-day mean relative humidity at 2 meters | Rolling 3 days | % | ECMWF ERA5-Land |

---

## 2. Hydrological & Soil Moisture Features

| Feature Name | Description & Formula | Unit | Source |
| :--- | :--- | :---: | :--- |
| `soil_moisture_root_zone` | Weighted shallow and root-zone soil water content: $0.4 \cdot SM_{0-7} + 0.6 \cdot SM_{7-28}$ | $m^3/m^3$ | ECMWF ERA5-Land |
| `soil_moisture_deep` | Intermediate and deep subsoil water storage: $SM_{28-100}$ | $m^3/m^3$ | ECMWF ERA5-Land |
| `soil_saturation_ratio` | Ratio of root-zone soil moisture to maximum effective field capacity (porosity $\approx 0.48$ for Kerala lateritic soils): $\min(SM_{rz} / 0.48, 1.0)$ | Dimensionless $[0, 1]$ | Derived |

---

## 3. Terrain & Geomorphological Features

| Feature Name | Description & Formula | Unit | Source |
| :--- | :--- | :---: | :--- |
| `elevation_m` | Absolute elevation above mean sea level | meters | USGS SRTM 30m DEM (GEE) |
| `slope_deg` | Topographic slope angle: $\theta = \arctan(\sqrt{(\partial z / \partial x)^2 + (\partial z / \partial y)^2})$ | degrees (°) | USGS SRTM 30m DEM (GEE) |
| `dist_river_km` | Euclidean distance from station centroid to primary river centerline | kilometers | CWC River Basin Atlas |
| `twi_proxy` | Topographic Wetness Index proxy: $\ln(100 / \tan(\max(\theta, 0.5^\circ)))$ | Dimensionless | Hydrological Formula |
| `elevation_risk_factor` | Lowland inundation susceptibility decay: $\exp(-\text{elevation} / 50.0)$ | Dimensionless $[0, 1]$ | Geomorphic Formula |

---

## 4. Remote Sensing SAR & Optical Proxies

| Feature Name | Description & Formula | Unit | Source |
| :--- | :--- | :---: | :--- |
| `sar_vv` | Sentinel-1 C-band SAR VV backscatter coefficient ($\sigma^\circ$) with water specular attenuation | decibels (dB) | Sentinel-1 SAR GRD |
| `sar_vh` | Sentinel-1 C-band SAR VH cross-polarization backscatter ($\sigma^\circ$) | decibels (dB) | Sentinel-1 SAR GRD |
| `sar_cross_ratio` | Dual-polarization ratio: $VH / (VV - 0.001)$ | Dimensionless | Radar Polarimetry |
| `ndwi_proxy` | Normalized Difference Water Index surface moisture proxy: $\min(\max(0.8 \cdot S_{ratio} - 0.3 + \text{Rain}_{3d}/300, -0.5), 0.8)$ | Dimensionless $[-0.5, 0.8]$ | Derived Spectral Proxy |
| `ndvi_proxy` | Normalized Difference Vegetation Index canopy density proxy: $\min(\max(0.75 - \text{Rain}_{7d}/600, 0.2), 0.85)$ | Dimensionless $[0.2, 0.85]$ | Derived Spectral Proxy |

---

## 5. Exposure, Vulnerability & Target Variables

| Feature Name | Description & Formula | Unit | Source |
| :--- | :--- | :---: | :--- |
| `population_density` | Gridded human population per square kilometer | persons / $km^2$ | CIESIN GPWv4.11 (GEE) |
| `exposure_index` | Normalized population exposure index: $\ln(1 + \text{Pop}) / \ln(1 + 5000)$ | Dimensionless $[0, 1]$ | Derived Vulnerability Proxy |
| `drainage_vulnerability` | Multi-criteria drainage bottleneck score combining elevation risk, slope deficit, and river proximity | Dimensionless $[0, 1]$ | Multi-Criteria Index |
| `vulnerability_index` | Baseline composite socio-environmental vulnerability: $0.6 \cdot V_{drainage} + 0.4 \cdot E_{pop}$ | Dimensionless $[0, 1]$ | Disaster Risk Framework |
| `flood_occurred` | **Primary Target**: Binary indicator (1 if flood occurred, 0 otherwise) | Binary $\{0, 1\}$ | CWC/KSDMA Ground Truth |
| `hazard_level` | **Ordinal Target**: 0: Low, 1: Moderate, 2: High, 3: Very High | Ordinal $\{0, 1, 2, 3\}$ | Multi-Tier Hazard Scale |
