# Dataset Architecture & Geographic Representation

**Project Title:** AI-Based Flood Prediction, Dynamic Risk Mapping and Resilience Planning  
**Application:** FloodAI Engine — "AI for a Safer, Resilient India"

---

## 1. Dataset Overview

The FloodAI Engine utilizes a multimodal, spatiotemporally aligned dataset spanning 6 full years (January 1, 2018 to December 31, 2023) across 36 strategic national flood-monitoring stations in 12 major Indian river basins, comprising **78,876 daily continuous observations**.

### Key Dataset Statistics
- **Primary Data Format**: Apache Parquet (`data/processed/india/pan_india_features.parquet`) with snappy compression (2.19 MB).
- **Partitioning**: Partitioned by river basin in `data/processed/basins/` for scalable lazy loading.
- **SHA-256 Hash**: `137af604b4c035c19a927373cb1311f22df976b7f80ce9213410df202fb14692`
- **Total Flood Events**: 1,353 days (1.72% natural flood incidence, reflecting genuine severe flood sparsity in India).

---

## 2. Multi-Source Provenance & Data Stack

| Source | Domain / Variables | Spatial Resolution | Access Method / License | Role in System |
| :--- | :--- | :--- | :--- | :--- |
| **ECMWF ERA5-Land** | Precipitation, Temperature, Humidity, Soil Moisture (0–7cm, 7–28cm) | 0.1° (~9 km) | Open-Meteo Historical Weather API / CC BY 4.0 | Provides continuous 2018–2023 atmospheric and root-zone moisture conditions. |
| **USGS SRTM 30m DEM** | Elevation (m), Topographic Slope (°), Aspect, TWI proxy | 30 meters | Google Earth Engine (`USGS/SRTMGL1_003`) / Public Domain | Provides physical geomorphic constraints on runoff accumulation and gravity drainage. |
| **Copernicus Sentinel-1** | Dual-polarization C-band SAR (VV and VH backscatter in dB) | 10–20 meters | Google Earth Engine (`COPERNICUS/S1_GRD`) / Copernicus Open Access | Captures surface water extent via specular radar backscatter attenuation. |
| **CIESIN GPWv4** | Population count and density (persons / km²) | 1 km (30 arc-sec) | GEE (`CIESIN/GPWv411/GPW_POPULATION_DENSITY`) / NASA Open Data | Quantifies human demographic exposure for dynamic disaster risk scoring. |
| **Central Water Commission (CWC)** | Historical danger levels, barrage discharges, and flood reports | Station Gauges | CWC Flood Forecasting / Open Data | Validates historical flood occurrences and seasonal surge timing. |

---

## 3. Geographic Coverage: 12 Major Indian River Basins

The 36 national stations span all primary geographic regions of India:
1. **Brahmaputra & Barak Basins (Assam)**: Guwahati, Dibrugarh, Silchar, Kaziranga (extreme rainfall, braided low-gradient channels).
2. **Yamuna Basin (Delhi & UP)**: Delhi Old Railway Bridge / ITO, Mathura.
3. **Ganga Basin (Uttarakhand, UP, Bihar, West Bengal)**: Haridwar, Kanpur, Prayagraj, Varanasi, Patna, Bhagalpur, Kolkata Hooghly.
4. **Kosi Basin (Bihar)**: Saharsa / Supaul (frequent avulsion, rapid Himalayan runoff).
5. **Mahanadi Basin (Odisha)**: Sambalpur Hirakud, Cuttack Delta, Puri Lowlands (cyclonic storm surge, delta backwater locking).
6. **Godavari Basin (Maharashtra, Telangana, AP)**: Nashik, Nanded, Bhadrachalam, Rajahmundry.
7. **Krishna Basin (Maharashtra, Karnataka, AP)**: Sangli / Kolhapur Panchaganga, Vijayawada Prakasham Barrage.
8. **Narmada Basin (MP, Gujarat)**: Jabalpur, Hoshangabad, Bharuch.
9. **Tapi Basin (MP, Gujarat)**: Burhanpur, Surat.
10. **Kaveri Basin (Karnataka, Tamil Nadu)**: KRS Dam Mysore, Tiruchirappalli, Thanjavur Delta.
11. **Periyar, Chalakkudy & Pamba Basins (Kerala)**: Aluva, Chalakkudy, Kuttanad / Mankombu (-1m below sea level).
12. **Northern Basins (J&K, Punjab, Himachal)**: Srinagar Jhelum, Firozpur Sutlej, Kullu / Mandi Beas.

---

## 4. Chronological & Spatial Partitions

1. **Chronological Splits (Temporal Evaluation)**:
   - **Train (2018–2021)**: 52,596 observations (929 flood events)
   - **Validation (2022)**: 13,140 observations (237 flood events)
   - **Test (2023)**: 13,140 observations (187 flood events)
2. **Spatial Holdout Splits (Cross-Basin Generalization)**:
   - **Training Basins (10 Basins)**: 67,921 observations
   - **Held-Out Basins (Brahmaputra & Mahanadi)**: 10,955 observations (415 flood events)
