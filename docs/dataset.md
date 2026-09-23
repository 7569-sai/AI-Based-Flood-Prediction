# Dataset Documentation: Kerala River Basins Hydrometeorological & Geospatial Dataset

## 1. Overview & Provenance

* **Dataset Name**: Kerala River Basins Hydrometeorological and Remote-Sensing Dataset (2018–2023)
* **Coverage**: 16 monitoring stations across 8 major river basins in Kerala, Southwestern India
* **Temporal Resolution**: Daily observations from January 1, 2018 to December 31, 2023 (6 full years, 2,191 days per station)
* **Total Records**: 35,056 records
* **Raw File**: `data/raw/kerala_flood_raw.csv` (SHA-256: `a7e9fc5d40372261051c98e9f2a0d17c3ebbbdfa9d1889a8786a5ee37766f60f`)
* **Manifest**: `data/DATA_MANIFEST.json`

---

## 2. Data Sources & Licensing

| Domain | Source Provider | Specific Product / API | Spatial/Temporal Resolution | License |
| :--- | :--- | :--- | :--- | :--- |
| **Meteorology & Hydrology** | ECMWF / Open-Meteo | ERA5-Land Reanalysis Archive | ~9 km gridded, Daily | CC BY 4.0 |
| **Digital Elevation Model** | USGS / NASA | SRTM GL1 30m (`USGS/SRTMGL1_003`) via GEE | 30 meter spatial resolution | NASA Open Data Policy |
| **Population Density** | CIESIN / Columbia Univ. | GPWv4.11 (`CIESIN/GPWv411/GPW_UNWPP-Adjusted`) | 1 km gridded | CC BY 4.0 |
| **SAR Remote Sensing** | ESA Copernicus | Sentinel-1 C-band SAR GRD (`COPERNICUS/S1_GRD`) | 10 meter spatial resolution | Copernicus Open Access |
| **Ground Truth Flood Events** | CWC / KSDMA | Central Water Commission & Kerala State Disaster Management Authority | Event-specific emergency bulletins | Public Domain / Open Gov Data |

---

## 3. Station Roster & Geographic Distribution

The 16 monitoring stations represent distinct hydrological sub-catchments and physiographic zones:

| Station ID | Station Name | Basin | District | Latitude (°N) | Longitude (°E) | Elevation (m) | Slope (°) | Pop. Density (ppl/km²) |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `KL_01` | Aluva | Periyar | Ernakulam | 10.1076 | 76.3516 | 10.0 | 4.72 | 1,359 |
| `KL_02` | Kalady | Periyar | Ernakulam | 10.1667 | 76.4333 | 12.0 | 3.88 | 654 |
| `KL_03` | Malayattoor | Periyar | Ernakulam | 10.1982 | 76.4975 | 21.0 | 4.63 | 1,340 |
| `KL_04` | Chalakkudy | Chalakkudy | Thrissur | 10.3070 | 76.3330 | 21.0 | 4.68 | 636 |
| `KL_05` | Thrissur Town | Karuvannur | Thrissur | 10.5276 | 76.2144 | 20.0 | 2.10 | 1,267 |
| `KL_06` | Chengannur | Pamba | Alappuzha | 9.3176 | 76.6111 | 13.0 | 0.94 | 1,225 |
| `KL_07` | Ranni | Pamba | Pathanamthitta | 9.3800 | 76.7800 | 5.0 | 2.08 | 950 |
| `KL_08` | Kuttanad_Mankombu | Pamba-Meenachil | Alappuzha | 9.4500 | 76.4833 | -1.0 | 3.35 | 687 |
| `KL_09` | Alappuzha Coastal | Coastal Plain | Alappuzha | 9.4981 | 76.3388 | 7.0 | 0.00 | 3,874 |
| `KL_10` | Kottayam_Meenachil | Meenachil | Kottayam | 9.5916 | 76.5222 | 28.0 | 5.63 | 1,425 |
| `KL_11` | Idukki_Painavu | Upper Periyar | Idukki | 9.8500 | 76.9700 | 672.0 | 28.21 | 422 |
| `KL_12` | Munnar | Muthirapuzha | Idukki | 10.0889 | 77.0595 | 1,454.0 | 3.87 | 107 |
| `KL_13` | Palakkad | Bharathapuzha | Palakkad | 10.7867 | 76.6548 | 77.0 | 0.93 | 773 |
| `KL_14` | Kozhikode_Feroke | Chaliyar | Kozhikode | 11.1833 | 75.8333 | 12.0 | 2.98 | 1,628 |
| `KL_15` | Mananthavady | Kabini | Wayanad | 11.8025 | 76.0034 | 756.0 | 5.86 | 319 |
| `KL_16` | Thiruvananthapuram | Karamana | Thiruvananthapuram | 8.5241 | 76.9366 | 19.0 | 8.38 | 4,314 |

---

## 4. Feature Dictionary

A comprehensive feature dictionary mapping raw and engineered variables:

### A. Raw Variables
* `date`: Observation date (`YYYY-MM-DD`).
* `precipitation_sum`: Total daily precipitation ($mm$).
* `rain_sum`: Total daily liquid rain ($mm$).
* `temperature_2m_max`, `temperature_2m_min`, `temperature_2m_mean`: Daily 2-meter air temperature (°C).
* `relative_humidity_2m_mean`: Daily mean relative humidity (%).
* `surface_pressure_mean`: Atmospheric surface pressure ($hPa$).
* `soil_moisture_0_to_7cm_mean`: Topsoil volumetric water content ($m^3/m^3$).
* `soil_moisture_7_to_28cm_mean`: Subsurface root-zone soil moisture ($m^3/m^3$).
* `soil_moisture_28_to_100cm_mean`: Intermediate deep soil moisture ($m^3/m^3$).
* `soil_moisture_100_to_255cm_mean`: Deep groundwater reservoir moisture ($m^3/m^3$).
* `elevation_m`: Elevation above sea level ($m$) from SRTM 30m.
* `slope_deg`: Terrain slope in degrees ($^\circ$) from SRTM.
* `population_density`: Population density ($persons/km^2$) from CIESIN GPWv4.
* `dist_river_km`: Approximate distance to primary river centerline ($km$).
* `sar_vv_ref`, `sar_vh_ref`: Baseline C-band SAR backscatter values ($dB$).

---

## 5. Limitations & Ethical Considerations

1. **Reanalysis Resolution**: ECMWF ERA5-Land reanalysis is resolved at 0.1° (~9km), smoothing extreme hyper-local micro-scale cloudburst peaks.
2. **Terrain Invariance**: Elevation and slope are static features per station centroid.
3. **No Direct Dam Operation Feature**: Dam gate openings (e.g. Idukki and Mullaperiyar dams in August 2018) contribute significantly to downstream river stage, which is captured indirectly via antecedent rainfall and runoff volume.
