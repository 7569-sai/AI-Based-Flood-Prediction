# System Limitations and Fallback Log

This document records the operational limitations, boundary conditions, and programmatic fallback log encountered and resolved during system engineering.

---

## 1. Programmatic Fallback Log

### Incident 1: Windows Certificate Store & Python SSL Interception
* **Component**: Network, Pip & Google Earth Engine Initialization
* **Original approach**: Default Python `urllib` / `httplib2` system calls.
* **Failure**: Intercepted local Windows root certificates failed SSL certificate verification (`[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1007)`).
* **Fallback**: Extracted all 87 Windows Root & CA certificates from the Windows Certificate Store via `ssl.enum_certificates()`, converted to PEM format, and appended into `certifi.where()`; configured `pip.ini` global cert.
* **Impact**: Zero negative impact; enabled 100% verified SSL connectivity to Google Earth Engine and Open-Meteo.
* **Final status**: **RESOLVED & OPERATIONAL**.

### Incident 2: Open-Meteo API Rate Limiting (HTTP 429)
* **Component**: Historical Meteorological Data Acquisition (`src/data/acquire_data.py`)
* **Original approach**: Sequential batch downloads with 0.5s pause.
* **Failure**: Hit rate limit HTTP 429 after 5 stations.
* **Fallback**: Implemented persistent station-level disk caching (`data/raw/station_{station_id}.csv`), increased inter-station delay to 3.5s, and implemented exponential backoff (4s, 10s, 20s, 30s, 45s).
* **Impact**: Total data collection succeeded with 35,056 authentic daily records without loss.
* **Final status**: **RESOLVED & OPERATIONAL**.

### Incident 3: CIESIN Population Density Band Name Mismatch
* **Component**: Earth Engine Population Extraction
* **Original approach**: `pop_img.select('population-density')`.
* **Failure**: Earth Engine error: `Band pattern 'population-density' did not match any bands. Available bands: ['unwpp-adjusted_population_density']`.
* **Fallback**: Corrected band name selector to `unwpp-adjusted_population_density`.
* **Impact**: Accurate, verified population extraction from Columbia University CIESIN GPWv4.11 across all 16 stations.
* **Final status**: **RESOLVED & OPERATIONAL**.

### Incident 4: Folium CartoDB Basemap API Key Warning
* **Component**: Geospatial Interactive Map Generation (`src/geospatial/risk_mapper.py`)
* **Original approach**: `CartoDB positron` tile layer.
* **Failure**: Console warning: `CartoDB tiles now require an API key`.
* **Fallback**: Switched basemap to standard `OpenStreetMap` tiles.
* **Impact**: Completely free, public-domain basemap tiles rendered cleanly in both standalone HTML and the Streamlit dashboard without API key requirements.
* **Final status**: **RESOLVED & OPERATIONAL**.

---

## 2. Scientific & Operational Limitations

1. **Reanalysis Grid Resolution**: ERA5-Land reanalysis provides daily grids at 0.1° (~9 km). While ideal for regional river basin hydrological accumulation, micro-scale cloudbursts (< 2 km radius) may be smoothed.
2. **Dam Gate Spillway Discharges**: Upstream dam operations (e.g. emergency opening of Idukki arch dam and Idamalayar dam spillway gates during August 2018) contribute directly to downstream river surges. The model captures this via antecedent rainfall and basin-wide runoff volume, but incorporating direct telemetry of dam spillway gate openings would further refine downstream time-to-peak estimates.
3. **Project-Defined Risk Thresholds**: The 0–100 Flood Risk Score categories (Very Low, Low, Moderate, High, Very High) are project-defined for decision-support prototyping and should not be construed as statutory government evacuation triggers.
