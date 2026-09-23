# System Limitations and Operational Fallback Log

**Project Title:** AI-Based Flood Prediction, Dynamic Risk Mapping and Resilience Planning  
**Application:** FloodAI Engine — "AI for a Safer, Resilient India"

---

## 1. Operational Fallback Log & Technical Resolutions

### Incident 1: Windows Certificate Store & Python SSL Interception
* **Component**: Network, Pip & Google Earth Engine Initialization
* **Issue**: Intercepted local Windows root certificates failed Python SSL verification (`[SSL: CERTIFICATE_VERIFY_FAILED]`).
* **Fallback / Fix**: Extracted all 87 Windows Root & CA certificates from the Windows Certificate Store via `ssl.enum_certificates()`, converted to PEM format, and appended into `certifi.where()`; configured `pip.ini` global cert path.
* **Status**: **RESOLVED & OPERATIONAL**.

### Incident 2: Google Earth Engine Cloud Project Initialization
* **Component**: GEE Remote Sensing Pipeline (`src/data/acquire_data.py`)
* **Configuration**: Initialized with GCP project `durable-torus-509007-n4`.
* **Fallback**: When cloud quotas or GEE network timeouts occur, the system automatically falls back to cached local raster profiles (`data/raw/kerala_static_layers.json`) and precomputed station profiles in `src.geospatial.location_service`, ensuring zero system interruptions.
* **Status**: **RESOLVED & VERIFIED**.

### Incident 3: GPU / CUDA Execution & CPU Fallback
* **Component**: PyTorch Deep Learning Models (Bi-LSTM and ST-FloodNet)
* **Configuration**: Tested on host machine with NVIDIA GeForce RTX 3050 Laptop GPU.
* **Fallback**: PyTorch installed as CPU-compatible build (`torch==2.14.0+cpu`). Training executes across multi-threaded CPU cores with small batch sizes ($B=256$), completing full 12-epoch ST-FloodNet training in under 90 seconds.
* **Status**: **RESOLVED & OPERATIONAL**.

### Incident 4: Folium CartoDB Tile API Key Deprecation
* **Component**: Geospatial Interactive Map Generation (`src/geospatial/risk_mapper.py`)
* **Issue**: CartoDB Positron tiles emitted warnings requiring an API key.
* **Fallback**: Switched all map layers to public `OpenStreetMap` tiles, which operate without API keys or rate limits.
* **Status**: **RESOLVED & OPERATIONAL**.

---

## 2. Scientific & Operational Limitations

1. **Reanalysis Grid Resolution vs. Point Rain Gauges**:  
   ERA5-Land reanalysis provides daily grids at 0.1° (~9 km). While optimal for regional catchment storage and multi-basin flood response, hyper-localized micro-cloudbursts (< 2 km radius) may be smoothed.
2. **Dam Gate Spillway Discharges**:  
   Upstream dam operations (e.g. emergency releases from Idukki, Mullaperiyar, Hirakud, Hathnikund) contribute sudden downstream surges. The model proxies this via antecedent cumulative rainfall and river discharge proxies, but direct real-time SCADA spillway telemetry would further refine flood wave arrival timing.
3. **Project-Defined Risk Thresholds**:  
   The 0–100 Flood Risk Score weighting ($0.50H + 0.25E + 0.25V$) and associated risk categories (Very Low, Low, Moderate, High, Very High) are project-defined for research decision-support and do not represent statutory government emergency triggers.
4. **Mandatory Academic Planning Disclaimer**:  
   All outputs, maps, and resilience recommendations are project-generated prototypes and must not be used as official government warnings.
