# Missing-Data & Sensor Telemetry Outage Robustness Analysis

**Project Title:** AI-Based Flood Prediction, Dynamic Risk Mapping and Resilience Planning  
**Application:** FloodAI Engine — "AI for a Safer, Resilient India"  
**Standard:** Operational Fault-Tolerance & Sensor Degradation Profiling  

---

## 1. Operational Telemetry Realities in Disaster Zones

In real-world operational hydrological forecasting across India, telemetry networks are vulnerable to severe environmental disruptions during extreme monsoon depressions:
1. **Automated Weather Station (AWS) Gauge Failure:** Severe convective storms, lightning strikes, localized flash floods washing away riverbanks, and communication tower outages routinely sever real-time telemetry from IMD and state rain-gauges.
2. **Satellite Earth Observation Blackouts:** Polar-orbiting Synthetic Aperture Radar (SAR) constellations (Sentinel-1A/C, NISAR) have orbital repeat cycles of 6 to 12 days; during rapidly developing convective deluges, fresh SAR imagery may not be available for up to 5 days.
3. **Soil Moisture In-Situ Sensor Siltation:** Subsurface capacitance soil probes suffer from sediment fouling, water ingress, and battery drainage after prolonged ground immersion.
4. **Packet Loss & Bandwidth Throttling:** Remote IoT sensor nodes in Himalayan headwaters and northeastern tribal belts frequently suffer 5% to 20% random packet loss over cellular GPRS/4G links.

An AI model that performs flawlessly on clean benchmark data but crashes or issues erroneous zero-probability forecasts when a single sensor fails is unacceptable for civil protection.

---

## 2. Experimental Degradation Protocols

To systematically evaluate the operational resilience of HF-STFloodNet and baseline models under real-world sensor degradation, six fault conditions were simulated on the 2023 out-of-time test partition:

### 2.1 Random Feature Missingness (5%, 10%, 20%)
- At each percentage level $p \in \{0.05, 0.10, 0.20\}$, each input feature across the 13,140 test records is randomly and independently masked out with probability $p$.
- Missing values are imputed using historical feature medians established during the 2018–2021 training baseline (standard standardized value of $0.0$).

### 2.2 Complete Satellite SAR Radar Blackout
- Simulates conditions where no fresh Sentinel-1 radar pass is available.
- Features completely masked: `sar_vv`, `sar_vh`, `sar_cross_ratio` (100% missing).

### 2.3 Rain Gauge Telemetry Failure
- Simulates automated weather station network collapse during a catastrophic cloudburst.
- Features completely masked: `rainfall_1d`, `rainfall_3d`, `rainfall_7d`, `rainfall_14d`, `api_14d` (100% missing).

### 2.4 Soil Moisture Sensor Outage
- Simulates subsurface hydrometric sensor failure.
- Features completely masked: `soil_moisture_root_zone`, `soil_moisture_deep`, `soil_saturation_ratio` (100% missing).

---

## 3. Resilience Mechanisms in HF-STFloodNet

HF-STFloodNet incorporates three architectural mechanisms that impart fault-tolerance:
1. **Multimodal Decoupling:** Because hydrometeorological, topographic, radar, and socioeconomic features pass through independent neural sub-encoders before gated fusion, failure in the radar encoder does not corrupt the internal representations learned by the terrain or hydrometeorological encoders.
2. **Gated Evidence Scaling:** The sigmoid gating unit dynamically scales down weights along latent paths that transmit noise or anomalous imputed defaults.
3. **Hybrid Ensemble Redundancy:** When tabular gradient boosting (LightGBM) encounters missing feature values, it branches along default splitting directions; simultaneously, ST-FloodNet provides a complementary smooth latent representation, mitigating sudden step-wise prediction collapses.

---

## 4. Empirical Results & Operational Guidelines

Detailed metrics across all degradation experiments are logged in `outputs/metrics/robustness_results.csv` and visualized in `outputs/figures/sensor_missingness_robustness.png`.

**Key Operational Takeaways:**
- Under random missingness up to 10%, HF-STFloodNet preserves over 94% of its baseline F1 score, demonstrating high operational tolerance to intermittent network packet loss.
- In the event of a total SAR Radar Blackout, the model gracefully falls back on antecedent rainfall accumulation, soil saturation, and topographic low-elevation indices, maintaining robust detection recall ($\ge 85\%$).
- Complete rain gauge telemetry failure represents the most severe failure mode; however, the presence of upstream discharge proxies, river stage anomalies, and satellite radar prevents complete warning failure.
