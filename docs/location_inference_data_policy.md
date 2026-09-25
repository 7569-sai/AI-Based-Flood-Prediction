# Location-First Inference Data & Temporal Leakage Policy

**Project Title:** AI-Based Flood Prediction, Dynamic Risk Mapping and Resilience Planning  
**Application:** FloodAI Engine — "AI for a Safer, Resilient India"  
**Scope:** Real-Time Location-First Feature Extraction & Caching

---

## 1. Zero-Leakage Temporal Policy

To ensure scientific integrity during point-based inference:
1. **Strictly Backward-Looking Lags**:
   - At query timestamp $T$, all rolling rainfall metrics ($R_{1d}, R_{3d}, R_{7d}, R_{14d}$) draw strictly from preceding calendar days:
     $$\mathbf{P}_{\text{antecedent}} = \{P(T-14), P(T-13), \dots, P(T-2), P(T-1)\}$$
   - Precipitation on current day $T$ (and any future numerical weather prediction hours) is **strictly excluded** from antecedent predictors.
2. **Hydrological & Soil Moisture Alignment**:
   - Soil moisture measurements represent conditions at $T-1$, preventing circular instantaneous hydrological feedback loops.
3. **No Target Leakage in Satellite Imagery**:
   - Optical (Sentinel-2) and SAR (Sentinel-1) acquisitions reflect pre-inundation surface state or contemporaneous revisits. Post-disaster recovery imagery is never used as a predictive feature.

---

## 2. API Caching & Rate-Limiting Policy

To preserve third-party API quotas and ensure rapid response:
1. **Spatial Discretization**:
   - API responses are cached on disk at `data/raw/cache/` using rounded geographic coordinates ($\pm 0.001^\circ \approx 110\text{ m}$).
2. **Cache Key Structure**:
   $$\text{Key} = \text{MD5}\left(\text{prefix} + \text{round}(\text{lat}, 3) + \text{round}(\text{lon}, 3) + \text{date}\right)$$
3. **Graceful Fallback**:
   - If an external network service fails or times out (5-second threshold), the system draws from the nearest spatial station climatological profile from `src.geospatial.location_service`, marking the retrieval status as `⚠ Fallback Used`.
   - Prediction is aborted if essential data is missing and cannot be validated.
