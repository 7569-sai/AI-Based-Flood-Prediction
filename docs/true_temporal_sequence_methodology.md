# True Chronological Temporal Sequence Methodology

**Project Title:** AI-Based Flood Prediction, Dynamic Risk Mapping and Resilience Planning  
**Application:** FloodAI Engine — "AI for a Safer, Resilient India"  
**Module:** Genuine Temporal Sequence Tensor Formulation (Zero Pseudo-Sequence)  
**Document Status:** Approved Scientific Protocol  

---

## 1. Motivation: Eliminating Pseudo-Sequence Approximations

In preliminary development iterations, spatiotemporal deep learning models frequently adopt computational shortcuts when processing tabular time-series data—such as unsqueezing a single day's 1D feature vector and repeating it across $T$ pseudo-timesteps (`x_dyn.unsqueeze(-1).repeat(1, 1, 3)`). While this satisfies neural network tensor dimensions ($[B, T, F]$), it is **scientifically invalid** as a true temporal sequence model because:
1. It does not provide historical lag dynamics or hydrograph shape.
2. It assumes time-invariance across the lookback window, preventing the recurrent layers (Bi-GRU) and temporal self-attention from discovering storm burst inflection points.

Under this final scientific correction pass, all pseudo-sequence repetitions are strictly removed from the research training path. The system now ingests **100% genuine chronological time-series tensors** constructed directly from consecutive daily station observations.

---

## 2. Temporal Sequence Construction & Alignment

### 2.1 Window Formulation
For any monitoring station $s \in \{1, \dots, 36\}$ on prediction calendar day $t$:
- The dynamic hydrometeorological state sequence is formed by the trailing $T = 14$ consecutive days:
$$\mathbf{X}_{\text{dynamic}}^{(s)}(t) = \begin{bmatrix}
\mathbf{x}_{\text{dyn}}^{(s)}(t - 13) \\
\mathbf{x}_{\text{dyn}}^{(s)}(t - 12) \\
\vdots \\
\mathbf{x}_{\text{dyn}}^{(s)}(t)
\end{bmatrix} \in \mathbb{R}^{14 \times 13}$$

- The static / snapshot state vector at prediction day $t$ comprises:
$$\mathbf{x}_{\text{static}}^{(s)}(t) = \left[ \mathbf{x}_{\text{terrain}}^{(s)}, \, \mathbf{x}_{\text{RS}}^{(s)}(t), \, \mathbf{x}_{\text{exposure}}^{(s)} \right] \in \mathbb{R}^{14}$$
where:
  - $\mathbf{x}_{\text{terrain}}^{(s)} \in \mathbb{R}^6$: Elevation, slope, river distance, TWI, elevation risk factor, drainage vulnerability.
  - $\mathbf{x}_{\text{RS}}^{(s)}(t) \in \mathbb{R}^5$: Sentinel-1 SAR VV/VH, polarimetric cross-ratio, Sentinel-2 NDWI proxy, NDVI proxy at day $t$.
  - $\mathbf{x}_{\text{exposure}}^{(s)} \in \mathbb{R}^3$: Population density, exposure index, vulnerability index.

### 2.2 Target Alignment
The target labels (binary flood hazard $y^{(s)}(t) \in \{0, 1\}$ and ordinal severity $s^{(s)}(t) \in \{0, 1, 2, 3\}$) correspond strictly to calendar day $t$.

---

## 3. Discard Policy for Incomplete Historical Windows

To prevent synthetic row-tiling or historical fabrication:
1. **Zero Padding / Repeating Policy:** A sequence is valid if and only if all 14 consecutive daily records exist in authentic station chronological logs.
2. **First-Year Boundary Window Discard:** Because observational logs begin on 2018-01-01, dates from 2018-01-01 to 2018-01-13 possess fewer than 14 days of history. These initial 13 days per station are **strictly dropped**.
3. **Seamless Multi-Year Continuity:** Because station observations are consecutive across the multi-year timeline (2018–2023):
   - The validation split (starting 2022-01-01) draws its 14-day trailing history from late December 2021.
   - The test split (starting 2023-01-01) draws its 14-day trailing history from late December 2022.
   - Zero days are discarded from validation or test sets.

### Sequence Accounting Table

| Data Split | Calendar Range | Raw Records | Discarded Windows | Valid 14-Day Sequences | Percentage Retained |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Training Split** | 2018-01-01 to 2021-12-31 | 52,596 | 468 (first 13 days × 36 stations) | **52,128** | 99.11% |
| **Validation Split** | 2022-01-01 to 2022-12-31 | 13,140 | 0 (trailing history from Dec 2021) | **13,140** | 100.00% |
| **Test Split** | 2023-01-01 to 2023-12-31 | 13,140 | 0 (trailing history from Dec 2022) | **13,140** | 100.00% |
| **Full Multi-Basin Stack** | 2018-01-01 to 2023-12-31 | **78,876** | **468** | **78,408** | **99.41%** |

---

## 4. 13 Dynamic Timestep Features vs 14 Static/Snapshot Features

### Dynamic Sequence Features ($F_{\text{dyn}} = 13$, evaluated at each $\tau \in [t-13, t]$):
1. `rainfall_1d`: Daily precipitation accumulation (mm)
2. `rainfall_3d`: 3-day backward rolling sum (mm)
3. `rainfall_7d`: 7-day backward rolling sum (mm)
4. `rainfall_14d`: 14-day backward rolling sum (mm)
5. `rainfall_anomaly_30d`: Monthly precipitation departure (mm)
6. `api_14d`: Antecedent Precipitation Index (mm)
7. `temp_mean_3d`: 2m air temperature (°C)
8. `humidity_mean_3d`: Relative humidity (%)
9. `soil_moisture_root_zone`: 0–7 cm soil moisture ($\text{m}^3/\text{m}^3$)
10. `soil_moisture_deep`: 7–28 cm deep soil moisture ($\text{m}^3/\text{m}^3$)
11. `soil_saturation_ratio`: Root-zone saturation relative to field capacity
12. `river_discharge_proxy`: Upstream catchment discharge proxy ($\text{m}^3/\text{s}$)
13. `river_stage_anomaly`: Water level departure from danger mark (m)

### Static / Snapshot Features ($F_{\text{static}} = 14$, evaluated at day $t$):
- **Topographic Terrain (6 features):** `elevation_m`, `slope_deg`, `dist_river_km`, `twi_proxy`, `elevation_risk_factor`, `drainage_vulnerability`
- **Remote Sensing Radar & Optical (5 features):** `sar_vv`, `sar_vh`, `sar_cross_ratio`, `ndwi_proxy`, `ndvi_proxy`
- **Socioeconomic Exposure (3 features):** `population_density`, `exposure_index`, `vulnerability_index`

---

## 5. Neural Information Flow

```
Dynamic Sequence [B, 14, 13]
      │
      ▼
Parallel 1D Temporal Convolutions (k=3, k=5, channels=16 each)
      │
      ▼
Concatenated Temporal Features [B, 14, 32]
      │
      ▼
Bidirectional GRU (2 layers, hidden_size=32 each dir) [B, 14, 64]
      │
      ▼
Temporal Self-Attention Context Pooling
      │
      ▼
Attended Dynamic Hydrometeorology Embedding h_temp [B, 64]
      │
      ├──────────────────────┬──────────────────────┬──────────────────────┐
      │                      │                      │                      │
      ▼                      ▼                      ▼                      ▼
  h_temp [B, 64]      h_terrain [B, 32]        h_sat [B, 16]          h_exp [B, 16]
(from Dynamic Seq)   (from Spatial MLP)    (from Satellite MLP)    (from Exposure MLP)
      │                      │                      │                      │
      └──────────────────────┴──────────┬───────────┴──────────────────────┘
                                        │
                                        ▼
                         Concatenation Vector [B, 128]
                                        │
                                        ▼
                       Dynamic Sigmoid Gating Unit [B, 128]
                                        │
                                        ▼
                   Gated Representation Bottleneck [B, 64]
                                        │
                        ┌───────────────┴───────────────┐
                        ▼                               ▼
                 Hazard Head [B, 1]              Severity Head [B, 4]
```
