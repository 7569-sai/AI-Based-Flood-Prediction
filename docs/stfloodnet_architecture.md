# ST-FloodNet Architecture Specification & Tensor Dimensions

**Project Title:** AI-Based Flood Prediction, Dynamic Risk Mapping and Resilience Planning  
**Application:** FloodAI Engine — "AI for a Safer, Resilient India"  
**Model Name:** Spatiotemporal Multimodal Flood Network (ST-FloodNet)  
**Status:** Approved Research Architecture  

---

## 1. Architectural Overview & Design Rationale

ST-FloodNet is a multi-modal, deep neural architecture designed to address the complex physical interplay of flood genesis in heterogeneous river basins. Hydrological flooding is neither purely a time-series forecasting problem nor solely a spatial terrain classification problem; it is driven by non-linear interactions across four distinct physical domains:
1. **Dynamic Hydrometeorology:** Trailing precipitation bursts, antecedent soil saturation, and upstream catchment discharge anomalies.
2. **Topographic Geomorphology:** Lowland depression, slope steepness, river channel proximity, and drainage bottlenecks.
3. **Satellite Earth Observation:** C-band Synthetic Aperture Radar (SAR) specular water attenuation and optical surface water/vegetation indices.
4. **Socioeconomic Vulnerability:** Population density concentration and structural coping capacity deficits.

ST-FloodNet decouples these modalities into specialized neural encoders, recombines them via **Gated Cross-Modal Fusion**, and outputs multi-task predictions (flood hazard probability and 4-tier flood severity) alongside epistemic uncertainty quantified through **Monte Carlo Dropout**.

```
                           ST-FLOODNET ARCHITECTURE
                           
 [ Dynamic Sequence: B × T × 13 ] ────────┐
   │ (Precipitation, API, Soil Moisture,  │
   │  Discharge, Stage Anomaly)           ▼
   ├── Multi-Scale 1D Conv (k=3, k=5) ───► [ Temporal Memory: Bi-GRU (64) ]
   │                                                    │
   └────────────────────────────────────────────► [ Temporal Attention ] ──► [ h_temp: 64 ]
                                                                                   │
 [ Terrain Geomorphic: B × 6 ] ─────────────────► [ Spatial MLP + BN + Dropout ] ─► [ h_spat: 32 ]
   (Elevation, Slope, Dist River, TWI, etc.)                                       │
                                                                                   │
 [ Remote Sensing: B × 5 ] ─────────────────────► [ Satellite Radar/Optical MLP ] ─► [ h_sat:  16 ]
   (SAR VV/VH, Cross-ratio, NDWI, NDVI)                                            │
                                                                                   │
 [ Socioeconomic Exposure: B × 3 ] ─────────────► [ Exposure MLP + BN + Dropout ] ─► [ h_socio:16 ]
   (Population Density, Exposure, Vulnerability)                                   │
                                                                                   ▼
                                                             [ Concatenation: B × 128 ]
                                                                       │
                                                             [ Gated Cross-Modal Unit ]
                                                             Gate = σ(W_g · h_cat)
                                                             h_gated = h_cat ⊙ Gate
                                                                       │
                                                             [ Bottleneck FC: B × 64 ]
                                                                       │
                                                      ┌────────────────┴────────────────┐
                                                      ▼                                 ▼
                                             [ Head A: Hazard ]               [ Head B: Severity ]
                                             Binary Logit: B × 1              4-Class Logits: B × 4
                                             (P(Flood) via σ)                 (Softmax Distribution)
```

---

## 2. Complete 27-Feature Modular Routing

| Module | Features Included (Indices) | Input Dim | Sub-Layers & Operations | Output Dim |
| :--- | :--- | :---: | :--- | :---: |
| **Module 1: Dynamic Hydrometeorological** | `rainfall_1d`, `rainfall_3d`, `rainfall_7d`, `rainfall_14d`, `rainfall_anomaly_30d`, `api_14d`, `temp_mean_3d`, `humidity_mean_3d`, `soil_moisture_root_zone`, `soil_moisture_deep`, `soil_saturation_ratio`, `river_discharge_proxy`, `river_stage_anomaly` *(Indices 0–12)* | $[B, T, 13]$ | • Multi-scale Conv1D ($k_1=3, k_2=5$, ch=16 each)<br>• 2-layer Bidirectional GRU (hidden=32)<br>• Self-Attention Context Pooling | $[B, 64]$ |
| **Module 2: Topographic Geomorphology** | `elevation_m`, `slope_deg`, `dist_river_km`, `twi_proxy`, `elevation_risk_factor`, `drainage_vulnerability` *(Indices 13–18)* | $[B, 6]$ | • Dense(6 $\to$ 32)<br>• BatchNorm1d(32)<br>• ReLU + Dropout($p=0.35$) | $[B, 32]$ |
| **Module 3: Satellite Earth Observation** | `sar_vv`, `sar_vh`, `sar_cross_ratio`, `ndwi_proxy`, `ndvi_proxy` *(Indices 19–23)* | $[B, 5]$ | • Dense(5 $\to$ 16)<br>• BatchNorm1d(16)<br>• ReLU + Dropout($p=0.35$) | $[B, 16]$ |
| **Module 4: Socioeconomic Exposure** | `population_density`, `exposure_index`, `vulnerability_index` *(Indices 24–26)* | $[B, 3]$ | • Dense(3 $\to$ 16)<br>• BatchNorm1d(16)<br>• ReLU + Dropout($p=0.35$) | $[B, 16]$ |
| **Cross-Modal Fusion** | Concatenation of Modules 1–4 | $[B, 128]$ | • Dynamic Fusion Gate: $\mathbf{g} = \sigma(\mathbf{W}_g \mathbf{f} + \mathbf{b}_g)$<br>• Gated Elementwise Scaling: $\mathbf{f}_{\text{gated}} = \mathbf{f} \odot \mathbf{g}$<br>• Representation Bottleneck: Dense(128 $\to$ 64) + ReLU | $[B, 64]$ |
| **Multi-Task Heads** | Bottleneck Latent Vector | $[B, 64]$ | • Head A: Dense(64 $\to$ 1) (Hazard Logit)<br>• Head B: Dense(64 $\to$ 4) (Severity Logits) | $[B, 1]$<br>$[B, 4]$ |

**Verification Check:**  
$13 + 6 + 5 + 3 = 27$ features. Exactly 100% of input variables are routed into physical modules. Zero features are dropped, bypassed, or unused.

---

## 3. Mathematical Operations & Tensor Dimensions

### 3.1 Temporal Hydrometeorological Attention
Given dynamic input sequence $\mathbf{X}_{\text{dyn}} \in \mathbb{R}^{B \times T \times 13}$:
1. Permute to channel-first representation: $\mathbf{X}_{\text{conv}} \in \mathbb{R}^{B \times 13 \times T}$.
2. Parallel 1D convolutions:
   $$\mathbf{C}_1 = \text{ReLU}(\text{Conv1D}_{k=3}(\mathbf{X}_{\text{conv}})) \in \mathbb{R}^{B \times 16 \times T}$$
   $$\mathbf{C}_2 = \text{ReLU}(\text{Conv1D}_{k=5}(\mathbf{X}_{\text{conv}})) \in \mathbb{R}^{B \times 16 \times T}$$
3. Feature channel concatenation: $\mathbf{C} = [\mathbf{C}_1; \mathbf{C}_2] \in \mathbb{R}^{B \times 32 \times T}$.
4. Permute back to time-first: $\mathbf{C}_{\text{time}} \in \mathbb{R}^{B \times T \times 32}$.
5. Bidirectional GRU recurrent processing:
   $$\mathbf{H} = \text{BiGRU}(\mathbf{C}_{\text{time}}) \in \mathbb{R}^{B \times T \times 64}$$
6. Additive temporal attention pooling:
   $$u_t = \mathbf{v}^T \tanh(\mathbf{W}_a \mathbf{h}_t + \mathbf{b}_a), \quad \alpha_t = \frac{\exp(u_t)}{\sum_{\tau=1}^T \exp(u_\tau)}$$
   $$\mathbf{h}_{\text{temp}} = \sum_{t=1}^T \alpha_t \mathbf{h}_t \in \mathbb{R}^{B \times 64}$$

### 3.2 Gated Cross-Modal Fusion
Concatenate feature representations across all four physical domains:
$$\mathbf{f}_{\text{fusion}} = [\mathbf{h}_{\text{temp}}; \mathbf{h}_{\text{terrain}}; \mathbf{h}_{\text{satellite}}; \mathbf{h}_{\text{exposure}}] \in \mathbb{R}^{B \times 128}$$

Compute cross-modal gating coefficient vector:
$$\mathbf{g} = \sigma(\mathbf{W}_g \mathbf{f}_{\text{fusion}} + \mathbf{b}_g) \in (0, 1)^{B \times 128}$$
$$\mathbf{f}_{\text{gated}} = \mathbf{f}_{\text{fusion}} \odot \mathbf{g} \in \mathbb{R}^{B \times 128}$$

Project into cross-modal bottleneck:
$$\mathbf{h}_{\text{rep}} = \text{ReLU}(\mathbf{W}_b \mathbf{f}_{\text{gated}} + \mathbf{b}_b) \in \mathbb{R}^{B \times 64}$$

### 3.3 Epistemic Uncertainty via Monte Carlo Dropout
At inference time, keep dropout active ($p=0.35$) and execute $M=30$ stochastic forward passes:
$$\hat{p}_m = \sigma(\text{Head}_A(\mathbf{h}_{\text{rep}}^{(m)})), \quad m \in \{1, \dots, M\}$$
$$\bar{p} = \frac{1}{M} \sum_{m=1}^M \hat{p}_m, \quad \sigma^2_{\text{epistemic}} = \frac{1}{M} \sum_{m=1}^M (\hat{p}_m - \bar{p})^2$$
$$\mathcal{H}(\bar{p}) = -\bar{p} \log_2 \bar{p} - (1 - \bar{p}) \log_2 (1 - \bar{p})$$
where $\sigma^2_{\text{epistemic}}$ measures model lack-of-knowledge (epistemic uncertainty) and $\mathcal{H}$ measures total predictive entropy.
