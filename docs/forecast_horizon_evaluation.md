# Multi-Day Forecast Horizon Evaluation Methodology

**Project Title:** AI-Based Flood Prediction, Dynamic Risk Mapping and Resilience Planning  
**Application:** FloodAI Engine — "AI for a Safer, Resilient India"  
**Module:** Temporal Horizon Degradation & Predictive Lead Analysis  

---

## 1. Operational Rationale for Multi-Horizon Forecasting

Flood hazard early warning systems operate across distinct temporal horizons, each dictating specific operational disaster management protocols:

```
  [ Day T: Today ] ──────────► [ T+1 Day: Tactical ] ──────────► [ T+3 Days: Operational ] ──────────► [ T+7 Days: Strategic ]
  Current Sensor Stack           Nowcasting & Immediate Action      Reservoir Rule-Curve Action        District Resource Staging
  • Trailing Precipitation       • Evacuate low-lying riverbanks    • Controlled dam pre-releases      • Mobilize NDRF battalions
  • Soil Moisture Saturation     • Close bridges and underpasses    • Pre-position rescue boats        • Issue advisories to farmers
  • Satellite SAR & Optical      • Direct traffic from floodplains  • Open emergency relief shelters   • Secure supply chains & fuel
```

Evaluating a model solely at $T+0$ or $T+1$ obscures its degradation dynamics across extended planning windows. The FloodAI Engine evaluates predictive accuracy, discriminability, and calibration across three standardized horizons: **$T+1$ Day**, **$T+3$ Days**, and **$T+7$ Days**.

---

## 2. Mathematical Horizon Formulation

Let $\mathbf{x}_t^{(s)} \in \mathbb{R}^{27}$ represent the standardized feature vector observed at station $s$ up to calendar day $t$.  
For horizon $h \in \{1, 3, 7\}$, the forecasting objective is to estimate the probability that significant inundation occurs on day $t + h$:

$$\hat{p}_{t+h|t}^{(s)} = \mathbb{P}\left(y_{t+h}^{(s)} = 1 \mid \mathbf{x}_t^{(s)}; \boldsymbol{\theta}\right)$$

### Strict Temporal Causality Guarantee
To prevent temporal leakage:
1. Feature extraction $\mathbf{x}_t^{(s)}$ incorporates strictly backward-looking information ($t, t-1, \dots$).
2. The model parameters $\boldsymbol{\theta}$ remain strictly frozen from the 2018–2021 training partition.
3. For each station $s$, input features on day $t$ are aligned against ground truth labels on day $t + h$:
$$\mathcal{D}_{\text{eval}}^{(h)} = \left\{ \left(\mathbf{x}_t^{(s)}, y_{t+h}^{(s)}\right) : t \in [1, N_s - h], \, s \in \mathcal{S}_{\text{test}} \right\}$$
4. Evaluation is performed over the full 2023 out-of-time test partition across all 36 stations.

---

## 3. Physical Degradation Dynamics

As the lookahead window increases from 24 hours to 7 days, model performance faces natural physical constraints:
1. **Atmospheric Determinism Loss:** Precipitation beyond 72 hours is governed by mesoscale convective dynamics and tropical depression trajectories with increasing spatial dispersion.
2. **Soil Infiltration Memory Decay:** While antecedent precipitation index (`api_14d`) and deep soil moisture (`soil_moisture_deep`) exhibit several days of hydraulic memory, surface root-zone moisture rapidly fluctuates due to evapotranspiration and subsurface lateral drainage.
3. **Catchment Routing Lag:** Large river basins (e.g. Ganga, Godavari) possess propagation times of 4–8 days between upper catchment headwaters and downstream delta outlets. Consequently, deep representations capturing upstream antecedent rainfall (`rainfall_7d`, `rainfall_14d`, `river_discharge_proxy`) retain meaningful predictive skill even at $T+7$.

---

## 4. Benchmark Metric Schema

For each horizon $h \in \{1, 3, 7\}$ and model configuration, the evaluation pipeline computes:
- **Precision:** $\frac{\text{TP}}{\text{TP} + \text{FP}}$ (Assesses warning confidence)
- **Recall:** $\frac{\text{TP}}{\text{TP} + \text{FN}}$ (Assesses missed disaster risk)
- **F1 Score:** Harmonic mean of precision and recall
- **ROC-AUC & PR-AUC:** Global discriminability under class imbalance
- **Brier Score:** Mean squared probability calibration error

Empirical results are recorded in `outputs/metrics/forecast_horizon_results.csv` and plotted in `outputs/figures/forecast_horizon_decay.png`.
