# Disaster Risk Methodology: Hazard, Exposure, Vulnerability & Risk

This document details the mathematical, physical, and geospatial methodology implemented in the AI-Based Flood Prediction and Resilience Planning system.

---

## 1. Conceptual Framework (UNDRR / IPCC Standard)

We adopt the United Nations Office for Disaster Risk Reduction (UNDRR) and Intergovernmental Panel on Climate Change (IPCC AR5/AR6) tripartite disaster risk formulation:

$$\text{Risk} = \text{Hazard} \times \text{Exposure} \times \text{Vulnerability}$$

To ensure scientific defensibility, each component is strictly bounded and computed independently before integration.

```
┌─────────────────────────────────┐
│     HAZARD (H ∈ [0, 1])         │
│  Model Inundation Probability   │
│  (Rainfall, Soil Saturation,    │
│   Antecedent Memory, Runoff)    │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐      ┌─────────────────────────────────┐
│    EXPOSURE (E ∈ [0, 1])        │      │   VULNERABILITY (V ∈ [0, 1])    │
│  Human Lives & Physical Assets  │  +   │   Physical & Drainage Bottleneck│
│  (Population Density, Settled   │      │   (Elevation Deficit, Slope,    │
│   Area, Key Corridors)          │      │    River Proximity, Infiltration)
└────────────────┬────────────────┘      └────────────────┬────────────────┘
                 │                                        │
                 └───────────────────┬────────────────────┘
                                     │
                                     ▼
                     ┌───────────────────────────────┐
                     │    COMPOSITE FLOOD RISK (R)   │
                     │    Project-Defined 0–100      │
                     └───────────────────────────────┘
```

---

## 2. Mathematical Formulations

### A. Flood Hazard ($H \in [0, 1]$)
Hazard represents the intrinsic physical likelihood of flood waters inundating a specific spatial cell or monitoring station. It is computed directly as the calibrated output probability from the Random Forest model:

$$H = P(\text{Flood} \mid X_{\text{hydro}}, X_{\text{terrain}}, X_{\text{remote-sensing}})$$

### B. Exposure Index ($E \in [0, 1]$)
Exposure quantifies the presence of people, settlements, and infrastructure situated in flood-prone zones. Using gridded population density ($\text{Pop}$, persons/$km^2$):

$$E = \min\left(1.0, \frac{\ln(1 + \text{Pop})}{\ln(1 + \text{Pop}_{\max})}\right)$$

Where $\text{Pop}_{\max} = 5,000\text{ ppl/km}^2$ reflects the high-density urban core ceiling (e.g. coastal Kochi / Thiruvananthapuram).

### C. Vulnerability Index ($V \in [0, 1]$)
Vulnerability reflects physical susceptibility to drainage failure, topography-induced water ponding, and proximity to river overtopping:

$$V = w_1 \cdot V_{\text{elev}} + w_2 \cdot V_{\text{river}} + w_3 \cdot V_{\text{slope}}$$

Where:
* $V_{\text{elev}} = \exp(-\text{elevation} / 50.0) \in [0, 1]$ (exponential susceptibility in lowlands $< 50m$)
* $V_{\text{river}} = \frac{1}{1 + d_{\text{river}}} \in [0, 1]$ (hyperbolic decay with distance to river in km)
* $V_{\text{slope}} = \frac{1}{1 + \text{slope}_{\text{deg}}} \in [0, 1]$ (flat terrain impairs gravitational runoff)
* Default weights: $w_1 = 0.45, w_2 = 0.35, w_3 = 0.20$ ($\sum w_i = 1.0$).

### D. Composite Flood Risk Score ($R \in [0, 100]$)
The final risk score combines hazard, exposure, and vulnerability into an intuitive 0–100 scale:

$$R = \left[ 0.50 \cdot H + 0.25 \cdot E + 0.25 \cdot V \right] \times 100$$

Where:
* Hazard carries the dominant weighting ($0.50$) because without meteorological/hydrologic hazard, exposure alone does not produce flood disaster.
* Exposure ($0.25$) and Vulnerability ($0.25$) modulate the impact severity.

---

## 3. Project-Defined Risk Categories

> [!IMPORTANT]
> The thresholds below are **PROJECT-DEFINED** for academic and planning prototyping, and do not constitute statutory government warning thresholds.

| Score Range | Project-Defined Category | Operational Interpretation |
| :---: | :---: | :--- |
| **0 – 20** | **Very Low Risk** | Normal dry or light seasonal conditions; zero emergency intervention required. |
| **21 – 40** | **Low Risk** | Moderate waterlogging possible in poor drains; routine monitoring. |
| **41 – 60** | **Moderate Risk** | Rising saturation, localized lowland inundation possible; advisory alerts. |
| **61 – 80** | **High Risk** | Substantial flood threat, low-lying evacuation preparedness, shelter readiness. |
| **81 – 100** | **Very High Risk** | Critical catastrophic inundation hazard, mandatory evacuation, emergency staging. |
