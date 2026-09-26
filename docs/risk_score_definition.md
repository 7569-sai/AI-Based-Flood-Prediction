# Project-Defined Composite Risk Score Definition & Scale Specification

**Project Title:** AI-Based Flood Prediction, Dynamic Risk Mapping and Resilience Planning  
**Document:** Risk Score Scale, Mathematical Formulation & Threshold Specification  
**Version:** 3.0 (Pan-India Multi-Basin Standard)  

---

## 1. Mathematical Formulation

FloodAI Engine implements the United Nations Office for Disaster Risk Reduction (UNDRR) and IPCC AR5/AR6 tripartite disaster risk formulation:

$$\text{Risk} = \text{Hazard} \times \text{Exposure} \times \text{Vulnerability}$$

To allow intuitive operational decision support, each component is strictly bounded on the unit interval $[0, 1]$ before linear synthesis into a **0–100 continuous composite scale**:

### A. Normalized Tripartite Components (Scale: $0.0$ to $1.0$)
1. **Hazard Probability ($H \in [0, 1]$)**:
   - Derived directly from the calibrated champion ensemble model inference:
     $$H = P(\text{Flood} \mid \mathbf{X}_{27})$$
   - Weight: **50% ($0.50$)** because without meteorological and hydrological water surplus, exposure alone does not generate flood inundation.

2. **Demographic Exposure Index ($E \in [0, 1]$)**:
   - Log-scaled human population exposure index derived from CIESIN GPWv4 / Census grids:
     $$E = \text{clip}\left(\frac{\ln(1 + \text{Population Density})}{\ln(1 + 5000)}, 0.0, 1.0\right)$$
   - Weight: **25% ($0.25$)**.

3. **Physical Vulnerability Index ($V \in [0, 1]$)**:
   - Topographic, geomorphic, and drainage bottleneck composite:
     $$V = 0.45 \cdot e^{-\text{elevation}/50} + 0.35 \cdot \frac{1}{1 + d_{\text{river}}} + 0.20 \cdot \frac{1}{1 + \text{radians}(\text{slope})}$$
   - Weight: **25% ($0.25$)**.

---

## 2. Composite Risk Score Scale (Scale: $0$ to $100$)

The composite risk score $R$ is defined mathematically as:

$$R = \left[ 0.50 \cdot H + 0.25 \cdot E + 0.25 \cdot V \right] \times 100.0$$

- **Scale Range:** $0.00$ to $100.00$
- **Units:** Dimensionless composite index (reported as $R / 100$)
- **Data Type:** Float rounded to 2 decimal places (e.g. `36.64 / 100` or `64.00 / 100`)

> [!IMPORTANT]
> The unscaled combination $[0.50H + 0.25E + 0.25V]$ yields a number in $[0.0, 1.0]$ (e.g. $0.3664$ or $0.6400$).  
> The **Composite Risk Score** is strictly multiplied by $100.0$ to yield $36.64 / 100$ or $64.00 / 100$.  
> The system UI must NEVER display an unscaled unit-interval value like `0.64 / 100`.

---

## 3. Project-Defined Risk Category Thresholds

The continuous risk score $R \in [0, 100]$ maps to categorical risk levels using the project-documented thresholds:

| Score Range ($R$) | Risk Category | Color Code | Operational Meaning |
| :---: | :---: | :---: | :--- |
| **$0.0 \le R \le 20.0$** | **Very Low** | Green (`#20A464` / `#2ca02c`) | Normal dry/baseline conditions; routine observation. |
| **$20.0 < R \le 40.0$** | **Low** | Light Green (`#22B8CF` / `#98df8a`) | Minor localized runoff potential; advisory monitoring. |
| **$40.0 < R \le 60.0$** | **Moderate** | Amber (`#F4A261` / `#ffbb78`) | Soil saturation high; localized inundation in lowlands. |
| **$60.0 < R \le 80.0$** | **High** | Orange (`#F97316` / `#ff7f0e`) | Significant inundation hazard; shelter & rescue staging. |
| **$80.0 < R \le 100.0$** | **Very High / Critical** | Red (`#EF4444` / `#d62728`) | Severe catastrophic inundation threat; emergency response. |

---

## 4. Verification Check & Defensive UI Safeguard

The system software enforces the following verification in `PredictionService` and `LocalRiskAnalyzer`:
```python
# Guaranteed 0-100 scale contract
assert 0.0 <= risk_score <= 100.0, f"Risk score {risk_score} out of bounds!"

# Dynamic category mapping strictly adheres to:
def get_risk_category(risk_score):
    if risk_score <= 20.0:
        return "Very Low", "#20A464"
    elif risk_score <= 40.0:
        return "Low", "#22B8CF"
    elif risk_score <= 60.0:
        return "Moderate", "#F4A261"
    elif risk_score <= 80.0:
        return "High", "#F97316"
    else:
        return "Very High", "#EF4444"
```
