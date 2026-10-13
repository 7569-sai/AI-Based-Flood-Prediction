# Flood Ground Truth Labeling Methodology & Operational Limitations

**Project Title:** AI-Based Flood Prediction, Dynamic Risk Mapping and Resilience Planning  
**Application:** FloodAI Engine — "AI for a Safer, Resilient India"  
**Module:** Target Variable Definition & Validation Protocols  

---

## 1. Ground Truth Formulation

In hydrological machine learning, obtaining continuous, high-resolution, pan-India observational ground truth for flood inundation is a recognized scientific bottleneck. Stream gauge networks (Central Water Commission) monitor river stages at discrete cross-sections but do not record overland 2D inundation extents; conversely, polar-orbiting SAR satellites (Sentinel-1) have revisit intervals of 6–12 days and suffer from radar shadow, specular surface confusion in urban environments, and dielectric variation.

To establish a coherent, pan-India benchmark dataset across 36 representative stations from 2018 to 2023 (78,876 daily station records), ground truth flood occurrence ($y \in \{0, 1\}$) and severity ($s \in \{0, 1, 2, 3\}$) are generated through a **Physically Calibrated Multi-Criteria Hydrometeorological Trigger Formulation** coupled with documented historical flood disaster inventories.

---

## 2. Mathematical Definition of Multi-Criteria Trigger

Let station $i$ on day $t$ have:
- 3-day rainfall $R_{3d}(i, t)$ and 7-day rainfall $R_{7d}(i, t)$
- Root-zone soil saturation ratio $S_{\text{sat}}(i, t) \in [0, 1]$
- River stage anomaly $\Delta H_{\text{river}}(i, t)$ (meters relative to bankfull danger mark)
- Calibrated Sentinel-1 SAR VV backscatter $\sigma^0_{\text{vv}}(i, t)$ (dB)
- Surface water proxy index $\text{NDWI}(i, t)$
- Drainage vulnerability $V_{\text{drain}}(i)$

A compound physical flood potential score $\Phi(i, t) \in [0, 1]$ is defined:

$$\Phi(i, t) = w_{\text{rain}} \cdot \psi_R(i, t) + w_{\text{soil}} \cdot S_{\text{sat}}(i, t) + w_{\text{stage}} \cdot \psi_H(i, t) + w_{\text{sat}} \cdot \psi_S(i, t)$$

where:
- $\psi_R(i, t) = \min\left(1.0, \frac{R_{3d}(i, t)}{150.0} + \frac{R_{7d}(i, t)}{300.0}\right)$ (Precipitation accumulation forcing)
- $\psi_H(i, t) = \text{sigmoid}\left(1.8 \cdot (\Delta H_{\text{river}}(i, t) - 0.5)\right)$ (Hydraulic overtopping probability)
- $\psi_S(i, t) = 0.5 \cdot \mathbf{1}_{\{\sigma^0_{\text{vv}} < -17.0\text{ dB}\}} + 0.5 \cdot \mathbf{1}_{\{\text{NDWI} > 0.30\}}$ (Sensor evidence)
- Weights: $w_{\text{rain}} = 0.35, w_{\text{soil}} = 0.25, w_{\text{stage}} = 0.25, w_{\text{sat}} = 0.15$.

### Binary Occurrence Rule
$$y(i, t) = \begin{cases} 1 & \text{if } \Phi(i, t) \ge 0.58 \text{ and } (R_{3d} \ge 65.0 \text{ or } \Delta H_{\text{river}} \ge 0.8) \\ 0 & \text{otherwise} \end{cases}$$

### Ordinal Severity Classification
$$s(i, t) = \begin{cases}
0 & \text{if } y(i, t) = 0 \quad (\text{No Inundation}) \\
1 & \text{if } 0.58 \le \Phi(i, t) < 0.70 \quad (\text{Minor: Shallow waterlogging, localized ponding}) \\
2 & \text{if } 0.70 \le \Phi(i, t) < 0.84 \quad (\text{Moderate: Overbank flooding, agricultural submersion}) \\
3 & \text{if } \Phi(i, t) \ge 0.84 \quad (\text{Severe/Catastrophic: Breached embankments, critical infrastructure inundated})
\end{cases}$$

---

## 3. Historical Event Alignment

The synthetic trigger rule was calibrated against documented historical flood events across India:
1. **Kerala August 2018 Great Floods:** Extreme orographic rainfall over the Western Ghats; station records for Periyar, Chalakkudy, and Pamba basins reflect catastrophic flood triggers ($\Phi > 0.85$, $s=3$).
2. **Delhi Yamuna July 2023 Floods:** Record water level at Old Railway Bridge reaching 208.66 m; stations in Yamuna and Delhi reflect severe overtopping triggers.
3. **Assam Brahmaputra Annual Floods (2019, 2020, 2022, 2023):** Multiple monsoon wave surges across Kaziranga, Guwahati, and Dibrugarh stations with prolonged inundation.
4. **Mahanadi Basin 2020 & Godavari Basin 2022 Surges:** Extensive coastal delta inundation following depression systems in the Bay of Bengal.

---

## 4. Class Imbalance Characteristics

Across the 78,876 daily station records (2018–2023):
- Total Positive Flood Days: ~3,480 (~4.41% positive class prevalence).
- Severity Distribution among Positives:
  - Class 1 (Minor): ~48%
  - Class 2 (Moderate): ~34%
  - Class 3 (Severe): ~18%

This severe class imbalance (~21:1 negative to positive ratio) reflects true physical conditions in Indian catchments where catastrophic flooding is episodic and seasonally bounded to the Southwest and Northeast monsoon windows (June–November).

---

## 5. Operational Limitations & Caveats

1. **Hydrometric Gauge Density:** The proxy discharge and stage anomalies represent idealized catchment responses; localized backwater effects from unmodeled downstream barrages, sluice gates, and tidal surges are not explicitly resolved.
2. **Urban Micro-Drainage:** While population density and elevation risk factors are included, sub-kilometer stormwater network blockages and localized culvert failures require hydraulic hydrodynamic modeling (e.g. SWMM / HEC-RAS 2D).
3. **Operational Recommendation:** When deploying the FloodAI Engine in real-time operational centers (e.g. SDMA, NDMA), ground truth feeds should be ingested via automated API connectors to CWC Telemetric Flood Forecasting Stations and the National Remote Sensing Centre (NRSC) Bhuvan flood inundation maps.
