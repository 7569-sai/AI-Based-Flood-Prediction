# Problem Definition: AI-Based Flood Prediction and Resilience Planning

## 1. Executive Summary & Prediction Scenario

Floods constitute the most destructive hydrometeorological disaster globally and across India, inflicting catastrophic human casualties, displacement, and economic losses. In Kerala, the South-West and North-East monsoons interact with the steep topography of the Western Ghats to generate rapid runoff, sudden river stage surges, and widespread inundation across coastal and deltaic plains.

### Primary Prediction Scenario
* **Classification Horizon**: **24- to 72-Hour Flood Hazard Prediction**.
* **Framing**: Given antecedent hydrometeorological accumulations (1-day, 3-day, 7-day, 14-day rainfall), root-zone and deep soil moisture saturation, geomorphological constraints (elevation, slope, river proximity), and remote sensing baseline indices, predict the occurrence and severity of flood hazard at a designated monitoring location.
* **Scientific Justification**:
  * Antecedent precipitation and soil moisture provide genuine multi-day memory in catchment hydrologic response. A 100mm deluge onto dry, unsaturated soil rarely causes catastrophic flooding, whereas a 60mm rainfall event onto fully saturated soil with full reservoir catchments precipitates catastrophic overflow (as observed in the August 2018 Kerala floods).
  * The dataset genuinely supports this multi-day lead prediction because antecedent features are computed strictly from historical windows preceding the target day $T$.

---

## 2. Target Variable Formulation

The target variable is formulated using documented official disaster response records from the Central Water Commission (CWC), Kerala State Disaster Management Authority (KSDMA), and physical hydrometeorological thresholds:

### A. Binary Target (`flood_occurred` $\in \{0, 1\}$)
* **Class 0 (No Flood)**: River stage below warning level, surface runoff and soil saturation within normal seasonal baselines.
* **Class 1 (Flood Event)**: Significant river overtopping, localized or basin-wide inundation, corresponding to official emergency declaration periods (e.g. August 8–20, 2018; August 8–14, 2019; October 16–22, 2021) or composite extreme hydrologic saturation:
  $$\text{3-Day Rainfall} \ge 120\text{ mm} \quad \text{AND} \quad \text{Soil Moisture (0–28cm)} \ge 0.41\text{ m}^3/\text{m}^3 \quad \text{AND} \quad \text{Elevation} \le 30\text{ m}$$
  OR extreme cloudburst runoff exceeding 99th percentile station historical threshold.

### B. Multi-Tier Flood Hazard Level (`hazard_level` $\in \{0, 1, 2, 3\}$)
To support granular risk mapping and operational disaster response, a 4-tier hazard classification is defined:
* **0 — Low / No Hazard**: Dry to moderate conditions, soil capacity available ($P(\text{Flood}) < 0.25$).
* **1 — Moderate Hazard**: Sustained moderate rainfall, rising soil moisture, potential waterlogging in low-lying depressions ($0.25 \le P < 0.50$).
* **2 — High Hazard**: Intense antecedent accumulation, near-saturated soil profile, river approaching danger mark ($0.50 \le P < 0.75$).
* **3 — Very High Hazard**: Severe deluge, complete profile saturation, river stage exceeding danger mark, widespread inundation and structural threat ($P \ge 0.75$).

---

## 3. Study Area Rationale: Kerala River Basins

Kerala was selected as the primary study area due to:
1. **Severe Historical Precedence**: The August 2018 disaster represented a 1-in-100-year return period flood affecting over 5.4 million people across 14 districts, followed by severe recurrence in 2019 and 2021.
2. **Topographic Heterogeneity**: Within a narrow 120 km cross-section, elevation plunges from over 2,600m (Anamudi / Western Ghats) to -1m below sea level (Kuttanad polders), creating rapid time-of-concentration river basins (Periyar, Chalakkudy, Pamba, Bharathapuzha).
3. **Data Availability**: High density of meteorological reanalysis, dual-polarization Sentinel-1 SAR observations, and validated CWC hydrological gauging stations.

---

## 4. Distinction Between Modules

To prevent scientific conflation:
* **Hazard (Module A)**: The intrinsic physical probability and intensity of flooding driven strictly by meteorological and hydrological processes ($H \in [0, 1]$).
* **Exposure (Module B)**: Assets, human lives, and infrastructure located in the hazard zone ($E \in [0, 1]$).
* **Vulnerability (Module B)**: Susceptibility and lack of coping capacity ($V \in [0, 1]$).
* **Risk (Module B)**: The intersection $R = f(H, E, V)$.
* **Resilience Planning (Module C)**: Prescriptive, rule-based mitigation protocols informed by the risk components.
