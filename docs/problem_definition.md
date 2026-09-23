# Problem Definition: Pan-India Multi-Basin Flood Prediction & Dynamic Risk Mapping

**Project Title:** AI-Based Flood Prediction, Dynamic Risk Mapping and Resilience Planning  
**Application:** FloodAI Engine — "AI for a Safer, Resilient India"

---

## 1. Executive Summary & Prediction Scenarios

Floods constitute India's most destructive and economically damaging natural hazard. The South-West and North-East monsoons interact with complex terrain—from the Himalayan headwaters of the Brahmaputra, Ganga, and Yamuna to the peninsular deltas of the Mahanadi, Godavari, Krishna, and Kaveri, and the steep Western Ghats in Kerala—producing catastrophic river breaches, flash floods, and coastal delta backwater stagnation.

### A. Operational Horizon & Dual Framing
1. **24- to 72-Hour Lead Time Hazard Prediction**:
   - Given antecedent multi-day hydrometeorology ($1\text{d}, 3\text{d}, 7\text{d}, 14\text{d}$ cumulative rainfall, API anomaly), soil moisture saturation, geomorphic elevation deficits, and satellite radar backscatter at time $T$, predict flood occurrence and severity across a 24–72 hour forward horizon.
2. **Retrospective Event Classification & Risk Attribution**:
   - For historical event analysis and post-disaster forensics (e.g. 2018 Kerala, 2019 Assam, 2020 Hyderabad, 2022 Silchar, 2023 Delhi Yamuna), classify flood state, attribute risk drivers using TreeSHAP, and evaluate chronic low-flood susceptibility.

---

## 2. Target Variable Definitions

### A. Binary Target (`flood_occurred` $\in \{0, 1\}$)
- **Class 0 (No Flood)**: River stage below warning level; surface runoff, soil saturation, and precipitation within non-disaster baselines.
- **Class 1 (Flood Event)**: Significant river overtopping, bank breach, or extensive surface inundation exceeding local danger levels, validated against official Central Water Commission (CWC) gauging records, India Flood Inventories, and state disaster declarations.

### B. Multi-Tier Flood Severity (`flood_severity` $\in \{0, 1, 2, 3\}$)
- **0 — None**: Normal dry/wet conditions; ample soil infiltration capacity.
- **1 — Minor Waterlogging**: Localized drainage congestion or shallow inundation under isolated heavy downpours.
- **2 — Moderate Riverine Flood**: River swelling near danger marks; low-lying agricultural polders or floodplains submerged.
- **3 — Severe / Catastrophic Deluge**: Major levee breaches, widespread residential inundation, emergency evacuation orders triggered (e.g., Aluva August 2018, Delhi Ring Road July 2023, Silchar June 2022).

---

## 3. Pan-India Geographic Architecture

The platform supports a 7-tier geographic hierarchy:
$$\text{India} \to \text{Region} \to \text{State} \to \text{District / City} \to \text{River Basin} \to \text{Sub-Basin} \to \text{Monitoring Station / Grid}$$

### Geographic Coverage (36 National Stations across 12 River Basins):
- **Northeast India**: Brahmaputra and Barak basins (Guwahati, Dibrugarh, Silchar, Kaziranga).
- **North India**: Yamuna and Upper Ganga basins (Delhi ITO, Mathura, Haridwar, Kullu/Mandi Beas, Srinagar Jhelum, Firozpur Sutlej).
- **East India**: Middle/Lower Ganga, Kosi, Damodar, Subernarekha, Mahanadi (Patna, Bhagalpur, Saharsa Kosi, Cuttack Delta, Sambalpur Hirakud, Asansol, Jamshedpur, Kolkata Hooghly).
- **Central India**: Narmada and Tapi basins (Hoshangabad, Jabalpur, Bharuch, Burhanpur, Surat).
- **West India**: Mumbai coastal & Mithi river, Sangli/Kolhapur Panchaganga.
- **South India**: Godavari, Krishna, Kaveri, Periyar/Kuttanad, Netravati (Bhadrachalam, Rajahmundry, Vijayawada, Tiruchirappalli, Thanjavur, Aluva, Kuttanad, Chalakkudy, Mangalore).

---

## 4. Separation of Scientific Modules

To maintain scientific integrity and prevent conceptual conflation:
1. **Module A (Flood Prediction)**: Evaluates physical hydrometeorological hazard $H \in [0, 1]$ using baseline models and the proposed ST-FloodNet architecture.
2. **Module B (Dynamic Risk Mapping)**: Computes exposure $E$ from population density and vulnerability $V$ from topographic drainage deficits, synthesizing a 0–100 dynamic risk score:
   $$\text{Risk} = [0.50 H + 0.25 E + 0.25 V] \times 100$$
3. **Module C (Resilience Planning)**: Prescriptive decision support generating agency-tagged emergency, structural, policy, and agricultural action plans.
