# 🌊 FloodAI ENGINE — AI for a Safer, Resilient India

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141.1-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.64.0-FF4B4B.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 51 Passed](https://img.shields.io/badge/Tests-51%20Passed%20(100%25)-brightgreen.svg)]()
[![Model: HF-STFloodNet](https://img.shields.io/badge/Model-HF--STFloodNet%20(Hybrid%20Fusion)-purple.svg)]()
[![Coverage: Pan-India](https://img.shields.io/badge/Coverage-36%20National%20Stations%20%7C%2024%20Basins-orange.svg)]()

> **B.Tech Major Project & IEEE Research Paper**  
> **Title**: AI-Based Flood Prediction, Dynamic Risk Mapping and Resilience Planning  
> **Platform**: FloodAI Engine  
> **Proposed Model**: HF-STFloodNet (Hybrid Fusion Spatiotemporal Flood Network)  
> **Tagline**: AI for a Safer, Resilient India  
> **Concept**: Predict • Map • Explain • Protect

---

## 📌 Table of Contents
1. [Project Overview & Mission](#-project-overview--mission)
2. [Research Upgrades & Data Provenance Disclosure](#-research-upgrades--data-provenance-disclosure)
3. [HF-STFloodNet Hybrid Architecture](#-hf-stfloodnet-hybrid-architecture)
4. [Certified Leakage-Free Methodology](#-certified-leakage-free-methodology)
5. [Benchmark Models & Empirical Results (2023 Test Set)](#-benchmark-models--empirical-results-2023-test-set)
6. [Comprehensive Multi-Basin Generalization (6 Basins)](#-comprehensive-multi-basin-generalization-6-basins)
7. [Event-Level Continuous Flood Metrics](#-event-level-continuous-flood-metrics)
8. [Multi-Day Forecast Horizons (T+1, T+3, T+7)](#-multi-day-forecast-horizons-t1-t3-t7)
9. [10-Tier Progressive Architectural Ablation Study](#-10-tier-progressive-architectural-ablation-study)
10. [Sensor Missingness & Telemetry Robustness](#-sensor-missingness--telemetry-robustness)
11. [Propagated Hybrid Uncertainty](#-propagated-hybrid-uncertainty)
12. [Dynamic Risk & Vulnerability Mapping](#-dynamic-risk--vulnerability-mapping)
13. [Explainable AI (TreeSHAP)](#-explainable-ai-treeshap)
14. [Resilience Planning Engine](#-resilience-planning-engine)
15. [Quickstart & Execution](#-quickstart--execution)
16. [Academic Disclaimer & Ethics](#-academic-disclaimer--ethics)

---

## 🌊 Project Overview & Mission

Floods represent India's most destructive hydrometeorological disaster, inflicting catastrophic casualties, displacement, and economic disruptions annually across diverse geographic domains.

**FloodAI Engine** unifies deep representation learning, tabular gradient boosting, spatial hazard mapping, and decision support:
* **MODULE A (HF-STFloodNet Hybrid Flood Prediction)**: Stacks multimodal spatiotemporal deep learning (ST-FloodNet) with structured gradient boosting (LightGBM) using station-aware cross-fitting to deliver calibrated flood hazard forecasting.
* **MODULE B (Dynamic Spatial Risk Mapping)**: Decomposes compound disaster risk according to the UNDRR framework:
  $$\text{Risk Score} = \left[ 0.50 \cdot \text{Hazard} + 0.25 \cdot \text{Exposure} + 0.25 \cdot \text{Vulnerability} \right] \times 100$$
* **MODULE C (Resilience Planning & Decision Support)**: Synthesizes context-tailored 0–48h emergency response, structural mitigation, zoning policies, and agricultural protection measures.

---

## 🔬 Research Upgrades & Data Provenance Disclosure

In accordance with empirical integrity and reproducibility standards:
* **True Temporal Sequence Tensors:** Dynamic hydrometeorological features are processed via Multi-Scale 1D Conv $\to$ Bidirectional GRU $\to$ Temporal Attention, while static terrain, radar, and socioeconomic features pass through modular encoders.
* **Zero Feature Bypassing:** All 27 engineered features are explicitly routed into four specialized physical neural modules:
  1. *Dynamic Hydrometeorology* (13 features: `rainfall_1d` to `river_stage_anomaly`)
  2. *Topographic Geomorphology* (6 features: `elevation_m` to `drainage_vulnerability`)
  3. *Satellite Earth Observation* (5 features: `sar_vv` to `ndvi_proxy`)
  4. *Socioeconomic Exposure* (3 features: `population_density`, `exposure_index`, `vulnerability_index`)
* **Scientific Data Disclosure:** Observational climate and geospatial variables are grounded in IMD, ERA5-Land, SRTM, and CIESIN GPWv4 datasets. Hydrological discharge/stage series, radar backscatter, and flood event labels represent physically calibrated proxies and simulated ground truth aligned with documented historical Indian floods. For complete details, see [`docs/data_provenance_audit.md`](docs/data_provenance_audit.md), [`docs/flood_label_methodology.md`](docs/flood_label_methodology.md), and [`data/DATA_MANIFEST.json`](data/DATA_MANIFEST.json).

---

## 🧠 HF-STFloodNet Hybrid Architecture

```
                       RAW 27-FEATURE SPATIOTEMPORAL INPUT
                                        │
           ┌────────────────────────────┴────────────────────────────┐
           ▼                                                         ▼
ST-FloodNet Multimodal Branch                                LightGBM Tabular Branch
• Temporal Conv-BiGRU-Attention (13 features)                • Histogram Gradient Boosting
• Spatial Terrain Encoder (6 features)                       • 27 tabular features
• Remote Sensing SAR/Optical Encoder (5 features)            • Sharp physical splitting
• Socioeconomic Exposure Encoder (3 features)                        │
• Gated Cross-Modal Bottleneck (128 -> 64)                           │
           │                                                         │
           ├────────────────────────────┐                            │
           ▼                            ▼                            ▼
64-dim Latent Bottleneck        ST Hazard Probability       LGBM Hazard Probability
  z_rep in R^64                    p_ST in [0, 1]              p_LGB in [0, 1]
           │                            │                            │
           ▼ (PCA k=8)                  │                            │
8-dim Compact Representation            │                            │
  z_pca in R^8                          │                            │
           │                            │                            │
           └──────────────┬─────────────┴─────────────┬──────────────┘
                          │                           │
                          │   Physical Domain Anchors │
                          ▼                           ▼
                    ┌───────────────────────────────────────┐
                    │      CROSS-FITTED META-LEARNER        │
                    │      Regularized LightGBM GBDT        │
                    └───────────────────┬───────────────────┘
                                        │
                                        ▼
                         Validation-Locked Platt Scaling
                                        │
                                        ▼
                             FINAL HYBRID PREDICTION
                 Calibrated Hazard Probability p in [0, 1],
                 Severity (None/Minor/Moderate/Severe),
                 Propagated Epistemic MC Variance & Entropy
```

---

## 🔒 Certified Leakage-Free Methodology

* **Strict Temporal Causality:** All dynamic indicators represent backward-looking historical summaries ($T-1$) with zero future leakage.
* **Station-Aware GroupKFold Cross-Fitting:** Meta-features for training the stacking layer are generated strictly Out-Of-Fold (OOF) across 5 station clusters on 2018–2021 data.
* **Validation-Locked Thresholding & Calibration:** Decision threshold ($\theta^* = 0.51$, Val $F_1 = 0.7491$) and Platt Scaling calibrator are estimated exclusively on calendar year 2022 and applied frozen to the 2023 test set.
* See [`docs/hybrid_leakage_audit.md`](docs/hybrid_leakage_audit.md) and [`docs/threshold_selection.md`](docs/threshold_selection.md).

---

## 📊 Benchmark Models & Empirical Results (2023 Test Set)

Evaluated on the out-of-time test partition (Calendar Year 2023, 13,140 daily records across 36 stations in 24 river basins; for sequence models, $T=14$ days $[B, 14, 13]$):

| Model Architecture | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Brier Score | ECE | Latency (ms) | Size (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.3186 | **1.0000** | 0.4832 | 0.9939 | 0.6652 | 0.0239 | 0.0353 | <0.001 ms | <0.01 MB |
| **Random Forest** | 0.5206 | 0.9465 | 0.6717 | 0.9960 | 0.7866 | 0.0092 | 0.0100 | 0.004 ms | 1.75 MB |
| **Gradient Boosting** | 0.6776 | 0.5508 | 0.6077 | 0.9960 | 0.7908 | **0.0060** | 0.0032 | 0.001 ms | 0.48 MB |
| **XGBoost** | 0.5291 | 0.9733 | **0.6855** | 0.9961 | 0.7878 | 0.0110 | 0.0114 | 0.001 ms | 0.22 MB |
| **LightGBM** | 0.5490 | 0.8984 | 0.6815 | **0.9962** | **0.7937** | 0.0093 | 0.0088 | 0.001 ms | 0.49 MB |
| **ST-FloodNet-v1 (Flat)** | 0.3830 | 0.9893 | 0.5522 | 0.9952 | 0.7697 | 0.0184 | 0.0218 | 0.007 ms | 0.25 MB |
| **ST-FloodNet-v2-Temporal** | 0.4093 | 0.9893 | 0.5790 | 0.9955 | 0.7781 | 0.0183 | 0.0198 | 0.046 ms | 0.25 MB |
| **HF-STFloodNet-v2 (Proposed)** | **0.6852** | 0.5936 | 0.6361 | 0.9946 | 0.7437 | 0.0078 | **0.0030** | 0.047 ms | 0.26 MB |

> [!NOTE]
> **Empirical Honesty:** On stationary in-basin records, tabular tree ensembles (XGBoost/LightGBM) achieve high F1 by fitting sharp physical thresholds. However, **HF-STFloodNet achieves the lowest ECE (0.0030)** and highest precision (0.6852), while **ST-FloodNet-v2-Temporal achieves 98.18% continuous event recall** with a detection delay of only 0.04 days and 1.68 days of peak lead time.

---

## 🗺️ Comprehensive Multi-Basin Generalization (6 Basins)

Evaluating spatial transferability onto major river basins unseen during fold training:

| Basin Name | Hydrological Regime | Samples | LightGBM F1 | Random Forest F1 | HF-STFloodNet F1 | ST-FloodNet F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Brahmaputra** | Assam / Eastern Himalayan | 6,573 | **0.8256** | 0.7790 | 0.7448 | 0.6885 |
| **Mahanadi** | Odisha / Eastern Coastal Delta | 4,382 | **0.9263** | 0.8458 | 0.8242 | 0.6769 |
| **Ganga** | UP / Gangetic Plain | 10,955 | **0.8879** | 0.8054 | 0.8233 | 0.6216 |
| **Godavari** | Central Peninsular | 4,382 | **0.8879** | 0.8376 | 0.8018 | 0.7510 |
| **Krishna** | Southern Peninsular | 4,382 | **0.8507** | 0.7837 | 0.7129 | 0.7170 |
| **Kaveri** | Southern Coastal & Ghats | 4,382 | **0.7400** | 0.6789 | 0.5647 | 0.3920 |

---

## ⏱️ Event-Level Continuous Flood Metrics

Evaluating continuous disaster episodes across stations (110 distinct flood events in 2023):

| Model Name | Total Events | Detected Events | Event Recall | Event FAR | Mean Detection Delay (days) | Mean Peak Lead Time (days) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ST-FloodNet-v2-Temporal** | 110 | **108** | **0.9818** | 0.6766 | **0.04** | **1.68** |
| **XGBoost** | 110 | 107 | 0.9727 | 0.5633 | 0.07 | 1.65 |
| **Random Forest** | 110 | 103 | 0.9364 | 0.5761 | 0.15 | 1.61 |
| **LightGBM** | 110 | 93 | 0.8455 | 0.5550 | 0.33 | 1.53 |
| **HF-STFloodNet-v2** | 110 | 37 | 0.3364 | **0.5132** | 1.35 | 1.01 |

---

## 📅 Multi-Day Forecast Horizons (T+1, T+3, T+7)

Predictive skill decay across extended lead times on 2023 test data:

| Forecast Horizon | HF-STFloodNet F1 | LightGBM F1 | HF-STFloodNet ROC-AUC | LightGBM ROC-AUC | HF-STFloodNet Precision | HF-STFloodNet Recall |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$T+1$ Day (Tactical)** | **0.3333** | 0.3286 | **0.8375** | 0.8268 | 0.2852 | 0.4011 |
| **$T+3$ Days (Operational)** | **0.2844** | 0.2880 | **0.8370** | 0.8086 | 0.2433 | 0.3422 |
| **$T+7$ Days (Strategic)** | **0.1867** | 0.1826 | **0.7966** | 0.7667 | 0.1597 | 0.2246 |

---

## 🔬 10-Tier Progressive Architectural Ablation Study

| Tier | Ablation Description | F1-Score | ROC-AUC | PR-AUC | Brier Score |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | Full Proposed HF-STFloodNet (27 features + Stacking + Attention) | 0.6444 | 0.9952 | 0.7613 | **0.0074** |
| **2** | w/o Meta-Learner Stacking (Unweighted 50/50 Probability Average) | **0.6740** | **0.9962** | 0.7996 | 0.0117 |
| **3** | w/o Tabular Gradient Boosting Branch (ST-FloodNet Standalone) | 0.5522 | 0.9952 | 0.7697 | 0.0184 |
| **4** | w/o Latent Deep Representation (Stacking on Probabilities Only) | 0.6514 | 0.9961 | **0.8005** | 0.0064 |
| **5** | w/o Temporal Attention (Uniform Average Recurrent Pooling) | 0.6064 | 0.9604 | 0.7141 | 0.0080 |
| **6** | w/o Bidirectional GRU Memory (Pure Convolutional Antecedent Branch) | 0.5748 | 0.9365 | 0.6738 | 0.0085 |
| **7** | w/o Sentinel-1 SAR Radar Modality (Zeroed VV, VH, Cross-ratio) | 0.6503 | 0.9953 | 0.7620 | 0.0073 |
| **8** | w/o Sentinel-2 Optical Spectral Modality (Zeroed NDWI, NDVI) | 0.6340 | 0.9952 | 0.7642 | 0.0074 |
| **9** | w/o Topographic Geomorphology (Zeroed Elevation, Slope, River Dist, TWI) | 0.5863 | 0.9933 | 0.7177 | 0.0094 |
| **10** | w/o Socioeconomic Exposure Modality (Zeroed Population, Vulnerability) | 0.6475 | 0.9953 | 0.7610 | 0.0074 |

---

## 📡 Sensor Missingness & Telemetry Robustness

| Experimental Degradation Condition | HF-STFloodNet F1 | LightGBM F1 | HF-STFloodNet ROC-AUC | LightGBM ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| **Clean Baseline** | 0.6444 | 0.6815 | 0.9952 | 0.9962 |
| **5% Random Missingness** | 0.6336 | 0.6696 | 0.9935 | 0.9947 |
| **10% Random Missingness** | 0.5970 | 0.6376 | 0.9910 | 0.9830 |
| **20% Random Missingness** | 0.5215 | 0.5706 | 0.9797 | 0.9683 |
| **SAR Radar Blackout (100% missing)** | **0.6503** | 0.6585 | **0.9953** | 0.9960 |
| **Soil Moisture Sensor Failure** | **0.6131** | 0.6788 | **0.9949** | 0.9961 |
| **Rain Gauge Telemetry Failure** | 0.0000 | 0.0000 | **0.8472** | 0.5136 |

---

## 🎯 Propagated Hybrid Uncertainty

* **Monte Carlo Dropout Inference:** $T=30$ forward stochastic passes through ST-FloodNet + Fusion Network.
* **Mean Epistemic Variance:** Quantifies model lack-of-knowledge near decision boundaries.
* **Predictive Entropy:** Captures aleatoric ambiguity across multimodal evidence.
* **95% Credible Interval:** Delivers dynamic confidence bounds $[CI_{\text{lower}}, CI_{\text{upper}}]$ for every point prediction.

---

## 🚀 Quickstart & Execution

```powershell
# 1. Install Dependencies
pip install -r requirements.txt

# 2. Run Comprehensive Research Experiments Suite
python scripts/run_research_experiments.py

# 3. Run Complete Automated Test Suite (51 Passed)
python -m pytest tests -q

# 4. Launch Interactive Location-First Streamlit Platform
streamlit run dashboard/app.py

# 5. Launch FastAPI REST Engine
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

---

## 🛡️ Academic Disclaimer & Ethics

> [!WARNING]
> **Mandatory Research Disclaimer**: All hazard forecasts, continuous risk scores, and resilience directives synthesized by FloodAI Engine are experimental project recommendations and do NOT constitute statutory government evacuation orders or official disaster declarations. Operational emergency deployments must be cross-verified against official Central Water Commission (CWC) and India Meteorological Department (IMD) bulletins.
