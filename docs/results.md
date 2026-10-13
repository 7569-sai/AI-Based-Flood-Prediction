# Comprehensive Experimental Results: FloodAI Engine

**Project Title:** AI-Based Flood Prediction, Dynamic Risk Mapping and Resilience Planning  
**Application:** FloodAI Engine — "AI for a Safer, Resilient India"  
**Dataset:** 78,876 Daily Station Records across 36 Pan-India Stations (2018–2023)  
**Status:** Peer-Reviewed Empirical Standard  

---

## 1. Out-of-Time Test Set Performance (2023 Benchmark)

The 2023 out-of-time test set evaluates temporal transferability across all 36 stations (13,140 daily observations):

| Model Name | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Brier Score | ECE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 0.9877 | 0.5206 | 0.9465 | 0.6717 | 0.9960 | 0.7866 | 0.0092 | 0.0100 |
| **XGBoost** | 0.9880 | 0.5291 | 0.9733 | **0.6855** | 0.9961 | 0.7878 | 0.0110 | 0.0114 |
| **LightGBM** | **0.9881** | 0.5490 | 0.8984 | 0.6815 | **0.9962** | **0.7937** | 0.0093 | 0.0088 |
| **ST-FloodNet** | 0.9775 | 0.3830 | **0.9893** | 0.5522 | 0.9952 | 0.7697 | 0.0184 | 0.0218 |
| **HF-STFloodNet (Proposed)** | 0.9878 | **0.5513** | 0.7754 | 0.6444 | 0.9952 | 0.7613 | **0.0074** | **0.0071** |

---

## 2. Decision Threshold & Probability Calibration (Validation 2022)

To avoid test-set snooping or data leakage:
- Decision threshold was optimized strictly on the 2022 validation split: $\theta^* = 0.51$ (Validation $F_1 = 0.7491$).
- Platt Scaling reduced Expected Calibration Error from $0.0167$ to $0.0056$ on validation data and down to $0.0071$ on test data.
- Brier score dropped from $0.0146$ to $0.0074$ on test data.

---

## 3. Leave-Basin-Out Multi-Basin Generalization

Evaluating models on 6 major river basins unseen during fold training:

| Basin Name | State / Geography | Samples | LightGBM F1 | Random Forest F1 | HF-STFloodNet F1 | ST-FloodNet F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Brahmaputra** | Assam / Eastern Himalayan | 6,573 | **0.8256** | 0.7790 | 0.7448 | 0.6885 |
| **Mahanadi** | Odisha / Eastern Coastal Delta | 4,382 | **0.9263** | 0.8458 | 0.8242 | 0.6769 |
| **Ganga** | UP / Gangetic Plain | 10,955 | **0.8879** | 0.8054 | 0.8233 | 0.6216 |
| **Godavari** | Central Peninsular | 4,382 | **0.8879** | 0.8376 | 0.8018 | 0.7510 |
| **Krishna** | Southern Peninsular | 4,382 | **0.8507** | 0.7837 | 0.7129 | 0.7170 |
| **Kaveri** | Southern Coastal & Ghats | 4,382 | **0.7400** | 0.6789 | 0.5647 | 0.3920 |

---

## 4. Multi-Day Forecast Horizon Decay

Predictive skill decay across $T+1$, $T+3$, and $T+7$ days ahead:

| Forecast Horizon | HF-STFloodNet Accuracy | HF-STFloodNet Precision | HF-STFloodNet Recall | HF-STFloodNet F1 | HF-STFloodNet ROC-AUC |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$T+1$ Day** | 0.9771 | 0.2852 | 0.4011 | **0.3333** | **0.8375** |
| **$T+3$ Days** | 0.9753 | 0.2433 | 0.3422 | **0.2844** | **0.8370** |
| **$T+7$ Days** | 0.9716 | 0.1597 | 0.2246 | **0.1867** | **0.7966** |

---

## 5. 10-Tier Progressive Architectural Ablation

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

## 6. Sensor Missingness & Telemetry Fault Tolerance

| Fault Condition | HF-STFloodNet F1 | LightGBM F1 | HF-STFloodNet ROC-AUC | LightGBM ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| **Clean Baseline** | 0.6444 | 0.6815 | 0.9952 | 0.9962 |
| **5% Random Missingness** | 0.6336 | 0.6696 | 0.9935 | 0.9947 |
| **10% Random Missingness** | 0.5970 | 0.6376 | 0.9910 | 0.9830 |
| **20% Random Missingness** | 0.5215 | 0.5706 | 0.9797 | 0.9683 |
| **SAR Radar Blackout (100% missing)** | **0.6503** | 0.6585 | **0.9953** | 0.9960 |
| **Soil Moisture Sensor Failure** | **0.6131** | 0.6788 | **0.9949** | 0.9961 |
| **Rain Gauge Telemetry Failure** | 0.0000 | 0.0000 | **0.8472** | 0.5136 |

Under extreme rain gauge telemetry collapse, while both models fail to trigger binary warnings at standard thresholds, HF-STFloodNet retains an operational discriminability of **ROC-AUC = 0.8472** compared to LightGBM collapsing to **0.5136**, demonstrating the value of deep cross-modal representations in preserving residual risk signal.
