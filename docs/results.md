# Experimental Results: AI-Based Flood Prediction & Resilience Planning

This document reports the verified empirical findings obtained from executing the flood prediction and resilience framework on the Kerala River Basins benchmark dataset.

---

## 1. Out-of-Time Test Set Performance (2023 Chronological Split)

| Model Name | Accuracy | Precision | Recall | F1-Score | PR-AUC | ROC-AUC | Brier Score | Latency (ms/sample) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 0.9202 | 0.1637 | **0.9891** | 0.2809 | 0.6218 | 0.9934 | 0.0566 | **0.0005** |
| **Random Forest (Champion)** | **0.9949** | **0.7627** | 0.9783 | **0.8571** | 0.9888 | **0.9997** | **0.0090** | 0.0082 |
| **Gradient Boosting** | 0.9577 | 0.2700 | **0.9891** | 0.4242 | 0.9849 | 0.9992 | 0.0271 | 0.0012 |
| **XGBoost** | 0.9825 | 0.4740 | **0.9891** | 0.6408 | **0.9893** | 0.9994 | 0.0133 | 0.0012 |
| **Bi-LSTM (Sequential DL)** | 0.9330 | 0.1816 | 0.8804 | 0.3011 | 0.5097 | 0.9672 | 0.0546 | 0.0503 |

*Source: `outputs/metrics/model_comparison.json`*

---

## 2. Test Set Confusion Matrix Breakdown

* **Total Test Observations**: 5,840 daily records
* **Actual Flood Days**: 92
* **Actual Non-Flood Days**: 5,748

| Model | True Negatives | False Positives | False Negatives | True Positives | Flood Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 5,283 | 465 | **1** | 91 | 98.91% |
| **Random Forest** | **5,720** | **28** | 2 | 90 | 97.83% |
| **Gradient Boosting** | 5,502 | 246 | **1** | 91 | 98.91% |
| **XGBoost** | 5,647 | 101 | **1** | 91 | 98.91% |
| **Bi-LSTM** | 5,159 | 365 | 11 | 81 | 88.04% |

---

## 3. Multimodal Feature Ablation Study

Progression of performance as multimodal data layers are incrementally added to the feature space:

| Experiment | Feature Set | Number of Features | F1-Score | Recall | PR-AUC | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Exp A** | Meteorological Features Only | 8 | 0.5294 | 0.9783 | 0.8026 | 0.9967 |
| **Exp B** | Meteorological + Terrain DEM | 16 | 0.6040 | 0.9783 | 0.9796 | 0.9989 |
| **Exp C** | Meteorological + Terrain + Remote Sensing SAR | 21 | 0.6190 | **0.9891** | **0.9900** | **0.9994** |
| **Exp D** | All Features (including Exposure & Vulnerability) | 25 | **0.6408** | **0.9891** | 0.9893 | **0.9994** |

### Key Ablation Insights:
1. **Adding Terrain Parameters (Exp B)**: Produces an immediate **+17.7% increase in PR-AUC** (0.8026 to 0.9796), demonstrating that topography (elevation and slope) is vital for constraining runoff accumulation.
2. **Adding Sentinel-1 SAR Backscatter (Exp C)**: Boosts Recall from 97.83% to 98.91% and achieves the peak PR-AUC of 0.9900.
3. **Full Multimodal Stack (Exp D)**: Delivers the highest overall F1 score (0.6408) with near-zero false negatives.
