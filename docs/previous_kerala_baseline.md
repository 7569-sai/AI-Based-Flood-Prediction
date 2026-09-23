# Previous Kerala Baseline Results (Archive)

> [!NOTE]
> This document archives the initial benchmark metrics evaluated strictly on the Kerala River Basins out-of-time test set (2023 chronological holdout, 5,840 daily records, 92 flood events). These metrics serve as the documented regional baseline and must NOT be conflated with the subsequent Pan-India multi-basin evaluation.

---

## 1. Study Area & Dataset Configuration (Kerala Baseline)
- **Geographic Domain**: Kerala River Basins (8 major basins: Periyar, Chalakkudy, Pamba, Meenachil, Bharathapuzha, Achenkovil, Muvattupuzha, Manimala).
- **Monitoring Stations**: 16 CWC/IMD stations.
- **Temporal Span**: January 1, 2018 to December 31, 2023 (6 full years, 35,056 daily records).
- **Chronological Split**:
  - **Train**: 2018–2021 (23,376 records)
  - **Validation**: 2022 (5,840 records)
  - **Test**: 2023 (5,840 records, 92 ground-truth flood occurrences)
- **Features**: 22 hydrometeorological, topographic, and satellite variables (ERA5-Land reanalysis, SRTM 30m DEM, Sentinel-1 SAR backscatter, CIESIN GPWv4 population density).

---

## 2. Model Performance Summary (Chronological 2023 Test Set)

| Model Architecture | Accuracy | Precision | Recall | F1-Score | PR-AUC | ROC-AUC | Brier Score | Training Time (s) | Inference Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Balanced)** | **0.9949** | **0.7627** | **0.9783** | **0.8571** | **0.9888** | **0.9997** | **0.0090** | 1.84 s | 0.021 ms |
| **Gradient Boosting** | 0.9901 | 0.6094 | 0.8478 | 0.7091 | 0.8872 | 0.9959 | 0.0084 | 4.12 s | 0.015 ms |
| **XGBoost (scale_pos_weight)** | 0.9825 | 0.4740 | 0.9891 | 0.6408 | 0.9893 | 0.9994 | 0.0135 | 0.85 s | 0.018 ms |
| **PyTorch Bi-LSTM** | 0.9841 | 0.4906 | 0.5652 | 0.5253 | 0.5367 | 0.9304 | 0.0142 | 48.20 s | 0.120 ms |
| **Logistic Regression (Class-Weighted)** | 0.8998 | 0.1294 | 0.7826 | 0.2222 | 0.4194 | 0.9328 | 0.0764 | 0.12 s | 0.008 ms |

---

## 3. Multimodal Feature Ablation Study

| Experiment Tier | Multimodal Feature Configuration | Accuracy | Precision | Recall | F1-Score | PR-AUC | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp A** | Meteorology Only (Rainfall 1d/3d/7d/14d, Anomaly, API) | 0.9880 | 0.5620 | 0.8804 | 0.6864 | 0.8712 | 0.9891 |
| **Exp B** | Meteorology + DEM Topography (Elevation, Slope, TWI) | 0.9918 | 0.6723 | 0.9348 | 0.7818 | 0.9340 | 0.9942 |
| **Exp C** | Meteorology + DEM + Sentinel-1 SAR (VV, VH, Cross-Ratio) | 0.9935 | 0.7258 | 0.9674 | 0.8295 | 0.9715 | 0.9978 |
| **Exp D** | Full Multimodal Stack (+ Exposure & Drainage Vulnerability) | **0.9949** | **0.7627** | **0.9783** | **0.8571** | **0.9888** | **0.9997** |

---

## 4. Key Takeaways
1. Random Forest delivered the highest balanced performance with 97.8% recall (detecting 90 out of 92 historical flood events) and only 28 false positives across 5,840 out-of-time test instances.
2. The multimodal ablation confirmed that satellite SAR backscatter and topographic DEM features are essential, increasing F1 from 0.6864 to 0.8571 (+17.07% F1 gain).
3. These results are preserved as the regional baseline prior to the Pan-India multi-basin expansion.
