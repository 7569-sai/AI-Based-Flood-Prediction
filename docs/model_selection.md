# Model Selection & Architecture Comparison

**Project Title:** AI-Based Flood Prediction, Dynamic Risk Mapping and Resilience Planning  
**Application:** FloodAI Engine — "AI for a Safer, Resilient India"

---

## 1. Comprehensive Model Evaluation Matrix (Pan-India 2023 Chronological Test Set)

Evaluated on 13,140 out-of-time daily observations across 36 national stations in 12 major Indian river basins (187 ground-truth flood occurrences):

| Category | Model Architecture | Accuracy | Precision | Recall | F1-Score | PR-AUC | ROC-AUC | Brier Score | Latency (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | Logistic Regression (Class-Weighted) | 0.9696 | 0.3186 | 1.0000 | 0.4832 | 0.6652 | 0.9939 | 0.0248 | 0.007 ms |
| **Baseline** | Random Forest (Balanced) | 0.9868 | 0.5206 | 0.9465 | 0.6717 | 0.7866 | 0.9960 | **0.0108** | 0.020 ms |
| **Baseline** | Gradient Boosting | 0.9899 | **0.6776** | 0.5508 | 0.6077 | 0.7908 | 0.9960 | 0.0094 | 0.015 ms |
| **Baseline** | XGBoost (scale_pos_weight) | 0.9873 | 0.5291 | 0.9733 | **0.6855** | 0.7878 | 0.9961 | 0.0112 | 0.016 ms |
| **Baseline** | LightGBM (Scalable Histogram) | **0.9881** | 0.5490 | 0.8984 | 0.6815 | **0.7937** | **0.9962** | 0.0103 | **0.009 ms** |
| **Baseline** | PyTorch Bi-LSTM (Temporal Sequence) | 0.9751 | 0.3638 | 1.0000 | 0.5335 | 0.7704 | 0.9952 | 0.0205 | 0.110 ms |
| **PROPOSED** | **ST-FloodNet (Multimodal + Uncertainty)** | 0.9709 | 0.3286 | **1.0000** | 0.4947 | 0.7622 | 0.9949 | 0.0234 | 0.145 ms |

---

## 2. Spatial Cross-Basin Generalization (Held-out Unseen Basins)

Evaluated on completely unseen river basins (Brahmaputra and Mahanadi, 10,955 daily observations, 415 flood events):

| Model Architecture | Accuracy | Precision | Recall (Detection Rate) | F1-Score | PR-AUC | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM (Strongest Tabular Baseline)** | **0.9803** | **0.8100** | 0.6265 | 0.7065 | 0.8670 | 0.9927 |
| **ST-FloodNet (Proposed Research Model)** | 0.9697 | 0.5556 | **1.0000** | **0.7143** | **0.8751** | **0.9943** |

---

## 3. Scientific Rationale & Role Separation

### A. Role of Baseline Models
- **Conventional Tabular Baselines (Logistic Regression, Random Forest, Gradient Boosting, XGBoost)**: Provide established benchmarks. Random Forest and XGBoost achieve strong in-domain F1 scores (0.6717 and 0.6855).
- **LightGBM**: Implemented as the primary scalable tabular baseline. It demonstrates the highest inference efficiency (0.009 ms/sample) and the highest in-domain PR-AUC (0.7937). However, as an established algorithm, LightGBM is treated strictly as an engineering baseline, NOT a research contribution.

### B. Role of Proposed Research Architecture: ST-FloodNet
- **Novelty & Purpose**: ST-FloodNet is designed specifically to solve the cross-basin generalization gap and prediction uncertainty:
  1. **Multi-Scale Temporal Convolution**: Extracts multi-frequency hydrologic signals across $1\text{d}, 3\text{d}, 7\text{d}, 14\text{d}$ antecedent scales.
  2. **Gated Multimodal Fusion**: Dynamically balances satellite radar backscatter against physical terrain elevation deficits.
  3. **Multi-Task Learning**: Jointly predicts binary flood hazard probability and ordinal flood severity (None, Minor, Moderate, Severe).
  4. **Uncertainty Quantification**: Uses Monte Carlo Dropout ($T=30$) to compute epistemic predictive variance ($\sigma_{\text{MC}}^2$), alerting operators to ambiguous hydrological regimes.
  5. **Cross-Basin Transfer**: Achieved **100% recall (415/415 events)** and the highest F1 (**0.7143**) on unseen river basins.
