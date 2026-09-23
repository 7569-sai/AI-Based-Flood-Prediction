# Model Comparison: AI-Based Flood Prediction

This report presents empirical performance benchmarks across four machine-learning architectures and one sequential deep-learning architecture on the Kerala River Basins benchmark dataset.

All models were evaluated on the **2023 chronological out-of-time test set** (5,840 records, 92 documented flood events).

---

## 1. Comprehensive Performance Benchmark

| Model Name | Type | Accuracy | Precision | Recall | F1-Score | PR-AUC | ROC-AUC | Brier Score | Training Time (s) | Inference Latency (ms/sample) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A: Logistic Regression** | Linear Baseline | 0.9202 | 0.1637 | **0.9891** | 0.2809 | 0.6218 | 0.9934 | 0.0566 | **0.186** | **0.0005** |
| **Model B: Random Forest** | Bagging Ensemble | **0.9949** | **0.7627** | 0.9783 | **0.8571** | 0.9888 | **0.9997** | **0.0090** | 0.662 | 0.0082 |
| **Model C: Gradient Boosting** | Boosting Ensemble | 0.9577 | 0.2700 | **0.9891** | 0.4242 | 0.9849 | 0.9992 | 0.0271 | 0.437 | 0.0012 |
| **Model D: XGBoost** | Gradient Tree Boosting | 0.9825 | 0.4740 | **0.9891** | 0.6408 | **0.9893** | 0.9994 | 0.0133 | 2.205 | 0.0012 |
| **Model E: Bi-LSTM** | Sequential Deep Learning | 0.9330 | 0.1816 | 0.8804 | 0.3011 | 0.5097 | 0.9672 | 0.0546 | 31.557 | 0.0503 |

*Note: All reported metrics are verified empirical outputs from `outputs/metrics/model_comparison.json`.*

---

## 2. Confusion Matrix Analysis (Test Set: 5,840 Observations)

| Model | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | True Positives (TP) | Detection Rate (TP / Actual Floods) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 5,283 | 465 | **1** | 91 / 92 | 98.91% |
| **Random Forest** | **5,720** | **28** | 2 | 90 / 92 | 97.83% |
| **Gradient Boosting** | 5,502 | 246 | **1** | 91 / 92 | 98.91% |
| **XGBoost** | 5,647 | 101 | **1** | 91 / 92 | 98.91% |
| **Bi-LSTM** | 5,159 | 365 | 11 | 81 / 92 | 88.04% |

---

## 3. Analysis & Key Takeaways

1. **Ensemble Trees Dramatically Outperform Linear Baselines**:
   * While Logistic Regression achieves high recall (0.9891), it suffers from 465 false alarms (Precision: 0.1637, F1: 0.2809).
   * **Random Forest** cuts false positives down from 465 to just **28**, driving Precision to **0.7627** and F1 to **0.8571** while maintaining a near-perfect recall of **0.9783** (missing only 2 events out of 92).

2. **XGBoost Precision vs Recall Tradeoff**:
   * XGBoost achieves the highest PR-AUC (**0.9893**) and an outstanding Recall of **0.9891** (missing only 1 event), with an F1 score of **0.6408**.

3. **Deep Learning (Bi-LSTM) Findings**:
   * The Bidirectional LSTM trained on 14-day sliding sequences achieved an 88.04% recall and 0.5097 PR-AUC, but required 48x longer training time and produced 11 false negatives (5.5x more misses than Random Forest).
   * The explicit engineered antecedent features (`rainfall_3d`, `rainfall_7d`, `api_14d`, `soil_saturation_ratio`) already capture the key catchment hydrologic memory far more effectively and with lower inductive bias than unconstrained LSTM sequence representations.
