# Model Card: Bidirectional LSTM Sequential Flood Predictor

## 1. Model Overview
* **Model Name**: Bidirectional LSTM Sequential Flood Predictor
* **Architecture**: 2-layer Bidirectional LSTM + Linear Projection Head (PyTorch)
* **Trained File**: `models/trained/model_bilstm.pt`
* **Input**: 14-day sliding sequential tensors $[B, 14, 25]$

## 2. Metrics (2023 Chronological Test Set)
* **Accuracy**: 0.9330
* **Precision**: 0.1816
* **Recall**: 0.8804 (81 / 92 flood events detected, 11 missed)
* **F1-Score**: 0.3011
* **PR-AUC**: 0.5097
* **ROC-AUC**: 0.9672
* **Brier Score**: 0.0546
* **Training Time**: 31.56 seconds
* **Inference Latency**: 0.0503 ms per sample

## 3. Comparative Assessment
* While the Bi-LSTM model effectively learns temporal dynamics, it yields 11 false negatives on the test set compared to only 2 for Random Forest.
* In flood hazard warning, avoiding false negatives is paramount. Engineered multi-day rolling sums and Antecedent Precipitation Index ($API$) provide clearer signals to tree-based models with lower sample variance than free-form recurrent sequence representations.
