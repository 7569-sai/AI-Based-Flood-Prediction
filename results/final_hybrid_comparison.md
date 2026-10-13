# Pan-India Flood Prediction: Benchmark & Hybrid Comparison (2023 Test Set)

| model_name | accuracy | precision | recall | f1 | roc_auc | pr_auc | mcc | balanced_accuracy | brier_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Model A: LightGBM Only | 0.9881 | 0.549 | 0.8984 | 0.6815 | 0.9962 | 0.7937 | 0.6972 | 0.9439 | 0.0093 |
| Model B: ST-FloodNet Only | 0.9775 | 0.388 | 1.0 | 0.559 | 0.9956 | 0.7754 | 0.6157 | 0.9886 | 0.0198 |
| Model C: Simple Blend (Ablation) | 0.9855 | 0.4947 | 0.9947 | 0.6607 | 0.9963 | 0.8004 | 0.6962 | 0.99 | 0.0122 |
| Model E: Embedding + LightGBM | 0.9873 | 0.5336 | 0.8503 | 0.6557 | 0.9959 | 0.7915 | 0.6679 | 0.9198 | 0.0098 |
| Model D: HF-STFloodNet (Proposed) | 0.9781 | 0.3937 | 1.0 | 0.565 | 0.9951 | 0.7488 | 0.6204 | 0.9889 | 0.0175 |

*Evaluation on held-out 2023 chronological test set across 36 Indian national stations (13,140 daily records).*
