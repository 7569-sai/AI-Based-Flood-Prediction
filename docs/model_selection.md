# Model Selection Rationale

## 1. Selected Final Model: **Random Forest Classifier**

* **Architecture**: Random Forest (150 estimators, max depth 12, min samples split 6, balanced class weighting)
* **Artifact Path**: `models/trained/model_random_forest.joblib`
* **Test Performance (2023 Chronological Out-of-Time Test Set)**:
  * **F1-Score**: **0.8571** (Highest across all evaluated models)
  * **Recall**: **0.9783** (Detects 90 of 92 historical flood days)
  * **Precision**: **0.7627** (Only 28 false alarms across 5,840 daily station observations)
  * **PR-AUC**: **0.9888**
  * **ROC-AUC**: **0.9997**
  * **Brier Calibration Score**: **0.0090** (Best calibrated probability output)
  * **Inference Latency**: **0.0082 ms / sample** (Enables real-time API response < 10ms)

---

## 2. Decision Framework & Elimination Rationale

### A. Why Not Logistic Regression?
While Logistic Regression was an effective baseline with 0.9891 recall, its linear decision surface cannot capture nonlinear thresholds (e.g. antecedent rainfall interacting with soil saturation and elevation). This resulted in **465 false alarms** (Precision: 0.1637, F1: 0.2809), which would lead to alert fatigue in emergency management.

### B. Why Random Forest Over XGBoost?
* Both tree ensemble methods achieved near-perfect recall (0.9783 for RF vs 0.9891 for XGBoost).
* However, Random Forest demonstrated substantially higher precision (**0.7627 vs 0.4740**) and fewer false alarms (**28 vs 101**), resulting in a far superior F1 score (**0.8571 vs 0.6408**).
* Random Forest also showed the lowest Brier score (**0.0090 vs 0.0133**), indicating superior probability calibration, which is essential when translating probabilities into continuous 0–100 flood risk scores.

### C. Why Not Deep Learning (Bi-LSTM)?
* The Bi-LSTM produced 11 false negatives (88.04% recall) compared to only 2 false negatives for Random Forest (97.83% recall).
* In disaster risk reduction, false negatives (failing to alert a flooded population) have life-threatening consequences.
* Furthermore, Random Forest executes in 0.008 ms/sample (6.1x faster than LSTM) and can be directly interpreted using TreeSHAP with exact game-theoretic guarantees.

---

## 3. Operational Deployment Recommendation

Deploy `model_random_forest.joblib` as the primary operational scoring engine for:
1. Point-level flood probability prediction (`POST /predict`)
2. Composite flood-risk scoring (`POST /risk-score`)
3. Global and local TreeSHAP explanations (`POST /explain`)
4. Geospatial flood risk mapping (`POST /geospatial-risk`)
