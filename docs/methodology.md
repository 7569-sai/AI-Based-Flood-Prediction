# Methodology: AI-Based Flood Prediction & Resilience Planning

## 1. Study Area & Dataset Construction

The empirical study focuses on Kerala, Southwestern India, an ecologically sensitive region subject to intense South-West monsoon and North-East monsoon deluges interacting with the steep hydraulic drop of the Western Ghats into coastal backwaters.

### Monitoring Network
16 monitoring stations spanning 8 major river basins:
* **Periyar Basin**: Aluva (`KL_01`), Kalady (`KL_02`), Malayattoor (`KL_03`), Idukki (`KL_11`)
* **Chalakkudy Basin**: Chalakkudy (`KL_04`)
* **Karuvannur Basin**: Thrissur Town (`KL_05`)
* **Pamba Basin**: Chengannur (`KL_06`), Ranni (`KL_07`)
* **Pamba-Meenachil Delta**: Kuttanad / Mankombu (`KL_08`), Alappuzha Coastal (`KL_09`), Kottayam (`KL_10`)
* **Muthirapuzha (Western Ghats)**: Munnar (`KL_12`)
* **Bharathapuzha Basin**: Palakkad (`KL_13`)
* **Chaliyar Basin**: Kozhikode / Feroke (`KL_14`)
* **Kabini Basin**: Mananthavady / Wayanad (`KL_15`)
* **Karamana Basin**: Thiruvananthapuram (`KL_16`)

---

## 2. Leakage-Free Preprocessing & Chronological Splitting

To ensure scientific validity and eliminate temporal, spatial, target, and preprocessing leakage:
1. **Strict Shift Operation**: All rolling accumulations apply `shift(1)` to ensure features strictly use information available at $T-1$ to predict day $T$.
2. **Chronological Splitting**:
   * **Training Set**: 2018-01-01 to 2021-12-31 (23,376 records, covering the catastrophic 2018, 2019, and 2021 Kerala floods).
   * **Validation Set**: 2022-01-01 to 2022-12-31 (5,840 records).
   * **Out-of-Time Test Set**: 2023-01-01 to 2023-12-31 (5,840 records, 92 flood events).
3. **Scaler Freezing**: The `StandardScaler` is fitted strictly on the Training set and frozen before transforming the Validation and Test sets.

---

## 3. Model Development & Evaluation

Models were evaluated using precision-recall metrics prioritizing **Recall, F1-Score, and PR-AUC** (critical in natural hazard prediction to prevent catastrophic false negatives):

1. **Model A (Baseline)**: Balanced Logistic Regression ($L_2$ regularization)
2. **Model B (Champion)**: Random Forest Classifier (150 trees, max depth 12, min samples split 6, balanced class weighting)
3. **Model C**: HistGradientBoostingClassifier with class frequency balancing
4. **Model D**: XGBoost Classifier with `scale_pos_weight = 20.0`
5. **Model E**: Bidirectional LSTM (2 layers, hidden size 48) trained on 14-day sequential sliding windows.

---

## 4. Disaster Risk Integration Framework

Follows the UNDRR tripartite framework:
$$\text{Risk} = \text{Hazard} \times \text{Exposure} \times \text{Vulnerability}$$
* **Hazard ($H \in [0, 1]$)**: Model output probability $P(\text{Flood} \mid X)$.
* **Exposure ($E \in [0, 1]$)**: Normalized log population density.
* **Vulnerability ($V \in [0, 1]$)**: Multi-criteria index evaluating elevation deficit, slope, and river proximity.
* **Composite Risk Score ($R \in [0, 100]$)**:
  $$R = (0.50 \cdot H + 0.25 \cdot E + 0.25 \cdot V) \times 100$$
