# Hybrid Architecture Leakage Audit & Cross-Fitting Protocol

**Project Title:** AI-Based Flood Prediction, Dynamic Risk Mapping and Resilience Planning  
**Application:** FloodAI Engine — "AI for a Safer, Resilient India"  
**Audit Standard:** Leakage-Free Machine Learning Stacking & Spatiotemporal Partitioning  

---

## 1. Mathematical Formulation of the Stacking Problem

In stacked generalization (Wolpert, 1992; Breiman, 1996), combining predictions from a deep neural network $f_{\text{DL}}(\mathbf{x}; \boldsymbol{\theta}_{\text{DL}})$ and a gradient-boosted decision tree ensemble $f_{\text{GBDT}}(\mathbf{x}; \boldsymbol{\theta}_{\text{GBDT}})$ into a meta-learner $g(\cdot; \boldsymbol{\theta}_{\text{meta}})$ introduces grave risks of target leakage if base-model predictions on the training partition are used directly. 

Because base models have partially memorized training samples, their in-sample predictions exhibit artificially low residual error, causing the meta-learner to assign disproportionate weight to overconfident predictions and catastrophic overfitting on unseen spatial or temporal test data.

Furthermore, in spatiotemporal environmental modeling, two distinct leakage pathways exist:
1. **Temporal Leakage:** Using future observations (day $t + k$) to predict day $t$, or computing rolling statistics with centered rather than trailing backward windows.
2. **Spatial Leakage:** Randomly splitting geographically correlated station observations across train and test folds, allowing the model to memorize local topographic signatures rather than learning invariant hydrological physics.

---

## 2. Leakage Mitigation Architecture

The FloodAI Engine implements a rigorous, three-tier spatiotemporal partition and cross-fitting protocol:

```
[ Dataset: 78,876 Daily Station Records Across 36 Pan-India Stations ]
                                  │
         ┌────────────────────────┴────────────────────────┐
         │                                                 │
[ Training Split: 2018–2021 ]                     [ Temporal Holdouts ]
 (52,596 records, 36 stations)                             │
         │                                       ┌─────────┴─────────┐
         ▼                                       ▼                   ▼
  5-Fold Station-Aware GroupKFold          [ Validation: 2022 ] [ Test: 2023 ]
   Fold 1: Stations {1..7}                   (13,140 records)     (13,140 records)
   Fold 2: Stations {8..14}                         │                   │
   Fold 3: Stations {15..21}                        │                   │
   Fold 4: Stations {22..28}                        │                   │
   Fold 5: Stations {29..36}                        │                   │
         │                                          │                   │
         ├───────────────────────────────┐          │                   │
         ▼                               ▼          │                   │
Base DL: ST-FloodNet            Base GBDT: LightGBM │                   │
  Train on 4 station groups       Train on 4 groups │                   │
  Predict OOF on held-out group   Predict OOF       │                   │
         │                               │          │                   │
         └───────────────┬───────────────┘          │                   │
                         ▼                          │                   │
         [ Out-Of-Fold Feature Matrix $\mathbf{Z}_{\text{OOF}}$ ]      │                   │
                         │                          │                   │
                         ▼                          │                   │
               Train Meta-Learner                   │                   │
              $g(\mathbf{Z}; \boldsymbol{\theta}_{\text{meta}})$              │                   │
                         │                          │                   │
                         └───────────────┬──────────┘                   │
                                         ▼                              │
                        Validation Optimization (2022)                  │
                        1. Threshold Tuning: $\theta^* \in [0.05, 0.95]$ │
                        2. Platt Scaling / Isotonic Calibration         │
                                         │                              │
                                         └──────────────┬───────────────┘
                                                        ▼
                                             Frozen Test Evaluation (2023)
                                             Zero parameter updates
                                             Unbiased empirical benchmark
```

---

## 3. Strict Algorithmic Guarantees

### Guarantee 1: Strict Temporal Causality ($T-1$)
All 27 input features represent backward-looking historical summaries or static geospatial baselines. Specifically:
- `rainfall_1d`: Precipitation on day $t-1$.
- `rainfall_3d`, `rainfall_7d`, `rainfall_14d`: Trailing sums $\sum_{i=1}^k P(t-i)$.
- `river_stage_anomaly`, `soil_moisture`: Measured prior to the flood prediction forecast window.
- **Verification:** No centered rolling windows or forward-looking terms ($t+1, t+2$) exist in the feature extraction pipeline.

### Guarantee 2: Station-Aware Grouped Out-Of-Fold Cross-Fitting
To construct the training dataset for the meta-learner:
$$\mathcal{D}_{\text{train}} = \bigcup_{k=1}^K \mathcal{S}_k, \quad \mathcal{S}_j \cap \mathcal{S}_k = \emptyset \quad \forall j \ne k$$
For each fold $k \in \{1, \dots, 5\}$:
1. Base ST-FloodNet and LightGBM models are trained strictly on $\mathcal{D}_{\text{train}} \setminus \mathcal{S}_k$.
2. Inference is executed strictly on the unseen station group $\mathcal{S}_k$ to generate out-of-fold probability vectors $\hat{\mathbf{p}}_{\text{ST}}^{(k)}$ and $\hat{\mathbf{p}}_{\text{LGB}}^{(k)}$, along with latent representation embeddings $\mathbf{h}_{\text{ST}}^{(k)}$.
3. The stacked training matrix is assembled:
$$\mathbf{Z}_{\text{OOF}} = \left[ \hat{\mathbf{p}}_{\text{ST}}, \hat{\mathbf{p}}_{\text{LGB}}, \hat{\mathbf{p}}_{\text{ST}} - \hat{\mathbf{p}}_{\text{LGB}}, \hat{\mathbf{p}}_{\text{ST}} \odot \hat{\mathbf{p}}_{\text{LGB}}, \mathbf{h}_{\text{ST}}^{\text{PCA}}, \mathbf{X}_{\text{struct}} \right]$$
4. The meta-learner is fitted exclusively on $\mathbf{Z}_{\text{OOF}}$ against the true binary labels $\mathbf{y}_{\text{train}}$.

### Guarantee 3: Isolation of Validation Split (2022)
The validation partition (all 36 stations throughout calendar year 2022) is completely withheld from:
- Neural network backpropagation
- Gradient boosting tree splitting
- Out-of-fold meta-learner training
- PCA latent projection fitting

The validation set is used exclusively for:
1. **Decision Threshold Selection:** Scanning $\theta \in [0.05, 0.95]$ with step $0.01$ to identify $\theta^* = \arg\max_\theta F_1(\theta; \text{Val})$.
2. **Probability Calibration:** Fitting a univariate Platt scaling logistic transformation $\sigma(\alpha \hat{p} + \beta)$ to minimize Brier score and Expected Calibration Error (ECE).

### Guarantee 4: Frozen Evaluation on Test Split (2023)
The 2023 test set represents an untouched temporal benchmark:
- All scalers (StandardScaler fitted on 2018–2021) are applied frozen.
- Base models and meta-learner execute with frozen weights.
- The decision threshold $\theta^*$ and calibration mapping $f_{\text{calib}}$ derived from 2022 are applied frozen.
- No post-hoc tuning or hyperparameter searching is conducted on test data.

---

## 4. Verification Check Script

The repository includes automated checks within `src/data/audit_pan_india_leakage.py` and `tests/test_hybrid_model.py` verifying:
1. Station ID intersection between fold train and fold test is null.
2. Temporal bounds: $\max(\text{Date}_{\text{train}}) < \min(\text{Date}_{\text{val}}) < \min(\text{Date}_{\text{test}})$.
3. No feature correlation with target exceeds physical plausibility ($r < 0.90$).
