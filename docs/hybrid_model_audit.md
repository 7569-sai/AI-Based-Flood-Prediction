# Comprehensive Scientific Audit: Model Architecture, Data Flow, and Hybridization Roadmap

**Project Title:** AI-Based Flood Prediction, Dynamic Risk Mapping and Resilience Planning  
**Application:** FloodAI Engine (Pan-India Multi-Basin Deployment)  
**Document Purpose:** Baseline audit of existing model implementations (ST-FloodNet, LightGBM, Random Forest), data pipelines, feature representations, and specifications for creating **HF-STFloodNet** (Hybrid Fusion Spatiotemporal Flood Network).  
**Date:** September 2026  

---

## 1. Executive Findings & Baseline Architecture Audit

### A. What Features ST-FloodNet Currently Receives
ST-FloodNet (`src/models/train_pan_india.py:STFloodNet`) is instantiated to accept a 1D feature vector of dimension 27 (`x_all` $\in \mathbb{R}^{B \times 27}$). Internally, its forward pass partitions `x_all` as follows:
1. `x_meteo = x_all[:, :8].unsqueeze(1)` ($\mathbb{R}^{B \times 1 \times 8}$): Meteorological & antecedent features:
   - `rainfall_1d`, `rainfall_3d`, `rainfall_7d`, `rainfall_14d`, `rainfall_anomaly_30d`, `api_14d`, `temp_mean_3d`, `humidity_mean_3d`.
   - Processed by two 1D convolutional layers (`temp_conv1`, kernel=3, channels=16; `temp_conv2`, kernel=5, channels=16), concatenated along channel dimension to $\mathbb{R}^{B \times 8 \times 32}$, and fed into a 2-layer Bidirectional GRU (`hidden_dim=32`).
2. `x_terrain = x_all[:, 8:16]` ($\mathbb{R}^{B \times 8}$): Hydrological & terrain features:
   - `soil_moisture_root_zone`, `soil_moisture_deep`, `soil_saturation_ratio`, `river_discharge_proxy`, `river_stage_anomaly`, `elevation_m`, `slope_deg`, `dist_river_km`.
   - Processed by `spatial_encoder` (Linear $8 \to 32$, BatchNorm1d, ReLU, Dropout $p=0.35$).
3. `x_sar = x_all[:, 16:21]` ($\mathbb{R}^{B \times 5}$): Geomorphic indices & SAR backscatter:
   - `twi_proxy`, `elevation_risk_factor`, `drainage_vulnerability`, `sar_vv`, `sar_vh`.
   - Processed by `sar_encoder` (Linear $5 \to 16$, BatchNorm1d, ReLU, Dropout $p=0.35$).
4. **Architectural Gap in Baseline Implementation:**
   Features 21 to 26 (`sar_cross_ratio`, `ndwi_proxy`, `ndvi_proxy`, `population_density`, `exposure_index`, `vulnerability_index`) are part of the 27-feature schema, but in the baseline `STFloodNet.forward()` they were not routed into any sub-encoder branch.

### B. What Features LightGBM Currently Receives
LightGBM (`model_pan_india_lightgbm.joblib`) receives all **27 scaled features** as a continuous tabular matrix:
- 8 Meteorological & antecedent features
- 5 Hydrological features
- 6 Terrain & geomorphological features
- 5 Remote sensing / SAR features
- 3 Demographic & exposure features

It operates using histogram-based gradient boosting (`num_leaves=31`, `learning_rate=0.07`, `n_estimators=150`, `scale_pos_weight=pos_weight`).

### C. How the Scaler is Fitted
- The scaler (`StandardScaler`) is fitted **strictly on the chronological training split** (`data/processed/india/train.parquet`, 52,596 rows corresponding to years 2018–2021).
- Validation (`val.parquet`, year 2022) and test sets (`test.parquet`, year 2023) are transformed using the fitted train scaler parameters ($\mu_{\text{train}}, \sigma_{\text{train}}$).
- **Leakage Status:** Clean. No validation or test statistics leak into the feature normalization.

### D. How Train/Validation/Test Splits are Created
The dataset (`78,876` records across 36 stations in 24 river basins) is split using two distinct, scientifically justified paradigms:
1. **Chronological Splitting (Temporal Generalization):**
   - **Train:** 2018-01-01 to 2021-12-31 ($N = 52,596$, 66.7%)
   - **Validation:** 2022-01-01 to 2022-12-31 ($N = 13,140$, 16.7%)
   - **Test:** 2023-01-01 to 2023-12-31 ($N = 13,140$, 16.7%)
2. **Spatial Holdout Splitting (Cross-Basin Generalization):**
   - **Training Basins:** 22 river basins across Peninsular, Gangetic, and Himalayan India.
   - **Unseen Held-out Basins:** 2 geographically and hydrologically distinct basins:
     - `Brahmaputra` (High-gradient antecedent snowmelt & intense monsoon regime in North-East India)
     - `Mahanadi` (East-coast cyclone- and delta-prone river basin)

### E. Temporal Feature Representation: Antecedent Encoding vs True Sequences
> [!IMPORTANT]
> **Scientific Integrity Verification:**
> The current dataset does **NOT** provide raw continuous 3D sequential tensors of shape $[B, T, F]$ (e.g. 30 daily consecutive steps per instance).
> Instead, each sample is a 1D tabular vector representing a single daily timestamp containing **engineered multi-scale antecedent and lag windows** ($T-1, T-3, T-7, T-14, T-30$ precipitation sums, API, rolling temperature, and stage anomalies).
> 
> The 1D Conv and Bi-GRU in `STFloodNet` operate across the ordered channels of these multi-scale temporal features, functioning as an **antecedent multi-scale feature encoder**, rather than a true long-term recurrent sequence forecaster.
> In this project, this will be reported accurately as **multi-scale antecedent temporal encoding**; sequence claims will not be fabricated.

### F. How ST-FloodNet Currently Produces Outputs
1. Sub-branch embeddings:
   - Temporal Bi-GRU: $\mathbb{R}^{B \times 64}$
   - Terrain encoder: $\mathbb{R}^{B \times 32}$
   - SAR encoder: $\mathbb{R}^{B \times 16}$
2. Concatenation: $\mathbb{R}^{B \times (64 + 32 + 16)} = \mathbb{R}^{B \times 112}$
3. Gated Fusion:
   $$\mathbf{g} = \sigma(\mathbf{W}_g \mathbf{f} + \mathbf{b}_g) \in \mathbb{R}^{B \times 112}$$
   $$\mathbf{f}_{\text{gated}} = \mathbf{f} \odot \mathbf{g}$$
4. Latent Representation:
   $$\mathbf{z} = \text{Dropout}(\text{ReLU}(\mathbf{W}_f \mathbf{f}_{\text{gated}} + \mathbf{b}_f), p=0.35) \in \mathbb{R}^{B \times 64}$$
5. Multi-task Heads:
   - Hazard logit: $\hat{y}_{\text{haz}} = \mathbf{W}_h \mathbf{z} + b_h \in \mathbb{R}^{B \times 1}$ (evaluated with `BCEWithLogitsLoss`)
   - Severity logits: $\hat{\mathbf{y}}_{\text{sev}} = \mathbf{W}_s \mathbf{z} + \mathbf{b}_s \in \mathbb{R}^{B \times 4}$ (evaluated with `CrossEntropyLoss`)
6. **Latent Embedding Extraction:** Currently, `forward()` only returns `(hazard_logit, severity_logits)`. The latent vector $\mathbf{z}$ is discarded after the forward pass.

### G. How LightGBM Currently Produces Predictions
- LightGBM takes the tabular vector $\mathbf{x} \in \mathbb{R}^{27}$, traverses an ensemble of 150 gradient-boosted regression trees, and computes margin sums transformed via sigmoid to obtain probability $\hat{p}_{\text{LGBM}} \in [0, 1]$.
- Excels at non-linear axis-aligned thresholding, feature interactions, and tabular structure, but lacks spatial/cross-modal representation learning.

### H. How `prediction_service.py` Selects the Active Model
- `src/models/prediction_service.py` currently loads `model_pan_india_random_forest.joblib` directly.
- The model selection is hardcoded without a dynamic registry or modular model selector.
- It does not load `ST-FloodNet` or `LightGBM` during runtime inference.

### I. How the Dashboard Gets Predictions
- `dashboard/app.py` caches `PredictionService` via `@st.cache_resource`.
- On user click $\to$ coordinate capture $\to$ `feature_extractor.retrieve_location_data()` retrieves 27 features $\to$ `prediction_service.predict(features)` generates probability, severity, uncertainty, and UNDRR risk score.

### J. How Uncertainty is Currently Calculated
- In `prediction_service.py`: Random Forest tree variance $\sigma_{\text{trees}} = \text{std}(\{\hat{p}_t\}_{t=1}^T)$.
- In `train_pan_india.py`: ST-FloodNet Monte Carlo Dropout (30 stochastic forward passes with dropout active at test time):
  $$\bar{p} = \frac{1}{T} \sum_{t=1}^T \sigma(\hat{y}_t), \quad \sigma^2_{\text{epistemic}} = \frac{1}{T} \sum_{t=1}^T (\sigma(\hat{y}_t) - \bar{p})^2$$

---

## 2. Leakage Risks in Naive Hybrid Architectures

| Stacking Step | Severe Leakage Risk | Mitigation Strategy |
| :--- | :--- | :--- |
| **Meta-Learner Training** | Training the meta-learner on predictions generated by base models evaluated on the *same* training samples leads to extreme overfitting and overconfidence (base models overfit train labels; meta-learner learns to trust overfitted probabilities). | **Leakage-Safe Out-Of-Fold (OOF) Stacking:** Partition the training set (2018–2021) into $K$ cross-validation folds ($K=5$). For each fold $k$, train base models on $K-1$ folds and predict on the holdout fold $k$. Meta-learner is trained strictly on cross-fitted OOF representations. |
| **Spatial Correlation** | Random $K$-Fold cross-validation mixes samples from the same station/basin across train and validation folds, causing spatial data leakage. | **Grouped / Station-Aware K-Fold:** Group records by station ID or river basin so that no station's records appear in both fold-train and fold-validation simultaneously. |
| **Temporal Causality** | Using future observations to predict past events during stacking. | Maintain strict backward-looking antecedent features ($T-1$) and evaluate final models out-of-time (2022 validation, 2023 holdout test). |
| **Feature Duplication** | Blindly concatenating all 27 raw features with base model predictions in the meta-learner dilutes the meta-learning signal. | Feature selection for the fusion model: combine ST-FloodNet latent embedding ($\mathbf{z} \in \mathbb{R}^{16\text{ or }32}$ or PCA-reduced), ST-FloodNet probability, LightGBM probability, and key structured domain indicators (`rainfall_3d`, `soil_saturation_ratio`, `elevation_risk_factor`, `dist_river_km`). |

---

## 3. Proposed Hybrid Architecture: HF-STFloodNet (STF-LGBM)

```
                            RAW 27-FEATURE VECTOR
                                      │
                   ┌──────────────────┴──────────────────┐
                   ▼                                     ▼
        ST-FloodNet Branch                        LightGBM Branch
   (Multimodal Deep Representation)           (Tabular Histogram GBDT)
                   │                                     │
         ┌─────────┴─────────┐                           │
         ▼                   ▼                           ▼
Latent Embedding      Hazard Probability         Hazard Probability
 z in R^32 (or 16)         p_ST in [0, 1]              p_LGBM in [0, 1]
         │                   │                           │
         └─────────┬─────────┴───────────┬───────────────┘
                   │                     │
                   │   Selected Key Structured Indicators
                   │   (rainfall_3d, soil_sat, elev_risk, dist_riv)
                   ▼                     ▼
             ┌───────────────────────────────────┐
             │       HYBRID FUSION LAYER         │
             │   Leakage-Safe Stacking / GBDT    │
             │     or Calibrated Ridge Meta      │
             └─────────────────┬─────────────────┘
                               ▼
                    FINAL HYBRID PREDICTION
                 p_hybrid in [0, 1], Severity,
              Hybrid Epistemic MC Uncertainty
```

---

## 4. Exact Changes Required for Hybridization

1. **Modify `STFloodNet`:**
   - Update `STFloodNet` in `src/models/` to support `return_embedding=True` in `forward()`, returning `(hazard_logit, severity_logits, latent_rep)`.
   - Ensure complete 27-feature routing (route remaining SAR/optical and exposure indices into encoders).
2. **Implement `src/models/hybrid_flood_model.py`:**
   - Create class `HFSTFloodNet` / `HybridFloodModel`.
   - Include cross-fitting OOF generator (`generate_oof_predictions`).
   - Meta-learner fusion layer (evaluating both LightGBM Meta and Regularized Logistic/Calibrated Fusion).
   - Hybrid Monte Carlo Dropout inference for epistemic uncertainty.
3. **Create Training Pipeline `src/models/train_hybrid.py`:**
   - Reproducible script executing:
     1. Grouped station-aware $K$-Fold cross-fitting on 2018–2021 train set.
     2. Generation of OOF predictions and latent representations.
     3. Meta-learner training on OOF features.
     4. Final base model training on full train split.
     5. Out-of-time evaluation on 2022 validation and 2023 test sets.
     6. Spatial generalization evaluation on held-out Brahmaputra & Mahanadi basins.
     7. 5-Tier ablation study ($A_1$ to $A_5$).
     8. Saving models, metrics, and plots.
4. **Update `src/models/prediction_service.py`:**
   - Add model registry (`MODEL_REGISTRY = {"st_floodnet": ..., "lightgbm": ..., "random_forest": ..., "hf_stfloodnet": ...}`).
   - Support active model selection and backward compatibility.
5. **Update API & Dashboard:**
   - Expose model name, hybrid architecture, and uncertainty.
   - Cleanly integrate into existing 3-state Location Analysis and 6 Advanced Analysis sub-tabs.
6. **Comprehensive Automated Tests:**
   - Add unit tests for hybrid loading, OOF generation, probability bounds $[0, 1]$, uncertainty, and API compatibility.
