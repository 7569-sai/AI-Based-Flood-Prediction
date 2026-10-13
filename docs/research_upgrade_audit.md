# Research-Grade Upgrade Audit & Gap Analysis

**Project Title:** AI-Based Flood Prediction, Dynamic Risk Mapping and Resilience Planning  
**Application:** FloodAI Engine — "AI for a Safer, Resilient India"  
**Audit Date:** September 2026  
**Document Status:** Approved Research Protocol  

---

## 1. Executive Summary

This document presents a comprehensive scientific audit of the FloodAI Engine codebase, detailing the architectural, methodological, and experimental gaps identified in previous development iterations, alongside the concrete specifications for the research-grade upgrade.

While earlier iterations demonstrated an operational end-to-end framework—integrating multi-source geospatial data, a deep neural network (ST-FloodNet), gradient-boosted decision trees (LightGBM), a FastAPI backend, and an interactive Location-First Streamlit dashboard—a rigorous scientific examination identified several critical technical and methodological vulnerabilities. This upgrade addresses each vulnerability systematically, ensuring empirical integrity, zero data leakage, and peer-review scientific validity.

---

## 2. Identified Vulnerabilities & Remediation Plan

### 2.1 Feature Routing Deficit in Deep Learning Branch
- **Identified Flaw:** In the baseline `STFloodNet` implementation, input tensors were hardcoded to route only 21 features (`x_meteo`: indices 0–7, `x_terrain`: indices 8–15, `x_sar`: indices 16–20). Features 21 through 26 (`sar_cross_ratio`, `ndwi_proxy`, `ndvi_proxy`, `population_density`, `exposure_index`, `vulnerability_index`) were completely bypassed during forward propagation. Furthermore, hydrological indicators (`soil_moisture`, `soil_saturation_ratio`, `river_discharge_proxy`, `river_stage_anomaly`) were inappropriately routed into the spatial terrain encoder rather than hydrometeorological modeling.
- **Scientific Consequence:** The deep learning branch was deprived of critical surface water spectral indicators (NDWI), canopy moisture (NDVI), SAR polarimetric cross-ratio, and demographic vulnerability, impairing the latent representation fed to the hybrid meta-learner.
- **Remediation Specification:** Redesign feature routing into four explicit physical modules covering all 27 features:
  1. *Hydrometeorological & Temporal Dynamic Encoder* (13 features: `rainfall_1d`, `rainfall_3d`, `rainfall_7d`, `rainfall_14d`, `rainfall_anomaly_30d`, `api_14d`, `temp_mean_3d`, `humidity_mean_3d`, `soil_moisture_root_zone`, `soil_moisture_deep`, `soil_saturation_ratio`, `river_discharge_proxy`, `river_stage_anomaly`).
  2. *Topographic & Terrain Encoder* (6 features: `elevation_m`, `slope_deg`, `dist_river_km`, `twi_proxy`, `elevation_risk_factor`, `drainage_vulnerability`).
  3. *Remote Sensing Earth Observation Encoder* (5 features: `sar_vv`, `sar_vh`, `sar_cross_ratio`, `ndwi_proxy`, `ndvi_proxy`).
  4. *Socioeconomic Exposure & Vulnerability Encoder* (3 features: `population_density`, `exposure_index`, `vulnerability_index`).
  - Strict assertion: Zero features are bypassed during inference.

### 2.2 Temporal Modeling: Sequence Tensor Integration
- **Identified Flaw:** The initial temporal convolution operated over engineered antecedent rainfall features (1d, 3d, 7d, 14d) flattened into a 1D vector rather than processing a chronological sequence of daily states $[B, T, F]$.
- **Scientific Consequence:** Intra-week temporal hydrographs, lag dynamics, and the non-linear inflection of soil saturation preceding convective storm events were represented solely via static summary statistics.
- **Remediation Specification:** Implement a dual-mode input pipeline:
  1. *True Sliding-Window Sequence Generator:* For batch training, evaluation, and time-series replay, generate tensors of shape $[B, T, F_{\text{temp}}]$ where $T \in \{7, 14, 30\}$ days of daily hydrometeorological observations per station.
  2. *Multi-Scale 1D Conv $\to$ Bidirectional GRU $\to$ Temporal Attention:* Convolve across temporal scales (3-day flash vs 7-day riverine windows), capture recurrent memory via Bi-GRU, and apply a temporal attention mechanism to automatically identify critical precursor storm days.
  3. *Static Modality Fusion:* Concatenate the attended temporal representation ($d=64$) with terrain embeddings ($d=32$), radar/optical embeddings ($d=16$), and exposure embeddings ($d=16$) prior to gated cross-modal fusion.
  4. *2D Backward Compatibility:* Retain a mathematically sound 2D tensor fallback for real-time single-point dashboard queries when multi-day antecedent historical sequences are synthesized from recent API lags.

### 2.3 Stacking Leakage & Meta-Feature Calibration
- **Identified Flaw:** Early hybrid prototypes risk target leakage if base model out-of-fold predictions share station observations across folds, or if decision thresholds and probability calibration curves are fitted on the test partition.
- **Scientific Consequence:** Artificially inflated test metrics, optimistic confidence intervals, and poor spatial transferability.
- **Remediation Specification:**
  - Implement strict 5-fold `GroupKFold` split grouped by station ID during training (2018–2021). Base model predictions for training the meta-learner must be strictly out-of-fold (OOF).
  - Validation Split (2022) is dedicated exclusively to:
    1. Decision threshold optimization ($\theta \in [0.05, 0.95]$ maximizing $F_1$ or cost-sensitive disaster response utility).
    2. Probability calibration via Platt Scaling (Logistic Sigmoidal fitting) and Isotonic Regression.
  - Test Split (2023) is held out untouched until inference. The optimal threshold $\theta^*$ and calibration mapping $f_{\text{calib}}$ are applied frozen to test data.

### 2.4 Empirical Honesty & Model Superiority Policy
- **Identified Flaw:** Tendency to assume or claim the hybrid model unconditionally outperforms single models across all benchmark metrics.
- **Scientific Consequence:** Violates peer-review empirical standards. In tabular hydrological datasets with high class imbalance, gradient boosting (LightGBM) frequently achieves higher raw precision, while deep multimodal representations often yield superior cross-basin spatial generalization and recall under covariate shift.
- **Remediation Specification:** Enforce complete reporting transparency. Present all metrics (Accuracy, Precision, Recall, $F_1$, ROC-AUC, PR-AUC, Brier Score, ECE) as measured experimentally. Document where LightGBM excels (e.g. in-basin precision) and where the HF-STFloodNet hybrid excels (e.g. spatial generalization, recall, cross-basin transfer, and epistemic uncertainty quantification).

### 2.5 Evaluation Scope Expansion
- **Identified Flaw:** Primary evaluations were restricted to temporal test (2023) and 2 regional holdouts (Brahmaputra, Mahanadi).
- **Remediation Specification:**
  1. *Comprehensive Leave-Basin-Out Validation:* Evaluate across $\ge 6$ distinct major river basins (Brahmaputra, Mahanadi, Ganga, Godavari, Krishna, Kaveri) to quantify geographic transferability across Himalayan, Gangetic, Peninsular, and Coastal hydrological regimes.
  2. *Event-Level Performance Metrics:* Beyond daily sample-level classification, evaluate continuous flood events measuring Event Detection Delay (days), Forecast Lead Time (days), and Event False Alarm Rate (FAR).
  3. *Forecast Horizon Degradation:* Evaluate multi-day lookaheads ($T+1, T+3, T+7$ days) to characterize the operational decay of predictive skill.
  4. *Missing-Data Sensor Robustness:* Quantify model degradation under realistic telemetry failure modes (5%, 10%, 20% random feature dropouts; total SAR blackout; rainfall gauge failure; soil moisture sensor offline).
  5. *Ablation Matrix:* Perform a 10-tier ablation dissecting the marginal contribution of temporal attention, Bi-GRU, SAR radar, optical indices, terrain encoders, and meta-learner stacking.

---

## 3. Implementation Verification Checklist

| Phase | Milestone | Primary Artifact | Target Status |
| :--- | :--- | :--- | :--- |
| **Phase 1** | Gap Analysis & Architecture Audit | `docs/research_upgrade_audit.md` | Completed |
| **Phase 2** | Data Provenance & Ground Truth Audit | `docs/data_provenance_audit.md` | In Progress |
| **Phase 3** | Flood Label Methodology Audit | `docs/flood_label_methodology.md` | In Progress |
| **Phase 4** | Leakage-Free Stacking Protocol | `docs/hybrid_leakage_audit.md` | In Progress |
| **Phase 5** | 3D Sequence Tensor Pipeline | `src/data/sequence_generator.py` | In Progress |
| **Phase 6** | 27-Feature Multimodal ST-FloodNet | `src/models/train_pan_india.py` | In Progress |
| **Phase 7** | HF-STFloodNet Calibration & Thresholding | `src/models/hybrid_flood_model.py` | In Progress |
| **Phase 8** | Multi-Basin Spatial Generalization | `outputs/metrics/cross_basin_results.csv` | In Progress |
| **Phase 9** | Event-Level Evaluation | `outputs/metrics/event_level_metrics.csv` | In Progress |
| **Phase 10** | Multi-Horizon Forecast Analysis | `outputs/metrics/forecast_horizon_results.csv` | In Progress |
| **Phase 11** | Sensor Missing-Data Robustness | `outputs/metrics/robustness_results.csv` | In Progress |
| **Phase 12** | 10-Tier Ablation Study | `outputs/metrics/ablation_results.csv` | In Progress |
| **Phase 13** | Model Registry & Data Manifest | `models/model_registry.json`, `data/DATA_MANIFEST.json` | In Progress |
| **Phase 14** | Test Suite Validation (100% Pass) | `pytest -q` | Pending |
