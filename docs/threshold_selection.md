# Decision Threshold Selection & Probability Calibration Report

**Optimization Partition:** Calendar Year 2022 (`data/processed/india/val.parquet`, N=13,140)
**Validation Positive Prevalence:** 1.80%
**Optimal Threshold Selected:** $\theta^* = 0.51$
**Validation F1 at $\theta^*$:** 0.7491

## Calibration Comparison (Validation Set)

| Method | Brier Score | Expected Calibration Error (ECE) | Log Loss |
| :--- | :---: | :---: | :---: |
| Raw Uncalibrated | 0.0146 | 0.0167 | 0.0484 |
| Platt Scaling (Sigmoidal) | 0.0076 | 0.0056 | 0.0230 |
| Isotonic Regression | 0.0059 | 0.0000 | 0.0173 |

Platt Scaling was chosen as the operational calibrator to prevent step-discontinuities and preserve monotonic differentiability.
