"""
HF-STFloodNet: Hybrid Fusion Spatiotemporal Flood Network (STF-LGBM)
Combines multimodal deep representation learning (ST-FloodNet) with structured tabular
gradient boosting (LightGBM) via leakage-safe cross-fitted stacking.

Architecture:
- Branch 1: ST-FloodNet (Multi-scale temporal CNN/Bi-GRU + Spatial Terrain + SAR Radar Encoder)
- Branch 2: LightGBM (Histogram Gradient Boosted Decision Trees on full 27 features)
- Latent Representation: 64-dim cross-modal embedding from ST-FloodNet pre-prediction bottleneck
- Fusion Layer: Regularized Meta-Learner trained on Out-Of-Fold (OOF) cross-fitted representations
- Epistemic Uncertainty: Propagated Monte Carlo Dropout through ST-FloodNet + Fusion Network
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.models.train_pan_india import STFloodNet

# Key domain indicators that physically govern flood runoff and watershed response
KEY_STRUCTURED_FEATURES = [
    "rainfall_3d",
    "rainfall_14d",
    "soil_saturation_ratio",
    "elevation_risk_factor",
    "dist_river_km",
    "drainage_vulnerability",
    "sar_vv"
]


class HFSTFloodNet:
    """
    Hybrid Fusion Spatiotemporal Flood Network (HF-STFloodNet).
    Orchestrates deep representation extraction, gradient boosting tabular inference,
    and meta-learned cross-modal fusion with rigorous uncertainty quantification.
    """
    def __init__(
        self,
        st_model: STFloodNet = None,
        lgb_model = None,
        fusion_model = None,
        pca_model = None,
        feature_cols = None,
        meta_features = None,
        threshold: float = 0.50,
        calibrator = None,
        model_name: str = "HF-STFloodNet"
    ):
        self.st_model = st_model
        self.lgb_model = lgb_model
        self.fusion_model = fusion_model
        self.pca_model = pca_model
        self.feature_cols = feature_cols or []
        self.meta_features = meta_features or []
        self.threshold = threshold
        self.calibrator = calibrator
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.st_model is not None:
            self.st_model.to(self.device)
            self.st_model.eval()

    def build_hybrid_features(self, X_scaled, x_seq=None, st_probs=None, st_embs=None, lgb_probs=None):
        """
        Construct the meta-feature matrix for the fusion layer:
        - ST-FloodNet hazard probability (using x_seq if available)
        - LightGBM hazard probability
        - Difference & interaction features
        - Projected latent multimodal embedding from ST-FloodNet
        - Key structured physical domain anchors
        """
        N = len(X_scaled)
        
        # 1. Base model probabilities if not precomputed
        if st_probs is None or st_embs is None:
            self.st_model.eval()
            with torch.no_grad():
                xt = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)
                x_seq_t = torch.tensor(x_seq, dtype=torch.float32, device=self.device) if x_seq is not None else None
                h_logit, _, rep = self.st_model(xt, x_seq=x_seq_t, return_embedding=True, mc_dropout=False)
                st_probs = torch.sigmoid(h_logit).cpu().numpy().flatten()
                st_embs = rep.cpu().numpy()

        if lgb_probs is None:
            lgb_probs = self.lgb_model.predict_proba(X_scaled)[:, 1]

        # 2. Probability interactions
        p_st = st_probs.reshape(-1, 1)
        p_lgb = lgb_probs.reshape(-1, 1)
        p_diff = p_st - p_lgb
        p_prod = p_st * p_lgb
        p_mean = 0.5 * (p_st + p_lgb)

        # 3. Projected / Reduced Latent Embedding
        if self.pca_model is not None:
            emb_features = self.pca_model.transform(st_embs)
        else:
            # If no PCA, use first 8 latent dimensions
            emb_features = st_embs[:, :8]

        # 4. Key Structured Physical Indicators
        feat_indices = [self.feature_cols.index(col) for col in KEY_STRUCTURED_FEATURES if col in self.feature_cols]
        if feat_indices:
            struct_features = X_scaled[:, feat_indices]
        else:
            struct_features = np.zeros((N, 0))

        # Concatenate into unified meta-feature array
        meta_X = np.hstack([p_st, p_lgb, p_diff, p_prod, p_mean, emb_features, struct_features])
        return meta_X

    def predict_proba(self, X_scaled, x_seq=None, calibrate: bool = True):
        """Compute hybrid flood hazard probability in [0, 1]."""
        meta_X = self.build_hybrid_features(X_scaled, x_seq=x_seq)
        if hasattr(self.fusion_model, "predict_proba"):
            probs = self.fusion_model.predict_proba(meta_X)[:, 1]
        elif hasattr(self.fusion_model, "predict"):
            probs = self.fusion_model.predict(meta_X)
            # If linear or regression, clip to [0, 1]
            probs = np.clip(probs, 0.0, 1.0)
        else:
            # Fallback blend
            x_seq_t = torch.tensor(x_seq, dtype=torch.float32, device=self.device) if x_seq is not None else None
            p_st = torch.sigmoid(self.st_model(torch.tensor(X_scaled, dtype=torch.float32).to(self.device), x_seq=x_seq_t)[0]).cpu().numpy().flatten()
            p_lgb = self.lgb_model.predict_proba(X_scaled)[:, 1]
            probs = 0.5 * p_st + 0.5 * p_lgb

        probs = np.clip(probs, 0.0, 1.0)
        if calibrate and self.calibrator is not None:
            if hasattr(self.calibrator, "predict_proba"):
                probs = self.calibrator.predict_proba(probs.reshape(-1, 1))[:, 1]
            elif hasattr(self.calibrator, "predict"):
                probs = self.calibrator.predict(probs.reshape(-1, 1))
            probs = np.clip(probs, 0.0, 1.0)

        return probs

    def predict(self, X_scaled, x_seq=None, threshold: float = None):
        """Binary flood prediction using calibrated decision threshold."""
        th = self.threshold if threshold is None else threshold
        probs = self.predict_proba(X_scaled, x_seq=x_seq)
        return (probs >= th).astype(int)

    def predict_severity(self, X_scaled, x_seq=None):
        """Classify multi-level flood severity (0=None, 1=Minor, 2=Moderate, 3=Severe)."""
        self.st_model.eval()
        with torch.no_grad():
            xt = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)
            x_seq_t = torch.tensor(x_seq, dtype=torch.float32, device=self.device) if x_seq is not None else None
            _, sev_logits, _ = self.st_model(xt, x_seq=x_seq_t, return_embedding=True, mc_dropout=False)
            sev_classes = torch.argmax(sev_logits, dim=1).cpu().numpy()
        return sev_classes

    def predict_with_uncertainty(self, X_scaled, x_seq=None, n_forward_passes: int = 30):
        """
        Scientifically grounded Hybrid Epistemic Uncertainty Estimation.
        Propagates Monte Carlo Dropout stochastic samples through the full stacked pipeline:
        For each pass t:
        1. Evaluates ST-FloodNet stochastic embedding z^(t) and probability p_ST^(t) under dropout
        2. Constructs hybrid meta-feature vector using z^(t) and p_ST^(t)
        3. Evaluates fusion model to generate hybrid probability p_hybrid^(t)
        4. Calculates epistemic variance, predictive entropy, and 95% credible intervals.
        """
        self.st_model.eval()
        xt = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)
        x_seq_t = torch.tensor(x_seq, dtype=torch.float32, device=self.device) if x_seq is not None else None
        lgb_probs = self.lgb_model.predict_proba(X_scaled)[:, 1]

        hybrid_probs = []
        with torch.no_grad():
            for _ in range(n_forward_passes):
                h_logit, _, rep = self.st_model(xt, x_seq=x_seq_t, return_embedding=True, mc_dropout=True)
                p_st_pass = torch.sigmoid(h_logit).cpu().numpy().flatten()
                rep_pass = rep.cpu().numpy()

                meta_x_pass = self.build_hybrid_features(
                    X_scaled, x_seq=x_seq, st_probs=p_st_pass, st_embs=rep_pass, lgb_probs=lgb_probs
                )
                if hasattr(self.fusion_model, "predict_proba"):
                    p_hyb_pass = self.fusion_model.predict_proba(meta_x_pass)[:, 1]
                else:
                    p_hyb_pass = np.clip(self.fusion_model.predict(meta_x_pass), 0.0, 1.0)

                hybrid_probs.append(p_hyb_pass)

        hybrid_probs = np.array(hybrid_probs)  # [T, N]
        mean_prob = np.mean(hybrid_probs, axis=0)
        epistemic_var = np.var(hybrid_probs, axis=0)
        epistemic_std = np.sqrt(np.maximum(epistemic_var, 0.0))

        # Predictive Entropy
        p_safe = np.clip(mean_prob, 1e-6, 1.0 - 1e-6)
        entropy = -(p_safe * np.log2(p_safe) + (1.0 - p_safe) * np.log2(1.0 - p_safe))

        # 95% Credible Interval
        ci_lower = np.percentile(hybrid_probs, 2.5, axis=0)
        ci_upper = np.percentile(hybrid_probs, 97.5, axis=0)

        uncertainty_pct = np.clip(epistemic_std * 100.0, 2.0, 45.0)

        return {
            "mean_probability": mean_prob,
            "epistemic_variance": epistemic_var,
            "epistemic_std": epistemic_std,
            "predictive_entropy": entropy,
            "uncertainty_pct": uncertainty_pct,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper
        }

    def save(self, output_dir: str, fusion_filename: str = "hybrid_fusion_model.joblib", st_weights_filename: str = "model_pan_india_st_floodnet.pt"):
        """Save hybrid pipeline artifacts cleanly to disk."""
        os.makedirs(output_dir, exist_ok=True)
        # 1. Base ST-FloodNet weights
        torch.save(self.st_model.state_dict(), os.path.join(output_dir, st_weights_filename))
        # 2. Base LightGBM model
        joblib.dump(self.lgb_model, os.path.join(output_dir, "model_pan_india_lightgbm.joblib"))
        # 3. Fusion Meta-Learner & Preprocessors
        fusion_artifact = {
            "fusion_model": self.fusion_model,
            "pca_model": self.pca_model,
            "feature_cols": self.feature_cols,
            "key_structured_features": KEY_STRUCTURED_FEATURES,
            "meta_features": self.meta_features,
            "threshold": self.threshold,
            "calibrator": self.calibrator,
            "model_name": self.model_name,
            "st_weights_filename": st_weights_filename
        }
        joblib.dump(fusion_artifact, os.path.join(output_dir, fusion_filename))
        print(f"HF-STFloodNet hybrid model saved successfully to {os.path.join(output_dir, fusion_filename)}")

    @classmethod
    def load(cls, models_dir: str, fusion_filename: str = "hybrid_fusion_model.joblib", st_weights_filename: str = None, device="cpu"):
        """Load trained HF-STFloodNet hybrid pipeline from disk."""
        fusion_path = os.path.join(models_dir, fusion_filename)
        assert os.path.exists(fusion_path), f"Fusion model missing at {fusion_path}"
        fusion_artifact = joblib.load(fusion_path)
        feature_cols = fusion_artifact["feature_cols"]

        if st_weights_filename is None:
            st_weights_filename = fusion_artifact.get("st_weights_filename", "model_pan_india_st_floodnet.pt")
        st_path = os.path.join(models_dir, st_weights_filename)
        lgb_path = os.path.join(models_dir, "model_pan_india_lightgbm.joblib")

        assert os.path.exists(st_path), f"ST-FloodNet weights missing at {st_path}"
        assert os.path.exists(lgb_path), f"LightGBM model missing at {lgb_path}"

        st_model = STFloodNet(num_features=len(feature_cols))
        st_model.load_state_dict(torch.load(st_path, map_location=device))
        st_model.to(device)
        st_model.eval()

        lgb_model = joblib.load(lgb_path)

        return cls(
            st_model=st_model,
            lgb_model=lgb_model,
            fusion_model=fusion_artifact["fusion_model"],
            pca_model=fusion_artifact.get("pca_model"),
            feature_cols=feature_cols,
            meta_features=fusion_artifact.get("meta_features", []),
            threshold=fusion_artifact.get("threshold", 0.50),
            calibrator=fusion_artifact.get("calibrator", None),
            model_name=fusion_artifact.get("model_name", "HF-STFloodNet")
        )


# Alias for research paper naming convention
STFLGBM = HFSTFloodNet
HybridFloodModel = HFSTFloodNet
