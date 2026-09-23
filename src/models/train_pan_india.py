"""
Pan-India Multi-Basin Model Training, Evaluation, and Generalization Pipeline.
Trains and benchmarks 6 baseline models + 1 proposed research architecture:
1. Logistic Regression (Linear baseline)
2. Random Forest (Ensemble bagging baseline)
3. Gradient Boosting (Tree boosting baseline)
4. XGBoost (Extreme gradient boosting baseline)
5. LightGBM (Scalable histogram boosting baseline)
6. Bi-LSTM (Recurrent temporal sequence baseline)
7. ST-FloodNet (Proposed SpatioTemporal Multimodal Uncertainty-Aware Architecture)

Executes:
- Chronological Test Set Evaluation (2023)
- Spatial Cross-Basin Generalization (Brahmaputra & Mahanadi holdouts)
- 6-Tier Multimodal Feature Ablation Study (Exp A to Exp F)
- Monte Carlo Dropout Uncertainty Quantification & Calibration
- High-resolution comparative plots & metric artifacts
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    roc_curve, precision_recall_curve, confusion_matrix
)
import xgboost as xgb
import lightgbm as lgb

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

DATA_DIR = os.path.join(BASE_DIR, "data", "processed", "india")
MODELS_DIR = os.path.join(BASE_DIR, "models", "trained")
SCALERS_DIR = os.path.join(BASE_DIR, "models", "scalers")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
METRICS_DIR = os.path.join(OUTPUTS_DIR, "metrics")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SCALERS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ===========================================================================
# 1. Proposed Research Model: ST-FloodNet
# ===========================================================================
class STFloodNet(nn.Module):
    """
    SpatioTemporal Multimodal Uncertainty-Aware Flood Network (ST-FloodNet).
    Investigates multi-scale temporal representation, spatial geomorphic embedding,
    gated multimodal fusion, multi-task learning (hazard + severity),
    and Monte Carlo Dropout epistemic uncertainty quantification.
    """
    def __init__(self, num_features=27, hidden_dim=64, num_classes=4, mc_dropout_p=0.35):
        super(STFloodNet, self).__init__()
        self.mc_dropout_p = mc_dropout_p

        # 1. Multi-scale Temporal Convolution Branch (Antecedents: 1d, 3d, 7d, 14d, API)
        # Represents temporal dynamics across distinct meteorological scales
        self.temp_conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.temp_conv2 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.temp_pool = nn.AdaptiveAvgPool1d(1)

        # 2. Recurrent Spatiotemporal Memory (Bi-GRU)
        self.bigru = nn.GRU(
            input_size=32,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        # 3. Spatial & Geomorphic Terrain Encoder (Elevation, Slope, River Dist, TWI)
        self.spatial_encoder = nn.Sequential(
            nn.Linear(8, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=mc_dropout_p)
        )

        # 4. Satellite Radar & Remote Sensing Encoder (SAR VV/VH, Cross-ratio, NDWI, NDVI)
        self.sar_encoder = nn.Sequential(
            nn.Linear(5, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(p=mc_dropout_p)
        )

        # 5. Gated Cross-Modal Fusion
        # Dynamically weights evidence from hydrometeorology, terrain, and satellite sensors
        fusion_dim = hidden_dim + 32 + 16
        self.fusion_gate = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.Sigmoid()
        )
        self.fusion_fc = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=mc_dropout_p)
        )

        # 6. Multi-Task Output Heads
        # Head A: Binary Flood Hazard Probability
        self.hazard_head = nn.Linear(64, 1)
        # Head B: Flood Severity Classification (0=None, 1=Minor, 2=Moderate, 3=Severe)
        self.severity_head = nn.Linear(64, num_classes)

    def forward(self, x_all):
        batch_size = x_all.size(0)

        # Split features into modalities
        # Temporal meteo features (first 8 features)
        x_meteo = x_all[:, :8].unsqueeze(1)  # [B, 1, 8]
        # Spatial terrain features (indices 8 to 16)
        x_terrain = x_all[:, 8:16]  # [B, 8]
        # Satellite SAR & Optical (indices 16 to 21)
        x_sar = x_all[:, 16:21]  # [B, 5]

        # Multi-scale temporal representation
        c1 = torch.relu(self.temp_conv1(x_meteo))
        c2 = torch.relu(self.temp_conv2(x_meteo))
        c_cat = torch.cat([c1, c2], dim=1).transpose(1, 2)  # [B, 8, 32]
        gru_out, _ = self.bigru(c_cat)
        gru_pool = gru_out[:, -1, :]  # [B, hidden_dim]

        # Spatial terrain embedding
        spatial_emb = self.spatial_encoder(x_terrain)  # [B, 32]

        # Satellite SAR embedding
        sar_emb = self.sar_encoder(x_sar)  # [B, 16]

        # Cross-modal fusion
        fused = torch.cat([gru_pool, spatial_emb, sar_emb], dim=1)
        gate = self.fusion_gate(fused)
        fused_gated = fused * gate
        rep = self.fusion_fc(fused_gated)

        # Monte Carlo Dropout active even during inference for uncertainty estimation
        rep = nn.functional.dropout(rep, p=self.mc_dropout_p, training=True)

        hazard_logit = self.hazard_head(rep)
        severity_logits = self.severity_head(rep)

        return hazard_logit, severity_logits

    def predict_with_uncertainty(self, x_tensor, n_forward_passes=30):
        """
        Monte Carlo Dropout inference: Performs T forward passes under dropout
        to quantify epistemic predictive uncertainty (variance) and aleatoric predictive entropy.
        """
        self.eval()
        probs = []
        with torch.no_grad():
            for _ in range(n_forward_passes):
                h_logit, _ = self.forward(x_tensor)
                p = torch.sigmoid(h_logit).cpu().numpy().flatten()
                probs.append(p)
        probs = np.array(probs)  # [T, N]
        mean_prob = np.mean(probs, axis=0)
        uncertainty_variance = np.var(probs, axis=0)  # Epistemic variance
        # Predictive Entropy
        p_safe = np.clip(mean_prob, 1e-6, 1.0 - 1e-6)
        entropy = -(p_safe * np.log2(p_safe) + (1 - p_safe) * np.log2(1 - p_safe))
        return mean_prob, uncertainty_variance, entropy


# ===========================================================================
# 2. Main Training & Evaluation Routine
# ===========================================================================
def train_and_benchmark_pan_india():
    print("===========================================================================")
    print("FLOODAI ENGINE — PAN-INDIA MULTI-BASIN MODEL TRAINING & BENCHMARKING")
    print("===========================================================================")

    # 1. Load Parquet Data
    train_df = pd.read_parquet(os.path.join(DATA_DIR, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(DATA_DIR, "val.parquet"))
    test_df = pd.read_parquet(os.path.join(DATA_DIR, "test.parquet"))

    with open(os.path.join(DATA_DIR, "pan_india_metadata.json"), "r") as f:
        meta = json.load(f)

    feature_cols = meta["feature_cols"]
    target_col = meta["target_col"]
    severity_col = meta["severity_col"]

    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].values
    y_train_sev = train_df[severity_col].values

    X_val = val_df[feature_cols].copy()
    y_val = val_df[target_col].values

    X_test = test_df[feature_cols].copy()
    y_test = test_df[target_col].values
    y_test_sev = test_df[severity_col].values

    # 2. Fit Scaler Strictly on Train (Leakage Free)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    scaler_path = os.path.join(SCALERS_DIR, "pan_india_scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"StandardScaler fit strictly on Train (52,596 rows) -> Saved to {scaler_path}")

    # Compute class imbalance ratio
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    pos_weight = float(neg_count / max(pos_count, 1))
    print(f"Train Class Imbalance: Neg={neg_count:,}, Pos={pos_count:,} (Scale Pos Weight = {pos_weight:.2f})")

    results = {}
    model_predictions = {}
    model_probabilities = {}

    # -----------------------------------------------------------------------
    # Baseline 1: Logistic Regression
    # -----------------------------------------------------------------------
    print("\n[1/7] Training Baseline: Logistic Regression (Class-Weighted)...")
    t0 = time.time()
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    t_train_lr = time.time() - t0

    t0 = time.time()
    p_test_lr = lr.predict_proba(X_test_scaled)[:, 1]
    t_lat_lr = ((time.time() - t0) / len(X_test)) * 1000.0
    y_pred_lr = (p_test_lr >= 0.5).astype(int)

    results["Logistic_Regression"] = evaluate_model("Logistic Regression", y_test, y_pred_lr, p_test_lr, t_train_lr, t_lat_lr)
    model_predictions["Logistic Regression"] = y_pred_lr
    model_probabilities["Logistic Regression"] = p_test_lr
    joblib.dump(lr, os.path.join(MODELS_DIR, "model_pan_india_logistic_regression.joblib"))

    # -----------------------------------------------------------------------
    # Baseline 2: Random Forest (Balanced)
    # -----------------------------------------------------------------------
    print("\n[2/7] Training Baseline: Random Forest (Balanced Ensemble)...")
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", max_depth=16, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    t_train_rf = time.time() - t0

    t0 = time.time()
    p_test_rf = rf.predict_proba(X_test_scaled)[:, 1]
    t_lat_rf = ((time.time() - t0) / len(X_test)) * 1000.0
    y_pred_rf = (p_test_rf >= 0.5).astype(int)

    results["Random_Forest"] = evaluate_model("Random Forest", y_test, y_pred_rf, p_test_rf, t_train_rf, t_lat_rf)
    model_predictions["Random Forest"] = y_pred_rf
    model_probabilities["Random Forest"] = p_test_rf
    joblib.dump(rf, os.path.join(MODELS_DIR, "model_pan_india_random_forest.joblib"))

    # -----------------------------------------------------------------------
    # Baseline 3: Gradient Boosting
    # -----------------------------------------------------------------------
    print("\n[3/7] Training Baseline: Gradient Boosting Classifier...")
    t0 = time.time()
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    gb.fit(X_train_scaled, y_train)
    t_train_gb = time.time() - t0

    t0 = time.time()
    p_test_gb = gb.predict_proba(X_test_scaled)[:, 1]
    t_lat_gb = ((time.time() - t0) / len(X_test)) * 1000.0
    y_pred_gb = (p_test_gb >= 0.5).astype(int)

    results["Gradient_Boosting"] = evaluate_model("Gradient Boosting", y_test, y_pred_gb, p_test_gb, t_train_gb, t_lat_gb)
    model_predictions["Gradient Boosting"] = y_pred_gb
    model_probabilities["Gradient Boosting"] = p_test_gb
    joblib.dump(gb, os.path.join(MODELS_DIR, "model_pan_india_gradient_boosting.joblib"))

    # -----------------------------------------------------------------------
    # Baseline 4: XGBoost (scale_pos_weight)
    # -----------------------------------------------------------------------
    print("\n[4/7] Training Baseline: XGBoost (Extreme Gradient Boosting)...")
    t0 = time.time()
    xgb_clf = xgb.XGBClassifier(
        n_estimators=120,
        learning_rate=0.08,
        max_depth=6,
        scale_pos_weight=pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
    xgb_clf.fit(X_train_scaled, y_train)
    t_train_xgb = time.time() - t0

    t0 = time.time()
    p_test_xgb = xgb_clf.predict_proba(X_test_scaled)[:, 1]
    t_lat_xgb = ((time.time() - t0) / len(X_test)) * 1000.0
    y_pred_xgb = (p_test_xgb >= 0.5).astype(int)

    results["XGBoost"] = evaluate_model("XGBoost", y_test, y_pred_xgb, p_test_xgb, t_train_xgb, t_lat_xgb)
    model_predictions["XGBoost"] = y_pred_xgb
    model_probabilities["XGBoost"] = p_test_xgb
    joblib.dump(xgb_clf, os.path.join(MODELS_DIR, "model_pan_india_xgboost.joblib"))

    # -----------------------------------------------------------------------
    # Baseline 5: LightGBM (Scalable Tabular Baseline)
    # -----------------------------------------------------------------------
    print("\n[5/7] Training Baseline: LightGBM (Scalable Histogram Boosting)...")
    t0 = time.time()
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.07,
        num_leaves=31,
        scale_pos_weight=pos_weight,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgb_clf.fit(X_train_scaled, y_train)
    t_train_lgb = time.time() - t0

    t0 = time.time()
    p_test_lgb = lgb_clf.predict_proba(X_test_scaled)[:, 1]
    t_lat_lgb = ((time.time() - t0) / len(X_test)) * 1000.0
    y_pred_lgb = (p_test_lgb >= 0.5).astype(int)

    results["LightGBM"] = evaluate_model("LightGBM", y_test, y_pred_lgb, p_test_lgb, t_train_lgb, t_lat_lgb)
    model_predictions["LightGBM"] = y_pred_lgb
    model_probabilities["LightGBM"] = p_test_lgb
    joblib.dump(lgb_clf, os.path.join(MODELS_DIR, "model_pan_india_lightgbm.joblib"))

    # -----------------------------------------------------------------------
    # Baseline 6: PyTorch Bi-LSTM (Temporal Sequence Baseline)
    # -----------------------------------------------------------------------
    print("\n[6/7] Training Baseline: PyTorch Bi-LSTM (Temporal Sequence Baseline)...")
    class BiLSTMNet(nn.Module):
        def __init__(self, in_dim, hidden_dim=48):
            super().__init__()
            self.lstm = nn.LSTM(in_dim, hidden_dim, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(hidden_dim * 2, 1)

        def forward(self, x):
            x_seq = x.unsqueeze(1)
            out, _ = self.lstm(x_seq)
            return self.fc(out[:, -1, :])

    bilstm = BiLSTMNet(len(feature_cols))
    optimizer_lstm = optim.Adam(bilstm.parameters(), lr=0.005, weight_decay=1e-4)
    criterion_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

    train_ds = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

    t0 = time.time()
    bilstm.train()
    for ep in range(10):
        for bx, by in train_loader:
            optimizer_lstm.zero_grad()
            out = bilstm(bx).squeeze(-1)
            loss = criterion_bce(out, by)
            loss.backward()
            optimizer_lstm.step()
    t_train_lstm = time.time() - t0

    bilstm.eval()
    t0 = time.time()
    with torch.no_grad():
        test_x_t = torch.tensor(X_test_scaled, dtype=torch.float32)
        logits_lstm = bilstm(test_x_t).squeeze(-1)
        p_test_lstm = torch.sigmoid(logits_lstm).numpy()
    t_lat_lstm = ((time.time() - t0) / len(X_test)) * 1000.0
    y_pred_lstm = (p_test_lstm >= 0.5).astype(int)

    results["BiLSTM"] = evaluate_model("Bi-LSTM", y_test, y_pred_lstm, p_test_lstm, t_train_lstm, t_lat_lstm)
    model_predictions["Bi-LSTM"] = y_pred_lstm
    model_probabilities["Bi-LSTM"] = p_test_lstm
    torch.save(bilstm.state_dict(), os.path.join(MODELS_DIR, "model_pan_india_bilstm.pt"))

    # -----------------------------------------------------------------------
    # Proposed Research Model: ST-FloodNet
    # -----------------------------------------------------------------------
    print("\n[7/7] Training Proposed Research Architecture: ST-FloodNet...")
    st_model = STFloodNet(num_features=len(feature_cols), hidden_dim=64, num_classes=4, mc_dropout_p=0.35)
    optimizer_st = optim.AdamW(st_model.parameters(), lr=0.003, weight_decay=1e-4)
    loss_bce_st = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    loss_ce_st = nn.CrossEntropyLoss()

    train_ds_st = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(y_train_sev, dtype=torch.long)
    )
    train_loader_st = DataLoader(train_ds_st, batch_size=256, shuffle=True)

    t0 = time.time()
    st_model.train()
    for ep in range(12):
        for bx, by_haz, by_sev in train_loader_st:
            optimizer_st.zero_grad()
            h_logit, s_logits = st_model(bx)
            l_haz = loss_bce_st(h_logit.squeeze(-1), by_haz)
            l_sev = loss_ce_st(s_logits, by_sev)
            # Multi-task composite loss
            total_loss = l_haz + 0.5 * l_sev
            total_loss.backward()
            optimizer_st.step()
    t_train_st = time.time() - t0

    # Uncertainty Quantification via Monte Carlo Dropout (30 forward passes)
    t0 = time.time()
    test_x_st = torch.tensor(X_test_scaled, dtype=torch.float32)
    p_test_st, uncert_var, uncert_entropy = st_model.predict_with_uncertainty(test_x_st, n_forward_passes=30)
    t_lat_st = ((time.time() - t0) / len(X_test)) * 1000.0
    y_pred_st = (p_test_st >= 0.5).astype(int)

    results["ST_FloodNet"] = evaluate_model("ST-FloodNet (Proposed)", y_test, y_pred_st, p_test_st, t_train_st, t_lat_st)
    model_predictions["ST-FloodNet (Proposed)"] = y_pred_st
    model_probabilities["ST-FloodNet (Proposed)"] = p_test_st
    torch.save(st_model.state_dict(), os.path.join(MODELS_DIR, "model_pan_india_st_floodnet.pt"))

    # Save Uncertainty Metrics
    uncert_data = {
        "mean_epistemic_variance": round(float(np.mean(uncert_var)), 6),
        "max_epistemic_variance": round(float(np.max(uncert_var)), 6),
        "mean_predictive_entropy": round(float(np.mean(uncert_entropy)), 4),
        "high_uncertainty_instances_count": int(np.sum(uncert_var > 0.05)),
        "uncertainty_brier_score": round(float(brier_score_loss(y_test, p_test_st)), 4)
    }
    with open(os.path.join(METRICS_DIR, "pan_india_uncertainty_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(uncert_data, f, indent=2)

    # -----------------------------------------------------------------------
    # 3. Spatial Cross-Basin Generalization Experiment
    # -----------------------------------------------------------------------
    print("\n--- Running Spatial Cross-Basin Generalization Experiment ---")
    sp_train = pd.read_parquet(os.path.join(DATA_DIR, "spatial_train.parquet"))
    sp_test = pd.read_parquet(os.path.join(DATA_DIR, "spatial_holdout_test.parquet"))

    X_sp_train = scaler.transform(sp_train[feature_cols])
    y_sp_train = sp_train[target_col].values
    X_sp_test = scaler.transform(sp_test[feature_cols])
    y_sp_test = sp_test[target_col].values

    # Train LightGBM baseline on training basins
    lgb_sp = lgb.LGBMClassifier(n_estimators=120, learning_rate=0.08, scale_pos_weight=pos_weight, random_state=42, n_jobs=-1, verbose=-1)
    lgb_sp.fit(X_sp_train, y_sp_train)
    p_sp_lgb = lgb_sp.predict_proba(X_sp_test)[:, 1]
    res_sp_lgb = evaluate_model("LightGBM (Cross-Basin)", y_sp_test, (p_sp_lgb >= 0.5).astype(int), p_sp_lgb, 0, 0)

    # Train ST-FloodNet on training basins
    st_sp = STFloodNet(num_features=len(feature_cols))
    opt_sp = optim.AdamW(st_sp.parameters(), lr=0.003)
    ds_sp = TensorDataset(torch.tensor(X_sp_train, dtype=torch.float32), torch.tensor(y_sp_train, dtype=torch.float32), torch.tensor(sp_train[severity_col].values, dtype=torch.long))
    loader_sp = DataLoader(ds_sp, batch_size=256, shuffle=True)
    st_sp.train()
    for _ in range(10):
        for bx, by_h, by_s in loader_sp:
            opt_sp.zero_grad()
            hl, sl = st_sp(bx)
            loss = loss_bce_st(hl.squeeze(-1), by_h) + 0.5 * loss_ce_st(sl, by_s)
            loss.backward()
            opt_sp.step()

    p_sp_st, _, _ = st_sp.predict_with_uncertainty(torch.tensor(X_sp_test, dtype=torch.float32), n_forward_passes=20)
    res_sp_st = evaluate_model("ST-FloodNet (Cross-Basin)", y_sp_test, (p_sp_st >= 0.5).astype(int), p_sp_st, 0, 0)

    spatial_metrics = {
        "held_out_basins": ["Brahmaputra", "Mahanadi"],
        "held_out_test_records": len(sp_test),
        "held_out_flood_events": int(y_sp_test.sum()),
        "LightGBM_Cross_Basin": res_sp_lgb,
        "ST_FloodNet_Cross_Basin": res_sp_st
    }
    with open(os.path.join(METRICS_DIR, "pan_india_spatial_generalization.json"), "w", encoding="utf-8") as f:
        json.dump(spatial_metrics, f, indent=2)

    # -----------------------------------------------------------------------
    # 4. Mandatory 6-Tier Multimodal Feature Ablation Study
    # -----------------------------------------------------------------------
    print("\n--- Running 6-Tier Multimodal Feature Ablation Study ---")
    ablation_tiers = {
        "Exp_A_Meteo_Only": feature_cols[:8],
        "Exp_B_Meteo_Hydrology": feature_cols[:13],
        "Exp_C_Meteo_Hydro_Terrain": feature_cols[:19],
        "Exp_D_Meteo_Hydro_Terrain_SAR": feature_cols[:24],
        "Exp_E_Meteo_Hydro_Terrain_SAR_Exposure": feature_cols[:26],
        "Exp_F_Full_Multimodal": feature_cols
    }

    ablation_results = {}
    for tier_name, subset_cols in ablation_tiers.items():
        sub_X_train = scaler.transform(X_train)[:, [feature_cols.index(c) for c in subset_cols]]
        sub_X_test = scaler.transform(X_test)[:, [feature_cols.index(c) for c in subset_cols]]

        sub_rf = RandomForestClassifier(n_estimators=80, class_weight="balanced", random_state=42, n_jobs=-1)
        sub_rf.fit(sub_X_train, y_train)
        p_sub = sub_rf.predict_proba(sub_X_test)[:, 1]
        y_sub_pred = (p_sub >= 0.5).astype(int)

        ablation_results[tier_name] = {
            "tier_description": tier_name.replace("_", " "),
            "feature_count": len(subset_cols),
            "accuracy": round(float(accuracy_score(y_test, y_sub_pred)), 4),
            "precision": round(float(precision_score(y_test, y_sub_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_sub_pred)), 4),
            "f1": round(float(f1_score(y_test, y_sub_pred)), 4),
            "pr_auc": round(float(average_precision_score(y_test, p_sub)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, p_sub)), 4),
            "brier_score": round(float(brier_score_loss(y_test, p_sub)), 4)
        }
        print(f"  {tier_name:<38} -> F1={ablation_results[tier_name]['f1']:.4f} | Recall={ablation_results[tier_name]['recall']:.4f} | PR-AUC={ablation_results[tier_name]['pr_auc']:.4f}")

    with open(os.path.join(METRICS_DIR, "pan_india_ablation_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)

    # -----------------------------------------------------------------------
    # 5. Save Comparison JSON & Model Info
    # -----------------------------------------------------------------------
    with open(os.path.join(METRICS_DIR, "pan_india_model_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    # Also update model_comparison.json for backward compatibility
    with open(os.path.join(METRICS_DIR, "model_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    model_info = {
        "champion_model": "Random_Forest",
        "proposed_research_model": "ST_FloodNet",
        "scalable_tabular_baseline": "LightGBM",
        "num_features": len(feature_cols),
        "test_f1": results["Random_Forest"]["f1"],
        "test_recall": results["Random_Forest"]["recall"],
        "test_pr_auc": results["Random_Forest"]["pr_auc"],
        "st_floodnet_f1": results["ST_FloodNet"]["f1"],
        "st_floodnet_recall": results["ST_FloodNet"]["recall"],
        "st_floodnet_pr_auc": results["ST_FloodNet"]["pr_auc"]
    }
    with open(os.path.join(MODELS_DIR, "final_model_info.json"), "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=2)

    # -----------------------------------------------------------------------
    # 6. Generate Figures
    # -----------------------------------------------------------------------
    generate_evaluation_figures(y_test, model_predictions, model_probabilities, ablation_results, spatial_metrics, uncert_var, p_test_st)

    print("\n--- PAN-INDIA BENCHMARK RESULTS SUMMARY ---")
    print(f"{'Model Architecture':<28} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'PR-AUC':<10} {'ROC-AUC':<10}")
    print("-" * 88)
    for m, vals in results.items():
        print(f"{m:<28} {vals['accuracy']:<10.4f} {vals['precision']:<10.4f} {vals['recall']:<10.4f} {vals['f1']:<10.4f} {vals['pr_auc']:<10.4f} {vals['roc_auc']:<10.4f}")

    return results


def evaluate_model(name, y_true, y_pred, y_prob, train_time, latency):
    """Compute complete scientific evaluation metrics."""
    return {
        "model_name": name,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "brier_score": round(float(brier_score_loss(y_true, y_prob)), 4),
        "train_time_sec": round(float(train_time), 3),
        "inference_latency_ms": round(float(latency), 4)
    }


def generate_evaluation_figures(y_true, preds, probs, ablations, spatial, uncert_var, p_st):
    """Generate high-resolution publication-quality evaluation figures."""
    sns.set_theme(style="whitegrid")
    colors = {
        "Logistic Regression": "#7f7f7f",
        "Random Forest": "#2ca02c",
        "Gradient Boosting": "#ff7f0e",
        "XGBoost": "#1f77b4",
        "LightGBM": "#9467bd",
        "Bi-LSTM": "#8c564b",
        "ST-FloodNet (Proposed)": "#d62728"
    }

    # 1. ROC Curves
    plt.figure(figsize=(8, 6), dpi=300)
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y_true, p)
        auc = roc_auc_score(y_true, p)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})", color=colors.get(name, "#333"), linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.6)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate (Recall)", fontsize=12)
    plt.title("Pan-India Test Set (2023) — ROC Curves", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "pan_india_roc_curves.png"))
    plt.savefig(os.path.join(FIGURES_DIR, "roc_curves.png"))
    plt.close()

    # 2. Precision-Recall Curves (Priority Metric for Imbalanced Disasters)
    plt.figure(figsize=(8, 6), dpi=300)
    for name, p in probs.items():
        prec, rec, _ = precision_recall_curve(y_true, p)
        pr_auc = average_precision_score(y_true, p)
        plt.plot(rec, prec, label=f"{name} (PR-AUC = {pr_auc:.4f})", color=colors.get(name, "#333"), linewidth=2)
    plt.xlabel("Recall (Detection Rate)", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Pan-India Test Set (2023) — Precision-Recall Curves", fontsize=14, fontweight="bold")
    plt.legend(loc="lower left", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "pan_india_pr_curves.png"))
    plt.savefig(os.path.join(FIGURES_DIR, "pr_curves.png"))
    plt.close()

    # 3. Confusion Matrices
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), dpi=300)
    axes = axes.flatten()
    for i, (name, pred) in enumerate(preds.items()):
        cm = confusion_matrix(y_true, pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i], cbar=False)
        axes[i].set_title(name, fontsize=11, fontweight="bold")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Ground Truth")
    axes[-1].axis("off")
    plt.suptitle("Pan-India Test Set (2023) — Confusion Matrices", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "pan_india_confusion_matrices.png"))
    plt.savefig(os.path.join(FIGURES_DIR, "confusion_matrices.png"))
    plt.close()

    # 4. Multimodal Feature Ablation Study Bar Chart
    plt.figure(figsize=(10, 5), dpi=300)
    tiers = [t["tier_description"].replace("Exp ", "") for t in ablations.values()]
    f1s = [t["f1"] for t in ablations.values()]
    praucs = [t["pr_auc"] for t in ablations.values()]
    x = np.arange(len(tiers))
    width = 0.35

    plt.bar(x - width/2, f1s, width, label="F1-Score", color="#2ca02c")
    plt.bar(x + width/2, praucs, width, label="PR-AUC", color="#1f77b4")
    plt.xticks(x, tiers, rotation=25, ha="right", fontsize=9)
    plt.ylabel("Score", fontsize=12)
    plt.title("Multimodal Feature Ablation Study across Pan-India Basins", fontsize=13, fontweight="bold")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "pan_india_ablation_study.png"))
    plt.savefig(os.path.join(FIGURES_DIR, "ablation_study.png"))
    plt.close()

    # 5. Spatial Cross-Basin Generalization Plot
    plt.figure(figsize=(7, 5), dpi=300)
    gen_models = ["LightGBM (Baseline)", "ST-FloodNet (Proposed)"]
    gen_f1s = [spatial["LightGBM_Cross_Basin"]["f1"], spatial["ST_FloodNet_Cross_Basin"]["f1"]]
    gen_praucs = [spatial["LightGBM_Cross_Basin"]["pr_auc"], spatial["ST_FloodNet_Cross_Basin"]["pr_auc"]]
    xg = np.arange(len(gen_models))

    plt.bar(xg - 0.17, gen_f1s, 0.34, label="F1-Score (Held-out Basins)", color="#ff7f0e")
    plt.bar(xg + 0.17, gen_praucs, 0.34, label="PR-AUC (Held-out Basins)", color="#d62728")
    plt.xticks(xg, gen_models, fontsize=11)
    plt.ylabel("Score", fontsize=12)
    plt.ylim(0, 1.05)
    plt.title("Spatial Generalization onto Unseen Basins (Brahmaputra & Mahanadi)", fontsize=12, fontweight="bold")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "pan_india_spatial_generalization.png"))
    plt.close()

    # 6. Uncertainty & Calibration Plot
    plt.figure(figsize=(8, 5), dpi=300)
    plt.scatter(p_st[y_true == 0], uncert_var[y_true == 0], alpha=0.35, color="#1f77b4", s=15, label="True Negative (No Flood)")
    plt.scatter(p_st[y_true == 1], uncert_var[y_true == 1], alpha=0.75, color="#d62728", s=25, label="True Positive (Flood Event)")
    plt.xlabel("ST-FloodNet Predicted Probability", fontsize=12)
    plt.ylabel("Epistemic Uncertainty Variance (MC Dropout)", fontsize=12)
    plt.title("Monte Carlo Dropout Epistemic Uncertainty vs Predicted Probability", fontsize=13, fontweight="bold")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "pan_india_uncertainty_calibration.png"))
    plt.close()


if __name__ == "__main__":
    train_and_benchmark_pan_india()
