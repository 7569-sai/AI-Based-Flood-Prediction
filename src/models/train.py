"""
Model Training, Evaluation, Ablation Study, and Deep Learning Comparison.
Trains:
- Model A: Baseline Logistic Regression (balanced)
- Model B: Random Forest Classifier
- Model C: Gradient Boosting Classifier
- Model D: XGBoost Classifier
- Model E: Sequential PyTorch LSTM
Performs 4-tier feature ablation study, generates evaluation curves,
and exports all metrics and model artifacts.
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve, brier_score_loss
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models", "trained")
SCALER_DIR = os.path.join(BASE_DIR, "models", "scalers")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")
METRICS_DIR = os.path.join(BASE_DIR, "outputs", "metrics")
CARDS_DIR = os.path.join(BASE_DIR, "models", "MODEL_CARDS")
DOCS_DIR = os.path.join(BASE_DIR, "docs")

for d in [MODELS_DIR, SCALER_DIR, FIGURES_DIR, METRICS_DIR, CARDS_DIR, DOCS_DIR]:
    os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# Deep Learning PyTorch Dataset & LSTM Model
# ---------------------------------------------------------------------------
class FloodSequenceDataset(Dataset):
    def __init__(self, df, feature_cols, scaler, seq_length=14):
        self.seq_length = seq_length
        self.sequences = []
        self.targets = []
        
        # Scale features
        X_scaled = scaler.transform(df[feature_cols])
        y = df["flood_occurred"].values
        stations = df["station_id"].values
        
        # Construct sliding sequences per station
        unique_stations = np.unique(stations)
        for st in unique_stations:
            st_idx = np.where(stations == st)[0]
            st_X = X_scaled[st_idx]
            st_y = y[st_idx]
            for i in range(len(st_X) - seq_length):
                self.sequences.append(st_X[i : i + seq_length])
                self.targets.append(st_y[i + seq_length])
                
        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx]), torch.tensor(self.targets[idx])


class FloodLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super(FloodLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        self.fc1 = nn.Linear(hidden_dim * 2, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        # Take the output of the last sequence step
        last_step = out[:, -1, :]
        h = self.dropout(self.relu(self.fc1(last_step)))
        out = self.fc2(h)
        return out


# ---------------------------------------------------------------------------
# Evaluation Utility
# ---------------------------------------------------------------------------
def evaluate_predictions(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = 0.5
        
    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except Exception:
        pr_auc = 0.0
        
    brier = brier_score_loss(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred).tolist()
    
    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "brier_score": round(float(brier), 4),
        "confusion_matrix": cm,
        "support_positive": int(sum(y_true)),
        "support_total": int(len(y_true))
    }


def run_training_pipeline():
    print("=" * 70)
    print("PHASE 5: Model Training & Evaluation")
    print("=" * 70)

    # 1. Load data
    df_train = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    df_val = pd.read_csv(os.path.join(PROCESSED_DIR, "val.csv"))
    df_test = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))
    
    with open(os.path.join(PROCESSED_DIR, "feature_metadata.json"), "r") as f:
        meta = json.load(f)
        
    feature_cols = meta["feature_cols"]
    scaler = joblib.load(os.path.join(SCALER_DIR, "feature_scaler.joblib"))

    X_train_raw = df_train[feature_cols]
    y_train = df_train["flood_occurred"].values

    X_val_raw = df_val[feature_cols]
    y_val = df_val["flood_occurred"].values

    X_test_raw = df_test[feature_cols]
    y_test = df_test["flood_occurred"].values

    X_train_scaled = scaler.transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    print(f"Features: {len(feature_cols)} | Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")
    pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    print(f"Positive class weight ratio: {pos_weight:.2f}")

    models = {}
    metrics_summary = {}
    model_predictions = {}

    # -----------------------------------------------------------------------
    # Model A: Baseline Logistic Regression
    # -----------------------------------------------------------------------
    print("\n--- Training Model A: Logistic Regression (Baseline) ---")
    t0 = time.time()
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_train_time = time.time() - t0

    t0 = time.time()
    lr_prob = lr.predict_proba(X_test_scaled)[:, 1]
    lr_inf_time = (time.time() - t0) / len(X_test_scaled) * 1000  # ms per sample
    lr_pred = (lr_prob >= 0.5).astype(int)

    models["Logistic_Regression"] = lr
    joblib.dump(lr, os.path.join(MODELS_DIR, "model_logistic_regression.joblib"))
    metrics_summary["Logistic_Regression"] = evaluate_predictions(y_test, lr_pred, lr_prob)
    metrics_summary["Logistic_Regression"]["train_time_sec"] = round(lr_train_time, 3)
    metrics_summary["Logistic_Regression"]["inference_latency_ms"] = round(lr_inf_time, 4)
    model_predictions["Logistic_Regression"] = {"prob": lr_prob, "pred": lr_pred}
    print(f"Logistic Regression: Recall={metrics_summary['Logistic_Regression']['recall']}, F1={metrics_summary['Logistic_Regression']['f1']}, PR-AUC={metrics_summary['Logistic_Regression']['pr_auc']}")

    # -----------------------------------------------------------------------
    # Model B: Random Forest Classifier
    # -----------------------------------------------------------------------
    print("\n--- Training Model B: Random Forest Classifier ---")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=150, max_depth=12, min_samples_split=6,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X_train_raw, y_train)  # Tree models can use raw scale directly
    rf_train_time = time.time() - t0

    t0 = time.time()
    rf_prob = rf.predict_proba(X_test_raw)[:, 1]
    rf_inf_time = (time.time() - t0) / len(X_test_raw) * 1000
    rf_pred = (rf_prob >= 0.5).astype(int)

    models["Random_Forest"] = rf
    joblib.dump(rf, os.path.join(MODELS_DIR, "model_random_forest.joblib"))
    metrics_summary["Random_Forest"] = evaluate_predictions(y_test, rf_pred, rf_prob)
    metrics_summary["Random_Forest"]["train_time_sec"] = round(rf_train_time, 3)
    metrics_summary["Random_Forest"]["inference_latency_ms"] = round(rf_inf_time, 4)
    model_predictions["Random_Forest"] = {"prob": rf_prob, "pred": rf_pred}
    print(f"Random Forest: Recall={metrics_summary['Random_Forest']['recall']}, F1={metrics_summary['Random_Forest']['f1']}, PR-AUC={metrics_summary['Random_Forest']['pr_auc']}")

    # -----------------------------------------------------------------------
    # Model C: Gradient Boosting Classifier
    # -----------------------------------------------------------------------
    print("\n--- Training Model C: Gradient Boosting Classifier ---")
    t0 = time.time()
    # Compute sample weights to balance gradient boosting
    sample_weights = np.where(y_train == 1, pos_weight, 1.0)
    gb = HistGradientBoostingClassifier(
        max_iter=120, max_depth=8, min_samples_leaf=15,
        random_state=42
    )
    gb.fit(X_train_raw, y_train, sample_weight=sample_weights)
    gb_train_time = time.time() - t0

    t0 = time.time()
    gb_prob = gb.predict_proba(X_test_raw)[:, 1]
    gb_inf_time = (time.time() - t0) / len(X_test_raw) * 1000
    gb_pred = (gb_prob >= 0.5).astype(int)

    models["Gradient_Boosting"] = gb
    joblib.dump(gb, os.path.join(MODELS_DIR, "model_gradient_boosting.joblib"))
    metrics_summary["Gradient_Boosting"] = evaluate_predictions(y_test, gb_pred, gb_prob)
    metrics_summary["Gradient_Boosting"]["train_time_sec"] = round(gb_train_time, 3)
    metrics_summary["Gradient_Boosting"]["inference_latency_ms"] = round(gb_inf_time, 4)
    model_predictions["Gradient_Boosting"] = {"prob": gb_prob, "pred": gb_pred}
    print(f"Gradient Boosting: Recall={metrics_summary['Gradient_Boosting']['recall']}, F1={metrics_summary['Gradient_Boosting']['f1']}, PR-AUC={metrics_summary['Gradient_Boosting']['pr_auc']}")

    # -----------------------------------------------------------------------
    # Model D: XGBoost Classifier
    # -----------------------------------------------------------------------
    print("\n--- Training Model D: XGBoost Classifier ---")
    t0 = time.time()
    xgb = XGBClassifier(
        n_estimators=180, max_depth=6, learning_rate=0.06,
        scale_pos_weight=pos_weight * 0.75, subsample=0.85,
        colsample_bytree=0.85, eval_metric="logloss",
        random_state=42, n_jobs=-1
    )
    xgb.fit(X_train_raw, y_train)
    xgb_train_time = time.time() - t0

    t0 = time.time()
    xgb_prob = xgb.predict_proba(X_test_raw)[:, 1]
    xgb_inf_time = (time.time() - t0) / len(X_test_raw) * 1000
    xgb_pred = (xgb_prob >= 0.5).astype(int)

    models["XGBoost"] = xgb
    joblib.dump(xgb, os.path.join(MODELS_DIR, "model_xgboost.joblib"))
    metrics_summary["XGBoost"] = evaluate_predictions(y_test, xgb_pred, xgb_prob)
    metrics_summary["XGBoost"]["train_time_sec"] = round(xgb_train_time, 3)
    metrics_summary["XGBoost"]["inference_latency_ms"] = round(xgb_inf_time, 4)
    model_predictions["XGBoost"] = {"prob": xgb_prob, "pred": xgb_pred}
    print(f"XGBoost: Recall={metrics_summary['XGBoost']['recall']}, F1={metrics_summary['XGBoost']['f1']}, PR-AUC={metrics_summary['XGBoost']['pr_auc']}")

    # -----------------------------------------------------------------------
    # Model E: Sequential Deep Learning (PyTorch Bidirectional LSTM)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 7: Sequential Deep Learning (PyTorch Bi-LSTM)")
    print("=" * 70)
    seq_len = 14
    train_ds = FloodSequenceDataset(df_train, feature_cols, scaler, seq_length=seq_len)
    test_ds = FloodSequenceDataset(df_test, feature_cols, scaler, seq_length=seq_len)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    lstm_model = FloodLSTM(input_dim=len(feature_cols), hidden_dim=48, num_layers=2)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight * 0.75]))
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.002, weight_decay=1e-4)

    t0 = time.time()
    lstm_model.train()
    epochs = 8
    for epoch in range(epochs):
        epoch_loss = 0.0
        for seq_b, target_b in train_loader:
            optimizer.zero_grad()
            out = lstm_model(seq_b).squeeze(-1)
            loss = criterion(out, target_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(target_b)
        epoch_loss /= len(train_ds)
        print(f"  Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")
    lstm_train_time = time.time() - t0

    # Evaluate LSTM
    lstm_model.eval()
    t0 = time.time()
    lstm_probs = []
    lstm_targets = []
    with torch.no_grad():
        for seq_b, target_b in test_loader:
            logits = lstm_model(seq_b).squeeze(-1)
            probs = torch.sigmoid(logits).numpy()
            lstm_probs.extend(probs)
            lstm_targets.extend(target_b.numpy())
    lstm_inf_time = (time.time() - t0) / len(test_ds) * 1000
    
    lstm_probs = np.array(lstm_probs)
    lstm_targets = np.array(lstm_targets)
    lstm_pred = (lstm_probs >= 0.5).astype(int)

    torch.save(lstm_model.state_dict(), os.path.join(MODELS_DIR, "model_bilstm.pt"))
    metrics_summary["Bi_LSTM"] = evaluate_predictions(lstm_targets, lstm_pred, lstm_probs)
    metrics_summary["Bi_LSTM"]["train_time_sec"] = round(lstm_train_time, 3)
    metrics_summary["Bi_LSTM"]["inference_latency_ms"] = round(lstm_inf_time, 4)
    model_predictions["Bi_LSTM"] = {"prob": lstm_probs, "pred": lstm_pred, "y_true": lstm_targets}
    print(f"Bi-LSTM: Recall={metrics_summary['Bi_LSTM']['recall']}, F1={metrics_summary['Bi_LSTM']['f1']}, PR-AUC={metrics_summary['Bi_LSTM']['pr_auc']}")

    # -----------------------------------------------------------------------
    # Phase 6: Mandatory Ablation Study
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 6: Mandatory Ablation Study (4-Tier Feature Subsets)")
    print("=" * 70)
    ablation_subsets = meta["feature_subsets"]
    ablation_results = {}

    for exp_name, subset_cols in ablation_subsets.items():
        print(f"Running Ablation Experiment: {exp_name} ({len(subset_cols)} features)...")
        clf = XGBClassifier(
            n_estimators=180, max_depth=6, learning_rate=0.06,
            scale_pos_weight=pos_weight * 0.75, subsample=0.85,
            colsample_bytree=0.85, eval_metric="logloss",
            random_state=42, n_jobs=-1
        )
        clf.fit(df_train[subset_cols], y_train)
        pred_prob = clf.predict_proba(df_test[subset_cols])[:, 1]
        pred_bin = (pred_prob >= 0.5).astype(int)
        
        res = evaluate_predictions(y_test, pred_bin, pred_prob)
        res["n_features"] = len(subset_cols)
        res["features"] = subset_cols
        ablation_results[exp_name] = res
        print(f"  {exp_name}: F1={res['f1']}, Recall={res['recall']}, PR-AUC={res['pr_auc']}, ROC-AUC={res['roc_auc']}")

    # Save metrics JSON
    with open(os.path.join(METRICS_DIR, "model_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    with open(os.path.join(METRICS_DIR, "ablation_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)

    # -----------------------------------------------------------------------
    # Generate Plots & Visualizations
    # -----------------------------------------------------------------------
    print("\nGenerating evaluation charts...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Confusion Matrices
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    disp_models = ["Logistic_Regression", "Random_Forest", "Gradient_Boosting", "XGBoost"]
    for idx, m_name in enumerate(disp_models):
        cm = metrics_summary[m_name]["confusion_matrix"]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[idx],
                    xticklabels=["No Flood", "Flood"], yticklabels=["No Flood", "Flood"])
        axes[idx].set_title(f"{m_name.replace('_', ' ')}\nF1: {metrics_summary[m_name]['f1']} | Rec: {metrics_summary[m_name]['recall']}")
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("True" if idx == 0 else "")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "confusion_matrices.png"), dpi=200)
    plt.close()

    # 2. ROC Curves
    plt.figure(figsize=(8, 6))
    for m_name in disp_models:
        fpr, tpr, _ = roc_curve(y_test, model_predictions[m_name]["prob"])
        auc_val = metrics_summary[m_name]["roc_auc"]
        plt.plot(fpr, tpr, label=f"{m_name.replace('_', ' ')} (AUC = {auc_val:.3f})", lw=2)
    # Also plot Bi-LSTM
    fpr_l, tpr_l, _ = roc_curve(model_predictions["Bi_LSTM"]["y_true"], model_predictions["Bi_LSTM"]["prob"])
    plt.plot(fpr_l, tpr_l, label=f"Bi-LSTM (AUC = {metrics_summary['Bi_LSTM']['roc_auc']:.3f})", lw=2, linestyle="--")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Comparison on Chronological Test Set (2023)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "roc_curves.png"), dpi=200)
    plt.close()

    # 3. Precision-Recall Curves
    plt.figure(figsize=(8, 6))
    for m_name in disp_models:
        prec, rec, _ = precision_recall_curve(y_test, model_predictions[m_name]["prob"])
        pr_val = metrics_summary[m_name]["pr_auc"]
        plt.plot(rec, prec, label=f"{m_name.replace('_', ' ')} (PR-AUC = {pr_val:.3f})", lw=2)
    prec_l, rec_l, _ = precision_recall_curve(model_predictions["Bi_LSTM"]["y_true"], model_predictions["Bi_LSTM"]["prob"])
    plt.plot(rec_l, prec_l, label=f"Bi-LSTM (PR-AUC = {metrics_summary['Bi_LSTM']['pr_auc']:.3f})", lw=2, linestyle="--")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves (Priority Metric for Imbalanced Flood Data)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "pr_curves.png"), dpi=200)
    plt.close()

    # 4. Feature Importance (XGBoost)
    plt.figure(figsize=(10, 7))
    xgb_best = models["XGBoost"]
    importances = pd.Series(xgb_best.feature_importances_, index=feature_cols).sort_values(ascending=True)
    importances.tail(15).plot(kind="barh", color="steelblue")
    plt.title("Top 15 Feature Importances (XGBoost Final Model)")
    plt.xlabel("Gini / Gain Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "feature_importances.png"), dpi=200)
    plt.close()

    # 5. Ablation Study Bar Chart
    plt.figure(figsize=(9, 5))
    exp_labels = ["Exp A: Met Only", "Exp B: Met + Terrain", "Exp C: Met+Terr+RS", "Exp D: All Features"]
    f1_vals = [ablation_results[k]["f1"] for k in ["exp_a_met", "exp_b_met_terrain", "exp_c_met_terrain_rs", "exp_d_all_features"]]
    rec_vals = [ablation_results[k]["recall"] for k in ["exp_a_met", "exp_b_met_terrain", "exp_c_met_terrain_rs", "exp_d_all_features"]]
    pr_vals = [ablation_results[k]["pr_auc"] for k in ["exp_a_met", "exp_b_met_terrain", "exp_c_met_terrain_rs", "exp_d_all_features"]]

    x = np.arange(len(exp_labels))
    width = 0.25
    plt.bar(x - width, f1_vals, width, label="F1 Score", color="#2b5c8f")
    plt.bar(x, rec_vals, width, label="Recall", color="#d95f02")
    plt.bar(x + width, pr_vals, width, label="PR-AUC", color="#7570b3")
    plt.xticks(x, exp_labels, rotation=15)
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.title("Feature Ablation Study: Impact of Adding Multimodal Layers")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "ablation_study.png"), dpi=200)
    plt.close()

    # -----------------------------------------------------------------------
    # Model Selection & Documentation
    # -----------------------------------------------------------------------
    # Determine best model based on F1 and PR-AUC
    ranked = sorted(metrics_summary.items(), key=lambda kv: (kv[1]["f1"] + kv[1]["pr_auc"]) / 2, reverse=True)
    best_name, best_metrics = ranked[0]
    print(f"\nFinal Selected Model: {best_name} (F1: {best_metrics['f1']}, Recall: {best_metrics['recall']}, PR-AUC: {best_metrics['pr_auc']})")

    # Save selection pointer
    selection_meta = {
        "final_model_name": best_name,
        "model_file": f"model_{best_name.lower()}.joblib",
        "metrics": best_metrics,
        "selection_rationale": "Selected based on highest composite F1-score, Recall, and PR-AUC on chronological out-of-time test set (2023). High recall is essential in flood hazard modeling to minimize catastrophic false negatives."
    }
    with open(os.path.join(MODELS_DIR, "final_model_info.json"), "w", encoding="utf-8") as f:
        json.dump(selection_meta, f, indent=2)

    return metrics_summary, ablation_results

if __name__ == "__main__":
    run_training_pipeline()
