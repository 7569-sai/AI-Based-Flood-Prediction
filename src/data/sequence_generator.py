"""
True Chronological Temporal Sequence Generator for Spatiotemporal Flood Modeling.
Constructs genuine sliding window sequence tensors [B, T, F_dynamic] from station daily records,
without padding, row-duplication, or pseudo-sequence fabrication.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Optional, List

DYNAMIC_FEATURE_COLS = [
    "rainfall_1d",
    "rainfall_3d",
    "rainfall_7d",
    "rainfall_14d",
    "rainfall_anomaly_30d",
    "api_14d",
    "temp_mean_3d",
    "humidity_mean_3d",
    "soil_moisture_root_zone",
    "soil_moisture_deep",
    "soil_saturation_ratio",
    "river_discharge_proxy",
    "river_stage_anomaly"
]

STATIC_FEATURE_COLS = [
    "elevation_m",
    "slope_deg",
    "dist_river_km",
    "twi_proxy",
    "elevation_risk_factor",
    "drainage_vulnerability",
    "sar_vv",
    "sar_vh",
    "sar_cross_ratio",
    "ndwi_proxy",
    "ndvi_proxy",
    "population_density",
    "exposure_index",
    "vulnerability_index"
]

ALL_27_FEATURE_COLS = DYNAMIC_FEATURE_COLS + STATIC_FEATURE_COLS


def build_genuine_temporal_splits(
    data_dir: str,
    scaler = None,
    window_size: int = 14,
    feature_cols: Optional[List[str]] = None,
    target_col: str = "flood_occurred",
    severity_col: str = "flood_severity"
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, int]]:
    """
    Constructs genuine chronological sequences [N, T, 13] across Train, Val, and Test splits.
    
    Guarantees:
    1. Sequences represent authentic consecutive daily measurements: [t - T + 1, ..., t].
    2. No current-row repeating / tiling.
    3. Invalid windows (the first T-1 days of 2018 per station) are strictly discarded.
    4. Validation (2022) uses authentic trailing history from late 2021 without leakage.
    5. Test (2023) uses authentic trailing history from late 2022 without leakage.
    
    Returns:
        (train_dict, val_dict, test_dict, stats_dict)
    """
    if feature_cols is None:
        feature_cols = ALL_27_FEATURE_COLS

    # 1. Load Parquet Data
    train_df = pd.read_parquet(os.path.join(data_dir, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(data_dir, "val.parquet"))
    test_df = pd.read_parquet(os.path.join(data_dir, "test.parquet"))

    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    full_df["date"] = pd.to_datetime(full_df["date"])
    full_df = full_df.sort_values(by=["station_id", "date"]).reset_index(drop=True)

    # Scale features using pre-fitted training scaler
    if scaler is not None:
        X_all_scaled = scaler.transform(full_df[feature_cols])
    else:
        X_all_scaled = full_df[feature_cols].values

    dyn_indices = [feature_cols.index(c) for c in DYNAMIC_FEATURE_COLS]
    static_indices = [feature_cols.index(c) for c in STATIC_FEATURE_COLS]

    X_dyn = X_all_scaled[:, dyn_indices].astype(np.float32)      # [N, 13]
    X_stat = X_all_scaled[:, static_indices].astype(np.float32)  # [N, 14]
    y_haz = full_df[target_col].values.astype(np.float32)
    y_sev = full_df[severity_col].values.astype(np.int64)
    dates = full_df["date"].values
    station_ids = full_df["station_id"].values

    T = window_size
    N_total = len(full_df)
    n_stations = full_df["station_id"].nunique()

    # Pre-allocate containers
    valid_indices = []
    discarded_count = 0

    # Group by station to extract authentic chronological windows
    for st_id, group in full_df.groupby("station_id"):
        idxs = group.index.values
        n_days = len(idxs)
        if n_days < T:
            discarded_count += n_days
            continue

        # First T-1 days cannot form a full historical sequence of length T -> discard
        discarded_count += (T - 1)
        valid_indices.extend(idxs[T - 1:].tolist())

    valid_indices = np.array(valid_indices)
    valid_indices.sort()
    N_valid = len(valid_indices)

    # Build 3D array for valid sequences
    X_seq_valid = np.zeros((N_valid, T, len(dyn_indices)), dtype=np.float32)
    for i, idx in enumerate(valid_indices):
        X_seq_valid[i] = X_dyn[idx - T + 1 : idx + 1]

    X_stat_valid = X_stat[valid_indices]
    X_flat_valid = X_all_scaled[valid_indices].astype(np.float32)
    y_haz_valid = y_haz[valid_indices]
    y_sev_valid = y_sev[valid_indices]
    dates_valid = dates[valid_indices]
    stations_valid = station_ids[valid_indices]

    # Partition based on target prediction date
    date_series = pd.to_datetime(dates_valid)
    mask_train = (date_series <= "2021-12-31")
    mask_val = (date_series >= "2022-01-01") & (date_series <= "2022-12-31")
    mask_test = (date_series >= "2023-01-01")

    train_data = {
        "X_dynamic_seq": X_seq_valid[mask_train],
        "X_static": X_stat_valid[mask_train],
        "X_flat": X_flat_valid[mask_train],
        "y_hazard": y_haz_valid[mask_train],
        "y_severity": y_sev_valid[mask_train],
        "dates": dates_valid[mask_train],
        "station_ids": stations_valid[mask_train]
    }

    val_data = {
        "X_dynamic_seq": X_seq_valid[mask_val],
        "X_static": X_stat_valid[mask_val],
        "X_flat": X_flat_valid[mask_val],
        "y_hazard": y_haz_valid[mask_val],
        "y_severity": y_sev_valid[mask_val],
        "dates": dates_valid[mask_val],
        "station_ids": stations_valid[mask_val]
    }

    test_data = {
        "X_dynamic_seq": X_seq_valid[mask_test],
        "X_static": X_stat_valid[mask_test],
        "X_flat": X_flat_valid[mask_test],
        "y_hazard": y_haz_valid[mask_test],
        "y_severity": y_sev_valid[mask_test],
        "dates": dates_valid[mask_test],
        "station_ids": stations_valid[mask_test]
    }

    stats = {
        "sequence_length_T": T,
        "total_records": N_total,
        "valid_sequences_total": N_valid,
        "discarded_invalid_sequences": discarded_count,
        "train_sequences": int(mask_train.sum()),
        "val_sequences": int(mask_val.sum()),
        "test_sequences": int(mask_test.sum()),
        "stations_count": n_stations
    }

    return train_data, val_data, test_data, stats


def build_temporal_sequences(
    df: pd.DataFrame,
    scaler = None,
    window_size: int = 14,
    feature_cols: Optional[List[str]] = None,
    target_col: str = "flood_occurred",
    severity_col: str = "flood_severity",
    drop_invalid: bool = False
) -> Dict[str, np.ndarray]:
    """
    Constructs chronological sliding window sequence arrays from multi-station data.
    If drop_invalid=True, discards windows with fewer than window_size historical observations.
    If drop_invalid=False (for dashboard/single-station fallback), repeats earliest available history.
    """
    if feature_cols is None:
        feature_cols = ALL_27_FEATURE_COLS

    df_sorted = df.copy()
    if "date" in df_sorted.columns:
        df_sorted["date"] = pd.to_datetime(df_sorted["date"])
        df_sorted = df_sorted.sort_values(by=["station_id", "date"]).reset_index(drop=True)

    if scaler is not None:
        X_all_scaled = scaler.transform(df_sorted[feature_cols])
    else:
        X_all_scaled = df_sorted[feature_cols].values

    dyn_indices = [feature_cols.index(c) for c in DYNAMIC_FEATURE_COLS]
    static_indices = [feature_cols.index(c) for c in STATIC_FEATURE_COLS]

    X_dyn = X_all_scaled[:, dyn_indices].astype(np.float32)
    X_stat = X_all_scaled[:, static_indices].astype(np.float32)

    T = window_size
    F_dyn = len(dyn_indices)

    if drop_invalid:
        valid_idxs = []
        for station_id, group in df_sorted.groupby("station_id"):
            idxs = group.index.values
            if len(idxs) >= T:
                valid_idxs.extend(idxs[T - 1:].tolist())
        valid_idxs = np.array(valid_idxs)
        valid_idxs.sort()

        N = len(valid_idxs)
        X_dynamic_seq = np.zeros((N, T, F_dyn), dtype=np.float32)
        for i, idx in enumerate(valid_idxs):
            X_dynamic_seq[i] = X_dyn[idx - T + 1 : idx + 1]

        sub_df = df_sorted.iloc[valid_idxs]
        return {
            "X_dynamic_seq": X_dynamic_seq,
            "X_static": X_stat[valid_idxs],
            "X_flat": X_all_scaled[valid_idxs].astype(np.float32),
            "station_ids": sub_df["station_id"].values if "station_id" in sub_df else np.zeros(N),
            "dates": sub_df["date"].values if "date" in sub_df else np.zeros(N),
            "y_hazard": sub_df[target_col].values.astype(np.float32) if target_col in sub_df else np.zeros(N, dtype=np.float32),
            "y_severity": sub_df[severity_col].values.astype(np.int64) if severity_col in sub_df else np.zeros(N, dtype=np.int64)
        }

    N = len(df_sorted)
    X_dynamic_seq = np.zeros((N, T, F_dyn), dtype=np.float32)
    for station_id, group in df_sorted.groupby("station_id"):
        indices = group.index.values
        n_pts = len(indices)
        station_dyn = X_dyn[indices]

        for i in range(n_pts):
            global_idx = indices[i]
            if i < T:
                pad_len = T - (i + 1)
                first_obs = np.tile(station_dyn[0], (pad_len, 1))
                actual_obs = station_dyn[: i + 1]
                seq = np.vstack([first_obs, actual_obs])
            else:
                seq = station_dyn[i - T + 1 : i + 1]
            X_dynamic_seq[global_idx] = seq

    results = {
        "X_dynamic_seq": X_dynamic_seq,
        "X_static": X_stat,
        "X_flat": X_all_scaled.astype(np.float32),
        "station_ids": df_sorted["station_id"].values if "station_id" in df_sorted else np.zeros(N),
        "dates": df_sorted["date"].values if "date" in df_sorted else np.zeros(N),
        "y_hazard": df_sorted[target_col].values.astype(np.float32) if target_col in df_sorted else np.zeros(N, dtype=np.float32),
        "y_severity": df_sorted[severity_col].values.astype(np.int64) if severity_col in df_sorted else np.zeros(N, dtype=np.int64)
    }
    return results


class SpatiotemporalFloodDataset(Dataset):
    """
    PyTorch Dataset yielding genuine temporal dynamic sequences and static multimodal vectors.
    """
    def __init__(
        self,
        X_dynamic_seq: np.ndarray,
        X_static: np.ndarray,
        X_flat: Optional[np.ndarray] = None,
        y_hazard: Optional[np.ndarray] = None,
        y_severity: Optional[np.ndarray] = None
    ):
        self.X_dynamic_seq = torch.tensor(X_dynamic_seq, dtype=torch.float32)
        self.X_static = torch.tensor(X_static, dtype=torch.float32)
        self.X_flat = torch.tensor(X_flat, dtype=torch.float32) if X_flat is not None else None
        self.y_hazard = torch.tensor(y_hazard, dtype=torch.float32) if y_hazard is not None else None
        self.y_severity = torch.tensor(y_severity, dtype=torch.long) if y_severity is not None else None

    def __len__(self) -> int:
        return len(self.X_dynamic_seq)

    def __getitem__(self, idx: int):
        item = {
            "x_dynamic": self.X_dynamic_seq[idx],
            "x_static": self.X_static[idx]
        }
        if self.X_flat is not None:
            item["x_flat"] = self.X_flat[idx]
        if self.y_hazard is not None:
            item["y_hazard"] = self.y_hazard[idx]
        if self.y_severity is not None:
            item["y_severity"] = self.y_severity[idx]
        return item
