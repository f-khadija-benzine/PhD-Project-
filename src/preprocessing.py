"""
Unified Preprocessing Pipeline with BI Integration
Chains all preprocessing steps from Section III-B:

    1. RUL clipping (piecewise linear, Eq. 1-2)
    2. Normalize sensor features (via DataNormalizer)
    3. Fuse with BI data (bi_fusion.py — Section III-B3)
    3b. Normalize continuous BI variables
    4. BI-aware feature selection (feature_selection.py — Section III-B4)
    5. Sliding window generation (Section III-B5)

Author: Fatima Khadija Benzine
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import sys
current_file = Path(__file__).resolve()
src_dir = current_file.parent
project_root = src_dir.parent
sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(project_root))

from bi_fusion import BIFusionPipeline, BIDataLoader, BI_FEATURE_NAMES, CATEGORICAL_BI_VARS, CONTINUOUS_BI_VARS
from feature_selection import BIAwareFeatureSelector


class DataNormalizer:
    """
    Handles data normalization. Reproduced from existing notebook code
    to keep src/ self-contained.
    """

    def __init__(self, method: str = 'minmax'):
        self.method = method
        self.scalers = {}
        self.fitted = False
        self.columns = None

    def fit(self, data: pd.DataFrame, columns: List[str]) -> 'DataNormalizer':
        if self.method == 'minmax':
            scaler = MinMaxScaler()
        elif self.method == 'standard':
            scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown method: {self.method}")
        scaler.fit(data[columns])
        self.scalers['global'] = scaler
        self.columns = columns
        self.fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise ValueError("Normalizer must be fitted before transform")
        data = data.copy()
        data[self.columns] = self.scalers['global'].transform(data[self.columns])
        return data

    def fit_transform(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        self.fit(data, columns)
        return self.transform(data)


class PreprocessingPipelineBI:
    """
    Complete preprocessing pipeline with BI integration.

    Usage:
        pipeline = PreprocessingPipelineBI()
        train_processed = pipeline.fit_transform(train_df, 'FD001')
        test_processed  = pipeline.transform(test_df)
    """

    def __init__(self,
                 normalization_method: str = 'minmax',
                 variance_threshold: float = 0.01,
                 correlation_threshold: float = 0.95,
                 rul_max: int = 125,
                 bi_data_dir: Path = None):
        """
        Args:
            normalization_method: 'minmax' or 'standard'
            variance_threshold: for sensor feature selection
            correlation_threshold: for correlation-based filtering
            rul_max: piecewise linear RUL cap (Eq. 2)
            bi_data_dir: path to BI CSV files
        """
        self.normalizer = DataNormalizer(method=normalization_method)
        self.bi_normalizer = DataNormalizer(method=normalization_method)
        self.bi_fusion = BIFusionPipeline(bi_data_dir)
        self.feature_selector = BIAwareFeatureSelector(
            variance_threshold=variance_threshold,
            correlation_threshold=correlation_threshold,
        )
        self.rul_max = rul_max
        self.fitted = False

        # Will be set during fit
        self.sensor_cols = []
        self.setting_cols = []
        self.bi_cols = []
        self.meta_cols = ['unit', 'cycle', 'rul']

    def fit_transform(self,
                      train_df: pd.DataFrame,
                      dataset_name: str) -> pd.DataFrame:
        """
        Fit the full pipeline on training data and return processed result.

        Args:
            train_df: raw training DataFrame from MultiDatasetLoader
            dataset_name: 'FD001', 'FD002', 'FD003', 'FD004'

        Returns:
            Preprocessed training DataFrame
        """
        print("=" * 60)
        print(f"PREPROCESSING PIPELINE — {dataset_name}")
        print("=" * 60)

        df = train_df.copy()

        # ---- Step 0: Clip RUL (Eq. 2) ----
        if 'rul' in df.columns and self.rul_max is not None:
            df['rul'] = df['rul'].clip(upper=self.rul_max)
            print(f"\n[Step 0] RUL clipped to max={self.rul_max}")

        # ---- Step 1: Identify column groups ----
        self.sensor_cols = [c for c in df.columns if c.startswith('sensor_')]
        self.setting_cols = [c for c in df.columns if c.startswith('setting_')]
        normalize_cols = self.sensor_cols + self.setting_cols

        print(f"\n[Step 1] Columns: {len(self.sensor_cols)} sensor, "
              f"{len(self.setting_cols)} setting")

        # ---- Step 2: Normalize sensor + setting features ----
        # Normalize BEFORE fusion so BI variables keep their natural scale
        df = self.normalizer.fit_transform(df, normalize_cols)
        print(f"\n[Step 2] Normalized {len(normalize_cols)} features "
              f"({self.normalizer.method})")

        # ---- Step 3: Fuse with BI data (Section III-B3) ----
        df = self.bi_fusion.fuse(df, dataset_name, split='train', encode_categoricals=True)
        self.bi_cols = self.bi_fusion.get_bi_columns(df)

        # ---- Step 3b: Normalize continuous BI variables ----
        # Binary (0/1) and one-hot encoded variables are NOT normalized
        self.bi_continuous_cols = [c for c in CONTINUOUS_BI_VARS if c in df.columns]
        if self.bi_continuous_cols:
            df = self.bi_normalizer.fit_transform(df, self.bi_continuous_cols)
            print(f"\n[Step 3b] Normalized {len(self.bi_continuous_cols)} continuous BI features "
                  f"({self.bi_normalizer.method})")
            print(f"  Normalized: {self.bi_continuous_cols}")
            bi_binary = [c for c in self.bi_cols if c not in self.bi_continuous_cols]
            print(f"  Kept as-is (binary/one-hot): {len(bi_binary)} features")

        # ---- Step 4: BI-aware feature selection (Section III-B4) ----
        self.feature_selector.select_features(
            data=df,
            sensor_cols=self.sensor_cols,
            bi_cols=self.bi_cols,
            setting_cols=self.setting_cols,
            exclude_cols=self.meta_cols,
        )

        df = self.feature_selector.transform(df, keep_cols=self.meta_cols)

        self.fitted = True
        print(f"\n[Done] Final shape: {df.shape}")
        print("=" * 60)
        return df

    def transform(self, test_df: pd.DataFrame,
                  dataset_name: str) -> pd.DataFrame:
        """
        Transform test data using fitted pipeline.

        Args:
            test_df: raw test DataFrame from MultiDatasetLoader
            dataset_name: needed to load BI data for test units

        Returns:
            Preprocessed test DataFrame
        """
        if not self.fitted:
            raise ValueError("Pipeline must be fitted before transform. "
                             "Call fit_transform() first.")

        df = test_df.copy()

        # Clip RUL if present
        if 'rul' in df.columns and self.rul_max is not None:
            df['rul'] = df['rul'].clip(upper=self.rul_max)

        # Normalize using fitted scaler
        df = self.normalizer.transform(df)

        # Fuse with BI
        df = self.bi_fusion.fuse(df, dataset_name, split='test', encode_categoricals=True)

        # Normalize continuous BI variables using fitted scaler
        if self.bi_continuous_cols:
            df = self.bi_normalizer.transform(df)

        # Select same features as training
        df = self.feature_selector.transform(df, keep_cols=self.meta_cols)

        return df

    def get_feature_groups(self) -> Dict[str, List[str]]:
        """Return selected features grouped by type."""
        selected = self.feature_selector.selected_features or []
        return {
            'sensor': [c for c in selected if c.startswith('sensor_')],
            'setting': [c for c in selected if c.startswith('setting_')],
            'bi': [c for c in selected if c in self.bi_cols],
            'all': selected,
        }

    def get_selection_report(self) -> Dict:
        """Return feature selection report."""
        return self.feature_selector.get_report()


def create_sliding_windows(df: pd.DataFrame,
                           window_size: int = 30,
                           feature_cols: List[str] = None,
                           target_col: str = 'rul',
                           unit_col: str = 'unit',
                           pad: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate sliding window sequences from preprocessed data (Section III-B5).

    For each unit at time step t, the window is:
        X[i] = features[t-W+1 : t+1]   shape (W, n_features)
        y[i] = RUL[t]                   scalar

    Units with fewer than W time steps are zero-padded at the beginning
    if pad=True, otherwise excluded.

    Args:
        df: preprocessed DataFrame (output of PreprocessingPipelineBI)
        window_size: W — number of past time steps per sample
        feature_cols: columns to include as features (if None, auto-detect)
        target_col: RUL column name
        unit_col: unit identifier column
        pad: if True, zero-pad short sequences; if False, exclude them

    Returns:
        X: np.ndarray of shape (n_samples, window_size, n_features)
        y: np.ndarray of shape (n_samples,)
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in [unit_col, 'cycle', target_col]]

    n_features = len(feature_cols)
    X_list = []
    y_list = []
    units_padded = 0
    units_excluded = 0

    for unit_id, unit_df in df.groupby(unit_col):
        unit_df = unit_df.sort_values('cycle')
        values = unit_df[feature_cols].values      # (T, n_features)
        targets = unit_df[target_col].values       # (T,)
        T = len(values)

        if T < window_size and not pad:
            units_excluded += 1
            continue

        if pad:
            # Always prepend (W-1) rows of zeros so that every time step
            # from t=0 to t=T-1 produces a full window
            pad_rows = window_size - 1
            values = np.vstack([np.zeros((pad_rows, n_features)), values])
            units_padded += 1 if T < window_size else 0

            for t in range(T):
                start = t  # in padded array, window for original t starts at t
                end = t + window_size
                X_list.append(values[start:end])
                y_list.append(targets[t])
        else:
            # No padding: skip first (W-1) time steps
            for t in range(window_size - 1, T):
                start = t - window_size + 1
                X_list.append(values[start:t + 1])
                y_list.append(targets[t])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    n_units = df[unit_col].nunique()
    print(f"\n[Sliding Window] W={window_size}, features={n_features}")
    print(f"  Units: {n_units} total, {units_padded} padded, {units_excluded} excluded")
    print(f"  Output: X={X.shape}, y={y.shape}")

    return X, y


def evaluate_per_unit(y_true: np.ndarray,
                      y_pred: np.ndarray,
                      df: 'pd.DataFrame',
                      window_size: int = 30,
                      pad: bool = True,
                      unit_col: str = 'unit') -> dict:
    """
    Compute standard per-unit evaluation metrics (last prediction per unit).

    This is the standard evaluation protocol for RUL prediction:
    one prediction per unit, compared against the true RUL at truncation.

    Args:
        y_true: true RUL from create_sliding_windows (n_samples,)
        y_pred: predicted RUL (n_samples,)
        df: the preprocessed DataFrame BEFORE sliding window (to count rows per unit)
        window_size: W used in create_sliding_windows
        pad: must match the pad parameter used in create_sliding_windows
        unit_col: unit identifier column

    Returns:
        dict with rmse_last, score_last, rmse_mean, score_mean, n_units, details
    """
    # Count windows per unit (must match sliding window logic)
    windows_per_unit = []
    unit_ids = []
    for u in sorted(df[unit_col].unique()):
        T = len(df[df[unit_col] == u])
        if pad:
            n_windows = T
        else:
            n_windows = max(T - (window_size - 1), 0)
        windows_per_unit.append(n_windows)
        unit_ids.append(u)

    # Verify alignment
    total = sum(windows_per_unit)
    assert total == len(y_true), (
        f"Window count mismatch: sum={total}, y_true={len(y_true)}. "
        f"Check that pad={pad} matches the sliding window call."
    )

    # Split by unit
    splits = np.cumsum(windows_per_unit)[:-1]
    preds_per_unit = np.split(y_pred, splits)
    true_per_unit = np.split(y_true, splits)

    # Last and mean predictions
    valid_units = [(u, p, t) for u, p, t in zip(unit_ids, preds_per_unit, true_per_unit)
                   if len(p) > 0]

    preds_last = np.array([p[-1] for _, p, _ in valid_units])
    true_last = np.array([t[-1] for _, _, t in valid_units])
    preds_mean = np.array([np.mean(p) for _, p, _ in valid_units])
    valid_ids = [u for u, _, _ in valid_units]

    # Metrics
    def _nasa_score(y_t, y_p):
        diff = y_p - y_t
        return np.sum(np.where(diff < 0, np.exp(-diff / 13) - 1, np.exp(diff / 10) - 1))

    rmse_last = float(np.sqrt(np.mean((true_last - preds_last) ** 2)))
    rmse_mean = float(np.sqrt(np.mean((true_last - preds_mean) ** 2)))
    score_last = float(_nasa_score(true_last, preds_last))
    score_mean = float(_nasa_score(true_last, preds_mean))
    mae_last = float(np.mean(np.abs(true_last - preds_last)))

    n_valid = len(valid_units)
    n_total = len(unit_ids)

    print(f"\n=== Per-Unit Evaluation ({n_valid}/{n_total} units) ===")
    print(f"  Last window:  RMSE={rmse_last:.2f}  MAE={mae_last:.2f}  Score={score_last:.2f}")
    print(f"  Mean window:  RMSE={rmse_mean:.2f}  Score={score_mean:.2f}")

    return {
        'rmse_last': rmse_last,
        'mae_last': mae_last,
        'score_last': score_last,
        'rmse_mean': rmse_mean,
        'score_mean': score_mean,
        'n_units': n_valid,
        'preds_last': preds_last,
        'true_last': true_last,
        'unit_ids': valid_ids,
    }
