"""
Business Intelligence Data Loader and Fusion Module
Handles loading, temporal alignment, and fusion of BI data with sensor data

Implements Section III-B3 of the methodology:
- Source-driven update frequencies (variable-specific Delta_k)
- Forward-fill temporal alignment (Eq. 3-4)
- Feature-level fusion via concatenation (Eq. 6)
- One-hot encoding of categorical BI variables (Eq. 5)

Author: Fatima Khadija Benzine
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys
current_file = Path(__file__).resolve()
src_dir = current_file.parent
project_root = src_dir.parent
sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(project_root))

from config import RAW_DATA_DIR


# ============================================================================
# BI Variable Configuration
# ============================================================================

# Variable-specific update frequencies (Delta_k in cycles)
# Determined by source enterprise systems, NOT tunable hyperparameters
BI_DELTA_CONFIG = {
    # ERP cost structure — monthly refresh (~50 flights)
    'pm_cost':              {'delta': 50, 'source': 'ERP', 'type': 'continuous'},
    'cm_cost':              {'delta': 50, 'source': 'ERP', 'type': 'continuous'},
    'labor_rate_standard':  {'delta': 50, 'source': 'ERP', 'type': 'continuous'},
    'labor_rate_overtime':  {'delta': 50, 'source': 'ERP', 'type': 'continuous'},
    # MES production — per shift (~10 flights)
    'production_priority':  {'delta': 10, 'source': 'MES', 'type': 'categorical'},
    'downtime_penalty':     {'delta': 10, 'source': 'MES', 'type': 'continuous'},
    'maintenance_window':   {'delta': 10, 'source': 'MES', 'type': 'binary'},
    'shift_pattern':        {'delta': 10, 'source': 'MES', 'type': 'categorical'},
    # Inventory / HR — daily-weekly (~25 flights)
    'revenue_per_hour':     {'delta': 25, 'source': 'ERP/Finance', 'type': 'continuous'},
    'spare_parts_available':{'delta': 25, 'source': 'ERP/Inventory', 'type': 'binary'},
    'spare_parts_lead_time':{'delta': 25, 'source': 'ERP/Inventory', 'type': 'continuous'},
    'technician_available': {'delta': 25, 'source': 'HR/Scheduling', 'type': 'binary'},
    'contract_penalty_active': {'delta': 25, 'source': 'ERP/Contracts', 'type': 'binary'},
}

# Variables requiring one-hot encoding (Eq. 5)
CATEGORICAL_BI_VARS = ['production_priority', 'shift_pattern']

# Continuous BI variables that NEED normalization (Section III-B4)
# Binary and one-hot encoded variables are excluded (already 0/1)
CONTINUOUS_BI_VARS = [
    name for name, cfg in BI_DELTA_CONFIG.items()
    if cfg['type'] == 'continuous'
]

# Binary BI variables — no normalization needed
BINARY_BI_VARS = [
    name for name, cfg in BI_DELTA_CONFIG.items()
    if cfg['type'] == 'binary'
]

# Variables exempt from variance-based filtering (Section III-B4)
BI_FEATURE_NAMES = list(BI_DELTA_CONFIG.keys())


class BIDataLoader:
    """
    Loads and aligns Business Intelligence data with C-MAPSS sensor data.
    """

    def __init__(self, bi_data_dir: Path = None):
        if bi_data_dir is None:
            self.bi_data_dir = Path(RAW_DATA_DIR) / "BI_data"
        else:
            self.bi_data_dir = Path(bi_data_dir)

    def load_bi(self, dataset_name: str, split: str = 'train') -> pd.DataFrame:
        """
        Load BI CSV for a given C-MAPSS sub-dataset and split.

        Args:
            dataset_name: 'FD001', 'FD002', 'FD003', or 'FD004'
            split: 'train' or 'test'

        Returns:
            DataFrame with columns [unit_id, cycle, <bi_vars>...]
        """
        filepath = self.bi_data_dir / f"BI_{dataset_name}_{split}.csv"
        if not filepath.exists():
            # Fallback to old naming convention (single file)
            filepath = self.bi_data_dir / f"BI_{dataset_name}.csv"
            if not filepath.exists():
                raise FileNotFoundError(
                    f"BI data not found for {dataset_name} ({split})\n"
                    f"Looked for: BI_{dataset_name}_{split}.csv or BI_{dataset_name}.csv\n"
                    f"in {self.bi_data_dir}"
                )

        df = pd.read_csv(filepath)

        # Validate expected columns
        expected = ['unit_id', 'cycle'] + BI_FEATURE_NAMES
        missing = [c for c in expected if c not in df.columns]
        if missing:
            raise ValueError(f"BI file missing columns: {missing}")

        print(f"  BI data loaded: {len(df)} rows, {df['unit_id'].nunique()} units")
        return df

    def get_delta_summary(self) -> pd.DataFrame:
        """Return a summary table of BI variable update frequencies."""
        rows = []
        for var, cfg in BI_DELTA_CONFIG.items():
            rows.append({
                'variable': var,
                'delta_cycles': cfg['delta'],
                'source_system': cfg['source'],
                'var_type': cfg['type'],
            })
        return pd.DataFrame(rows)


class BIFusionPipeline:
    """
    Fuses BI data with sensor data following Section III-B3.

    Pipeline:
        1. Load BI data
        2. Align column naming (unit_id -> unit)
        3. Merge on (unit, cycle) — feature-level fusion (Eq. 6)
        4. One-hot encode categorical BI variables (Eq. 5)
    """

    def __init__(self, bi_data_dir: Path = None):
        self.bi_loader = BIDataLoader(bi_data_dir)
        self.encoded_columns = []  # tracks one-hot columns after encoding

    def fuse(self,
             sensor_df: pd.DataFrame,
             dataset_name: str,
             split: str = 'train',
             encode_categoricals: bool = True) -> pd.DataFrame:
        """
        Fuse sensor data with BI data.

        Args:
            sensor_df: C-MAPSS sensor DataFrame with columns [unit, cycle, ...]
            dataset_name: 'FD001', 'FD002', 'FD003', or 'FD004'
            split: 'train' or 'test'
            encode_categoricals: whether to one-hot encode categorical BI vars

        Returns:
            Fused DataFrame with sensor + BI features
        """
        print(f"\n=== BI Fusion: {dataset_name} ({split}) ===")
        print(f"  Sensor data: {sensor_df.shape}")

        # Step 1: Load BI
        bi_df = self.bi_loader.load_bi(dataset_name, split=split)

        # Step 2: Align column names (data_loader uses 'unit', BI uses 'unit_id')
        bi_df = bi_df.rename(columns={'unit_id': 'unit'})

        # Step 3: Merge — inner join keeps only matching (unit, cycle) pairs
        fused = sensor_df.merge(bi_df, on=['unit', 'cycle'], how='left')

        n_missing = fused[BI_FEATURE_NAMES[0]].isna().sum()
        if n_missing > 0:
            print(f"  Warning: {n_missing} rows without BI match "
                  f"({n_missing/len(fused)*100:.1f}%) — filling forward per unit")
            # Forward fill within each unit for unmatched cycles
            bi_cols = [c for c in BI_FEATURE_NAMES if c in fused.columns]
            fused[bi_cols] = fused.groupby('unit')[bi_cols].ffill()
            # Backfill any remaining NaN at the start of a unit's life
            fused[bi_cols] = fused.groupby('unit')[bi_cols].bfill()

        # Step 4: One-hot encode categoricals (Eq. 5)
        if encode_categoricals:
            fused = self._encode_categoricals(fused)

        print(f"  Fused data: {fused.shape}")
        print(f"  Features: {len([c for c in fused.columns if c.startswith('sensor_')])} sensor "
              f"+ {len(self.get_bi_columns(fused))} BI")

        return fused

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode categorical BI variables (Eq. 5)."""
        cats_present = [c for c in CATEGORICAL_BI_VARS if c in df.columns]
        if not cats_present:
            return df

        # Convert to int first to get clean dummy names
        for col in cats_present:
            df[col] = df[col].astype(int)

        df = pd.get_dummies(df, columns=cats_present,
                            prefix={c: c for c in cats_present})

        # Track encoded column names
        self.encoded_columns = [c for c in df.columns
                                if any(c.startswith(cat + '_') for cat in cats_present)]

        return df

    @staticmethod
    def get_bi_columns(df: pd.DataFrame) -> List[str]:
        """Get all BI-related column names from a fused DataFrame."""
        bi_cols = []
        for col in df.columns:
            # Original BI columns
            if col in BI_FEATURE_NAMES:
                bi_cols.append(col)
            # One-hot encoded versions
            elif any(col.startswith(cat + '_') for cat in CATEGORICAL_BI_VARS):
                bi_cols.append(col)
        return bi_cols

    @staticmethod
    def get_sensor_columns(df: pd.DataFrame) -> List[str]:
        """Get sensor column names."""
        return [c for c in df.columns if c.startswith('sensor_')]

    @staticmethod
    def get_setting_columns(df: pd.DataFrame) -> List[str]:
        """Get operational setting column names."""
        return [c for c in df.columns if c.startswith('setting_')]
