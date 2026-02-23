"""
BI-Aware Feature Selection Module
Extends the existing FeatureSelector with BI-specific logic from Section III-B4:

- BI variables are EXEMPT from variance-based filtering
  (their low variance is inherent to enterprise systems, not uninformative)
- Correlation-based filtering applies to ALL features (sensor + BI)
- When a sensor and BI feature are highly correlated, the BI feature is retained
  (business interpretability is prioritized)

Author: Fatima Khadija Benzine
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


class BIAwareFeatureSelector:
    """
    Feature selector that treats BI and sensor features differently.

    Variance filtering:  sensor features only (BI exempt)
    Correlation filtering: all features, but BI prioritized over sensor
    """

    def __init__(self,
                 variance_threshold: float = 0.01,
                 correlation_threshold: float = 0.95):
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.selected_features = None
        self.selection_report = {}

    def select_features(self,
                        data: pd.DataFrame,
                        sensor_cols: List[str],
                        bi_cols: List[str],
                        setting_cols: List[str] = None,
                        exclude_cols: List[str] = None) -> List[str]:
        """
        Select features with BI-aware logic.

        Args:
            data: Fused DataFrame (sensor + BI)
            sensor_cols: sensor feature column names
            bi_cols: BI feature column names (exempt from variance filter)
            setting_cols: operational setting column names
            exclude_cols: non-feature columns (unit, cycle, rul)

        Returns:
            List of selected feature names
        """
        if exclude_cols is None:
            exclude_cols = []
        if setting_cols is None:
            setting_cols = []

        all_feature_cols = sensor_cols + bi_cols + setting_cols
        print(f"\n=== BI-Aware Feature Selection ===")
        print(f"  Input: {len(sensor_cols)} sensor + {len(bi_cols)} BI "
              f"+ {len(setting_cols)} setting = {len(all_feature_cols)} total")

        # ---- Step 1: Variance filtering (SENSOR + SETTINGS ONLY) ----
        sensor_setting_cols = sensor_cols + setting_cols
        variances = data[sensor_setting_cols].var()
        low_var = variances[variances <= self.variance_threshold].index.tolist()
        high_var_sensor = [c for c in sensor_setting_cols if c not in low_var]

        self.selection_report['variance_removed'] = low_var
        print(f"  Variance filter (sensor/settings only):")
        print(f"    Removed {len(low_var)}: {low_var}")
        print(f"    Kept {len(high_var_sensor)} sensor/setting features")
        print(f"    BI features: {len(bi_cols)} (all exempt, all kept)")

        # Candidates after variance filtering: surviving sensors + ALL BI
        candidates = high_var_sensor + bi_cols

        # ---- Step 2: Correlation filtering (ALL features) ----
        if len(candidates) > 1:
            corr_matrix = data[candidates].corr().abs()
            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )

            to_drop = set()
            for col in upper.columns:
                correlated_with = upper.index[upper[col] > self.correlation_threshold].tolist()
                for corr_col in correlated_with:
                    # If one is BI and the other is sensor → drop the sensor
                    col_is_bi = col in bi_cols
                    corr_is_bi = corr_col in bi_cols

                    if col_is_bi and not corr_is_bi:
                        to_drop.add(corr_col)  # drop sensor
                    elif corr_is_bi and not col_is_bi:
                        to_drop.add(col)  # drop sensor
                    else:
                        # Both same type — drop the one that appears later
                        to_drop.add(corr_col)

            selected = [c for c in candidates if c not in to_drop]
            self.selection_report['correlation_removed'] = list(to_drop)
            print(f"  Correlation filter (tau={self.correlation_threshold}):")
            print(f"    Removed {len(to_drop)}: {list(to_drop)}")
        else:
            selected = candidates
            self.selection_report['correlation_removed'] = []

        self.selected_features = selected
        self.selection_report['selected'] = selected

        n_sensor = len([c for c in selected if c in sensor_cols + setting_cols])
        n_bi = len([c for c in selected if c in bi_cols])
        print(f"  Final: {len(selected)} features ({n_sensor} sensor/setting + {n_bi} BI)")

        return selected

    def transform(self, data: pd.DataFrame,
                  keep_cols: List[str] = None) -> pd.DataFrame:
        """Keep only selected features + specified columns."""
        if self.selected_features is None:
            raise ValueError("Must call select_features before transform")
        if keep_cols is None:
            keep_cols = []

        cols = [c for c in keep_cols + self.selected_features if c in data.columns]
        return data[cols]

    def get_report(self) -> Dict:
        """Return a summary of the selection process."""
        return self.selection_report
