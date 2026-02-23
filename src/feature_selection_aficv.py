"""
AFICv Feature Selection — Aggregated Feature Importances with Cross-Validation
Implements Section III-B4, Eq. 13 of the methodology.

Two modes:
  - Global:      AFICv on all features together (original, biased toward high-rate)
  - Stratified:  AFICv separately on sensor and BI groups, then combine
                 (handles multi-rate bias — recommended for BI-fused data)

Author: Fatima Khadija Benzine
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor


class AFICvFeatureSelector:
    """
    Aggregated Feature Importances with Cross-Validation (AFICv).

    Usage (stratified — recommended):
        selector = AFICvFeatureSelector(base_learner='xgboost')
        selected = selector.select_features_stratified(
            data, sensor_cols, bi_cols, setting_cols,
            target_col='rul', group_col='unit'
        )
        df_filtered = selector.transform(data, keep_cols=['unit', 'cycle', 'rul'])

    Usage (global — for comparison):
        selected = selector.select_features(data, all_feature_cols, target_col='rul')
    """

    def __init__(self,
                 base_learner: str = 'xgboost',
                 n_folds: int = 5,
                 cumulative_threshold: float = 0.70,
                 random_state: int = 42):
        """
        Args:
            base_learner: 'xgboost', 'random_forest', or 'gradient_boosting'
            n_folds: number of CV folds (K)
            cumulative_threshold: cumulative importance cutoff (default 70%)
            random_state: for reproducibility
        """
        self.base_learner = base_learner
        self.n_folds = n_folds
        self.cumulative_threshold = cumulative_threshold
        self.random_state = random_state

        self.selected_features = None
        self.importance_df = None
        self.importance_sensor = None
        self.importance_bi = None
        self.selection_report = {}

    def _get_learner(self):
        """Instantiate the base learner."""
        if self.base_learner == 'xgboost':
            return XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=0,
            )
        elif self.base_learner == 'random_forest':
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1,
            )
        elif self.base_learner == 'gradient_boosting':
            return GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.random_state,
            )
        else:
            raise ValueError(f"Unknown learner: {self.base_learner}")

    def _run_aficv(self,
                   data: pd.DataFrame,
                   feature_cols: List[str],
                   target_col: str,
                   group_col: str,
                   label: str = '') -> pd.DataFrame:
        """
        Core AFICv routine (Eq. 13). Returns importance DataFrame.
        """
        prefix = f"  [{label}] " if label else "  "

        X = data[feature_cols].values
        y = data[target_col].values

        units = data[group_col].values
        unique_units = np.unique(units)
        kf = KFold(n_splits=self.n_folds, shuffle=True,
                    random_state=self.random_state)

        fold_importances = np.zeros((self.n_folds, len(feature_cols)))

        for k, (train_idx, val_idx) in enumerate(kf.split(unique_units)):
            train_units = unique_units[train_idx]
            val_units = unique_units[val_idx]

            train_mask = np.isin(units, train_units)
            val_mask = np.isin(units, val_units)

            X_train, y_train = X[train_mask], y[train_mask]
            X_val, y_val = X[val_mask], y[val_mask]

            model = self._get_learner()
            model.fit(X_train, y_train)
            fold_importances[k] = model.feature_importances_

            score = model.score(X_val, y_val)
            print(f"{prefix}Fold {k+1}/{self.n_folds}: R²={score:.4f}")

        mean_imp = fold_importances.mean(axis=0)
        std_imp = fold_importances.std(axis=0)

        imp_df = pd.DataFrame({
            'feature': feature_cols,
            'mean_importance': mean_imp,
            'std_importance': std_imp,
        }).sort_values('mean_importance', ascending=False).reset_index(drop=True)

        total = imp_df['mean_importance'].sum()
        imp_df['normalized'] = imp_df['mean_importance'] / total
        imp_df['cumulative'] = imp_df['normalized'].cumsum()

        return imp_df

    def _apply_threshold(self, imp_df: pd.DataFrame,
                         threshold: float = None) -> List[str]:
        """Apply cumulative threshold and return selected feature names."""
        thresh = threshold if threshold is not None else self.cumulative_threshold
        mask = imp_df['cumulative'] <= thresh
        n_selected = max(mask.sum(), 1)
        if mask.sum() < len(imp_df):
            n_selected = mask.sum() + 1  # include the one that crosses
        return imp_df.iloc[:n_selected]['feature'].tolist()

    # ==================================================================
    # Global mode (single AFICv on all features)
    # ==================================================================

    def select_features(self,
                        data: pd.DataFrame,
                        feature_cols: List[str],
                        target_col: str = 'rul',
                        group_col: str = 'unit') -> List[str]:
        """
        Global AFICv on all features together.
        Note: biased toward high-frequency features (sensors).
        """
        print(f"\n=== AFICv Feature Selection (Global) ===")
        print(f"  Learner: {self.base_learner}, K={self.n_folds}, "
              f"threshold={self.cumulative_threshold*100:.0f}%")
        print(f"  Candidates: {len(feature_cols)}")

        self.importance_df = self._run_aficv(
            data, feature_cols, target_col, group_col, label='Global')

        selected = self._apply_threshold(self.importance_df)
        self.selected_features = selected

        coverage = self.importance_df.iloc[:len(selected)]['cumulative'].iloc[-1]
        self.selection_report = {
            'mode': 'global',
            'n_candidates': len(feature_cols),
            'n_selected': len(selected),
            'cumulative_coverage': coverage,
        }

        print(f"\n  Selected {len(selected)}/{len(feature_cols)} "
              f"(coverage={coverage*100:.1f}%)")
        return selected

    # ==================================================================
    # Stratified mode (sensor and BI selected independently)
    # ==================================================================

    def select_features_stratified(self,
                                   data: pd.DataFrame,
                                   sensor_cols: List[str],
                                   bi_cols: List[str],
                                   setting_cols: List[str] = None,
                                   target_col: str = 'rul',
                                   group_col: str = 'unit',
                                   sensor_threshold: float = None,
                                   bi_threshold: float = None) -> List[str]:
        """
        Stratified AFICv: run separately on sensor/setting and BI groups,
        then combine. Avoids multi-rate frequency bias.

        Args:
            data: fused DataFrame
            sensor_cols: sensor feature names
            bi_cols: BI feature names (after one-hot encoding)
            setting_cols: operational setting names
            target_col: RUL column
            group_col: unit column
            sensor_threshold: override threshold for sensor group (default: same)
            bi_threshold: override threshold for BI group (default: same)

        Returns:
            Combined list of selected features
        """
        if setting_cols is None:
            setting_cols = []

        s_thresh = sensor_threshold if sensor_threshold is not None else self.cumulative_threshold
        b_thresh = bi_threshold if bi_threshold is not None else self.cumulative_threshold

        tech_cols = sensor_cols + setting_cols

        print(f"\n{'='*60}")
        print(f"AFICv Feature Selection (Stratified)")
        print(f"{'='*60}")
        print(f"  Learner: {self.base_learner}, K={self.n_folds}")
        print(f"  Sensor/Setting: {len(tech_cols)} candidates, "
              f"threshold={s_thresh*100:.0f}%")
        print(f"  BI:             {len(bi_cols)} candidates, "
              f"threshold={b_thresh*100:.0f}%")

        # ---- Group 1: Sensor + Settings ----
        print(f"\n--- Sensor/Setting Group ({len(tech_cols)} features) ---")
        self.importance_sensor = self._run_aficv(
            data, tech_cols, target_col, group_col, label='Sensor')
        sensor_selected = self._apply_threshold(self.importance_sensor, s_thresh)

        s_cov = self.importance_sensor.iloc[:len(sensor_selected)]['cumulative'].iloc[-1]
        print(f"\n  Sensor selected: {len(sensor_selected)}/{len(tech_cols)} "
              f"(coverage={s_cov*100:.1f}%)")
        print(f"  → {sensor_selected}")

        # ---- Group 2: BI ----
        print(f"\n--- BI Group ({len(bi_cols)} features) ---")
        self.importance_bi = self._run_aficv(
            data, bi_cols, target_col, group_col, label='BI')
        bi_selected = self._apply_threshold(self.importance_bi, b_thresh)

        b_cov = self.importance_bi.iloc[:len(bi_selected)]['cumulative'].iloc[-1]
        print(f"\n  BI selected: {len(bi_selected)}/{len(bi_cols)} "
              f"(coverage={b_cov*100:.1f}%)")
        print(f"  → {bi_selected}")

        # ---- Combine ----
        selected = sensor_selected + bi_selected

        # Build combined importance table
        imp_s = self.importance_sensor.copy()
        imp_s['group'] = 'sensor/setting'
        imp_b = self.importance_bi.copy()
        imp_b['group'] = 'BI'
        self.importance_df = pd.concat([imp_s, imp_b], ignore_index=True)
        self.importance_df['selected'] = self.importance_df['feature'].isin(selected)

        self.selected_features = selected
        self.selection_report = {
            'mode': 'stratified',
            'sensor_candidates': len(tech_cols),
            'sensor_selected': len(sensor_selected),
            'sensor_coverage': s_cov,
            'sensor_features': sensor_selected,
            'bi_candidates': len(bi_cols),
            'bi_selected': len(bi_selected),
            'bi_coverage': b_cov,
            'bi_features': bi_selected,
            'total_selected': len(selected),
        }

        print(f"\n{'='*60}")
        print(f"  TOTAL: {len(selected)} features "
              f"({len(sensor_selected)} sensor/setting + {len(bi_selected)} BI)")
        print(f"{'='*60}")

        return selected

    # ==================================================================
    # Transform & reporting
    # ==================================================================

    def transform(self, data: pd.DataFrame,
                  keep_cols: List[str] = None) -> pd.DataFrame:
        """Keep only selected features + specified columns."""
        if self.selected_features is None:
            raise ValueError("Must call select_features or "
                             "select_features_stratified before transform")
        if keep_cols is None:
            keep_cols = []
        cols = [c for c in keep_cols + self.selected_features if c in data.columns]
        return data[cols]

    def get_importance_table(self, group: str = None) -> pd.DataFrame:
        """
        Return importance ranking table.
        Args:
            group: None=combined, 'sensor'=sensor only, 'bi'=BI only
        """
        if group == 'sensor' and self.importance_sensor is not None:
            return self.importance_sensor
        elif group == 'bi' and self.importance_bi is not None:
            return self.importance_bi
        return self.importance_df

    def get_report(self) -> Dict:
        """Return selection summary."""
        return self.selection_report
