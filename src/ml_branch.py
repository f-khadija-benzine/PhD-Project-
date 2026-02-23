"""
ML Branch for Hybrid RUL Prediction (Section III-C3)

Implements the machine learning component of the hybrid model:
    - Flattens sliding window input for tabular learners
    - Supports XGBoost and Random Forest
    - Provides feature importance for comparison with attention weights
    - Hybrid fusion: α × RUL_DL + (1-α) × RUL_ML

Author: Fatima Khadija Benzine
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error


class MLBranch:
    """
    ML branch of the hybrid model.

    Takes the same sliding window input as the DL branch,
    flattens it, and trains a tree-based model.

    Two flattening strategies:
        - 'flatten': concatenate all time steps → (batch, W * n_features)
        - 'statistics': extract statistical features per variable
                        (mean, std, min, max, last, trend) → (batch, n_features * 6)
    """

    def __init__(self,
                 model_type: str = 'xgboost',
                 flatten_strategy: str = 'statistics',
                 random_state: int = 42,
                 **model_params):
        """
        Args:
            model_type: 'xgboost' or 'random_forest'
            flatten_strategy: 'flatten' or 'statistics'
            random_state: random seed
            **model_params: passed to the underlying model
        """
        self.model_type = model_type
        self.flatten_strategy = flatten_strategy
        self.random_state = random_state
        self.model = None
        self.feature_names_flat = None

        # Default parameters (CPU-friendly)
        if model_type == 'xgboost':
            import xgboost as xgb
            defaults = {
                'n_estimators': 300,
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': random_state,
                'n_jobs': -1,
            }
            defaults.update(model_params)
            self.model = xgb.XGBRegressor(**defaults)

        elif model_type == 'random_forest':
            from sklearn.ensemble import RandomForestRegressor
            defaults = {
                'n_estimators': 300,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': random_state,
                'n_jobs': -1,
            }
            defaults.update(model_params)
            self.model = RandomForestRegressor(**defaults)

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def _flatten(self, X: np.ndarray, feature_names: List[str] = None) -> np.ndarray:
        """
        Convert 3D sliding window to 2D tabular data.

        Args:
            X: (n_samples, W, n_features)
            feature_names: original feature names

        Returns:
            X_flat: (n_samples, n_flat_features)
        """
        n_samples, W, n_features = X.shape

        if self.flatten_strategy == 'flatten':
            # Simple concatenation
            X_flat = X.reshape(n_samples, -1)
            if feature_names and self.feature_names_flat is None:
                self.feature_names_flat = [
                    f"{name}_t{t}" for t in range(W) for name in feature_names
                ]

        elif self.flatten_strategy == 'statistics':
            # Statistical features per variable
            stats = []
            names = []

            for j in range(n_features):
                col = X[:, :, j]  # (n_samples, W)
                fname = feature_names[j] if feature_names else f"f{j}"

                stats.append(np.mean(col, axis=1))
                names.append(f"{fname}_mean")

                stats.append(np.std(col, axis=1))
                names.append(f"{fname}_std")

                stats.append(np.min(col, axis=1))
                names.append(f"{fname}_min")

                stats.append(np.max(col, axis=1))
                names.append(f"{fname}_max")

                stats.append(col[:, -1])  # last value
                names.append(f"{fname}_last")

                # Trend: slope of linear fit over window
                t_axis = np.arange(W, dtype=np.float32)
                t_mean = t_axis.mean()
                t_var = np.sum((t_axis - t_mean) ** 2)
                col_demean = col - col.mean(axis=1, keepdims=True)
                slope = np.sum(col_demean * (t_axis - t_mean), axis=1) / (t_var + 1e-8)
                stats.append(slope)
                names.append(f"{fname}_trend")

            X_flat = np.column_stack(stats)
            if self.feature_names_flat is None:
                self.feature_names_flat = names

        else:
            raise ValueError(f"Unknown flatten_strategy: {self.flatten_strategy}")

        return X_flat

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: List[str] = None,
            eval_set: Tuple[np.ndarray, np.ndarray] = None,
            verbose: bool = True):
        """
        Train the ML model.

        Args:
            X: (n_samples, W, n_features) sliding window input
            y: (n_samples,) RUL targets
            feature_names: original feature names
            eval_set: optional (X_val, y_val) for early stopping
            verbose: print progress
        """
        X_flat = self._flatten(X, feature_names)

        if verbose:
            print(f"\n[ML Branch] {self.model_type} — {self.flatten_strategy}")
            print(f"  Input: {X.shape} → Flattened: {X_flat.shape}")

        if eval_set is not None and self.model_type == 'xgboost':
            X_val_flat = self._flatten(eval_set[0], feature_names)
            self.model.fit(
                X_flat, y,
                eval_set=[(X_val_flat, eval_set[1])],
                verbose=False,
            )
        else:
            self.model.fit(X_flat, y)

        if verbose:
            print(f"  Training complete ✓")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict RUL.

        Args:
            X: (n_samples, W, n_features)

        Returns:
            y_pred: (n_samples,)
        """
        X_flat = self._flatten(X)
        return self.model.predict(X_flat)

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance from the tree model.

        Returns:
            DataFrame with feature name, importance, and original feature type
        """
        importances = self.model.feature_importances_
        names = self.feature_names_flat or [f"f{i}" for i in range(len(importances))]

        df = pd.DataFrame({
            'feature': names,
            'importance': importances,
        }).sort_values('importance', ascending=False).reset_index(drop=True)

        # Extract original feature name and type
        def original_name(flat_name):
            # Remove _mean, _std, _min, _max, _last, _trend, _tN suffixes
            for suffix in ['_mean', '_std', '_min', '_max', '_last', '_trend']:
                if flat_name.endswith(suffix):
                    return flat_name[:-len(suffix)]
            # Handle _tN for flatten strategy
            parts = flat_name.rsplit('_t', 1)
            return parts[0] if len(parts) == 2 and parts[1].isdigit() else flat_name

        def feature_type(name):
            if name.startswith('sensor_'):
                return 'sensor'
            elif name.startswith('setting_'):
                return 'setting'
            else:
                return 'BI'

        df['original_feature'] = df['feature'].apply(original_name)
        df['type'] = df['original_feature'].apply(feature_type)

        return df.head(top_n)


class HybridPredictor:
    """
    Hybrid fusion of DL and ML branches (Section III-C5).

    RUL_final = α × RUL_DL + (1 - α) × RUL_ML

    α can be fixed or optimized on validation data.
    """

    def __init__(self, alpha: float = 0.5):
        """
        Args:
            alpha: weight for DL branch (0 = ML only, 1 = DL only)
        """
        self.alpha = alpha
        self.optimal_alpha = None

    def predict(self, y_dl: np.ndarray, y_ml: np.ndarray,
                alpha: float = None) -> np.ndarray:
        """
        Fuse DL and ML predictions.

        Args:
            y_dl: DL branch predictions (n_samples,)
            y_ml: ML branch predictions (n_samples,)
            alpha: override weight (uses self.alpha if None)

        Returns:
            y_fused: (n_samples,)
        """
        a = alpha if alpha is not None else self.alpha
        return a * y_dl + (1 - a) * y_ml

    def optimize_alpha(self, y_dl: np.ndarray, y_ml: np.ndarray,
                       y_true: np.ndarray,
                       metric: str = 'rmse',
                       n_steps: int = 101) -> float:
        """
        Find optimal α on validation data via grid search.

        Args:
            y_dl: DL predictions on validation set
            y_ml: ML predictions on validation set
            y_true: true RUL values
            metric: 'rmse' or 'score'
            n_steps: grid resolution

        Returns:
            optimal_alpha: best α value
        """
        alphas = np.linspace(0, 1, n_steps)
        best_metric = float('inf')
        best_alpha = 0.5

        for a in alphas:
            y_fused = a * y_dl + (1 - a) * y_ml

            if metric == 'rmse':
                val = np.sqrt(mean_squared_error(y_true, y_fused))
            elif metric == 'score':
                diff = y_fused - y_true
                val = np.sum(np.where(diff < 0, np.exp(-diff / 13) - 1,
                                      np.exp(diff / 10) - 1))
            else:
                raise ValueError(f"Unknown metric: {metric}")

            if val < best_metric:
                best_metric = val
                best_alpha = a

        self.optimal_alpha = best_alpha
        self.alpha = best_alpha

        print(f"\n[Hybrid] Optimal α = {best_alpha:.2f} "
              f"({metric} = {best_metric:.2f})")
        print(f"  α=0.0 (ML only): {metric}={self._compute(y_ml, y_true, metric):.2f}")
        print(f"  α=0.5 (equal):   {metric}={self._compute(0.5*y_dl + 0.5*y_ml, y_true, metric):.2f}")
        print(f"  α=1.0 (DL only): {metric}={self._compute(y_dl, y_true, metric):.2f}")

        return best_alpha

    def _compute(self, y_pred, y_true, metric):
        if metric == 'rmse':
            return np.sqrt(mean_squared_error(y_true, y_pred))
        diff = y_pred - y_true
        return np.sum(np.where(diff < 0, np.exp(-diff / 13) - 1,
                               np.exp(diff / 10) - 1))
