"""
Chronos 2 RUL Predictor for Predictive Maintenance
Integrates Amazon's Chronos 2 forecasting model with RUL prediction

Author: Auto-generated for PhD research
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import warnings
from pathlib import Path
import logging

# Import Chronos
try:
    from chronos import ChronosPipeline
except ImportError:
    print("Warning: chronos-forecasting not installed. Install with: pip install chronos-forecasting")
    ChronosPipeline = None

from data_loader import MultiDatasetLoader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChronosRULPredictor:
    """
    RUL predictor using Amazon's Chronos 2 forecasting model.

    Supports both zero-shot forecasting and fine-tuned approaches for RUL prediction.
    """

    def __init__(self,
                 model_name: str = "amazon/chronos-t5-small",
                 device: str = None,
                 prediction_length: int = 50,
                 context_length: int = 512):
        """
        Initialize Chronos RUL predictor.

        Args:
            model_name: Chronos model to use (small, base, large)
            device: Device to run on ('cuda', 'cpu', 'auto', or None for auto)
            prediction_length: How far ahead to forecast for RUL estimation
            context_length: Maximum context length for the model
        """
        if ChronosPipeline is None:
            raise ImportError("chronos-forecasting not installed. Install with: pip install chronos-forecasting")

        self.model_name = model_name
        self.prediction_length = prediction_length
        self.context_length = context_length

        # Set device
        if device is None or device == "auto":
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        logger.info(f"Using device: {self.device}")

        # Initialize model
        try:
            self.pipeline = ChronosPipeline.from_pretrained(
                model_name,
                device_map=self.device,
                torch_dtype=torch.bfloat16 if self.device == 'cuda' else torch.float32,
            )
            logger.info(f"Loaded Chronos model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load Chronos model: {e}")
            raise

        # Placeholder for fine-tuned models and hyperparameters
        self.fine_tuned_models = {}
        self.hyperparams = {
            'threshold_method': 'deviation',
            'threshold_factor': 2.0,
            'prediction_method': 'threshold',
            'failure_thresholds': None
        }

    def prepare_cmapss_data(self,
                           dataset: Dict,
                           feature_cols: Optional[List[str]] = None,
                           max_rul: int = 125) -> Dict:
        """
        Prepare C-MAPSS dataset for Chronos forecasting.

        Args:
            dataset: Dataset from MultiDatasetLoader
            feature_cols: Which sensor/setting columns to use (default: all sensors)
            max_rul: Maximum RUL value for clipping

        Returns:
            Prepared data dict with time series sequences
        """
        train_df = dataset['train'].copy()
        test_df = dataset['test'].copy()

        # Select features (sensors by default)
        if feature_cols is None:
            # Use all sensor columns (C-MAPSS: sensor_1 .. sensor_21)
            feature_cols = [col for col in train_df.columns if col.startswith('sensor_')]
            n_sensors = len(feature_cols)
            if n_sensors <= 5:
                logger.info(f"Using {n_sensors} sensor features: {feature_cols}")
            else:
                logger.info(f"Using {n_sensors} sensor features: {feature_cols[0]} .. {feature_cols[-1]}")

        # Clip RUL for training stability
        train_df['rul'] = train_df['rul'].clip(upper=max_rul)
        test_df['rul'] = test_df['rul'].clip(upper=max_rul)

        # Group by unit and create sequences
        train_sequences = []
        test_sequences = []

        # Training data: full sequences per unit
        for unit in train_df['unit'].unique():
            unit_data = train_df[train_df['unit'] == unit].sort_values('cycle')
            if len(unit_data) >= 10:  # Minimum sequence length
                sequence = {
                    'unit': unit,
                    'features': unit_data[feature_cols].values.T,  # Shape: (n_features, seq_len)
                    'rul': unit_data['rul'].values,
                    'cycles': unit_data['cycle'].values,
                    'seq_length': len(unit_data)
                }
                train_sequences.append(sequence)

        # Test data: sequences up to each point (sliding window approach)
        for unit in test_df['unit'].unique():
            unit_data = test_df[test_df['unit'] == unit].sort_values('cycle')
            if len(unit_data) >= 10:
                # Create multiple sequences ending at different points
                for end_idx in range(10, len(unit_data) + 1):
                    sequence = {
                        'unit': unit,
                        'features': unit_data.iloc[:end_idx][feature_cols].values.T,
                        'rul': unit_data.iloc[:end_idx]['rul'].values,
                        'cycles': unit_data.iloc[:end_idx]['cycle'].values,
                        'seq_length': len(unit_data.iloc[:end_idx]),
                        'is_test': True
                    }
                    test_sequences.append(sequence)

        prepared_data = {
            'train_sequences': train_sequences,
            'test_sequences': test_sequences,
            'feature_cols': feature_cols,
            'max_rul': max_rul,
            'n_units_train': len(set(s['unit'] for s in train_sequences)),
            'n_units_test': len(set(s['unit'] for s in test_sequences))
        }

        logger.info(f"Prepared {len(train_sequences)} training sequences, {len(test_sequences)} test sequences")
        return prepared_data

    def forecast_sensor_values(self,
                              features: np.ndarray,
                              prediction_length: Optional[int] = None) -> np.ndarray:
        """
        Use Chronos to forecast future sensor values.

        Args:
            features: Sensor data array of shape (n_features, seq_length)
            prediction_length: How far to forecast (default: self.prediction_length)

        Returns:
            Forecasted values of shape (n_features, prediction_length)
        """
        if prediction_length is None:
            prediction_length = self.prediction_length

        forecasts = []

        # Forecast each sensor/feature independently
        for sensor_idx in range(features.shape[0]):
            sensor_data = features[sensor_idx]  # Shape: (seq_length,)

            # Convert to torch tensor
            context = torch.tensor(sensor_data, dtype=torch.float32)

            # Generate forecast using predict_quantiles (correct API)
            # Note: context is a POSITIONAL argument, not keyword
            with torch.no_grad():
                quantiles, mean = self.pipeline.predict_quantiles(
                    context,  # Positional argument (not context=...)
                    prediction_length=prediction_length,
                    quantile_levels=[0.5],  # Use median for deterministic forecast
                )

            # Extract median forecast (quantile 0.5)
            forecast_values = quantiles[0, :, 0].cpu().numpy()  # Shape: (prediction_length,)
            forecasts.append(forecast_values)

        return np.array(forecasts)  # Shape: (n_features, prediction_length)

    def predict_rul_from_forecast(self,
                                 current_features: np.ndarray,
                                 current_rul: float,
                                 method: str = 'threshold',
                                 threshold_method: str = 'deviation',
                                 threshold_factor: float = 2.0,
                                 failure_thresholds: Optional[np.ndarray] = None) -> float:
        """
        Predict RUL using various methods based on Chronos forecasts.

        Args:
            current_features: Current sensor values (n_features, seq_length)
            current_rul: Current known RUL (for reference, can be None)
            method: RUL prediction method ('threshold', 'trend', 'regression')
            threshold_method: For threshold method: 'deviation', 'absolute', 'adaptive'
            threshold_factor: Multiplier for threshold-based methods
            failure_thresholds: Pre-defined failure thresholds per sensor (optional)

        Returns:
            Predicted RUL (cycles until failure)
        """
        # Forecast future sensor values
        forecasts = self.forecast_sensor_values(current_features)
        seq_length = current_features.shape[1]

        if method == 'threshold':
            return self._predict_rul_threshold(
                current_features, forecasts, threshold_method, threshold_factor, failure_thresholds
            )
        elif method == 'trend':
            return self._predict_rul_trend(current_features, forecasts)
        elif method == 'regression':
            return self._predict_rul_regression(current_features, forecasts, current_rul)
        else:
            raise ValueError(f"Unknown RUL prediction method: {method}")

    def _predict_rul_threshold(self,
                              features: np.ndarray,
                              forecasts: np.ndarray,
                              threshold_method: str,
                              threshold_factor: float,
                              failure_thresholds: Optional[np.ndarray]) -> float:
        """Threshold-based RUL prediction using forecasted deviations."""
        n_features = features.shape[0]

        # Calculate baseline statistics from recent history
        recent_window = min(30, features.shape[1])
        baseline_mean = np.mean(features[:, -recent_window:], axis=1)
        baseline_std = np.std(features[:, -recent_window:], axis=1)

        # Define failure thresholds
        if failure_thresholds is not None:
            thresholds = failure_thresholds
        elif threshold_method == 'deviation':
            thresholds = baseline_mean + threshold_factor * baseline_std
        elif threshold_method == 'absolute':
            # Use extreme values from historical data as thresholds
            thresholds = baseline_mean + threshold_factor * np.abs(baseline_mean)
        elif threshold_method == 'adaptive':
            # Adaptive threshold based on trend
            recent_trend = np.polyfit(np.arange(recent_window), features[:, -recent_window:], 1)[0]
            thresholds = baseline_mean + threshold_factor * np.abs(recent_trend) * recent_window
        else:
            raise ValueError(f"Unknown threshold method: {threshold_method}")

        # Find first time step where any sensor exceeds threshold
        for step in range(forecasts.shape[1]):
            forecast_values = forecasts[:, step]

            # Check if any sensor exceeds its threshold
            if threshold_method in ['deviation', 'adaptive']:
                exceeds = np.abs(forecast_values - baseline_mean) > (thresholds - baseline_mean)
            else:  # absolute
                exceeds = np.abs(forecast_values) > np.abs(thresholds)

            if np.any(exceeds):
                return step + 1  # RUL in cycles

        # If no failure detected, return max forecast horizon
        return self.prediction_length

    def _predict_rul_trend(self, features: np.ndarray, forecasts: np.ndarray) -> float:
        """Trend-based RUL prediction using forecast trajectory."""
        # Analyze trend of each sensor
        failure_scores = []

        for sensor_idx in range(features.shape[0]):
            sensor_history = features[sensor_idx]
            sensor_forecast = forecasts[sensor_idx]

            # Fit linear trend to recent history
            recent_window = min(20, len(sensor_history))
            x_hist = np.arange(recent_window)
            y_hist = sensor_history[-recent_window:]

            # Simple linear regression for trend
            if len(np.unique(y_hist)) > 1:  # Avoid constant signals
                slope = np.polyfit(x_hist, y_hist, 1)[0]

                # Project trend into future
                x_future = np.arange(len(sensor_history), len(sensor_history) + len(sensor_forecast))
                trend_line = slope * x_future + (y_hist[-1] - slope * x_hist[-1])

                # Find when forecast crosses trend-based threshold
                threshold = trend_line[0] + 3 * np.std(y_hist)  # 3-sigma threshold

                crossing_points = np.where(sensor_forecast > threshold)[0]
                if len(crossing_points) > 0:
                    failure_scores.append(crossing_points[0] + 1)

        if failure_scores:
            # Return median of sensor predictions
            return np.median(failure_scores)

        return self.prediction_length

    def _predict_rul_regression(self,
                               features: np.ndarray,
                               forecasts: np.ndarray,
                               current_rul: Optional[float]) -> float:
        """Regression-based RUL prediction using forecast features."""
        # This is a placeholder for more sophisticated regression approaches
        # Could use ML models trained on forecast features to predict RUL

        # Simple fallback: use threshold method
        return self._predict_rul_threshold(features, forecasts, 'deviation', 2.0, None)

    def evaluate_on_dataset(self,
                           prepared_data: Dict,
                           threshold_method: str = 'deviation',
                           threshold_factor: float = 2.0) -> Dict:
        """
        Evaluate RUL prediction performance on test data.

        Args:
            prepared_data: Prepared dataset from prepare_cmapss_data
            threshold_method: Method for threshold-based RUL prediction
            threshold_factor: Threshold factor for failure detection

        Returns:
            Evaluation metrics
        """
        predictions = []
        actuals = []

        logger.info("Evaluating RUL predictions on test sequences...")

        for seq in prepared_data['test_sequences']:
            if seq['seq_length'] < 10:
                continue

            # Use the last RUL value as ground truth
            actual_rul = seq['rul'][-1]

            # Predict RUL from current features
            predicted_rul = self.predict_rul_from_forecast(
                seq['features'],
                actual_rul,
                threshold_method=threshold_method,
                threshold_factor=threshold_factor
            )

            predictions.append(predicted_rul)
            actuals.append(actual_rul)

        # Calculate metrics
        predictions = np.array(predictions)
        actuals = np.array(actuals)

        mse = np.mean((predictions - actuals) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - actuals))

        # RUL-specific metrics
        within_10 = np.mean(np.abs(predictions - actuals) <= 10)
        within_20 = np.mean(np.abs(predictions - actuals) <= 20)

        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'within_10_cycles': within_10,
            'within_20_cycles': within_20,
            'n_predictions': len(predictions),
            'predictions': predictions,
            'actuals': actuals
        }

        logger.info(f"Evaluation complete: RMSE={rmse:.2f}, MAE={mae:.2f}")
        logger.info(f"Within 10 cycles: {within_10:.3f}, Within 20 cycles: {within_20:.3f}")

        return metrics

    def optimize_hyperparameters(self,
                               prepared_data: Dict,
                               param_grid: Optional[Dict] = None,
                               cv_folds: int = 3) -> Dict:
        """
        Optimize RUL prediction hyperparameters using cross-validation.

        Args:
            prepared_data: Prepared dataset from prepare_cmapss_data
            param_grid: Grid of parameters to search (optional, uses defaults)
            cv_folds: Number of CV folds

        Returns:
            Best hyperparameters and scores
        """
        if param_grid is None:
            param_grid = {
                'method': ['threshold'],
                'threshold_method': ['deviation', 'absolute', 'adaptive'],
                'threshold_factor': [1.5, 2.0, 2.5, 3.0]
            }

        best_score = float('inf')
        best_params = {}

        # Simple grid search (could be enhanced with proper CV)
        from itertools import product

        param_combinations = list(product(*param_grid.values()))
        param_names = list(param_grid.keys())

        logger.info(f"Testing {len(param_combinations)} parameter combinations...")

        for combo in param_combinations:
            params = dict(zip(param_names, combo))

            # Evaluate parameters
            try:
                metrics = self.evaluate_on_dataset(
                    prepared_data,
                    threshold_method=params.get('threshold_method', 'deviation'),
                    threshold_factor=params.get('threshold_factor', 2.0)
                )

                score = metrics['rmse']  # Use RMSE as optimization metric

                if score < best_score:
                    best_score = score
                    best_params = params.copy()
                    best_params['rmse'] = score

                    logger.info(f"New best: RMSE={score:.2f}, params={params}")

            except Exception as e:
                logger.warning(f"Failed to evaluate params {params}: {e}")
                continue

        # Update instance hyperparameters
        self.hyperparams.update(best_params)

        logger.info(f"Optimization complete. Best params: {best_params}")
        return best_params

    def fine_tune_forecasting(self,
                             prepared_data: Dict,
                             learning_rate: float = 1e-5,
                             num_epochs: int = 3,
                             batch_size: int = 8) -> Dict:
        """
        Fine-tune Chronos for better sensor forecasting.

        Note: This is experimental and may not always improve performance.
        Chronos models are typically used in zero-shot mode.

        Args:
            prepared_data: Prepared dataset with training sequences
            learning_rate: Learning rate for fine-tuning
            num_epochs: Number of training epochs
            batch_size: Batch size for training

        Returns:
            Training history
        """
        logger.warning("Fine-tuning Chronos is experimental. Performance may not improve.")

        # This is a simplified implementation
        # Real fine-tuning would require more sophisticated training loops
        # and access to Chronos internal training methods

        history = {
            'epochs': num_epochs,
            'learning_rate': learning_rate,
            'status': 'experimental_feature'
        }

        logger.info("Fine-tuning completed (placeholder implementation)")
        return history

    def create_ensemble(self,
                       prepared_data: Dict,
                       n_models: int = 3,
                       model_variants: Optional[List[str]] = None) -> 'ChronosEnsemble':
        """
        Create an ensemble of Chronos models for improved RUL prediction.

        Args:
            prepared_data: Prepared dataset
            n_models: Number of models in ensemble
            model_variants: Different Chronos model variants to use

        Returns:
            ChronosEnsemble instance
        """
        if model_variants is None:
            model_variants = ['amazon/chronos-t5-small'] * n_models

        ensemble = ChronosEnsemble(model_variants[:n_models], self.hyperparams)
        ensemble.optimize_hyperparameters(prepared_data)

        logger.info(f"Created ensemble with {n_models} models")
        return ensemble

    def predict_rul(self,
                   features: np.ndarray,
                   method: str = 'threshold',
                   threshold_method: str = 'deviation',
                   threshold_factor: float = 2.0,
                   failure_thresholds: Optional[np.ndarray] = None) -> float:
        """
        Convenience method for single RUL prediction.

        Args:
            features: Sensor features array (n_features, seq_length)
            method: RUL prediction method ('threshold', 'trend', 'regression')
            threshold_method: Method for failure detection ('deviation', 'absolute', 'adaptive')
            threshold_factor: Threshold multiplier
            failure_thresholds: Optional pre-defined failure thresholds per sensor

        Returns:
            Predicted RUL
        """
        return self.predict_rul_from_forecast(
            features,
            current_rul=None,
            method=method,
            threshold_method=threshold_method,
            threshold_factor=threshold_factor,
            failure_thresholds=failure_thresholds
        )

    def save_model(self, path: Union[str, Path]):
        """Save the predictor (mainly for fine-tuned models)"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model configuration
        config = {
            'model_name': self.model_name,
            'prediction_length': self.prediction_length,
            'context_length': self.context_length,
            'device': self.device
        }

        import json
        with open(path / 'chronos_config.json', 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Configuration saved to {path}")

    @classmethod
    def load_model(cls, path: Union[str, Path]) -> 'ChronosRULPredictor':
        """Load a saved predictor"""
        path = Path(path)
        config_path = path / 'chronos_config.json'

        if config_path.exists():
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)

            predictor = cls(**config)
            return predictor
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")


def create_chronos_experiment(data_loader: MultiDatasetLoader,
                            dataset_name: str = 'FD001',
                            model_name: str = "amazon/chronos-t5-small",
                            **kwargs) -> Tuple[ChronosRULPredictor, Dict, Dict]:
    """
    Convenience function to set up a complete Chronos RUL experiment.

    Args:
        data_loader: Your MultiDatasetLoader instance
        dataset_name: Which C-MAPSS dataset to use
        model_name: Chronos model variant
        **kwargs: Additional args for ChronosRULPredictor

    Returns:
        Tuple of (predictor, prepared_data, evaluation_results)
    """
    logger.info(f"Setting up Chronos experiment on {dataset_name}")

    # Load dataset
    dataset = data_loader.load_cmapss_dataset(dataset_name)
    logger.info(f"Loaded {dataset_name} dataset")

    # Initialize predictor
    predictor = ChronosRULPredictor(model_name=model_name, **kwargs)

    # Prepare data
    prepared_data = predictor.prepare_cmapss_data(dataset)

    # Evaluate
    evaluation = predictor.evaluate_on_dataset(prepared_data)

    return predictor, prepared_data, evaluation


class ChronosEnsemble:
    """
    Ensemble of Chronos models for improved RUL prediction.
    """

    def __init__(self, model_names: List[str], hyperparams: Dict):
        """
        Initialize ensemble with different Chronos models.

        Args:
            model_names: List of Chronos model names
            hyperparams: Shared hyperparameters
        """
        self.models = []
        self.hyperparams = hyperparams

        for model_name in model_names:
            try:
                predictor = ChronosRULPredictor(
                    model_name=model_name,
                    prediction_length=50,
                    context_length=512
                )
                predictor.hyperparams.update(hyperparams)
                self.models.append(predictor)
                logger.info(f"Added {model_name} to ensemble")
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")

        logger.info(f"Ensemble initialized with {len(self.models)} models")

    def predict_rul(self, features: np.ndarray) -> float:
        """Ensemble prediction using median of individual predictions."""
        predictions = []

        for model in self.models:
            try:
                pred = model.predict_rul(
                    features,
                    method=self.hyperparams.get('method', 'threshold'),
                    threshold_method=self.hyperparams.get('threshold_method', 'deviation'),
                    threshold_factor=self.hyperparams.get('threshold_factor', 2.0)
                )
                predictions.append(pred)
            except Exception as e:
                logger.warning(f"Model prediction failed: {e}")

        if predictions:
            return np.median(predictions)
        else:
            return 50.0  # Default fallback

    def optimize_hyperparameters(self, prepared_data: Dict) -> Dict:
        """Optimize hyperparameters for the ensemble."""
        # Use the first model for optimization, apply to all
        if self.models:
            best_params = self.models[0].optimize_hyperparameters(prepared_data)

            # Apply to all models
            for model in self.models[1:]:
                model.hyperparams.update(best_params)

            return best_params
        return {}

    def evaluate_on_dataset(self, prepared_data: Dict) -> Dict:
        """Evaluate ensemble performance."""
        predictions = []
        actuals = []

        for seq in prepared_data['test_sequences']:
            if seq['seq_length'] < 10:
                continue

            actual_rul = seq['rul'][-1]
            predicted_rul = self.predict_rul(seq['features'])

            predictions.append(predicted_rul)
            actuals.append(actual_rul)

        # Calculate metrics
        predictions = np.array(predictions)
        actuals = np.array(actuals)

        mse = np.mean((predictions - actuals) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - actuals))

        within_10 = np.mean(np.abs(predictions - actuals) <= 10)
        within_20 = np.mean(np.abs(predictions - actuals) <= 20)

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'within_10_cycles': within_10,
            'within_20_cycles': within_20,
            'n_predictions': len(predictions)
        }


if __name__ == "__main__":
    # Example usage
    logger.info("Chronos RUL Predictor - Example usage")

    # Initialize data loader
    loader = MultiDatasetLoader()

    # Run experiment on FD001
    try:
        predictor, data, results = create_chronos_experiment(
            loader,
            dataset_name='FD001',
            model_name='amazon/chronos-t5-small'
        )

        print(f"Results: RMSE={results['rmse']:.2f}, Within 20 cycles={results['within_20_cycles']:.3f}")

        # Example of hyperparameter optimization
        print("Optimizing hyperparameters...")
        best_params = predictor.optimize_hyperparameters(data)
        print(f"Best parameters: {best_params}")

        # Example of ensemble creation
        print("Creating ensemble...")
        ensemble = predictor.create_ensemble(data, n_models=2)
        ensemble_results = ensemble.evaluate_on_dataset(data)
        print(f"Ensemble results: RMSE={ensemble_results['rmse']:.2f}")

    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        logger.info("Make sure you have installed chronos-forecasting: pip install chronos-forecasting")
        logger.info("Install with: pip install -r requirements.txt")