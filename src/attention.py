"""
Dual Attention Mechanism for BI-Aware RUL Prediction
Implements Section III-C2 of the methodology.

Two attention layers:
    1. Feature Attention (Eq. 16-17): learns adaptive weights per feature
       at each time step — key for BI vs sensor adaptive weighting
    2. Temporal Attention (Eq. 18): learns which time steps in the window
       are most informative for RUL prediction

Architecture:
    Input (batch, W, n_features)
        → Feature Attention → weighted features
        → BiLSTM → hidden states (batch, W, 2*hidden)
        → Temporal Attention → context vector (batch, 2*hidden)
        → Dense → RUL

Author: Fatima Khadija Benzine
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, backend as K
from typing import Optional, Tuple


class FeatureAttention(layers.Layer):
    """
    Feature-level attention (Eq. 16-17).

    At each time step t, computes importance weights α_j for each feature j.
    Allows the model to dynamically focus on different features (sensor vs BI)
    depending on the degradation context.

    Input:  (batch, W, n_features)
    Output: (batch, W, n_features)  — element-wise weighted
    Weights: (batch, W, n_features) — accessible for interpretability
    """

    def __init__(self, hidden_dim: int = 32, **kwargs):
        """
        Args:
            hidden_dim: intermediate dimension for attention scoring
        """
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim

    def build(self, input_shape):
        n_features = input_shape[-1]
        # Attention scoring network: x_t → score_j for each feature
        self.W_feat = self.add_weight(
            name='W_feat',
            shape=(n_features, self.hidden_dim),
            initializer='glorot_uniform',
            trainable=True,
        )
        self.b_feat = self.add_weight(
            name='b_feat',
            shape=(self.hidden_dim,),
            initializer='zeros',
            trainable=True,
        )
        self.v_feat = self.add_weight(
            name='v_feat',
            shape=(self.hidden_dim, n_features),
            initializer='glorot_uniform',
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs, return_weights=False):
        """
        Args:
            inputs: (batch, W, n_features)
            return_weights: if True, return (output, weights)

        Returns:
            output: (batch, W, n_features) — weighted features
            weights: (batch, W, n_features) — attention weights (if requested)
        """
        # Score computation: tanh(X @ W + b) @ v → (batch, W, n_features)
        score = tf.tanh(tf.matmul(inputs, self.W_feat) + self.b_feat)
        score = tf.matmul(score, self.v_feat)  # (batch, W, n_features)

        # Softmax over features dimension
        alpha = tf.nn.softmax(score, axis=-1)  # (batch, W, n_features)

        # Element-wise weighting
        output = inputs * alpha  # (batch, W, n_features)

        if return_weights:
            return output, alpha
        return output

    def get_config(self):
        config = super().get_config()
        config.update({'hidden_dim': self.hidden_dim})
        return config


class TemporalAttention(layers.Layer):
    """
    Temporal attention over LSTM hidden states (Eq. 18).

    Computes importance weights β_t for each time step in the window,
    then produces a weighted context vector summarizing the sequence.

    Input:  (batch, W, hidden_dim)  — LSTM hidden states
    Output: (batch, hidden_dim)     — context vector
    Weights: (batch, W)             — accessible for interpretability
    """

    def __init__(self, attention_dim: int = 64, **kwargs):
        """
        Args:
            attention_dim: intermediate dimension for attention scoring
        """
        super().__init__(**kwargs)
        self.attention_dim = attention_dim

    def build(self, input_shape):
        hidden_dim = input_shape[-1]
        self.W_temp = self.add_weight(
            name='W_temp',
            shape=(hidden_dim, self.attention_dim),
            initializer='glorot_uniform',
            trainable=True,
        )
        self.b_temp = self.add_weight(
            name='b_temp',
            shape=(self.attention_dim,),
            initializer='zeros',
            trainable=True,
        )
        self.v_temp = self.add_weight(
            name='v_temp',
            shape=(self.attention_dim, 1),
            initializer='glorot_uniform',
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs, return_weights=False):
        """
        Args:
            inputs: (batch, W, hidden_dim) — LSTM outputs
            return_weights: if True, return (context, weights)

        Returns:
            context: (batch, hidden_dim) — weighted sum of hidden states
            weights: (batch, W) — temporal attention weights (if requested)
        """
        # Score: tanh(H @ W + b) @ v → (batch, W, 1)
        score = tf.tanh(tf.matmul(inputs, self.W_temp) + self.b_temp)
        score = tf.matmul(score, self.v_temp)  # (batch, W, 1)
        score = tf.squeeze(score, axis=-1)     # (batch, W)

        # Softmax over time dimension
        beta = tf.nn.softmax(score, axis=-1)   # (batch, W)

        # Weighted sum: context = Σ β_t * h_t
        beta_expanded = tf.expand_dims(beta, axis=-1)  # (batch, W, 1)
        context = tf.reduce_sum(inputs * beta_expanded, axis=1)  # (batch, hidden_dim)

        if return_weights:
            return context, beta
        return context

    def get_config(self):
        config = super().get_config()
        config.update({'attention_dim': self.attention_dim})
        return config


def build_dual_attention_bilstm(
    window_size: int = 30,
    n_features: int = 32,
    lstm_units: int = 64,
    feature_attention_dim: int = 32,
    temporal_attention_dim: int = 64,
    dropout_rate: float = 0.3,
    dense_units: int = 32,
    learning_rate: float = 0.001,
) -> Tuple[Model, Model]:
    """
    Build Dual-Attention BiLSTM model for RUL prediction.

    Architecture:
        Input → Feature Attention → BiLSTM → Temporal Attention → Dense → RUL

    Args:
        window_size: W — sliding window length
        n_features: number of input features (sensor + BI after selection)
        lstm_units: hidden units per LSTM direction (total = 2 * lstm_units)
        feature_attention_dim: hidden dim for feature attention scoring
        temporal_attention_dim: hidden dim for temporal attention scoring
        dropout_rate: dropout after BiLSTM and dense layers
        dense_units: units in the final dense layer before output
        learning_rate: Adam optimizer learning rate

    Returns:
        model: compiled Keras model for training
        attention_model: model that also outputs attention weights
    """
    # --- Input ---
    inputs = layers.Input(shape=(window_size, n_features), name='input')

    # --- Feature Attention (Eq. 16-17) ---
    feat_attn = FeatureAttention(
        hidden_dim=feature_attention_dim,
        name='feature_attention',
    )
    x_weighted, alpha = feat_attn(inputs, return_weights=True)

    # --- BiLSTM ---
    x = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True),
        name='bilstm',
    )(x_weighted)
    x = layers.Dropout(dropout_rate, name='dropout_lstm')(x)

    # --- Temporal Attention (Eq. 18) ---
    temp_attn = TemporalAttention(
        attention_dim=temporal_attention_dim,
        name='temporal_attention',
    )
    context, beta = temp_attn(x, return_weights=True)

    # --- Output head ---
    x = layers.Dense(dense_units, activation='relu', name='dense_1')(context)
    x = layers.Dropout(dropout_rate, name='dropout_dense')(x)
    output = layers.Dense(1, activation='linear', name='rul_output')(x)

    # --- Compile training model ---
    model = Model(inputs=inputs, outputs=output, name='DualAttention_BiLSTM')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae'],
    )

    # --- Attention model (for interpretability) ---
    attention_model = Model(
        inputs=inputs,
        outputs={
            'rul': output,
            'feature_weights': alpha,    # (batch, W, n_features)
            'temporal_weights': beta,     # (batch, W)
        },
        name='DualAttention_BiLSTM_Interpretable',
    )

    return model, attention_model


# ==============================================================================
# Attention Weight Extraction & Storage (for Recommendation System)
# ==============================================================================

def extract_attention_weights(
    attention_model: Model,
    X: 'np.ndarray',
    feature_names: list,
    unit_ids: 'np.ndarray' = None,
    batch_size: int = 256,
) -> dict:
    """
    Extract and structure attention weights for downstream use
    (recommendation system, explainability, SHAP comparison).

    Args:
        attention_model: the interpretable model from build_dual_attention_bilstm
        X: input data (n_samples, W, n_features)
        feature_names: list of feature names matching n_features
        unit_ids: optional unit identifiers per sample
        batch_size: prediction batch size

    Returns:
        dict with:
            'rul_pred': (n_samples,) predicted RUL
            'feature_weights': (n_samples, W, n_features) raw weights
            'temporal_weights': (n_samples, W) raw weights
            'feature_importance': DataFrame — mean weight per feature across all samples
            'feature_names': list of feature names
            'per_sample_importance': (n_samples, n_features) mean over W per sample
    """
    import numpy as np
    import pandas as pd

    results = attention_model.predict(X, batch_size=batch_size)

    rul_pred = results['rul'].flatten()
    feat_w = results['feature_weights']   # (n_samples, W, n_features)
    temp_w = results['temporal_weights']   # (n_samples, W)

    # Per-sample feature importance: average over the window dimension
    per_sample_imp = np.mean(feat_w, axis=1)  # (n_samples, n_features)

    # Global feature importance: average over all samples and time steps
    global_imp = np.mean(feat_w, axis=(0, 1))  # (n_features,)

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'mean_attention_weight': global_imp,
    }).sort_values('mean_attention_weight', ascending=False).reset_index(drop=True)

    # Tag feature type
    def feature_type(name):
        if name.startswith('sensor_'):
            return 'sensor'
        elif name.startswith('setting_'):
            return 'setting'
        else:
            return 'BI'
    importance_df['type'] = importance_df['feature'].apply(feature_type)

    output = {
        'rul_pred': rul_pred,
        'feature_weights': feat_w,
        'temporal_weights': temp_w,
        'feature_importance': importance_df,
        'feature_names': feature_names,
        'per_sample_importance': per_sample_imp,
    }

    if unit_ids is not None:
        output['unit_ids'] = unit_ids

    return output


def save_attention_weights(
    weights_dict: dict,
    save_dir: str,
    dataset_name: str = 'M1',
    prefix: str = 'attn',
):
    """
    Save attention weights to disk for the recommendation system.

    Saves:
        - {prefix}_{dataset}_feature_importance.csv  (global ranking)
        - {prefix}_{dataset}_per_sample.npy          (per-sample weights)
        - {prefix}_{dataset}_temporal.npy             (temporal weights)
        - {prefix}_{dataset}_predictions.csv          (RUL predictions)

    Args:
        weights_dict: output of extract_attention_weights
        save_dir: directory to save files
        dataset_name: identifier (M1, M2, etc.)
        prefix: file prefix
    """
    import numpy as np
    import pandas as pd
    from pathlib import Path

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    tag = f"{prefix}_{dataset_name}"

    # Global feature importance
    weights_dict['feature_importance'].to_csv(
        save_path / f"{tag}_feature_importance.csv", index=False
    )

    # Per-sample feature importance (n_samples, n_features)
    np.save(
        save_path / f"{tag}_per_sample.npy",
        weights_dict['per_sample_importance']
    )

    # Temporal weights (n_samples, W)
    np.save(
        save_path / f"{tag}_temporal.npy",
        weights_dict['temporal_weights']
    )

    # Predictions
    pred_df = pd.DataFrame({'rul_pred': weights_dict['rul_pred']})
    if 'unit_ids' in weights_dict:
        pred_df['unit'] = weights_dict['unit_ids']
    pred_df.to_csv(save_path / f"{tag}_predictions.csv", index=False)

    print(f"Saved attention weights to {save_path}/")
    print(f"  {tag}_feature_importance.csv  — global feature ranking")
    print(f"  {tag}_per_sample.npy          — ({weights_dict['per_sample_importance'].shape})")
    print(f"  {tag}_temporal.npy            — ({weights_dict['temporal_weights'].shape})")
    print(f"  {tag}_predictions.csv         — {len(weights_dict['rul_pred'])} predictions")
