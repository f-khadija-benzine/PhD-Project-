"""
Multi-Level Explainability Module (Section IV.D.1)

Three levels of interpretability:
    Level 1: Global Feature Importance (Tree-based, from XGBoost)
    Level 2: Temporal Feature Dynamics (Attention weights, from BiLSTM)
    Level 3: Instance-Level Explanations (SHAP values)

Author: Fatima Khadija Benzine
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from pathlib import Path


# ==============================================================================
# Level 1: Global Feature Importance (Tree-Based)
# ==============================================================================

def get_tree_importance(ml_model, feature_names: List[str], top_k: int = 15) -> pd.DataFrame:
    """
    Extract global feature importance from XGBoost.
    
    Args:
        ml_model: MLBranch instance (fitted)
        feature_names: list of original feature names
        top_k: number of top features to return
    
    Returns:
        DataFrame with columns [feature, importance, category]
    """
    # Get raw importance from the model
    raw_importance = ml_model.model.feature_importances_
    
    # Map flattened feature names back to original features
    # MLBranch with statistics creates: feat_mean, feat_std, feat_min, feat_max, feat_last, feat_trend
    # MLBranch with flatten creates: feat_t0, feat_t1, ..., feat_tW
    strategy = ml_model.flatten_strategy
    
    if strategy == 'statistics':
        stats = ['mean', 'std', 'min', 'max', 'last', 'trend']
        flat_names = []
        for fname in feature_names:
            for s in stats:
                flat_names.append(f"{fname}_{s}")
    else:  # flatten
        W = len(raw_importance) // len(feature_names)
        flat_names = []
        for fname in feature_names:
            for t in range(W):
                flat_names.append(f"{fname}_t{t}")
    
    # Aggregate importance per original feature
    feat_importance = {}
    for i, flat_name in enumerate(flat_names):
        if i >= len(raw_importance):
            break
        original = flat_name.rsplit('_', 1)[0]
        # For statistics: feat_mean -> feat
        for s in (['mean', 'std', 'min', 'max', 'last', 'trend'] if strategy == 'statistics' 
                  else [f't{t}' for t in range(100)]):
            if flat_name.endswith(f'_{s}'):
                original = flat_name[:-(len(s)+1)]
                break
        
        if original not in feat_importance:
            feat_importance[original] = 0.0
        feat_importance[original] += raw_importance[i]
    
    # Build DataFrame
    df = pd.DataFrame([
        {'feature': k, 'importance': v} for k, v in feat_importance.items()
    ]).sort_values('importance', ascending=False).head(top_k)
    
    # Categorize features
    df['category'] = df['feature'].apply(_categorize_feature)
    
    # Normalize
    df['importance'] = df['importance'] / df['importance'].sum()
    
    return df.reset_index(drop=True)


def plot_tree_importance(df: pd.DataFrame, save_path: str = None,
                         title: str = 'Global Feature Importance (XGBoost)'):
    """Plot horizontal bar chart of feature importance."""
    colors = {'sensor': '#2196F3', 'setting': '#4CAF50', 'bi': '#FF9800'}
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(df)), df['importance'].values,
                    color=[colors.get(c, '#9E9E9E') for c in df['category']])
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['feature'].values)
    ax.set_xlabel('Normalized Importance')
    ax.set_title(title, fontweight='bold')
    ax.invert_yaxis()
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[k], label=k.capitalize()) 
                       for k in colors if k in df['category'].values]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig


# ==============================================================================
# Level 2: Temporal Feature Dynamics (Attention Weights)
# ==============================================================================

def analyze_attention_temporal(attn_model, X_test: np.ndarray,
                                feature_names: List[str],
                                unit_indices: List[int] = None) -> Dict:
    """
    Extract and analyze attention weights for temporal dynamics.
    
    Args:
        attn_model: attention extraction model (from build_dual_attention_bilstm)
        X_test: test data (n_samples, window_size, n_features)
        feature_names: feature names
        unit_indices: indices of specific units to analyze (default: all)
    
    Returns:
        dict with feature_weights (n_samples, n_features) and
        temporal_weights (n_samples, window_size)
    """
    # Get attention weights
    outputs = attn_model.predict(X_test, batch_size=256, verbose=0)
    
    # outputs depends on model structure
    # Typically: [prediction, feature_weights, temporal_weights]
    # Get attention weights
    outputs = attn_model.predict(X_test, batch_size=256, verbose=0)
    
    if isinstance(outputs, dict):
        feature_weights = outputs['feature_weights']
        temporal_weights = outputs['temporal_weights']
    elif isinstance(outputs, list) and len(outputs) >= 3:
        feature_weights = outputs[1]
        temporal_weights = outputs[2]
    else:
        raise ValueError(f"Unexpected attention model output: {type(outputs)}")
    result = {
        'feature_weights': feature_weights,
        'temporal_weights': temporal_weights,
        'feature_names': feature_names,
        'mean_feature_weights': np.mean(feature_weights, axis=0),
        'mean_temporal_weights': np.mean(temporal_weights, axis=0),
    }
    
    return result


def plot_attention_heatmap(attn_result: Dict, unit_idx: int = 0,
                            save_path: str = None):
    """Plot attention weights heatmap for a specific unit."""
    feat_w = attn_result['feature_weights']
    feat_names = attn_result['feature_names']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Feature attention (average across all samples)
    mean_fw = attn_result['mean_feature_weights']
    sorted_idx = np.argsort(mean_fw)[::-1][:15]
    axes[0].barh(range(len(sorted_idx)), mean_fw[sorted_idx],
                  color=[('#FF9800' if 'sensor' not in feat_names[i] and 
                          'setting' not in feat_names[i] else '#2196F3')
                         for i in sorted_idx])
    axes[0].set_yticks(range(len(sorted_idx)))
    axes[0].set_yticklabels([feat_names[i] for i in sorted_idx])
    axes[0].set_xlabel('Average Attention Weight')
    axes[0].set_title('Feature Attention (mean across samples)')
    axes[0].invert_yaxis()
    
    # Temporal attention (average across all samples)
    mean_tw = attn_result['mean_temporal_weights']
    axes[1].plot(range(len(mean_tw)), mean_tw, 'b-', linewidth=2)
    axes[1].fill_between(range(len(mean_tw)), mean_tw, alpha=0.3)
    axes[1].set_xlabel('Timestep in Window')
    axes[1].set_ylabel('Average Attention Weight')
    axes[1].set_title('Temporal Attention (mean across samples)')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig


# ==============================================================================
# Level 3: SHAP Explanations
# ==============================================================================

def compute_shap_values(ml_model, X_test: np.ndarray,
                         feature_names: List[str],
                         n_background: int = 100) -> Dict:
    """
    Compute SHAP values for the ML branch (XGBoost).
    
    Args:
        ml_model: MLBranch instance (fitted)
        X_test: test data (n_samples, window_size, n_features)
        feature_names: original feature names
        n_background: number of background samples for SHAP
    
    Returns:
        dict with shap_values, expected_value, feature_names_flat
    """
    import shap
    
    # Flatten test data
    X_flat = ml_model._flatten(X_test, feature_names)
    
    # Get flattened feature names
    strategy = ml_model.flatten_strategy
    if strategy == 'statistics':
        stats = ['mean', 'std', 'min', 'max', 'last', 'trend']
        flat_names = [f"{f}_{s}" for f in feature_names for s in stats]
    else:
        W = X_test.shape[1]
        flat_names = [f"{f}_t{t}" for f in feature_names for t in range(W)]
    
    # Compute SHAP values
    explainer = shap.TreeExplainer(ml_model.model)
    shap_values = explainer.shap_values(X_flat)
    
    return {
        'shap_values': shap_values,
        'expected_value': explainer.expected_value,
        'X_flat': X_flat,
        'feature_names_flat': flat_names,
        'feature_names_original': feature_names,
    }


def aggregate_shap_to_original(shap_result: Dict) -> pd.DataFrame:
    """
    Aggregate SHAP values from flattened features back to original features.
    
    Returns:
        DataFrame with columns [feature, mean_abs_shap, category]
    """
    shap_vals = shap_result['shap_values']
    flat_names = shap_result['feature_names_flat']
    original_names = shap_result['feature_names_original']
    
    # Mean absolute SHAP per flattened feature
    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    
    # Aggregate to original features
    agg = {}
    for i, flat_name in enumerate(flat_names):
        for orig in original_names:
            if flat_name.startswith(orig + '_'):
                if orig not in agg:
                    agg[orig] = 0.0
                agg[orig] += mean_abs[i]
                break
    
    df = pd.DataFrame([
        {'feature': k, 'mean_abs_shap': v, 'category': _categorize_feature(k)}
        for k, v in agg.items()
    ]).sort_values('mean_abs_shap', ascending=False)
    
    return df.reset_index(drop=True)


def plot_shap_summary(shap_result: Dict, max_display: int = 15,
                       save_path: str = None):
    """Plot SHAP summary (beeswarm) plot."""
    import shap
    
    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_result['shap_values'],
        shap_result['X_flat'],
        feature_names=shap_result['feature_names_flat'],
        max_display=max_display,
        show=False,
    )
    plt.title('SHAP Summary — Feature Impact on RUL Prediction', fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


def plot_shap_waterfall(shap_result: Dict, sample_idx: int = 0,
                         save_path: str = None):
    """Plot SHAP waterfall for a single prediction."""
    import shap
    
    fig = plt.figure(figsize=(10, 8))
    explanation = shap.Explanation(
        values=shap_result['shap_values'][sample_idx],
        base_values=shap_result['expected_value'],
        data=shap_result['X_flat'][sample_idx],
        feature_names=shap_result['feature_names_flat'],
    )
    shap.waterfall_plot(explanation, max_display=15, show=False)
    plt.title(f'SHAP Waterfall — Sample {sample_idx}', fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


def plot_shap_unit_evolution(shap_result: Dict, X_test: np.ndarray,
                              test_df: pd.DataFrame, unit_id: int,
                              feature_names: List[str],
                              top_k: int = 5, window_size: int = 30,
                              pad: bool = False, save_path: str = None):
    """
    Plot SHAP value evolution over time for a specific unit.
    Shows how each feature's contribution changes as the unit degrades.
    """
    # Find sample indices for this unit
    unit_sizes = []
    for u in sorted(test_df['unit'].unique()):
        T = len(test_df[test_df['unit'] == u])
        n_win = T if pad else max(T - (window_size - 1), 0)
        unit_sizes.append((u, n_win))
    
    start_idx = 0
    n_samples = 0
    for u, n in unit_sizes:
        if u == unit_id:
            n_samples = n
            break
        start_idx += n
    
    if n_samples == 0:
        print(f"Unit {unit_id} not found")
        return None
    
    unit_shap = shap_result['shap_values'][start_idx:start_idx + n_samples]
    
    # Aggregate to original features
    flat_names = shap_result['feature_names_flat']
    original_names = feature_names
    
    agg_shap = np.zeros((n_samples, len(original_names)))
    for i, flat_name in enumerate(flat_names):
        for j, orig in enumerate(original_names):
            if flat_name.startswith(orig + '_'):
                agg_shap[:, j] += unit_shap[:, i]
                break
    
    # Top-k features by mean absolute contribution for this unit
    mean_abs = np.mean(np.abs(agg_shap), axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:top_k]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    for idx in top_idx:
        color = '#FF9800' if _categorize_feature(original_names[idx]) == 'bi' else '#2196F3'
        ax.plot(range(n_samples), agg_shap[:, idx], label=original_names[idx],
                linewidth=2, alpha=0.8, color=color if top_k <= 5 else None)
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time Step (cycles)')
    ax.set_ylabel('SHAP Value (contribution to RUL)')
    ax.set_title(f'SHAP Evolution — Unit {unit_id}', fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig


# ==============================================================================
# Synthesis: Cross-Level Comparison
# ==============================================================================

def compare_importance_levels(tree_importance: pd.DataFrame,
                               shap_agg: pd.DataFrame,
                               attn_result: Dict = None,
                               save_path: str = None) -> pd.DataFrame:
    """
    Compare feature importance across all three levels.
    
    Returns:
        DataFrame with columns [feature, tree_importance, shap_importance, 
        attention_importance, category]
    """
    # Merge tree and SHAP
    comparison = tree_importance[['feature', 'importance', 'category']].rename(
        columns={'importance': 'tree_importance'})
    
    shap_norm = shap_agg[['feature', 'mean_abs_shap']].copy()
    shap_norm['shap_importance'] = shap_norm['mean_abs_shap'] / shap_norm['mean_abs_shap'].sum()
    
    comparison = comparison.merge(
        shap_norm[['feature', 'shap_importance']], on='feature', how='outer')
    
    # Add attention weights if available
    if attn_result is not None:
        feat_names = attn_result['feature_names']
        mean_fw = attn_result['mean_feature_weights']
        attn_norm = mean_fw / mean_fw.sum()
        attn_df = pd.DataFrame({
            'feature': feat_names,
            'attention_importance': attn_norm
        })
        comparison = comparison.merge(attn_df, on='feature', how='outer')
    
    comparison = comparison.fillna(0).sort_values('tree_importance', ascending=False)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(min(len(comparison), 15))
    comp_top = comparison.head(15)
    
    width = 0.25
    ax.bar([i - width for i in x], comp_top['tree_importance'], width,
           label='Tree Importance', color='#2196F3', alpha=0.8)
    ax.bar(x, comp_top['shap_importance'], width,
           label='SHAP Importance', color='#4CAF50', alpha=0.8)
    if 'attention_importance' in comp_top.columns:
        ax.bar([i + width for i in x], comp_top['attention_importance'], width,
               label='Attention Weight', color='#FF9800', alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(comp_top['feature'], rotation=45, ha='right')
    ax.set_ylabel('Normalized Importance')
    ax.set_title('Cross-Level Feature Importance Comparison', fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return comparison


# ==============================================================================
# Helpers
# ==============================================================================

def _categorize_feature(name: str) -> str:
    """Categorize a feature as sensor, setting, or bi."""
    if name.startswith('sensor_'):
        return 'sensor'
    elif name.startswith('setting_'):
        return 'setting'
    else:
        return 'bi'
