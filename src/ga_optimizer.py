"""
Genetic Algorithm Hyperparameter Optimization (Section III-B4)

Two-stage strategy for CPU-friendly optimization:
    Stage 1: GA on ML branch (XGBoost) — fast (~1h)
    Stage 2: GA on DL branch (BiLSTM + Attention) — slower (~6-12h)

Fitness: RMSE on validation set (per-unit, last window)

Author: Fatima Khadija Benzine
"""

import numpy as np
import pandas as pd
import random
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error


# ==============================================================================
# Search Spaces
# ==============================================================================

ML_SEARCH_SPACE = {
    'n_estimators':       [100, 200, 300, 500],
    'max_depth':          [3, 4, 5, 6, 8],
    'learning_rate_xgb':  [0.01, 0.05, 0.1],
    'subsample':          [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree':   [0.6, 0.7, 0.8, 0.9, 1.0],
    'reg_alpha':          [0.0, 0.01, 0.1, 1.0],
    'reg_lambda':         [0.1, 0.5, 1.0, 2.0],
    'flatten_strategy':   ['statistics', 'flatten'],
    'feature_selection':  ['correlation', 'aficv', 'sensor_only'],
}

DL_SEARCH_SPACE = {
    'lstm_units':              [32, 64, 128],
    'feature_attention_dim':   [16, 32, 64],
    'temporal_attention_dim':  [32, 64, 128],
    'dropout_rate':            [0.1, 0.2, 0.3, 0.4, 0.5],
    'dense_units':             [16, 32, 64],
    'learning_rate':           [0.0001, 0.0005, 0.001, 0.005],
    'batch_size':              [64, 128, 256],
    'feature_selection':       ['correlation', 'aficv', 'sensor_only'],
}


# ==============================================================================
# Core GA Engine
# ==============================================================================

class GeneticAlgorithm:
    """
    Generic GA engine for hyperparameter optimization.

    Args:
        search_space: dict of {param_name: [possible_values]}
        fitness_fn: callable(params_dict) → float (lower is better)
        pop_size: population size
        n_generations: number of generations
        crossover_rate: probability of crossover
        mutation_rate: probability of mutation per gene
        random_state: random seed
    """

    def __init__(self,
                 search_space: Dict[str, list],
                 fitness_fn,
                 pop_size: int = 20,
                 n_generations: int = 30,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.15,
                 stagnation_limit: int = 5,
                 random_state: int = 42):

        self.search_space = search_space
        self.fitness_fn = fitness_fn
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.base_mutation_rate = mutation_rate
        self.stagnation_limit = stagnation_limit
        self.param_names = list(search_space.keys())

        random.seed(random_state)
        np.random.seed(random_state)

        # Tracking
        self.history = []
        self.best_individual = None
        self.best_fitness = float('inf')
        self._stagnation_counter = 0

    def _random_individual(self) -> Dict:
        return {k: random.choice(v) for k, v in self.search_space.items()}

    def _crossover(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        # Uniform crossover
        child1, child2 = {}, {}
        for key in self.param_names:
            if random.random() < 0.5:
                child1[key] = parent1[key]
                child2[key] = parent2[key]
            else:
                child1[key] = parent2[key]
                child2[key] = parent1[key]
        return child1, child2

    def _mutate(self, individual: Dict) -> Dict:
        mutated = individual.copy()
        for key in self.param_names:
            if random.random() < self.mutation_rate:
                mutated[key] = random.choice(self.search_space[key])
        return mutated

    def _tournament_select(self, population: List[Dict],
                           fitness_scores: List[float],
                           k: int = 3) -> Dict:
        indices = random.sample(range(len(population)), k)
        best_idx = min(indices, key=lambda i: fitness_scores[i])
        return population[best_idx].copy()

    def _population_diversity(self, population: List[Dict]) -> float:
        """Measure diversity as average pairwise gene difference ratio."""
        if len(population) < 2:
            return 1.0
        diffs = 0
        comparisons = 0
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                for key in self.param_names:
                    if population[i][key] != population[j][key]:
                        diffs += 1
                comparisons += len(self.param_names)
        return diffs / comparisons if comparisons > 0 else 0.0

    def run(self, verbose: bool = True) -> Dict:
        """
        Run the GA optimization.

        Diversity mechanisms:
            1. Adaptive mutation: rate doubles after stagnation_limit gens without improvement
            2. Diversity injection: replace 30% of population with random individuals on stagnation
            3. Population diversity tracking

        Returns:
            best_params: dict of best hyperparameters
        """
        start_time = time.time()

        # Initialize population
        population = [self._random_individual() for _ in range(self.pop_size)]

        for gen in range(self.n_generations):
            gen_start = time.time()

            # Evaluate fitness
            fitness_scores = []
            for i, individual in enumerate(population):
                try:
                    fitness = self.fitness_fn(individual)
                except Exception as e:
                    if verbose:
                        print(f"  [Gen {gen+1}] Individual {i+1} failed: {e}")
                    fitness = float('inf')
                fitness_scores.append(fitness)

            # Track best
            gen_best_idx = np.argmin(fitness_scores)
            gen_best_fitness = fitness_scores[gen_best_idx]
            gen_mean_fitness = np.mean([f for f in fitness_scores if f < float('inf')])

            # Stagnation detection
            improved = False
            if gen_best_fitness < self.best_fitness:
                self.best_fitness = gen_best_fitness
                self.best_individual = population[gen_best_idx].copy()
                self._stagnation_counter = 0
                self.mutation_rate = self.base_mutation_rate
                improved = True
            else:
                self._stagnation_counter += 1

            # Adaptive mutation: increase rate on stagnation
            if self._stagnation_counter >= self.stagnation_limit:
                self.mutation_rate = min(self.base_mutation_rate * 2, 0.5)

            diversity = self._population_diversity(population)

            self.history.append({
                'generation': gen + 1,
                'best_fitness': gen_best_fitness,
                'global_best': self.best_fitness,
                'mean_fitness': gen_mean_fitness,
                'best_params': population[gen_best_idx].copy(),
                'time': time.time() - gen_start,
                'diversity': diversity,
                'mutation_rate': self.mutation_rate,
                'stagnation': self._stagnation_counter,
            })

            if verbose:
                elapsed = time.time() - start_time
                stag_flag = " ⚠ STAGNATION" if self._stagnation_counter >= self.stagnation_limit else ""
                print(f"  Gen {gen+1:3d}/{self.n_generations} | "
                      f"Best: {gen_best_fitness:.2f} | "
                      f"Global: {self.best_fitness:.2f} | "
                      f"Mean: {gen_mean_fitness:.2f} | "
                      f"Div: {diversity:.2f} | "
                      f"Mut: {self.mutation_rate:.2f} | "
                      f"Time: {elapsed:.0f}s{stag_flag}")

            # Diversity injection on stagnation
            if self._stagnation_counter >= self.stagnation_limit:
                n_inject = max(int(self.pop_size * 0.3), 2)
                if verbose:
                    print(f"    → Injecting {n_inject} random individuals")
                self._stagnation_counter = 0  # Reset counter after injection

            # Elitism: keep best individual
            new_population = [self.best_individual.copy()]

            # Diversity injection: add random individuals if stagnating
            if self._stagnation_counter == 0 and not improved and gen > 0:
                n_inject = max(int(self.pop_size * 0.3), 2)
                for _ in range(n_inject):
                    new_population.append(self._random_individual())

            # Generate rest via selection + crossover + mutation
            while len(new_population) < self.pop_size:
                parent1 = self._tournament_select(population, fitness_scores)
                parent2 = self._tournament_select(population, fitness_scores)
                child1, child2 = self._crossover(parent1, parent2)
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                new_population.extend([child1, child2])

            population = new_population[:self.pop_size]

        total_time = time.time() - start_time
        if verbose:
            print(f"\n{'='*60}")
            print(f"GA Complete — {total_time:.0f}s ({total_time/60:.1f} min)")
            print(f"Best RMSE: {self.best_fitness:.4f}")
            print(f"Best params: {self.best_individual}")
            print(f"{'='*60}")

        return self.best_individual

    def get_history_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)

    def save_results(self, save_dir: str, tag: str = 'ga'):
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)

        # Best params
        with open(path / f"{tag}_best_params.json", 'w') as f:
            json.dump(self.best_individual, f, indent=2)

        # History
        self.get_history_df().to_csv(path / f"{tag}_history.csv", index=False)

        print(f"Saved: {path}/{tag}_best_params.json, {tag}_history.csv")


# ==============================================================================
# Stage 1: ML Branch GA
# ==============================================================================

def run_ml_ga(
    X_train_dict: Dict[str, np.ndarray],
    y_train_dict: Dict[str, np.ndarray],
    train_df_dict: Dict[str, 'pd.DataFrame'],
    feature_names_dict: Dict[str, List[str]],
    window_size: int = 30,
    pad: bool = False,
    pop_size: int = 20,
    n_generations: int = 30,
    val_ratio: float = 0.2,
    random_state: int = 42,
    save_dir: str = None,
) -> Dict:
    """
    Run GA to optimize XGBoost hyperparameters including feature selection.

    Args:
        X_train_dict: {'correlation': X, 'aficv': X, 'sensor_only': X}
        y_train_dict: {'correlation': y, 'aficv': y, 'sensor_only': y}
        train_df_dict: {'correlation': df, 'aficv': df, 'sensor_only': df}
        feature_names_dict: {'correlation': [...], 'aficv': [...], 'sensor_only': [...]}
        Other params: same as before

    Fitness = RMSE on validation set (per-unit, last window).
    """
    from ml_branch import MLBranch

    # --- Precompute train/val splits for each feature selection ---
    splits = {}
    for fs_key in X_train_dict:
        X = X_train_dict[fs_key]
        y = y_train_dict[fs_key]
        df = train_df_dict[fs_key]

        windows_per_unit = []
        unit_ids_ordered = []
        for u in sorted(df['unit'].unique()):
            T = len(df[df['unit'] == u])
            n_win = T if pad else max(T - (window_size - 1), 0)
            windows_per_unit.append(n_win)
            unit_ids_ordered.append(u)

        sample_units = np.concatenate([
            np.full(n, u) for u, n in zip(unit_ids_ordered, windows_per_unit)
        ])

        gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=random_state)
        train_idx, val_idx = next(gss.split(X, y, groups=sample_units))

        splits[fs_key] = {
            'X_tr': X[train_idx], 'X_val': X[val_idx],
            'y_tr': y[train_idx], 'y_val': y[val_idx],
            'val_units': sample_units[val_idx],
            'n_train_units': len(set(sample_units[train_idx])),
            'n_val_units': len(set(sample_units[val_idx])),
        }

    for fs_key, s in splits.items():
        print(f"[ML GA] {fs_key}: Train {len(s['X_tr'])} ({s['n_train_units']} units), "
              f"Val {len(s['X_val'])} ({s['n_val_units']} units)")

    def _eval_per_unit_rmse(y_true, y_pred, unit_labels):
        preds_last = []
        true_last = []
        for u in sorted(set(unit_labels)):
            mask = unit_labels == u
            if mask.sum() > 0:
                preds_last.append(y_pred[mask][-1])
                true_last.append(y_true[mask][-1])
        return np.sqrt(mean_squared_error(true_last, preds_last))

    # --- Fitness function ---
    def fitness_fn(params):
        fs = params['feature_selection']
        s = splits[fs]

        ml = MLBranch(
            model_type='xgboost',
            flatten_strategy=params['flatten_strategy'],
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate_xgb'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            reg_alpha=params['reg_alpha'],
            reg_lambda=params['reg_lambda'],
            random_state=random_state,
        )
        ml.fit(s['X_tr'], s['y_tr'], feature_names=feature_names_dict[fs], verbose=False)
        y_pred_val = ml.predict(s['X_val'])
        return _eval_per_unit_rmse(s['y_val'], y_pred_val, s['val_units'])

    # --- Run GA ---
    print(f"\n{'='*60}")
    print(f"ML GA — {pop_size} pop × {n_generations} gen")
    print(f"Search space includes: feature_selection, flatten_strategy, XGBoost params")
    print(f"{'='*60}")

    ga = GeneticAlgorithm(
        search_space=ML_SEARCH_SPACE,
        fitness_fn=fitness_fn,
        pop_size=pop_size,
        n_generations=n_generations,
        random_state=random_state,
    )
    best_params = ga.run(verbose=True)

    if save_dir:
        ga.save_results(save_dir, tag='ga_ml')

    return {
        'best_params': best_params,
        'best_rmse': ga.best_fitness,
        'ga': ga,
    }


# ==============================================================================
# Stage 2: DL Branch GA
# ==============================================================================

def run_dl_ga(
    X_train_dict: Dict[str, np.ndarray],
    y_train_dict: Dict[str, np.ndarray],
    train_df_dict: Dict[str, 'pd.DataFrame'],
    feature_names_dict: Dict[str, List[str]],
    window_size: int = 30,
    pad: bool = False,
    pop_size: int = 20,
    n_generations: int = 30,
    max_epochs: int = 50,
    val_ratio: float = 0.2,
    random_state: int = 42,
    save_dir: str = None,
) -> Dict:
    """
    Run GA to optimize BiLSTM + Attention hyperparameters including feature selection.

    Args:
        X_train_dict: {'correlation': X, 'aficv': X, 'sensor_only': X}
        y_train_dict: {'correlation': y, 'aficv': y, 'sensor_only': y}
        train_df_dict: {'correlation': df, 'aficv': df, 'sensor_only': df}
        feature_names_dict: {'correlation': [...], 'aficv': [...], 'sensor_only': [...]}

    Fitness = RMSE on validation set (per-unit, last window).
    EarlyStopping with patience=5 to speed up evaluation.
    """
    import tensorflow as tf
    from attention import build_dual_attention_bilstm

    # --- Precompute train/val splits for each feature selection ---
    splits = {}
    for fs_key in X_train_dict:
        X = X_train_dict[fs_key]
        y = y_train_dict[fs_key]
        df = train_df_dict[fs_key]

        windows_per_unit = []
        unit_ids_ordered = []
        for u in sorted(df['unit'].unique()):
            T = len(df[df['unit'] == u])
            n_win = T if pad else max(T - (window_size - 1), 0)
            windows_per_unit.append(n_win)
            unit_ids_ordered.append(u)

        sample_units = np.concatenate([
            np.full(n, u) for u, n in zip(unit_ids_ordered, windows_per_unit)
        ])

        gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=random_state)
        train_idx, val_idx = next(gss.split(X, y, groups=sample_units))

        splits[fs_key] = {
            'X_tr': X[train_idx], 'X_val': X[val_idx],
            'y_tr': y[train_idx], 'y_val': y[val_idx],
            'val_units': sample_units[val_idx],
            'n_features': X.shape[2],
            'n_train_units': len(set(sample_units[train_idx])),
            'n_val_units': len(set(sample_units[val_idx])),
        }

    for fs_key, s in splits.items():
        print(f"[DL GA] {fs_key}: Train {len(s['X_tr'])} ({s['n_train_units']} units), "
              f"Val {len(s['X_val'])} ({s['n_val_units']} units), "
              f"Features: {s['n_features']}")

    def _eval_per_unit_rmse(y_true, y_pred, unit_labels):
        preds_last = []
        true_last = []
        for u in sorted(set(unit_labels)):
            mask = unit_labels == u
            if mask.sum() > 0:
                preds_last.append(y_pred[mask][-1])
                true_last.append(y_true[mask][-1])
        return np.sqrt(mean_squared_error(true_last, preds_last))

    # --- Fitness function ---
    def fitness_fn(params):
        fs = params['feature_selection']
        s = splits[fs]

        tf.keras.backend.clear_session()

        model, _ = build_dual_attention_bilstm(
            window_size=window_size,
            n_features=s['n_features'],
            lstm_units=params['lstm_units'],
            feature_attention_dim=params['feature_attention_dim'],
            temporal_attention_dim=params['temporal_attention_dim'],
            dropout_rate=params['dropout_rate'],
            dense_units=params['dense_units'],
            learning_rate=params['learning_rate'],
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=5, restore_best_weights=True
            ),
        ]

        model.fit(
            s['X_tr'], s['y_tr'],
            epochs=max_epochs,
            batch_size=params['batch_size'],
            validation_data=(s['X_val'], s['y_val']),
            callbacks=callbacks,
            verbose=0,
        )

        y_pred_val = model.predict(s['X_val'], batch_size=256, verbose=0).flatten()
        rmse = _eval_per_unit_rmse(s['y_val'], y_pred_val, s['val_units'])

        del model
        tf.keras.backend.clear_session()

        return rmse

    # --- Run GA ---
    print(f"\n{'='*60}")
    print(f"DL GA — {pop_size} pop × {n_generations} gen (max {max_epochs} epochs each)")
    print(f"Search space includes: feature_selection, BiLSTM+Attention params")
    print(f"{'='*60}")

    ga = GeneticAlgorithm(
        search_space=DL_SEARCH_SPACE,
        fitness_fn=fitness_fn,
        pop_size=pop_size,
        n_generations=n_generations,
        random_state=random_state,
    )
    best_params = ga.run(verbose=True)

    if save_dir:
        ga.save_results(save_dir, tag='ga_dl')

    return {
        'best_params': best_params,
        'best_rmse': ga.best_fitness,
        'ga': ga,
    }
