"""
Asset-History BI Generator (controlled experiment) — C-MAPSS FD001
==================================================================
Generates a SEPARATE BI block describing each engine's history, as it would
be recorded in enterprise systems (ERP traceability, CMMS maintenance log):

    component_batch            categorical  B1..B8      ERP / traceability
    supplier                   binary       0=OEM, 1=Alternative   ERP / purchasing
    last_overhaul_type         binary       0=Light, 1=Full        CMMS
    prior_unplanned_removals   count        0, 1, 2, ...           CMMS
    hours_since_overhaul_entry continuous   hours                  CMMS

All five variables are STATIC: one value per engine, known when the engine
enters service, never updated cycle by cycle. They are joined to the sensor
data on `unit` only.

Why they can carry lifetime information (real-world motivation)
---------------------------------------------------------------
Fleet records describe an engine's history, which sensors cannot observe:
production-lot defects (e.g. the 2023 PW1100G powder-metal HPT-disk issue,
where risk was identified from part serial numbers and production dates),
repair workscope, supplier, and "bad actor" removal history. Reliability
engineers routinely compare life distributions by batch or supplier.

How the link is simulated
-------------------------
C-MAPSS lifetimes are fixed by the simulator, so the link is created in the
generator. Each engine gets a hidden robustness score

    r_u = rho * z(life_u) + sqrt(1 - rho^2) * e_u,   e_u ~ N(0, 1)

where z(life_u) is the normal score of the engine's total life and rho
controls the strength of the link. The visible variables are then drawn
from r_u with additional noise, so each one is only a weak hint and none
is a deterministic function of lifetime. rho = 0 gives pure noise.

The same random draws (e_u and all secondary noise) are reused for every
rho, so the files differ ONLY by the strength of the link.

Leakage safeguards
------------------
- Only train_FD001.txt (run-to-failure) is read. The test file and the
  RUL ground-truth file are never opened. The experiment uses unit-level
  cross-validation on the 100 training engines.
- Variables are static per engine: no cycle-by-cycle dependence on
  degradation.
- The hidden score r_u is NOT written to the asset files. It is written
  to a separate audit file, used only for checks, never as a feature.

Output (does not touch existing BI files)
-----------------------------------------
    data/raw_data/BI_data/asset/BI_FD001_asset_rho{00,02,04,06}.csv
    data/raw_data/BI_data/asset/BI_FD001_asset_audit.csv

Run from the project root:
    python src/generate_bi_asset_data.py

Author: Fatima Khadija Benzine
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

SEED = 2026
RHO_LEVELS = [0.0, 0.2, 0.4, 0.6]
N_BATCHES = 8
BATCH_REASSIGN_FRAC = 0.20       # share of engines moved to a random batch
FLIGHT_HOURS_PER_CYCLE = 1.5      # 1 cycle ~ 1 flight ~ 90 min (Saxena et al., 2008)

ASSET_VARS = ['component_batch', 'supplier', 'last_overhaul_type',
              'prior_unplanned_removals', 'hours_since_overhaul_entry']
ASSET_CATEGORICAL = ['component_batch']
ASSET_BINARY = ['supplier', 'last_overhaul_type']
ASSET_CONTINUOUS = ['prior_unplanned_removals', 'hours_since_overhaul_entry']


def rho_tag(rho: float) -> str:
    return f"rho{int(round(rho * 10)):02d}"


def read_train_lifetimes(cmapss_dir: Path, dataset_name: str = 'FD001') -> pd.Series:
    """Total life per training unit (run-to-failure). Test/RUL files not read."""
    cols = ['unit', 'cycle'] + [f'op_{i}' for i in range(1, 4)] + \
           [f's_{i}' for i in range(1, 22)]
    path = cmapss_dir / f"train_{dataset_name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Train file not found: {path}")
    df = pd.read_csv(path, sep=r'\s+', header=None, names=cols, engine='python')
    return df.groupby('unit')['cycle'].max().rename('total_life')


def normal_scores(x: pd.Series) -> np.ndarray:
    """Rank-based normal scores (robust standardisation of lifetime)."""
    ranks = x.rank(method='average').values
    return norm.ppf((ranks - 0.5) / len(x))


def draw_common_noise(n_units: int, rng: np.random.Generator) -> dict:
    """All random draws, shared across rho levels."""
    return {
        'e_latent': rng.standard_normal(n_units),
        'batch_reassign_mask': rng.random(n_units) < BATCH_REASSIGN_FRAC,
        'batch_random': rng.integers(0, N_BATCHES, n_units),
        'batch_label_perm': rng.permutation(N_BATCHES),
        'u_supplier': rng.random(n_units),
        'u_overhaul': rng.random(n_units),
        'u_removals': rng.random(n_units),
        'e_hours': rng.standard_normal(n_units),
    }


def generate_asset_block(lifetimes: pd.Series, rho: float, noise: dict) -> pd.DataFrame:
    """Static asset-history variables for every unit at link strength rho."""
    units = lifetimes.index.values
    z_life = normal_scores(lifetimes)
    r = rho * z_life + np.sqrt(1.0 - rho ** 2) * noise['e_latent']

    # --- component_batch: production lots ordered by robustness, with mixing.
    # Engines are cut into 8 equal groups by r, then 20% are moved to a random
    # batch (lots overlap in quality). Labels are permuted so B1 is not
    # automatically the best or worst lot.
    order = np.argsort(np.argsort(r))                      # rank 0..n-1
    batch_idx = (order * N_BATCHES) // len(units)
    batch_idx = np.where(noise['batch_reassign_mask'], noise['batch_random'], batch_idx)
    batch_idx = noise['batch_label_perm'][batch_idx]
    component_batch = np.array([f"B{i + 1}" for i in batch_idx])

    # --- supplier: less robust engines more often carry alternative-supplier parts
    p_alt = 1.0 / (1.0 + np.exp(0.8 * r + 0.4))            # ~40% alternative overall
    supplier = (noise['u_supplier'] < p_alt).astype(int)

    # --- last_overhaul_type: robust engines more often had a full overhaul
    p_full = 1.0 / (1.0 + np.exp(-0.8 * r))                # ~50% full overall
    last_overhaul_type = (noise['u_overhaul'] < p_full).astype(int)

    # --- prior_unplanned_removals: Poisson, "bad actors" remove more often
    lam = 0.8 * np.exp(-0.5 * r)
    from scipy.stats import poisson
    prior_removals = poisson.ppf(noise['u_removals'], lam).astype(int)

    # --- hours_since_overhaul_entry: weaker engines enter with more hours
    hours = np.exp(7.0 - 0.3 * r + 0.35 * noise['e_hours'])   # median ~1100 h
    hours = np.clip(hours, 50, 4000).round(0)

    return pd.DataFrame({
        'unit_id': units,
        'component_batch': component_batch,
        'supplier': supplier,
        'last_overhaul_type': last_overhaul_type,
        'prior_unplanned_removals': prior_removals,
        'hours_since_overhaul_entry': hours,
    }), r


def generate_all(cmapss_dir: Path, output_dir: Path, dataset_name: str = 'FD001',
                 seed: int = SEED, rho_levels=RHO_LEVELS) -> dict:
    """Write one asset file per rho plus an audit file. Returns {rho: DataFrame}."""
    output_dir.mkdir(parents=True, exist_ok=True)
    lifetimes = read_train_lifetimes(cmapss_dir, dataset_name)
    rng = np.random.default_rng(seed)
    noise = draw_common_noise(len(lifetimes), rng)

    audit = pd.DataFrame({'unit_id': lifetimes.index.values,
                          'total_life': lifetimes.values})
    out = {}
    for rho in rho_levels:
        block, r = generate_asset_block(lifetimes, rho, noise)
        path = output_dir / f"BI_{dataset_name}_asset_{rho_tag(rho)}.csv"
        block.to_csv(path, index=False)
        audit[f"latent_{rho_tag(rho)}"] = r
        out[rho] = block
        print(f"  {path.name}: {block.shape}")
    audit_path = output_dir / f"BI_{dataset_name}_asset_audit.csv"
    audit.to_csv(audit_path, index=False)
    print(f"  {audit_path.name}: audit only (never used as a feature)")
    return out


if __name__ == '__main__':
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    candidates = [project_root / 'data' / 'raw_data' / 'C_MAPSS',
                  project_root / 'data' / 'raw' / 'C_MAPSS',
                  Path('data') / 'raw_data' / 'C_MAPSS']
    cmapss_dir = next((c for c in candidates if (c / 'train_FD001.txt').exists()), None)
    if cmapss_dir is None:
        print("ERROR: train_FD001.txt not found.")
        sys.exit(1)
    output_dir = project_root / 'data' / 'raw_data' / 'BI_data' / 'asset'
    print(f"C-MAPSS: {cmapss_dir.resolve()}\nOutput:  {output_dir.resolve()}")
    generate_all(cmapss_dir, output_dir)
    print("✓ Done.")
