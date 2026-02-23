"""
Synthetic BI Data Generator for C-MAPSS Datasets (FD001-FD004)
==============================================================
Generates realistic Business Intelligence data ALIGNED with the actual
C-MAPSS turbofan sensor data — reads real train files to match exact
(unit_id, cycle) pairs.

Run from your project root:
    python src/generate_bi_data.py

Requires:
    - C-MAPSS train files in data/raw/C_MAPSS/train_FD00X.txt
    - Output goes to data/BI_data/BI_FD00X.csv

Design Principles:
    - Multi-Delta_k update frequencies (realistic enterprise system refresh)
    - Mixed correlation strategy:
        * production_priority: degradation-correlated (operational decision)
        * downtime_penalty: degradation-correlated (risk exposure)
        * pm_cost, cm_cost, labor rates: INDEPENDENT (contract/market-driven)
        * spare parts, technician: stochastic with autocorrelation
    - Realistic aerospace MRO economics (CM/PM ratio: 5-10x)
    - 1 cycle = 1 complete flight (~90 min) per Saxena et al. (2008)

Author: Fatima Khadija Benzine
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

np.random.seed(42)


# ============================================================================
# Cost parameters per dataset (scaled for simulation)
# ============================================================================
COST_PARAMS = {
    'FD001': {  # 1 op condition, 1 fault mode — standard fleet
        'pm_cost_base': 500, 'pm_cost_std': 75,
        'cm_cost_base': 3500, 'cm_cost_std': 500,
        'downtime_base': 5000, 'downtime_high': 15000,
        'revenue_base': 8000, 'revenue_std': 1000,
        'labor_std': 85, 'labor_ot': 135,
    },
    'FD002': {  # 6 op conditions, 1 fault mode — variable costs
        'pm_cost_base': 600, 'pm_cost_std': 120,
        'cm_cost_base': 4200, 'cm_cost_std': 800,
        'downtime_base': 6000, 'downtime_high': 18000,
        'revenue_base': 9000, 'revenue_std': 1500,
        'labor_std': 90, 'labor_ot': 145,
    },
    'FD003': {  # 1 op condition, 2 fault modes — higher uncertainty
        'pm_cost_base': 550, 'pm_cost_std': 100,
        'cm_cost_base': 4000, 'cm_cost_std': 700,
        'downtime_base': 5500, 'downtime_high': 16000,
        'revenue_base': 8500, 'revenue_std': 1200,
        'labor_std': 88, 'labor_ot': 140,
    },
    'FD004': {  # 6 op conditions + 2 faults — most complex
        'pm_cost_base': 650, 'pm_cost_std': 140,
        'cm_cost_base': 4800, 'cm_cost_std': 900,
        'downtime_base': 7000, 'downtime_high': 20000,
        'revenue_base': 10000, 'revenue_std': 2000,
        'labor_std': 95, 'labor_ot': 150,
    },
}


# ============================================================================
# Read real C-MAPSS lifetimes
# ============================================================================

def read_cmapss_lifetimes(cmapss_dir: Path, dataset_name: str) -> dict:
    """
    Read C-MAPSS train AND test files, extract max cycle per unit.

    For train units: max_cycle = actual run-to-failure length (full trajectory).
    For test units: max_cycle = last observed cycle + true RUL (= full life).
        This ensures BI data covers the ENTIRE life of each unit, including
        the unobserved portion, so degradation correlation is realistic.

    Returns:
        dict: {('train', unit_id): max_cycle, ('test', unit_id): max_cycle}
              keyed by (split, unit_id) to avoid collisions
    """
    cols = ['unit', 'cycle'] + [f'op_{i}' for i in range(1, 4)] + \
           [f's_{i}' for i in range(1, 22)]

    # --- Train ---
    train_file = cmapss_dir / f"train_{dataset_name}.txt"
    if not train_file.exists():
        raise FileNotFoundError(
            f"C-MAPSS train file not found: {train_file}\n"
            f"Expected in: {cmapss_dir}"
        )
    train_df = pd.read_csv(train_file, sep=r'\s+', header=None, names=cols, engine='python')
    train_lifetimes = train_df.groupby('unit')['cycle'].max().to_dict()
    print(f"  Train: {len(train_lifetimes)} units, "
          f"cycles [{min(train_lifetimes.values())}, {max(train_lifetimes.values())}]")

    # --- Test ---
    test_file = cmapss_dir / f"test_{dataset_name}.txt"
    rul_file = cmapss_dir / f"RUL_{dataset_name}.txt"

    test_lifetimes = {}
    if test_file.exists() and rul_file.exists():
        test_df = pd.read_csv(test_file, sep=r'\s+', header=None, names=cols, engine='python')
        rul_df = pd.read_csv(rul_file, sep=r'\s+', header=None, names=['rul'], engine='python')
        test_max_cycles = test_df.groupby('unit')['cycle'].max()

        for i, (unit_id, max_obs_cycle) in enumerate(test_max_cycles.items()):
            # Full life = observed cycles + remaining RUL
            true_rul = rul_df.iloc[i]['rul']
            test_lifetimes[unit_id] = int(max_obs_cycle + true_rul)

        print(f"  Test:  {len(test_lifetimes)} units, "
              f"full life [{min(test_lifetimes.values())}, {max(test_lifetimes.values())}]")
    else:
        print(f"  Test:  files not found, skipping test units")

    return train_lifetimes, test_lifetimes


# ============================================================================
# BI variable generators
# ============================================================================

def compute_health_index(cycle, max_cycle):
    """1.0 (healthy) -> 0.0 (failure)."""
    return max(0.0, 1.0 - cycle / max_cycle)


def expand_to_cycles(values, delta, max_cycle):
    """Expand BI values sampled at delta intervals to per-cycle (Eq. 4)."""
    expanded = np.empty(max_cycle)
    for t in range(max_cycle):
        idx = min(t // delta, len(values) - 1)
        expanded[t] = values[idx]
    return expanded


# --- Independent variables (contract/market-driven) ---

def gen_pm_cost(n_periods, p):
    base, std = p['pm_cost_base'], p['pm_cost_std']
    vals = [base + np.random.normal(0, std * 0.3)]
    for _ in range(1, n_periods):
        vals.append(np.clip(vals[-1] + np.random.normal(0, std * 0.05),
                            base * 0.7, base * 1.4))
    return np.array(vals).round(2)


def gen_cm_cost(n_periods, p):
    base, std = p['cm_cost_base'], p['cm_cost_std']
    vals = [base + np.random.normal(0, std * 0.3)]
    for _ in range(1, n_periods):
        spike = np.random.choice([0, 1], p=[0.92, 0.08])
        drift = np.random.normal(0, std * 0.08) + spike * np.random.uniform(500, 1500)
        vals.append(np.clip(vals[-1] + drift, base * 0.6, base * 1.8))
    return np.array(vals).round(2)


def gen_revenue(n_periods, p):
    base, std = p['revenue_base'], p['revenue_std']
    t = np.arange(n_periods)
    seasonal = 0.1 * base * np.sin(2 * np.pi * t / max(n_periods, 1) * 2)
    noise = np.random.normal(0, std * 0.3, n_periods)
    return np.clip(base + seasonal + noise, base * 0.6, base * 1.5).round(2)


# --- Degradation-correlated variables ---

def gen_production_priority(n_periods, health_indices):
    """Correlated: priority increases as health degrades."""
    priorities = []
    for hi in health_indices:
        if hi > 0.7:
            priorities.append(np.random.choice([0, 1, 2], p=[0.50, 0.35, 0.15]))
        elif hi > 0.4:
            priorities.append(np.random.choice([0, 1, 2], p=[0.15, 0.50, 0.35]))
        else:
            priorities.append(np.random.choice([0, 1, 2], p=[0.05, 0.25, 0.70]))
    return np.array(priorities)


def gen_downtime_penalty(n_periods, health_indices, p):
    """Correlated: penalty exposure increases with degradation."""
    base = p['downtime_base']
    penalties = []
    for hi in health_indices:
        severity = 1.0 + 2.0 * (1.0 - hi)  # 1x healthy -> 3x near failure
        noise = np.random.normal(0, base * 0.1)
        penalties.append(np.clip(base * severity + noise,
                                 base * 0.5, p['downtime_high'] * 1.2))
    return np.array(penalties).round(2)


# --- Stochastic autocorrelated variables ---

def gen_spare_parts(n_periods):
    avail = [np.random.choice([0, 1], p=[0.15, 0.85])]
    for _ in range(1, n_periods):
        if avail[-1] == 1:
            avail.append(np.random.choice([0, 1], p=[0.08, 0.92]))
        else:
            avail.append(np.random.choice([0, 1], p=[0.40, 0.60]))
    return np.array(avail)


def gen_lead_time(n_periods, parts_avail):
    return np.array([
        np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1]) if a == 1
        else np.random.choice([7, 14, 21, 30], p=[0.3, 0.35, 0.25, 0.1])
        for a in parts_avail
    ])


def gen_technician(n_periods):
    avail = [np.random.choice([0, 1], p=[0.10, 0.90])]
    for _ in range(1, n_periods):
        if avail[-1] == 1:
            avail.append(np.random.choice([0, 1], p=[0.05, 0.95]))
        else:
            avail.append(np.random.choice([0, 1], p=[0.55, 0.45]))
    return np.array(avail)


def gen_contract_penalty(n_periods):
    active = [np.random.choice([0, 1], p=[0.80, 0.20])]
    for _ in range(1, n_periods):
        if active[-1] == 1:
            active.append(np.random.choice([0, 1], p=[0.40, 0.60]))
        else:
            active.append(np.random.choice([0, 1], p=[0.88, 0.12]))
    return np.array(active)


# --- Pattern-based variables ---

def gen_maintenance_window(n_periods):
    return np.array([
        1 if i % 3 == 0 else np.random.choice([0, 1], p=[0.85, 0.15])
        for i in range(n_periods)
    ])


def gen_shift_pattern(n_periods):
    return np.array([i % 3 for i in range(n_periods)])


# ============================================================================
# Generate BI for one unit
# ============================================================================

def generate_unit_bi(unit_id, max_cycle, params):
    """Generate all BI variables for a single unit with exact max_cycle."""

    # Number of update periods at each Delta_k
    n10 = len(range(0, max_cycle, 10))
    n25 = len(range(0, max_cycle, 25))
    n50 = len(range(0, max_cycle, 50))

    # Health indices at each Delta's sampling points
    hi_10 = [compute_health_index(c, max_cycle) for c in range(0, max_cycle, 10)]
    hi_25 = [compute_health_index(c, max_cycle) for c in range(0, max_cycle, 25)]

    # Generate at native Delta, then expand to per-cycle
    cycles = np.arange(1, max_cycle + 1)

    # Cost structure (Delta=50, independent)
    pm = expand_to_cycles(gen_pm_cost(n50, params), 50, max_cycle)
    cm = expand_to_cycles(gen_cm_cost(n50, params), 50, max_cycle)
    labor_std = expand_to_cycles(
        (np.full(n50, params['labor_std']) + np.random.normal(0, 3, n50)).round(2),
        50, max_cycle)
    labor_ot = expand_to_cycles(
        (np.full(n50, params['labor_ot']) + np.random.normal(0, 5, n50)).round(2),
        50, max_cycle)

    # Production economics (Delta=10, correlated)
    priority = expand_to_cycles(gen_production_priority(n10, hi_10), 10, max_cycle)
    downtime = expand_to_cycles(gen_downtime_penalty(n10, hi_10, params), 10, max_cycle)

    # Revenue (Delta=25, independent)
    revenue = expand_to_cycles(gen_revenue(n25, params), 25, max_cycle)

    # Resources (Delta=25, stochastic)
    parts_raw = gen_spare_parts(n25)
    parts = expand_to_cycles(parts_raw, 25, max_cycle)
    lead = expand_to_cycles(gen_lead_time(n25, parts_raw), 25, max_cycle)
    tech = expand_to_cycles(gen_technician(n25), 25, max_cycle)
    contract = expand_to_cycles(gen_contract_penalty(n25), 25, max_cycle)

    # Operational (Delta=10, pattern)
    window = expand_to_cycles(gen_maintenance_window(n10), 10, max_cycle)
    shift = expand_to_cycles(gen_shift_pattern(n10), 10, max_cycle)

    return pd.DataFrame({
        'unit_id': np.full(max_cycle, unit_id, dtype=int),
        'cycle': cycles,
        'pm_cost': pm,
        'cm_cost': cm,
        'labor_rate_standard': labor_std,
        'labor_rate_overtime': labor_ot,
        'production_priority': priority.astype(int),
        'downtime_penalty': downtime,
        'revenue_per_hour': revenue,
        'spare_parts_available': parts.astype(int),
        'spare_parts_lead_time': lead.astype(int),
        'technician_available': tech.astype(int),
        'maintenance_window': window.astype(int),
        'shift_pattern': shift.astype(int),
        'contract_penalty_active': contract.astype(int),
    })


# ============================================================================
# Main
# ============================================================================

def generate_dataset(dataset_name, cmapss_dir, output_dir):
    """Generate BI data for one C-MAPSS dataset using real lifetimes."""
    print(f"\n{'='*50}")
    print(f"Generating BI data for {dataset_name}")
    print(f"{'='*50}")

    params = COST_PARAMS[dataset_name]
    train_lifetimes, test_lifetimes = read_cmapss_lifetimes(cmapss_dir, dataset_name)

    # Generate train BI
    train_units = []
    for unit_id, max_cycle in train_lifetimes.items():
        train_units.append(generate_unit_bi(unit_id, max_cycle, params))
    train_bi = pd.concat(train_units, ignore_index=True)

    train_path = output_dir / f"BI_{dataset_name}_train.csv"
    train_bi.to_csv(train_path, index=False)
    print(f"  Train BI: {train_path.name} — {train_bi.shape}")

    # Generate test BI (using full life = observed + RUL for degradation correlation)
    if test_lifetimes:
        test_units = []
        for unit_id, full_life in test_lifetimes.items():
            test_units.append(generate_unit_bi(unit_id, full_life, params))
        test_bi = pd.concat(test_units, ignore_index=True)

        test_path = output_dir / f"BI_{dataset_name}_test.csv"
        test_bi.to_csv(test_path, index=False)
        print(f"  Test BI:  {test_path.name} — {test_bi.shape}")

    print(f"  PM={train_bi['pm_cost'].mean():.0f}, CM={train_bi['cm_cost'].mean():.0f}, "
          f"Ratio={train_bi['cm_cost'].mean()/train_bi['pm_cost'].mean():.1f}x")


if __name__ == '__main__':
    # Auto-detect project structure
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent  # if script is in src/

    # Try common data locations
    cmapss_candidates = [
        project_root / 'data' / 'raw_data' / 'C_MAPSS',
        project_root / 'data' / 'raw' / 'C_MAPSS',
        project_root / 'data' / 'C_MAPSS',
        Path('data') / 'raw_data' / 'C_MAPSS',
        Path('data') / 'raw' / 'C_MAPSS',
        Path('data') / 'C_MAPSS',
    ]

    cmapss_dir = None
    for candidate in cmapss_candidates:
        if (candidate / 'train_FD001.txt').exists():
            cmapss_dir = candidate
            break

    if cmapss_dir is None:
        print("ERROR: Could not find C-MAPSS train files.")
        print("Searched in:")
        for c in cmapss_candidates:
            print(f"  {c.resolve()}")
        print("\nPlease set cmapss_dir manually in the script or pass as argument.")
        sys.exit(1)

    print(f"C-MAPSS directory: {cmapss_dir.resolve()}")

    # Output directory
    output_dir = project_root / 'data' / 'raw_data' / 'BI_data'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.resolve()}")

    for ds in ['FD001', 'FD002', 'FD003', 'FD004']:
        try:
            generate_dataset(ds, cmapss_dir, output_dir)
        except FileNotFoundError as e:
            print(f"  Skipping {ds}: {e}")

    print("\n✓ Done.")
