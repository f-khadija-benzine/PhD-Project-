"""
Synthetic BI Data Generator for C-MAPSS Datasets (FD001-FD004)
==============================================================
Generates realistic Business Intelligence data ALIGNED with the actual
C-MAPSS turbofan sensor data — reads real train/test files to match exact
(unit_id, cycle) pairs.

Run from your project root:
    python src/generate_bi_data.py

Requires:
    - C-MAPSS files in data/raw_data/C_MAPSS/{train,test}_FD00X.txt
    - Output goes to data/raw_data/BI_data/BI_FD00X_{train,test}.csv

Design Principles (v2 — leakage-free):
    - Multi-Delta_k update frequencies (realistic enterprise system refresh)
    - NO BI variable is a function of degradation, lifetime, or RUL.
      BI describes the OPERATIONAL and ECONOMIC CONTEXT the engine runs in:
        * production_priority: driven by a per-unit criticality class
          (fleet role, fixed for the engine's life) + schedule fluctuation
        * downtime_penalty: driven by priority x shift x contract x market
        * pm_cost, cm_cost, labor rates: contract/market-driven,
          mean-reverting (no time trend)
        * revenue_per_hour: seasonal with an ABSOLUTE period and a random
          per-unit phase (not scaled to each unit's lifetime)
        * spare parts, technician, contract: stochastic, autocorrelated
    - Test BI is generated ONLY for observed cycles; the ground-truth
      RUL file is never read.
    - Realistic aerospace MRO economics (CM/PM ratio: 5-10x)
    - 1 cycle = 1 complete flight (~90 min) per Saxena et al. (2008)

Change log:
    v1: production_priority and downtime_penalty were functions of
        health_index = 1 - cycle/max_cycle (= normalized RUL), giving
        within-unit rho(downtime_penalty, RUL) = -0.985. Test BI used
        max_cycle = observed + true RUL, leaking test ground truth.
    v2: all degradation/lifetime dependence removed (this file).

Author: Fatima Khadija Benzine
"""

import pandas as pd
import numpy as np
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

# Revenue seasonality: absolute period in cycles (~1 cycle = 1 flight),
# identical for every unit, so it carries no lifetime information.
REVENUE_PERIOD_CYCLES = 300


# ============================================================================
# Read real C-MAPSS observed lengths
# ============================================================================

def read_cmapss_lengths(cmapss_dir: Path, dataset_name: str):
    """
    Read C-MAPSS train and test files, return the last OBSERVED cycle per unit.

    Train: last observed cycle = failure cycle (run-to-failure).
    Test:  last observed cycle = truncation point. The RUL ground-truth file
           is deliberately NOT read — BI must not depend on it.

    Returns:
        (train_lengths, test_lengths): dicts {unit_id: n_cycles}
    """
    cols = ['unit', 'cycle'] + [f'op_{i}' for i in range(1, 4)] + \
           [f's_{i}' for i in range(1, 22)]

    train_file = cmapss_dir / f"train_{dataset_name}.txt"
    if not train_file.exists():
        raise FileNotFoundError(
            f"C-MAPSS train file not found: {train_file}\n"
            f"Expected in: {cmapss_dir}"
        )
    train_df = pd.read_csv(train_file, sep=r'\s+', header=None, names=cols, engine='python')
    train_lengths = train_df.groupby('unit')['cycle'].max().to_dict()
    print(f"  Train: {len(train_lengths)} units, "
          f"cycles [{min(train_lengths.values())}, {max(train_lengths.values())}]")

    test_lengths = {}
    test_file = cmapss_dir / f"test_{dataset_name}.txt"
    if test_file.exists():
        test_df = pd.read_csv(test_file, sep=r'\s+', header=None, names=cols, engine='python')
        test_lengths = test_df.groupby('unit')['cycle'].max().to_dict()
        print(f"  Test:  {len(test_lengths)} units, observed cycles "
              f"[{min(test_lengths.values())}, {max(test_lengths.values())}]")
    else:
        print("  Test:  file not found, skipping test units")

    return train_lengths, test_lengths


# ============================================================================
# Helpers
# ============================================================================

def expand_to_cycles(values, delta, n_cycles):
    """Expand BI values sampled at delta intervals to per-cycle (Eq. 4)."""
    expanded = np.empty(n_cycles)
    for t in range(n_cycles):
        idx = min(t // delta, len(values) - 1)
        expanded[t] = values[idx]
    return expanded


def resample_to_delta(values, src_delta, dst_delta, n_dst):
    """Read a Delta=src series at the sampling points of a Delta=dst series."""
    return np.array([
        values[min((i * dst_delta) // src_delta, len(values) - 1)]
        for i in range(n_dst)
    ])


# ============================================================================
# BI variable generators — none takes cycle, lifetime, or RUL as input
# ============================================================================

# --- Contract / market-driven (Delta=50) ---

def gen_pm_cost(n_periods, p):
    """Mean-reverting around the contract price (Ornstein-Uhlenbeck style)."""
    base, std = p['pm_cost_base'], p['pm_cost_std']
    vals = [base + np.random.normal(0, std * 0.3)]
    for _ in range(1, n_periods):
        nxt = vals[-1] + 0.3 * (base - vals[-1]) + np.random.normal(0, std * 0.15)
        vals.append(np.clip(nxt, base * 0.7, base * 1.4))
    return np.array(vals).round(2)


def gen_cm_cost(n_periods, p):
    """Mean-reverting with occasional supply-chain shocks that decay."""
    base, std = p['cm_cost_base'], p['cm_cost_std']
    vals = [base + np.random.normal(0, std * 0.3)]
    for _ in range(1, n_periods):
        shock = np.random.choice([0, 1], p=[0.90, 0.10]) * np.random.uniform(500, 1500)
        nxt = vals[-1] + 0.4 * (base - vals[-1]) + np.random.normal(0, std * 0.15) + shock
        vals.append(np.clip(nxt, base * 0.6, base * 1.8))
    return np.array(vals).round(2)


def gen_labor(n_periods, level, std):
    return (np.full(n_periods, level) + np.random.normal(0, std, n_periods)).round(2)


# --- Finance (Delta=25) ---

def gen_revenue(n_periods, p, phase):
    """Seasonal demand with an absolute period and a per-unit random phase."""
    base, std = p['revenue_base'], p['revenue_std']
    t_cycles = np.arange(n_periods) * 25
    seasonal = 0.1 * base * np.sin(2 * np.pi * t_cycles / REVENUE_PERIOD_CYCLES + phase)
    noise = np.random.normal(0, std * 0.3, n_periods)
    return np.clip(base + seasonal + noise, base * 0.6, base * 1.5).round(2)


# --- Production context (Delta=10) ---

def gen_production_priority(n_periods, unit_criticality):
    """
    Priority set by the engine's fleet role (criticality, fixed per unit)
    plus schedule fluctuation. Independent of degradation.
    Categories: Low=0, Medium=1, High=2
    """
    if unit_criticality >= 0.7:
        probs = [0.10, 0.30, 0.60]
    elif unit_criticality >= 0.3:
        probs = [0.30, 0.45, 0.25]
    else:
        probs = [0.55, 0.35, 0.10]
    return np.random.choice([0, 1, 2], size=n_periods, p=probs)


def gen_downtime_penalty(n_periods, p, priority, shift, contract):
    """
    Cost of taking the engine offline, driven by operational context:
      priority  Low/Med/High -> 1.0 / 1.4 / 1.8 x
      shift     Day/Eve/Night -> 1.15 / 0.90 / 0.75 x
      contract  active -> 1.25 x
      market    ~N(1, 0.15), clipped to [0.6, 1.5]
    """
    base, high = p['downtime_base'], p['downtime_high']
    shift_mult = {0: 1.15, 1: 0.90, 2: 0.75}
    out = []
    for i in range(n_periods):
        mult = (1.0 + 0.4 * priority[i]) \
               * shift_mult.get(int(shift[i]), 1.0) \
               * (1.25 if contract[i] else 1.0) \
               * np.clip(np.random.normal(1.0, 0.15), 0.6, 1.5)
        out.append(np.clip(base * mult, base * 0.3, high * 1.2))
    return np.array(out).round(2)


# --- Resources (Delta=25, stochastic autocorrelated) ---

def gen_spare_parts(n_periods):
    avail = [np.random.choice([0, 1], p=[0.15, 0.85])]
    for _ in range(1, n_periods):
        if avail[-1] == 1:
            avail.append(np.random.choice([0, 1], p=[0.08, 0.92]))
        else:
            avail.append(np.random.choice([0, 1], p=[0.40, 0.60]))
    return np.array(avail)


def gen_lead_time(parts_avail):
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


# --- Operational patterns (Delta=10) ---

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

def generate_unit_bi(unit_id, n_cycles, params):
    """Generate all BI variables for one unit over its OBSERVED cycles."""
    n10 = len(range(0, n_cycles, 10))
    n25 = len(range(0, n_cycles, 25))
    n50 = len(range(0, n_cycles, 50))

    # Per-unit context drawn once (fleet role, seasonal phase)
    criticality = np.random.uniform(0, 1)
    revenue_phase = np.random.uniform(0, 2 * np.pi)

    # Cost structure (Delta=50)
    pm = gen_pm_cost(n50, params)
    cm = gen_cm_cost(n50, params)
    labor_std = gen_labor(n50, params['labor_std'], 3)
    labor_ot = gen_labor(n50, params['labor_ot'], 5)

    # Resources & contracts (Delta=25)
    parts = gen_spare_parts(n25)
    lead = gen_lead_time(parts)
    tech = gen_technician(n25)
    contract = gen_contract_penalty(n25)
    revenue = gen_revenue(n25, params, revenue_phase)

    # Production context (Delta=10)
    shift = gen_shift_pattern(n10)
    window = gen_maintenance_window(n10)
    priority = gen_production_priority(n10, criticality)
    contract_10 = resample_to_delta(contract, 25, 10, n10)
    downtime = gen_downtime_penalty(n10, params, priority, shift, contract_10)

    return pd.DataFrame({
        'unit_id': np.full(n_cycles, unit_id, dtype=int),
        'cycle': np.arange(1, n_cycles + 1),
        'pm_cost': expand_to_cycles(pm, 50, n_cycles),
        'cm_cost': expand_to_cycles(cm, 50, n_cycles),
        'labor_rate_standard': expand_to_cycles(labor_std, 50, n_cycles),
        'labor_rate_overtime': expand_to_cycles(labor_ot, 50, n_cycles),
        'production_priority': expand_to_cycles(priority, 10, n_cycles).astype(int),
        'downtime_penalty': expand_to_cycles(downtime, 10, n_cycles),
        'revenue_per_hour': expand_to_cycles(revenue, 25, n_cycles),
        'spare_parts_available': expand_to_cycles(parts, 25, n_cycles).astype(int),
        'spare_parts_lead_time': expand_to_cycles(lead, 25, n_cycles).astype(int),
        'technician_available': expand_to_cycles(tech, 25, n_cycles).astype(int),
        'maintenance_window': expand_to_cycles(window, 10, n_cycles).astype(int),
        'shift_pattern': expand_to_cycles(shift, 10, n_cycles).astype(int),
        'contract_penalty_active': expand_to_cycles(contract, 25, n_cycles).astype(int),
    })


# ============================================================================
# Leakage self-check
# ============================================================================

def leakage_check(train_bi):
    """Print within-unit Spearman correlation of each continuous BI var with RUL."""
    df = train_bi.copy()
    df['RUL'] = df.groupby('unit_id')['cycle'].transform('max') - df['cycle']
    print("  Within-unit Spearman rho with RUL (mean over units):")
    for col in ['downtime_penalty', 'production_priority', 'cm_cost',
                'pm_cost', 'revenue_per_hour']:
        rhos = df.groupby('unit_id').apply(
            lambda g: g[col].corr(g['RUL'], method='spearman'))
        print(f"    {col:22s} {rhos.mean():+.3f}")


# ============================================================================
# Main
# ============================================================================

def generate_dataset(dataset_name, cmapss_dir, output_dir):
    """Generate BI data for one C-MAPSS dataset using real observed lengths."""
    print(f"\n{'='*50}")
    print(f"Generating BI data for {dataset_name}")
    print(f"{'='*50}")

    params = COST_PARAMS[dataset_name]
    train_lengths, test_lengths = read_cmapss_lengths(cmapss_dir, dataset_name)

    train_bi = pd.concat(
        [generate_unit_bi(u, n, params) for u, n in train_lengths.items()],
        ignore_index=True)
    train_path = output_dir / f"BI_{dataset_name}_train.csv"
    train_bi.to_csv(train_path, index=False)
    print(f"  Train BI: {train_path.name} — {train_bi.shape}")

    if test_lengths:
        test_bi = pd.concat(
            [generate_unit_bi(u, n, params) for u, n in test_lengths.items()],
            ignore_index=True)
        test_path = output_dir / f"BI_{dataset_name}_test.csv"
        test_bi.to_csv(test_path, index=False)
        print(f"  Test BI:  {test_path.name} — {test_bi.shape}")

    print(f"  PM={train_bi['pm_cost'].mean():.0f}, CM={train_bi['cm_cost'].mean():.0f}, "
          f"Ratio={train_bi['cm_cost'].mean()/train_bi['pm_cost'].mean():.1f}x")
    leakage_check(train_bi)


if __name__ == '__main__':
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent  # script lives in src/

    cmapss_candidates = [
        project_root / 'data' / 'raw_data' / 'C_MAPSS',
        project_root / 'data' / 'raw' / 'C_MAPSS',
        project_root / 'data' / 'C_MAPSS',
        Path('data') / 'raw_data' / 'C_MAPSS',
        Path('data') / 'raw' / 'C_MAPSS',
        Path('data') / 'C_MAPSS',
    ]
    cmapss_dir = next((c for c in cmapss_candidates
                       if (c / 'train_FD001.txt').exists()), None)
    if cmapss_dir is None:
        print("ERROR: Could not find C-MAPSS train files. Searched in:")
        for c in cmapss_candidates:
            print(f"  {c.resolve()}")
        sys.exit(1)
    print(f"C-MAPSS directory: {cmapss_dir.resolve()}")

    output_dir = project_root / 'data' / 'raw_data' / 'BI_data'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.resolve()}")

    for ds in ['FD001', 'FD002', 'FD003', 'FD004']:
        try:
            generate_dataset(ds, cmapss_dir, output_dir)
        except FileNotFoundError as e:
            print(f"  Skipping {ds}: {e}")

    print("\n✓ Done.")
