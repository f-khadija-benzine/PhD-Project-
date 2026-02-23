"""
Synthetic BI Data Generator for C-MAPSS Datasets (FD001-FD004)
==============================================================
Generates realistic Business Intelligence data aligned with turbofan engine
sensor data for the proposed framework's BI fusion pipeline.

Design Principles:
- Multi-Δ update frequencies (realistic enterprise system refresh rates)
- Mixed correlation: some BI vars correlated with degradation, others independent
- Realistic aerospace MRO (Maintenance, Repair, Overhaul) economics
- Consistent with Section III-B3 of the paper (temporal alignment, encoding)

Reference costs based on real aerospace MRO industry data:
- Turbofan engine PM (scheduled shop visit): $50,000 - $150,000
- Turbofan engine CM (unplanned removal): $200,000 - $800,000
- CM/PM ratio: typically 3x-8x in aerospace
- Aircraft on ground (AOG) penalty: $10,000 - $150,000/hour
- Engine shop visit duration: 40-90 days (PM), 60-120+ days (CM)
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)

# ===========================================================================
# C-MAPSS Dataset Specifications
# ===========================================================================
CMAPSS_SPECS = {
    'FD001': {'n_train_units': 100, 'n_test_units': 100, 'op_conditions': 1, 'fault_modes': 1,
              'avg_life': 206, 'min_life': 128, 'max_life': 362},
    'FD002': {'n_train_units': 260, 'n_test_units': 259, 'op_conditions': 6, 'fault_modes': 1,
              'avg_life': 206, 'min_life': 128, 'max_life': 378},
    'FD003': {'n_train_units': 100, 'n_test_units': 100, 'op_conditions': 1, 'fault_modes': 2,
              'avg_life': 247, 'min_life': 145, 'max_life': 525},
    'FD004': {'n_train_units': 249, 'n_test_units': 248, 'op_conditions': 6, 'fault_modes': 2,
              'avg_life': 246, 'min_life': 128, 'max_life': 543},
}

# ===========================================================================
# BI Variable Definitions with Update Frequencies (Δ in cycles)
# ===========================================================================
# Based on Section III-B3 and BI Data Collection Guide
BI_VARIABLES = {
    # --- Cost Structure (updated infrequently, from ERP) ---
    'pm_cost':              {'delta': 50, 'type': 'continuous', 'correlation': 'independent'},
    'cm_cost':              {'delta': 50, 'type': 'continuous', 'correlation': 'independent'},
    'labor_rate_standard':  {'delta': 50, 'type': 'continuous', 'correlation': 'independent'},
    'labor_rate_overtime':  {'delta': 50, 'type': 'continuous', 'correlation': 'independent'},

    # --- Production Economics (updated per shift ~ every 10 cycles) ---
    'production_priority':  {'delta': 10, 'type': 'categorical', 'correlation': 'degradation_correlated'},
    'downtime_penalty':     {'delta': 10, 'type': 'continuous', 'correlation': 'degradation_correlated'},
    'revenue_per_hour':     {'delta': 25, 'type': 'continuous', 'correlation': 'independent'},

    # --- Resource Availability (updated daily ~ every 25 cycles) ---
    'spare_parts_available':{'delta': 25, 'type': 'binary',     'correlation': 'stochastic'},
    'spare_parts_lead_time':{'delta': 25, 'type': 'continuous', 'correlation': 'stochastic'},
    'technician_available': {'delta': 25, 'type': 'binary',     'correlation': 'stochastic'},

    # --- Operational Constraints ---
    'maintenance_window':   {'delta': 10, 'type': 'binary',     'correlation': 'pattern_based'},
    'shift_pattern':        {'delta': 10, 'type': 'categorical', 'correlation': 'pattern_based'},
    'contract_penalty_active': {'delta': 25, 'type': 'binary',  'correlation': 'stochastic'},
}

# ===========================================================================
# Realistic Aerospace MRO Cost Parameters
# ===========================================================================
# Base costs in thousands of USD — scaled to C-MAPSS simulation context
# We use a scaled-down version to represent per-cycle operational costs
COST_PARAMS = {
    'FD001': {  # Single condition, single fault — standard fleet
        'pm_cost_base': 500, 'pm_cost_std': 75,        # $/event (scaled)
        'cm_cost_base': 3500, 'cm_cost_std': 500,      # $/event (scaled)
        'downtime_base': 5000, 'downtime_high': 15000,  # $/hour
        'revenue_base': 8000, 'revenue_std': 1000,       # $/hour
        'labor_std': 85, 'labor_ot': 135,                # $/hour
    },
    'FD002': {  # 6 conditions — variable operating costs
        'pm_cost_base': 600, 'pm_cost_std': 120,
        'cm_cost_base': 4200, 'cm_cost_std': 800,
        'downtime_base': 6000, 'downtime_high': 18000,
        'revenue_base': 9000, 'revenue_std': 1500,
        'labor_std': 90, 'labor_ot': 145,
    },
    'FD003': {  # 2 fault modes — higher uncertainty in costs
        'pm_cost_base': 550, 'pm_cost_std': 100,
        'cm_cost_base': 4000, 'cm_cost_std': 700,
        'downtime_base': 5500, 'downtime_high': 16000,
        'revenue_base': 8500, 'revenue_std': 1200,
        'labor_std': 88, 'labor_ot': 140,
    },
    'FD004': {  # 6 conditions + 2 faults — most complex, highest costs
        'pm_cost_base': 650, 'pm_cost_std': 140,
        'cm_cost_base': 4800, 'cm_cost_std': 900,
        'downtime_base': 7000, 'downtime_high': 20000,
        'revenue_base': 10000, 'revenue_std': 2000,
        'labor_std': 95, 'labor_ot': 150,
    },
}

# ===========================================================================
# Generator Functions
# ===========================================================================

def generate_unit_lifetime(specs):
    """Generate realistic lifetimes for each unit based on dataset statistics."""
    n_units = specs['n_train_units']
    lifetimes = np.random.randint(specs['min_life'], specs['max_life'] + 1, size=n_units)
    return lifetimes


def compute_health_index(cycle, max_cycle, fault_modes=1):
    """
    Compute normalized health index: 1.0 (healthy) -> 0.0 (failure).
    Used to correlate BI variables with degradation state.
    """
    hi = 1.0 - (cycle / max_cycle)
    return np.clip(hi, 0.0, 1.0)


def generate_pm_cost(n_periods, params):
    """PM costs: independent of degradation, slight random walk (market fluctuations)."""
    base = params['pm_cost_base']
    std = params['pm_cost_std']
    costs = [base + np.random.normal(0, std * 0.3)]
    for _ in range(1, n_periods):
        drift = np.random.normal(0, std * 0.05)
        costs.append(np.clip(costs[-1] + drift, base * 0.7, base * 1.4))
    return np.array(costs).round(2)


def generate_cm_cost(n_periods, params):
    """CM costs: independent, higher variance, occasional spikes (supply chain)."""
    base = params['cm_cost_base']
    std = params['cm_cost_std']
    costs = [base + np.random.normal(0, std * 0.3)]
    for _ in range(1, n_periods):
        spike = np.random.choice([0, 1], p=[0.92, 0.08])
        drift = np.random.normal(0, std * 0.08) + spike * np.random.uniform(500, 1500)
        costs.append(np.clip(costs[-1] + drift, base * 0.6, base * 1.8))
    return np.array(costs).round(2)


def generate_production_priority(n_periods, health_indices):
    """
    Production priority: CORRELATED with degradation.
    As health degrades, priority tends to increase (more urgent to keep running).
    Categories: Low=0, Medium=1, High=2
    """
    priorities = []
    for i, hi in enumerate(health_indices):
        if hi > 0.7:
            p = np.random.choice([0, 1, 2], p=[0.5, 0.35, 0.15])
        elif hi > 0.4:
            p = np.random.choice([0, 1, 2], p=[0.15, 0.50, 0.35])
        else:
            p = np.random.choice([0, 1, 2], p=[0.05, 0.25, 0.70])
        priorities.append(p)
    return np.array(priorities)


def generate_downtime_penalty(n_periods, health_indices, params):
    """
    Downtime penalty: CORRELATED with degradation and priority.
    Higher penalty as equipment becomes more critical (degraded but still running).
    """
    base = params['downtime_base']
    high = params['downtime_high']
    penalties = []
    for hi in health_indices:
        severity_factor = 1.0 + 2.0 * (1.0 - hi)  # 1x at healthy, 3x near failure
        noise = np.random.normal(0, base * 0.1)
        penalty = base * severity_factor + noise
        penalties.append(np.clip(penalty, base * 0.5, high * 1.2))
    return np.array(penalties).round(2)


def generate_revenue_per_hour(n_periods, params):
    """Revenue: independent, slight seasonal pattern."""
    base = params['revenue_base']
    std = params['revenue_std']
    t = np.arange(n_periods)
    seasonal = 0.1 * base * np.sin(2 * np.pi * t / max(n_periods, 1) * 2)
    noise = np.random.normal(0, std * 0.3, n_periods)
    return np.clip(base + seasonal + noise, base * 0.6, base * 1.5).round(2)


def generate_spare_parts(n_periods):
    """Spare parts availability: stochastic with autocorrelation."""
    available = [np.random.choice([0, 1], p=[0.15, 0.85])]
    for _ in range(1, n_periods):
        if available[-1] == 1:
            available.append(np.random.choice([0, 1], p=[0.08, 0.92]))
        else:
            available.append(np.random.choice([0, 1], p=[0.40, 0.60]))
    return np.array(available)


def generate_lead_time(n_periods, parts_available):
    """Lead time: depends on parts availability."""
    lead_times = []
    for avail in parts_available:
        if avail == 1:
            lead_times.append(np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1]))
        else:
            lead_times.append(np.random.choice([7, 14, 21, 30], p=[0.3, 0.35, 0.25, 0.1]))
    return np.array(lead_times)


def generate_technician_availability(n_periods):
    """Technician availability: stochastic, slightly autocorrelated."""
    avail = [np.random.choice([0, 1], p=[0.10, 0.90])]
    for _ in range(1, n_periods):
        if avail[-1] == 1:
            avail.append(np.random.choice([0, 1], p=[0.05, 0.95]))
        else:
            avail.append(np.random.choice([0, 1], p=[0.55, 0.45]))
    return np.array(avail)


def generate_maintenance_window(n_periods):
    """Maintenance window: pattern-based (every 3rd period is a window)."""
    pattern = []
    for i in range(n_periods):
        # ~30% of periods have a maintenance window (night shift, weekends)
        if i % 3 == 0:
            pattern.append(1)
        else:
            pattern.append(np.random.choice([0, 1], p=[0.85, 0.15]))
    return np.array(pattern)


def generate_shift_pattern(n_periods):
    """Shift pattern: categorical cycling (Day=0, Evening=1, Night=2)."""
    return np.array([i % 3 for i in range(n_periods)])


def generate_contract_penalty(n_periods):
    """Contract penalty active: stochastic, ~20% of time."""
    active = [np.random.choice([0, 1], p=[0.80, 0.20])]
    for _ in range(1, n_periods):
        if active[-1] == 1:
            active.append(np.random.choice([0, 1], p=[0.40, 0.60]))
        else:
            active.append(np.random.choice([0, 1], p=[0.88, 0.12]))
    return np.array(active)


def expand_bi_to_cycles(bi_values, delta, max_cycle):
    """
    Expand BI values (sampled at delta intervals) to per-cycle values.
    Implements Eq. 4 from Section III-B3: b_k(t) = b_k^(p) for all t in [pΔ, (p+1)Δ)
    """
    expanded = np.empty(max_cycle)
    for cycle in range(max_cycle):
        period_idx = min(cycle // delta, len(bi_values) - 1)
        expanded[cycle] = bi_values[period_idx]
    return expanded


def generate_bi_for_unit(unit_id, max_cycle, params, dataset_name):
    """Generate complete BI data for a single engine unit."""
    cycles = np.arange(1, max_cycle + 1)
    health_indices_10 = [compute_health_index(c, max_cycle) for c in range(0, max_cycle, 10)]
    health_indices_25 = [compute_health_index(c, max_cycle) for c in range(0, max_cycle, 25)]
    
    n_periods_10 = len(range(0, max_cycle, 10))
    n_periods_25 = len(range(0, max_cycle, 25))
    n_periods_50 = len(range(0, max_cycle, 50))

    # Generate BI variables at their native Δ frequencies
    pm_cost_raw = generate_pm_cost(n_periods_50, params)
    cm_cost_raw = generate_cm_cost(n_periods_50, params)
    labor_std_raw = np.full(n_periods_50, params['labor_std']) + np.random.normal(0, 3, n_periods_50)
    labor_ot_raw = np.full(n_periods_50, params['labor_ot']) + np.random.normal(0, 5, n_periods_50)

    priority_raw = generate_production_priority(n_periods_10, health_indices_10)
    downtime_raw = generate_downtime_penalty(n_periods_10, health_indices_10, params)
    revenue_raw = generate_revenue_per_hour(n_periods_25, params)

    parts_avail_raw = generate_spare_parts(n_periods_25)
    lead_time_raw = generate_lead_time(n_periods_25, parts_avail_raw)
    tech_avail_raw = generate_technician_availability(n_periods_25)

    maint_window_raw = generate_maintenance_window(n_periods_10)
    shift_raw = generate_shift_pattern(n_periods_10)
    contract_raw = generate_contract_penalty(n_periods_25)

    # Expand to per-cycle using Eq. 4
    data = {
        'unit_id': np.full(max_cycle, unit_id, dtype=int),
        'cycle': cycles,
        'pm_cost': expand_bi_to_cycles(pm_cost_raw, 50, max_cycle),
        'cm_cost': expand_bi_to_cycles(cm_cost_raw, 50, max_cycle),
        'labor_rate_standard': expand_bi_to_cycles(labor_std_raw.round(2), 50, max_cycle),
        'labor_rate_overtime': expand_bi_to_cycles(labor_ot_raw.round(2), 50, max_cycle),
        'production_priority': expand_bi_to_cycles(priority_raw, 10, max_cycle),
        'downtime_penalty': expand_bi_to_cycles(downtime_raw, 10, max_cycle),
        'revenue_per_hour': expand_bi_to_cycles(revenue_raw, 25, max_cycle),
        'spare_parts_available': expand_bi_to_cycles(parts_avail_raw, 25, max_cycle),
        'spare_parts_lead_time': expand_bi_to_cycles(lead_time_raw, 25, max_cycle),
        'technician_available': expand_bi_to_cycles(tech_avail_raw, 25, max_cycle),
        'maintenance_window': expand_bi_to_cycles(maint_window_raw, 10, max_cycle),
        'shift_pattern': expand_bi_to_cycles(shift_raw, 10, max_cycle),
        'contract_penalty_active': expand_bi_to_cycles(contract_raw, 25, max_cycle),
    }

    return pd.DataFrame(data)


def generate_dataset_bi(dataset_name):
    """Generate BI data for an entire C-MAPSS sub-dataset."""
    specs = CMAPSS_SPECS[dataset_name]
    params = COST_PARAMS[dataset_name]
    lifetimes = generate_unit_lifetime(specs)

    all_units = []
    for unit_idx in range(specs['n_train_units']):
        unit_id = unit_idx + 1
        max_cycle = lifetimes[unit_idx]
        unit_df = generate_bi_for_unit(unit_id, max_cycle, params, dataset_name)
        all_units.append(unit_df)

    df = pd.concat(all_units, ignore_index=True)

    # Cast integer columns
    int_cols = ['unit_id', 'cycle', 'production_priority', 'spare_parts_available',
                'spare_parts_lead_time', 'technician_available', 'maintenance_window',
                'shift_pattern', 'contract_penalty_active']
    for col in int_cols:
        df[col] = df[col].astype(int)

    return df


# ===========================================================================
# Main Generation
# ===========================================================================
if __name__ == '__main__':
    output_dir = '/home/claude/bi_data'
    os.makedirs(output_dir, exist_ok=True)

    for ds_name in ['FD001', 'FD002', 'FD003', 'FD004']:
        print(f"Generating BI data for {ds_name}...")
        df = generate_dataset_bi(ds_name)
        filepath = os.path.join(output_dir, f'BI_{ds_name}.csv')
        df.to_csv(filepath, index=False)
        print(f"  -> {filepath}: {len(df)} rows, {df['unit_id'].nunique()} units")
        print(f"     Columns: {list(df.columns)}")
        print(f"     Sample costs: PM={df['pm_cost'].mean():.0f}, CM={df['cm_cost'].mean():.0f}, "
              f"Ratio={df['cm_cost'].mean()/df['pm_cost'].mean():.1f}x")
        print()

    print("All datasets generated successfully.")
