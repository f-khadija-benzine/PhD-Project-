# Synthetic BI Data for C-MAPSS — Documentation

## Overview

This dataset provides synthetic Business Intelligence (BI) data for the four C-MAPSS sub-datasets (FD001-FD004), designed to support the experimental validation of the "Integrated Framework for Contextualized RUL Prediction and Maintenance Decision Support".

The BI data implements the temporal alignment and fusion methodology described in Section III-B3 of the paper, with multi-delta update frequencies reflecting realistic enterprise system refresh rates.

## Files

| File         | Dataset | Units | Rows   | Op. Conditions | Fault Modes |
|:-------------|:--------|------:|-------:|---------------:|------------:|
| BI_FD001.csv | FD001   |   100 | 24,122 |              1 |           1 |
| BI_FD002.csv | FD002   |   260 | 65,055 |              6 |           1 |
| BI_FD003.csv | FD003   |   100 | 32,620 |              1 |           2 |
| BI_FD004.csv | FD004   |   249 | 87,504 |              6 |           2 |

## Variable Dictionary

### Key Columns

| Column  | Type | Description                              |
|:--------|:-----|:-----------------------------------------|
| unit_id | int  | Engine unit identifier (matches C-MAPSS) |
| cycle   | int  | Operational cycle (matches C-MAPSS)      |

### Cost Structure (source: ERP, delta = 50 cycles)

| Column              | Type  | Range     | Correlation | Description                               |
|:--------------------|:------|:----------|:------------|:------------------------------------------|
| pm_cost             | float | 350-910   | Independent | Preventive maintenance cost per event ($) |
| cm_cost             | float | 2100-8640 | Independent | Corrective maintenance cost per event ($) |
| labor_rate_standard | float | 78-108    | Independent | Standard technician hourly rate ($/hr)    |
| labor_rate_overtime | float | 120-170   | Independent | Overtime/emergency hourly rate ($/hr)     |

### Production Economics (source: MES, delta = 10-25 cycles)

| Column              | Type  | Delta | Range      | Correlation            | Description                   |
|:--------------------|:------|------:|:-----------|:-----------------------|:------------------------------|
| production_priority | int   |    10 | 0, 1, 2    | Degradation-correlated | 0=Low, 1=Medium, 2=High      |
| downtime_penalty    | float |    10 | 2500-24000 | Degradation-correlated | Production loss rate ($/hr)   |
| revenue_per_hour    | float |    25 | 4800-15000 | Independent            | Equipment output value ($/hr) |

### Resource Availability (delta = 25 cycles)

| Column                | Type | Range | Correlation               | Description                     |
|:----------------------|:-----|:------|:--------------------------|:--------------------------------|
| spare_parts_available | int  | 0, 1  | Stochastic autocorrelated | 1=In stock, 0=Must order        |
| spare_parts_lead_time | int  | 0-30  | Depends on availability   | Days to procure if not in stock |
| technician_available  | int  | 0, 1  | Stochastic autocorrelated | 1=Available, 0=Unavailable      |

### Operational Constraints (delta = 10-25 cycles)

| Column                  | Type | Delta | Range   | Correlation   | Description                   |
|:------------------------|:-----|------:|:--------|:--------------|:------------------------------|
| maintenance_window      | int  |    10 | 0, 1    | Pattern-based | 1=Window open for maintenance |
| shift_pattern           | int  |    10 | 0, 1, 2 | Pattern-based | 0=Day, 1=Evening, 2=Night     |
| contract_penalty_active | int  |    25 | 0, 1    | Stochastic    | 1=SLA penalty risk active     |

## Update Frequencies (delta)

The multi-delta design reflects realistic enterprise system update rates:

| Delta (cycles) | Industrial equivalent | Variables                                                     |
|:--------------:|:----------------------|:--------------------------------------------------------------|
|             10 | Per shift (~8h)       | production_priority, downtime_penalty, maintenance_window     |
|             25 | Daily / weekly        | revenue_per_hour, spare_parts_*, technician, contract_penalty |
|             50 | Monthly               | pm_cost, cm_cost, labor_rate_standard, labor_rate_overtime    |

Within each delta period, values remain constant, implementing Eq. 4 from the paper:

    b_k(t) = b_k^(p),  for all t in [p*delta, (p+1)*delta)

### Making Delta Learnable

The generated CSVs store data expanded at per-cycle granularity with fixed delta values. To implement learnable delta via the Genetic Algorithm (Eq. 14), two strategies are possible:

1. **Re-sample approach** (recommended): Store BI data at the finest native frequency (delta=10) and re-sample to delta_candidate during preprocessing at each GA evaluation.
2. **Re-generate approach**: Re-run the generator with a different delta at each GA iteration (cleaner but slower).

The GA then optimizes delta alongside other hyperparameters to find the optimal balance between BI information granularity and temporal stability.

## Correlation Design (Mixed Approach)

This is critical for validating that the attention mechanism learns meaningful BI-sensor interactions.

### Degradation-Correlated Variables

- **production_priority**: Shifts toward High as health degrades. Early life mean ~0.66, near-failure mean ~1.66. Rationale: degraded engines on active routes become more critical to keep operational.
- **downtime_penalty**: Increases approximately 2.5x from healthy to near-failure. Rationale: failing engines serving critical missions incur higher downtime costs.

### Independent Variables

- **pm_cost, cm_cost**: Follow random walks with occasional supply-chain spikes. Independent of engine health (market-driven pricing).
- **revenue_per_hour**: Slight seasonal pattern, no degradation link.
- **labor_rate_***: Stable with minor noise (contract-driven).

### Stochastic Autocorrelated Variables

- **spare_parts_available**: Markov-chain dynamics. If available: 92% stays available next period. If unavailable: 60% becomes available.
- **technician_available**: Similar autocorrelation reflecting staffing inertia.
- **contract_penalty_active**: Approximately 20% active with persistence.

### Pattern-Based Variables

- **maintenance_window**: Every 3rd delta-period is a guaranteed window, with 15% random windows otherwise.
- **shift_pattern**: Deterministic cycling (Day, Evening, Night).

## Cost Realism

Reference: aerospace MRO (Maintenance, Repair, Overhaul) industry benchmarks.

| Metric           | FD001 | FD002 | FD003 | FD004 | Industry standard |
|:-----------------|------:|------:|------:|------:|:------------------|
| Mean PM Cost ($) |   499 |   594 |   552 |   647 | —                 |
| Mean CM Cost ($) | 3,605 | 4,370 | 4,263 | 5,055 | —                 |
| CM/PM Ratio      |  7.2x |  7.4x |  7.7x |  7.8x | 5-10x             |

Costs increase across datasets to reflect growing operational complexity (more operating conditions and fault modes).

## How to Use with C-MAPSS Sensor Data

### Step 1: Load and Merge (Feature-Level Fusion, Eq. 6)

```python
import pandas as pd

# Load sensor data (standard C-MAPSS format)
sensor_cols = ['unit_id', 'cycle', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]
sensor_df = pd.read_csv('train_FD001.txt', sep=' ', header=None, names=sensor_cols)
sensor_df = sensor_df.dropna(axis=1)

# Load BI data
bi_df = pd.read_csv('BI_FD001.csv')

# Merge on (unit_id, cycle) — implements Eq. 6 feature-level fusion
fused_df = sensor_df.merge(bi_df, on=['unit_id', 'cycle'], how='left')

# Result: sensor features (21) + BI features (13) = 34+ features before selection
```

### Step 2: One-Hot Encoding (Eq. 5)

```python
fused_df = pd.get_dummies(
    fused_df,
    columns=['production_priority', 'shift_pattern'],
    prefix=['prio', 'shift']
)
```

### Step 3: Feature Selection (Section III-B4)

Important: BI variables are exempt from variance-based filtering. Their lower variance is an inherent characteristic of business systems, not a lack of predictive information. Apply correlation-based filtering (Eq. 12) to all features, prioritizing BI retention when choosing between correlated sensor-BI pairs.

## Ablation Study Design

The mixed correlation design enables key experiments:

| Experiment                | Configuration                                    | What it tests                                |
|:--------------------------|:-------------------------------------------------|:---------------------------------------------|
| Full BI integration       | All 13 BI variables + sensors                    | Complete framework performance               |
| Sensor-only baseline      | No BI columns                                    | Baseline without business context            |
| Correlated BI only        | production_priority + downtime_penalty + sensors  | Minimal BI contribution                      |
| Independent BI only       | Cost and resource variables + sensors             | Value of non-correlated business context     |
| Delta sensitivity         | Re-generate with delta = 5, 10, 25, 50, 100      | Optimal BI update granularity                |
| Attention weight analysis | Full BI, inspect learned attention weights        | Verify BI-to-sensor shift pattern (Sec III-C2)|

## Reproducibility

- Random seed: 42
- Generator script: generate_bi_data.py (included)
- Dependencies: pandas, numpy

```bash
python generate_bi_data.py
```
