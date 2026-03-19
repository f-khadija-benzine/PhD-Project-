"""
Scenario Generation & Cost-Benefit Analysis (Section IV.D.2-3)

Generates maintenance scenarios for a given unit based on:
    - Predicted RUL from the hybrid model
    - BI data (costs, penalties, resource availability)
    
Evaluates scenarios through cost-benefit analysis and ranks them.

Author: Fatima Khadija Benzine
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


# ==============================================================================
# Data Classes
# ==============================================================================

@dataclass
class MaintenanceScenario:
    """A single maintenance scenario with cost and risk estimates."""
    name: str
    action: str                    # 'do_nothing', 'pm_early', 'pm_optimal', 'pm_late'
    intervention_time: int         # cycle at which maintenance occurs (0 = no intervention)
    restoration_level: float       # η_PM ∈ [0, 1], 0 = no restoration, 1 = full
    
    # Costs (computed)
    intervention_cost: float = 0.0
    downtime_cost: float = 0.0
    production_loss_cost: float = 0.0
    total_cost: float = 0.0
    
    # Risk
    failure_probability: float = 0.0
    
    # Score (composite)
    score: float = 0.0
    
    # Extended RUL after PM
    rul_after_pm: float = 0.0


@dataclass
class UnitContext:
    """BI context for a specific unit."""
    unit_id: int
    current_cycle: int
    predicted_rul: float
    
    # BI indicators — costs
    pm_cost: float = 0.0
    cm_cost: float = 0.0
    downtime_penalty: float = 0.0     # cost per hour of downtime
    revenue_per_hour: float = 0.0
    
    # BI indicators — resources
    technician_available: int = 1
    spare_parts_available: int = 1
    spare_parts_lead_time: float = 0.0
    labor_rate_standard: float = 0.0
    labor_rate_overtime: float = 0.0
    contract_penalty_active: int = 0
    
    # BI indicators — operational context
    production_priority: int = 1      # 0=low, 1=medium, 2=high
    maintenance_window: int = 1       # 1=window available, 0=not available
    shift_pattern: int = 0            # 0=day, 1=evening, 2=night
    
    @property
    def eol(self) -> int:
        """Estimated end-of-life cycle."""
        return self.current_cycle + int(self.predicted_rul)
    
    @property
    def cm_to_pm_ratio(self) -> float:
        """How much more expensive is CM vs PM."""
        if self.pm_cost > 0:
            return self.cm_cost / self.pm_cost
        return 5.0  # default
    
    @property
    def priority_multiplier(self) -> float:
        """Cost multiplier based on production priority."""
        # High priority → downtime costs more
        return {0: 0.7, 1: 1.0, 2: 1.5}[self.production_priority]
    
    @property
    def priority_label(self) -> str:
        return {0: 'Low', 1: 'Medium', 2: 'High'}[self.production_priority]
    
    @property
    def shift_label(self) -> str:
        return {0: 'Day', 1: 'Evening', 2: 'Night'}[self.shift_pattern]
    
    @property
    def shift_labor_multiplier(self) -> float:
        """Labor cost multiplier based on shift pattern."""
        # Night shift → more expensive
        return {0: 1.0, 1: 1.15, 2: 1.3}[self.shift_pattern]


# ==============================================================================
# Scenario Generation
# ==============================================================================

class ScenarioGenerator:
    """
    Generates and evaluates maintenance scenarios for a given unit.
    
    Args:
        w_cost: weight for cost in composite score (default 0.6)
        w_risk: weight for risk in composite score (default 0.4)
        pm_downtime_hours: hours of downtime for planned PM (default 8)
        cm_downtime_hours: hours of downtime for unplanned CM (default 48)
    """
    
    def __init__(self, w_cost: float = 0.6, w_risk: float = 0.4,
                 pm_downtime_hours: float = 8, cm_downtime_hours: float = 48):
        self.w_cost = w_cost
        self.w_risk = w_risk
        self.pm_downtime_hours = pm_downtime_hours
        self.cm_downtime_hours = cm_downtime_hours
    
    def generate_scenarios(self, context: UnitContext,
                            n_pm_times: int = 5,
                            restoration_levels: List[float] = [0.5, 0.7, 0.9]
                            ) -> List[MaintenanceScenario]:
        """
        Generate maintenance scenarios for a unit.
        
        Scenarios:
            - Do Nothing (baseline): wait until failure
            - PM at various times before predicted EoL
            - Each PM time × each restoration level
        
        Returns:
            List of MaintenanceScenario, sorted by score (best first)
        """
        scenarios = []
        
        # --- Scenario 0: Do Nothing ---
        s0 = self._do_nothing_scenario(context)
        scenarios.append(s0)
        
        # --- PM Scenarios ---
        rul = context.predicted_rul
        if rul <= 5:
            # Too close to failure, only immediate PM
            pm_times = [context.current_cycle + 1]
        else:
            # Generate PM times: from 20% to 90% of remaining life
            fractions = np.linspace(0.2, 0.9, n_pm_times)
            pm_times = [context.current_cycle + int(f * rul) for f in fractions]
        
        for t_pm in pm_times:
            for eta in restoration_levels:
                s = self._pm_scenario(context, t_pm, eta)
                scenarios.append(s)
        
        # --- Rank by composite score ---
        max_cost = max(s.total_cost for s in scenarios) if scenarios else 1.0
        if max_cost == 0:
            max_cost = 1.0
        
        for s in scenarios:
            s.score = (self.w_cost * (s.total_cost / max_cost) +
                       self.w_risk * s.failure_probability)
        
        scenarios.sort(key=lambda s: s.score)
        
        return scenarios
    
    def _do_nothing_scenario(self, ctx: UnitContext) -> MaintenanceScenario:
        """Baseline: operate until failure, then corrective maintenance."""
        # Costs — adjusted by production priority and shift
        intervention_cost = ctx.cm_cost * ctx.shift_labor_multiplier
        downtime_cost = ctx.downtime_penalty * self.cm_downtime_hours * ctx.priority_multiplier
        
        # Production loss: degraded performance over remaining life
        # Higher priority → more revenue lost per cycle
        production_loss = ctx.revenue_per_hour * ctx.predicted_rul * 0.1 * ctx.priority_multiplier
        
        total = intervention_cost + downtime_cost + production_loss
        
        # Risk: high probability of failure (by definition)
        p_failure = min(0.95, 1.0 - (ctx.predicted_rul / 125.0))
        
        return MaintenanceScenario(
            name='Do Nothing',
            action='do_nothing',
            intervention_time=ctx.eol,
            restoration_level=0.0,
            intervention_cost=intervention_cost,
            downtime_cost=downtime_cost,
            production_loss_cost=production_loss,
            total_cost=total,
            failure_probability=p_failure,
            rul_after_pm=0.0,
        )
    
    def _pm_scenario(self, ctx: UnitContext, t_pm: int,
                      eta: float) -> MaintenanceScenario:
        """Preventive maintenance at time t_pm with restoration η."""
        cycles_until_pm = t_pm - ctx.current_cycle
        cycles_remaining = ctx.predicted_rul
        
        # Label
        fraction = cycles_until_pm / max(cycles_remaining, 1)
        if fraction < 0.4:
            label = f'PM Early (t={t_pm}, η={eta})'
        elif fraction < 0.7:
            label = f'PM Optimal (t={t_pm}, η={eta})'
        else:
            label = f'PM Late (t={t_pm}, η={eta})'
        
        # --- Intervention cost ---
        intervention_cost = ctx.pm_cost
        
        # Shift pattern: night/evening → higher labor cost
        intervention_cost *= ctx.shift_labor_multiplier
        
        # No maintenance window → must use overtime
        if not ctx.maintenance_window:
            intervention_cost += ctx.labor_rate_overtime * self.pm_downtime_hours
            label += ' [no window]'
        
        # Overtime if no technician available
        if not ctx.technician_available:
            intervention_cost += ctx.labor_rate_overtime * self.pm_downtime_hours
        
        # Spare parts lead time adds cost if not available
        if not ctx.spare_parts_available:
            intervention_cost += ctx.spare_parts_lead_time * ctx.labor_rate_standard
        
        # Contract penalty if active
        if ctx.contract_penalty_active:
            intervention_cost *= 1.2  # 20% surcharge
        
        # --- Downtime cost — adjusted by production priority ---
        downtime_cost = ctx.downtime_penalty * self.pm_downtime_hours * ctx.priority_multiplier
        
        # --- Production loss until PM — higher priority = more revenue at risk ---
        production_loss = ctx.revenue_per_hour * cycles_until_pm * 0.05 * ctx.priority_multiplier
        
        total = intervention_cost + downtime_cost + production_loss
        
        # --- Risk: probability of failure before PM ---
        if cycles_remaining > 0:
            p_failure = max(0.0, min(0.9, (fraction ** 2) * 0.8))
        else:
            p_failure = 0.95
        
        # High priority + late PM → extra risk penalty
        if ctx.production_priority == 2 and fraction > 0.7:
            p_failure = min(0.95, p_failure * 1.3)
        
        # RUL after PM
        rul_after = 125.0 - (1 - eta) * (125.0 - max(cycles_remaining - cycles_until_pm, 0))
        
        return MaintenanceScenario(
            name=label,
            action='preventive_maintenance',
            intervention_time=t_pm,
            restoration_level=eta,
            intervention_cost=intervention_cost,
            downtime_cost=downtime_cost,
            production_loss_cost=production_loss,
            total_cost=total,
            failure_probability=p_failure,
            rul_after_pm=rul_after,
        )


# ==============================================================================
# Cost-Benefit Analysis
# ==============================================================================

def cost_benefit_table(scenarios: List[MaintenanceScenario]) -> pd.DataFrame:
    """Create a comparison table of all scenarios."""
    rows = []
    for s in scenarios:
        rows.append({
            'Scenario': s.name,
            'Action': s.action,
            'Intervention Cycle': s.intervention_time,
            'Restoration (η)': s.restoration_level,
            'Intervention Cost': round(s.intervention_cost, 2),
            'Downtime Cost': round(s.downtime_cost, 2),
            'Production Loss': round(s.production_loss_cost, 2),
            'Total Cost': round(s.total_cost, 2),
            'Failure Risk': round(s.failure_probability, 3),
            'RUL After PM': round(s.rul_after_pm, 1),
            'Score': round(s.score, 4),
        })
    return pd.DataFrame(rows)


def generate_recommendation(scenarios: List[MaintenanceScenario],
                             context: UnitContext) -> str:
    """Generate a human-readable recommendation from ranked scenarios."""
    best = scenarios[0]
    do_nothing = [s for s in scenarios if s.action == 'do_nothing'][0]
    
    savings = do_nothing.total_cost - best.total_cost
    risk_reduction = do_nothing.failure_probability - best.failure_probability
    
    # Context summary
    ctx_lines = []
    ctx_lines.append(f"  Production priority: {context.priority_label}")
    ctx_lines.append(f"  Maintenance window:  {'Available' if context.maintenance_window else 'NOT available (overtime required)'}")
    ctx_lines.append(f"  Shift pattern:       {context.shift_label} (labor multiplier: {context.shift_labor_multiplier:.2f}x)")
    ctx_lines.append(f"  Technician:          {'Available' if context.technician_available else 'NOT available'}")
    ctx_lines.append(f"  Spare parts:         {'Available' if context.spare_parts_available else f'NOT available (lead time: {context.spare_parts_lead_time:.0f}h)'}")
    ctx_summary = '\n'.join(ctx_lines)
    
    if best.action == 'do_nothing':
        rec = (f"RECOMMENDATION for Unit {context.unit_id}:\n"
               f"  Action: Continue monitoring\n"
               f"  Predicted RUL: {context.predicted_rul:.0f} cycles\n"
               f"  Failure risk: {best.failure_probability:.1%}\n"
               f"  Reason: No preventive action is cost-effective at this time.\n"
               f"  Next review: in {max(int(context.predicted_rul * 0.3), 5)} cycles\n"
               f"\n  --- Operational Context ---\n{ctx_summary}")
    else:
        rec = (f"RECOMMENDATION for Unit {context.unit_id}:\n"
               f"  Action: {best.name}\n"
               f"  Schedule maintenance at cycle {best.intervention_time}\n"
               f"  Restoration level: {best.restoration_level:.0%}\n"
               f"\n  --- Cost Analysis ---\n"
               f"  Expected cost:         ${best.total_cost:,.0f}\n"
               f"    Intervention:        ${best.intervention_cost:,.0f}\n"
               f"    Downtime:            ${best.downtime_cost:,.0f}\n"
               f"    Production loss:     ${best.production_loss_cost:,.0f}\n"
               f"  Cost vs. failure:      ${savings:,.0f} saved "
               f"({savings/max(do_nothing.total_cost,1)*100:.0f}%)\n"
               f"  Risk reduction:        {risk_reduction:.1%}\n"
               f"  RUL after maintenance: {best.rul_after_pm:.0f} cycles\n"
               f"\n  --- Operational Context ---\n{ctx_summary}")
        
        # Warnings
        warnings = []
        if context.production_priority == 2:
            warnings.append("⚠ HIGH PRIORITY unit — downtime costs are elevated")
        if not context.maintenance_window:
            warnings.append("⚠ No maintenance window — overtime rates apply")
        if not context.technician_available:
            warnings.append("⚠ Technician not available — scheduling required")
        if not context.spare_parts_available:
            warnings.append(f"⚠ Spare parts not available — {context.spare_parts_lead_time:.0f}h lead time")
        if context.contract_penalty_active:
            warnings.append("⚠ Contract penalty active — 20% surcharge on intervention")
        
        if warnings:
            rec += "\n\n  --- Warnings ---\n  " + "\n  ".join(warnings)
    
    return rec


# ==============================================================================
# Visualization
# ==============================================================================

def plot_scenario_comparison(scenarios: List[MaintenanceScenario],
                              context: UnitContext,
                              save_path: str = None):
    """Plot cost and risk comparison of all scenarios."""
    df = cost_benefit_table(scenarios)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # --- Cost breakdown ---
    labels = [s.name for s in scenarios[:6]]  # top 6
    x = range(len(labels))
    
    intervention = [s.intervention_cost for s in scenarios[:6]]
    downtime = [s.downtime_cost for s in scenarios[:6]]
    production = [s.production_loss_cost for s in scenarios[:6]]
    
    axes[0].bar(x, intervention, label='Intervention', color='#E53935')
    axes[0].bar(x, downtime, bottom=intervention, label='Downtime', color='#FB8C00')
    axes[0].bar(x, production,
                bottom=[i+d for i, d in zip(intervention, downtime)],
                label='Production Loss', color='#FDD835')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    axes[0].set_ylabel('Cost ($)')
    axes[0].set_title('Cost Breakdown', fontweight='bold')
    axes[0].legend(fontsize=8)
    
    # --- Cost vs Risk scatter ---
    costs = [s.total_cost for s in scenarios]
    risks = [s.failure_probability for s in scenarios]
    colors = ['#E53935' if s.action == 'do_nothing' else '#1E88E5' for s in scenarios]
    
    axes[1].scatter(costs, risks, c=colors, s=100, edgecolors='black', zorder=3)
    for s in scenarios:
        axes[1].annotate(s.name, (s.total_cost, s.failure_probability),
                         fontsize=7, ha='center', va='bottom')
    axes[1].set_xlabel('Total Cost ($)')
    axes[1].set_ylabel('Failure Probability')
    axes[1].set_title('Cost vs Risk', fontweight='bold')
    axes[1].axhline(y=0.5, color='red', linestyle='--', alpha=0.3, label='High risk threshold')
    axes[1].legend(fontsize=8)
    
    # --- Timeline ---
    t_curr = context.current_cycle
    t_eol = context.eol
    
    axes[2].axvline(x=t_curr, color='green', linestyle='-', linewidth=2, label='Current')
    axes[2].axvline(x=t_eol, color='red', linestyle='-', linewidth=2, label='Predicted EoL')
    axes[2].axvspan(t_eol - 10, t_eol, alpha=0.2, color='red', label='Danger zone')
    
    y_pos = 0.9
    for s in scenarios[:6]:
        if s.action != 'do_nothing':
            color = '#1E88E5' if s.score == scenarios[0].score else '#90CAF9'
            axes[2].axvline(x=s.intervention_time, color=color,
                           linestyle='--', alpha=0.7)
            axes[2].text(s.intervention_time, y_pos, s.name,
                        fontsize=7, rotation=90, va='top', ha='right')
            y_pos -= 0.15
    
    axes[2].set_xlabel('Cycle')
    axes[2].set_title('Maintenance Timeline', fontweight='bold')
    axes[2].legend(fontsize=8, loc='upper left')
    axes[2].set_xlim(t_curr - 5, t_eol + 10)
    axes[2].set_ylim(0, 1)
    axes[2].set_yticks([])
    
    plt.suptitle(f'Scenario Analysis — Unit {context.unit_id} '
                 f'(Predicted RUL = {context.predicted_rul:.0f} cycles)',
                 fontweight='bold', fontsize=13)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig


def plot_degradation_trajectories(scenarios: List[MaintenanceScenario],
                                   context: UnitContext,
                                   save_path: str = None):
    """Plot projected degradation under each scenario."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    t_curr = context.current_cycle
    t_eol = context.eol
    rul = context.predicted_rul
    
    # Health index: 1.0 (healthy) → 0.0 (failure)
    cycles = np.arange(t_curr, t_eol + 20)
    
    # Do Nothing trajectory
    health_dn = np.clip(1.0 - (cycles - t_curr) / max(rul, 1), 0, 1)
    ax.plot(cycles, health_dn, 'r-', linewidth=2, label='Do Nothing')
    ax.fill_between(cycles, 0, health_dn, alpha=0.1, color='red')
    
    # PM trajectories (top 3 PM scenarios)
    pm_scenarios = [s for s in scenarios if s.action != 'do_nothing'][:3]
    colors_pm = ['#1E88E5', '#43A047', '#FB8C00']
    
    for s, color in zip(pm_scenarios, colors_pm):
        t_pm = s.intervention_time
        eta = s.restoration_level
        
        # Before PM: same degradation
        before = cycles[cycles <= t_pm]
        health_before = np.clip(1.0 - (before - t_curr) / max(rul, 1), 0, 1)
        
        # At PM: restoration
        health_at_pm = health_before[-1] if len(health_before) > 0 else 0.5
        health_restored = health_at_pm + eta * (1.0 - health_at_pm)
        
        # After PM: slower degradation from restored state
        after = cycles[cycles > t_pm]
        new_rul = s.rul_after_pm
        health_after = np.clip(health_restored - (after - t_pm) / max(new_rul, 1) * health_restored, 0, 1)
        
        full_health = np.concatenate([health_before, health_after])
        ax.plot(cycles[:len(full_health)], full_health, color=color,
                linewidth=2, linestyle='--', label=s.name)
        ax.axvline(x=t_pm, color=color, linestyle=':', alpha=0.5)
    
    # Failure threshold
    ax.axhline(y=0.0, color='black', linestyle='-', alpha=0.3)
    ax.axhline(y=0.2, color='red', linestyle='--', alpha=0.3, label='Critical threshold')
    ax.axvline(x=t_curr, color='green', linewidth=2, alpha=0.5, label='Current')
    
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Health Index')
    ax.set_title(f'Degradation Trajectories — Unit {context.unit_id}', fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig
