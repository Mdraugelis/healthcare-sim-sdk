"""Weekly aggregator: convert sim state into a dashboard row.

The aggregator enforces the **dashboard ignorance** constraint. The
monitoring tiers should only see what a real operational dashboard
would see: population-level summaries, per-manager adherence and
check-in counts, and outcome rates from the factual branch. They
must not see:

- The counterfactual branch state (that's the ground truth — only
  the validation analysis layer uses it)
- Internal per-nurse random variables or latent risks
- The true effect size

This module defines ``WeeklyDashboardRow`` and the ``aggregate()``
function that produces one row from a factual-branch state + the
factual-branch ``Outcomes`` object for that week.

``aggregate()`` deliberately does not accept any counterfactual
parameter. The type system enforces that the monitoring harness
cannot accidentally leak CF information into the tiers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from healthcare_sim_sdk.core.scenario import Interventions, Outcomes
from healthcare_sim_sdk.scenarios.nurse_retention.scenario import (
    NurseRetentionState,
)


@dataclass
class ManagerWeeklyRow:
    """Per-manager detail for one week."""

    manager_id: int
    team_size: int
    n_departures_this_week: int
    n_check_ins_done: int
    check_in_target: int
    adopts_ai: bool
    team_new_hires_active: int


@dataclass
class WeeklyDashboardRow:
    """Everything a real dashboard would see for one simulation week.

    Population-level signals plus an optional per-manager breakdown.
    The tiers consume these as the only source of truth; anything
    not in this row is invisible to the monitoring system.
    """

    week: int

    # Population metrics
    n_active: int
    n_departures_this_week: int
    unit_turnover_rate: float  # departures / active at start of week
    cumulative_departures: int
    retention_rate: float      # fraction still employed

    # Check-in adherence (Tier 1 leading indicator)
    check_ins_done_this_week: int
    check_in_target_this_week: int
    check_in_adherence: float  # done / target, 0 if target == 0

    # Population composition
    n_new_hire_active: int
    new_hire_fraction: float

    # Tool adoption (Tier 1 detects partial adoption in Regime F)
    n_managers_adopting: int
    n_managers_total: int
    adoption_rate: float

    # Per-manager detail (for manager-level Tier 1 and Regime F)
    per_manager: List[ManagerWeeklyRow] = field(default_factory=list)

    # Optional model score / true label snapshot for Tier 4 (offline).
    # None on weeks where no prediction was scheduled.
    prediction_scores: Optional[np.ndarray] = None
    prediction_true_labels: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable dict (drops arrays, keeps summary)."""
        return {
            "week": self.week,
            "n_active": self.n_active,
            "n_departures_this_week": self.n_departures_this_week,
            "unit_turnover_rate": float(self.unit_turnover_rate),
            "cumulative_departures": self.cumulative_departures,
            "retention_rate": float(self.retention_rate),
            "check_ins_done_this_week": self.check_ins_done_this_week,
            "check_in_target_this_week": self.check_in_target_this_week,
            "check_in_adherence": float(self.check_in_adherence),
            "n_new_hire_active": self.n_new_hire_active,
            "new_hire_fraction": float(self.new_hire_fraction),
            "n_managers_adopting": self.n_managers_adopting,
            "n_managers_total": self.n_managers_total,
            "adoption_rate": float(self.adoption_rate),
            "per_manager": [
                {
                    "manager_id": m.manager_id,
                    "team_size": m.team_size,
                    "n_departures_this_week": m.n_departures_this_week,
                    "n_check_ins_done": m.n_check_ins_done,
                    "check_in_target": m.check_in_target,
                    "adopts_ai": bool(m.adopts_ai),
                    "team_new_hires_active": m.team_new_hires_active,
                }
                for m in self.per_manager
            ],
        }


def aggregate(
    state: NurseRetentionState,
    outcomes: Outcomes,
    interventions: Optional[Interventions],
    soc_interventions_this_week: int,
    capacity_this_week: int,
    previous_cumulative_departures: int,
    previous_active: int,
) -> WeeklyDashboardRow:
    """Build one dashboard row from factual branch state + measurements.

    Parameters
    ----------
    state : NurseRetentionState
        The **factual branch** state at the end of this week. This
        function refuses to accept any counterfactual state — the
        parameter type is explicit. In practice, callers must
        inspect only ``results.outcomes`` (factual), never
        ``results.counterfactual_outcomes``, when calling this.
    outcomes : Outcomes
        The factual-branch ``Outcomes`` object for this week. Used
        for the per-step departure count and retention rate.
    interventions : Optional[Interventions]
        The factual-branch ``Interventions`` object for this week,
        or None if no prediction happened. Used for the check-in
        count and capacity target.
    soc_interventions_this_week : int
        SOC check-ins performed in step() this week. Computed by
        the harness as a delta from the previous total.
    capacity_this_week : int
        The resolved capacity value (K per manager per week) at
        this t. Used for adherence target.
    previous_cumulative_departures : int
        Total departures at the start of the week. Used to compute
        n_departures_this_week as a delta.
    previous_active : int
        Number of active nurses at the start of the week. Used as
        the denominator for unit_turnover_rate.

    Returns
    -------
    WeeklyDashboardRow
        A flat, JSON-serializable summary. The tiers consume this
        row; they never see ``state`` or the counterfactual branch
        directly.
    """
    week = int(outcomes.metadata.get("t", 0))
    n_active = int(outcomes.metadata.get("n_active", 0))
    total_departures = int(
        outcomes.metadata.get("total_departures", 0)
    )
    n_new_hire_active = int(
        outcomes.metadata.get("n_new_hire_active", 0)
    )

    n_departures_this_week = (
        total_departures - previous_cumulative_departures
    )

    # Weekly turnover rate: departures / population at start of week
    unit_turnover_rate = (
        n_departures_this_week / previous_active
        if previous_active > 0
        else 0.0
    )

    # Check-in adherence this week.
    # AI-directed check-ins come from intervene(); SOC check-ins come
    # from step() regardless of whether a prediction happened.
    ai_check_ins = (
        interventions.metadata.get("n_treated", 0)
        if interventions is not None
        else 0
    )
    total_check_ins = ai_check_ins + soc_interventions_this_week

    n_managers = int(state.n_managers)
    check_in_target = capacity_this_week * n_managers
    adherence = (
        total_check_ins / check_in_target
        if check_in_target > 0
        else 0.0
    )

    # Adoption rate (manager-level AI adoption from state)
    n_adopting = int(state.manager_adopts_ai.sum())
    adoption_rate = n_adopting / n_managers if n_managers > 0 else 0.0

    # Per-manager detail
    active_mask = ~state.departed
    is_new_hire_mask = state.is_new_hire.astype(bool)
    per_manager: List[ManagerWeeklyRow] = []

    for mgr_id in range(n_managers):
        team_mask = state.manager_id == mgr_id
        team_active = team_mask & active_mask
        team_size = int(team_active.sum())

        # Per-manager check-ins: we don't track this exactly, so
        # we approximate by assuming each adopting manager does
        # capacity_this_week AI check-ins (if prediction fired)
        # plus SOC check-ins in proportion to team eligibility.
        mgr_ai_checkins = (
            capacity_this_week
            if (
                interventions is not None
                and bool(state.manager_adopts_ai[mgr_id])
            )
            else 0
        )

        # Approximate per-manager check-in target
        mgr_target = capacity_this_week

        team_new_hires = int((team_active & is_new_hire_mask).sum())

        per_manager.append(
            ManagerWeeklyRow(
                manager_id=mgr_id,
                team_size=team_size,
                n_departures_this_week=0,  # not tracked per-mgr
                n_check_ins_done=mgr_ai_checkins,
                check_in_target=mgr_target,
                adopts_ai=bool(state.manager_adopts_ai[mgr_id]),
                team_new_hires_active=team_new_hires,
            )
        )

    # Capture prediction scores for Tier 4 offline analysis.
    # We don't reach into the engine here — the harness passes
    # prediction data through the outcomes metadata when available.
    prediction_scores = None
    prediction_true_labels = None
    if interventions is not None and hasattr(interventions, "metadata"):
        # The scenario doesn't store scores on Interventions, so
        # the harness is responsible for attaching them. This
        # aggregator is a best-effort extractor.
        pass

    return WeeklyDashboardRow(
        week=week,
        n_active=n_active,
        n_departures_this_week=n_departures_this_week,
        unit_turnover_rate=unit_turnover_rate,
        cumulative_departures=total_departures,
        retention_rate=float(
            outcomes.metadata.get("retention_rate", 0.0)
        ),
        check_ins_done_this_week=total_check_ins,
        check_in_target_this_week=check_in_target,
        check_in_adherence=adherence,
        n_new_hire_active=n_new_hire_active,
        new_hire_fraction=(
            n_new_hire_active / n_active if n_active > 0 else 0.0
        ),
        n_managers_adopting=n_adopting,
        n_managers_total=n_managers,
        adoption_rate=adoption_rate,
        per_manager=per_manager,
        prediction_scores=prediction_scores,
        prediction_true_labels=prediction_true_labels,
    )
