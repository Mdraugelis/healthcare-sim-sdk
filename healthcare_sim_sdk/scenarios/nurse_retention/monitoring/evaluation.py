"""Evaluate declared monitoring events against change time and follow-up.

Processing completeness is separate from monitor success. A pre-change alert
never becomes a detection of a later change. Alert counts are per run; they do
not estimate a false-alarm probability until aggregated across independent runs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np

from .events import MonitoringRun


@dataclass(frozen=True)
class MonitoringClaim:
    claim_id: str
    tier: int
    metric: str
    direction: str
    change_week: int | None
    horizon: int
    monitoring_start: int = 0
    eligibility_start: int = 0
    rule: str | None = None

    def __post_init__(self) -> None:
        if self.tier not in (1, 2, 3, 4):
            raise ValueError("tier must be 1-4")
        if self.direction not in ("up", "down", "level", "any"):
            raise ValueError("invalid claim direction")
        if min(self.horizon, self.monitoring_start, self.eligibility_start) < 0:
            raise ValueError("monitoring weeks must be nonnegative")
        if self.change_week is not None and self.change_week < 0:
            raise ValueError("change_week must be nonnegative or None for a null claim")
        if self.monitoring_start > self.horizon:
            raise ValueError("monitoring_start must not exceed horizon")


def evaluate_claim(run: MonitoringRun, claim: MonitoringClaim) -> dict[str, Any]:
    """Return directional detection, false alarms, delay and censoring.

    Windows are inclusive. ``change_week=None`` means no target event is
    expected; all alerts matching the declared metric/rule are false alerts
    for that null claim. Direction filters detection, never hides false alerts.
    A missing observation week makes evaluation INCONCLUSIVE.
    """
    observed = {int(row["week"]) for row in run.weekly_history}
    required = set(range(claim.monitoring_start, claim.horizon + 1))
    missing = sorted(required - observed)
    observed_end = min(max(observed, default=-1), claim.horizon)
    relevant = sorted([
        event for event in run.detection_events
        if event.tier == claim.tier and event.metric == claim.metric
        and (claim.rule is None or event.rule == claim.rule)
        and claim.monitoring_start <= event.week <= observed_end
        and event.week in observed
    ], key=lambda event: event.week)
    result = {
        **asdict(claim), "seed": run.seed, "regime": run.regime,
        "evaluation_status": "PASS" if not missing else "INCONCLUSIVE",
        "status_scope": "evaluation_completeness_not_monitor_efficacy",
        "observed_through_week": observed_end,
        "missing_observation_weeks": missing,
        "n_alerts_full_observed_horizon": len(relevant),
        "full_horizon_any_alert": bool(relevant),
        "pre_change_alert_count": 0,
        "post_change_wrong_direction_alert_count": 0,
        "first_post_change_detection_week": None,
        "detection_delay_weeks": None,
        "detected": None,
        "right_censored": False,
        "censoring_week": None,
        "follow_up_weeks": 0,
    }
    if claim.change_week is None:
        result.update({
            "null_claim": True, "false_alert_count": len(relevant),
            "false_alert_any": bool(relevant),
            "reason": "no target change declared; estimate full-horizon false alarms",
        })
        if observed_end < claim.eligibility_start:
            result["evaluation_status"] = "NOT_RUN"
            result["reason"] = "no observed eligible monitoring follow-up"
        return result

    pre_change = [event for event in relevant if event.week < claim.change_week]
    eligible_start = max(
        claim.change_week, claim.eligibility_start, claim.monitoring_start,
    )
    eligible = [event for event in relevant if event.week >= eligible_start]
    matches = [event for event in eligible if (
        claim.direction == "any" or event.direction == claim.direction
    )]
    result.update({
        "null_claim": False,
        "pre_change_alert_count": len(pre_change),
        "pre_change_any_alert": bool(pre_change),
        "post_change_wrong_direction_alert_count": len(eligible) - len(matches),
        "post_change_ineligible_alert_count": sum(
            claim.change_week <= event.week < eligible_start for event in relevant
        ),
        "eligible_detection_start_week": eligible_start,
        "follow_up_weeks": max(0, observed_end - eligible_start + 1),
    })
    if observed_end < eligible_start:
        result.update({
            "evaluation_status": "NOT_RUN",
            "reason": "no observed eligible post-change follow-up",
        })
    elif matches:
        first = matches[0]
        result.update({
            "detected": True,
            "first_post_change_detection_week": first.week,
            "detection_delay_weeks": first.week - claim.change_week,
            "reason": "matching directional alert in post-change eligibility window",
        })
    else:
        result.update({
            "detected": False,
            "right_censored": True, "censoring_week": observed_end,
            "reason": "no matching detection through observed follow-up",
        })
    return result


def evaluate_claims(
    run: MonitoringRun, claims: Iterable[MonitoringClaim],
) -> list[dict[str, Any]]:
    records = [evaluate_claim(run, claim) for claim in claims]
    ids = [record["claim_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("monitoring claim IDs must be unique")
    return records


def summarize_claim_evaluations(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarize independent runs with explicit Monte Carlo uncertainty.

    Incomplete evaluations are counted and excluded explicitly. Delay is
    conditional on detection; censored nondetections are counted separately.
    Do not interpret this summary as valid for correlated or tuned-on seeds.
    """
    rows = list(records)
    if not rows:
        return {"status": "NOT_RUN", "n_runs": 0}
    definitions = [
        {key: row[key] for key in MonitoringClaim.__dataclass_fields__}
        for row in rows
    ]
    if any(definition != definitions[0] for definition in definitions[1:]):
        raise ValueError("cannot combine different monitoring claims or horizons")
    if len({row["regime"] for row in rows}) != 1:
        raise ValueError("cannot pool different monitoring regimes")
    if len({row["seed"] for row in rows}) != len(rows):
        raise ValueError("independent validation requires unique seeds")
    complete = [row for row in rows if row["evaluation_status"] == "PASS"]
    binary_key = "false_alert_any" if rows[0]["null_claim"] else "detected"
    valid = [row for row in complete if row.get(binary_key) is not None]
    values = np.asarray([row[binary_key] for row in valid], dtype=float)
    n = len(values)
    p = float(values.mean()) if n else None
    # Exact one-sided bound is informative even for zero observed alerts.
    from scipy.stats import beta
    upper = (
        float(beta.ppf(0.95, values.sum() + 1, n - values.sum()))
        if n and values.sum() < n else (1.0 if n else None)
    )
    exact_interval = None
    if n:
        successes = int(values.sum())
        exact_interval = [
            float(beta.ppf(0.025, successes, n - successes + 1))
            if successes else 0.0,
            float(beta.ppf(0.975, successes + 1, n - successes))
            if successes < n else 1.0,
        ]
    delays = [row["detection_delay_weeks"] for row in valid
              if row.get("detected") is True]
    return {
        "claim": definitions[0],
        "status": "PASS" if n == len(rows) else "INCONCLUSIVE",
        "status_scope": "evaluation_completeness_not_monitor_efficacy",
        "n_runs": len(rows), "n_complete_runs": n,
        "n_excluded_incomplete": len(rows) - n,
        "probability_metric": binary_key,
        "estimated_probability": p,
        "monte_carlo_standard_error": (
            float(np.sqrt(p * (1 - p) / n)) if p is not None and n else None
        ),
        "one_sided_95pct_exact_binomial_upper_bound": upper,
        "probability_exact_95pct_interval": exact_interval,
        "n_nondetections_censored": sum(row["right_censored"] for row in valid),
        "mean_delay_weeks_conditional_on_detection": (
            float(np.mean(delays)) if delays else None
        ),
        "independent_validation_seeds_required": True,
    }
