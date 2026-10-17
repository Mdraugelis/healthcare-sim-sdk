"""Tier 2: CUSUM charts for lagging outcomes (turnover rate).

Based on Page (1954). Two-sided CUSUM with reference value k and
decision interval h.

This is a generic chart. A no-effect AI regime is not necessarily stationary:
a closed cohort loses high-risk nurses first, so raw turnover declines. The
monitoring harness uses prospective observed-minus-predicted departure residuals
and a decision interval calibrated by simulation (currently h=96 at n=1000).
The defaults below are textbook examples, not deployment operating guarantees.

CUSUM basics (Page 1954, Montgomery 7th ed., Ch. 9):

Let the in-control mean be ``mu0`` (estimated from the baseline) and
the process standard deviation be ``sigma`` (also estimated from the
baseline). The reference value ``k`` is half the shift we want to
detect quickly, typically ``k = 0.5 * sigma``. The decision interval
``h`` is typically ``4 * sigma`` to 5 * sigma; we use 4 per the task
spec.

At each new observation ``x_i``:

- Upper CUSUM: ``C_upper_i = max(0, C_upper_{i-1} + (x_i - mu0) - k)``
- Lower CUSUM: ``C_lower_i = max(0, C_lower_{i-1} - (x_i - mu0) - k)``

A detection fires when either exceeds ``h``. After detection, only the arm that fired is reset to zero.

Note on direction semantics for turnover rate:
- **Upper CUSUM** detects an INCREASE in turnover (bad — program
  is losing effect or capacity collapsed)
- **Lower CUSUM** detects a DECREASE in turnover (good — program
  is working better than baseline). Usually uninteresting for
  monitoring, but we track both for completeness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from healthcare_sim_sdk.scenarios.nurse_retention.monitoring.events import (
    DetectionEvent,
)


@dataclass
class Tier2CUSUM:
    """Page (1954) two-sided CUSUM chart.

    Parameters
    ----------
    metric : str
        Name of the metric being monitored (emitted in events).
    baseline_weeks : int
        Observations used to estimate ``mu0`` and ``sigma``.
    k_multiplier : float
        Reference value as a fraction of sigma (default 0.5).
    h_multiplier : float
        Decision interval as a fraction of sigma (default 4.0).
    reset_on_detection : bool
        Whether to reset the firing CUSUM arm after a detection (default
        True). Standard practice.

    Internal state
    --------------
    history : list of float
        All observations seen so far.
    mu0 : float or None
        In-control mean estimated from baseline.
    sigma : float or None
        Process standard deviation estimated from baseline.
    k : float or None
        Reference value (frozen after baseline).
    h : float or None
        Decision interval (frozen after baseline).
    c_upper, c_lower : float
        Current CUSUM values.
    """

    metric: str
    baseline_weeks: int = 8
    k_multiplier: float = 0.5
    h_multiplier: float = 4.0
    reset_on_detection: bool = True

    history: List[float] = field(default_factory=list)
    mu0: Optional[float] = None
    sigma: Optional[float] = None
    k: Optional[float] = None
    h: Optional[float] = None
    c_upper: float = 0.0
    c_lower: float = 0.0
    upper_trajectory: List[float] = field(default_factory=list)
    lower_trajectory: List[float] = field(default_factory=list)

    def update(self, week: int, value: float) -> List[DetectionEvent]:
        """Add a new observation and return any detections."""
        self.history.append(float(value))
        events: List[DetectionEvent] = []

        # Baseline period: accumulate observations
        if len(self.history) < self.baseline_weeks:
            self.upper_trajectory.append(0.0)
            self.lower_trajectory.append(0.0)
            return events
        if len(self.history) == self.baseline_weeks:
            self._initialize_from_baseline()
            self.upper_trajectory.append(0.0)
            self.lower_trajectory.append(0.0)
            return events

        # Post-baseline: update CUSUM statistics
        if self.mu0 is None or self.sigma is None:
            return events

        deviation = value - self.mu0

        # Upper CUSUM: accumulates positive deviations above k
        self.c_upper = max(0.0, self.c_upper + deviation - self.k)
        # Lower CUSUM: accumulates negative deviations below -k
        self.c_lower = max(0.0, self.c_lower - deviation - self.k)

        self.upper_trajectory.append(self.c_upper)
        self.lower_trajectory.append(self.c_lower)

        # Detection: either CUSUM exceeds decision interval h
        if self.c_upper > self.h:
            events.append(
                DetectionEvent(
                    tier=2,
                    metric=self.metric,
                    week=week,
                    severity="critical",
                    value=self.c_upper,
                    direction="up",
                    rule="cusum_upper",
                )
            )
            if self.reset_on_detection:
                self.c_upper = 0.0

        if self.c_lower > self.h:
            events.append(
                DetectionEvent(
                    tier=2,
                    metric=self.metric,
                    week=week,
                    severity="info",
                    value=self.c_lower,
                    direction="down",
                    rule="cusum_lower",
                )
            )
            if self.reset_on_detection:
                self.c_lower = 0.0

        return events

    def _initialize_from_baseline(self) -> None:
        """Estimate mu0 and sigma from the baseline, freeze k and h."""
        baseline = np.array(self.history[: self.baseline_weeks])
        self.mu0 = float(np.mean(baseline))
        # Use sample standard deviation with ddof=1
        self.sigma = float(np.std(baseline, ddof=1))
        # If sigma is zero (flat baseline), fall back to a small
        # non-zero value so the CUSUM can still accumulate
        if self.sigma < 1e-9:
            self.sigma = 1e-6
        self.k = self.k_multiplier * self.sigma
        self.h = self.h_multiplier * self.sigma


def calibrate_h(
    residual_series,
    candidates,
    max_alarm_probability=0.10,
    baseline_weeks=8,
    k_multiplier=0.5,
):
    """Choose the smallest decision interval meeting a simulated horizon target.

    Supply independent no-AI calibration runs. Report holdout false-alarm rates
    separately; the observed calibration fraction is not a guarantee about ARL.
    """
    if not residual_series or not candidates:
        raise ValueError("Calibration requires residual series and h candidates")
    rates = {}
    upper_bounds = {}
    for h in sorted(candidates):
        alarms = 0
        for series in residual_series:
            chart = Tier2CUSUM(
                "risk_adjusted_departures", baseline_weeks, k_multiplier, float(h)
            )
            alarm = False
            for week, value in enumerate(series):
                if value is not None:
                    alarm |= bool(chart.update(week, float(value)))
            alarms += alarm
        rates[float(h)] = alarms / len(residual_series)
        # Wilson 95% upper confidence bound: selecting on a noisy point
        # estimate alone underestimates the false-alarm risk.
        n_runs = len(residual_series)
        z = 1.959963984540054
        rate = rates[float(h)]
        upper = (
            rate
            + z * z / (2 * n_runs)
            + z * np.sqrt(rate * (1 - rate) / n_runs + z * z / (4 * n_runs * n_runs))
        ) / (1 + z * z / n_runs)
        upper_bounds[float(h)] = float(upper)
        if upper <= max_alarm_probability:
            return {
                "h": float(h),
                "calibration_rates": rates,
                "calibration_upper_95": upper_bounds,
                "n_calibration_runs": len(residual_series),
                "target_horizon_alarm_probability": max_alarm_probability,
            }
    raise ValueError(f"No h candidate met the target; calibration rates: {rates}")
