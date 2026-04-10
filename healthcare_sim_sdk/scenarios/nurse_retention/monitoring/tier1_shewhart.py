"""Tier 1: Shewhart XmR control chart for leading indicators.

This tier monitors operational leading indicators like check-in
adherence, new-hire contact rate, and tool adoption. It uses the
standard Shewhart Individuals and Moving-Range (XmR) control chart
with Western Electric rules for detection.

The expected detection profile:

- **Capacity collapse** (Regime D): adherence drops to ~33% immediately
  when capacity is cut from 6 to 2. One point outside the lower 3σ
  limit triggers WE1 within 1-2 weeks.
- **Partial adoption** (Regime F): per-manager chart for non-adopting
  managers shows zero check-ins sustained, triggering WE2 (8
  consecutive points below centerline) within ~8 weeks.
- **Null / calibrated / gradual decay**: adherence stays at target, no
  detection. This is the intended silence.

Shewhart XmR basics (see Montgomery 7th ed., Ch. 6):

- Baseline period (default 8 weeks): estimate the process mean
  ``xbar`` and average moving range ``mr_bar``.
- Process standard deviation estimate: ``sigma_hat = mr_bar / 1.128``
  (d2 constant for moving range of 2 consecutive observations).
- Control limits:
  * UCL = xbar + 3 * sigma_hat
  * LCL = xbar - 3 * sigma_hat
  * centerline = xbar
- After the baseline, each new observation is classified against the
  limits. Two Western Electric rules are implemented:
  * **WE1**: one point outside the 3σ limits
  * **WE2**: 8 consecutive points on one side of the centerline
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from healthcare_sim_sdk.scenarios.nurse_retention.monitoring.events import (
    DetectionEvent,
)

# d2 constant for moving range of 2 consecutive observations
# (Montgomery, Table VI)
D2_N2 = 1.128


@dataclass
class ShewhartChart:
    """A Shewhart XmR control chart for a single metric.

    Stateful: call ``update(week, value)`` once per observation to
    advance the chart. Returns a list of ``DetectionEvent`` for
    detections that fired on this observation (may be empty).

    Parameters
    ----------
    metric : str
        Name of the metric being monitored. Used in emitted events.
    baseline_weeks : int
        Number of initial observations to use for control limit
        estimation. Control limits are frozen after this many points.
    we_rules : tuple of str
        Which Western Electric rules to enable. Supported:
        ``"WE1"`` (single point outside 3σ) and ``"WE2"`` (8 consecutive
        points on one side of centerline).
    unit_id : Optional[int]
        Manager ID if this chart monitors a single manager; None for
        population-level charts.
    """

    metric: str
    baseline_weeks: int = 8
    we_rules: tuple = ("WE1", "WE2")
    unit_id: Optional[int] = None

    # Internal state (populated as update() is called)
    history: List[float] = field(default_factory=list)
    xbar: Optional[float] = None
    sigma_hat: Optional[float] = None
    ucl: Optional[float] = None
    lcl: Optional[float] = None
    centerline: Optional[float] = None
    _above_count: int = 0
    _below_count: int = 0

    def update(self, week: int, value: float) -> List[DetectionEvent]:
        """Add a new observation and return any detections that fire.

        Before the baseline is complete, the chart accumulates
        observations but emits no detections.
        """
        self.history.append(float(value))
        events: List[DetectionEvent] = []

        # Baseline period: accumulate observations, compute limits
        # when we hit baseline_weeks points
        if len(self.history) < self.baseline_weeks:
            return events
        if len(self.history) == self.baseline_weeks:
            self._compute_limits()
            return events

        # Post-baseline: check rules
        if self.xbar is None or self.sigma_hat is None:
            return events

        # WE1: single point outside 3-sigma limits
        if "WE1" in self.we_rules:
            if value > self.ucl:
                events.append(
                    DetectionEvent(
                        tier=1,
                        metric=self.metric,
                        week=week,
                        severity="critical",
                        value=value,
                        direction="up",
                        rule="WE1_3sigma_upper",
                        unit_id=self.unit_id,
                    )
                )
            elif value < self.lcl:
                events.append(
                    DetectionEvent(
                        tier=1,
                        metric=self.metric,
                        week=week,
                        severity="critical",
                        value=value,
                        direction="down",
                        rule="WE1_3sigma_lower",
                        unit_id=self.unit_id,
                    )
                )

        # WE2: 8 consecutive points on one side of centerline
        if "WE2" in self.we_rules:
            if value > self.centerline:
                self._above_count += 1
                self._below_count = 0
            elif value < self.centerline:
                self._below_count += 1
                self._above_count = 0
            else:  # exactly at centerline
                self._above_count = 0
                self._below_count = 0

            if self._above_count >= 8:
                events.append(
                    DetectionEvent(
                        tier=1,
                        metric=self.metric,
                        week=week,
                        severity="warning",
                        value=value,
                        direction="up",
                        rule="WE2_8consecutive_above",
                        unit_id=self.unit_id,
                    )
                )
                # Reset after firing to avoid spamming every subsequent
                # week
                self._above_count = 0
            elif self._below_count >= 8:
                events.append(
                    DetectionEvent(
                        tier=1,
                        metric=self.metric,
                        week=week,
                        severity="warning",
                        value=value,
                        direction="down",
                        rule="WE2_8consecutive_below",
                        unit_id=self.unit_id,
                    )
                )
                self._below_count = 0

        return events

    def _compute_limits(self) -> None:
        """Compute control limits from the baseline observations."""
        baseline = np.array(self.history[: self.baseline_weeks])
        self.xbar = float(np.mean(baseline))
        self.centerline = self.xbar

        # Moving range of adjacent observations
        if len(baseline) < 2:
            self.sigma_hat = 0.0
        else:
            moving_ranges = np.abs(np.diff(baseline))
            mr_bar = float(np.mean(moving_ranges))
            self.sigma_hat = mr_bar / D2_N2

        self.ucl = self.xbar + 3.0 * self.sigma_hat
        self.lcl = self.xbar - 3.0 * self.sigma_hat


class Tier1Shewhart:
    """Tier 1: Shewhart XmR charts for leading indicators.

    Maintains one ``ShewhartChart`` per monitored metric. Add metrics
    via ``add_metric()`` before calling ``update()``.

    Example
    -------
    >>> tier1 = Tier1Shewhart()
    >>> tier1.add_metric("check_in_adherence", baseline_weeks=8)
    >>> for week in range(52):
    ...     row = get_row(week)  # doctest: +SKIP
    ...     events = tier1.update(week, row)  # doctest: +SKIP
    """

    def __init__(self) -> None:
        self.charts: dict = {}

    def add_metric(
        self,
        metric: str,
        baseline_weeks: int = 8,
        unit_id: Optional[int] = None,
    ) -> None:
        """Add a new metric to monitor.

        The chart key is ``(metric, unit_id)`` so that per-manager
        charts don't collide with the population-level chart.
        """
        key = (metric, unit_id)
        self.charts[key] = ShewhartChart(
            metric=metric,
            baseline_weeks=baseline_weeks,
            unit_id=unit_id,
        )

    def update(
        self,
        week: int,
        metric_values: dict,
    ) -> List[DetectionEvent]:
        """Update all charts with this week's values.

        ``metric_values`` is a dict mapping ``(metric_name, unit_id)``
        keys to float values. Unit_id is None for population-level
        metrics. Only charts that have been added via ``add_metric()``
        are updated; extra entries are ignored.
        """
        events: List[DetectionEvent] = []
        for key, chart in self.charts.items():
            if key not in metric_values:
                continue
            value = metric_values[key]
            events.extend(chart.update(week, value))
        return events
