"""Detection event and monitoring run dataclasses.

These are the dashboard's output types. The monitoring tiers emit
``DetectionEvent`` instances when their rules fire, and the
``MonitoringHarness`` collects them into a ``MonitoringRun`` alongside
the full weekly dashboard history and (for validation purposes) the
ground truth trajectory from the simulation's counterfactual branch.

The dashboard-vs-ground-truth separation is maintained in the type
system: the tiers only see ``WeeklyDashboardRow`` objects from the
aggregator, and the ground truth trajectory is stored alongside but
never passed into the tier ``update()`` methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class DetectionEvent:
    """A single detection fired by one of the monitoring tiers.

    Attributes
    ----------
    tier : int
        Tier number (1=Shewhart, 2=CUSUM, 3=CITS, 4=Model drift).
    metric : str
        Name of the metric that triggered detection, e.g.
        ``"check_in_adherence"`` (Tier 1), ``"turnover_rate"``
        (Tier 2), ``"program_effect"`` (Tier 3), ``"realized_auc"``
        (Tier 4).
    week : int
        Simulation week at which the detection fired.
    severity : str
        Qualitative severity: ``"info"``, ``"warning"``, or
        ``"critical"``.
    unit_id : Optional[int]
        Manager ID if the detection is manager-level (Tier 1 can be
        per-manager); None for population-level detections.
    value : float
        The statistic value that triggered the rule (e.g., the
        CUSUM value, the AUC estimate, the point estimate from the
        segmented regression).
    direction : str
        ``"up"``, ``"down"``, or ``"level"`` — the direction of the
        deviation.
    rule : str
        Short identifier of the detection rule that fired, e.g.
        ``"WE1_3sigma"``, ``"WE2_8consecutive"``, ``"cusum_upper"``,
        ``"auc_below_threshold"``.
    """

    tier: int
    metric: str
    week: int
    severity: str
    value: float
    direction: str
    rule: str
    unit_id: Optional[int] = None

    def __post_init__(self) -> None:
        if self.tier not in (1, 2, 3, 4):
            raise ValueError(f"tier must be 1-4, got {self.tier}")
        if self.severity not in ("info", "warning", "critical"):
            raise ValueError(
                f"severity must be info|warning|critical, "
                f"got {self.severity!r}"
            )
        if self.direction not in ("up", "down", "level"):
            raise ValueError(
                f"direction must be up|down|level, "
                f"got {self.direction!r}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable dict representation."""
        return {
            "tier": self.tier,
            "metric": self.metric,
            "week": self.week,
            "severity": self.severity,
            "value": float(self.value),
            "direction": self.direction,
            "rule": self.rule,
            "unit_id": self.unit_id,
        }


@dataclass
class Tier3Estimate:
    """One quarterly Tier 3 CITS estimate.

    Captures the segmented-regression point estimate and 95% CI for
    the program effect at a given refit week.
    """

    week: int
    effect_estimate: float
    ci_lower: float
    ci_upper: float
    p_value: float
    n_observations: int
    mode: str  # "cits_with_cf" or "its_only"

    def is_significant(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha

    def to_dict(self) -> Dict[str, Any]:
        return {
            "week": self.week,
            "effect_estimate": float(self.effect_estimate),
            "ci_lower": float(self.ci_lower),
            "ci_upper": float(self.ci_upper),
            "p_value": float(self.p_value),
            "n_observations": self.n_observations,
            "mode": self.mode,
        }


@dataclass
class MonitoringRun:
    """Complete output of one MonitoringHarness run.

    Contains the full weekly dashboard history, all detection events
    from all tiers, quarterly Tier 3 estimates, and — for validation
    purposes only — the ground truth factual and counterfactual
    turnover trajectories from the simulation. The ground truth
    fields are used by the analysis scripts to compare "what the
    dashboard saw" against "what the simulation knew." The tiers
    themselves never see the ground truth.
    """

    regime: str
    seed: int
    n_weeks: int

    # Weekly dashboard history (list of WeeklyDashboardRow dicts)
    weekly_history: List[Dict[str, Any]] = field(default_factory=list)

    # All detection events collected from the four tiers
    detection_events: List[DetectionEvent] = field(default_factory=list)

    # Quarterly Tier 3 estimates
    tier3_estimates: List[Tier3Estimate] = field(default_factory=list)

    # Ground truth (from the sim's branched result, for validation)
    # These are NOT visible to the tiers — they are passed through
    # separately for the analysis layer.
    ground_truth_factual_departures: Optional[np.ndarray] = None
    ground_truth_counterfactual_departures: Optional[np.ndarray] = None
    ground_truth_factual_retention: Optional[np.ndarray] = None
    ground_truth_counterfactual_retention: Optional[np.ndarray] = None

    # For offline Tier 4: arrays of (week, scores, true_labels) triples
    # Populated by the harness from the engine's prediction history.
    tier4_prediction_log: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_event(self, event: DetectionEvent) -> None:
        self.detection_events.append(event)

    def events_by_tier(self, tier: int) -> List[DetectionEvent]:
        return [e for e in self.detection_events if e.tier == tier]

    def first_detection(
        self, tier: int, metric: Optional[str] = None,
    ) -> Optional[DetectionEvent]:
        """Return the earliest detection for a given tier (and metric)."""
        events = [
            e for e in self.detection_events
            if e.tier == tier
            and (metric is None or e.metric == metric)
        ]
        if not events:
            return None
        return min(events, key=lambda e: e.week)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable dict (excluding numpy arrays)."""
        return {
            "regime": self.regime,
            "seed": self.seed,
            "n_weeks": self.n_weeks,
            "weekly_history": self.weekly_history,
            "detection_events": [
                e.to_dict() for e in self.detection_events
            ],
            "tier3_estimates": [
                e.to_dict() for e in self.tier3_estimates
            ],
            "metadata": self.metadata,
        }
