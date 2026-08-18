"""Tier 4: Offline model calibration and AUC drift monitoring.

Per the task plan (open question #3), Tier 4 is computed **offline**
from saved run artifacts rather than instrumented live during
``predict()``. This keeps the scenario's predict() method clean and
avoids model-internals leakage.

The input is a list of (week, scores, true_labels) triples captured
from the engine's prediction history. The function computes rolling
window statistics and emits DetectionEvents when the configured
drift thresholds are crossed.

Statistics (following Cox 1958 for calibration and standard
convention for AUC and Brier score):

- **Realized AUC**: sklearn.metrics.roc_auc_score over a rolling
  window of the last N weeks of score-outcome pairs
- **Calibration slope**: the slope coefficient from regressing the
  observed binary outcome on the logit of the predicted probability
- **Calibration intercept**: mean observed - mean predicted
- **Brier score**: mean squared error of (score - label)

Detection rules (from task spec):

- AUC below 0.65 for 4 consecutive weeks → critical
- Calibration slope deviates by more than 20% from the initial
  fitted value → warning

The function is stateless — it takes the full prediction log and
returns the full list of events and rolling statistics. The
``MonitoringHarness`` calls it at the end of a run, after all
weekly data has been collected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

from healthcare_sim_sdk.scenarios.nurse_retention.monitoring.events import (
    DetectionEvent,
)

# Task-specified thresholds
AUC_DETECTION_THRESHOLD = 0.65
AUC_CONSECUTIVE_WEEKS = 4
CALIBRATION_SLOPE_DEVIATION = 0.20

# Rolling window length in weeks
ROLLING_WINDOW_WEEKS = 12


@dataclass
class Tier4RollingStats:
    """One week's worth of rolling Tier 4 statistics."""

    week: int
    n_predictions: int
    realized_auc: Optional[float]
    calibration_slope: Optional[float]
    calibration_intercept: Optional[float]
    brier_score: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "week": self.week,
            "n_predictions": self.n_predictions,
            "realized_auc": (
                float(self.realized_auc)
                if self.realized_auc is not None
                else None
            ),
            "calibration_slope": (
                float(self.calibration_slope)
                if self.calibration_slope is not None
                else None
            ),
            "calibration_intercept": (
                float(self.calibration_intercept)
                if self.calibration_intercept is not None
                else None
            ),
            "brier_score": (
                float(self.brier_score)
                if self.brier_score is not None
                else None
            ),
        }


@dataclass
class Tier4Result:
    """Full Tier 4 analysis output."""

    rolling_stats: List[Tier4RollingStats] = field(default_factory=list)
    detection_events: List[DetectionEvent] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rolling_stats": [s.to_dict() for s in self.rolling_stats],
            "detection_events": [
                e.to_dict() for e in self.detection_events
            ],
        }


def _calibration_slope_intercept(
    predicted: np.ndarray,
    observed: np.ndarray,
) -> (Optional[float], Optional[float]):
    """Compute calibration slope and intercept via logistic regression.

    Cox (1958): regress the observed binary outcome on the predicted
    probability (or its logit). A well-calibrated model has slope 1
    and intercept 0.

    Returns (None, None) if the input is degenerate (too few
    distinct predictions or all observations are the same class).
    """
    if len(predicted) < 10:
        return (None, None)
    if len(np.unique(observed)) < 2:
        return (None, None)

    # Clip predicted probabilities away from 0 and 1 before logit
    eps = 1e-6
    p_clipped = np.clip(predicted, eps, 1.0 - eps)
    logit_p = np.log(p_clipped / (1.0 - p_clipped))

    # Simple linear regression: observed ~ intercept + slope*logit_p
    # Use numpy polyfit for robustness
    try:
        slope, intercept = np.polyfit(logit_p, observed, 1)
    except np.linalg.LinAlgError:
        return (None, None)

    return (float(slope), float(intercept))


def _safe_auc(
    predicted: np.ndarray, observed: np.ndarray,
) -> Optional[float]:
    """Compute AUC, returning None for degenerate cases."""
    if len(predicted) < 10:
        return None
    if len(np.unique(observed)) < 2:
        return None
    try:
        return float(roc_auc_score(observed, predicted))
    except ValueError:
        return None


def _safe_brier(
    predicted: np.ndarray, observed: np.ndarray,
) -> Optional[float]:
    if len(predicted) == 0:
        return None
    try:
        return float(brier_score_loss(observed, predicted))
    except ValueError:
        return None


def analyze_tier4(
    prediction_log: List[Dict[str, Any]],
    rolling_window_weeks: int = ROLLING_WINDOW_WEEKS,
    auc_threshold: float = AUC_DETECTION_THRESHOLD,
    auc_consecutive_weeks: int = AUC_CONSECUTIVE_WEEKS,
    slope_deviation: float = CALIBRATION_SLOPE_DEVIATION,
) -> Tier4Result:
    """Analyze a prediction log and return drift statistics + events.

    Parameters
    ----------
    prediction_log : list of dict
        Each entry is a dict with keys:
        * ``"week"``: int, simulation week of the prediction
        * ``"scores"``: np.ndarray, model score per active nurse
        * ``"true_labels"``: np.ndarray, binary label for whether
          the nurse departed within the prediction horizon
        The harness populates this from the engine's prediction
        metadata during a run.
    rolling_window_weeks : int
        Window length for rolling statistics (default 12).
    auc_threshold : float
        AUC below this value for several consecutive weeks
        constitutes a drift detection (default 0.65).
    auc_consecutive_weeks : int
        Number of consecutive weeks below the AUC threshold
        required to fire a detection (default 4).
    slope_deviation : float
        Fractional deviation of calibration slope from its initial
        value that triggers a warning (default 0.20 = 20%).

    Returns
    -------
    Tier4Result
        Full rolling statistics and all detection events.
    """
    result = Tier4Result()
    if not prediction_log:
        return result

    # Sort by week
    sorted_log = sorted(prediction_log, key=lambda r: r["week"])
    n_entries = len(sorted_log)

    # Baseline calibration slope (from the first fit we can compute)
    initial_slope: Optional[float] = None

    # Track consecutive weeks where AUC is below threshold
    consecutive_below = 0
    below_event_emitted = False

    for i in range(n_entries):
        week = int(sorted_log[i]["week"])

        # Build the rolling window: entries within [week - W, week]
        window_entries = [
            entry for entry in sorted_log[: i + 1]
            if week - entry["week"] < rolling_window_weeks
        ]

        # Concatenate scores and labels across the window
        scores_list = []
        labels_list = []
        for entry in window_entries:
            s = np.asarray(entry["scores"])
            y = np.asarray(entry["true_labels"])
            # Keep only scores with a real prediction (scored > 0)
            mask = s > 0
            scores_list.append(s[mask])
            labels_list.append(y[mask])

        if not scores_list:
            result.rolling_stats.append(
                Tier4RollingStats(
                    week=week,
                    n_predictions=0,
                    realized_auc=None,
                    calibration_slope=None,
                    calibration_intercept=None,
                    brier_score=None,
                )
            )
            continue

        scores = np.concatenate(scores_list)
        labels = np.concatenate(labels_list)

        auc = _safe_auc(scores, labels)
        brier = _safe_brier(scores, labels)
        slope, intercept = _calibration_slope_intercept(scores, labels)

        result.rolling_stats.append(
            Tier4RollingStats(
                week=week,
                n_predictions=len(scores),
                realized_auc=auc,
                calibration_slope=slope,
                calibration_intercept=intercept,
                brier_score=brier,
            )
        )

        # Lock in the initial slope on the first successful fit
        if initial_slope is None and slope is not None:
            initial_slope = slope

        # Detection: AUC below threshold for N consecutive weeks
        if auc is not None and auc < auc_threshold:
            consecutive_below += 1
        else:
            consecutive_below = 0
            below_event_emitted = False

        if (
            consecutive_below >= auc_consecutive_weeks
            and not below_event_emitted
            and auc is not None
        ):
            result.detection_events.append(
                DetectionEvent(
                    tier=4,
                    metric="realized_auc",
                    week=week,
                    severity="critical",
                    value=auc,
                    direction="down",
                    rule=(
                        f"auc_below_{auc_threshold}_"
                        f"{auc_consecutive_weeks}_consecutive"
                    ),
                )
            )
            below_event_emitted = True

        # Detection: calibration slope deviates by more than threshold
        if (
            initial_slope is not None
            and slope is not None
            and abs(initial_slope) > 1e-6
        ):
            deviation = abs(slope - initial_slope) / abs(initial_slope)
            if deviation > slope_deviation:
                # Only emit once per run (avoid spamming every week)
                already_emitted = any(
                    e.tier == 4
                    and e.metric == "calibration_slope"
                    for e in result.detection_events
                )
                if not already_emitted:
                    result.detection_events.append(
                        DetectionEvent(
                            tier=4,
                            metric="calibration_slope",
                            week=week,
                            severity="warning",
                            value=slope,
                            direction=(
                                "up" if slope > initial_slope
                                else "down"
                            ),
                            rule=(
                                f"calibration_slope_"
                                f"deviation_over_{slope_deviation}"
                            ),
                        )
                    )

    return result
