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
- **Calibration intercept**: intercept in the logistic recalibration fit
- **Brier score**: mean squared error of (score - label)

Detection rules (from task spec):

- AUC below 0.65 for 4 consecutive evaluations (8 weeks at a two-week cadence) → critical
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
AUC_CONSECUTIVE_EVALUATIONS = 4
AUC_CONSECUTIVE_WEEKS = AUC_CONSECUTIVE_EVALUATIONS  # legacy import alias
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
                float(self.realized_auc) if self.realized_auc is not None else None
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
                float(self.brier_score) if self.brier_score is not None else None
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
            "detection_events": [e.to_dict() for e in self.detection_events],
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

    if np.std(logit_p) < 1e-8:
        return (None, None)
    from scipy.optimize import minimize
    from scipy.special import expit

    design = np.column_stack([np.ones(len(logit_p)), logit_p])

    def objective(beta):
        eta = design @ beta
        loss = np.mean(np.logaddexp(0, eta) - observed * eta)
        gradient = design.T @ (expit(eta) - observed) / len(observed)
        return loss, gradient

    fit = minimize(objective, [0.0, 1.0], jac=True, method="BFGS")
    if not fit.success or not np.all(np.isfinite(fit.x)):
        return (None, None)
    return (float(fit.x[1]), float(fit.x[0]))


def _safe_auc(
    predicted: np.ndarray,
    observed: np.ndarray,
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
    predicted: np.ndarray,
    observed: np.ndarray,
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
    auc_consecutive_evaluations: int = AUC_CONSECUTIVE_EVALUATIONS,
    slope_deviation: float = CALIBRATION_SLOPE_DEVIATION,
    baseline_weeks: int = 12,
    min_baseline_events: int = 100,
    auc_consecutive_weeks: Optional[int] = None,
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
        The harness matches saved scores to mature future factual
        departures, never to synthetic fit labels in prediction metadata.
    rolling_window_weeks : int
        Window length for rolling statistics (default 12).
    auc_threshold : float
        AUC below this value for several consecutive weeks
        constitutes a drift detection (default 0.65).
    auc_consecutive_evaluations : int
        Number of consecutive evaluations below the AUC threshold
        required to fire a detection (default 4, i.e. 8 weeks at two-week cadence).
    slope_deviation : float
        Fractional deviation of calibration slope from a pooled baseline with at least min_baseline_events events that triggers a warning (default 0.20 = 20%).

    Returns
    -------
    Tier4Result
        Full rolling statistics and all detection events.
    """
    if auc_consecutive_weeks is not None:
        import warnings

        warnings.warn(
            "auc_consecutive_weeks counts evaluations; use "
            "auc_consecutive_evaluations instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        auc_consecutive_evaluations = auc_consecutive_weeks
    result = Tier4Result()
    if not prediction_log:
        return result

    # Sort by week
    sorted_log = sorted(prediction_log, key=lambda r: r["week"])
    n_entries = len(sorted_log)
    # Use at least the configured baseline duration and continue pooling
    # until enough events are observed. If that never occurs, report stats
    # without claiming a drift comparison against an unstable reference.
    earliest_end = int(sorted_log[0]["week"]) + baseline_weeks
    baseline_end = float("inf")
    baseline_scores_list, baseline_labels_list = [], []
    for entry in sorted_log:
        baseline_scores_list.append(np.asarray(entry["scores"]))
        baseline_labels_list.append(np.asarray(entry["true_labels"]))
        baseline_labels = np.concatenate(baseline_labels_list)
        enough = (
            min(baseline_labels.sum(), len(baseline_labels) - baseline_labels.sum())
            >= min_baseline_events
        )
        if entry["week"] >= earliest_end and enough:
            baseline_end = int(entry["week"]) + 1
            break
    baseline_scores = np.concatenate(baseline_scores_list)
    baseline_labels = np.concatenate(baseline_labels_list)
    baseline_mask = np.isfinite(baseline_scores)
    baseline_scores, baseline_labels = (
        baseline_scores[baseline_mask],
        baseline_labels[baseline_mask],
    )
    baseline_ready = np.isfinite(baseline_end)

    # Baseline calibration slope (from the first fit we can compute)
    initial_slope = (
        _calibration_slope_intercept(baseline_scores, baseline_labels)[0]
        if baseline_ready
        else None
    )

    # Track consecutive weeks where AUC is below threshold
    consecutive_below = 0
    below_event_emitted = False

    for i in range(n_entries):
        week = int(sorted_log[i]["week"])

        # Build the rolling window: entries within [week - W, week]
        window_entries = [
            entry
            for entry in sorted_log[: i + 1]
            if week - entry["week"] < rolling_window_weeks
        ]

        # Concatenate scores and labels across the window
        scores_list = []
        labels_list = []
        for entry in window_entries:
            s = np.asarray(entry["scores"])
            y = np.asarray(entry["true_labels"])
            # Keep only scores with a real prediction (scored > 0)
            mask = np.isfinite(s)
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

        # Freeze a pooled baseline only after its time window has elapsed.
        # If it contains too few events, report statistics without drift claims.
        if week < baseline_end or not baseline_ready:
            continue

        # Detection: AUC below threshold for N consecutive weeks
        if auc is not None and auc < auc_threshold:
            consecutive_below += 1
        else:
            consecutive_below = 0
            below_event_emitted = False

        if (
            consecutive_below >= auc_consecutive_evaluations
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
                        f"{auc_consecutive_evaluations}_consecutive_evaluations"
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
                    e.tier == 4 and e.metric == "calibration_slope"
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
                            direction=("up" if slope > initial_slope else "down"),
                            rule=(
                                f"calibration_slope_deviation_over_{slope_deviation}"
                            ),
                        )
                    )

    return result
