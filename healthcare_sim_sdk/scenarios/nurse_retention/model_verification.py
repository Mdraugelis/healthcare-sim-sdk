"""Explicitly scoped discrimination checks for the deployed scoring path.

Synthetic labels used by the score generator and realized future departures
answer different questions. These records carry their label provenance so a
passing synthetic-label check cannot certify clinical outcome discrimination.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any, cast

import numpy as np

from scipy.stats import rankdata


def score_diagnostics(
    labels: np.ndarray,
    scores: np.ndarray,
    target_auc: float,
    *,
    tolerance: float = 0.05,
    min_positives: int = 20,
    min_negatives: int = 20,
    label_provenance: str,
) -> dict[str, Any]:
    """Report actual scores, sample adequacy, and an explicit AUC gate.

    The tolerance is an absolute engineering acceptance criterion, not a
    statistical confidence interval. Too few positive/negative outcomes yield
    INCONCLUSIVE even when the observed AUC happens to match its target.
    """
    y, s = np.asarray(labels), np.asarray(scores)
    if y.ndim != 1 or s.shape != y.shape:
        raise ValueError("labels and scores must be aligned one-dimensional arrays")
    if not np.isfinite(target_auc) or not 0 <= target_auc <= 1:
        raise ValueError("target_auc must be finite and in [0, 1]")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    for name, count in (("min_positives", min_positives),
                        ("min_negatives", min_negatives)):
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(f"{name} must be a positive integer")
    valid = bool(
        np.isfinite(s).all() and np.isin(y, [0, 1]).all()
        and ((s >= 0) & (s <= 1)).all()
    )
    n_positive, n_negative = int((y == 1).sum()), int((y == 0).sum())
    auc = (
        float((rankdata(s)[y == 1].sum() - n_positive * (n_positive + 1) / 2)
              / (n_positive * n_negative))
        if valid and n_positive and n_negative else None
    )
    if not valid:
        status, reason = "FAIL", "invalid scores or binary labels"
    elif n_positive < min_positives or n_negative < min_negatives:
        status, reason = "INCONCLUSIVE", "insufficient outcome counts"
    elif abs(cast(float, auc) - target_auc) > tolerance:
        status, reason = "FAIL", "deployed score AUC misses configured tolerance"
    else:
        status, reason = "PASS", "deployed score AUC is within configured tolerance"
    return {
        "status": status, "reason": reason,
        "label_provenance": label_provenance,
        "auc_method": "rank_statistic_with_average_ties",
        "n_predictions": len(y), "n_positive": n_positive,
        "n_negative": n_negative, "auc": auc, "target_auc": float(target_auc),
        "auc_absolute_tolerance": float(tolerance),
        "min_positives": min_positives, "min_negatives": min_negatives,
        "brier_score": float(np.mean((s - y) ** 2)) if valid and len(y) else None,
        "tolerance_is_confidence_interval": False,
    }


def verify_frozen_scoring_path(
    scenario: Any, *, validation_seed: int, n_entities: int = 5000, t: int = 0,
) -> dict[str, Any]:
    """Evaluate a frozen fitted scorer on a separately generated population.

    Uses the same scenario ``predict`` method and independent partitioned
    population/prediction RNG streams. Does not refit, consume the reviewed
    run's random numbers, or equate synthetic labels with realized outcomes.
    The initial independent population is not a certification of every later
    time-varying case in the original run.
    """
    if not scenario._classifier_fitted:
        return {"status": "NOT_RUN", "reason": "scorer has not been fitted"}
    if validation_seed == scenario.seed:
        raise ValueError("validation_seed must differ from the simulation seed")
    # Import here to keep the scenario's diagnostic import acyclic.
    from .scenario import NurseRetentionScenario
    from .time_varying import resolve

    target_auc = resolve(scenario.config.model_auc, t)
    if abs(target_auc - scenario._fitted_auc) > 1e-9:
        return {
            "status": "NOT_RUN",
            "reason": "frozen scorer is incompatible with target at requested week",
        }
    independent = NurseRetentionScenario(
        config=replace(scenario.config, n_nurses=n_entities), seed=validation_seed,
    )
    independent._classifier = copy.deepcopy(scenario._classifier)
    independent._classifier_fitted = True
    independent._fitted_auc = scenario._fitted_auc
    independent._fit_report = copy.deepcopy(scenario._fit_report)
    population = independent.create_population(n_entities)
    prediction = independent.predict(population, t)
    diagnostic = prediction.metadata["model_performance"]["deployed_score_path"]
    return {
        **diagnostic, "validation_seed": validation_seed,
        "population_scope": "independent_initial_population",
        "frozen_model": True, "refit": False, "week": t,
        "actual_future_outcomes_status": "NOT_RUN",
    }
