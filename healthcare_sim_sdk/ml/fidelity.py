"""Diagnostics on actual synthetic scores, separate from model-fitting reports."""

from typing import Any, Dict

import numpy as np
from scipy.stats import rankdata  # type: ignore[import-untyped]


def auc_fidelity(
    true_labels: np.ndarray,
    scores: np.ndarray,
    target_auc: float,
    *,
    tolerance: float = 0.05,
    min_class_count: int = 20,
    independent_cohort: bool = False,
    scope: str = "observed scores on the supplied cohort",
) -> Dict[str, Any]:
    """Compare realized AUC with an explicit target and minimum class counts.

    PASS means the observed score diagnostic met these criteria. It is not a
    confidence interval, proof of population performance, or independent
    certification unless the caller supplies a genuinely independent cohort.
    Repeated timesteps on the same entities are not independent replications.
    """
    if not np.isfinite(target_auc) or not 0 <= target_auc <= 1:
        raise ValueError("target_auc must be finite and in [0, 1]")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    if (isinstance(min_class_count, bool) or not isinstance(min_class_count, int)
            or min_class_count < 1):
        raise ValueError("min_class_count must be a positive integer")
    labels = np.asarray(true_labels)
    values = np.asarray(scores)
    report: Dict[str, Any] = {
        "status": "INCONCLUSIVE", "target_auc": float(target_auc),
        "achieved_auc": None, "absolute_error": None, "tolerance": float(tolerance),
        "min_class_count": min_class_count, "n": int(labels.size),
        "n_positive": int(np.sum(labels == 1)), "n_negative": int(np.sum(labels == 0)),
        "independent_cohort": bool(independent_cohort), "scope": scope,
        "interpretation": (
            "Observed-score criterion; no population confidence interval is asserted."),
        "auc_method": "exact Mann-Whitney rank statistic with average ranks for ties",
        "detail": "",
    }
    if labels.ndim != 1 or values.ndim != 1 or labels.shape != values.shape:
        report.update(status="FAIL", detail=(
            "Labels and scores must be aligned one-dimensional arrays."))
        return report
    if not np.all(np.isin(labels, [0, 1])):
        report.update(status="FAIL", detail="Labels must be binary.")
        return report
    try:
        valid_scores = bool(
            np.all(np.isfinite(values)) and np.all((values >= 0) & (values <= 1)))
    except TypeError:
        valid_scores = False
    if not valid_scores:
        report.update(status="FAIL", detail=(
            "Scores must be finite numeric probabilities in [0, 1]."))
        return report
    if not report["n_positive"] or not report["n_negative"]:
        report["detail"] = (
            "AUC is undefined without both classes; "
            "no chance-level value was substituted.")
        return report
    # Use an independent exact calculation: the legacy fitting metric samples
    # 200 thresholds, which can miss ranking differences in a narrow score band.
    ranks = rankdata(values, method="average")
    n_positive, n_negative = report["n_positive"], report["n_negative"]
    achieved = float((ranks[labels == 1].sum() - n_positive * (n_positive + 1) / 2)
                     / (n_positive * n_negative))
    report.update(achieved_auc=achieved, absolute_error=abs(achieved - target_auc))
    if min(report["n_positive"], report["n_negative"]) < min_class_count:
        report["detail"] = (
            "Too few examples in one class for the declared diagnostic criterion.")
        return report
    report["status"] = "PASS" if report["absolute_error"] <= tolerance else "FAIL"
    report["detail"] = ("Observed AUC is within the declared absolute tolerance."
                        if report["status"] == "PASS"
                        else "Observed AUC misses the declared absolute tolerance.")
    return report
