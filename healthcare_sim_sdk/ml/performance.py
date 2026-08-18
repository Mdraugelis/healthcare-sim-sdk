"""ML model performance metrics and calibration utilities."""

from typing import Dict, Tuple

import numpy as np


def confusion_matrix_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute classification metrics at a given threshold."""
    labels = (y_pred >= threshold).astype(int)
    tp = np.sum((labels == 1) & (y_true == 1))
    fp = np.sum((labels == 1) & (y_true == 0))
    tn = np.sum((labels == 0) & (y_true == 0))
    fn = np.sum((labels == 0) & (y_true == 1))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    n_total = tp + fp + tn + fn
    accuracy = (tp + tn) / n_total if n_total > 0 else 0.0
    precision_recall_sum = ppv + sensitivity
    f1 = (
        2 * ppv * sensitivity / precision_recall_sum
        if precision_recall_sum > 0
        else 0.0
    )
    flag_rate = (tp + fp) / n_total if n_total > 0 else 0.0

    return {
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "accuracy": accuracy,
        "f1": f1,
        "flag_rate": flag_rate,
    }


def roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    n_thresholds: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ROC curve (FPR, TPR, thresholds)."""
    thresholds = np.linspace(0, 1, n_thresholds)
    fprs, tprs = [], []
    for t in thresholds:
        m = confusion_matrix_metrics(y_true, y_scores, threshold=t)
        tprs.append(m["sensitivity"])
        fprs.append(1 - m["specificity"])
    return np.array(fprs), np.array(tprs), thresholds


def auc_score(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute AUC using the trapezoidal rule on ROC curve."""
    fprs, tprs, _ = roc_curve(y_true, y_scores)
    # Sort by fpr then tpr for correct integration (stable sort preserves order
    # among ties; secondary tpr sort ensures (0,0) precedes (0,1) etc.)
    order = np.lexsort((tprs, fprs))
    trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(trapz(tprs[order], fprs[order]))


def hosmer_lemeshow_test(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, float]:
    """Hosmer-Lemeshow goodness-of-fit test for calibration.

    Returns (HL statistic, p-value). p > 0.05 indicates good calibration.
    """
    from scipy import stats

    order = np.argsort(y_pred)
    y_true_sorted = y_true[order]
    y_pred_sorted = y_pred[order]

    bins = np.array_split(np.arange(len(y_true)), n_bins)
    hl_stat = 0.0
    for bin_idx in bins:
        if len(bin_idx) == 0:
            continue
        observed = y_true_sorted[bin_idx].sum()
        expected = y_pred_sorted[bin_idx].sum()
        n = len(bin_idx)
        mean_pred = y_pred_sorted[bin_idx].mean()
        denom = n * mean_pred * (1 - mean_pred)
        if denom > 1e-10:
            hl_stat += (observed - expected) ** 2 / denom

    p_value = 1 - stats.chi2.cdf(hl_stat, df=max(n_bins - 2, 1))
    return float(hl_stat), float(p_value)


def calibration_slope(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute calibration slope (predicted vs observed bin means).

    Returns (slope, bin_predicted_means, bin_observed_means).
    Slope near 1.0 indicates good calibration.
    """
    order = np.argsort(y_pred)
    y_true_sorted = y_true[order]
    y_pred_sorted = y_pred[order]

    bins = np.array_split(np.arange(len(y_true)), n_bins)
    pred_means, obs_means = [], []
    for bin_idx in bins:
        if len(bin_idx) == 0:
            continue
        pred_means.append(y_pred_sorted[bin_idx].mean())
        obs_means.append(y_true_sorted[bin_idx].mean())

    pred_arr = np.array(pred_means)
    obs_arr = np.array(obs_means)

    if len(pred_arr) < 2 or np.std(pred_arr) < 1e-10:
        return 1.0, pred_arr, obs_arr

    slope = np.polyfit(pred_arr, obs_arr, 1)[0]
    return float(slope), pred_arr, obs_arr


# -- Ranking metrics (precision-recall family) -----------------------------

def auprc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Area under the precision-recall curve (average precision).

    Uses the standard step-interpolated average-precision formulation:
    AP = mean over positives of (precision at each positive's rank).

    Scores must be NaN-free (callers pass the scoreable mask); NaN would
    silently poison the ranking, so it raises instead.
    Returns 0.0 if there are no positives.
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores, dtype=float)
    if np.isnan(y_scores).any():
        raise ValueError("auprc: y_scores contains NaN; mask before calling")
    n_pos = int(np.sum(y_true == 1))
    if n_pos == 0:
        return 0.0
    order = np.argsort(-y_scores, kind="stable")
    hits = (y_true[order] == 1)
    cum_hits = np.cumsum(hits)
    ranks = np.arange(1, len(y_true) + 1)
    precision_at_hits = cum_hits[hits] / ranks[hits]
    return float(precision_at_hits.sum() / n_pos)


def _top_k_indices(y_scores: np.ndarray, k_fraction: float) -> np.ndarray:
    """Indices of the top k-fraction of scores (descending, stable ties)."""
    y_scores = np.asarray(y_scores, dtype=float)
    if np.isnan(y_scores).any():
        raise ValueError(
            "top-k metrics: y_scores contains NaN; mask before calling"
        )
    if not 0.0 < k_fraction <= 1.0:
        raise ValueError(f"k_fraction must be in (0, 1], got {k_fraction}")
    k = max(1, int(round(k_fraction * len(y_scores))))
    order = np.argsort(-y_scores, kind="stable")
    return order[:k]


def precision_at_k(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    k_fraction: float,
) -> float:
    """Precision among the top k-fraction of entities by score."""
    y_true = np.asarray(y_true)
    top = _top_k_indices(y_scores, k_fraction)
    return float(np.sum(y_true[top] == 1) / len(top))


def capture_at_k(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    k_fraction: float,
) -> float:
    """Fraction of all positives captured in the top k-fraction (recall@K).

    Returns 0.0 if there are no positives.
    """
    y_true = np.asarray(y_true)
    n_pos = int(np.sum(y_true == 1))
    if n_pos == 0:
        return 0.0
    top = _top_k_indices(y_scores, k_fraction)
    return float(np.sum(y_true[top] == 1) / n_pos)


# -- Prevalence-aware performance bounds ----------------------------------

def theoretical_ppv(
    prevalence: float,
    sensitivity: float,
    specificity: float,
) -> float:
    """Compute PPV from prevalence, sensitivity, and specificity.

    Bayes' theorem:
    PPV = (sens * prev) / (sens * prev + (1-spec) * (1-prev))
    """
    numer = sensitivity * prevalence
    denom = numer + (1 - specificity) * (1 - prevalence)
    if denom < 1e-12:
        return 0.0
    return numer / denom


def theoretical_ppv_bounds(
    prevalence: float,
    sensitivity_range: np.ndarray = None,
    specificities: list = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Compute PPV bounds across sensitivity and specificity values.

    Returns (sensitivity_range, dict of PPV arrays keyed by specificity).
    Shows the maximum achievable PPV at each operating point.
    """
    if sensitivity_range is None:
        sensitivity_range = np.linspace(0.1, 1.0, 50)
    if specificities is None:
        specificities = [0.70, 0.80, 0.90, 0.95, 0.99]

    results = {}
    for spec in specificities:
        ppvs = np.array([
            theoretical_ppv(prevalence, sens, spec)
            for sens in sensitivity_range
        ])
        results[f"spec_{spec:.2f}"] = ppvs

    return sensitivity_range, results


def check_target_feasibility(
    prevalence: float,
    target_ppv: float,
    target_sensitivity: float = 0.80,
) -> Dict[str, float]:
    """Check whether a target PPV is ARITHMETICALLY self-consistent.

    WARNING -- `feasible` here does NOT mean "your model can do this". It
    is pure Bayes arithmetic: it asks only whether some specificity in
    (0, 1) yields the target PPV at the target sensitivity and
    prevalence. It never consults the model's discriminative ability, so
    it returns True for operating points no real classifier could reach.

    A worked case: prevalence 0.008, PPV 0.15, sensitivity 0.80 gives
    required_specificity 0.9625 < 1.0, so `feasible` is True -- yet
    holding sensitivity 0.80 AT that specificity needs AUC ~0.97, while
    the fitted model reached 0.82 and delivered sensitivity 0.049.

    For a check that accounts for achievable AUC, use
    `required_auc_for_operating_point()` and drive the model in
    `operating_point` mode, which gates on empirically achieved metrics.

    Returns dict with:
    - feasible: bool -- arithmetic consistency ONLY (see warning above)
    - max_ppv_at_spec_95: maximum PPV at 95% specificity
    - max_ppv_at_spec_99: maximum PPV at 99% specificity
    - required_specificity: specificity needed to achieve target PPV
    """
    max_95 = theoretical_ppv(prevalence, target_sensitivity, 0.95)
    max_99 = theoretical_ppv(prevalence, target_sensitivity, 0.99)

    # Find required specificity for target PPV
    # PPV = (sens * prev) / (sens * prev + (1-spec) * (1-prev))
    # Solve for spec:
    # PPV * (sens*prev + (1-spec)*(1-prev)) = sens * prev
    # PPV * (1-spec) * (1-prev) = sens*prev * (1 - PPV)
    # (1-spec) = sens*prev*(1-PPV) / (PPV*(1-prev))
    # spec = 1 - sens*prev*(1-PPV) / (PPV*(1-prev))
    if target_ppv > 0 and (1 - prevalence) > 0:
        required_spec = 1 - (
            target_sensitivity * prevalence * (1 - target_ppv)
            / (target_ppv * (1 - prevalence))
        )
    else:
        required_spec = 1.0

    feasible = required_spec < 1.0 and required_spec > 0

    return {
        "feasible": feasible,
        "max_ppv_at_spec_95": max_95,
        "max_ppv_at_spec_99": max_99,
        "required_specificity": max(0.0, required_spec),
        "prevalence": prevalence,
        "target_ppv": target_ppv,
        "target_sensitivity": target_sensitivity,
    }


# -- Operating-point geometry ---------------------------------------------
#
# Fixing PPV at a given prevalence pins a straight ray through the origin
# in ROC space:
#
#     FPR = k * TPR,   k = prev * (1 - ppv) / (ppv * (1 - prev))
#
# Any ROC curve crosses that ray exactly once, so (prevalence, PPV) plus a
# curve determines sensitivity uniquely -- sensitivity is NOT a free knob.
# Fixing sensitivity instead pins a horizontal line, which likewise crosses
# any curve once. Either metric can therefore serve as the binding
# constraint; the other is then a consequence of the model's AUC.
#
# The helpers below use a binormal ROC (equal-variance) to answer "what AUC
# would this operating point require?". Validated against the SDK's beta
# risk distributions: absolute sensitivity error 0.004-0.011 at AUC >= 0.83,
# rising to ~0.09 at AUC ~0.68. That is accurate enough to REPORT a required
# AUC and to explain an infeasible ask, but NOT accurate enough to gate
# pass/fail -- always gate on empirically achieved metrics.

def iso_ppv_slope(prevalence: float, ppv: float) -> float:
    """Slope (TPR per unit FPR) of the constant-PPV ray in ROC space.

    A slope <= 1 puts the ray on or below the chance diagonal, which asks
    for a worse-than-random classifier. That happens exactly when the PPV
    target does not exceed prevalence.
    """
    if not 0.0 < ppv < 1.0 or not 0.0 < prevalence < 1.0:
        raise ValueError(
            f"prevalence and ppv must lie in (0, 1); got "
            f"prevalence={prevalence}, ppv={ppv}"
        )
    k = prevalence * (1 - ppv) / (ppv * (1 - prevalence))
    return 1.0 / k


def validate_operating_point(
    prevalence: float,
    ppv: float,
) -> Dict:
    """Reject operating points that are degenerate before any fitting.

    Returns a dict with 'valid', 'reason' (None when valid) and the
    iso-PPV ray slope.
    """
    slope = iso_ppv_slope(prevalence, ppv)
    if ppv <= prevalence:
        return {
            "valid": False,
            "slope": slope,
            "reason": (
                f"target PPV {ppv:.4f} does not exceed prevalence "
                f"{prevalence:.4f}. A flagged set no more enriched than "
                f"the base rate describes a worse-than-random model; the "
                f"constant-PPV ray (slope {slope:.3f}) lies on or below "
                f"the chance diagonal. Raise the PPV target above "
                f"prevalence."
            ),
        }
    return {"valid": True, "slope": slope, "reason": None}


def sensitivity_at_ppv(
    auc: float,
    prevalence: float,
    ppv: float,
) -> float:
    """Sensitivity attainable at a given PPV, for a binormal ROC of `auc`.

    Solves for the crossing of the ROC curve with the constant-PPV ray.
    Returns 0.0 when the ray lies entirely above the curve (the PPV is
    unreachable at any non-trivial operating point for this AUC).
    """
    from scipy.optimize import brentq
    from scipy.stats import norm

    if not 0.5 <= auc < 1.0:
        raise ValueError(f"auc must lie in [0.5, 1.0); got {auc}")
    k = prevalence * (1 - ppv) / (ppv * (1 - prevalence))
    d_prime = norm.ppf(auc) * np.sqrt(2.0)

    def gap(tpr: float) -> float:
        fpr = float(np.clip(k * tpr, 1e-12, 1 - 1e-12))
        return float(norm.cdf(norm.ppf(fpr) + d_prime) - tpr)

    hi = min(1.0 / k, 1.0) - 1e-9
    if hi <= 1e-9:
        return 0.0
    if gap(1e-9) <= 0.0:
        return 0.0          # ray above the curve everywhere
    if gap(hi) >= 0.0:
        return float(hi)    # curve still above the ray at the ceiling
    return float(brentq(gap, 1e-9, hi, xtol=1e-12))


def ppv_at_sensitivity(
    auc: float,
    prevalence: float,
    sensitivity: float,
) -> float:
    """PPV attainable at a given sensitivity, for a binormal ROC of `auc`."""
    from scipy.stats import norm

    if not 0.5 <= auc < 1.0:
        raise ValueError(f"auc must lie in [0.5, 1.0); got {auc}")
    if not 0.0 < sensitivity < 1.0:
        raise ValueError(
            f"sensitivity must lie in (0, 1); got {sensitivity}"
        )
    d_prime = norm.ppf(auc) * np.sqrt(2.0)
    # Invert TPR = Phi(Phi^-1(FPR) + d') for FPR.
    fpr = float(norm.cdf(norm.ppf(sensitivity) - d_prime))
    specificity = 1.0 - fpr
    return theoretical_ppv(prevalence, sensitivity, specificity)


def required_auc_for_operating_point(
    prevalence: float,
    binding: str,
    binding_value: float,
    goal_value: float,
    auc_bounds: Tuple[float, float] = (0.5, 0.9999),
) -> Dict:
    """Smallest AUC whose ROC reaches `goal_value` at the binding point.

    `binding` is 'ppv' (goal is sensitivity) or 'sensitivity' (goal is
    PPV). Returns a dict with 'required_auc' (None when the goal is not
    reachable below auc_bounds[1]), plus the value attainable at each
    bound so callers can explain the shortfall.
    """
    from scipy.optimize import brentq

    if binding not in ("ppv", "sensitivity"):
        raise ValueError(
            f"binding must be 'ppv' or 'sensitivity'; got '{binding}'"
        )

    if binding == "ppv":
        def attain(auc: float) -> float:
            return sensitivity_at_ppv(auc, prevalence, binding_value)
    else:
        def attain(auc: float) -> float:
            return ppv_at_sensitivity(auc, prevalence, binding_value)

    lo, hi = auc_bounds
    at_lo, at_hi = attain(lo), attain(hi)
    out = {
        "binding": binding,
        "binding_value": binding_value,
        "goal_value": goal_value,
        "attainable_at_auc_lo": at_lo,
        "attainable_at_auc_hi": at_hi,
        "auc_bounds": auc_bounds,
        "required_auc": None,
    }
    if at_hi < goal_value:
        return out          # unreachable even at the AUC ceiling
    if at_lo >= goal_value:
        out["required_auc"] = lo
        return out
    out["required_auc"] = float(
        brentq(lambda a: attain(a) - goal_value, lo, hi, xtol=1e-6)
    )
    return out
