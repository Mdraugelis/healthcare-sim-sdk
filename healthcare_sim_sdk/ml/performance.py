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
    # Sort by fpr for correct integration
    order = np.argsort(fprs)
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
    """Check if target PPV is achievable given prevalence.

    Returns dict with:
    - feasible: bool (achievable at specificity < 1.0)
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
