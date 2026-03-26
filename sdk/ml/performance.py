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
