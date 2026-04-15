"""Interrupted Time Series (ITS) segmented regression estimators.

This module provides three estimators for post-hoc analysis of
simulation output:

1. ``segmented_regression`` — classic pre/post ITS with a break at a
   known timepoint. Fits

       Y_t = beta0 + beta1 * t + beta2 * post_t + beta3 * (t - T*) * post_t

   and returns the level change (``beta2``), the slope change
   (``beta3``), and the pre-period trend (``beta1``) for validity
   checking. Supports Newey-West HAC standard errors to handle
   first-order residual autocorrelation (the SE-inflation penalty
   that a naive OLS fit would miss).

2. ``cits_with_control`` — stacked-group controlled ITS with a control
   series (typically the counterfactual branch). Lifted from
   ``nurse_retention/monitoring/tier3_cits._fit_cits_with_cf``. Fits

       Y = b0 + b1 * t + b2 * group + b3 * group * t

   and reports the program effect at a chosen timepoint as
   ``b2 + b3 * t``.

3. ``its_slope_only`` — floor estimator. Fits ``Y = b0 + b1 * t`` on a
   single series and returns the slope. Lifted from
   ``nurse_retention/monitoring/tier3_cits._fit_its_only``.

Both lifted modes produce results that are numerically identical to
the original ``tier3_cits.py`` fits; the extraction exists so other
scenarios (notably ``perinatal_efm_early_warning``) can use the same
math without importing nurse-retention-specific modules.

References
----------
- Wagner, A. K. et al. (2002). Segmented regression analysis of
  interrupted time series studies in medication use research.
  *J Clin Pharm Ther* 27(4):299-309.
- Bernal, J. L., Cummins, S., Gasparrini, A. (2017). Interrupted
  time series regression for the evaluation of public health
  interventions. *Int J Epidemiol* 46(1):348-355.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import statsmodels.api as sm
from scipy.stats import norm


@dataclass
class ITSResult:
    """Result of a pre/post segmented regression.

    Attributes
    ----------
    level_change : float
        ``beta2`` — instantaneous level shift at the break. Negative
        values indicate a post-period level below the pre-period
        projection.
    level_change_se : float
        Standard error of ``beta2`` (HAC-adjusted if
        ``hac_maxlags > 0``).
    level_change_pvalue : float
        Two-sided p-value for ``beta2 = 0``.
    slope_change : float
        ``beta3`` — change in slope after the break.
    slope_change_se : float
        Standard error of ``beta3``.
    slope_change_pvalue : float
        Two-sided p-value for ``beta3 = 0``.
    pre_slope : float
        ``beta1`` — pre-period time trend. Used as a validity check;
        if this is significantly non-zero, the pre-period has a
        confounding trend and the ITS interpretation is compromised.
    pre_slope_se : float
        Standard error of ``beta1``.
    pre_slope_pvalue : float
        Two-sided p-value for ``beta1 = 0``.
    intercept : float
        ``beta0`` — pre-period intercept.
    n_pre : int
        Number of observations in the pre-period (t < break_index).
    n_post : int
        Number of observations in the post-period (t >= break_index).
    hac_maxlags : int
        Newey-West lag used. ``0`` means plain OLS standard errors.
    direction : str
        ``"decrease"`` if ``beta2`` is negative and its magnitude
        exceeds ``direction_tolerance_ses * level_change_se``;
        ``"increase"`` if positive beyond the same tolerance;
        ``"none"`` otherwise. A non-``"none"`` direction does not
        imply statistical significance — see ``level_change_pvalue``.
    """

    level_change: float
    level_change_se: float
    level_change_pvalue: float
    slope_change: float
    slope_change_se: float
    slope_change_pvalue: float
    pre_slope: float
    pre_slope_se: float
    pre_slope_pvalue: float
    intercept: float
    n_pre: int
    n_post: int
    hac_maxlags: int
    direction: str

    def is_significant(self, alpha: float = 0.05) -> bool:
        """Whether the level change is significant at ``alpha``."""
        return self.level_change_pvalue < alpha


@dataclass
class CITSWithControlResult:
    """Result of a stacked-group CITS fit with a control series.

    Mirrors the output shape of
    ``nurse_retention/monitoring/tier3_cits._fit_cits_with_cf``.
    """

    effect_estimate: float
    effect_se: float
    ci_lower: float
    ci_upper: float
    p_value: float
    n_observations: int


@dataclass
class SlopeOnlyResult:
    """Result of a slope-only fit on a single series.

    Mirrors the output shape of
    ``nurse_retention/monitoring/tier3_cits._fit_its_only``.
    """

    slope: float
    slope_se: float
    ci_lower: float
    ci_upper: float
    p_value: float
    n_observations: int


def segmented_regression(
    series: np.ndarray,
    break_index: int,
    *,
    hac_maxlags: int = 1,
    direction_tolerance_ses: float = 2.0,
) -> ITSResult:
    """Classic pre/post ITS segmented regression.

    Fits

        Y_t = beta0 + beta1 * t + beta2 * post_t
              + beta3 * (t - T*) * post_t + epsilon_t

    where ``post_t`` is 1 for ``t >= break_index`` and 0 otherwise,
    and ``T*`` is the break index. ``beta2`` is the instantaneous
    level change at the break; ``beta3`` is the change in slope after
    the break; ``beta1`` is the pre-period trend (a validity check).

    Newey-West HAC standard errors handle first-order residual
    autocorrelation and produce the SE-inflation penalty that a naive
    OLS fit would miss. For perinatal-scale monthly series with
    moderate persistence, ``hac_maxlags=1`` is a reasonable default;
    for weekly or noisier series, 4 is also common.

    Parameters
    ----------
    series : np.ndarray
        1-D observations indexed by time.
    break_index : int
        Index in ``series`` where the post-period begins (0-indexed).
        Must leave at least 2 observations on each side.
    hac_maxlags : int, default 1
        Newey-West lag. ``0`` disables HAC and uses plain OLS
        standard errors.
    direction_tolerance_ses : float, default 2.0
        ``direction`` is ``"decrease"``/``"increase"`` only if
        ``|beta2| > direction_tolerance_ses * SE(beta2)``; otherwise
        ``"none"``. This is a stricter convention than a bare sign
        check so that near-zero level changes on noisy series are
        not classified.

    Returns
    -------
    ITSResult
        All segmented-regression coefficients, their standard errors,
        and two-sided p-values.

    Raises
    ------
    ValueError
        If ``series`` is shorter than 4 points or ``break_index``
        does not leave at least 2 points on each side.
    """
    y = np.asarray(series, dtype=float)
    n = len(y)
    if n < 4:
        raise ValueError(
            f"segmented_regression requires at least 4 observations, "
            f"got {n}"
        )
    if break_index < 2 or break_index > n - 2:
        raise ValueError(
            f"break_index must be between 2 and {n - 2} (inclusive) "
            f"to leave at least 2 observations on each side, "
            f"got {break_index}"
        )
    if hac_maxlags < 0:
        raise ValueError(
            f"hac_maxlags must be non-negative, got {hac_maxlags}"
        )

    t = np.arange(n, dtype=float)
    post = (t >= break_index).astype(float)
    time_since_break = np.where(post > 0, t - break_index, 0.0)

    X = np.column_stack([np.ones(n), t, post, time_since_break])
    model = sm.OLS(y, X)
    if hac_maxlags > 0:
        fit = model.fit(
            cov_type="HAC", cov_kwds={"maxlags": hac_maxlags},
        )
    else:
        fit = model.fit()

    beta0, beta1, beta2, beta3 = fit.params
    se0, se1, se2, se3 = fit.bse
    p1 = float(fit.pvalues[1])
    p2 = float(fit.pvalues[2])
    p3 = float(fit.pvalues[3])

    if abs(beta2) > direction_tolerance_ses * se2:
        direction = "decrease" if beta2 < 0 else "increase"
    else:
        direction = "none"

    return ITSResult(
        level_change=float(beta2),
        level_change_se=float(se2),
        level_change_pvalue=p2,
        slope_change=float(beta3),
        slope_change_se=float(se3),
        slope_change_pvalue=p3,
        pre_slope=float(beta1),
        pre_slope_se=float(se1),
        pre_slope_pvalue=p1,
        intercept=float(beta0),
        n_pre=int(break_index),
        n_post=int(n - break_index),
        hac_maxlags=int(hac_maxlags),
        direction=direction,
    )


def power_across_seeds(
    series_list: List[np.ndarray],
    break_index: int,
    *,
    alpha: float = 0.05,
    expected_direction: str = "decrease",
    hac_maxlags: int = 1,
) -> float:
    """Fraction of series whose ITS level change is significant and
    in the expected direction.

    Matches the E3 estimand in the perinatal protocol: per cell,
    power is the fraction of seeds with ``level_change_pvalue < alpha``
    AND the sign of ``level_change`` matching ``expected_direction``.

    Parameters
    ----------
    series_list : list of np.ndarray
        One 1-D series per seed.
    break_index : int
        Index where the post-period begins. Must be valid for every
        series.
    alpha : float, default 0.05
        Significance threshold.
    expected_direction : {"decrease", "increase"}, default "decrease"
        Series whose ``level_change`` has the opposite sign do not
        count toward power even if ``p < alpha``.
    hac_maxlags : int, default 1
        Passed through to ``segmented_regression``.

    Returns
    -------
    float
        Power in ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``series_list`` is empty or ``expected_direction`` is not
        ``"decrease"`` or ``"increase"``.
    """
    if not series_list:
        raise ValueError("series_list is empty")
    if expected_direction not in ("decrease", "increase"):
        raise ValueError(
            f"expected_direction must be 'decrease' or 'increase', "
            f"got {expected_direction!r}"
        )

    hits = 0
    for series in series_list:
        result = segmented_regression(
            series, break_index, hac_maxlags=hac_maxlags,
        )
        if result.level_change_pvalue >= alpha:
            continue
        sign = (
            "decrease" if result.level_change < 0 else "increase"
        )
        if sign == expected_direction:
            hits += 1
    return hits / len(series_list)


def cits_with_control(
    treatment_series: np.ndarray,
    control_series: np.ndarray,
    *,
    effect_timepoint: Optional[int] = None,
    z_critical: float = 1.96,
) -> CITSWithControlResult:
    """Stacked-group CITS with a control series.

    Lifted from ``tier3_cits._fit_cits_with_cf``. Fits

        Y = b0 + b1 * t + b2 * group + b3 * group * t

    where ``group = 1`` for the treatment series and ``0`` for the
    control series. The program effect at ``effect_timepoint`` is
    ``b2 + b3 * effect_timepoint``. When the treatment is on from
    t=0 (no pre-period) and the control series is the counterfactual
    branch, this is the right estimator — the nurse-retention
    monitoring harness uses it in exactly that configuration.

    Parameters
    ----------
    treatment_series : np.ndarray
        Factual-branch observations.
    control_series : np.ndarray
        Counterfactual-branch observations. Must be the same length
        as ``treatment_series``.
    effect_timepoint : int, optional
        Timepoint at which the effect is reported. Default is the
        last observation (``n - 1``), which matches the tier3
        behaviour.
    z_critical : float, default 1.96
        Z-score for the confidence interval. Default is the 95%
        two-sided value.

    Returns
    -------
    CITSWithControlResult
        Effect estimate, standard error, 95% CI, two-sided p-value,
        and the total observation count (``2 * n`` after stacking).

    Raises
    ------
    ValueError
        If the two series have different lengths or are shorter than
        3 observations.
    """
    y_t = np.asarray(treatment_series, dtype=float)
    y_c = np.asarray(control_series, dtype=float)
    if len(y_t) != len(y_c):
        raise ValueError(
            f"treatment and control series must be the same length, "
            f"got {len(y_t)} and {len(y_c)}"
        )
    n = len(y_t)
    if n < 3:
        raise ValueError(
            f"cits_with_control requires at least 3 observations, "
            f"got {n}"
        )

    t_vals = np.arange(n, dtype=float)
    Y = np.concatenate([y_t, y_c])
    time = np.concatenate([t_vals, t_vals])
    group = np.concatenate([np.ones(n), np.zeros(n)])
    X = np.column_stack(
        [np.ones(2 * n), time, group, group * time],
    )

    fit = sm.OLS(Y, X).fit()
    beta2 = fit.params[2]
    beta3 = fit.params[3]
    T = int(effect_timepoint) if effect_timepoint is not None else n - 1

    effect = beta2 + beta3 * T

    cov = fit.cov_params()
    var_effect = (
        cov[2, 2] + (T ** 2) * cov[3, 3] + 2.0 * T * cov[2, 3]
    )
    se_effect = float(np.sqrt(max(var_effect, 0.0)))

    ci_half = z_critical * se_effect
    if se_effect > 0:
        z_stat = effect / se_effect
        p_value = 2.0 * float(norm.sf(abs(z_stat)))
    else:
        p_value = 1.0

    return CITSWithControlResult(
        effect_estimate=float(effect),
        effect_se=se_effect,
        ci_lower=float(effect - ci_half),
        ci_upper=float(effect + ci_half),
        p_value=p_value,
        n_observations=2 * n,
    )


def its_slope_only(
    series: np.ndarray,
    *,
    z_critical: float = 1.96,
) -> SlopeOnlyResult:
    """Slope-only fit on a single series.

    Lifted from ``tier3_cits._fit_its_only``. Fits
    ``Y = b0 + b1 * t + epsilon`` on ``series`` and returns the slope
    with its standard error, 95% CI, and p-value. This is a weak
    floor estimator for scenarios where no valid control or
    pre-period is available.

    Parameters
    ----------
    series : np.ndarray
        1-D observations indexed by time.
    z_critical : float, default 1.96
        Z-score for the confidence interval.

    Returns
    -------
    SlopeOnlyResult

    Raises
    ------
    ValueError
        If ``series`` is shorter than 3 observations.
    """
    y = np.asarray(series, dtype=float)
    n = len(y)
    if n < 3:
        raise ValueError(
            f"its_slope_only requires at least 3 observations, "
            f"got {n}"
        )

    t_vals = np.arange(n, dtype=float)
    X = np.column_stack([np.ones(n), t_vals])
    fit = sm.OLS(y, X).fit()

    slope = float(fit.params[1])
    se_slope = float(fit.bse[1])
    ci_half = z_critical * se_slope
    p_value = float(fit.pvalues[1])

    return SlopeOnlyResult(
        slope=slope,
        slope_se=se_slope,
        ci_lower=slope - ci_half,
        ci_upper=slope + ci_half,
        p_value=p_value,
        n_observations=n,
    )


__all__ = [
    "ITSResult",
    "CITSWithControlResult",
    "SlopeOnlyResult",
    "segmented_regression",
    "power_across_seeds",
    "cits_with_control",
    "its_slope_only",
]
