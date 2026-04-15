"""Tier 3: Rolling Comparative Interrupted Time Series (CITS).

Two analysis modes:

1. **CITS with counterfactual as control series** (oracle mode). At
   each quarterly refit, stack the factual and counterfactual weekly
   turnover series and fit:

       Y = β0 + β1*t + β2*treat + β3*treat*t + ε

   where ``treat`` is 1 for factual and 0 for counterfactual. β2 is
   the level difference between the two series; β3 is how that
   difference evolves over time.

   For the nurse retention scenario there is no pre/post interruption
   — the program is on from week 0 — so the CITS simplifies to a
   direct factual-vs-CF comparison with a time trend. The "treatment
   effect at week T" is computed as the cumulative prevented
   departures through week T, estimated from the fitted group
   difference on the turnover rate series.

2. **Pure ITS (floor mode)**. Fit the factual series alone against a
   constant baseline and a time trend:

       Y = β0 + β1*t + ε

   The "effect" is inferred from the residual trajectory relative to
   some pre-specified expected rate. This is deliberately weaker than
   the CITS-with-CF mode and represents the worst-case operational
   conditions where no valid control series is available.

Wagner et al. (2002) is the canonical reference for the segmented
regression interpretation. Since the nurse retention scenario has no
pre-period, we lean on the stacked-group interpretation from
Bernal, Cummins, Gasparrini (2017).

The rolling refit cadence is quarterly (13 weeks). At each refit,
the most recent quarter's data is added and the model is re-fit over
the full window so far. This mirrors how a real operational dashboard
would update the estimate as new weeks accumulate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import statsmodels.api as sm

from healthcare_sim_sdk.scenarios.nurse_retention.monitoring.events import (
    DetectionEvent,
    Tier3Estimate,
)


@dataclass
class Tier3CITS:
    """Rolling CITS estimator.

    Parameters
    ----------
    mode : str
        ``"cits_with_cf"`` (oracle, uses CF branch as control series)
        or ``"its_only"`` (factual series alone).
    refit_interval_weeks : int
        How often to refit. Default 13 (quarterly).
    min_observations : int
        Minimum number of weeks before the first fit. Default 13
        (one quarter of data).
    """

    mode: str = "cits_with_cf"
    refit_interval_weeks: int = 13
    min_observations: int = 13

    # Internal state
    factual_history: List[float] = field(default_factory=list)
    counterfactual_history: List[float] = field(default_factory=list)
    estimates: List[Tier3Estimate] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.mode not in ("cits_with_cf", "its_only"):
            raise ValueError(
                f"mode must be cits_with_cf or its_only, "
                f"got {self.mode!r}"
            )

    def update(
        self,
        week: int,
        factual_turnover: float,
        counterfactual_turnover: Optional[float] = None,
    ) -> List[DetectionEvent]:
        """Add one week of data and possibly refit.

        Parameters
        ----------
        week : int
            Current simulation week.
        factual_turnover : float
            Weekly turnover rate from the factual branch.
        counterfactual_turnover : Optional[float]
            Weekly turnover rate from the counterfactual branch.
            Required for ``cits_with_cf`` mode. Ignored otherwise.

        Returns
        -------
        list of DetectionEvent
            Empty unless a refit just happened and the new estimate
            is significantly different from the previous one, or the
            point estimate crosses zero in either direction.
        """
        self.factual_history.append(float(factual_turnover))
        if counterfactual_turnover is not None:
            self.counterfactual_history.append(
                float(counterfactual_turnover)
            )

        events: List[DetectionEvent] = []

        # Refit only at quarterly boundaries (or whenever we hit the
        # minimum observation count for the first time)
        n = len(self.factual_history)
        if n < self.min_observations:
            return events

        is_refit_week = (
            n == self.min_observations
            or (n - self.min_observations) % self.refit_interval_weeks
            == 0
        )
        if not is_refit_week:
            return events

        estimate = self._fit(week)
        if estimate is None:
            return events
        self.estimates.append(estimate)

        # Emit a detection event if the effect is significant AND
        # this is either the first estimate or the direction flipped
        if estimate.is_significant():
            # Determine direction from the estimate sign
            if estimate.effect_estimate > 0:
                direction = "up"
            elif estimate.effect_estimate < 0:
                direction = "down"
            else:
                direction = "level"

            # Only emit if we've crossed a significance threshold
            # (not every quarter once detected). We track this by
            # only emitting on the first significant estimate, and
            # again on any subsequent direction flip.
            emit = False
            if len(self.estimates) == 1:
                emit = True
            else:
                prev = self.estimates[-2]
                if not prev.is_significant():
                    emit = True
                elif (
                    (prev.effect_estimate > 0)
                    != (estimate.effect_estimate > 0)
                ):
                    emit = True

            if emit:
                events.append(
                    DetectionEvent(
                        tier=3,
                        metric="program_effect",
                        week=week,
                        severity="warning",
                        value=estimate.effect_estimate,
                        direction=direction,
                        rule=f"cits_quarterly_{self.mode}",
                    )
                )

        return events

    def _fit(self, current_week: int) -> Optional[Tier3Estimate]:
        """Fit the segmented regression model and return an estimate."""
        n_weeks = len(self.factual_history)
        if n_weeks < self.min_observations:
            return None

        if self.mode == "cits_with_cf":
            return self._fit_cits_with_cf(current_week)
        else:
            return self._fit_its_only(current_week)

    def _fit_cits_with_cf(
        self, current_week: int,
    ) -> Optional[Tier3Estimate]:
        """Fit CITS with counterfactual as the comparison series.

        Stack factual and CF series with a group dummy. Model:

            Y = β0 + β1*t + β2*group + β3*group*t + ε

        where group=1 for factual, 0 for CF. The program effect at
        week T is ``β2 + β3*T`` — the cumulative level difference
        adjusted for the group-specific slope.
        """
        if len(self.counterfactual_history) != len(self.factual_history):
            return None

        n = len(self.factual_history)
        t_vals = np.arange(n, dtype=float)

        # Stack factual (group=1) and CF (group=0)
        y_factual = np.array(self.factual_history)
        y_cf = np.array(self.counterfactual_history)

        Y = np.concatenate([y_factual, y_cf])
        time = np.concatenate([t_vals, t_vals])
        group = np.concatenate([np.ones(n), np.zeros(n)])

        # Design matrix: intercept, time, group, group*time
        X = np.column_stack(
            [
                np.ones(2 * n),
                time,
                group,
                group * time,
            ]
        )

        try:
            model = sm.OLS(Y, X).fit()
        except Exception:
            return None

        # Effect at current_week is β2 + β3 * current_week
        # But since the scenario is intervention-from-week-0, we
        # report the average level difference over the observed
        # window: β2 + β3 * (n-1)/2  (midpoint) is a reasonable
        # summary. Following the task spec, we report the effect
        # "at that quarter" which means the current endpoint.
        T = n - 1  # last observed week in the window
        beta2 = model.params[2]
        beta3 = model.params[3]
        effect = beta2 + beta3 * T

        # Variance of (beta2 + beta3*T):
        # Var(β2 + T*β3) = Var(β2) + T^2*Var(β3) + 2*T*Cov(β2,β3)
        cov = model.cov_params()
        var_effect = (
            cov[2, 2]
            + (T ** 2) * cov[3, 3]
            + 2.0 * T * cov[2, 3]
        )
        se_effect = float(np.sqrt(max(var_effect, 0.0)))

        ci_half = 1.96 * se_effect
        ci_lower = effect - ci_half
        ci_upper = effect + ci_half

        # Approximate p-value from z-stat
        if se_effect > 0:
            z_stat = effect / se_effect
            # Two-sided p via scipy.stats.norm.sf
            from scipy.stats import norm
            p_value = 2.0 * float(norm.sf(abs(z_stat)))
        else:
            p_value = 1.0

        return Tier3Estimate(
            week=current_week,
            effect_estimate=float(effect),
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            p_value=p_value,
            n_observations=2 * n,
            mode="cits_with_cf",
        )

    def _fit_its_only(
        self, current_week: int,
    ) -> Optional[Tier3Estimate]:
        """Pure ITS on the factual series alone.

        Fits ``Y = β0 + β1*t + ε`` on the factual history and
        returns the slope β1 as the "effect" (interpreted as the
        per-week change in turnover rate). This is a floor
        interpretation — a real operational ITS would segment against
        a pre-period baseline, which this scenario does not have.
        """
        n = len(self.factual_history)
        t_vals = np.arange(n, dtype=float)
        Y = np.array(self.factual_history)
        X = np.column_stack([np.ones(n), t_vals])

        try:
            model = sm.OLS(Y, X).fit()
        except Exception:
            return None

        slope = float(model.params[1])
        se_slope = float(model.bse[1])
        ci_half = 1.96 * se_slope
        ci_lower = slope - ci_half
        ci_upper = slope + ci_half
        p_value = float(model.pvalues[1])

        return Tier3Estimate(
            week=current_week,
            effect_estimate=slope,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            n_observations=n,
            mode="its_only",
        )
