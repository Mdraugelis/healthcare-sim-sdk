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

from healthcare_sim_sdk.experiments.analysis.its import (
    cits_with_control,
    its_slope_only,
)
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

        Delegates to
        ``experiments.analysis.its.cits_with_control``. The program
        effect is reported at the current endpoint (``n - 1``), which
        matches the original tier3 behaviour.
        """
        if len(self.counterfactual_history) != len(self.factual_history):
            return None

        try:
            result = cits_with_control(
                treatment_series=np.asarray(self.factual_history),
                control_series=np.asarray(self.counterfactual_history),
            )
        except ValueError:
            return None

        return Tier3Estimate(
            week=current_week,
            effect_estimate=result.effect_estimate,
            ci_lower=result.ci_lower,
            ci_upper=result.ci_upper,
            p_value=result.p_value,
            n_observations=result.n_observations,
            mode="cits_with_cf",
        )

    def _fit_its_only(
        self, current_week: int,
    ) -> Optional[Tier3Estimate]:
        """Pure ITS on the factual series alone.

        Delegates to ``experiments.analysis.its.its_slope_only``. The
        slope is interpreted as the per-week change in turnover rate;
        this is a floor estimator for the case where no valid control
        or pre-period is available.
        """
        try:
            result = its_slope_only(np.asarray(self.factual_history))
        except ValueError:
            return None

        return Tier3Estimate(
            week=current_week,
            effect_estimate=result.slope,
            ci_lower=result.ci_lower,
            ci_upper=result.ci_upper,
            p_value=result.p_value,
            n_observations=result.n_observations,
            mode="its_only",
        )
