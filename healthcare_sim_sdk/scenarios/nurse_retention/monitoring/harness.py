"""MonitoringHarness — orchestrates the four monitoring tiers.

The harness is a post-hoc consumer: give it a completed
``BranchedSimulationResults`` from the nurse retention scenario and
it replays the weekly data through Tiers 1, 2, 3 (online) and Tier 4
(offline, at the end).

This keeps the harness completely decoupled from the engine. The
simulation runs normally via ``BranchedSimulationEngine.run()``, and
then the harness walks the results to produce a ``MonitoringRun``.
There is no modification to the engine, the scenario contract, or
the RNG streams — the harness is a pure consumer of the factual
branch's outcomes.

The **dashboard ignorance** invariant is enforced by how data flows:

- The tiers only see ``WeeklyDashboardRow`` objects from the
  ``aggregator``
- The aggregator only accepts factual state and factual outcomes
- The ground truth from the counterfactual branch is stored on the
  ``MonitoringRun`` as separate attributes but is never passed into
  any tier's ``update()`` method

The one deliberate exception is Tier 3's ``cits_with_cf`` mode, which
uses the counterfactual branch as a "synthetic control series." This
is the oracle mode for best-case performance measurement. The
``its_only`` mode makes no reference to the counterfactual and is
the floor/realistic mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from healthcare_sim_sdk.scenarios.nurse_retention.scenario import (
    RetentionConfig,
)
from healthcare_sim_sdk.scenarios.nurse_retention.time_varying import (
    resolve,
)

from .aggregator import WeeklyDashboardRow, aggregate
from .events import MonitoringRun
from .tier1_shewhart import Tier1Shewhart
from .tier2_cusum import Tier2CUSUM
from .tier3_cits import Tier3CITS
from .tier4_model_drift import analyze_tier4


@dataclass
class MonitoringHarness:
    """Runs the four tiers over a completed simulation result.

    Parameters
    ----------
    regime : str
        Name of the regime being evaluated (informational).
    seed : int
        Seed used for the underlying simulation (informational).
    tier3_mode : str
        ``"cits_with_cf"`` (oracle) or ``"its_only"`` (floor).
    """

    regime: str = "unnamed"
    seed: int = 0
    tier3_mode: str = "cits_with_cf"

    def run_from_results(
        self,
        results: Any,
        config: RetentionConfig,
    ) -> MonitoringRun:
        """Replay a completed simulation through the four tiers.

        Parameters
        ----------
        results : BranchedSimulationResults
            A completed simulation result from
            ``BranchedSimulationEngine.run()``.
        config : RetentionConfig
            The scenario config (used to resolve time-varying
            parameters for the capacity targets).

        Returns
        -------
        MonitoringRun
            Full harness output: weekly history, detection events
            from all tiers, Tier 3 quarterly estimates, ground
            truth trajectories, Tier 4 rolling statistics.
        """
        n_weeks = len(results.outcomes)

        run = MonitoringRun(
            regime=self.regime,
            seed=self.seed,
            n_weeks=n_weeks,
        )

        # Initialize tiers. Tier 1 monitors operational leading
        # indicators that should NOT have a structural time trend
        # in the calibrated regime:
        #   - check_in_adherence: did managers do their check-ins?
        #   - adoption_rate: did managers actually use the tool?
        # We deliberately do NOT include new_hire_fraction because
        # it has a natural downward drift (new hires age into
        # established nurses at week 52). A proper "new-hire 30-day
        # contact rate" would be a better leading indicator but
        # requires per-manager state.
        tier1 = Tier1Shewhart()
        tier1.add_metric("check_in_adherence", baseline_weeks=8)
        tier1.add_metric("adoption_rate", baseline_weeks=8)

        tier2 = Tier2CUSUM(
            metric="unit_turnover_rate",
            baseline_weeks=8,
            k_multiplier=0.5,
            h_multiplier=4.0,
        )

        tier3 = Tier3CITS(mode=self.tier3_mode)

        # Walk weeks
        previous_active = None
        previous_cum_departures = 0

        for t in sorted(results.outcomes.keys()):
            f_out = results.outcomes[t]
            cf_out = results.counterfactual_outcomes[t]

            # Interventions only exist when predict() ran
            f_interventions = results.interventions.get(t)

            # SOC interventions: delta from previous
            f_soc_total = int(
                f_out.metadata.get("total_soc_interventions", 0)
            )

            # We track SOC as a delta across weeks
            prev_soc = (
                run.weekly_history[-1].get(
                    "_cumulative_soc_interventions", 0,
                )
                if run.weekly_history
                else 0
            )
            soc_this_week = f_soc_total - prev_soc

            # At week 0, previous_active is None; use current active
            # count (departures this week should be 0 anyway).
            n_active_now = int(f_out.metadata.get("n_active", 0))
            if previous_active is None:
                previous_active = n_active_now

            capacity_this_week = int(
                resolve(
                    config.max_interventions_per_manager_per_week, t,
                )
            )

            # Aggregate the factual-branch data for the tiers.
            # NOTE: we pull state from the engine's stored state, not
            # from any counterfactual path. results.outcomes[t] is
            # factual-branch only.
            # Some engines may not store state per step; the aggregator
            # needs the state to compute per-manager rows. If the
            # result object doesn't carry state, we build a lightweight
            # surrogate row from outcomes metadata alone.
            if (
                hasattr(results, "factual_states")
                and results.factual_states is not None
                and t in results.factual_states
            ):
                state = results.factual_states[t]
                row = aggregate(
                    state=state,
                    outcomes=f_out,
                    interventions=f_interventions,
                    soc_interventions_this_week=soc_this_week,
                    capacity_this_week=capacity_this_week,
                    previous_cumulative_departures=(
                        previous_cum_departures
                    ),
                    previous_active=previous_active,
                )
            else:
                # Fallback: build a row from outcomes metadata only.
                # Lose per-manager detail in exchange for not needing
                # stored state. Sufficient for Tier 1 (population
                # metrics) and Tier 2 (turnover rate).
                row = self._row_from_metadata_only(
                    t=t,
                    f_out=f_out,
                    f_interventions=f_interventions,
                    soc_this_week=soc_this_week,
                    capacity_this_week=capacity_this_week,
                    previous_cum_departures=previous_cum_departures,
                    previous_active=previous_active,
                    config=config,
                )

            # Capture prediction data for Tier 4 (offline analysis)
            if t in results.predictions:
                pred = results.predictions[t]
                scores_arr = np.asarray(pred.scores)
                true_labels = np.asarray(
                    pred.metadata.get(
                        "true_labels", np.zeros_like(scores_arr),
                    )
                )
                run.tier4_prediction_log.append(
                    {
                        "week": int(t),
                        "scores": scores_arr,
                        "true_labels": true_labels,
                    }
                )

            # Tier 1: update the Shewhart charts
            tier1_metrics = {
                ("check_in_adherence", None): row.check_in_adherence,
                ("adoption_rate", None): row.adoption_rate,
            }
            tier1_events = tier1.update(t, tier1_metrics)
            for e in tier1_events:
                run.add_event(e)

            # Tier 2: update CUSUM
            tier2_events = tier2.update(t, row.unit_turnover_rate)
            for e in tier2_events:
                run.add_event(e)

            # Tier 3: update rolling CITS
            # The CF turnover rate is the ground-truth counterfactual
            # series, used as the comparison for cits_with_cf mode.
            cf_total_depart = int(
                cf_out.metadata.get("total_departures", 0)
            )
            prev_cf_total = (
                run.weekly_history[-1].get(
                    "_cumulative_cf_departures", 0,
                )
                if run.weekly_history
                else 0
            )
            cf_delta = cf_total_depart - prev_cf_total
            cf_active = int(cf_out.metadata.get("n_active", 0))
            prev_cf_active = (
                run.weekly_history[-1].get(
                    "_previous_cf_active", cf_active,
                )
                if run.weekly_history
                else cf_active
            )
            cf_turnover_rate = (
                cf_delta / prev_cf_active
                if prev_cf_active > 0 else 0.0
            )

            tier3_events = tier3.update(
                week=t,
                factual_turnover=row.unit_turnover_rate,
                counterfactual_turnover=cf_turnover_rate,
            )
            for e in tier3_events:
                run.add_event(e)

            # Store weekly history. We attach a couple of private
            # keys used for delta computation across weeks.
            row_dict = row.to_dict()
            row_dict["_cumulative_soc_interventions"] = f_soc_total
            row_dict["_cumulative_cf_departures"] = cf_total_depart
            row_dict["_previous_cf_active"] = cf_active
            run.weekly_history.append(row_dict)

            previous_active = n_active_now
            previous_cum_departures = row.cumulative_departures

        # Persist Tier 3 estimates
        run.tier3_estimates.extend(tier3.estimates)

        # Ground truth trajectories (for validation analysis, never
        # fed back into the tiers)
        factual_departures = np.array(
            [
                results.outcomes[t].metadata.get("total_departures", 0)
                for t in sorted(results.outcomes.keys())
            ],
            dtype=float,
        )
        cf_departures = np.array(
            [
                results.counterfactual_outcomes[t].metadata.get(
                    "total_departures", 0,
                )
                for t in sorted(results.counterfactual_outcomes.keys())
            ],
            dtype=float,
        )
        factual_retention = np.array(
            [
                results.outcomes[t].metadata.get("retention_rate", 0.0)
                for t in sorted(results.outcomes.keys())
            ],
            dtype=float,
        )
        cf_retention = np.array(
            [
                results.counterfactual_outcomes[t].metadata.get(
                    "retention_rate", 0.0,
                )
                for t in sorted(results.counterfactual_outcomes.keys())
            ],
            dtype=float,
        )

        run.ground_truth_factual_departures = factual_departures
        run.ground_truth_counterfactual_departures = cf_departures
        run.ground_truth_factual_retention = factual_retention
        run.ground_truth_counterfactual_retention = cf_retention

        # Tier 4: run offline analysis on collected prediction data
        tier4_result = analyze_tier4(run.tier4_prediction_log)
        run.metadata["tier4_rolling_stats"] = [
            s.to_dict() for s in tier4_result.rolling_stats
        ]
        for e in tier4_result.detection_events:
            run.add_event(e)

        return run

    def _row_from_metadata_only(
        self,
        t: int,
        f_out,
        f_interventions,
        soc_this_week: int,
        capacity_this_week: int,
        previous_cum_departures: int,
        previous_active: int,
        config: RetentionConfig,
    ) -> WeeklyDashboardRow:
        """Build a dashboard row when per-step state isn't available.

        Uses only the Outcomes metadata, which is always populated.
        Loses per-manager detail; sufficient for population-level
        monitoring (Tiers 1, 2, 3).
        """
        n_active = int(f_out.metadata.get("n_active", 0))
        total_departures = int(f_out.metadata.get("total_departures", 0))
        n_new_hire_active = int(
            f_out.metadata.get("n_new_hire_active", 0)
        )
        retention_rate = float(f_out.metadata.get("retention_rate", 0.0))

        n_departures_this_week = (
            total_departures - previous_cum_departures
        )
        unit_turnover_rate = (
            n_departures_this_week / previous_active
            if previous_active > 0
            else 0.0
        )

        ai_check_ins = (
            f_interventions.metadata.get("n_treated", 0)
            if f_interventions is not None
            else 0
        )
        total_check_ins = ai_check_ins + soc_this_week

        # Approximate n_managers from config
        n_managers = max(1, config.n_nurses // config.nurses_per_manager)
        check_in_target = capacity_this_week * n_managers
        adherence = (
            total_check_ins / check_in_target
            if check_in_target > 0
            else 0.0
        )

        # Without per-manager state, assume all managers adopt
        # (this is only inaccurate for Regime F — if we don't have
        # state, the harness caller is responsible for providing
        # state via results.factual_states)
        adoption_rate = 1.0

        return WeeklyDashboardRow(
            week=t,
            n_active=n_active,
            n_departures_this_week=n_departures_this_week,
            unit_turnover_rate=unit_turnover_rate,
            cumulative_departures=total_departures,
            retention_rate=retention_rate,
            check_ins_done_this_week=total_check_ins,
            check_in_target_this_week=check_in_target,
            check_in_adherence=adherence,
            n_new_hire_active=n_new_hire_active,
            new_hire_fraction=(
                n_new_hire_active / n_active if n_active > 0 else 0.0
            ),
            n_managers_adopting=n_managers,
            n_managers_total=n_managers,
            adoption_rate=adoption_rate,
            per_manager=[],
        )
