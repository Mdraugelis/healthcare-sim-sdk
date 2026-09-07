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

from ..model_verification import score_diagnostics
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
    # None retains ground truth in oracle mode and withholds it in its_only.
    include_counterfactual_ground_truth: bool | None = None

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

        oracle = self.tier3_mode == "cits_with_cf"
        include_cf = (
            oracle if self.include_counterfactual_ground_truth is None
            else self.include_counterfactual_ground_truth
        )
        # In observable-only mode this attribute is never accessed. This
        # also supports factual-only result objects with no oracle payload.
        cf_outcomes = results.counterfactual_outcomes if oracle or include_cf else {}
        if oracle and set(cf_outcomes) != set(results.outcomes):
            raise ValueError("oracle monitoring requires matching counterfactual weeks")
        run.metadata.update({
            "monitoring_mode": self.tier3_mode,
            "uses_counterfactual_for_estimation": oracle,
            "includes_counterfactual_ground_truth": include_cf,
            "tier3_estimand": (
                "factual_minus_counterfactual_rate_at_endpoint" if oracle
                else "factual_turnover_time_trend_not_causal_program_effect"
            ),
            "capacity_model": "separate_additive_soc_and_ai_budgets",
            "tier4_label_provenance": "realized_future_factual_departures",
            "prediction_performance": [],
            "missing_adoption_weeks": [],
        })

        # Initialize tiers. Tier 1 monitors operational leading
        # indicators:
        #   - check_ins_done: raw count of check-ins per week
        #     (detects capacity collapse — drops from ~60 to ~20)
        #   - adoption_rate: did managers actually use the tool?
        # We use raw count rather than adherence ratio because the
        # adherence ratio divides by the current capacity target,
        # which changes with the capacity itself — so adherence
        # stays at 100% even when capacity collapses. The raw count
        # has a stable baseline from the first 8 weeks and detects
        # any drop via WE1/WE2.
        tier1 = Tier1Shewhart()
        tier1.add_metric("check_ins_done", baseline_weeks=8)
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
        previous_cf_total = 0
        previous_cf_active = config.n_nurses

        for t in sorted(results.outcomes.keys()):
            f_out = results.outcomes[t]

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

            # step() realizes departures at week 0, so its denominator
            # must be the initial population rather than survivors.
            n_active_now = int(f_out.metadata.get("n_active", 0))
            if previous_active is None:
                previous_active = int(f_out.metadata.get("n_nurses", config.n_nurses))

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

            # Preserve fit/synthetic-score diagnostics, without presenting
            # synthetic prediction labels as subsequently observed departures.
            if t in results.predictions:
                pred = results.predictions[t]
                run.metadata["prediction_performance"].append({
                    "score_week": int(t),
                    **pred.metadata.get("model_performance", {}),
                    "actual_future_outcomes_status": "NOT_RUN",
                    "actual_future_outcomes_reason": (
                        "prediction horizon not yet observed"
                    ),
                })

            # Outcomes for a score made at week p mature after the full
            # prediction horizon (p+1 through p+h), after intervening care.
            for p in results.predictions:
                pred = results.predictions[p]
                horizon = int(pred.metadata.get(
                    "prediction_horizon_weeks", config.prediction_interval,
                ))
                if p + horizon != t:
                    continue
                future_weeks = range(p + 1, t + 1)
                if not all(w in results.outcomes for w in future_weeks):
                    continue
                scores_arr = np.asarray(pred.scores)
                active_mask = pred.metadata.get("active_mask")
                if active_mask is None:
                    active_mask = results.outcomes[p].secondary["active_mask"]
                active_mask = np.asarray(active_mask, dtype=bool)
                labels = np.maximum.reduce([
                    np.asarray(results.outcomes[w].events) for w in future_weeks
                ]).astype(int)
                run.tier4_prediction_log.append({
                    "week": int(t), "score_week": int(p),
                    "scores": scores_arr[active_mask],
                    "true_labels": labels[active_mask],
                    "label_provenance": "realized_future_factual_departures",
                })
                diagnostic = score_diagnostics(
                    labels[active_mask], scores_arr[active_mask],
                    resolve(config.model_auc, p),
                    tolerance=config.model_auc_tolerance,
                    min_positives=config.model_validation_min_positives,
                    min_negatives=config.model_validation_min_negatives,
                    label_provenance="realized_future_factual_departures",
                )
                for record in run.metadata["prediction_performance"]:
                    if record["score_week"] == p:
                        record["actual_future_outcomes"] = diagnostic
                        record["actual_future_outcomes_status"] = diagnostic["status"]
                        record["actual_future_outcomes_reason"] = diagnostic["reason"]
                        record["outcomes_available_week"] = int(t)
                        record["actual_outcome_limitation"] = (
                            "Post-intervention factual outcomes; measures the deployed "
                            "program's prediction performance, not untreated risk."
                        )

            # Tier 1: update the Shewhart charts
            tier1_metrics = {
                ("check_ins_done", None): float(
                    row.check_ins_done_this_week,
                ),
            }
            if row.adoption_rate is not None:
                tier1_metrics[("adoption_rate", None)] = row.adoption_rate
            else:
                run.metadata["missing_adoption_weeks"].append(int(t))
            tier1_events = tier1.update(t, tier1_metrics)
            for e in tier1_events:
                run.add_event(e)

            # Tier 2: update CUSUM
            tier2_events = tier2.update(t, row.unit_turnover_rate)
            for e in tier2_events:
                run.add_event(e)

            # Read oracle data only when that explicit estimator is selected.
            cf_turnover_rate = None
            if oracle:
                cf_out = cf_outcomes[t]
                cf_total = int(cf_out.metadata.get("total_departures", 0))
                cf_turnover_rate = (
                    (cf_total - previous_cf_total) / previous_cf_active
                    if previous_cf_active > 0 else 0.0
                )
                previous_cf_total = cf_total
                previous_cf_active = int(cf_out.metadata.get("n_active", 0))

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
        factual_retention = np.array(
            [
                results.outcomes[t].metadata.get("retention_rate", 0.0)
                for t in sorted(results.outcomes.keys())
            ],
            dtype=float,
        )
        run.ground_truth_factual_departures = factual_departures
        run.ground_truth_factual_retention = factual_retention
        if include_cf:
            run.ground_truth_counterfactual_departures = np.array([
                cf_outcomes[t].metadata.get("total_departures", 0)
                for t in sorted(cf_outcomes)
            ], dtype=float)
            run.ground_truth_counterfactual_retention = np.array([
                cf_outcomes[t].metadata.get("retention_rate", 0.0)
                for t in sorted(cf_outcomes)
            ], dtype=float)

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

        n_managers = int(f_out.metadata.get(
            "n_managers", max(1, config.n_nurses // config.nurses_per_manager),
        ))
        # The scenario currently provides separate SOC and AI budgets.
        # Include both potential budgets on prediction weeks; adoption is
        # measured separately. This prevents adherence above one by design.
        check_in_target = capacity_this_week * n_managers * (
            2 if f_interventions is not None else 1
        )
        adherence = (
            total_check_ins / check_in_target
            if check_in_target > 0
            else 0.0
        )

        n_adopting = f_out.metadata.get("n_managers_adopting")
        adoption_rate = (
            float(n_adopting) / n_managers if n_adopting is not None else None
        )

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
            n_managers_adopting=n_adopting,
            n_managers_total=n_managers,
            adoption_rate=adoption_rate,
            per_manager=[],
        )
