"""Conservation law and boundary condition tests for targeted reminders."""

import numpy as np

from healthcare_sim_sdk.scenarios.noshow_targeted_reminders.scenario import (
    CallerConfig,
    NoShowTargetedReminderScenario,
)
from healthcare_sim_sdk.core.engine import (
    BranchedSimulationEngine,
    CounterfactualMode,
)
from healthcare_sim_sdk.core.scenario import TimeConfig


def _make_scenario(n_days=15, n_patients=500, seed=42, **kwargs):
    tc = TimeConfig(
        n_timesteps=n_days,
        timestep_duration=1 / 365,
        timestep_unit="day",
        prediction_schedule=list(range(n_days)),
    )
    defaults = dict(
        base_noshow_rate=0.20,
        model_auc=0.74,
    )
    defaults.update(kwargs)
    return NoShowTargetedReminderScenario(
        tc, seed=seed, n_patients=n_patients, **defaults
    )


def _run(sc, n_patients=500, mode=CounterfactualMode.BRANCHED):
    return BranchedSimulationEngine(sc, mode).run(n_patients)


class TestCapacityConservation:
    def test_calls_per_day_respects_capacity(self):
        """Calls made per day must never exceed call_capacity_per_day."""
        capacity = 8
        sc = _make_scenario(
            n_days=20, n_patients=500,
            caller_config=CallerConfig(
                call_capacity_per_day=capacity,
                call_success_rate=0.65,
                reminder_effectiveness=0.35,
            ),
        )
        results = _run(sc)

        for t, intv in results.interventions.items():
            n_called = len(intv.treated_indices)
            assert n_called <= capacity, (
                f"Calls at t={t}: {n_called} > capacity {capacity}"
            )

    def test_events_are_binary(self):
        """All outcome events must be 0.0 or 1.0."""
        sc = _make_scenario(n_days=10)
        results = _run(sc)

        for t in range(10):
            events = results.outcomes[t].events
            unique = np.unique(events)
            assert np.all(np.isin(unique, [0.0, 1.0])), (
                f"Non-binary events at t={t}: {unique}"
            )

    def test_no_nan_inf_in_predictions(self):
        """Predictions must contain no NaN or Inf."""
        sc = _make_scenario(n_days=10)
        results = _run(sc)

        for t, pred in results.predictions.items():
            assert not np.any(np.isnan(pred.scores)), (
                f"NaN in predictions at t={t}"
            )
            assert not np.any(np.isinf(pred.scores)), (
                f"Inf in predictions at t={t}"
            )
            assert np.all(pred.scores >= 0) and np.all(pred.scores <= 1), (
                f"Predictions out of [0,1] at t={t}"
            )

    def test_no_nan_inf_in_outcomes(self):
        """Outcomes must contain no NaN or Inf."""
        sc = _make_scenario(n_days=10)
        results = _run(sc)

        for t in range(10):
            events = results.outcomes[t].events
            assert not np.any(np.isnan(events)), (
                f"NaN in outcomes at t={t}"
            )
            assert not np.any(np.isinf(events)), (
                f"Inf in outcomes at t={t}"
            )


class TestBoundaryConditions:
    def test_zero_capacity_equals_counterfactual(self):
        """With zero call capacity, factual == counterfactual."""
        sc = _make_scenario(
            n_days=15,
            caller_config=CallerConfig(
                call_capacity_per_day=0,
                call_success_rate=0.65,
                reminder_effectiveness=0.35,
            ),
        )
        results = _run(sc)

        for t in range(15):
            np.testing.assert_array_equal(
                results.outcomes[t].events,
                results.counterfactual_outcomes[t].events,
                err_msg=f"Zero capacity: factual != CF at t={t}",
            )

    def test_zero_effectiveness_similar_to_counterfactual(self):
        """With zero effectiveness, factual ~= counterfactual."""
        sc = _make_scenario(
            n_days=20, n_patients=1000,
            caller_config=CallerConfig(
                call_capacity_per_day=20,
                call_success_rate=0.80,
                reminder_effectiveness=0.0,
            ),
        )
        results = _run(sc, n_patients=1000)

        f_total = sum(
            results.outcomes[t].events.sum() for t in range(20)
        )
        cf_total = sum(
            results.counterfactual_outcomes[t].events.sum()
            for t in range(20)
        )
        # With zero effectiveness, should be very close
        assert abs(f_total - cf_total) < 0.5 * max(f_total, cf_total, 1), (
            f"Zero effectiveness: factual={f_total}, CF={cf_total} "
            f"should be similar"
        )

    def test_zero_reach_rate_no_reached(self):
        """With call_success_rate=0, no patients should be reached."""
        sc = _make_scenario(
            n_days=15,
            caller_config=CallerConfig(
                call_capacity_per_day=10,
                call_success_rate=0.0,
                reminder_effectiveness=0.35,
            ),
        )
        results = _run(sc)

        last_t = max(results.outcomes.keys())
        total_reached = results.outcomes[last_t].metadata[
            "total_calls_reached"
        ]
        assert total_reached == 0, (
            f"Zero reach rate but total_calls_reached={total_reached}"
        )

    def test_perfect_reach_rate_all_reached(self):
        """With call_success_rate=1.0, all called == all reached."""
        sc = _make_scenario(
            n_days=15,
            caller_config=CallerConfig(
                call_capacity_per_day=10,
                call_success_rate=1.0,
                reminder_effectiveness=0.35,
            ),
        )
        results = _run(sc)

        last_t = max(results.outcomes.keys())
        meta = results.outcomes[last_t].metadata
        total_made = meta["total_calls_made"]
        total_reached = meta["total_calls_reached"]
        assert total_made == total_reached, (
            f"Perfect reach: calls_made={total_made} != "
            f"calls_reached={total_reached}"
        )


class TestMonotonicity:
    def test_higher_capacity_more_calls(self):
        """Higher call capacity should produce more total calls."""
        results_low = _run(_make_scenario(
            n_days=20,
            caller_config=CallerConfig(
                call_capacity_per_day=5,
                call_success_rate=0.65,
                reminder_effectiveness=0.35,
            ),
        ))
        results_high = _run(_make_scenario(
            n_days=20,
            caller_config=CallerConfig(
                call_capacity_per_day=20,
                call_success_rate=0.65,
                reminder_effectiveness=0.35,
            ),
        ))

        last_t = max(results_low.outcomes.keys())
        low_calls = results_low.outcomes[last_t].metadata["total_calls_made"]
        high_calls = results_high.outcomes[last_t].metadata[
            "total_calls_made"
        ]
        assert high_calls >= low_calls, (
            f"Higher capacity should mean more calls: "
            f"low={low_calls}, high={high_calls}"
        )

    def test_higher_effectiveness_fewer_noshows(self):
        """Higher reminder effectiveness should produce fewer no-shows."""
        results_low = _run(_make_scenario(
            n_days=30, n_patients=1000,
            caller_config=CallerConfig(
                n_providers=4,
                slots_per_provider_per_day=10,
                call_capacity_per_day=15,
                call_success_rate=0.80,
                reminder_effectiveness=0.10,
            ),
        ), n_patients=1000)
        results_high = _run(_make_scenario(
            n_days=30, n_patients=1000,
            caller_config=CallerConfig(
                n_providers=4,
                slots_per_provider_per_day=10,
                call_capacity_per_day=15,
                call_success_rate=0.80,
                reminder_effectiveness=0.80,
            ),
        ), n_patients=1000)

        low_noshows = sum(
            results_low.outcomes[t].events.sum() for t in range(30)
        )
        high_noshows = sum(
            results_high.outcomes[t].events.sum() for t in range(30)
        )
        assert high_noshows <= low_noshows, (
            f"Higher effectiveness should mean fewer no-shows: "
            f"low_eff={low_noshows}, high_eff={high_noshows}"
        )
