"""Integration tests for the No-Show Targeted Reminder scenario."""

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


def _make_scenario(
    n_days=15, n_patients=500, seed=42, **kwargs
):
    tc = TimeConfig(
        n_timesteps=n_days,
        timestep_duration=1 / 365,
        timestep_unit="day",
        prediction_schedule=list(range(n_days)),
    )
    defaults = dict(
        base_noshow_rate=0.20,
        model_auc=0.74,
        caller_config=CallerConfig(
            n_providers=3,
            slots_per_provider_per_day=8,
            call_capacity_per_day=10,
            call_success_rate=0.65,
            reminder_effectiveness=0.35,
        ),
    )
    defaults.update(kwargs)
    return NoShowTargetedReminderScenario(
        tc, seed=seed, n_patients=n_patients, **defaults
    )


class TestReminderPurity:
    def test_branched_vs_none_identity(self):
        """Factual branch must be identical in BRANCHED and NONE modes."""
        sc_b = _make_scenario()
        res_b = BranchedSimulationEngine(
            sc_b, CounterfactualMode.BRANCHED
        ).run(500)

        sc_n = _make_scenario()
        res_n = BranchedSimulationEngine(
            sc_n, CounterfactualMode.NONE
        ).run(500)

        for t in range(15):
            np.testing.assert_array_equal(
                res_b.outcomes[t].events,
                res_n.outcomes[t].events,
                err_msg=f"Purity violation at t={t}",
            )


class TestReminderEffect:
    def test_factual_fewer_noshows_than_counterfactual(self):
        """Factual branch should have fewer no-shows than CF."""
        sc = _make_scenario(
            n_days=30, n_patients=1000,
            caller_config=CallerConfig(
                n_providers=4,
                slots_per_provider_per_day=10,
                call_capacity_per_day=15,
                call_success_rate=0.80,
                reminder_effectiveness=0.50,
            ),
        )
        results = BranchedSimulationEngine(
            sc, CounterfactualMode.BRANCHED
        ).run(1000)

        f_total = sum(
            results.outcomes[t].events.sum()
            for t in range(30)
        )
        cf_total = sum(
            results.counterfactual_outcomes[t].events.sum()
            for t in range(30)
        )
        assert f_total < cf_total, (
            f"Reminders should reduce no-shows: "
            f"factual={f_total}, counterfactual={cf_total}"
        )

    def test_counterfactual_has_zero_calls(self):
        """Counterfactual branch has no calls (no interventions)."""
        sc = _make_scenario(n_days=15)
        results = BranchedSimulationEngine(
            sc, CounterfactualMode.BRANCHED
        ).run(500)

        for t in range(15):
            cf_calls = results.counterfactual_outcomes[t].metadata[
                "total_calls_made"
            ]
            assert cf_calls == 0, (
                f"CF total_calls_made should be 0 at t={t}: {cf_calls}"
            )


class TestReminderGenerality:
    def test_same_engine_api(self):
        """Same engine.run() works for reminders as for other scenarios."""
        sc = _make_scenario(n_days=10, n_patients=200)
        engine = BranchedSimulationEngine(
            sc, CounterfactualMode.BRANCHED
        )
        results = engine.run(200)

        assert results.counterfactual_mode == "branched"
        assert results.unit_of_analysis == "appointment"
        assert len(results.outcomes) == 10
        assert len(results.counterfactual_outcomes) == 10

    def test_analysis_exports(self):
        """All AnalysisDataset methods work on reminder results."""
        sc = _make_scenario(n_days=10, n_patients=200)
        results = BranchedSimulationEngine(
            sc, CounterfactualMode.BRANCHED
        ).run(200)

        analysis = results.to_analysis()

        ts = analysis.to_time_series()
        assert len(ts["outcomes"]) == 10

        panel = analysis.to_panel()
        assert panel["unit_of_analysis"] == "appointment"

        snap = analysis.to_entity_snapshots(t=0)
        assert snap["scores"] is not None


class TestReminderEquity:
    def test_demographic_labels_present(self):
        """Demographic labels are available for equity analysis."""
        sc = _make_scenario(n_days=5, n_patients=200)
        results = BranchedSimulationEngine(
            sc, CounterfactualMode.BRANCHED
        ).run(200)

        out = results.outcomes[1]
        assert "race_ethnicity" in out.secondary
        assert "insurance_type" in out.secondary
        assert "age_band" in out.secondary
        races = set(out.secondary["race_ethnicity"])
        assert len(races) > 1, "Should have multiple race/ethnicity groups"

    def test_subgroup_panel_export(self):
        """to_subgroup_panel works with demographic data."""
        sc = _make_scenario(n_days=5, n_patients=200)
        results = BranchedSimulationEngine(
            sc, CounterfactualMode.BRANCHED
        ).run(200)

        panel = results.to_analysis().to_subgroup_panel(
            subgroup_key="race_ethnicity"
        )
        assert "subgroup" in panel
        unique = set(panel["subgroup"])
        assert "White" in unique or "Black" in unique

    def test_call_metadata_present(self):
        """Call tracking metadata present in outcomes."""
        sc = _make_scenario(n_days=5, n_patients=200)
        results = BranchedSimulationEngine(
            sc, CounterfactualMode.BRANCHED
        ).run(200)

        out = results.outcomes[1]
        assert "was_called" in out.secondary
        assert "was_reached" in out.secondary
        assert "total_calls_made" in out.metadata
        assert "total_calls_reached" in out.metadata
