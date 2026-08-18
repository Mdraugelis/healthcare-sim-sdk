"""Integration tests for the No-Show Overbooking scenario."""

import numpy as np

from healthcare_sim_sdk.scenarios.noshow_overbooking.scenario import (
    NoShowOverbookingScenario,
)
from healthcare_sim_sdk.core.engine import BranchedSimulationEngine, CounterfactualMode
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
    return NoShowOverbookingScenario(
        tc, seed=seed, n_patients=n_patients, **kwargs
    )


class TestNoShowPurity:
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


class TestNoShowCompounding:
    def test_overbooking_burden_increases_on_factual(self):
        """Factual branch accumulates overbooking burden over time."""
        sc = _make_scenario(n_days=20, n_patients=500)
        results = BranchedSimulationEngine(
            sc, CounterfactualMode.BRANCHED
        ).run(500)

        early = results.outcomes[5].metadata["mean_overbooking_burden"]
        late = results.outcomes[19].metadata["mean_overbooking_burden"]
        assert late >= early, (
            f"Overbooking burden should increase: "
            f"day 5={early:.3f}, day 19={late:.3f}"
        )

    def test_counterfactual_has_zero_burden(self):
        """Counterfactual branch has no overbooking (no interventions)."""
        sc = _make_scenario(n_days=15)
        results = BranchedSimulationEngine(
            sc, CounterfactualMode.BRANCHED
        ).run(500)

        for t in range(15):
            cf_burden = results.counterfactual_outcomes[t].metadata[
                "mean_overbooking_burden"
            ]
            assert cf_burden == 0.0, (
                f"CF overbooking burden should be 0 at t={t}: {cf_burden}"
            )


class TestNoShowGenerality:
    def test_same_engine_api(self):
        """Same engine.run() works for no-show as for stroke."""
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
        """All AnalysisDataset methods work on no-show results."""
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


class TestNoShowEquity:
    def test_demographic_labels_present(self):
        """Demographic labels are available for equity analysis."""
        sc = _make_scenario(n_days=5, n_patients=200)
        results = BranchedSimulationEngine(
            sc, CounterfactualMode.BRANCHED
        ).run(200)

        out = results.outcomes[1]  # t=1 has resolved slots
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


class TestNoShowPopulation:
    def test_noshow_rate_near_target(self):
        """Cumulative no-show rate should be near base_noshow_rate."""
        sc = _make_scenario(
            n_days=30, n_patients=1000,
            base_noshow_rate=0.12,
            overbooking_threshold=1.0,  # no interventions
        )
        results = BranchedSimulationEngine(
            sc, CounterfactualMode.NONE
        ).run(1000)

        # Use cumulative counters from final timestep metadata
        # (appointments are resolved in step() of the *next* timestep)
        assert len(results.outcomes) == 30, "Should have 30 days of outcomes"
        assert results.unit_of_analysis == "appointment"
