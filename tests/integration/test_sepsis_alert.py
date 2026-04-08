"""Integration tests for the Sepsis Early Alert scenario."""

import numpy as np

from healthcare_sim_sdk.core.engine import BranchedSimulationEngine, CounterfactualMode
from healthcare_sim_sdk.core.scenario import TimeConfig
from healthcare_sim_sdk.scenarios.sepsis_early_alert.scenario import (
    SepsisConfig,
    SepsisEarlyAlertScenario,
)
from tests.integration.test_step_purity import assert_purity


class TestSepsisPurity:
    def test_branched_vs_none_identity(self):
        """The critical purity test on the real sepsis scenario.

        step() must be a pure function of (state, t, self.rng.temporal).
        The _cumulative_false_alerts counter is updated in intervene()
        (factual-only) but step() never reads it, so purity holds.
        """

        class _SepsisForPurity(SepsisEarlyAlertScenario):
            """Wrapper to match assert_purity interface."""
            def __init__(self, time_config, seed=None):
                cfg = SepsisConfig(n_patients=500, sepsis_incidence=0.04)
                super().__init__(time_config=time_config, seed=seed, config=cfg)

        assert_purity(
            _SepsisForPurity,
            n_entities=500,
            n_timesteps=18,
            prediction_schedule=list(range(18)),
        )

    def test_purity_with_baseline_detection(self):
        """Purity holds with baseline clinical detection enabled.

        Baseline detection in step() is purely deterministic (no RNG),
        so it cannot desynchronize branches.
        """

        class _SepsisBaselinePurity(SepsisEarlyAlertScenario):
            def __init__(self, time_config, seed=None):
                cfg = SepsisConfig(
                    n_patients=500, sepsis_incidence=0.04,
                    baseline_detection_enabled=True,
                )
                super().__init__(
                    time_config=time_config, seed=seed, config=cfg,
                )

        assert_purity(
            _SepsisBaselinePurity,
            n_entities=500,
            n_timesteps=18,
            prediction_schedule=list(range(18)),
        )


class TestSepsisIntervention:
    def test_treatment_reduces_progression(self):
        """Factual should have fewer deaths than counterfactual."""
        tc = TimeConfig(
            n_timesteps=42,
            timestep_duration=4 / 8760,
            timestep_unit="4hr_block",
            prediction_schedule=list(range(42)),
        )
        cfg = SepsisConfig(
            n_patients=2000,
            sepsis_incidence=0.04,
            treatment_effectiveness=0.35,
            rapid_response_capacity=8,
        )
        sc = SepsisEarlyAlertScenario(time_config=tc, seed=42, config=cfg)
        engine = BranchedSimulationEngine(sc, CounterfactualMode.BRANCHED)
        results = engine.run(2000)

        factual_deaths = results.outcomes[41].secondary["mortality"].sum()
        cf_deaths = results.counterfactual_outcomes[41].secondary["mortality"].sum()
        assert factual_deaths < cf_deaths, (
            f"Treatment should reduce deaths: "
            f"factual={factual_deaths}, counterfactual={cf_deaths}"
        )

    def test_cf_has_baseline_treated(self):
        """CF branch should have patients treated via baseline detection."""
        tc = TimeConfig(
            n_timesteps=42,
            timestep_duration=4 / 8760,
            timestep_unit="4hr_block",
            prediction_schedule=list(range(42)),
        )
        cfg = SepsisConfig(
            n_patients=2000, sepsis_incidence=0.04,
            baseline_detection_enabled=True,
        )
        sc = SepsisEarlyAlertScenario(time_config=tc, seed=42, config=cfg)
        engine = BranchedSimulationEngine(sc, CounterfactualMode.BRANCHED)
        results = engine.run(2000)

        cf_treated = results.counterfactual_outcomes[41].secondary["treated"]
        assert cf_treated.sum() > 0, (
            "CF branch should have baseline-treated patients"
        )

    def test_cf_no_treated_when_baseline_disabled(self):
        """With baseline detection disabled, CF has no treated patients."""
        tc = TimeConfig(
            n_timesteps=18,
            timestep_duration=4 / 8760,
            timestep_unit="4hr_block",
            prediction_schedule=list(range(18)),
        )
        cfg = SepsisConfig(
            n_patients=500, sepsis_incidence=0.04,
            baseline_detection_enabled=False,
        )
        sc = SepsisEarlyAlertScenario(time_config=tc, seed=42, config=cfg)
        engine = BranchedSimulationEngine(sc, CounterfactualMode.BRANCHED)
        results = engine.run(500)

        for t in range(18):
            cf_treated = results.counterfactual_outcomes[t].secondary["treated"]
            assert cf_treated.sum() == 0, (
                f"CF branch has treated patients at t={t}: {cf_treated.sum()}"
            )

    def test_analysis_exports(self):
        """All AnalysisDataset methods work on sepsis results."""
        tc = TimeConfig(
            n_timesteps=12,
            timestep_duration=4 / 8760,
            timestep_unit="4hr_block",
            prediction_schedule=list(range(12)),
        )
        cfg = SepsisConfig(n_patients=200, sepsis_incidence=0.04)
        sc = SepsisEarlyAlertScenario(time_config=tc, seed=42, config=cfg)
        results = BranchedSimulationEngine(
            sc, CounterfactualMode.BRANCHED,
        ).run(200)

        analysis = results.to_analysis()

        ts = analysis.to_time_series()
        assert len(ts["outcomes"]) == 12

        panel = analysis.to_panel()
        assert len(panel["entity_ids"]) == 200 * 12
        assert panel["unit_of_analysis"] == "patient_admission"

        snap = analysis.to_entity_snapshots(t=0)
        assert len(snap["entity_ids"]) == 200
        assert snap["scores"] is not None


class TestSepsisEquity:
    def test_demographic_labels_present(self):
        """Check demographic labels exist in secondary outcomes."""
        tc = TimeConfig(
            n_timesteps=6,
            timestep_duration=4 / 8760,
            timestep_unit="4hr_block",
            prediction_schedule=list(range(6)),
        )
        cfg = SepsisConfig(n_patients=200, sepsis_incidence=0.04)
        sc = SepsisEarlyAlertScenario(time_config=tc, seed=42, config=cfg)
        results = BranchedSimulationEngine(
            sc, CounterfactualMode.BRANCHED,
        ).run(200)

        secondary = results.outcomes[0].secondary
        assert "race_ethnicity" in secondary
        assert "insurance_type" in secondary
        assert "age_band" in secondary
        assert len(secondary["race_ethnicity"]) == 200

    def test_subgroup_panel_export(self):
        """Subgroup panel export works with race_ethnicity key."""
        tc = TimeConfig(
            n_timesteps=6,
            timestep_duration=4 / 8760,
            timestep_unit="4hr_block",
            prediction_schedule=list(range(6)),
        )
        cfg = SepsisConfig(n_patients=500, sepsis_incidence=0.04)
        sc = SepsisEarlyAlertScenario(time_config=tc, seed=42, config=cfg)
        results = BranchedSimulationEngine(
            sc, CounterfactualMode.BRANCHED,
        ).run(500)

        analysis = results.to_analysis()
        subgroup = analysis.to_subgroup_panel(subgroup_key="race_ethnicity")
        # Should have multiple race groups
        unique_groups = set(subgroup["subgroup"])
        assert len(unique_groups) > 1


class TestSepsisPopulation:
    def test_stage_accounting(self):
        """Sum of patients across all 6 stages == n_patients at every t."""
        tc = TimeConfig(
            n_timesteps=18,
            timestep_duration=4 / 8760,
            timestep_unit="4hr_block",
            prediction_schedule=list(range(18)),
        )
        cfg = SepsisConfig(n_patients=500, sepsis_incidence=0.04)
        sc = SepsisEarlyAlertScenario(time_config=tc, seed=42, config=cfg)
        results = BranchedSimulationEngine(
            sc, CounterfactualMode.BRANCHED,
        ).run(500)

        for t in range(18):
            counts = results.outcomes[t].metadata["stage_counts"]
            total = sum(counts.values())
            assert total == 500, (
                f"Stage sum {total} != 500 at t={t}: {counts}"
            )


class TestSepsisKumarDecay:
    def test_purity_with_kumar(self):
        """Purity test with Kumar decay enabled."""

        class _SepsisKumarPurity(SepsisEarlyAlertScenario):
            def __init__(self, time_config, seed=None):
                cfg = SepsisConfig(
                    n_patients=500, sepsis_incidence=0.04,
                    kumar_half_life_hours=6.0,
                    max_treatment_effectiveness=0.50,
                )
                super().__init__(
                    time_config=time_config, seed=seed, config=cfg,
                )

        assert_purity(
            _SepsisKumarPurity,
            n_entities=500,
            n_timesteps=18,
            prediction_schedule=list(range(18)),
        )

    def test_kumar_mode_high_auc_saves_more(self):
        """Higher AUC should save more lives in Kumar mode.

        This is the core test: with time-dependent effectiveness,
        earlier detection (higher AUC) leads to earlier treatment,
        which has higher effectiveness via the Kumar decay curve.
        """
        lives_saved_by_auc = {}
        for auc in [0.63, 0.82]:
            tc = TimeConfig(
                n_timesteps=42,
                timestep_duration=4 / 8760,
                timestep_unit="4hr_block",
                prediction_schedule=list(range(42)),
            )
            cfg = SepsisConfig(
                n_patients=3000,
                model_auc=auc,
                kumar_half_life_hours=6.0,
                max_treatment_effectiveness=0.50,
                sepsis_incidence=0.04,
                rapid_response_capacity=50,
                alert_threshold_percentile=90.0,
                initial_response_rate=0.80,
            )
            sc = SepsisEarlyAlertScenario(
                time_config=tc, seed=42, config=cfg,
            )
            results = BranchedSimulationEngine(
                sc, CounterfactualMode.BRANCHED,
            ).run(3000)
            f_deaths = results.outcomes[41].secondary["mortality"].sum()
            cf_deaths = (
                results.counterfactual_outcomes[41]
                .secondary["mortality"].sum()
            )
            lives_saved_by_auc[auc] = cf_deaths - f_deaths

        assert lives_saved_by_auc[0.82] > lives_saved_by_auc[0.63], (
            f"Higher AUC should save more lives in Kumar mode: "
            f"AUC=0.63 saved {lives_saved_by_auc[0.63]}, "
            f"AUC=0.82 saved {lives_saved_by_auc[0.82]}"
        )

    def test_kumar_zero_equals_flat_mode(self):
        """kumar_half_life_hours=0 should behave identically to flat mode."""
        tc = TimeConfig(
            n_timesteps=18,
            timestep_duration=4 / 8760,
            timestep_unit="4hr_block",
            prediction_schedule=list(range(18)),
        )
        cfg_flat = SepsisConfig(
            n_patients=500, treatment_effectiveness=0.35,
        )
        cfg_kumar0 = SepsisConfig(
            n_patients=500, treatment_effectiveness=0.35,
            kumar_half_life_hours=0.0,
        )

        sc_flat = SepsisEarlyAlertScenario(
            time_config=tc, seed=42, config=cfg_flat,
        )
        sc_kumar = SepsisEarlyAlertScenario(
            time_config=tc, seed=42, config=cfg_kumar0,
        )

        r_flat = BranchedSimulationEngine(
            sc_flat, CounterfactualMode.BRANCHED,
        ).run(500)
        r_kumar = BranchedSimulationEngine(
            sc_kumar, CounterfactualMode.BRANCHED,
        ).run(500)

        for t in range(18):
            np.testing.assert_array_equal(
                r_flat.outcomes[t].secondary["mortality"],
                r_kumar.outcomes[t].secondary["mortality"],
                err_msg=f"Flat vs kumar_half_life=0 differ at t={t}",
            )
