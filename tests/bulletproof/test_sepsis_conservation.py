"""Conservation law and invariant tests for the Sepsis Early Alert scenario.

Tests encode properties that must hold regardless of parameters:
- Population conservation (stage counts always sum to n_patients)
- No NaN/Inf in state arrays
- No backward stage transitions
- Monotonicity invariants
- Boundary conditions
- Treatment pathway validity
"""

import numpy as np

from healthcare_sim_sdk.core.engine import BranchedSimulationEngine, CounterfactualMode
from healthcare_sim_sdk.core.scenario import TimeConfig
from healthcare_sim_sdk.scenarios.sepsis_early_alert.scenario import (
    STAGE_DECEASED,
    STAGE_DISCHARGED,
    SepsisConfig,
    SepsisEarlyAlertScenario,
)


def _make_sepsis(n_patients=500, n_timesteps=18, seed=42, **cfg_overrides):
    """Helper to create a sepsis scenario and run it."""
    tc = TimeConfig(
        n_timesteps=n_timesteps,
        timestep_duration=4 / 8760,
        timestep_unit="4hr_block",
        prediction_schedule=list(range(n_timesteps)),
    )
    cfg = SepsisConfig(n_patients=n_patients, **cfg_overrides)
    sc = SepsisEarlyAlertScenario(time_config=tc, seed=seed, config=cfg)
    engine = BranchedSimulationEngine(sc, CounterfactualMode.BRANCHED)
    return engine.run(n_patients), n_patients, n_timesteps


# =====================================================================
# 1. POPULATION CONSERVATION
# =====================================================================

class TestSepsisPopulationConservation:
    """Stage counts must always sum to n_patients."""

    def test_stage_sum_equals_n(self):
        """For every t, sum(stage_counts) == n_patients."""
        results, n, n_t = _make_sepsis()
        for t in range(n_t):
            counts = results.outcomes[t].metadata["stage_counts"]
            total = sum(counts.values())
            assert total == n, (
                f"Stage sum {total} != {n} at t={t}: {counts}"
            )

    def test_stage_sum_counterfactual(self):
        """Stage conservation holds on CF branch too."""
        results, n, n_t = _make_sepsis()
        for t in range(n_t):
            counts = results.counterfactual_outcomes[t].metadata["stage_counts"]
            total = sum(counts.values())
            assert total == n, (
                f"CF stage sum {total} != {n} at t={t}: {counts}"
            )

    def test_no_nan_inf_in_state(self):
        """State arrays must never contain NaN or Inf."""
        tc = TimeConfig(
            n_timesteps=18,
            timestep_duration=4 / 8760,
            timestep_unit="4hr_block",
            prediction_schedule=list(range(18)),
        )
        cfg = SepsisConfig(n_patients=500, sepsis_incidence=0.04)
        sc = SepsisEarlyAlertScenario(time_config=tc, seed=42, config=cfg)
        engine = BranchedSimulationEngine(sc, CounterfactualMode.BRANCHED)
        results = engine.run(500)

        # Check outcomes at several timesteps
        for t in [0, 5, 10, 17]:
            mortality = results.outcomes[t].secondary["mortality"]
            assert not np.any(np.isnan(mortality)), f"NaN in mortality at t={t}"
            assert not np.any(np.isinf(mortality)), f"Inf in mortality at t={t}"

            stage = results.outcomes[t].secondary["stage"]
            assert not np.any(np.isnan(stage)), f"NaN in stage at t={t}"
            assert not np.any(np.isinf(stage)), f"Inf in stage at t={t}"

    def test_no_backward_stage_transitions(self):
        """Patients should never regress to a lower disease stage.

        Stage progression is one-way: at_risk -> sepsis -> severe -> shock
        -> deceased. Discharged is also terminal. No patient should ever
        go from a higher numeric stage back to a lower one.
        """
        results, n, n_t = _make_sepsis()

        # Track each patient's max stage seen so far
        max_stage = np.zeros(n)
        for t in range(n_t):
            stage = results.outcomes[t].secondary["stage"]
            # Check: no patient's stage is lower than their previous max,
            # UNLESS they were discharged (stage 5) — discharged is terminal
            active = (max_stage < STAGE_DECEASED) & (max_stage < STAGE_DISCHARGED)
            violations = active & (stage < max_stage)
            assert not np.any(violations), (
                f"Backward transition at t={t}: "
                f"patients {np.where(violations)[0][:5]} went from "
                f"stage {max_stage[violations][:5]} to {stage[violations][:5]}"
            )
            max_stage = np.maximum(max_stage, stage)


# =====================================================================
# 2. MONOTONICITY
# =====================================================================

class TestSepsisMonotonicity:
    """Higher effectiveness -> fewer deaths, etc."""

    def test_higher_effectiveness_fewer_deaths(self):
        """Increasing treatment effectiveness should reduce mortality."""
        death_counts = {}
        for eff in [0.0, 0.25, 0.50]:
            results, n, n_t = _make_sepsis(
                n_patients=2000,
                n_timesteps=42,
                treatment_effectiveness=eff,
                sepsis_incidence=0.04,
            )
            deaths = results.outcomes[41].secondary["mortality"].sum()
            death_counts[eff] = deaths

        effs = sorted(death_counts.keys())
        for i in range(1, len(effs)):
            assert death_counts[effs[i]] <= death_counts[effs[i - 1]] * 1.10, (
                f"Monotonicity violation: "
                f"eff={effs[i-1]}->{death_counts[effs[i-1]]}, "
                f"eff={effs[i]}->{death_counts[effs[i]]}"
            )

    def test_capacity_zero_equals_counterfactual(self):
        """With capacity=0, no alerts are responded to, so factual ~ CF."""
        results, n, _ = _make_sepsis(
            n_patients=1000,
            n_timesteps=42,
            rapid_response_capacity=0,
        )
        factual_deaths = results.outcomes[41].secondary["mortality"].sum()
        cf_deaths = (
            results.counterfactual_outcomes[41].secondary["mortality"].sum()
        )
        # Allow small tolerance due to stochastic effects
        assert abs(factual_deaths - cf_deaths) <= max(5, 0.1 * cf_deaths), (
            f"Capacity=0 should yield factual ~ CF: "
            f"factual={factual_deaths}, cf={cf_deaths}"
        )


# =====================================================================
# 3. BOUNDARY CONDITIONS
# =====================================================================

class TestSepsisBoundaryConditions:
    """Extreme parameter values must produce predictable behavior."""

    def test_threshold_100_flags_nobody(self):
        """alert_threshold_percentile=100 -> 0 flags -> factual ~ CF."""
        results, n, _ = _make_sepsis(
            n_patients=1000,
            n_timesteps=42,
            alert_threshold_percentile=100.0,
        )
        factual_deaths = results.outcomes[41].secondary["mortality"].sum()
        cf_deaths = (
            results.counterfactual_outcomes[41].secondary["mortality"].sum()
        )
        assert abs(factual_deaths - cf_deaths) <= max(5, 0.1 * cf_deaths), (
            f"Threshold=100 should yield factual ~ CF: "
            f"factual={factual_deaths}, cf={cf_deaths}"
        )

    def test_effectiveness_zero_no_benefit(self):
        """treatment_effectiveness=0 -> factual mortality ~ CF mortality."""
        results, n, _ = _make_sepsis(
            n_patients=1000,
            n_timesteps=42,
            treatment_effectiveness=0.0,
        )
        factual_deaths = results.outcomes[41].secondary["mortality"].sum()
        cf_deaths = (
            results.counterfactual_outcomes[41].secondary["mortality"].sum()
        )
        assert abs(factual_deaths - cf_deaths) <= max(5, 0.1 * cf_deaths), (
            f"Effectiveness=0 should yield factual ~ CF: "
            f"factual={factual_deaths}, cf={cf_deaths}"
        )


# =====================================================================
# 4. TREATMENT PATHWAY VALIDITY
# =====================================================================

class TestSepsisTreatmentPathway:
    """Treatment indices must be valid and consistent."""

    def test_treated_indices_valid_range(self):
        """All treated indices must be in [0, n_patients)."""
        results, n, n_t = _make_sepsis()
        for t, intv in results.interventions.items():
            indices = intv.treated_indices
            if len(indices) > 0:
                assert np.all(indices >= 0), (
                    f"Negative index at t={t}: {indices[indices < 0]}"
                )
                assert np.all(indices < n), (
                    f"Index >= n at t={t}: {indices[indices >= n]}"
                )

    def test_treated_indices_unique_per_timestep(self):
        """No duplicate treated indices per timestep."""
        results, n, n_t = _make_sepsis()
        for t, intv in results.interventions.items():
            indices = intv.treated_indices
            assert len(indices) == len(np.unique(indices)), (
                f"Duplicate treated indices at t={t}: "
                f"{len(indices)} total, {len(np.unique(indices))} unique"
            )

    def test_capacity_respected(self):
        """Number of treated per timestep <= rapid_response_capacity."""
        capacity = 8
        results, n, n_t = _make_sepsis(rapid_response_capacity=capacity)
        for t, intv in results.interventions.items():
            assert len(intv.treated_indices) <= capacity, (
                f"Capacity violated at t={t}: "
                f"{len(intv.treated_indices)} > {capacity}"
            )


# =====================================================================
# 5. KUMAR DECAY CONSERVATION
# =====================================================================

class TestSepsisKumarConservation:
    """Conservation laws hold with Kumar time-dependent effectiveness."""

    def test_stage_sum_with_kumar(self):
        """Population conservation holds with Kumar decay enabled."""
        results, n, n_t = _make_sepsis(
            kumar_half_life_hours=6.0,
            max_treatment_effectiveness=0.50,
        )
        for t in range(n_t):
            counts = results.outcomes[t].metadata["stage_counts"]
            total = sum(counts.values())
            assert total == n, f"Stage sum {total} != {n} at t={t}"

    def test_no_nan_inf_with_kumar(self):
        """No NaN/Inf in outcomes with Kumar decay enabled."""
        results, n, n_t = _make_sepsis(
            kumar_half_life_hours=6.0,
            max_treatment_effectiveness=0.50,
        )
        for t in [0, 5, 10, 17]:
            mortality = results.outcomes[t].secondary["mortality"]
            assert not np.any(np.isnan(mortality)), f"NaN at t={t}"
            assert not np.any(np.isinf(mortality)), f"Inf at t={t}"

    def test_no_backward_transitions_with_kumar(self):
        """No backward stage transitions with Kumar decay enabled."""
        results, n, n_t = _make_sepsis(
            kumar_half_life_hours=6.0,
            max_treatment_effectiveness=0.50,
        )
        max_stage = np.zeros(n)
        for t in range(n_t):
            stage = results.outcomes[t].secondary["stage"]
            active = (
                (max_stage < STAGE_DECEASED) & (max_stage < STAGE_DISCHARGED)
            )
            violations = active & (stage < max_stage)
            assert not np.any(violations), (
                f"Backward transition at t={t} with Kumar decay"
            )
            max_stage = np.maximum(max_stage, stage)


# =====================================================================
# 6. BASELINE CLINICAL DETECTION CONSERVATION
# =====================================================================

class TestBaselineDetectionConservation:
    """Conservation laws hold with baseline clinical detection."""

    def test_stage_sum_with_baseline(self):
        """Population conservation holds with baseline detection."""
        results, n, n_t = _make_sepsis(
            baseline_detection_enabled=True,
        )
        for t in range(n_t):
            counts = results.outcomes[t].metadata["stage_counts"]
            total = sum(counts.values())
            assert total == n, f"Stage sum {total} != {n} at t={t}"

    def test_no_nan_inf_with_baseline_and_kumar(self):
        """No NaN/Inf with both baseline detection and Kumar decay."""
        results, n, n_t = _make_sepsis(
            baseline_detection_enabled=True,
            kumar_half_life_hours=6.0,
            max_treatment_effectiveness=0.50,
        )
        for t in [0, 5, 10, 17]:
            mortality = results.outcomes[t].secondary["mortality"]
            assert not np.any(np.isnan(mortality)), f"NaN at t={t}"
            assert not np.any(np.isinf(mortality)), f"Inf at t={t}"
            onset = results.outcomes[t].secondary["onset_timestep"]
            assert not np.any(np.isnan(onset)), f"NaN in onset at t={t}"

    def test_no_backward_transitions_with_baseline(self):
        """No backward stage transitions with baseline detection."""
        results, n, n_t = _make_sepsis(
            baseline_detection_enabled=True,
        )
        max_stage = np.zeros(n)
        for t in range(n_t):
            stage = results.outcomes[t].secondary["stage"]
            active = (
                (max_stage < STAGE_DECEASED) & (max_stage < STAGE_DISCHARGED)
            )
            violations = active & (stage < max_stage)
            assert not np.any(violations), (
                f"Backward transition at t={t} with baseline detection"
            )
            max_stage = np.maximum(max_stage, stage)

    def test_baseline_disabled_matches_no_cf_treatment(self):
        """With baseline disabled, CF branch has no treated patients."""
        results, n, n_t = _make_sepsis(
            baseline_detection_enabled=False,
        )
        for t in range(n_t):
            cf_treated = (
                results.counterfactual_outcomes[t].secondary["treated"]
            )
            assert cf_treated.sum() == 0, (
                f"CF has treated patients at t={t} with baseline disabled"
            )
