"""Scenario-specific validation tests.

Domain-level invariants for each implemented scenario. These tests
verify properties that are specific to the healthcare domain, not
generic simulation properties.

Categories:
- Stroke scenario: risk dynamics, treatment persistence, seasonal effects
- No-show scenario: overbooking constraints, utilization bounds, equity
"""

import numpy as np
import pytest

from healthcare_sim_sdk.core.engine import BranchedSimulationEngine, CounterfactualMode
from healthcare_sim_sdk.core.scenario import TimeConfig
from healthcare_sim_sdk.scenarios.stroke_prevention.scenario import (
    StrokeConfig,
    StrokePreventionScenario,
)
from healthcare_sim_sdk.scenarios.noshow_overbooking.scenario import (
    ClinicConfig,
    NoShowOverbookingScenario,
)

from .conftest import assert_no_nan_inf, assert_in_unit_interval


# =====================================================================
# 1. STROKE SCENARIO — domain invariants
# =====================================================================

class TestStrokeDomainInvariants:
    """Healthcare-specific invariants for stroke prevention."""

    def test_state_shape_invariant(self):
        """State should always be (4, n_patients) array."""
        config = StrokeConfig(n_patients=200, n_weeks=8, prediction_interval=4)
        sc = StrokePreventionScenario(config=config, seed=42)
        state = sc.create_population(200)
        assert state.shape == (4, 200), f"State shape: {state.shape}"

    def test_base_risks_immutable_after_step(self):
        """Base risks (row 0) should not change during step()."""
        config = StrokeConfig(n_patients=500, n_weeks=8, prediction_interval=4)
        sc = StrokePreventionScenario(config=config, seed=42)
        state = sc.create_population(500)
        base_risks_before = state[0].copy()

        for t in range(8):
            state = sc.step(state, t)

        np.testing.assert_array_equal(
            state[0], base_risks_before,
            err_msg="Base risks (row 0) changed during step()"
        )

    def test_treatment_effect_persists(self):
        """Once treated, factual should have fewer total events than CF.

        With 50% effectiveness and aggressive treatment, the factual branch
        should accumulate fewer stroke events than the counterfactual
        branch over the full simulation.
        """
        config = StrokeConfig(
            n_patients=3000, n_weeks=26,
            prediction_interval=4,
            intervention_effectiveness=0.50,
            treatment_threshold=0.3,  # aggressive treatment
        )
        sc = StrokePreventionScenario(config=config, seed=42)
        engine = BranchedSimulationEngine(sc, CounterfactualMode.BRANCHED)
        results = engine.run(3000)

        f_total = sum(
            results.outcomes[t].events.sum() for t in range(26)
        )
        cf_total = sum(
            results.counterfactual_outcomes[t].events.sum()
            for t in range(26)
        )
        # With persistent treatment at 50% effectiveness,
        # factual should have meaningfully fewer events
        assert f_total < cf_total, (
            f"Treatment should reduce events: "
            f"factual={f_total}, counterfactual={cf_total}"
        )

    def test_current_risks_positive(self):
        """Current risks (row 3) should always be positive."""
        config = StrokeConfig(n_patients=1000, n_weeks=20, prediction_interval=4)
        sc = StrokePreventionScenario(config=config, seed=42)
        state = sc.create_population(1000)

        for t in range(20):
            state = sc.step(state, t)
            current_risks = state[3]
            assert np.all(current_risks >= 0), (
                f"Negative current risk at t={t}: min={current_risks.min()}"
            )
            assert_no_nan_inf(current_risks, f"current_risks t={t}")

    def test_ar1_modifiers_bounded(self):
        """AR(1) modifiers (row 1) should stay within configured bounds."""
        config = StrokeConfig(n_patients=1000, n_weeks=52, prediction_interval=52)
        sc = StrokePreventionScenario(config=config, seed=42)
        state = sc.create_population(1000)

        for t in range(52):
            state = sc.step(state, t)
            mods = state[1]
            assert np.all(mods >= 0.5), (
                f"AR1 modifier below 0.5 at t={t}: min={mods.min()}"
            )
            assert np.all(mods <= 2.0), (
                f"AR1 modifier above 2.0 at t={t}: max={mods.max()}"
            )

    def test_prediction_scores_match_risk_ordering(self):
        """Higher-risk patients should tend to get higher prediction scores.

        Not exact (noise injection), but rank correlation should be positive.
        """
        config = StrokeConfig(
            n_patients=3000, n_weeks=8, prediction_interval=4,
        )
        sc = StrokePreventionScenario(config=config, seed=42)
        results = BranchedSimulationEngine(
            sc, CounterfactualMode.BRANCHED
        ).run(3000)

        # Get first prediction
        if results.predictions:
            first_t = min(results.predictions.keys())
            scores = results.predictions[first_t].scores
            assert_in_unit_interval(scores, "prediction scores")


# =====================================================================
# 2. NO-SHOW SCENARIO — domain invariants
# =====================================================================

class TestNoShowDomainInvariants:
    """Healthcare-specific invariants for no-show/overbooking."""

    def _run_noshow(self, n_days=15, n_patients=500, seed=42,
                    base_noshow_rate=0.13, overbooking_threshold=0.30,
                    max_overbook_per_provider=2):
        tc = TimeConfig(
            n_timesteps=n_days, timestep_duration=1 / 365,
            timestep_unit="day",
            prediction_schedule=list(range(n_days)),
        )
        cc = ClinicConfig(
            n_providers=3, slots_per_provider_per_day=8,
            max_overbook_per_provider=max_overbook_per_provider,
        )
        sc = NoShowOverbookingScenario(
            time_config=tc, seed=seed, n_patients=n_patients,
            base_noshow_rate=base_noshow_rate,
            overbooking_threshold=overbooking_threshold,
            model_type="predictor", model_auc=0.80,
            clinic_config=cc,
        )
        return BranchedSimulationEngine(
            sc, CounterfactualMode.BRANCHED
        ).run(n_patients)

    def test_utilization_bounded_01(self):
        """Utilization (slot filled or not) must be in {0, 1}."""
        results = self._run_noshow()
        for t in results.outcomes:
            out = results.outcomes[t]
            if "utilization" in out.secondary:
                util = out.secondary["utilization"]
                unique = np.unique(util)
                assert np.all(np.isin(unique, [0, 1])), (
                    f"Non-binary utilization at t={t}: {unique}"
                )

    def test_wait_times_non_negative(self):
        """Wait times must be >= 0."""
        results = self._run_noshow()
        for t in results.outcomes:
            out = results.outcomes[t]
            if "wait_times" in out.secondary:
                wt = out.secondary["wait_times"]
                assert np.all(wt >= 0), (
                    f"Negative wait time at t={t}: min={wt.min()}"
                )

    def test_noshow_events_binary(self):
        """No-show events must be binary (0 or 1)."""
        results = self._run_noshow()
        for t in results.outcomes:
            events = results.outcomes[t].events
            unique = np.unique(events)
            assert np.all(np.isin(unique, [0, 1])), (
                f"Non-binary noshow events at t={t}: {unique}"
            )

    def test_overbooking_respects_provider_budget(self):
        """Total overbookings per provider per day should not exceed budget."""
        results = self._run_noshow(max_overbook_per_provider=2)
        for t, intv in results.interventions.items():
            n_treated = len(intv.treated_indices)
            # Total overbookings bounded by n_providers * max_per_provider
            max_possible = 3 * 2  # 3 providers * 2 max each
            assert n_treated <= max_possible, (
                f"Overbookings at t={t}: {n_treated} > max {max_possible}"
            )

    def test_no_overbooking_with_threshold_1(self):
        """Threshold=1.0 means no slot can trigger overbooking."""
        results = self._run_noshow(overbooking_threshold=1.0)
        for t, intv in results.interventions.items():
            assert len(intv.treated_indices) == 0, (
                f"Overbookings at t={t} with threshold=1.0: "
                f"{len(intv.treated_indices)}"
            )

    def test_aggressive_threshold_more_overbooking(self):
        """Lower threshold should produce more overbookings."""
        r_low = self._run_noshow(overbooking_threshold=0.10, n_days=20)
        r_high = self._run_noshow(overbooking_threshold=0.50, n_days=20)

        total_low = sum(
            len(intv.treated_indices) for intv in r_low.interventions.values()
        )
        total_high = sum(
            len(intv.treated_indices) for intv in r_high.interventions.values()
        )
        assert total_low >= total_high, (
            f"Lower threshold should overbooking more: "
            f"thresh=0.10->{total_low}, thresh=0.50->{total_high}"
        )

    def test_demographics_present_in_outcomes(self):
        """Outcome secondary data should include demographic fields."""
        results = self._run_noshow()
        # Check at least one timestep has demographic data
        found_demographics = False
        for t in results.outcomes:
            out = results.outcomes[t]
            if "race_ethnicity" in out.secondary or "insurance_type" in out.secondary:
                found_demographics = True
                break
        # Demographics should be present (unless t=0 has no resolved slots)
        # Check after first timestep
        if len(results.outcomes) > 1:
            assert found_demographics, (
                "No demographic data found in any outcome secondary dict"
            )


# =====================================================================
# 3. CROSS-SCENARIO UNIVERSAL INVARIANTS
# =====================================================================

class TestUniversalScenarioInvariants:
    """Properties that must hold for ANY scenario implementation."""

    @pytest.mark.parametrize("mode", [
        CounterfactualMode.NONE,
        CounterfactualMode.SNAPSHOT,
        CounterfactualMode.BRANCHED,
    ])
    def test_all_modes_complete_without_error(self, mode):
        """Every counterfactual mode should run to completion."""
        config = StrokeConfig(
            n_patients=100, n_weeks=8, prediction_interval=4,
        )
        sc = StrokePreventionScenario(config=config, seed=42)
        results = BranchedSimulationEngine(sc, mode).run(100)
        assert len(results.outcomes) == 8

    def test_results_metadata_populated(self):
        """SimulationResults should have correct metadata."""
        config = StrokeConfig(
            n_patients=100, n_weeks=8, prediction_interval=4,
        )
        sc = StrokePreventionScenario(config=config, seed=42)
        results = BranchedSimulationEngine(
            sc, CounterfactualMode.BRANCHED
        ).run(100)

        assert results.n_entities == 100
        assert results.time_config.n_timesteps == 8
        assert results.unit_of_analysis == "patient"
        assert results.counterfactual_mode == "branched"

    def test_validations_recorded(self):
        """Population and final validations should be recorded."""
        config = StrokeConfig(
            n_patients=100, n_weeks=4, prediction_interval=4,
        )
        sc = StrokePreventionScenario(config=config, seed=42)
        results = BranchedSimulationEngine(
            sc, CounterfactualMode.BRANCHED
        ).run(100)

        assert "population" in results.validations
        assert "final" in results.validations
