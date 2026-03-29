"""Numerical stability tests: NaN, Inf, overflow, underflow, degenerate inputs.

These tests ensure the simulation doesn't silently produce garbage when
given extreme, degenerate, or adversarial inputs. Every computation path
should either produce valid numbers or raise a clear error.
"""

import numpy as np
import pytest
import warnings

from healthcare_sim_sdk.core.engine import BranchedSimulationEngine, CounterfactualMode
from healthcare_sim_sdk.core.scenario import TimeConfig
from healthcare_sim_sdk.ml.model import ControlledMLModel
from healthcare_sim_sdk.ml.performance import (
    auc_score,
    confusion_matrix_metrics,
    calibration_slope,
    hosmer_lemeshow_test,
    theoretical_ppv,
)
from healthcare_sim_sdk.population.risk_distributions import beta_distributed_risks
from healthcare_sim_sdk.population.temporal_dynamics import (
    annual_risk_to_hazard,
    hazard_to_timestep_probability,
    AR1Process,
)
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
# 1. HAZARD CONVERSION EDGE CASES
# =====================================================================

class TestHazardNumericalStability:
    """Hazard conversions must not produce NaN/Inf at extremes."""

    def test_risk_near_1_does_not_produce_inf(self):
        """annual_risk close to 1.0 should produce large but finite hazard."""
        risks = np.array([0.99, 0.999, 0.9999, 0.99999])
        hazards = annual_risk_to_hazard(risks)
        assert_no_nan_inf(hazards, "hazard from near-1 risk")
        assert np.all(hazards > 0), "Hazards should be positive"
        assert np.all(np.isfinite(hazards)), "Hazards should be finite"

    def test_risk_exactly_0(self):
        """annual_risk = 0 -> hazard = 0, prob = 0."""
        h = annual_risk_to_hazard(np.array([0.0]))
        assert h[0] == 0.0
        p = hazard_to_timestep_probability(h, 1 / 52)
        assert p[0] == 0.0

    def test_risk_exactly_1_clipped(self):
        """annual_risk = 1.0 should be clipped to avoid log(0)."""
        h = annual_risk_to_hazard(np.array([1.0]))
        assert_no_nan_inf(h, "hazard from risk=1.0")
        assert np.all(np.isfinite(h))

    def test_negative_risk_clipped(self):
        """Negative risks should be clipped to 0."""
        h = annual_risk_to_hazard(np.array([-0.1, -1.0, -100.0]))
        assert_no_nan_inf(h, "hazard from negative risk")
        assert np.all(h >= 0)

    def test_very_small_risk_no_underflow(self):
        """Very small risks should not underflow to exactly 0 hazard."""
        tiny_risks = np.array([1e-10, 1e-15, 1e-20])
        hazards = annual_risk_to_hazard(tiny_risks)
        probs = hazard_to_timestep_probability(hazards, 1 / 52)
        assert_no_nan_inf(hazards, "tiny risk hazards")
        assert_no_nan_inf(probs, "tiny risk probs")

    def test_very_large_hazard_prob_saturates_at_1(self):
        """Huge hazard rates should give probability ~ 1, not > 1."""
        huge_hazards = np.array([100.0, 1000.0, 1e6])
        probs = hazard_to_timestep_probability(huge_hazards, 1.0)
        assert np.all(probs <= 1.0), f"Prob > 1: {probs}"
        assert np.all(probs >= 0.99), f"Prob should be ~1: {probs}"


# =====================================================================
# 2. ML MODEL NUMERICAL STABILITY
# =====================================================================

class TestMLModelNumericalStability:
    """ML model score generation must not produce NaN/Inf."""

    def test_scores_no_nan_inf_extreme_params(self):
        """Extreme noise parameters should not produce NaN/Inf scores."""
        rng = np.random.default_rng(42)
        n = 1000
        risks = rng.random(n)
        labels = (rng.random(n) > 0.5).astype(int)
        model = ControlledMLModel(mode="discrimination")

        extreme_params = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 10.0, 0.0),
            (0.99, 0.001, 5.0),
            (0.5, 5.0, 5.0),
        ]
        for corr, scale, lns in extreme_params:
            scores = model._generate_scores(
                risks, rng, labels, corr, scale, lns
            )
            assert_no_nan_inf(
                scores,
                f"scores(corr={corr}, scale={scale}, lns={lns})"
            )
            assert_in_unit_interval(
                scores,
                f"scores(corr={corr}, scale={scale}, lns={lns})"
            )

    def test_scores_with_constant_risks(self):
        """All-identical risk inputs should not crash or produce NaN."""
        rng = np.random.default_rng(42)
        n = 1000
        for const in [0.0, 0.5, 1.0]:
            risks = np.full(n, const)
            labels = (rng.random(n) > 0.5).astype(int)
            model = ControlledMLModel(mode="discrimination")
            scores = model._generate_scores(
                risks, rng, labels, 0.8, 0.1, 1.0
            )
            assert_no_nan_inf(scores, f"constant risk={const}")
            assert_in_unit_interval(scores, f"constant risk={const}")

    def test_scores_with_single_class(self):
        """All-positive or all-negative labels should not crash."""
        rng = np.random.default_rng(42)
        n = 1000
        risks = rng.random(n)

        for label_val in [0, 1]:
            labels = np.full(n, label_val)
            model = ControlledMLModel(mode="discrimination")
            scores = model._generate_scores(
                risks, rng, labels, 0.8, 0.1, 1.0
            )
            assert_no_nan_inf(scores, f"single class={label_val}")
            assert_in_unit_interval(scores, f"single class={label_val}")

    def test_auc_with_edge_cases(self):
        """AUC computation should handle degenerate inputs gracefully."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_scores = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        auc = auc_score(y_true, y_scores)
        assert np.isfinite(auc), f"AUC is not finite: {auc}"

        y_true_all0 = np.zeros(100)
        y_scores_rand = np.random.default_rng(42).random(100)
        auc = auc_score(y_true_all0, y_scores_rand)
        assert np.isfinite(auc)

    def test_confusion_matrix_all_zeros(self):
        """All-zero predictions should not produce NaN metrics."""
        y_true = np.array([0, 0, 1, 1, 0])
        y_pred = np.zeros(5)
        m = confusion_matrix_metrics(y_true, y_pred, threshold=0.5)
        assert np.isfinite(m["sensitivity"])
        assert np.isfinite(m["specificity"])
        assert np.isfinite(m["ppv"])

    def test_confusion_matrix_all_ones(self):
        """All-one predictions should not produce NaN metrics."""
        y_true = np.array([0, 0, 1, 1, 0])
        y_pred = np.ones(5)
        m = confusion_matrix_metrics(y_true, y_pred, threshold=0.5)
        assert np.isfinite(m["sensitivity"])
        assert np.isfinite(m["ppv"])


# =====================================================================
# 3. CALIBRATION & HOSMER-LEMESHOW EDGE CASES
# =====================================================================

class TestCalibrationNumericalStability:
    """Calibration metrics must handle edge cases without NaN/Inf."""

    def test_calibration_slope_constant_predictions(self):
        """Constant predictions -> slope should default gracefully."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 0])
        y_pred = np.full(10, 0.3)
        slope, pred_means, obs_means = calibration_slope(y_true, y_pred)
        assert np.isfinite(slope), f"Slope not finite: {slope}"

    def test_calibration_slope_perfect_predictions(self):
        """Perfect predictions should give slope ~ 1.0."""
        rng = np.random.default_rng(42)
        n = 1000
        y_true = (rng.random(n) > 0.7).astype(float)
        y_pred = y_true.copy()
        slope, _, _ = calibration_slope(y_true, y_pred)
        assert np.isfinite(slope)

    def test_hosmer_lemeshow_small_sample(self):
        """HL test should not crash with small samples."""
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0.2, 0.8, 0.3, 0.7, 0.1])
        hl, p = hosmer_lemeshow_test(y_true, y_pred, n_bins=3)
        assert np.isfinite(hl), f"HL stat not finite: {hl}"
        assert np.isfinite(p), f"p-value not finite: {p}"

    def test_theoretical_ppv_zero_denominator(self):
        """PPV with zero denominator should return 0, not NaN."""
        ppv = theoretical_ppv(prevalence=0.0, sensitivity=0.8, specificity=1.0)
        assert np.isfinite(ppv), f"PPV not finite: {ppv}"


# =====================================================================
# 4. SCENARIO-LEVEL NUMERICAL STABILITY
# =====================================================================

class TestScenarioNumericalStability:
    """Full scenarios must not produce NaN/Inf in any output."""

    def test_stroke_no_nan_inf_in_outcomes(self):
        """All stroke scenario outcomes should be finite."""
        config = StrokeConfig(
            n_patients=1000, n_weeks=26,
            prediction_interval=4,
        )
        sc = StrokePreventionScenario(config=config, seed=42)
        results = BranchedSimulationEngine(
            sc, CounterfactualMode.BRANCHED
        ).run(1000)

        for t in range(26):
            assert_no_nan_inf(
                results.outcomes[t].events,
                f"factual events t={t}"
            )
            assert_no_nan_inf(
                results.counterfactual_outcomes[t].events,
                f"counterfactual events t={t}"
            )

        for t, pred in results.predictions.items():
            assert_no_nan_inf(pred.scores, f"prediction scores t={t}")

    def test_stroke_no_nan_in_analysis_exports(self):
        """Analysis dataset exports should have no NaN/Inf."""
        config = StrokeConfig(
            n_patients=500, n_weeks=12,
            prediction_interval=4,
        )
        sc = StrokePreventionScenario(config=config, seed=42)
        results = BranchedSimulationEngine(
            sc, CounterfactualMode.BRANCHED
        ).run(500)
        analysis = results.to_analysis()

        ts = analysis.to_time_series()
        assert_no_nan_inf(ts["outcomes"], "time series outcomes")
        assert_no_nan_inf(ts["treatment_indicator"], "treatment indicator")

        panel = analysis.to_panel()
        assert_no_nan_inf(panel["outcomes"], "panel outcomes")
        assert_no_nan_inf(panel["treated"], "panel treated")

    def test_noshow_no_nan_inf_in_outcomes(self):
        """All no-show scenario outcomes should be finite."""
        tc = TimeConfig(
            n_timesteps=10, timestep_duration=1 / 365,
            timestep_unit="day",
            prediction_schedule=list(range(10)),
        )
        cc = ClinicConfig(n_providers=3, slots_per_provider_per_day=8,
                          max_overbook_per_provider=2)
        sc = NoShowOverbookingScenario(
            time_config=tc, seed=42, n_patients=300,
            base_noshow_rate=0.13, overbooking_threshold=0.30,
            model_type="predictor", model_auc=0.80,
            clinic_config=cc,
        )
        results = BranchedSimulationEngine(
            sc, CounterfactualMode.BRANCHED
        ).run(300)

        for t in range(10):
            assert_no_nan_inf(
                results.outcomes[t].events,
                f"noshow factual events t={t}"
            )

    def test_single_entity_simulation(self):
        """Simulation with n=1 should not crash or produce NaN."""
        config = StrokeConfig(
            n_patients=1, n_weeks=12,
            prediction_interval=4,
            annual_incident_rate=0.30,
        )
        sc = StrokePreventionScenario(config=config, seed=42)
        results = BranchedSimulationEngine(
            sc, CounterfactualMode.BRANCHED
        ).run(1)

        for t in range(12):
            assert results.outcomes[t].events.shape == (1,)
            assert_no_nan_inf(results.outcomes[t].events, f"n=1, t={t}")

    def test_very_short_simulation(self):
        """Single-timestep simulation should work."""
        config = StrokeConfig(
            n_patients=100, n_weeks=1,
            prediction_interval=1,
        )
        sc = StrokePreventionScenario(config=config, seed=42)
        results = BranchedSimulationEngine(
            sc, CounterfactualMode.BRANCHED
        ).run(100)

        assert 0 in results.outcomes
        assert_no_nan_inf(results.outcomes[0].events, "single step")


# =====================================================================
# 5. AR(1) PROCESS EDGE CASES
# =====================================================================

class TestAR1NumericalStability:
    """AR(1) process must stay finite under stress."""

    def test_high_sigma_stays_bounded(self):
        """Very high innovation sigma should be caught by bounds clipping."""
        rng = np.random.default_rng(42)
        ar1 = AR1Process(
            n_entities=1000, rho=0.9, sigma=10.0,
            mu=1.0, bounds=(0.5, 2.0),
        )
        for _ in range(100):
            vals = ar1.step(rng)
            assert_no_nan_inf(vals, "high sigma AR1")
            assert np.all(vals >= 0.5)
            assert np.all(vals <= 2.0)

    def test_rho_1_no_drift(self):
        """rho=1.0 (random walk) should still stay bounded."""
        rng = np.random.default_rng(42)
        ar1 = AR1Process(
            n_entities=1000, rho=1.0, sigma=0.1,
            mu=1.0, bounds=(0.1, 5.0),
        )
        for _ in range(500):
            vals = ar1.step(rng)
            assert_no_nan_inf(vals, "rho=1 AR1")
            assert np.all(vals >= 0.1)
            assert np.all(vals <= 5.0)

    def test_rho_0_iid(self):
        """rho=0 should produce i.i.d. draws (no memory)."""
        rng = np.random.default_rng(42)
        ar1 = AR1Process(
            n_entities=1000, rho=0.0, sigma=0.1,
            mu=1.0, bounds=(-10, 10),
        )
        for _ in range(50):
            vals = ar1.step(rng)
            assert_no_nan_inf(vals, "rho=0 AR1")
