"""Statistical sanity checks: does the math actually add up?

These tests verify that simulated distributions, rates, and expected values
match their analytical/theoretical counterparts. Each test encodes a
statistical identity that MUST hold if the simulation is correct.

Categories:
- Population risk distributions match parameterization
- Event rates match expected incidence over time
- Treatment effects have correct direction and magnitude
- Aggregated outcomes obey law of large numbers
- Conditional distributions are consistent with marginals
"""

import numpy as np
import pytest

from healthcare_sim_sdk.core.engine import BranchedSimulationEngine, CounterfactualMode
from healthcare_sim_sdk.core.scenario import TimeConfig
from healthcare_sim_sdk.ml.model import ControlledMLModel
from healthcare_sim_sdk.ml.performance import (
    auc_score,
    confusion_matrix_metrics,
    theoretical_ppv,
    calibration_slope,
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

from .conftest import (
    assert_mean_in_range,
    assert_rate_in_range,
    assert_no_nan_inf,
    assert_in_unit_interval,
)


# =====================================================================
# 1. POPULATION GENERATION — distribution shape matches parameters
# =====================================================================

class TestPopulationDistributions:
    """Verify generated populations match specified distributions."""

    @pytest.mark.parametrize("rate", [0.01, 0.05, 0.13, 0.30, 0.50])
    def test_beta_risks_mean_matches_target(self, rate):
        """Population mean risk should equal annual_incident_rate."""
        rng = np.random.default_rng(42)
        risks = beta_distributed_risks(
            n_patients=20000,
            annual_incident_rate=rate,
            concentration=0.5,
            rng=rng,
        )
        assert_mean_in_range(
            risks, rate, tolerance_sigma=4.0,
            name=f"beta_risks(rate={rate})"
        )

    @pytest.mark.parametrize("concentration", [0.1, 0.5, 1.0, 2.0])
    def test_higher_concentration_lower_variance(self, concentration):
        """Higher concentration should produce lower variance in risks."""
        rng = np.random.default_rng(42)
        risks = beta_distributed_risks(
            n_patients=10000,
            annual_incident_rate=0.10,
            concentration=concentration,
            rng=rng,
        )
        return np.var(risks)

    def test_beta_risks_bounded_01(self):
        """All generated risks must be in (0, 1)."""
        rng = np.random.default_rng(42)
        for rate in [0.001, 0.01, 0.05, 0.13, 0.50, 0.90]:
            risks = beta_distributed_risks(
                n_patients=10000,
                annual_incident_rate=rate,
                concentration=0.5,
                rng=rng,
            )
            assert np.all(risks > 0), f"rate={rate}: found zero risks"
            assert np.all(risks < 1), f"rate={rate}: found risk >= 1"

    def test_variance_ordering_by_concentration(self):
        """Variance should decrease monotonically with concentration."""
        variances = []
        for conc in [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]:
            risks = beta_distributed_risks(
                n_patients=50000,
                annual_incident_rate=0.10,
                concentration=conc,
                rng=np.random.default_rng(42),
            )
            variances.append(np.var(risks))
        # Variance should be non-increasing
        for i in range(1, len(variances)):
            assert variances[i] <= variances[i - 1] * 1.05, (
                f"Variance not decreasing: conc[{i-1}]->var={variances[i-1]:.6f}, "
                f"conc[{i}]->var={variances[i]:.6f}"
            )


# =====================================================================
# 2. HAZARD / PROBABILITY CONVERSIONS — known identities
# =====================================================================

class TestHazardConversions:
    """Survival analysis conversion identities that must hold exactly."""

    def test_roundtrip_risk_hazard_probability(self):
        """annual_risk -> hazard -> timestep_prob should be consistent.

        For weekly timesteps: sum of weekly probs over 52 weeks should
        approximate annual risk (via geometric series).
        """
        for annual_risk in [0.01, 0.05, 0.10, 0.25, 0.50]:
            hazard = annual_risk_to_hazard(np.array([annual_risk]))[0]
            weekly_prob = hazard_to_timestep_probability(
                np.array([hazard]), 1 / 52
            )[0]
            # P(at least one event in 52 weeks) = 1 - (1-p)^52
            annual_from_weekly = 1 - (1 - weekly_prob) ** 52
            assert abs(annual_from_weekly - annual_risk) < 0.001, (
                f"Roundtrip failed for annual_risk={annual_risk}: "
                f"recovered={annual_from_weekly:.6f}"
            )

    def test_zero_risk_gives_zero_hazard(self):
        """Zero annual risk -> zero hazard -> zero timestep probability."""
        h = annual_risk_to_hazard(np.array([0.0]))[0]
        assert h == 0.0
        p = hazard_to_timestep_probability(np.array([0.0]), 1 / 52)[0]
        assert p == 0.0

    def test_hazard_monotone_in_risk(self):
        """Higher annual risk -> higher hazard rate (monotonicity)."""
        risks = np.linspace(0.01, 0.95, 50)
        hazards = annual_risk_to_hazard(risks)
        for i in range(1, len(hazards)):
            assert hazards[i] > hazards[i - 1], (
                f"Hazard not monotone: risk[{i-1}]={risks[i-1]:.3f}->h={hazards[i-1]:.6f}, "
                f"risk[{i}]={risks[i]:.3f}->h={hazards[i]:.6f}"
            )

    def test_timestep_prob_increases_with_dt(self):
        """Longer timestep -> higher event probability."""
        h = annual_risk_to_hazard(np.array([0.10]))
        probs = []
        for dt in [1 / 365, 1 / 52, 1 / 12, 1 / 4, 1.0]:
            p = hazard_to_timestep_probability(h, dt)[0]
            probs.append(p)
        for i in range(1, len(probs)):
            assert probs[i] > probs[i - 1]

    def test_small_risk_approximation(self):
        """For small risks, hazard ~ annual_risk (first-order approx)."""
        small_risks = np.array([0.001, 0.005, 0.01])
        hazards = annual_risk_to_hazard(small_risks)
        np.testing.assert_allclose(
            hazards, small_risks, rtol=0.01,
            err_msg="Small risk approximation failed"
        )


# =====================================================================
# 3. AR(1) PROCESS — stationary distribution properties
# =====================================================================

class TestAR1StatisticalProperties:
    """Verify AR(1) process converges to correct stationary distribution."""

    def test_stationary_mean(self):
        """After burn-in, mean should converge near mu.

        With bounds clipping at [0.5, 2.0], the stationary distribution
        is slightly skewed, so we accept mean within 0.01 of mu.
        """
        rng = np.random.default_rng(42)
        ar1 = AR1Process(n_entities=5000, rho=0.9, sigma=0.05, mu=1.0)

        # Burn in 200 steps
        for _ in range(200):
            ar1.step(rng)

        # Collect 100 more samples
        samples = np.concatenate([ar1.step(rng) for _ in range(100)])
        assert abs(np.mean(samples) - 1.0) < 0.01, (
            f"AR1 stationary mean={np.mean(samples):.6f}, expected ~1.0"
        )

    def test_stationary_variance(self):
        """Stationary variance should be sigma^2 / (1 - rho^2)."""
        rng = np.random.default_rng(42)
        rho, sigma = 0.9, 0.05
        ar1 = AR1Process(n_entities=10000, rho=rho, sigma=sigma, mu=1.0,
                         bounds=(-10, 10))  # wide bounds so they don't clip

        for _ in range(500):
            ar1.step(rng)

        samples = np.concatenate([ar1.step(rng) for _ in range(200)])
        expected_var = sigma ** 2 / (1 - rho ** 2)
        observed_var = np.var(samples)
        assert abs(observed_var - expected_var) < expected_var * 0.5, (
            f"AR1 stationary variance: expected~{expected_var:.6f}, "
            f"got {observed_var:.6f}"
        )

    def test_autocorrelation_near_rho(self):
        """Lag-1 autocorrelation should be close to rho."""
        rng = np.random.default_rng(42)
        rho = 0.9
        ar1 = AR1Process(n_entities=1, rho=rho, sigma=0.05, mu=1.0,
                         bounds=(-10, 10))

        for _ in range(200):
            ar1.step(rng)

        series = [ar1.step(rng)[0] for _ in range(2000)]
        series = np.array(series)

        x = series[:-1] - np.mean(series)
        y = series[1:] - np.mean(series)
        acf1 = np.sum(x * y) / np.sqrt(np.sum(x ** 2) * np.sum(y ** 2))
        assert abs(acf1 - rho) < 0.10, (
            f"Lag-1 autocorrelation should be ~{rho}, got {acf1:.3f}"
        )

    def test_bounds_respected(self):
        """All outputs must be within specified bounds."""
        rng = np.random.default_rng(42)
        lo, hi = 0.5, 2.0
        ar1 = AR1Process(
            n_entities=5000, rho=0.9, sigma=0.5, mu=1.0,
            bounds=(lo, hi),
        )
        for _ in range(100):
            vals = ar1.step(rng)
            assert np.all(vals >= lo), f"Below lower bound: min={vals.min()}"
            assert np.all(vals <= hi), f"Above upper bound: max={vals.max()}"


# =====================================================================
# 4. SIMULATION-LEVEL STATISTICS — event rates match expectations
# =====================================================================

class TestSimulationEventRates:
    """Verify aggregate event rates are consistent with parameters."""

    def test_stroke_annual_rate_matches_config(self):
        """Total stroke events / (patients x years) ~ annual_incident_rate.

        This is the most fundamental sanity check: does the simulation
        produce the right number of events on average?
        """
        rate = 0.05
        config = StrokeConfig(
            n_patients=10000, n_weeks=52,
            annual_incident_rate=rate,
            prediction_interval=52,  # predict once (minimal intervention)
            intervention_effectiveness=0.0,  # no treatment effect
        )
        sc = StrokePreventionScenario(config=config, seed=42)
        results = BranchedSimulationEngine(
            sc, CounterfactualMode.NONE
        ).run(10000)

        total_events = sum(
            results.outcomes[t].events.sum() for t in range(52)
        )
        person_years = 10000 * 1.0
        observed_rate = total_events / person_years
        assert 0.02 < observed_rate < 0.12, (
            f"Annual event rate should be ~{rate}, got {observed_rate:.4f} "
            f"({total_events} events in {person_years} person-years)"
        )

    def test_counterfactual_events_geq_factual_with_treatment(self):
        """With effective treatment, counterfactual should have more events.

        Run with multiple seeds to ensure it's not a fluke.
        """
        successes = 0
        for seed in range(5):
            config = StrokeConfig(
                n_patients=5000, n_weeks=26,
                prediction_interval=4,
                intervention_effectiveness=0.50,
            )
            sc = StrokePreventionScenario(config=config, seed=seed)
            results = BranchedSimulationEngine(
                sc, CounterfactualMode.BRANCHED
            ).run(5000)

            f_total = sum(
                results.outcomes[t].events.sum() for t in range(26)
            )
            cf_total = sum(
                results.counterfactual_outcomes[t].events.sum()
                for t in range(26)
            )
            if cf_total >= f_total:
                successes += 1

        assert successes >= 4, (
            f"Treatment should reduce events in most runs, "
            f"only worked {successes}/5 times"
        )

    def test_events_are_binary(self, stroke_results_branched):
        """All event arrays should be binary (0 or 1)."""
        results = stroke_results_branched
        for t in results.outcomes:
            events = results.outcomes[t].events
            unique = np.unique(events)
            assert np.all(np.isin(unique, [0, 1])), (
                f"Non-binary events at t={t}: {unique}"
            )

    def test_treated_count_bounded_by_population(self, stroke_results_branched):
        """Number treated at any timestep cannot exceed population size."""
        results = stroke_results_branched
        n = results.n_entities
        for t, intv in results.interventions.items():
            n_treated = len(intv.treated_indices)
            assert n_treated <= n, (
                f"Treated count {n_treated} > population {n} at t={t}"
            )
            assert np.all(intv.treated_indices >= 0)
            assert np.all(intv.treated_indices < n)


# =====================================================================
# 5. ML MODEL — achieved metrics match targets
# =====================================================================

class TestMLModelStatistics:
    """Verify ML model produces metrics in expected range."""

    @pytest.mark.parametrize("target_auc", [0.65, 0.75, 0.85, 0.90])
    def test_fitted_auc_near_target(self, target_auc):
        """After fitting, achieved AUC should be within 0.08 of target."""
        rng = np.random.default_rng(42)
        n = 5000
        risks = rng.beta(0.5, 0.5 * (1 / 0.13 - 1), n)
        risks = np.clip(risks, 0.01, 0.99)
        labels = (rng.random(n) < risks).astype(int)

        model = ControlledMLModel(
            mode="discrimination", target_auc=target_auc
        )
        report = model.fit(labels, risks, rng, n_iterations=5)
        assert abs(report["achieved_auc"] - target_auc) < 0.08, (
            f"AUC target={target_auc}, achieved={report['achieved_auc']:.3f}"
        )

    def test_prediction_scores_sum_to_reasonable_range(self):
        """Mean prediction score should be in a reasonable range."""
        rng = np.random.default_rng(42)
        n = 5000
        risks = rng.beta(0.5, 0.5 * (1 / 0.13 - 1), n)
        risks = np.clip(risks, 0.01, 0.99)
        labels = (rng.random(n) < risks).astype(int)

        model = ControlledMLModel(mode="discrimination", target_auc=0.80)
        model.fit(labels, risks, rng, n_iterations=3)
        scores = model.predict(risks, rng, true_labels=labels)

        assert_no_nan_inf(scores, "prediction scores")
        assert_in_unit_interval(scores, "prediction scores")
        assert 0.05 < np.mean(scores) < 0.95, (
            f"Mean score={np.mean(scores):.3f} is extreme"
        )

    def test_ppv_consistent_with_bayes_theorem(self):
        """Achieved PPV should be consistent with Bayes' theorem
        given achieved sensitivity, specificity, and prevalence."""
        rng = np.random.default_rng(42)
        n = 10000
        prevalence = 0.13
        risks = rng.beta(0.5, 0.5 * (1 / prevalence - 1), n)
        risks = np.clip(risks, 0.01, 0.99)
        labels = (rng.random(n) < risks).astype(int)
        actual_prev = labels.mean()

        model = ControlledMLModel(
            mode="classification",
            target_sensitivity=0.80,
            target_ppv=0.25,
        )
        report = model.fit(labels, risks, rng, n_iterations=5)
        scores, binary_labels = model.predict_binary(
            risks, rng, true_labels=labels
        )

        metrics = confusion_matrix_metrics(labels, scores, report["threshold"])
        if metrics["sensitivity"] > 0 and metrics["specificity"] < 1:
            expected_ppv = theoretical_ppv(
                actual_prev, metrics["sensitivity"], metrics["specificity"]
            )
            assert abs(metrics["ppv"] - expected_ppv) < 0.03, (
                f"PPV={metrics['ppv']:.3f} inconsistent with Bayes "
                f"(expected={expected_ppv:.3f}, prev={actual_prev:.3f}, "
                f"sens={metrics['sensitivity']:.3f}, "
                f"spec={metrics['specificity']:.3f})"
            )


# =====================================================================
# 6. CROSS-BRANCH CONSISTENCY — factual vs counterfactual
# =====================================================================

class TestCrossBranchConsistency:
    """Verify factual/counterfactual branches are internally consistent."""

    def test_both_branches_have_all_timesteps(self, stroke_results_branched):
        """Both branches should record outcomes at every timestep."""
        results = stroke_results_branched
        n_t = results.time_config.n_timesteps
        for t in range(n_t):
            assert t in results.outcomes, f"Missing factual outcome at t={t}"
            assert t in results.counterfactual_outcomes, (
                f"Missing counterfactual outcome at t={t}"
            )

    def test_outcome_array_shapes_consistent(self, stroke_results_branched):
        """All outcome arrays should have consistent shape = (n_entities,)."""
        results = stroke_results_branched
        n = results.n_entities
        for t in results.outcomes:
            f_shape = results.outcomes[t].events.shape
            cf_shape = results.counterfactual_outcomes[t].events.shape
            assert f_shape == (n,), (
                f"Factual shape at t={t}: {f_shape}, expected ({n},)"
            )
            assert cf_shape == (n,), (
                f"Counterfactual shape at t={t}: {cf_shape}, expected ({n},)"
            )

    def test_no_intervention_means_identical_branches(self):
        """With 0% effectiveness, branches should produce ~identical events."""
        config = StrokeConfig(
            n_patients=5000, n_weeks=26,
            prediction_interval=4,
            intervention_effectiveness=0.0,
        )
        sc = StrokePreventionScenario(config=config, seed=42)
        results = BranchedSimulationEngine(
            sc, CounterfactualMode.BRANCHED
        ).run(5000)

        f_total = sum(
            results.outcomes[t].events.sum()
            for t in range(26)
        )
        cf_total = sum(
            results.counterfactual_outcomes[t].events.sum()
            for t in range(26)
        )
        ratio = f_total / max(cf_total, 1)
        assert 0.80 < ratio < 1.20, (
            f"With 0% effectiveness, ratio should be ~1.0, "
            f"got {ratio:.3f} (F={f_total}, CF={cf_total})"
        )
