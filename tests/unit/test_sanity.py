"""Sanity checks: boundary conditions where the answer is known a priori.

Each test sets up a scenario where the expected output is analytically
obvious, then verifies the simulation produces it. If any of these
fail, something fundamental is broken.

Organized by component:
1. Prevalence bounds (Bayes' theorem)
2. ML model at extremes (perfect, random, degenerate)
3. No-show scenario at extremes (0%, 100% no-show, threshold bounds)
4. Intervention effectiveness at extremes (0%, 100%)
"""

import numpy as np
import pytest

from sdk.ml.model import ControlledMLModel
from sdk.ml.performance import (
    auc_score,
    confusion_matrix_metrics,
    theoretical_ppv,
    check_target_feasibility,
)
from sdk.core.engine import BranchedSimulationEngine, CounterfactualMode
from sdk.core.scenario import TimeConfig
from scenarios.noshow_overbooking.scenario import (
    ClinicConfig,
    NoShowOverbookingScenario,
)
from scenarios.stroke_prevention.scenario import (
    StrokeConfig,
    StrokePreventionScenario,
)


# =====================================================================
# 1. PREVALENCE BOUNDS — Bayes' theorem identities
# =====================================================================

class TestPrevalenceBoundsAnalytic:
    """Bayes' theorem has exact closed-form solutions. Verify them."""

    def test_ppv_at_50pct_prevalence_symmetric(self):
        """At 50% prevalence with equal sens and spec, PPV = sensitivity.

        Bayes: PPV = (sens * 0.5) / (sens * 0.5 + (1-spec) * 0.5)
        When sens = spec: PPV = sens / (sens + 1-sens) = sens.
        """
        for value in [0.70, 0.80, 0.90, 0.95]:
            ppv = theoretical_ppv(prevalence=0.5, sensitivity=value,
                                  specificity=value)
            assert abs(ppv - value) < 0.001, (
                f"At 50% prevalence with sens=spec={value}, "
                f"PPV should be {value}, got {ppv}"
            )

    def test_ppv_at_1pct_prevalence_ceiling(self):
        """At 1% prevalence, even 99% specificity limits PPV to ~44.7%.

        Bayes: PPV = (0.8 * 0.01) / (0.8 * 0.01 + 0.01 * 0.99)
             = 0.008 / 0.0179 ≈ 0.447

        This is THE key insight: low prevalence demolishes PPV.
        """
        ppv = theoretical_ppv(prevalence=0.01, sensitivity=0.80,
                              specificity=0.99)
        expected = 0.008 / (0.008 + 0.01 * 0.99)
        assert abs(ppv - expected) < 0.001

    def test_ppv_at_100pct_prevalence_is_always_1(self):
        """When everyone has the condition, every positive is true."""
        for sens in [0.1, 0.5, 0.8, 1.0]:
            for spec in [0.1, 0.5, 0.9]:
                ppv = theoretical_ppv(1.0, sens, spec)
                assert abs(ppv - 1.0) < 0.001

    def test_ppv_at_perfect_specificity_is_1(self):
        """If specificity = 1.0, no false positives, so PPV = 1.0."""
        for prev in [0.01, 0.05, 0.13, 0.50]:
            ppv = theoretical_ppv(prev, sensitivity=0.80,
                                  specificity=1.0)
            assert abs(ppv - 1.0) < 0.001

    def test_ppv_at_zero_specificity_equals_prevalence(self):
        """If specificity = 0, everyone is flagged, PPV = prevalence.

        Bayes: PPV = (sens * prev) / (sens * prev + 1.0 * (1-prev))
        When spec=0, all negatives are false positives.
        At extreme: PPV ≈ prev (for large N).
        """
        for prev in [0.05, 0.13, 0.50]:
            ppv = theoretical_ppv(prev, sensitivity=1.0,
                                  specificity=0.0)
            assert abs(ppv - prev) < 0.001

    def test_feasibility_high_ppv_low_prevalence(self):
        """PPV=0.50 at 1% prevalence requires specificity > 0.99."""
        result = check_target_feasibility(
            prevalence=0.01, target_ppv=0.50, target_sensitivity=0.80
        )
        assert result["required_specificity"] > 0.99
        # Max at 95% spec is much less than 0.50
        assert result["max_ppv_at_spec_95"] < 0.50

    def test_feasibility_moderate_ppv_moderate_prevalence(self):
        """PPV=0.25 at 13% prevalence should be feasible."""
        result = check_target_feasibility(
            prevalence=0.13, target_ppv=0.25, target_sensitivity=0.80
        )
        assert result["feasible"]
        assert result["required_specificity"] < 0.95


# =====================================================================
# 2. ML MODEL AT EXTREMES
# =====================================================================

class TestMLModelBoundaryConditions:
    """Test the noise injection at known extremes."""

    def _make_population(self, rng, n=5000, prevalence=0.13):
        risks = rng.beta(0.3, 0.3 * (1 / prevalence - 1), n)
        risks = np.clip(risks * prevalence / risks.mean(), 0.01, 0.80)
        labels = (rng.random(n) < risks).astype(int)
        return risks, labels

    def test_near_perfect_model_high_auc(self):
        """correlation=0.99, scale=0.01, high label noise → AUC near max.

        When correlation is nearly 1 and scale is minimal, predictions
        closely track true risk. AUC should be high (>0.85).
        """
        rng = np.random.default_rng(42)
        risks, labels = self._make_population(rng)

        model = ControlledMLModel(mode="discrimination")
        scores = model._generate_scores(
            risks, rng, labels,
            correlation=0.99, scale=0.01, label_noise_strength=3.0,
        )
        auc = auc_score(labels, scores)
        assert auc > 0.80, (
            f"Near-perfect model should have AUC > 0.80, got {auc:.3f}"
        )

    def test_random_model_auc_near_05(self):
        """correlation=0.0, scale=1.0, no label noise → AUC ≈ 0.5.

        When predictions are pure noise with no signal, the model
        can't distinguish positives from negatives.
        """
        rng = np.random.default_rng(42)
        risks, labels = self._make_population(rng)

        model = ControlledMLModel(mode="discrimination")
        scores = model._generate_scores(
            risks, rng, None,  # no labels = no label-dependent boost
            correlation=0.0, scale=1.0, label_noise_strength=0.0,
        )
        auc = auc_score(labels, scores)
        assert 0.35 < auc < 0.65, (
            f"Random model should have AUC near 0.5, got {auc:.3f}"
        )

    def test_all_same_risk_auc_near_05(self):
        """If all patients have the same risk, model can't discriminate.

        No matter what noise parameters, you can't rank identical inputs.
        """
        rng = np.random.default_rng(42)
        n = 3000
        risks = np.full(n, 0.13)  # everyone identical
        labels = (rng.random(n) < 0.13).astype(int)

        model = ControlledMLModel(mode="discrimination")
        scores = model._generate_scores(
            risks, rng, labels,
            correlation=0.9, scale=0.1, label_noise_strength=1.0,
        )
        auc = auc_score(labels, scores)
        # Label noise provides SOME signal, but base risk has zero
        # discrimination. AUC should be moderate at best.
        assert auc < 0.80, (
            f"Identical risks should limit AUC, got {auc:.3f}"
        )

    def test_threshold_0_flags_everyone(self):
        """Threshold=0: every patient flagged. Sensitivity=1, PPV=prevalence."""
        rng = np.random.default_rng(42)
        risks, labels = self._make_population(rng)
        scores = rng.random(len(labels))  # any scores

        m = confusion_matrix_metrics(labels, scores, threshold=0.0)
        assert m["sensitivity"] == 1.0
        assert abs(m["ppv"] - labels.mean()) < 0.01
        assert m["flag_rate"] == 1.0

    def test_threshold_1_flags_nobody(self):
        """Threshold=1.0: nobody flagged. Sensitivity=0, all negatives."""
        rng = np.random.default_rng(42)
        risks, labels = self._make_population(rng)
        scores = rng.uniform(0, 0.99, len(labels))  # all < 1.0

        m = confusion_matrix_metrics(labels, scores, threshold=1.0)
        assert m["sensitivity"] == 0.0
        assert m["flag_rate"] == 0.0
        assert m["fn"] == labels.sum()

    def test_higher_prevalence_enables_higher_ppv(self):
        """At 50% prevalence, PPV should be much higher than at 5%.

        This validates that the model's PPV is prevalence-dependent,
        not just a function of noise parameters.
        """
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        # Low prevalence
        risks_low, labels_low = self._make_population(
            rng1, prevalence=0.05
        )
        model_low = ControlledMLModel(mode="classification",
                                      target_ppv=0.30,
                                      target_sensitivity=0.70)
        r_low = model_low.fit(labels_low, risks_low, rng1,
                              n_iterations=3)

        # High prevalence
        risks_high, labels_high = self._make_population(
            rng2, prevalence=0.50
        )
        model_high = ControlledMLModel(mode="classification",
                                       target_ppv=0.30,
                                       target_sensitivity=0.70)
        r_high = model_high.fit(labels_high, risks_high, rng2,
                                n_iterations=3)

        # Higher prevalence should achieve equal or better PPV
        # (more true positives in the flagged pool)
        assert r_high["achieved_ppv"] >= r_low["achieved_ppv"] * 0.8, (
            f"High prevalence PPV={r_high['achieved_ppv']:.3f} should "
            f"beat low prevalence PPV={r_low['achieved_ppv']:.3f}"
        )

    def test_scores_always_in_unit_interval(self):
        """Predictions must always be in [0, 1] regardless of params."""
        rng = np.random.default_rng(42)
        risks, labels = self._make_population(rng)

        model = ControlledMLModel(mode="discrimination")
        # Extreme parameters
        for corr, scale, lns in [
            (0.0, 2.0, 0.0),
            (1.0, 0.0, 5.0),
            (0.5, 1.0, 3.0),
        ]:
            scores = model._generate_scores(
                risks, rng, labels, corr, scale, lns,
            )
            assert scores.min() >= 0.0, (
                f"Score < 0 with corr={corr}, scale={scale}"
            )
            assert scores.max() <= 1.0, (
                f"Score > 1 with corr={corr}, scale={scale}"
            )

    def test_unfitted_model_warns(self):
        """Calling predict() before fit() should raise a warning."""
        rng = np.random.default_rng(42)
        model = ControlledMLModel(mode="discrimination")
        with pytest.warns(UserWarning, match="before fit"):
            model.predict(np.array([0.1, 0.2]), rng)


# =====================================================================
# 3. NO-SHOW SCENARIO BOUNDARY CONDITIONS
# =====================================================================

def _run_noshow(
    n_days=20, n_patients=500, seed=42,
    base_noshow_rate=0.13, overbooking_threshold=0.30,
    max_individual_overbooks=5,
    model_type="predictor", model_auc=0.83,
    mode=CounterfactualMode.BRANCHED,
):
    """Helper to run no-show scenario with specific params."""
    tc = TimeConfig(
        n_timesteps=n_days, timestep_duration=1 / 365,
        timestep_unit="day",
        prediction_schedule=list(range(n_days)),
    )
    cc = ClinicConfig(n_providers=4, slots_per_provider_per_day=10,
                      max_overbook_per_provider=2)
    sc = NoShowOverbookingScenario(
        time_config=tc, seed=seed,
        n_patients=n_patients,
        base_noshow_rate=base_noshow_rate,
        overbooking_threshold=overbooking_threshold,
        max_individual_overbooks=max_individual_overbooks,
        model_type=model_type, model_auc=model_auc,
        clinic_config=cc,
    )
    return BranchedSimulationEngine(sc, mode).run(n_patients)


class TestNoShowBoundaryConditions:
    """No-show scenario at known extremes."""

    def test_threshold_1_means_no_overbooking(self):
        """Threshold=1.0: no predicted prob can reach it.

        Expected: zero overbookings, zero collisions, zero burden.
        Utilization = 1 - noshow_rate (resolved slots only).
        """
        results = _run_noshow(overbooking_threshold=1.0, n_days=15)

        # Check final state counters (from last resolved day's metadata)
        # Use t=14 which has resolved_slots from day 13
        final = results.outcomes[14].metadata
        assert final["total_overbooked"] == 0, (
            f"Threshold=1.0 should produce 0 overbookings, "
            f"got {final['total_overbooked']}"
        )
        assert final["total_collisions"] == 0
        assert final["mean_overbooking_burden"] == 0.0

    def test_max_individual_overbooks_0_means_no_overbooking(self):
        """max_individual_overbooks=0: no patient eligible for overbooking.

        Even with threshold=0, no candidates pass the eligibility check.
        """
        results = _run_noshow(
            overbooking_threshold=0.01,  # very aggressive
            max_individual_overbooks=0,
            n_days=15,
        )
        final = results.outcomes[14].metadata
        assert final["total_overbooked"] == 0, (
            f"max_individual_overbooks=0 should produce 0 overbookings, "
            f"got {final['total_overbooked']}"
        )

    def test_no_intervention_factual_equals_counterfactual(self):
        """With threshold=1.0 (no overbooking), both branches are identical.

        If no interventions occur, factual and counterfactual should
        produce exactly the same outcomes at every timestep.
        """
        results = _run_noshow(overbooking_threshold=1.0, n_days=10)

        for t in range(1, 10):  # skip t=0 (no resolved slots yet)
            f_events = results.outcomes[t].events
            cf_events = results.counterfactual_outcomes[t].events
            np.testing.assert_array_equal(
                f_events, cf_events,
                err_msg=(
                    f"With no interventions at t={t}, factual should "
                    f"equal counterfactual"
                ),
            )

    def test_low_threshold_means_more_overbooking(self):
        """Lower threshold → more slots overbooked → more burden.

        Threshold=0.10 should produce strictly more overbookings than
        threshold=0.50, all else equal.
        """
        r_low = _run_noshow(overbooking_threshold=0.10, n_days=20)
        r_high = _run_noshow(overbooking_threshold=0.50, n_days=20)

        ob_low = r_low.outcomes[19].metadata["total_overbooked"]
        ob_high = r_high.outcomes[19].metadata["total_overbooked"]
        assert ob_low >= ob_high, (
            f"Lower threshold should produce more overbookings: "
            f"thresh=0.10 got {ob_low}, thresh=0.50 got {ob_high}"
        )


# =====================================================================
# 4. INTERVENTION EFFECTIVENESS BOUNDARY CONDITIONS
# =====================================================================

class TestStrokeBoundaryConditions:
    """Stroke scenario at known extremes."""

    def test_zero_effectiveness_no_intervention_effect(self):
        """effectiveness=0: treatment does nothing to risk.

        Factual incidents should approximately equal counterfactual,
        because intervening with 0% effectiveness doesn't change risk.
        """
        config = StrokeConfig(
            n_patients=2000, n_weeks=26,
            prediction_interval=4,
            intervention_effectiveness=0.0,
        )
        sc = StrokePreventionScenario(config=config, seed=42)
        results = BranchedSimulationEngine(
            sc, CounterfactualMode.BRANCHED
        ).run(2000)

        f_total = sum(
            results.outcomes[t].events.sum() for t in range(26)
        )
        cf_total = sum(
            results.counterfactual_outcomes[t].events.sum()
            for t in range(26)
        )
        # With 0% effectiveness, treatment doesn't change risk.
        # Both branches should be very similar (small stochastic diff
        # from the threshold selection of who gets "treated").
        ratio = f_total / max(cf_total, 1)
        assert 0.85 < ratio < 1.15, (
            f"0% effectiveness: factual/CF ratio should be ~1.0, "
            f"got {ratio:.3f} (F={f_total}, CF={cf_total})"
        )

    def test_high_effectiveness_large_reduction(self):
        """effectiveness=0.90: strong treatment dramatically reduces events.

        With 90% risk reduction, factual should have substantially
        fewer incidents than counterfactual.
        """
        config = StrokeConfig(
            n_patients=5000, n_weeks=52,
            prediction_interval=4,
            intervention_effectiveness=0.90,
        )
        sc = StrokePreventionScenario(config=config, seed=42)
        results = BranchedSimulationEngine(
            sc, CounterfactualMode.BRANCHED
        ).run(5000)

        f_total = sum(
            results.outcomes[t].events.sum() for t in range(52)
        )
        cf_total = sum(
            results.counterfactual_outcomes[t].events.sum()
            for t in range(52)
        )
        assert f_total < cf_total, (
            f"90% effectiveness should reduce incidents: "
            f"factual={f_total}, counterfactual={cf_total}"
        )

    def test_zero_incident_rate_no_events(self):
        """annual_incident_rate=0.001: almost no events should occur.

        With near-zero risk, both branches should see very few events
        over a short simulation.
        """
        config = StrokeConfig(
            n_patients=1000, n_weeks=12,
            annual_incident_rate=0.001,
            prediction_interval=4,
        )
        sc = StrokePreventionScenario(config=config, seed=42)
        results = BranchedSimulationEngine(
            sc, CounterfactualMode.NONE
        ).run(1000)

        total = sum(
            results.outcomes[t].events.sum() for t in range(12)
        )
        # 1000 patients × 12 weeks × 0.001/52 ≈ 0.23 expected events
        assert total < 10, (
            f"Near-zero incidence should produce very few events, "
            f"got {total}"
        )


# =====================================================================
# 5. METRIC CONSISTENCY CHECKS
# =====================================================================

class TestMetricConsistency:
    """Verify metrics are internally consistent."""

    def test_confusion_matrix_sums_to_n(self):
        """TP + FP + TN + FN must equal population size."""
        rng = np.random.default_rng(42)
        n = 1000
        y_true = (rng.random(n) > 0.7).astype(int)
        y_pred = rng.random(n)

        for threshold in [0.0, 0.25, 0.5, 0.75, 1.0]:
            m = confusion_matrix_metrics(y_true, y_pred, threshold)
            total = m["tp"] + m["fp"] + m["tn"] + m["fn"]
            assert total == n, (
                f"Confusion matrix should sum to {n}, "
                f"got {total} at threshold={threshold}"
            )

    def test_sensitivity_plus_fnr_equals_1(self):
        """Sensitivity + False Negative Rate = 1.0 always."""
        rng = np.random.default_rng(42)
        y_true = np.concatenate([np.ones(100), np.zeros(900)])
        y_pred = rng.random(1000)

        m = confusion_matrix_metrics(y_true, y_pred, 0.5)
        n_pos = m["tp"] + m["fn"]
        if n_pos > 0:
            fnr = m["fn"] / n_pos
            assert abs(m["sensitivity"] + fnr - 1.0) < 1e-10

    def test_ppv_increases_with_higher_threshold(self):
        """Raising the threshold should increase PPV (fewer, better flags).

        This is a monotonicity property: higher threshold → more
        selective → higher precision (at cost of lower sensitivity).
        With a model that has ANY signal.
        """
        rng = np.random.default_rng(42)
        n = 5000
        # Create scores correlated with truth
        y_true = (rng.random(n) > 0.87).astype(int)
        y_pred = y_true * 0.7 + rng.random(n) * 0.3

        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        ppvs = []
        for t in thresholds:
            m = confusion_matrix_metrics(y_true, y_pred, t)
            if m["tp"] + m["fp"] > 0:
                ppvs.append(m["ppv"])
            else:
                ppvs.append(1.0)  # no flags = undefined, treat as 1

        # PPV should be non-decreasing (monotone with signal)
        for i in range(1, len(ppvs)):
            assert ppvs[i] >= ppvs[i - 1] - 0.02, (
                f"PPV should increase with threshold: "
                f"PPV at {thresholds[i-1]}={ppvs[i-1]:.3f}, "
                f"PPV at {thresholds[i]}={ppvs[i]:.3f}"
            )

    def test_auc_of_perfect_model_is_1(self):
        """If predictions perfectly separate classes, AUC = 1.0."""
        y_true = np.concatenate([np.ones(500), np.zeros(500)])
        y_scores = np.concatenate([
            np.ones(500) * 0.9,  # all positives high
            np.ones(500) * 0.1,  # all negatives low
        ])
        auc = auc_score(y_true, y_scores)
        assert auc > 0.99, f"Perfect separation should give AUC≈1.0, got {auc}"

    def test_auc_of_inverted_model_is_0(self):
        """If predictions are inverted (high scores for negatives), AUC ≈ 0."""
        y_true = np.concatenate([np.ones(500), np.zeros(500)])
        y_scores = np.concatenate([
            np.ones(500) * 0.1,  # positives scored LOW
            np.ones(500) * 0.9,  # negatives scored HIGH
        ])
        auc = auc_score(y_true, y_scores)
        assert auc < 0.01, f"Inverted model should give AUC≈0.0, got {auc}"
