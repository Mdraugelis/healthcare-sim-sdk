"""Conservation law and invariant tests.

These tests encode properties that must hold across ALL valid simulations,
regardless of parameters. They are the most powerful class of tests because
a violation indicates a fundamental logic error, not a statistical fluke.

Categories:
- Population conservation (entity count doesn't change)
- Probability conservation (probabilities sum correctly)
- Monotonicity invariants (higher X always means higher Y)
- Symmetry properties (relabeling shouldn't change results)
- Accounting identities (things that must add up)
"""

import numpy as np
import pytest

from sdk.core.engine import BranchedSimulationEngine, CounterfactualMode
from sdk.core.scenario import TimeConfig
from sdk.ml.performance import (
    confusion_matrix_metrics,
    theoretical_ppv,
    check_target_feasibility,
)
from sdk.population.temporal_dynamics import (
    annual_risk_to_hazard,
    hazard_to_timestep_probability,
)
from scenarios.stroke_prevention.scenario import (
    StrokeConfig,
    StrokePreventionScenario,
)
from scenarios.noshow_overbooking.scenario import (
    ClinicConfig,
    NoShowOverbookingScenario,
)


# =====================================================================
# 1. POPULATION CONSERVATION
# =====================================================================

class TestPopulationConservation:
    """Entity count must not change during simulation."""

    def test_stroke_population_constant(self):
        """Number of entities in outcomes must equal n_entities at all t."""
        config = StrokeConfig(
            n_patients=500, n_weeks=12,
            prediction_interval=4,
        )
        sc = StrokePreventionScenario(config=config, seed=42)
        results = BranchedSimulationEngine(
            sc, CounterfactualMode.BRANCHED
        ).run(500)

        for t in range(12):
            f_n = len(results.outcomes[t].events)
            cf_n = len(results.counterfactual_outcomes[t].events)
            assert f_n == 500, (
                f"Factual population changed at t={t}: {f_n} != 500"
            )
            assert cf_n == 500, (
                f"CF population changed at t={t}: {cf_n} != 500"
            )

    def test_prediction_scores_match_population(self):
        """Prediction scores should have same length as population."""
        config = StrokeConfig(
            n_patients=300, n_weeks=12,
            prediction_interval=4,
        )
        sc = StrokePreventionScenario(config=config, seed=42)
        results = BranchedSimulationEngine(
            sc, CounterfactualMode.BRANCHED
        ).run(300)

        for t, pred in results.predictions.items():
            assert len(pred.scores) == 300, (
                f"Prediction scores length at t={t}: "
                f"{len(pred.scores)} != 300"
            )

    def test_entity_ids_consistent_across_time(self):
        """Entity IDs should be the same set at every timestep."""
        config = StrokeConfig(
            n_patients=200, n_weeks=8,
            prediction_interval=4,
        )
        sc = StrokePreventionScenario(config=config, seed=42)
        results = BranchedSimulationEngine(
            sc, CounterfactualMode.BRANCHED
        ).run(200)

        reference_ids = None
        for t in range(8):
            eids = results.outcomes[t].entity_ids
            if eids is not None:
                if reference_ids is None:
                    reference_ids = set(eids.tolist())
                else:
                    current_ids = set(eids.tolist())
                    assert current_ids == reference_ids, (
                        f"Entity IDs changed at t={t}: "
                        f"missing={reference_ids - current_ids}, "
                        f"extra={current_ids - reference_ids}"
                    )


# =====================================================================
# 2. CONFUSION MATRIX ACCOUNTING IDENTITIES
# =====================================================================

class TestConfusionMatrixAccounting:
    """Confusion matrix must satisfy exact identities."""

    @pytest.mark.parametrize("seed", range(10))
    def test_tp_fp_tn_fn_sum_to_n(self, seed):
        """TP + FP + TN + FN == N, always, for any threshold."""
        rng = np.random.default_rng(seed)
        n = np.random.default_rng(seed + 100).integers(10, 1000)
        y_true = (rng.random(n) > 0.7).astype(int)
        y_pred = rng.random(n)

        for threshold in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            m = confusion_matrix_metrics(y_true, y_pred, threshold)
            total = m["tp"] + m["fp"] + m["tn"] + m["fn"]
            assert total == n, (
                f"seed={seed}, n={n}, threshold={threshold}: "
                f"TP+FP+TN+FN={total} != {n}"
            )

    @pytest.mark.parametrize("seed", range(10))
    def test_sensitivity_specificity_bounded_01(self, seed):
        """Sensitivity and specificity must be in [0, 1]."""
        rng = np.random.default_rng(seed)
        n = 500
        y_true = (rng.random(n) > 0.7).astype(int)
        y_pred = rng.random(n)

        for threshold in np.linspace(0, 1, 20):
            m = confusion_matrix_metrics(y_true, y_pred, threshold)
            assert 0 <= m["sensitivity"] <= 1, (
                f"Sensitivity={m['sensitivity']} out of [0,1]"
            )
            assert 0 <= m["specificity"] <= 1, (
                f"Specificity={m['specificity']} out of [0,1]"
            )
            assert 0 <= m["ppv"] <= 1 or m["ppv"] == 0.0
            assert 0 <= m["npv"] <= 1 or m["npv"] == 0.0

    def test_f1_harmonic_mean_identity(self):
        """F1 = 2 * PPV * Sens / (PPV + Sens) when both > 0."""
        rng = np.random.default_rng(42)
        n = 1000
        y_true = (rng.random(n) > 0.7).astype(int)
        y_pred = y_true * 0.6 + rng.random(n) * 0.4

        for threshold in [0.3, 0.5, 0.7]:
            m = confusion_matrix_metrics(y_true, y_pred, threshold)
            if m["ppv"] > 0 and m["sensitivity"] > 0:
                expected_f1 = (
                    2 * m["ppv"] * m["sensitivity"]
                    / (m["ppv"] + m["sensitivity"])
                )
                assert abs(m["f1"] - expected_f1) < 1e-10, (
                    f"F1={m['f1']:.6f} != expected={expected_f1:.6f}"
                )

    def test_flag_rate_identity(self):
        """flag_rate = (TP + FP) / N."""
        rng = np.random.default_rng(42)
        n = 1000
        y_true = (rng.random(n) > 0.7).astype(int)
        y_pred = rng.random(n)

        for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
            m = confusion_matrix_metrics(y_true, y_pred, threshold)
            expected_flag = (m["tp"] + m["fp"]) / n
            assert abs(m["flag_rate"] - expected_flag) < 1e-10, (
                f"flag_rate={m['flag_rate']:.6f} != "
                f"(TP+FP)/N={expected_flag:.6f}"
            )


# =====================================================================
# 3. MONOTONICITY INVARIANTS
# =====================================================================

class TestMonotonicity:
    """Properties that must be monotone in their inputs."""

    def test_higher_effectiveness_fewer_factual_events(self):
        """More effective treatment should reduce factual events monotonically.

        Run multiple effectiveness levels and verify ordering.
        """
        event_counts = {}
        for eff in [0.0, 0.25, 0.50, 0.75]:
            config = StrokeConfig(
                n_patients=5000, n_weeks=26,
                prediction_interval=4,
                intervention_effectiveness=eff,
            )
            sc = StrokePreventionScenario(config=config, seed=42)
            results = BranchedSimulationEngine(
                sc, CounterfactualMode.BRANCHED
            ).run(5000)
            total = sum(
                results.outcomes[t].events.sum() for t in range(26)
            )
            event_counts[eff] = total

        # Should be non-increasing (more effectiveness = fewer events)
        effs = sorted(event_counts.keys())
        for i in range(1, len(effs)):
            assert event_counts[effs[i]] <= event_counts[effs[i-1]] * 1.10, (
                f"Monotonicity violation: "
                f"eff={effs[i-1]}->{event_counts[effs[i-1]]}, "
                f"eff={effs[i]}->{event_counts[effs[i]]}"
            )

    def test_higher_threshold_fewer_treated(self):
        """Higher treatment threshold should treat fewer patients."""
        treated_counts = {}
        for thresh in [0.2, 0.4, 0.6, 0.8]:
            config = StrokeConfig(
                n_patients=3000, n_weeks=12,
                prediction_interval=4,
                treatment_threshold=thresh,
            )
            sc = StrokePreventionScenario(config=config, seed=42)
            results = BranchedSimulationEngine(
                sc, CounterfactualMode.BRANCHED
            ).run(3000)

            total_treated = sum(
                len(intv.treated_indices)
                for intv in results.interventions.values()
            )
            treated_counts[thresh] = total_treated

        thresholds = sorted(treated_counts.keys())
        for i in range(1, len(thresholds)):
            assert treated_counts[thresholds[i]] <= treated_counts[thresholds[i-1]] * 1.05, (
                f"Higher threshold should treat fewer: "
                f"thresh={thresholds[i-1]}->{treated_counts[thresholds[i-1]]}, "
                f"thresh={thresholds[i]}->{treated_counts[thresholds[i]]}"
            )

    def test_theoretical_ppv_monotone_in_specificity(self):
        """PPV is monotonically increasing in specificity (at fixed prev, sens)."""
        specs = np.linspace(0.5, 0.99, 50)
        ppvs = [theoretical_ppv(0.13, 0.80, s) for s in specs]
        for i in range(1, len(ppvs)):
            assert ppvs[i] >= ppvs[i-1] - 1e-10, (
                f"PPV not monotone in specificity: "
                f"spec={specs[i-1]:.3f}->PPV={ppvs[i-1]:.4f}, "
                f"spec={specs[i]:.3f}->PPV={ppvs[i]:.4f}"
            )

    def test_theoretical_ppv_monotone_in_prevalence(self):
        """PPV is monotonically increasing in prevalence (at fixed sens, spec)."""
        prevs = np.linspace(0.01, 0.90, 50)
        ppvs = [theoretical_ppv(p, 0.80, 0.90) for p in prevs]
        for i in range(1, len(ppvs)):
            assert ppvs[i] >= ppvs[i-1] - 1e-10, (
                f"PPV not monotone in prevalence: "
                f"prev={prevs[i-1]:.3f}->PPV={ppvs[i-1]:.4f}, "
                f"prev={prevs[i]:.3f}->PPV={ppvs[i]:.4f}"
            )


# =====================================================================
# 4. TREATMENT INDICATOR CONSISTENCY
# =====================================================================

class TestTreatmentConsistency:
    """Treatment-related invariants that must hold."""

    def test_treated_indices_are_unique(self):
        """No entity should be treated twice at the same timestep."""
        config = StrokeConfig(
            n_patients=1000, n_weeks=12,
            prediction_interval=4,
        )
        sc = StrokePreventionScenario(config=config, seed=42)
        results = BranchedSimulationEngine(
            sc, CounterfactualMode.BRANCHED
        ).run(1000)

        for t, intv in results.interventions.items():
            indices = intv.treated_indices
            assert len(indices) == len(np.unique(indices)), (
                f"Duplicate treated indices at t={t}: "
                f"{len(indices)} total, {len(np.unique(indices))} unique"
            )

    def test_treated_indices_valid_range(self):
        """All treated indices must be valid entity indices."""
        config = StrokeConfig(
            n_patients=500, n_weeks=12,
            prediction_interval=4,
        )
        sc = StrokePreventionScenario(config=config, seed=42)
        results = BranchedSimulationEngine(
            sc, CounterfactualMode.BRANCHED
        ).run(500)

        for t, intv in results.interventions.items():
            assert np.all(intv.treated_indices >= 0), (
                f"Negative index at t={t}"
            )
            assert np.all(intv.treated_indices < 500), (
                f"Index >= n at t={t}: max={intv.treated_indices.max()}"
            )

    def test_intervention_only_at_prediction_times(self):
        """Interventions should only occur at scheduled prediction times."""
        config = StrokeConfig(
            n_patients=200, n_weeks=20,
            prediction_interval=4,
        )
        sc = StrokePreventionScenario(config=config, seed=42)
        results = BranchedSimulationEngine(
            sc, CounterfactualMode.BRANCHED
        ).run(200)

        pred_times = set(results.predictions.keys())
        intv_times = set(results.interventions.keys())
        assert intv_times.issubset(pred_times), (
            f"Interventions at non-prediction times: "
            f"{intv_times - pred_times}"
        )


# =====================================================================
# 5. HAZARD CONVERSION IDENTITIES
# =====================================================================

class TestHazardConservation:
    """Hazard conversion must preserve ordering and bounds."""

    def test_risk_ordering_preserved(self):
        """Higher input risk -> higher output probability (order preserving)."""
        risks = np.array([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90])
        hazards = annual_risk_to_hazard(risks)
        probs = hazard_to_timestep_probability(hazards, 1 / 52)

        for i in range(1, len(risks)):
            assert probs[i] > probs[i-1], (
                f"Ordering not preserved: "
                f"risk={risks[i-1]:.3f}->prob={probs[i-1]:.6f}, "
                f"risk={risks[i]:.3f}->prob={probs[i]:.6f}"
            )

    def test_probability_bounded_01(self):
        """All timestep probabilities must be in [0, 1]."""
        risks = np.linspace(0, 0.999, 1000)
        hazards = annual_risk_to_hazard(risks)
        for dt in [1/365, 1/52, 1/12, 1/4, 1.0]:
            probs = hazard_to_timestep_probability(hazards, dt)
            assert np.all(probs >= 0), f"Negative prob at dt={dt}"
            assert np.all(probs <= 1), f"Prob > 1 at dt={dt}"


# =====================================================================
# 6. SIMULATION COUNTERFACTUAL ACCOUNTING
# =====================================================================

class TestCounterfactualAccounting:
    """Counterfactual branch must satisfy structural invariants."""

    def test_counterfactual_has_all_timesteps(self):
        """Counterfactual outcomes must exist at every timestep in BRANCHED mode."""
        config = StrokeConfig(
            n_patients=200, n_weeks=12,
            prediction_interval=4,
        )
        sc = StrokePreventionScenario(config=config, seed=42)
        results = BranchedSimulationEngine(
            sc, CounterfactualMode.BRANCHED
        ).run(200)

        for t in range(12):
            assert t in results.counterfactual_outcomes, (
                f"Missing counterfactual at t={t}"
            )

    def test_snapshot_only_at_prediction_times(self):
        """In SNAPSHOT mode, counterfactual only at prediction times."""
        config = StrokeConfig(
            n_patients=200, n_weeks=12,
            prediction_interval=4,
        )
        sc = StrokePreventionScenario(config=config, seed=42)
        results = BranchedSimulationEngine(
            sc, CounterfactualMode.SNAPSHOT
        ).run(200)

        cf_times = set(results.counterfactual_outcomes.keys())
        pred_times = set(results.predictions.keys())
        assert cf_times.issubset(pred_times), (
            f"Snapshot CF at non-prediction times: {cf_times - pred_times}"
        )

    def test_none_mode_no_counterfactual(self):
        """In NONE mode, no counterfactual outcomes should exist."""
        config = StrokeConfig(
            n_patients=200, n_weeks=12,
            prediction_interval=4,
        )
        sc = StrokePreventionScenario(config=config, seed=42)
        results = BranchedSimulationEngine(
            sc, CounterfactualMode.NONE
        ).run(200)

        assert len(results.counterfactual_outcomes) == 0, (
            f"NONE mode should have no CF outcomes, "
            f"found {len(results.counterfactual_outcomes)}"
        )
