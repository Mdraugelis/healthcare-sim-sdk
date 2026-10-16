"""Regressions for the ADD register, including negative controls."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from healthcare_sim_sdk.core.engine import BranchedSimulationEngine, CounterfactualMode
from healthcare_sim_sdk.core.results import SimulationResults
from healthcare_sim_sdk.core.scenario import Outcomes, Interventions, TimeConfig
from healthcare_sim_sdk.ml.model import ControlledMLModel, OperatingPointError
from healthcare_sim_sdk.experiments.analysis.its import cits_with_control
from healthcare_sim_sdk.scenarios._template.scenario import CoinFlipScenario
from healthcare_sim_sdk.scenarios.nurse_retention.monitoring.tier4_model_drift import (
    _calibration_slope_intercept,
    analyze_tier4,
)


def test_scenarios_are_single_use_across_engines():
    sc = CoinFlipScenario(TimeConfig(3, 1 / 52))
    BranchedSimulationEngine(sc).run(10)
    with pytest.raises(RuntimeError, match="single-use"):
        BranchedSimulationEngine(sc).run(10)


def test_named_outcomes_and_treatment_ids():
    result = SimulationResults(2, TimeConfig(3, 1 / 52))
    for t in range(3):
        out = Outcomes(
            np.zeros(2),
            np.array([101, 305]),
            secondary={"mortality": np.array([t, 2 * t])},
        )
        result.record_outcomes(t, out)
        result.record_outcomes(t, out, counterfactual=True)
    result.record_interventions(0, Interventions(np.array([0])))
    result.record_interventions(2, Interventions(np.array([1])))
    a = result.to_analysis()
    np.testing.assert_array_equal(result.get_outcome_series("mortality"), [0, 3, 6])
    np.testing.assert_array_equal(
        result.get_counterfactual_outcome_series("mortality"), [0, 3, 6]
    )
    np.testing.assert_array_equal(a.to_time_series("mortality")["outcomes"], [0, 3, 6])
    np.testing.assert_array_equal(
        a.to_panel("mortality")["outcomes"], [0, 0, 1, 2, 2, 4]
    )
    np.testing.assert_array_equal(a.to_panel()["treated"], [1, 0, 1, 0, 1, 1])
    np.testing.assert_array_equal(
        a.to_panel(treatment_definition="ever")["treated"], np.ones(6)
    )
    np.testing.assert_array_equal(
        a.to_entity_snapshots(2, "mortality")["outcomes"], [2, 4]
    )
    np.testing.assert_array_equal(
        a.to_subgroup_panel("mortality")["outcomes"], [0, 0, 1, 2, 2, 4]
    )
    for export in [
        result.get_outcome_series,
        a.to_panel,
        a.to_time_series,
        a.to_subgroup_panel,
    ]:
        with pytest.raises(KeyError, match="Unknown outcome"):
            export("misspelled")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_ppv": 0.2},
        {"target_sensitivity": 0.8},
        {"binding": "ppv"},
        {"operating_threshold": None, "target_calibration_slope": 1},
    ],
)
def test_unused_arguments_raise(kwargs):
    with pytest.raises(ValueError, match="Unused arguments"):
        ControlledMLModel(mode="discrimination", **kwargs)


def test_refit_prevalence_is_current():
    rng = np.random.default_rng(5)
    m = ControlledMLModel(target_auc=0.6)
    for prevalence in [0.1, 0.3]:
        risks = rng.beta(0.3, 0.3 * (1 / prevalence - 1), 4000)
        labels = rng.random(len(risks)) < risks
        report = m.fit(
            labels,
            risks,
            rng,
            n_iterations=1,
            correlation_grid=np.array([0.4, 0.8]),
            scale_grid=np.array([0.1, 0.5]),
        )
        assert report["prevalence"] == labels.mean()


def test_classification_keeps_unreachable_requirement_and_warns():
    rng = np.random.default_rng(7)
    risks = rng.beta(2, 18, 5000)
    labels = (rng.random(len(risks)) < risks).astype(int)
    model = ControlledMLModel(
        mode="classification",
        target_ppv=0.8,
        target_sensitivity=0.8,
        allow_infeasible=True,
    )
    with pytest.warns(UserWarning):
        report = model.fit(
            labels,
            risks,
            rng,
            n_iterations=1,
            correlation_grid=np.array([0.4, 0.9]),
            scale_grid=np.array([0.01, 0.5]),
        )
    assert report["required_auc"] > report["population_auc_ceiling"]
    assert not report["sensitivity_within_tolerance"]
    assert report["gate_misses"]
    strict = ControlledMLModel(
        mode="classification", target_ppv=0.8, target_sensitivity=0.8
    )
    with pytest.raises(OperatingPointError):
        strict.fit(labels, risks, rng)


def test_reference_cohort_has_enough_events():
    rng = np.random.default_rng(7)
    risks = rng.beta(0.3, 3, 100)
    model = ControlledMLModel(target_auc=0.65)
    report = model.fit_reference(risks, rng)
    assert report["n_positives"] >= 100
    assert report["reference_cohort_size"] >= 5000


def test_paired_cits_cancels_shared_noise():
    rng = np.random.default_rng(11)
    control = rng.normal(0, 5, 100)
    factual = control + rng.normal(-0.5, 0.02, 100)
    paired = cits_with_control(factual, control, paired=True)
    independent = cits_with_control(factual, control)
    assert paired.effect_se < independent.effect_se / 10
    assert paired.ci_lower < -0.5 < paired.ci_upper
    assert cits_with_control(control, control, paired=True).p_value == 1


def test_logistic_recalibration_recovers_probability_scale():
    rng = np.random.default_rng(61)
    logits = rng.normal(-2, 1, 100000)
    p = 1 / (1 + np.exp(-logits))
    labels = rng.random(len(p)) < p
    slope, intercept = _calibration_slope_intercept(p, labels)
    assert abs(slope - 1) < 0.05
    assert abs(intercept) < 0.07


def test_tier4_does_not_claim_drift_on_tiny_baseline():
    rng = np.random.default_rng(6)
    log = [
        {
            "week": t,
            "scores": np.full(100, 0.01),
            "true_labels": (rng.random(100) < 0.01).astype(int),
        }
        for t in range(0, 30, 2)
    ]
    assert not analyze_tier4(log).detection_events


def test_seed_none_is_documented_deterministic_default():
    tc = TimeConfig(3, 1 / 52)
    a = BranchedSimulationEngine(CoinFlipScenario(tc, seed=None)).run(100)
    b = BranchedSimulationEngine(CoinFlipScenario(tc, seed=42)).run(100)
    for t in a.outcomes:
        np.testing.assert_array_equal(a.outcomes[t].events, b.outcomes[t].events)


def test_random_hook_checks_import_aliases_and_stdlib(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "random_hook", Path("scripts/check_no_bare_random.py")
    )
    hook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hook)
    for source in [
        "from numpy.random import default_rng",
        "import numpy.random as npr",
        "import random",
        "from random import random",
        "import numpy as alias\nx=alias.random.default_rng()",
    ]:
        path = tmp_path / "bad.py"
        path.write_text(source)
        assert hook.check_file(path), source
    path.write_text("import numpy as np\ndef f(rng: np.random.Generator): pass")
    assert not hook.check_file(path)


def test_overbooking_does_not_mutate_inputs():
    import pickle
    from healthcare_sim_sdk.scenarios.noshow_overbooking.scenario import (
        NoShowOverbookingScenario,
    )

    sc = NoShowOverbookingScenario(
        TimeConfig(3, 1 / 365), seed=4, model_type="baseline"
    )
    state = sc.create_population(100)
    before = pickle.dumps(state)
    next_state = sc.step(state, 0)
    assert pickle.dumps(state) == before
    pred = sc.predict(next_state, 0)
    before_intervention = pickle.dumps(next_state)
    sc.intervene(next_state, pred, 0)
    assert pickle.dumps(next_state) == before_intervention


@pytest.mark.parametrize("mode", [CounterfactualMode.NONE, CounterfactualMode.SNAPSHOT])
def test_retention_supports_partial_counterfactual_modes(mode):
    from healthcare_sim_sdk.scenarios.nurse_retention.scenario import (
        NurseRetentionScenario,
        RetentionConfig,
    )

    sc = NurseRetentionScenario(RetentionConfig(n_weeks=5, model_auc=0.7), seed=8)
    result = BranchedSimulationEngine(sc, mode).run(300)
    assert result.validations["final"]["counterfactual_available"] is False


def test_functional_ar1_has_documented_seasonal_transfer():
    from healthcare_sim_sdk.population.temporal_dynamics import ar1_step

    rho, amplitude, period = 0.95, 0.1, 52
    rng = np.random.default_rng(4)
    current, trajectory = np.ones(1), []
    for t in range(52 * 20):
        current = ar1_step(
            current,
            rng,
            rho=rho,
            sigma=0,
            mu=1 + amplitude * np.cos(2 * np.pi * t / period),
        )
        trajectory.append(current[0])
    y = np.array(trajectory[-period:]) - 1
    omega = 2 * np.pi / period
    harmonic = 2 * np.sum(y * np.exp(-1j * omega * np.arange(period))) / period
    assert abs(abs(harmonic) - 0.039) < 0.002
    assert 8 <= -np.angle(harmonic) / omega <= 10


def test_masai_routes_by_percentile_and_manz_uses_flags():
    from healthcare_sim_sdk.core.scenario import Predictions
    from healthcare_sim_sdk.scenarios.paper09_masai.scenario import MASAIScenario
    from healthcare_sim_sdk.scenarios.paper07_manz_nudges.scenario import (
        ManzNudgeScenario,
    )

    tc = TimeConfig(1, 1 / 52)
    masai = MASAIScenario(tc)
    state = masai.create_population(100)
    # Every score is far below the retired absolute cut 0.70.
    scores = np.linspace(0.001, 0.01, 100)
    _, assignment = masai.intervene(state, Predictions(scores), 0)
    assert len(assignment.treated_indices) == 30
    manz = ManzNudgeScenario(tc, stepped_wedge_start=0)
    state = manz.create_population(1000)
    _, no_flags = manz.intervene(
        state, Predictions(np.zeros(1000), np.zeros(1000, int)), 13
    )
    _, all_flags = manz.intervene(
        state, Predictions(np.ones(1000), np.ones(1000, int)), 13
    )
    assert len(no_flags.treated_indices) == 0
    assert len(all_flags.treated_indices) == 1000


def test_tier4_labels_are_mature_realized_outcomes():
    import copy
    from healthcare_sim_sdk.scenarios.nurse_retention.scenario import (
        NurseRetentionScenario,
        RetentionConfig,
    )
    from healthcare_sim_sdk.scenarios.nurse_retention.monitoring.harness import (
        MonitoringHarness,
    )

    cfg = RetentionConfig(n_weeks=15, model_auc=0.7)
    result = BranchedSimulationEngine(NurseRetentionScenario(cfg, seed=11)).run(300)
    harness = MonitoringHarness(tier3_mode="its_only")
    original = harness.run_from_results(result, cfg)
    poisoned = copy.deepcopy(result)
    for pred in poisoned.predictions.values():
        pred.metadata["true_labels"] = np.ones_like(pred.scores)
    repeated = harness.run_from_results(poisoned, cfg)
    for a, b in zip(original.tier4_prediction_log, repeated.tier4_prediction_log):
        np.testing.assert_array_equal(a["true_labels"], b["true_labels"])
        t = a["prediction_week"]
        active = result.outcomes[t].secondary["active_mask"].astype(bool)
        expected = np.maximum.reduce(
            [result.outcomes[u].events for u in range(t + 1, t + 3)]
        )
        np.testing.assert_array_equal(a["true_labels"], expected[active])
        assert a["week"] == t + cfg.prediction_interval


def test_cusum_calibration_uses_external_null_runs():
    from healthcare_sim_sdk.scenarios.nurse_retention.monitoring.tier2_cusum import (
        calibrate_h,
    )

    rng = np.random.default_rng(41)
    series = [rng.normal(size=104).tolist() for _ in range(100)]
    report = calibrate_h(series, [4, 8, 16, 32, 64], max_alarm_probability=0.1)
    assert report["calibration_rates"][report["h"]] <= 0.1
    # An irreversible trend in an alleged null must make this calibration fail.
    with pytest.raises(ValueError, match="No h candidate"):
        calibrate_h([np.arange(104).tolist()], [4, 8])


def test_strict_stroke_defaults_raise_on_unreachable_targets():
    from healthcare_sim_sdk.scenarios.stroke_prevention.scenario import (
        StrokeConfig,
        StrokePreventionScenario,
    )

    assert StrokeConfig().allow_infeasible_model is False
    with pytest.raises(OperatingPointError):
        BranchedSimulationEngine(StrokePreventionScenario(seed=42)).run(10000)


def test_engine_never_predicts_or_intervenes_on_counterfactual():
    class FactualOnly(CoinFlipScenario):
        def create_population(self, n):
            self.expected_rng = self.rng
            self.predict_calls = 0
            return super().create_population(n)

        def predict(self, state, t):
            assert self.rng is self.expected_rng
            self.predict_calls += 1
            return super().predict(state, t)

        def intervene(self, state, predictions, t):
            assert self.rng is self.expected_rng
            return super().intervene(state, predictions, t)

    scenario = FactualOnly(TimeConfig(5, 1 / 52, prediction_schedule=[0, 2, 4]))
    BranchedSimulationEngine(scenario).run(100)
    assert scenario.predict_calls == 3


@pytest.mark.parametrize(
    "expression", ["state[0]", "state.value", "getattr(state, 'value')"]
)
def test_state_opacity_hook_rejects_all_documented_accesses(
    tmp_path, monkeypatch, expression
):
    spec = importlib.util.spec_from_file_location(
        "invariants_hook", Path("scripts/check_invariants.py")
    )
    hook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hook)
    engine = tmp_path / "healthcare_sim_sdk/core/engine.py"
    engine.parent.mkdir(parents=True)
    engine.write_text(f"def evolve(state):\n    return {expression}\n")
    monkeypatch.chdir(tmp_path)
    assert not hook.check_engine_state_opacity()


def test_incident_onset_is_counted_even_with_same_step_discharge():
    from healthcare_sim_sdk.scenarios.paper01_epic_esm.scenario import (
        EpicESMConfig,
        EpicESMScenario,
        ROW_BASE_RISK,
        ROW_CURRENT_RISK,
        ROW_STAGE,
        ROW_LOS_REMAINING,
        STAGE_DISCHARGED,
    )

    sc = EpicESMScenario(
        TimeConfig(1, 1 / 2190),
        seed=3,
        config=EpicESMConfig(prog_at_risk=1.0, ar1_sigma=0),
    )
    state = sc.create_population(50)
    state[ROW_BASE_RISK] = 0.5
    state[ROW_CURRENT_RISK] = 0.5
    state[ROW_LOS_REMAINING] = 1
    next_state = sc.step(state, 0)
    assert np.all(next_state[ROW_STAGE] == STAGE_DISCHARGED)
    observed = sc.measure(next_state, 0)
    assert observed.events.sum() == 0  # occupancy at measurement
    assert (
        0 < observed.secondary["incident_onset"].sum() <= 50
    )  # transitions survive discharge


def test_manz_unflagged_patients_keep_standard_care():
    from healthcare_sim_sdk.core.scenario import Predictions
    from healthcare_sim_sdk.scenarios.paper07_manz_nudges.scenario import (
        ManzNudgeScenario,
    )

    scenario = ManzNudgeScenario(TimeConfig(1, 1 / 52), seed=42, stepped_wedge_start=0)
    control = scenario.create_population(1000)
    control.mortality_risk[:] = 0.9
    factual, _ = scenario.intervene(
        control, Predictions(np.zeros(1000), np.zeros(1000, int)), 13
    )
    f_out = scenario.measure(factual, 13)
    scenario.rng = scenario._partitioner.fork().create_streams()
    c_out = scenario.measure(control, 13)
    np.testing.assert_array_equal(f_out.events, c_out.events)
