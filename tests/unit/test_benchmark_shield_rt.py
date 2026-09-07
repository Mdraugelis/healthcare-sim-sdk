"""Structural and analytic oracles, using invented inputs only."""

import copy
import json

import numpy as np
import pytest

from healthcare_sim_sdk.core.engine import BranchedSimulationEngine, CounterfactualMode
from healthcare_sim_sdk.experiments.benchmark.shield_rt import (
    ShieldRTBenchmarkScenario, run_case, validate_inputs,
)


def fixed(value):
    return {"distribution": "fixed", "value": value}


def uniform(low, high):
    return {"distribution": "uniform", "low": low, "high": high}


def case(p0=0.3, effect=0.4, uptake=0.6, n_control=101, n_intervention=97):
    return {"inputs": {
        key: {"value": value, "source_id": "synthetic-test-only"}
        for key, value in {
            "n_control": n_control, "n_intervention": n_intervention,
            "baseline_risk": fixed(p0), "relative_effect": fixed(effect),
            "uptake": fixed(uptake),
        }.items()
    }}


def scenario(spec, seed=7):
    return ShieldRTBenchmarkScenario(**validate_inputs(spec), seed=seed)


@pytest.mark.parametrize("p0,effect,uptake,expected", [
    (0, 0.5, 1, (0, 0)),
    (1, 0, 1, (1, 1)),
    (1, 1, 1, (1, 0)),
    (1, 1, 0, (1, 1)),
    (0, -2, 1, (0, 0)),
])
def test_exact_boundary_outcomes(p0, effect, uptake, expected):
    result = run_case(case(p0, effect, uptake), 3)
    assert result["status"] == "PASS"
    assert all(check["status"] == "PASS" for check in result["checks"])
    assert result["predictions"]["control_risk"] == expected[0]
    assert result["predictions"]["intervention_risk"] == expected[1]
    assert result["predictions"]["risk_difference"] == expected[1] - expected[0]
    assert result["baselines"]["risk_difference"] == 0


@pytest.mark.parametrize("effect,uptake", [(0.4, 0.6), (-1.0, 0.7)])
def test_realized_trial_matches_analytic_oracle(effect, uptake):
    result = run_case(case(0.4, effect, uptake, 100_000, 120_000), 11)
    metrics = result["metrics"]
    expected = -0.4 * effect * uptake
    assert metrics["analytic_expected_difference"] == pytest.approx(expected)
    standard_error = np.sqrt(metrics["analytic_sampling_variance"])
    assert abs(result["predictions"]["risk_difference"] - expected) < 6 * standard_error
    assert metrics["treated_event_probability"] == pytest.approx(0.4 * (1 - effect))


def test_null_is_an_independent_arm_trial_not_forced_zero_difference():
    results = [run_case(case(effect=0), seed) for seed in range(32)]
    differences = [r["predictions"]["risk_difference"] for r in results]
    assert any(d != 0 for d in differences)
    assert any(d > 0 for d in differences) and any(d < 0 for d in differences)
    variance = results[0]["metrics"]["analytic_sampling_variance"]
    assert abs(np.mean(differences)) < 6 * np.sqrt(variance / len(results))
    assert all(r["metrics"]["analytic_expected_difference"] == 0 for r in results)
    assert any(
        r["baselines"]["risk_difference"] != d for r, d in zip(results, differences)
    )


def test_exact_randomization_no_rethresholding_and_no_compounding():
    sc = scenario(case(uptake=1))
    initial = sc.create_population(198)
    assert initial.assigned_intervention.sum() == 97
    assert np.unique(initial.assigned_intervention).size == 2
    predictions = sc.predict(initial, 0)
    predictions.scores[:] = -999  # Scores cannot exclude an already eligible course.
    treated, actions = sc.intervene(initial, predictions, 0)
    assert len(actions.treated_indices) == 97
    assert not treated.received_extra_evaluation[~treated.assigned_intervention].any()
    assert not initial.received_extra_evaluation.any()
    np.testing.assert_array_equal(initial.event_probability, np.full(198, 0.3))
    treated_twice, _ = sc.intervene(treated, predictions, 0)
    np.testing.assert_array_equal(
        treated_twice.event_probability, treated.event_probability
    )
    np.testing.assert_array_equal(treated_twice.received_extra_evaluation,
                                  treated.received_extra_evaluation)
    with pytest.raises(ValueError, match="exactly one"):
        sc.step(treated, 1)


def test_measure_is_repeatable_without_state_or_output_alias_mutation():
    sc = scenario(case())
    initial = sc.create_population(198)
    state, _ = sc.intervene(initial, sc.predict(initial, 0), 0)
    before = sc.clone_state(state)
    rng_before = copy.deepcopy(sc.rng.outcomes.bit_generator.state)
    first = sc.measure(state, 0)
    second = sc.measure(state, 0)
    np.testing.assert_array_equal(first.events, second.events)
    np.testing.assert_array_equal(state.event_probability, before.event_probability)
    np.testing.assert_array_equal(state.outcome_uniform, before.outcome_uniform)
    assert rng_before == sc.rng.outcomes.bit_generator.state
    first.events[:] = 99
    first.secondary["event_probability"][:] = 99
    third = sc.measure(state, 0)
    np.testing.assert_array_equal(third.events, second.events)
    np.testing.assert_array_equal(state.event_probability, before.event_probability)


def test_engine_uses_one_episode_without_counterfactual_branch():
    sc = scenario(case())
    results = BranchedSimulationEngine(sc, CounterfactualMode.NONE).run(198)
    assert list(results.outcomes) == [0]
    assert not results.counterfactual_outcomes
    assert len(np.unique(results.outcomes[0].entity_ids)) == 198
    with pytest.raises(ValueError, match="registered arm sizes"):
        sc.create_population(197)


def test_uncertainty_is_shared_once_per_replicate_and_reproducible():
    spec = case()
    for name, bounds in {
        "baseline_risk": (0.1, 0.4), "relative_effect": (-0.5, 0.6),
        "uptake": (0.2, 0.9),
    }.items():
        spec["inputs"][name]["value"] = uniform(*bounds)
    original = copy.deepcopy(spec)
    a, b = run_case(spec, 19), run_case(spec, 19)
    assert a == b
    assert spec == original
    assert a["metrics"] != run_case(spec, 20)["metrics"]
    sc = scenario(spec)
    state = sc.create_population(198)
    state, _ = sc.intervene(state, sc.predict(state, 0), 0)
    assert len(np.unique(state.event_probability)) <= 2
    assert np.all(state.event_probability[~state.received_extra_evaluation]
                  == state.baseline_risk)
    json.dumps(a, allow_nan=False)


def test_reference_baseline_does_not_depend_on_assumed_effect():
    a = run_case(case(effect=0.8), 27)
    b = run_case(case(effect=-0.8), 27)
    assert a["baselines"] == b["baselines"]
    assert a["metrics"]["null_control_events"] == b["metrics"]["null_control_events"]
    assert (
        a["metrics"]["null_intervention_events"]
        == b["metrics"]["null_intervention_events"]
    )


@pytest.mark.parametrize("name,value", [
    ("n_control", True), ("n_intervention", 0), ("n_control", 1.5),
    ("baseline_risk", fixed(float("nan"))),
    ("baseline_risk", fixed(1.01)), ("baseline_risk", fixed(-0.01)),
    ("uptake", uniform(0, 1.1)), ("uptake", fixed(float("inf"))),
    ("relative_effect", fixed(float("nan"))),
    ("relative_effect", fixed(1.01)),
    ("baseline_risk", {"distribution": "normal", "mean": 0.2}),
    ("baseline_risk", uniform(0.4, 0.2)),
    ("baseline_risk", fixed(True)),
])
def test_invalid_inputs_fail_before_simulation(name, value):
    spec = case()
    spec["inputs"][name]["value"] = value
    with pytest.raises(ValueError):
        validate_inputs(spec)


def test_invalid_joint_support_rejected_even_when_most_draws_would_be_valid():
    spec = case(uptake=0)
    spec["inputs"]["baseline_risk"]["value"] = uniform(0.1, 0.8)
    spec["inputs"]["relative_effect"]["value"] = uniform(-0.5, 0.5)
    with pytest.raises(ValueError, match="Joint"):
        validate_inputs(spec)


def test_no_implicit_inputs_or_missing_provenance():
    spec = case()
    del spec["inputs"]["uptake"]
    with pytest.raises(ValueError, match="five"):
        validate_inputs(spec)
    spec = case()
    spec["inputs"]["baseline_risk"]["source_id"] = " "
    with pytest.raises(ValueError, match="source_id"):
        validate_inputs(spec)
    spec = case()
    spec["inputs"]["empirical_outcome"] = {"value": 0.2, "source_id": "sealed"}
    with pytest.raises(ValueError, match="five"):
        validate_inputs(spec)


@pytest.mark.parametrize("seed", [-1, True, 1.5, 2**32])
def test_seed_domain_is_separate_from_null_reference(seed):
    with pytest.raises(ValueError, match="seed"):
        run_case(case(), seed)
