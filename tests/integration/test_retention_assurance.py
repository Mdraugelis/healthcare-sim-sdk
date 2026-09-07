"""Semantic guarantees for retention coupling, accounting, and monitoring."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from healthcare_sim_sdk.core.engine import BranchedSimulationEngine
from healthcare_sim_sdk.core.scenario import Outcomes, Predictions
from healthcare_sim_sdk.scenarios.nurse_retention.model_verification import (
    score_diagnostics, verify_frozen_scoring_path,
)
from healthcare_sim_sdk.scenarios.nurse_retention.monitoring import MonitoringHarness
from healthcare_sim_sdk.scenarios.nurse_retention.scenario import (
    NurseRetentionScenario, RetentionConfig,
)


class LabelScorer:
    """A deliberately label-dependent frozen scorer; no fit API available."""
    def predict(self, risk, rng, labels):
        return np.where(labels, 0.8, 0.2)


def _fast_scenario(**kwargs):
    scenario = NurseRetentionScenario(RetentionConfig(**kwargs), seed=17)
    scenario._classifier = LabelScorer()
    scenario._classifier_fitted = True
    scenario._fit_report = {"achieved_auc": 0.8}
    return scenario


@pytest.mark.parametrize("divergence", [
    "departures", "cooldowns", "tenure", "all_ineligible",
])
def test_temporal_draws_remain_aligned_after_state_diverges(divergence):
    factual = _fast_scenario(n_nurses=80, nurses_per_manager=20)
    control = _fast_scenario(n_nurses=80, nurses_per_manager=20)
    f_state = factual.create_population(80)
    c_state = control.create_population(80)
    if divergence == "departures":
        c_state.departed[::2] = True
    elif divergence == "cooldowns":
        c_state.soc_weeks_since_intervention[::3] = 0
    elif divergence == "tenure":
        c_state.tenure_weeks[:] = 100
    else:
        c_state.departed[:] = True
    for t in range(8):
        f_state = factual.step(f_state, t)
        c_state = control.step(c_state, t)
        assert (factual.rng.temporal.bit_generator.state
                == control.rng.temporal.bit_generator.state)
        np.testing.assert_array_equal(f_state.ar1_modifier, c_state.ar1_modifier)


def test_temporal_draws_remain_aligned_with_different_capacity():
    a = _fast_scenario(n_nurses=80, max_interventions_per_manager_per_week=0)
    b = _fast_scenario(n_nurses=80, max_interventions_per_manager_per_week=5)
    a.step(a.create_population(80), 0)
    b.step(b.create_population(80), 0)
    assert a.rng.temporal.bit_generator.state == b.rng.temporal.bit_generator.state


@pytest.mark.parametrize("overrides", [
    {"intervention_effectiveness": 0.0},
    {"max_interventions_per_manager_per_week": 0},
    {"manager_adherence_rate": 0.0},
])
def test_null_mechanisms_preserve_exact_branch_outcomes(overrides):
    scenario = _fast_scenario(n_nurses=150, n_weeks=20, **overrides)
    result = BranchedSimulationEngine(scenario).run(150)
    for t in result.outcomes:
        np.testing.assert_array_equal(
            result.outcomes[t].events, result.counterfactual_outcomes[t].events,
        )
        np.testing.assert_array_equal(
            result.outcomes[t].secondary["current_risk"],
            result.counterfactual_outcomes[t].secondary["current_risk"],
        )


def test_capacity_and_eligibility_for_each_separate_budget():
    scenario = _fast_scenario(n_nurses=80, nurses_per_manager=20,
                              max_interventions_per_manager_per_week=3)
    state = scenario.create_population(80)
    state.departed[:10] = True
    state.soc_weeks_since_intervention[10:20] = 0
    state.weeks_since_intervention[10:20] = 0
    after_soc = scenario.step(state, 0)
    selected_soc = after_soc.soc_weeks_since_intervention == 0
    assert not selected_soc[:20].any()
    assert np.bincount(state.manager_id[selected_soc], minlength=4).max() <= 3
    predictions = scenario.predict(after_soc, 0)
    after_ai, actions = scenario.intervene(after_soc, predictions, 0)
    assert not set(actions.treated_indices).intersection(range(20))
    ai_counts = np.bincount(state.manager_id[actions.treated_indices], minlength=4)
    assert ai_counts.max() <= 3
    meta = scenario.measure(after_ai, 0).metadata
    assert meta["capacity_model"] == "separate_additive_soc_and_ai_budgets"
    assert sum(meta["soc_per_manager_this_week"]) == int(selected_soc.sum())


def _handworked_results(n_weeks=13):
    outcomes, predictions = {}, {}
    departed = np.zeros(4, dtype=bool)
    for t in range(n_weeks):
        events = np.zeros(4)
        if t in (0, 1, 2):
            events[[0, 2, 1][t]] = 1
            departed |= events.astype(bool)
        outcomes[t] = Outcomes(
            events=events, entity_ids=np.arange(4),
            secondary={"active_mask": (~departed).astype(float)},
            metadata={"t": t, "n_nurses": 4, "n_managers": 2,
                      "n_managers_adopting": 1, "n_active": int((~departed).sum()),
                      "total_departures": int(departed.sum()),
                      "total_soc_interventions": 0,
                      "retention_rate": float((~departed).mean())},
        )
    predictions[0] = Predictions(
        scores=np.array([0.0, 0.6, 0.9, 0.1]), labels=np.array([0, 1, 1, 1]),
        metadata={"true_labels": np.zeros(4),
                  "active_mask": np.array([False, True, True, True]),
                  "prediction_horizon_weeks": 2},
    )
    return SimpleNamespace(outcomes=outcomes, predictions=predictions, interventions={})


def test_observable_mode_requires_no_counterfactual_and_is_invariant_to_perturbation():
    config = RetentionConfig(n_nurses=4, nurses_per_manager=2, n_weeks=13)
    result = _handworked_results()
    harness = MonitoringHarness(tier3_mode="its_only")
    baseline = harness.run_from_results(result, config)
    result.counterfactual_outcomes = {0: "corrupt and incomplete withheld data"}
    perturbed = harness.run_from_results(result, config)
    assert baseline.to_dict() == perturbed.to_dict()
    assert baseline.ground_truth_counterfactual_departures is None
    assert len(baseline.tier3_estimates) == 1
    assert all(not any("cf" in key for key in row) for row in baseline.weekly_history)


def test_dashboard_adoption_and_initial_denominator_are_factual():
    config = RetentionConfig(n_nurses=4, nurses_per_manager=2, n_weeks=13)
    result = _handworked_results()
    run = MonitoringHarness(tier3_mode="its_only").run_from_results(result, config)
    assert run.weekly_history[0]["unit_turnover_rate"] == 1 / 4
    assert run.weekly_history[1]["unit_turnover_rate"] == 1 / 3
    assert all(row["adoption_rate"] == 0.5 for row in run.weekly_history)


def test_missing_adoption_is_disclosed_not_assumed_complete():
    result = _handworked_results()
    for outcome in result.outcomes.values():
        del outcome.metadata["n_managers_adopting"]
    config = RetentionConfig(n_nurses=4, nurses_per_manager=2, n_weeks=13)
    run = MonitoringHarness(tier3_mode="its_only").run_from_results(result, config)
    assert run.metadata["missing_adoption_weeks"] == list(range(13))
    assert all(row["adoption_rate"] is None for row in run.weekly_history)


def test_tier4_uses_matured_future_events_not_synthetic_labels():
    config = RetentionConfig(n_nurses=4, nurses_per_manager=2, n_weeks=13)
    result = _handworked_results()
    run = MonitoringHarness(tier3_mode="its_only").run_from_results(result, config)
    entry, = run.tier4_prediction_log
    assert entry["week"] == 2  # available after the entire prediction horizon
    assert entry["score_week"] == 0
    np.testing.assert_array_equal(entry["true_labels"], [1, 1, 0])
    np.testing.assert_array_equal(entry["scores"], [0.6, 0.9, 0.1])
    performance = run.metadata["prediction_performance"][0]
    assert performance["actual_future_outcomes_status"] == "INCONCLUSIVE"
    short = _handworked_results(n_weeks=2)
    harness = MonitoringHarness(tier3_mode="its_only")
    incomplete = harness.run_from_results(short, config)
    assert incomplete.tier4_prediction_log == []
    performance = incomplete.metadata["prediction_performance"][0]
    assert performance["actual_future_outcomes_status"] == "NOT_RUN"


def test_deployed_score_diagnostic_can_fail_despite_fitted_auc():
    scenario = _fast_scenario(n_nurses=5000)
    before = copy.deepcopy(scenario.rng)
    result = verify_frozen_scoring_path(scenario, validation_seed=123, n_entities=5000)
    assert result["status"] == "FAIL"  # frozen LabelScorer yields AUC 1 vs target .8
    assert result["auc"] == 1.0
    assert result["frozen_model"] and not result["refit"]
    assert result["label_provenance"] == "synthetic_prediction_window_labels"
    assert result["actual_future_outcomes_status"] == "NOT_RUN"
    for stream in ("population", "temporal", "prediction", "intervention", "outcomes"):
        assert (getattr(before, stream).bit_generator.state
                == getattr(scenario.rng, stream).bit_generator.state)
    with pytest.raises(ValueError, match="must differ"):
        verify_frozen_scoring_path(scenario, validation_seed=17)


def test_degenerate_or_small_class_counts_do_not_pass_model_gate():
    assert score_diagnostics(np.ones(10), np.ones(10), 1.0,
                             label_provenance="test")["status"] == "INCONCLUSIVE"
    assert score_diagnostics(np.array([0, 1]), np.array([np.nan, 1]), 1.0,
                             label_provenance="test")["status"] == "FAIL"


@pytest.mark.parametrize("fault", [None, "missing_prediction", "id_order"])
def test_monitoring_packet_binds_actual_cases_and_keeps_missing_certification(
    tmp_path, fault,
):
    import json
    from dataclasses import asdict

    from healthcare_sim_sdk.experiments.review import verify_packet
    from healthcare_sim_sdk.scenarios.nurse_retention.monitoring import (
        MonitoringClaim, evaluate_claims,
    )
    from healthcare_sim_sdk.scenarios.nurse_retention.monitoring.review import (
        write_monitoring_review_packet,
    )
    scenario = _fast_scenario(n_nurses=80, n_weeks=6, intervention_effectiveness=0)
    results = BranchedSimulationEngine(scenario).run(80)
    harness = MonitoringHarness(regime="null", seed=17, tier3_mode="its_only")
    run = harness.run_from_results(results, scenario.config)
    run.metadata["monitoring_claim_evaluations"] = evaluate_claims(run, [
        MonitoringClaim("null", 1, "check_ins_done", "any", None, 5),
    ])
    run.metadata["independent_frozen_scoring_validation"] = {
        "status": "INCONCLUSIVE", "reason": "too few positives in this small test",
    }
    if fault == "missing_prediction":
        del results.predictions[min(results.predictions)]
    elif fault == "id_order":
        results.outcomes[1].entity_ids = results.outcomes[1].entity_ids[::-1]
    packet = write_monitoring_review_packet(
        results, scenario.config, run, asdict(scenario.config),
        {"departures_prevented": 0, "factual_retention_rate": 0.9,
         "tier1_n_events": 0}, tmp_path,
    )
    assert packet.exists()
    verified = {check["name"]: check["status"] for check in verify_packet(tmp_path)}
    assert verified["artifact_integrity"] == "PASS"
    if fault:
        assert verified["declared_machine_evidence"] == "FAIL"
        return
    assert "FAIL" not in verified.values()
    evidence_json = json.loads((tmp_path / "evidence.json").read_text())
    claims = {claim["id"]: claim for claim in evidence_json["claims"]}
    assert claims["run_integrity"]["status"] == "PASS"
    assert claims["operating_characteristics"]["status"] == "NOT_RUN"
    assert claims["workflow_plausibility"]["status"] == "NOT_RUN"
    traces = json.loads((tmp_path / "case_traces.json").read_text())
    for row in traces:
        t, nurse = row["week"], row["nurse_id"]
        assert row["factual_departure"] == results.outcomes[t].events[nurse]
        control_event = results.counterfactual_outcomes[t].events[nurse]
        assert row["counterfactual_departure"] == control_event
    cases = evidence_json["runs"][0]["cases"]
    assert sum(case["candidate_count"] == 0 for case in cases) == 2
    assert (tmp_path / "review_operations.md").exists()
    (tmp_path / "case_traces.json").write_text("[]")
    assert verify_packet(tmp_path)[0]["status"] == "FAIL"


def test_departed_nurse_does_not_repeat_last_soc_action_in_metadata():
    scenario = _fast_scenario(n_nurses=80, nurses_per_manager=20,
                              max_interventions_per_manager_per_week=3)
    state = scenario.step(scenario.create_population(80), 0)
    selected = state.soc_intervened_this_step.copy()
    assert selected.any()
    state.departed[selected] = True
    next_state = scenario.step(state, 1)
    assert not next_state.soc_intervened_this_step[selected].any()
    meta = scenario.measure(next_state, 1).metadata
    assert max(meta["soc_per_manager_this_week"]) <= 3
    assert sum(meta["soc_per_manager_this_week"]) == (
        next_state.total_soc_interventions - state.total_soc_interventions
    )
