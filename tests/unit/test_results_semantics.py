"""Hand-worked causal-export fixtures: values, identity and exposure timing."""
import numpy as np
import pytest

from healthcare_sim_sdk.core.results import SimulationResults
from healthcare_sim_sdk.core.scenario import Outcomes, Predictions, Interventions, TimeConfig


def golden_results():
    r = SimulationResults(time_config=TimeConfig(4, 1), unit_of_analysis="patient")
    # Different order in each period makes a positional-ID mixup observable.
    for t, ids, events, costs in [
        (0, [101, 909], [1, 0], [10, 20]),
        (1, [909, 101], [0, 0], [30, 40]),
        (2, [101, 909], [0, 1], [50, 60]),
        (3, [909, 101], [1, 1], [70, 80]),
    ]:
        out = Outcomes(np.array(events), np.array(ids),
                       secondary={"cost": np.array(costs)},
                       metadata={"group": ["A" if i == 101 else "B" for i in ids]})
        r.record_outcomes(t, out)
        r.record_outcomes(t, out, counterfactual=True)
    r.record_interventions(0, Interventions(np.array([0])))  # 101 at t=0
    r.record_interventions(2, Interventions(np.array([1])))  # 909 at t=2
    return r


def test_hand_computed_costs_in_every_export_and_branch():
    r = golden_results()
    a = r.to_analysis()
    for branch in ("factual", "counterfactual"):
        np.testing.assert_array_equal(a.to_time_series("cost", branch)["outcomes"], [30, 70, 110, 150])
        np.testing.assert_array_equal(a.to_panel("cost", branch)["outcomes"], [10, 20, 30, 40, 50, 60, 70, 80])
        np.testing.assert_array_equal(a.to_entity_snapshots(1, "cost", branch)["outcomes"], [30, 40])
        np.testing.assert_array_equal(a.to_subgroup_panel("cost", "group", branch)["subgroup"],
                                      ["A", "B", "B", "A", "A", "B", "B", "A"])
    np.testing.assert_array_equal(r.get_outcome_series("cost"), [30, 70, 110, 150])
    np.testing.assert_array_equal(r.get_counterfactual_outcome_series("cost"), [30, 70, 110, 150])


def test_staggered_identity_and_assignment_semantics():
    a = golden_results().to_analysis()
    np.testing.assert_array_equal(a.to_panel()["treated"], [1, 0, 0, 1, 1, 1, 1, 1])
    np.testing.assert_array_equal(a.to_panel(treatment_semantics="current_step")["treated"],
                                  [1, 0, 0, 0, 0, 1, 0, 0])
    assert not a.to_panel(branch="counterfactual")["treated"].any()
    assert not a.to_time_series(branch="counterfactual")["treatment_indicator"].any()


def test_prediction_identity_controls_lagged_slot_assignments():
    r = SimulationResults(time_config=TimeConfig(2, 1))
    r.record_predictions(0, Predictions(np.array([.1, .9]), metadata={"slot_ids": [100, 900]}))
    r.record_interventions(0, Interventions(np.array([1])))
    r.record_outcomes(0, Outcomes(np.array([0]), np.array([42])))
    r.record_outcomes(1, Outcomes(np.array([1, 0]), np.array([900, 100])))
    np.testing.assert_array_equal(r.to_analysis().to_panel()["treated"], [0, 1, 0])
    assert np.isnan(r.to_analysis().to_entity_snapshots(0)["scores"]).all()


def test_snapshots_align_scores_by_explicit_ids_and_omit_counterfactual_scores():
    r = golden_results()
    r.record_predictions(0, Predictions(np.array([.8, .2]), metadata={"entity_ids": [909, 101]}))
    np.testing.assert_array_equal(r.to_analysis().to_entity_snapshots(0)["scores"], [.2, .8])
    assert r.to_analysis().to_entity_snapshots(0, branch="counterfactual")["scores"] is None


def test_missing_measurements_are_unknown_but_empty_observed_is_zero():
    r = SimulationResults(time_config=TimeConfig(3, 1))
    r.record_outcomes(1, Outcomes(np.array([]), np.array([], dtype=int)))
    ts = r.to_analysis().to_time_series()
    np.testing.assert_array_equal(ts["observed"], [False, True, False])
    np.testing.assert_array_equal(ts["outcomes"], [np.nan, 0, np.nan])
    assert len(r.to_analysis().to_panel()["outcomes"]) == 0
    assert len(SimulationResults().to_analysis().to_subgroup_panel()["subgroup"]) == 0
    inferred = SimulationResults(outcomes={2: Outcomes(np.array([4]))})
    np.testing.assert_array_equal(inferred.get_outcome_series(), [np.nan, np.nan, 4])


def test_scheduled_assignment_does_not_imply_any_entity_was_treated():
    r = golden_results()
    r.interventions[0].treated_indices = np.array([], dtype=int)
    np.testing.assert_array_equal(r.get_treatment_indicator(), [1, 0, 1, 0])
    np.testing.assert_array_equal(r.get_treatment_indicator(treatment_semantics="any_treated"), [0, 0, 1, 0])


@pytest.mark.parametrize("export", ["to_time_series", "to_panel", "to_subgroup_panel", "to_entity_snapshots"])
def test_unknown_outcome_or_branch_is_rejected(export):
    fn = getattr(golden_results().to_analysis(), export)
    args = {"t": 0} if export == "to_entity_snapshots" else {}
    with pytest.raises(ValueError, match="not found"):
        fn(outcome_key="typo", **args)
    with pytest.raises(ValueError, match="branch"):
        fn(branch="typo", **args)


@pytest.mark.parametrize("mutation,match", [
    (lambda r: setattr(r.outcomes[3], "entity_ids", None), "entity_ids"),
    (lambda r: setattr(r.outcomes[3], "entity_ids", np.array([101, 101])), "unique"),
    (lambda r: r.outcomes[3].secondary.update(cost=np.array([1])), "length"),
    (lambda r: setattr(r.interventions[0], "treated_indices", np.array([-1])), "nonnegative"),
    (lambda r: setattr(r.interventions[0], "treated_indices", np.array([0.5])), "integer"),
    (lambda r: setattr(r.interventions[0], "treated_indices", np.array([2])), "outside"),
    (lambda r: r.record_predictions(0, Predictions(np.array([.1]))), "length"),
])
def test_invalid_alignment_fails_loudly(mutation, match):
    r = golden_results()
    mutation(r)
    with pytest.raises(ValueError, match=match):
        r.to_analysis().to_panel("cost")


def test_invalid_subgroups_and_timesteps_fail_loudly():
    r = golden_results()
    r.outcomes[3].metadata["group"] = ["A"]
    with pytest.raises(ValueError, match="length"):
        r.to_analysis().to_subgroup_panel(subgroup_key="group")
    r.record_outcomes(-1, Outcomes(np.array([1])))
    with pytest.raises(ValueError, match="nonnegative"):
        r.get_outcome_series()


def test_explicit_unobserved_flag_cannot_hide_nonempty_outcomes():
    r = SimulationResults(outcomes={
        0: Outcomes(np.array([1]), metadata={'outcome_observed': False})})
    with pytest.raises(ValueError, match='no entity values'):
        r.get_outcome_series()
    r.outcomes[0] = Outcomes(np.array([]), metadata={'outcome_observed': 'false'})
    with pytest.raises(ValueError, match='boolean'):
        r.get_outcome_series()


def test_nonfinite_entity_identity_is_rejected():
    r = golden_results()
    r.outcomes[0].entity_ids = np.array([101, np.nan])
    with pytest.raises(ValueError, match='nonfinite'):
        r.to_analysis().to_panel()
