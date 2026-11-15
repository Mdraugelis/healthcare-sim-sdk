"""Held-out scoring-path checks and honest missing-evidence diagnostics."""

import copy
import json

import numpy as np
import pytest

from healthcare_sim_sdk.core.rng import RNGPartitioner
from healthcare_sim_sdk.core.scenario import TimeConfig
from healthcare_sim_sdk.ml.fidelity import auc_fidelity
from healthcare_sim_sdk.scenarios.sepsis_early_alert.scenario import (
    ROW_CURRENT_RISK, ROW_STAGE, STAGE_AT_RISK, STAGE_DECEASED,
    STAGE_DISCHARGED, STAGE_SEPSIS, SepsisConfig, SepsisEarlyAlertScenario,
)


def test_hand_computed_auc_and_target_miss():
    # Three concordant pairs of four positive-negative pairs gives AUC 0.75.
    diagnostic = auc_fidelity(np.array([0, 1, 0, 1]), np.array([0.1, 0.4, 0.5, 0.9]),
                              0.75, tolerance=0, min_class_count=2)
    assert diagnostic["achieved_auc"] == 0.75 and diagnostic["status"] == "PASS"
    swapped = auc_fidelity(np.array([0, 1, 0, 1]), np.array([0.9, 0.5, 0.4, 0.1]),
                           0.75, tolerance=0.05, min_class_count=2)
    assert swapped["achieved_auc"] == 0.25 and swapped["status"] == "FAIL"


@pytest.mark.parametrize("labels", [np.array([]), np.zeros(100), np.ones(100)])
def test_undefined_auc_does_not_become_chance_or_pass(labels):
    diagnostic = auc_fidelity(labels, np.zeros(labels.size), 0.5)
    assert diagnostic["status"] == "INCONCLUSIVE"
    assert diagnostic["achieved_auc"] is None
    json.dumps(diagnostic, allow_nan=False)


def test_few_positives_inconclusive_even_if_point_target_matches():
    diagnostic = auc_fidelity(np.array([0, 1]), np.array([0.1, 0.9]), 1.0)
    assert diagnostic["achieved_auc"] == 1 and diagnostic["status"] == "INCONCLUSIVE"


def test_auc_is_exact_in_narrow_score_bands_and_credits_ties():
    labels = np.array([0, 1, 0, 1])
    scores = np.array([0.001, 0.0014, 0.0015, 0.0019])
    assert auc_fidelity(labels, scores, 0.75, min_class_count=2)["achieved_auc"] == 0.75
    assert auc_fidelity(labels, np.full(4, 0.001), 0.5, min_class_count=2)["achieved_auc"] == 0.5


@pytest.mark.parametrize("scores", [
    np.array([0.1, np.nan]), np.array([0.1, np.inf]),
    np.array([0.1, 1.1]), np.array([0.1]), np.array([[0.1, 0.9]]),
])
def test_invalid_scores_fail(scores):
    assert auc_fidelity(np.array([0, 1]), scores, 0.75)["status"] == "FAIL"


def _scenario(seed):
    return SepsisEarlyAlertScenario(
        TimeConfig(n_timesteps=2, timestep_duration=4 / 8760,
                   timestep_unit="4hr_block", prediction_schedule=[0, 1]),
        seed=seed, config=SepsisConfig(model_auc=0.76),
    )


def _draw_labels(state, seed):
    """Outcomes drawn from current risk on a stream the model never sees."""
    rng = RNGPartitioner(seed).create_streams().outcomes
    return (rng.random(state.shape[1]) < state[ROW_CURRENT_RISK]).astype(int)


def test_predict_is_label_free(monkeypatch):
    # main's ControlledMLModel scores from true risk only; the scenario must
    # never pass outcome labels into predict().
    scenario = _scenario(72)
    state = scenario.create_population(400)
    state[ROW_STAGE, :30] = STAGE_SEPSIS
    state[ROW_STAGE, 30:40] = STAGE_DECEASED
    state[ROW_STAGE, 40:50] = STAGE_DISCHARGED
    scenario.predict(state, 0)  # fit on first call
    seen = {}
    original = scenario._model.predict

    def scoring(risks, rng, *args, **kwargs):
        seen["args"], seen["kwargs"] = args, kwargs
        return original(risks, rng)

    monkeypatch.setattr(scenario._model, "predict", scoring)
    predicted = scenario.predict(state, 1)
    assert seen["args"] == () and "true_labels" not in seen["kwargs"]
    # Deceased and discharged patients are not scored.
    assert np.all(predicted.scores[30:50] == 0)


def test_frozen_sepsis_model_meets_target_on_independent_cohort():
    scenario = _scenario(41)
    training = scenario.create_population(4000)
    scenario.predict(training, 0)  # label-free fit on this population
    fitted = copy.deepcopy(scenario._model._fit_report)
    assert scenario._model_fitted

    # Independent population (different seed), scored through the real
    # scenario.predict path with the model frozen.
    heldout = _scenario(907).create_population(20000)
    labels = _draw_labels(heldout, 4242)
    heldout[ROW_STAGE] = np.where(labels == 1, STAGE_SEPSIS, STAGE_AT_RISK)
    predicted = scenario.predict(heldout, 1)
    assert scenario._model._fit_report == fitted  # not refitted
    diagnostic = auc_fidelity(labels, predicted.scores, scenario.cfg.model_auc,
                              independent_cohort=True,
                              scope="independent same-distribution synthetic cohort")
    assert diagnostic["status"] == "PASS", diagnostic
    assert diagnostic["independent_cohort"] is True
