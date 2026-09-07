"""Held-out scoring-path checks and honest missing-evidence diagnostics."""

import copy
import json

import numpy as np
import pytest

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


def test_unfitted_single_class_is_explicit_and_later_fit_is_possible():
    scenario = _scenario(72)
    state = scenario.create_population(100)
    predicted = scenario.predict(state, 0)
    assert not scenario._model_fitted
    diagnostic = predicted.metadata["model_fidelity"]
    assert diagnostic["status"] == "NOT_RUN"
    assert diagnostic["fit_auc"] is None and diagnostic["achieved_auc"] is None
    assert diagnostic["n_positive"] == 0


def test_actual_predict_receives_labels_and_excludes_inactive(monkeypatch):
    scenario = _scenario(72)
    state = scenario.create_population(100)
    state[ROW_STAGE, :30] = STAGE_SEPSIS
    state[ROW_STAGE, 30:40] = STAGE_DECEASED
    state[ROW_STAGE, 40:50] = STAGE_DISCHARGED
    scenario._model_fitted = True
    seen = {}

    def scoring(risks, rng, true_labels=None):
        seen["labels"] = true_labels
        return np.where(true_labels == 1, 0.9, 0.1)

    monkeypatch.setattr(scenario._model, "predict", scoring)
    predicted = scenario.predict(state, 1)
    assert seen["labels"] is not None
    assert seen["labels"].sum() == 30
    assert np.all(predicted.scores[30:50] == 0)
    diagnostic = predicted.metadata["model_fidelity"]
    assert diagnostic["n"] == 80 and diagnostic["n_positive"] == 30
    assert diagnostic["n_negative"] == 50
    assert diagnostic["status"] == "FAIL"  # Perfect scores miss the configured 0.76 target.
    assert not diagnostic["independent_cohort"]


def test_frozen_sepsis_model_meets_target_on_independent_same_distribution_cohort():
    scenario = _scenario(41)
    training = scenario.create_population(2000)
    labels_rng = np.random.default_rng(713)
    training[ROW_STAGE] = np.where(labels_rng.random(2000) < training[ROW_CURRENT_RISK],
                                   STAGE_SEPSIS, STAGE_AT_RISK)
    scenario.predict(training, 0)
    fitted_report = copy.deepcopy(scenario._model_fit_report)
    assert scenario._model_fitted

    # An independent scenario/population seed creates the holdout. The fitted
    # model is frozen, and scoring goes through the real scenario.predict path.
    independent = _scenario(907)
    heldout = independent.create_population(8000)
    heldout[ROW_STAGE] = np.where(labels_rng.random(8000) < heldout[ROW_CURRENT_RISK],
                                  STAGE_SEPSIS, STAGE_AT_RISK)
    predicted = scenario.predict(heldout, 1)
    assert scenario._model_fit_report == fitted_report
    diagnostic = auc_fidelity((heldout[ROW_STAGE] == STAGE_SEPSIS).astype(int),
                              predicted.scores, scenario.cfg.model_auc,
                              independent_cohort=True,
                              scope="independent same-distribution synthetic cohort")
    assert diagnostic["status"] == "PASS", diagnostic
    assert abs(diagnostic["achieved_auc"] - 0.76) <= 0.05
    # Fit metrics alone are not used as proof of this held-out result.
    assert predicted.metadata["model_fidelity"]["achieved_auc"] == diagnostic["achieved_auc"]
