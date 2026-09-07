"""A fidelity gate may not pass with invalid target or acceptance parameters."""

import numpy as np
import pytest

from healthcare_sim_sdk.scenarios.nurse_retention.model_verification import (
    score_diagnostics,
)


@pytest.mark.parametrize("invalid", [
    {"target_auc": np.nan}, {"target_auc": np.inf},
    {"target_auc": -0.01}, {"target_auc": 1.5, "tolerance": 1.0},
    {"tolerance": np.nan}, {"tolerance": np.inf}, {"tolerance": -0.1},
    {"min_positives": np.nan}, {"min_positives": 1.5},
    {"min_positives": True}, {"min_positives": 0},
    {"min_negatives": np.nan}, {"min_negatives": 1.5},
    {"min_negatives": True}, {"min_negatives": 0},
])
def test_invalid_acceptance_parameters_cannot_return_pass(invalid):
    kwargs = {"target_auc": 1.0, "tolerance": 0.05,
              "min_positives": 1, "min_negatives": 1, **invalid}
    with pytest.raises(ValueError):
        score_diagnostics(np.array([0, 1]), np.array([0.1, 0.9]),
                          label_provenance="test", **kwargs)


def test_valid_fidelity_parameter_boundaries():
    actual = score_diagnostics(np.array([0, 1]), np.array([0.1, 0.9]),
                               1.0, tolerance=0, min_positives=1,
                               min_negatives=1, label_provenance="test")
    assert actual["status"] == "PASS"
