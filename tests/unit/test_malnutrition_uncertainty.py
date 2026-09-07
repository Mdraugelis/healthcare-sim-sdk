"""Do not silently condition analyses on available files or finite seeds."""

import pytest

from healthcare_sim_sdk.scenarios.malnutrition_prioritization.estimands import (
    _agg_scalar,
)
from healthcare_sim_sdk.scenarios.malnutrition_prioritization.outcome_overlay import (
    _agg, expected_seed_files,
)


def test_undefined_stratum_does_not_drop_a_seed():
    with pytest.raises(ValueError, match="no seeds omitted"):
        _agg_scalar([{"effect": 1}, {"effect": float("nan")}], "effect")


def test_missing_seed_file_rejected(tmp_path):
    (tmp_path / "seed_0001.npz").touch()
    with pytest.raises(ValueError, match="seed arrays"):
        expected_seed_files(tmp_path, {"seeds": [1, 2]})
    with pytest.raises(ValueError, match="exact seed IDs"):
        expected_seed_files(tmp_path, {"n_seeds": 1})


def test_overlay_separate_mean_error_from_seed_spread():
    result = _agg([{"grid": {"point": x}} for x in [1, 2, 3]], "grid")["point"]
    assert result["n_seeds"] == 3
    assert result["seed_sd"] == 1
    assert result["mcse_mean"] == pytest.approx(1 / 3 ** .5)
    assert result["mean_ci95_low"] < result["seed_p025"]
