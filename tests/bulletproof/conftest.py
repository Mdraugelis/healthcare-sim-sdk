"""Shared fixtures and helpers for bulletproof tests.

Provides scenario runners, statistical assertion helpers, and
reusable population/model fixtures across all test modules.
"""

import numpy as np
import pytest

from healthcare_sim_sdk.core.engine import BranchedSimulationEngine, CounterfactualMode
from healthcare_sim_sdk.core.scenario import TimeConfig
from healthcare_sim_sdk.core.rng import RNGPartitioner
from healthcare_sim_sdk.ml.model import ControlledMLModel
from healthcare_sim_sdk.ml.performance import auc_score, confusion_matrix_metrics
from healthcare_sim_sdk.scenarios.stroke_prevention.scenario import (
    StrokeConfig,
    StrokePreventionScenario,
)
from healthcare_sim_sdk.scenarios.noshow_overbooking.scenario import (
    ClinicConfig,
    NoShowOverbookingScenario,
)


# ---- Statistical assertion helpers ----------------------------------------

def assert_mean_in_range(arr, expected_mean, tolerance_sigma=4.0, name=""):
    """Assert sample mean is within tolerance_sigma standard errors of expected.

    Uses CLT: stderr = std / sqrt(n). Default 4-sigma gives <0.007% false
    positive rate per check, which is tight enough for repeated runs.
    """
    n = len(arr)
    assert n > 0, f"{name}: empty array"
    sample_mean = np.mean(arr)
    sample_std = np.std(arr, ddof=1) if n > 1 else 0.0
    stderr = sample_std / np.sqrt(n) if n > 1 else abs(expected_mean) * 0.1
    lower = expected_mean - tolerance_sigma * stderr
    upper = expected_mean + tolerance_sigma * stderr
    assert lower <= sample_mean <= upper, (
        f"{name}: mean={sample_mean:.6f} outside "
        f"[{lower:.6f}, {upper:.6f}] "
        f"(expected={expected_mean:.6f}, n={n}, std={sample_std:.6f})"
    )


def assert_rate_in_range(
    count, n_trials, expected_rate, tolerance_sigma=4.0, name=""
):
    """Assert observed rate is within tolerance_sigma of expected binomial rate."""
    observed_rate = count / n_trials if n_trials > 0 else 0.0
    stderr = np.sqrt(expected_rate * (1 - expected_rate) / n_trials)
    lower = expected_rate - tolerance_sigma * stderr
    upper = expected_rate + tolerance_sigma * stderr
    assert lower <= observed_rate <= upper, (
        f"{name}: rate={observed_rate:.6f} ({count}/{n_trials}) outside "
        f"[{lower:.6f}, {upper:.6f}] "
        f"(expected={expected_rate:.6f})"
    )


def assert_no_nan_inf(arr, name=""):
    """Assert array has no NaN or Inf values."""
    arr = np.asarray(arr)
    n_nan = np.sum(np.isnan(arr))
    n_inf = np.sum(np.isinf(arr))
    assert n_nan == 0, f"{name}: found {n_nan} NaN values"
    assert n_inf == 0, f"{name}: found {n_inf} Inf values"


def assert_in_unit_interval(arr, name="", strict=False):
    """Assert all values in [0, 1] (or (0, 1) if strict)."""
    arr = np.asarray(arr)
    if strict:
        assert np.all(arr > 0) and np.all(arr < 1), (
            f"{name}: values outside (0,1): "
            f"min={arr.min():.8f}, max={arr.max():.8f}"
        )
    else:
        assert np.all(arr >= 0) and np.all(arr <= 1), (
            f"{name}: values outside [0,1]: "
            f"min={arr.min():.8f}, max={arr.max():.8f}"
        )


def assert_monotone_nondecreasing(seq, name=""):
    """Assert sequence is monotonically non-decreasing."""
    for i in range(1, len(seq)):
        assert seq[i] >= seq[i - 1] - 1e-10, (
            f"{name}: monotonicity violation at index {i}: "
            f"{seq[i-1]:.6f} -> {seq[i]:.6f}"
        )


def assert_arrays_equal_across_seeds(fn, seeds, name=""):
    """Assert function produces identical results with same seed."""
    reference = None
    for seed in seeds:
        result = fn(seed)
        if reference is None:
            reference = result
        else:
            np.testing.assert_array_equal(
                reference, result,
                err_msg=f"{name}: seed {seeds[0]} != seed {seed}"
            )


# ---- Scenario fixtures ----------------------------------------------------

@pytest.fixture
def rng():
    """Fresh RNG for test isolation."""
    return np.random.default_rng(42)


@pytest.fixture
def stroke_small():
    """Small stroke scenario for fast tests (1000 patients, 12 weeks)."""
    config = StrokeConfig(
        n_patients=1000,
        n_weeks=12,
        prediction_interval=4,
        annual_incident_rate=0.05,
        intervention_effectiveness=0.50,
    )
    sc = StrokePreventionScenario(config=config, seed=42)
    return sc, config


@pytest.fixture
def stroke_results_branched(stroke_small):
    """Pre-run stroke results in BRANCHED mode."""
    sc, config = stroke_small
    engine = BranchedSimulationEngine(sc, CounterfactualMode.BRANCHED)
    return engine.run(config.n_patients)


@pytest.fixture
def noshow_small():
    """Small no-show scenario for fast tests (500 patients, 15 days)."""
    tc = TimeConfig(
        n_timesteps=15,
        timestep_duration=1 / 365,
        timestep_unit="day",
        prediction_schedule=list(range(15)),
    )
    cc = ClinicConfig(
        n_providers=3,
        slots_per_provider_per_day=8,
        max_overbook_per_provider=2,
    )
    sc = NoShowOverbookingScenario(
        time_config=tc, seed=42,
        n_patients=500,
        base_noshow_rate=0.13,
        overbooking_threshold=0.30,
        model_type="predictor",
        model_auc=0.80,
        clinic_config=cc,
    )
    return sc, tc


@pytest.fixture
def noshow_results_branched(noshow_small):
    """Pre-run no-show results in BRANCHED mode."""
    sc, tc = noshow_small
    engine = BranchedSimulationEngine(sc, CounterfactualMode.BRANCHED)
    return engine.run(500)


# ---- Model fixtures -------------------------------------------------------

@pytest.fixture
def fitted_model():
    """A fitted ControlledMLModel in classification mode."""
    rng = np.random.default_rng(42)
    n = 5000
    risks = rng.beta(0.5, 0.5 * (1 / 0.13 - 1), n)
    risks = np.clip(risks, 0.01, 0.99)
    labels = (rng.random(n) < risks).astype(int)

    model = ControlledMLModel(
        mode="classification",
        target_sensitivity=0.80,
        target_ppv=0.25,
    )
    model.fit(labels, risks, rng, n_iterations=3)
    return model, risks, labels, rng
