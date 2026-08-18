"""Hand-computable fixtures for the ranking metrics in ml/performance.py.

Every expected value below is derivable by hand; if a test fails, recompute
on paper before touching the implementation.
"""

import numpy as np
import pytest

from healthcare_sim_sdk.ml.performance import (
    auprc,
    capture_at_k,
    precision_at_k,
)


class TestAUPRC:
    def test_hand_computed_four_elements(self):
        # Ranked by score: [1, 0, 1, 0].
        # Precision at hit ranks: 1/1 (rank 1), 2/3 (rank 3).
        # AP = (1 + 2/3) / 2 = 5/6.
        y = np.array([1, 0, 1, 0])
        s = np.array([0.9, 0.8, 0.7, 0.6])
        assert auprc(y, s) == pytest.approx(5.0 / 6.0)

    def test_perfect_ranking(self):
        y = np.array([1, 1, 0, 0])
        s = np.array([0.9, 0.8, 0.2, 0.1])
        assert auprc(y, s) == pytest.approx(1.0)

    def test_worst_ranking(self):
        # Positives ranked last: hits at ranks 3, 4.
        # AP = (1/3 + 2/4) / 2 = 5/12.
        y = np.array([0, 0, 1, 1])
        s = np.array([0.9, 0.8, 0.2, 0.1])
        assert auprc(y, s) == pytest.approx(5.0 / 12.0)

    def test_all_positive(self):
        y = np.ones(5)
        s = np.linspace(0.9, 0.1, 5)
        assert auprc(y, s) == pytest.approx(1.0)

    def test_all_negative_returns_zero(self):
        y = np.zeros(5)
        s = np.linspace(0.9, 0.1, 5)
        assert auprc(y, s) == 0.0

    def test_ties_are_deterministic(self):
        y = np.array([1, 0, 1, 0])
        s = np.array([0.5, 0.5, 0.5, 0.5])
        # Stable sort keeps input order: [1,0,1,0].
        # AP = (1/1 + 2/3) / 2 = 5/6, and repeatable.
        assert auprc(y, s) == pytest.approx(5.0 / 6.0)
        assert auprc(y, s) == auprc(y, s)

    def test_nan_raises(self):
        y = np.array([1, 0])
        s = np.array([0.9, np.nan])
        with pytest.raises(ValueError, match="NaN"):
            auprc(y, s)

    def test_random_scores_approach_prevalence(self):
        # AP of a random ranking converges to prevalence.
        rng = np.random.default_rng(42)
        y = (rng.random(20000) < 0.13).astype(int)
        s = rng.random(20000)
        assert auprc(y, s) == pytest.approx(0.13, abs=0.02)


class TestPrecisionAtK:
    def test_hand_computed_top_20pct(self):
        # n=10, k=2. Top-2 by score are indices 0, 1 -> labels 1, 0.
        y = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
        s = np.linspace(0.95, 0.05, 10)
        assert precision_at_k(y, s, 0.20) == pytest.approx(0.5)

    def test_k_full_population_equals_prevalence(self):
        y = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
        s = np.linspace(0.95, 0.05, 10)
        assert precision_at_k(y, s, 1.0) == pytest.approx(0.3)

    def test_minimum_k_is_one(self):
        y = np.array([1, 0, 0])
        s = np.array([0.9, 0.5, 0.1])
        # k_fraction tiny -> k floors at 1 -> top item is a positive.
        assert precision_at_k(y, s, 0.01) == pytest.approx(1.0)

    def test_invalid_k_raises(self):
        y = np.array([1, 0])
        s = np.array([0.9, 0.1])
        with pytest.raises(ValueError, match="k_fraction"):
            precision_at_k(y, s, 0.0)
        with pytest.raises(ValueError, match="k_fraction"):
            precision_at_k(y, s, 1.5)


class TestCaptureAtK:
    def test_hand_computed_top_20pct(self):
        # n=10, 3 positives; top-2 contains 1 of them -> capture 1/3.
        y = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
        s = np.linspace(0.95, 0.05, 10)
        assert capture_at_k(y, s, 0.20) == pytest.approx(1.0 / 3.0)

    def test_k_full_population_captures_all(self):
        y = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
        s = np.linspace(0.95, 0.05, 10)
        assert capture_at_k(y, s, 1.0) == pytest.approx(1.0)

    def test_all_negative_returns_zero(self):
        y = np.zeros(10)
        s = np.linspace(0.95, 0.05, 10)
        assert capture_at_k(y, s, 0.5) == 0.0

    def test_perfect_ranking_captures_early(self):
        # 2 positives ranked top; k=20% of 10 = 2 -> capture = 1.0.
        y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        s = np.linspace(0.95, 0.05, 10)
        assert capture_at_k(y, s, 0.20) == pytest.approx(1.0)

    def test_nan_raises(self):
        y = np.array([1, 0])
        s = np.array([np.nan, 0.1])
        with pytest.raises(ValueError, match="NaN"):
            capture_at_k(y, s, 0.5)
