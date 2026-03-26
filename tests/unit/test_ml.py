"""Unit tests for ML model simulators and performance utilities."""

import numpy as np

from sdk.ml.binary_classifier import ControlledBinaryClassifier
from sdk.ml.performance import (
    auc_score,
    calibration_slope,
    confusion_matrix_metrics,
    roc_curve,
)
from sdk.ml.probability_model import ControlledProbabilityModel


def _make_population(rng, n=2000, prevalence=0.1):
    """Create a test population with known labels and risks."""
    risks = rng.beta(0.5, 0.5 * (1 / prevalence - 1), n)
    risks = np.clip(risks * prevalence / risks.mean(), 0, 0.99)
    labels = (rng.random(n) < risks).astype(int)
    return risks, labels


class TestConfusionMatrixMetrics:
    def test_perfect_predictions(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0.9, 0.8, 0.1, 0.2])
        m = confusion_matrix_metrics(y_true, y_pred, threshold=0.5)
        assert m["sensitivity"] == 1.0
        assert m["specificity"] == 1.0
        assert m["ppv"] == 1.0

    def test_all_positive(self):
        y_true = np.array([1, 0, 0, 0])
        y_pred = np.array([0.9, 0.9, 0.9, 0.9])
        m = confusion_matrix_metrics(y_true, y_pred, threshold=0.5)
        assert m["sensitivity"] == 1.0
        assert m["ppv"] == 0.25  # 1 TP / 4 flagged

    def test_metrics_keys(self):
        m = confusion_matrix_metrics(
            np.array([1, 0]), np.array([0.9, 0.1])
        )
        expected_keys = {
            "tp", "fp", "tn", "fn", "sensitivity",
            "specificity", "ppv", "npv", "accuracy", "f1", "flag_rate",
        }
        assert set(m.keys()) == expected_keys


class TestROCAndAUC:
    def test_perfect_classifier_high_auc(self):
        rng = np.random.default_rng(42)
        y_true = np.concatenate([np.ones(100), np.zeros(100)])
        y_scores = np.concatenate([
            rng.uniform(0.7, 1.0, 100),
            rng.uniform(0.0, 0.3, 100),
        ])
        auc = auc_score(y_true, y_scores)
        assert auc > 0.9

    def test_random_classifier_near_05(self):
        rng = np.random.default_rng(42)
        y_true = (rng.random(1000) > 0.5).astype(int)
        y_scores = rng.random(1000)
        auc = auc_score(y_true, y_scores)
        assert 0.4 < auc < 0.6

    def test_roc_curve_shape(self):
        rng = np.random.default_rng(42)
        fprs, tprs, thresholds = roc_curve(
            (rng.random(100) > 0.5).astype(int),
            rng.random(100),
        )
        assert len(fprs) == len(tprs) == len(thresholds)


class TestBinaryClassifier:
    def test_predict_returns_scores_and_labels(self):
        rng = np.random.default_rng(42)
        risks, labels = _make_population(rng)
        clf = ControlledBinaryClassifier()
        scores, preds = clf.predict(labels, risks, rng)
        assert scores.shape == risks.shape
        assert preds.shape == risks.shape
        assert set(np.unique(preds)).issubset({0, 1})

    def test_optimize_improves_performance(self):
        rng = np.random.default_rng(42)
        risks, labels = _make_population(rng, n=5000, prevalence=0.1)
        clf = ControlledBinaryClassifier(
            target_sensitivity=0.7, target_ppv=0.15,
        )
        result = clf.optimize(labels, risks, rng, n_iterations=5)
        assert result["achieved_sensitivity"] > 0.3
        assert clf._optimized

    def test_scores_in_unit_interval(self):
        rng = np.random.default_rng(42)
        risks, labels = _make_population(rng)
        clf = ControlledBinaryClassifier()
        scores, _ = clf.predict(labels, risks, rng)
        assert scores.min() >= 0
        assert scores.max() <= 1


class TestProbabilityModel:
    def test_predict_returns_probabilities(self):
        rng = np.random.default_rng(42)
        true_probs = rng.beta(2, 20, 1000)
        model = ControlledProbabilityModel(target_auc=0.78)
        preds = model.predict(true_probs, rng)
        assert preds.shape == true_probs.shape
        assert preds.min() >= 0
        assert preds.max() <= 1

    def test_fit_adjusts_noise(self):
        rng = np.random.default_rng(42)
        true_probs = rng.beta(2, 20, 2000)
        model = ControlledProbabilityModel(target_auc=0.75)
        result = model.fit(true_probs, rng, n_iterations=3)
        assert "achieved_auc" in result
        assert "noise_scale" in result
        assert model._fitted

    def test_calibration_report(self):
        rng = np.random.default_rng(42)
        true_probs = rng.beta(2, 20, 1000)
        actuals = (rng.random(1000) < true_probs).astype(int)
        model = ControlledProbabilityModel()
        preds = model.predict(true_probs, rng)
        report = model.calibration_report(preds, actuals)
        assert "auc" in report
        assert "calibration_slope" in report
        assert "hosmer_lemeshow_p" in report


class TestCalibrationSlope:
    def test_well_calibrated(self):
        rng = np.random.default_rng(42)
        true_probs = rng.uniform(0, 1, 1000)
        actuals = (rng.random(1000) < true_probs).astype(int)
        slope, _, _ = calibration_slope(actuals, true_probs)
        assert abs(slope - 1.0) < 0.3
