"""Regression tests for the label-free model simulator.

These pin the defects fixed on fix/label-free-model-scores:

1. fit() tuned a label-dependent score boost that predict() dropped when
   called without outcomes, so the deployed AUC fell short of the fit
   report (0.90 configured -> ~0.79 delivered).
2. predict() called WITH outcomes kept the boost, letting a simulated
   model beat the population's Bayes ceiling (an oracle knowing every
   true risk).
3. Only operating_point mode was gated against the ceiling.
4. Platt scaling was fit on boosted scores, so deployed calibration
   slopes sat near 0.75 rather than 1.0.
5. ControlledProbabilityModel returned the ceiling whatever AUC was
   asked for.

Every check here scores a FRESH population through predict() -- never
the fit report alone -- because the fit report is what was wrong.
"""

import warnings

import numpy as np
import pytest
from scipy.optimize import minimize

from healthcare_sim_sdk.ml.model import ControlledMLModel, OperatingPointError
from healthcare_sim_sdk.ml.performance import auc_exact, auc_score
from healthcare_sim_sdk.ml.probability_model import (
    ControlledProbabilityModel,
)

N = 6000


def _population(seed, n=N, prevalence=0.05, concentration=4.0):
    rng = np.random.default_rng(seed)
    a = prevalence * concentration
    b = (1 - prevalence) * concentration
    risk = rng.beta(a, b, n)
    labels = (rng.random(n) < risk).astype(int)
    return risk, labels


def _logistic_calibration_slope(y, p):
    """Standard slope b in logit P(y=1) = a + b * logit(p)."""
    eps = 1e-7
    p = np.clip(p, eps, 1 - eps)
    lp = np.log(p / (1 - p))

    def nll(t):
        q = np.clip(1 / (1 + np.exp(-(t[0] + t[1] * lp))), eps, 1 - eps)
        return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))

    return float(minimize(nll, [0.0, 1.0], method="Nelder-Mead").x[1])


def _fit(target, **kw):
    risk, labels = _population(1, **kw)
    model = ControlledMLModel(mode="discrimination", target_auc=target)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(labels, risk, np.random.default_rng(2), n_iterations=3)
    return model


class TestAucExact:
    def test_matches_hand_count(self):
        # 3 of 4 positive-negative pairs concordant -> 0.75
        y = np.array([0, 1, 0, 1])
        s = np.array([0.1, 0.4, 0.5, 0.9])
        assert auc_exact(y, s) == pytest.approx(0.75)

    def test_ties_credit_half(self):
        assert auc_exact(np.array([0, 1]), np.array([0.3, 0.3])) == 0.5

    def test_single_class_is_nan(self):
        assert np.isnan(auc_exact(np.zeros(10), np.random.rand(10)))

    def test_resolves_narrow_score_band(self):
        """The 200-threshold grid AUC collapses when every score sits
        inside one grid step; the rank AUC does not."""
        y = np.array([0, 0, 1, 1])
        s = np.array([0.0010, 0.0011, 0.0012, 0.0013])
        assert auc_exact(y, s) == 1.0
        assert auc_score(y, s) < 1.0


class TestDeliveredAucMatchesTarget:
    @pytest.mark.parametrize("target", [0.65, 0.76, 0.85, 0.89])
    def test_heldout_population_hits_target(self, target):
        """Fit on one population; score a different one through
        predict() with no labels -- the way scenarios deploy."""
        model = _fit(target)
        risk, labels = _population(99)
        scores = model.predict(risk, np.random.default_rng(7))
        assert abs(auc_exact(labels, scores) - target) < 0.03

    def test_fit_report_describes_predict(self):
        """The report is measured on an independent noise draw, so it
        must agree with what predict() produces."""
        model = _fit(0.80)
        report = model._fit_report
        assert report["report_basis"] == "independent_noise_draw"
        assert report["label_free"] is True
        risk, labels = _population(1)
        scores = model.predict(risk, np.random.default_rng(11))
        assert abs(auc_exact(labels, scores)
                   - report["achieved_auc"]) < 0.03


class TestLabelsAreIgnored:
    def test_predict_identical_with_or_without_labels(self):
        model = _fit(0.80)
        risk, labels = _population(5)
        a = model.predict(risk, np.random.default_rng(3))
        b = model.predict(risk, np.random.default_rng(3), labels)
        np.testing.assert_array_equal(a, b)

    def test_predict_binary_identical_with_or_without_labels(self):
        model = _fit(0.80)
        risk, labels = _population(5)
        s1, l1 = model.predict_binary(risk, np.random.default_rng(3))
        s2, l2 = model.predict_binary(
            risk, np.random.default_rng(3), labels,
        )
        np.testing.assert_array_equal(s1, s2)
        np.testing.assert_array_equal(l1, l2)

    def test_never_beats_the_oracle(self):
        """Scores built from risk alone cannot out-rank the true risks,
        even when asked to and handed the outcomes. The old generator
        reached ~0.87 here against a 0.78 ceiling."""
        risk, labels = _population(3, prevalence=0.30)
        ceiling = auc_exact(labels, risk)
        model = ControlledMLModel(
            mode="discrimination", target_auc=ceiling + 0.08,
            allow_infeasible=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(labels, risk, np.random.default_rng(4),
                      n_iterations=3)
        scores = model.predict(risk, np.random.default_rng(5), labels)
        assert auc_exact(labels, scores) <= ceiling + 0.005


class TestCeilingGateEveryMode:
    def test_discrimination_above_ceiling_raises(self):
        risk, labels = _population(1)
        model = ControlledMLModel(mode="discrimination", target_auc=0.97)
        with pytest.raises(OperatingPointError, match="Bayes ceiling"):
            model.fit(labels, risk, np.random.default_rng(1))

    def test_threshold_ppv_above_ceiling_raises(self):
        risk, labels = _population(1)
        model = ControlledMLModel(
            mode="threshold_ppv", target_auc=0.97, target_ppv=0.2,
            operating_threshold=0.1,
        )
        with pytest.raises(OperatingPointError, match="Bayes ceiling"):
            model.fit(labels, risk, np.random.default_rng(1))

    def test_classification_infeasible_pair_raises(self):
        risk, labels = _population(1)
        model = ControlledMLModel(
            mode="classification", target_sensitivity=0.95,
            target_ppv=0.60,
        )
        with pytest.raises(OperatingPointError):
            model.fit(labels, risk, np.random.default_rng(1))

    def test_allow_infeasible_warns_and_records_gap(self):
        risk, labels = _population(1)
        model = ControlledMLModel(
            mode="discrimination", target_auc=0.97, allow_infeasible=True,
        )
        with pytest.warns(UserWarning, match="delivered AUC"):
            report = model.fit(labels, risk, np.random.default_rng(1))
        assert report["auc_within_tolerance"] is False
        assert report["achieved_auc"] <= (
            report["population_auc_ceiling"] + 0.01
        )

    def test_feasible_target_does_not_warn(self):
        risk, labels = _population(1)
        model = ControlledMLModel(mode="discrimination", target_auc=0.80)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            report = model.fit(
                labels, risk, np.random.default_rng(1), n_iterations=3,
            )
        assert report["auc_within_tolerance"] is True


class TestCalibrationOnHeldOut:
    @pytest.mark.parametrize("target", [0.70, 0.85])
    def test_slope_and_mean_on_fresh_population(self, target):
        model = _fit(target)
        risk, labels = _population(42)
        scores = model.predict(risk, np.random.default_rng(8))
        slope = _logistic_calibration_slope(labels, scores)
        assert 0.80 < slope < 1.20
        assert abs(scores.mean() - labels.mean()) < 0.01


class TestClassificationThresholdScale:
    def test_threshold_lives_on_calibrated_scale(self):
        """The threshold must be placed on the scores predict_binary
        returns (post-Platt), so the flagged set carries the PPV."""
        risk, labels = _population(1, prevalence=0.13, concentration=1.0)
        model = ControlledMLModel(
            mode="classification", target_sensitivity=0.60,
            target_ppv=0.30,
        )
        model.fit(labels, risk, np.random.default_rng(2), n_iterations=3)
        risk2, labels2 = _population(77, prevalence=0.13,
                                     concentration=1.0)
        _, flags = model.predict_binary(risk2, np.random.default_rng(3))
        ppv = labels2[flags == 1].mean()
        assert abs(ppv - 0.30) < 0.06


class TestUnderpoweredGate:
    def test_few_positives_warn_instead_of_raise(self):
        """A ceiling estimated from a handful of events is noise; the
        gate must not stop a run on it, but must say so."""
        rng = np.random.default_rng(0)
        risk = rng.beta(0.2, 20, 2000)
        labels = np.zeros(2000, dtype=int)
        labels[np.argsort(risk)[:3]] = 1   # 3 positives, worst-ranked
        model = ControlledMLModel(mode="discrimination", target_auc=0.80)
        with pytest.warns(UserWarning, match="only 3 positive"):
            report = model.fit(labels, risk, rng, n_iterations=2)
        assert report["gate_underpowered"] is True


class TestDegeneratePopulation:
    def test_no_positives_warns_and_returns(self):
        risk = np.full(500, 0.01)
        labels = np.zeros(500, dtype=int)
        model = ControlledMLModel(mode="classification")
        with pytest.warns(UserWarning, match="AUC is undefined"):
            report = model.fit(labels, risk, np.random.default_rng(1))
        assert report["degenerate_population"] is True
        scores = model.predict(risk, np.random.default_rng(2))
        assert np.all((scores >= 0) & (scores <= 1))


class TestProbabilityModelHonorsTarget:
    @pytest.mark.parametrize("target", [0.62, 0.70, 0.76])
    def test_heldout_auc_tracks_target(self, target):
        rng = np.random.default_rng(11)
        p = rng.beta(0.8, 3.2, N)   # ceiling ~0.80
        model = ControlledProbabilityModel(target_auc=target)
        model.fit(p, np.random.default_rng(12))
        actual = (np.random.default_rng(14).random(N) < p).astype(int)
        preds = model.predict(p, np.random.default_rng(13))
        assert abs(auc_exact(actual, preds) - target) < 0.03

    def test_different_targets_give_different_models(self):
        """The old search returned the ceiling for every target."""
        rng = np.random.default_rng(11)
        p = rng.beta(0.8, 3.2, N)
        scales = set()
        for target in (0.62, 0.70, 0.76):
            m = ControlledProbabilityModel(target_auc=target)
            m.fit(p, np.random.default_rng(12))
            scales.add(round(m.noise_scale, 3))
        assert len(scales) == 3

    def test_above_ceiling_raises(self):
        rng = np.random.default_rng(11)
        p = rng.beta(0.8, 3.2, N)
        with pytest.raises(OperatingPointError, match="Bayes ceiling"):
            ControlledProbabilityModel(target_auc=0.90).fit(
                p, np.random.default_rng(1),
            )
