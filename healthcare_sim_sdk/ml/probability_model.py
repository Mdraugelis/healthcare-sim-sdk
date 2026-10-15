"""Controlled probability model for simulation.

Generates probability predictions that achieve target AUC and
calibration characteristics. Unlike ControlledBinaryClassifier
which targets PPV/sensitivity at a threshold, this targets overall
discrimination (AUC) and calibration (slope near 1.0).
"""

import warnings
from typing import Dict, Optional

import numpy as np

from .performance import auc_exact, auc_score, calibration_slope


class ControlledProbabilityModel:
    """Probability estimator that hits target AUC and calibration.

    Given true probabilities, generates predicted probabilities that
    achieve specified AUC and calibration slope. Used for scenarios
    where the ML task is probability estimation (e.g., no-show
    prediction) rather than binary classification.

    Usage in a scenario's predict() method:
        model = ControlledProbabilityModel(target_auc=0.78)
        model.fit(true_probabilities, rng)
        predictions = model.predict(true_probabilities, rng)
    """

    def __init__(
        self,
        target_auc: float = 0.78,
        target_calibration_slope: float = 1.0,
        allow_infeasible: bool = False,
        tolerance_abs: float = 0.02,
    ):
        self.target_auc = target_auc
        self.target_calibration_slope = target_calibration_slope
        self.allow_infeasible = allow_infeasible
        self.tolerance_abs = tolerance_abs
        self.noise_scale: float = 0.3
        self._platt_a: float = 1.0
        self._platt_b: float = 0.0
        self._population_auc_ceiling: Optional[float] = None
        self._fitted = False

    def predict(
        self,
        true_probabilities: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate probability predictions with controlled AUC.

        Args:
            true_probabilities: True event probabilities in [0, 1].
            rng: Random generator (should be scenario's prediction stream).

        Returns:
            Predicted probabilities with target discrimination/calibration.
        """
        z = rng.normal(0, 1, len(true_probabilities))
        return self._calibrate(
            self._noisy_logits(true_probabilities, z, self.noise_scale)
        )

    def fit(
        self,
        true_probabilities: np.ndarray,
        rng: np.random.Generator,
        n_iterations: int = 10,
    ) -> Dict[str, float]:
        """Find the log-odds noise scale that achieves target AUC.

        AUC falls monotonically with noise, so the scale is found by
        bisection on AUC alone, using common random numbers across
        candidates. Calibration is then restored with Platt scaling
        rather than traded against AUC in the search objective -- that
        trade is what used to pin the noise near zero and return the
        population ceiling whatever AUC was asked for.

        Raises OperatingPointError when target_auc exceeds the
        population's Bayes ceiling (the AUC of the true probabilities),
        unless allow_infeasible is set.
        """
        from .model import OperatingPointError

        p = np.asarray(true_probabilities, dtype=float)
        n = len(p)
        actuals = (rng.random(n) < p).astype(int)
        ceiling = float(auc_exact(actuals, p))
        self._population_auc_ceiling = ceiling
        if self.target_auc > ceiling + self.tolerance_abs:
            msg = (
                f"target_auc {self.target_auc:.4f} exceeds this "
                f"population's Bayes ceiling {ceiling:.4f} (the AUC of "
                f"the true probabilities). No model can reach it."
            )
            if not self.allow_infeasible:
                raise OperatingPointError(
                    msg, report={"population_auc_ceiling": ceiling},
                )
            warnings.warn(msg, UserWarning, stacklevel=2)

        draws = [rng.normal(0, 1, n) for _ in range(max(n_iterations, 5))]

        def mean_auc(scale: float) -> float:
            return float(np.mean([
                auc_exact(actuals, self._noisy_logits(p, z, scale))
                for z in draws
            ]))

        lo, hi = 0.0, 12.0
        if mean_auc(lo) <= self.target_auc:
            scale = lo
        elif mean_auc(hi) >= self.target_auc:
            scale = hi
        else:
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                if mean_auc(mid) > self.target_auc:
                    lo = mid
                else:
                    hi = mid
            scale = 0.5 * (lo + hi)
        self.noise_scale = float(scale)

        # Platt scaling on a fit draw restores calibration.
        fit_logits = self._noisy_logits(p, rng.normal(0, 1, n), scale)
        self._fit_platt(fit_logits, actuals)
        self._fitted = True

        # Report on an independent draw -- what predict() delivers.
        preds = self.predict(p, rng)
        achieved_auc = auc_exact(actuals, preds)
        achieved_cal, _, _ = calibration_slope(actuals, preds)
        return {
            "noise_scale": self.noise_scale,
            "achieved_auc": achieved_auc,
            "achieved_calibration_slope": achieved_cal,
            "population_auc_ceiling": ceiling,
            "report_basis": "independent_noise_draw",
        }

    @staticmethod
    def _noisy_logits(
        p: np.ndarray, z: np.ndarray, scale: float,
    ) -> np.ndarray:
        eps = 1e-6
        clipped = np.clip(p, eps, 1 - eps)
        return np.log(clipped / (1 - clipped)) + scale * z

    def _calibrate(self, logits: np.ndarray) -> np.ndarray:
        out = 1.0 / (1.0 + np.exp(-(self._platt_a * logits + self._platt_b)))
        return np.clip(out, 0, 1)

    def _fit_platt(self, logits: np.ndarray, actuals: np.ndarray) -> None:
        from scipy.optimize import minimize

        eps = 1e-7

        def nll(params):
            q = 1.0 / (1.0 + np.exp(-(params[0] * logits + params[1])))
            q = np.clip(q, eps, 1 - eps)
            return -np.mean(
                actuals * np.log(q) + (1 - actuals) * np.log(1 - q)
            )

        res = minimize(nll, [1.0, 0.0], method="Nelder-Mead")
        self._platt_a, self._platt_b = float(res.x[0]), float(res.x[1])

    def calibration_report(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
    ) -> Dict[str, float]:
        """Generate calibration metrics for predictions."""
        from .performance import hosmer_lemeshow_test

        auc = auc_score(actuals, predictions)
        slope, pred_means, obs_means = calibration_slope(
            actuals, predictions,
        )
        hl_stat, hl_p = hosmer_lemeshow_test(actuals, predictions)

        return {
            "auc": auc,
            "calibration_slope": slope,
            "hosmer_lemeshow_stat": hl_stat,
            "hosmer_lemeshow_p": hl_p,
        }
