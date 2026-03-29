"""Controlled probability model for simulation.

Generates probability predictions that achieve target AUC and
calibration characteristics. Unlike ControlledBinaryClassifier
which targets PPV/sensitivity at a threshold, this targets overall
discrimination (AUC) and calibration (slope near 1.0).
"""

from typing import Dict

import numpy as np

from .performance import auc_score, calibration_slope


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
    ):
        self.target_auc = target_auc
        self.target_calibration_slope = target_calibration_slope
        self.noise_scale: float = 0.3
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
        return self._generate_predictions(
            true_probabilities, rng, self.noise_scale,
        )

    def fit(
        self,
        true_probabilities: np.ndarray,
        rng: np.random.Generator,
        n_iterations: int = 10,
    ) -> Dict[str, float]:
        """Find noise scale that achieves target AUC.

        Searches over noise levels to match target AUC. Higher noise
        reduces discrimination (lower AUC).
        """
        # Generate binary outcomes from true probs for AUC calculation
        actuals = (rng.random(len(true_probabilities))
                   < true_probabilities).astype(int)

        scales = np.linspace(0.05, 1.0, 20)
        best_score = float("inf")
        best_scale = 0.3

        for scale in scales:
            total_auc_diff = 0.0
            total_cal_diff = 0.0
            for _ in range(n_iterations):
                preds = self._generate_predictions(
                    true_probabilities, rng, scale,
                )
                auc = auc_score(actuals, preds)
                cal, _, _ = calibration_slope(actuals, preds)
                total_auc_diff += abs(auc - self.target_auc)
                total_cal_diff += abs(cal - self.target_calibration_slope)

            avg_auc_diff = total_auc_diff / n_iterations
            avg_cal_diff = total_cal_diff / n_iterations
            combined = avg_auc_diff + 0.5 * avg_cal_diff

            if combined < best_score:
                best_score = combined
                best_scale = scale

        self.noise_scale = best_scale
        self._fitted = True

        # Compute final metrics
        preds = self._generate_predictions(
            true_probabilities, rng, self.noise_scale,
        )
        achieved_auc = auc_score(actuals, preds)
        achieved_cal, _, _ = calibration_slope(actuals, preds)

        return {
            "noise_scale": self.noise_scale,
            "achieved_auc": achieved_auc,
            "achieved_calibration_slope": achieved_cal,
        }

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

    def _generate_predictions(
        self,
        true_probabilities: np.ndarray,
        rng: np.random.Generator,
        noise_scale: float,
    ) -> np.ndarray:
        """Generate noisy probability predictions."""
        n = len(true_probabilities)

        # Log-odds space noise for better calibration properties
        eps = 1e-6
        clipped = np.clip(true_probabilities, eps, 1 - eps)
        log_odds = np.log(clipped / (1 - clipped))

        # Add Gaussian noise in log-odds space
        noise = rng.normal(0, noise_scale, n)
        noisy_log_odds = log_odds + noise

        # Back to probability space
        predictions = 1.0 / (1.0 + np.exp(-noisy_log_odds))
        return np.clip(predictions, 0, 1)
