"""Controlled binary classifier for simulation.

Generates predictions that achieve target PPV and sensitivity by
injecting calibrated noise into true risk scores.
"""

from typing import Dict, Tuple

import numpy as np

from .performance import confusion_matrix_metrics


class ControlledBinaryClassifier:
    """Binary classifier that hits target PPV and sensitivity.

    Given known true labels and risk scores, generates predicted
    scores and binary labels that achieve the specified performance
    targets. Uses noise injection with grid-search optimization
    over correlation and scale parameters.

    Usage in a scenario's predict() method:
        classifier = ControlledBinaryClassifier(
            target_ppv=0.15, target_sensitivity=0.80
        )
        classifier.optimize(true_labels, risk_scores, rng)
        scores, labels = classifier.predict(true_labels, risk_scores, rng)
    """

    def __init__(
        self,
        target_sensitivity: float = 0.8,
        target_ppv: float = 0.3,
    ):
        self.target_sensitivity = target_sensitivity
        self.target_ppv = target_ppv
        self.noise_correlation: float = 0.7
        self.noise_scale: float = 0.3
        self.threshold: float = 0.5
        self._optimized = False

    def predict(
        self,
        true_labels: np.ndarray,
        risk_scores: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate predictions with controlled performance.

        Args:
            true_labels: Ground truth binary labels (0/1).
            risk_scores: True underlying risk scores in [0, 1].
            rng: Random generator (should be scenario's prediction stream).

        Returns:
            (scores, labels) tuple of predicted scores and binary labels.
        """
        scores = self._generate_scores(
            true_labels, risk_scores, rng,
            self.noise_correlation, self.noise_scale,
        )
        labels = (scores >= self.threshold).astype(int)
        return scores, labels

    def optimize(
        self,
        true_labels: np.ndarray,
        risk_scores: np.ndarray,
        rng: np.random.Generator,
        n_iterations: int = 10,
    ) -> Dict[str, float]:
        """Find noise parameters that hit target performance.

        Grid searches over correlation and scale values, averaging
        metrics across iterations for stability.
        """
        correlations = np.linspace(0.5, 0.95, 10)
        scales = np.linspace(0.1, 0.5, 10)

        best_score = float("inf")
        best_params = (0.7, 0.3, 0.5)

        for corr in correlations:
            for scale in scales:
                total_metric = 0.0
                for _ in range(n_iterations):
                    scores = self._generate_scores(
                        true_labels, risk_scores, rng, corr, scale,
                    )
                    # Find best threshold for these parameters
                    best_t, best_t_score = 0.5, float("inf")
                    for t in np.linspace(0.1, 0.9, 17):
                        m = confusion_matrix_metrics(
                            true_labels, scores, threshold=t,
                        )
                        dist = (
                            abs(m["sensitivity"] - self.target_sensitivity)
                            + 1.5 * abs(m["ppv"] - self.target_ppv)
                        )
                        if dist < best_t_score:
                            best_t_score = dist
                            best_t = t
                    total_metric += best_t_score

                avg_metric = total_metric / n_iterations
                if avg_metric < best_score:
                    best_score = avg_metric
                    best_params = (corr, scale, best_t)

        self.noise_correlation = best_params[0]
        self.noise_scale = best_params[1]
        self.threshold = best_params[2]
        self._optimized = True

        # Compute final metrics at optimized params
        scores = self._generate_scores(
            true_labels, risk_scores, rng,
            self.noise_correlation, self.noise_scale,
        )
        metrics = confusion_matrix_metrics(
            true_labels, scores, self.threshold,
        )
        return {
            "noise_correlation": self.noise_correlation,
            "noise_scale": self.noise_scale,
            "threshold": self.threshold,
            "achieved_sensitivity": metrics["sensitivity"],
            "achieved_ppv": metrics["ppv"],
        }

    def _generate_scores(
        self,
        true_labels: np.ndarray,
        risk_scores: np.ndarray,
        rng: np.random.Generator,
        correlation: float,
        scale: float,
    ) -> np.ndarray:
        """Generate noisy prediction scores."""
        n = len(risk_scores)
        base_scores = np.clip(risk_scores, 0, 1)
        noise = rng.random(n)

        # Correlated noise mixing
        noisy = correlation * base_scores + (1 - correlation) * noise

        # Label-dependent noise (positive cases get slight boost)
        label_noise = np.where(true_labels == 1, 0.05, -0.025)
        label_noise += rng.normal(0, 0.05, n)

        # Independent noise for variability
        independent_noise = rng.normal(0, 0.1, n)

        noisy += (label_noise + independent_noise) * scale

        # Sigmoid calibration
        scores = 1.0 / (1.0 + np.exp(-4.0 * (noisy - 0.5)))
        return np.clip(scores, 0, 1)
