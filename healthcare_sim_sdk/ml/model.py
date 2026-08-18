"""Unified controlled ML model for simulation.

Generates predictions with controlled performance characteristics using
4-component noise injection and 2D grid-search optimization. Supports
three operating modes targeting different metric combinations.

Based on the proven noise injection approach from pop-ml-simulator's
MLPredictionSimulator, extended with prevalence-aware bounds checking.
"""

import logging
import warnings
from typing import Dict, Optional, Tuple

import numpy as np

from .performance import (
    auc_score,
    calibration_slope,
    check_target_feasibility,
    confusion_matrix_metrics,
)

logger = logging.getLogger(__name__)


class ControlledMLModel:
    """ML model simulator with controlled performance characteristics.

    Uses 4-component noise injection to generate realistic predictions
    that achieve specified performance targets. Three operating modes
    support different evaluation needs.

    Modes:
        discrimination: Target AUC (probability ranking quality).
            Use for: comparing model discrimination, baseline evaluation.
        classification: Target PPV + sensitivity at optimized threshold.
            Use for: binary intervention decisions (treat/don't treat).
        threshold_ppv: Target AUC + PPV at a fixed operating threshold.
            Use for: probability models with a specific decision point.

    The model MUST be fitted before predictions are meaningful.
    Call fit() with representative data to optimize noise parameters.

    Usage:
        model = ControlledMLModel(mode="classification",
                                  target_sensitivity=0.80,
                                  target_ppv=0.15)
        report = model.fit(true_labels, risk_scores, rng)
        scores = model.predict(risk_scores, rng)
    """

    def __init__(
        self,
        mode: str = "discrimination",
        target_auc: float = 0.83,
        target_sensitivity: float = 0.80,
        target_ppv: float = 0.30,
        operating_threshold: Optional[float] = None,
        target_calibration_slope: float = 1.0,
        prevalence: Optional[float] = None,
    ):
        if mode not in ("discrimination", "classification", "threshold_ppv"):
            raise ValueError(
                f"mode must be 'discrimination', 'classification', "
                f"or 'threshold_ppv', got '{mode}'"
            )
        self.mode = mode
        self.target_auc = target_auc
        self.target_sensitivity = target_sensitivity
        self.target_ppv = target_ppv
        self.operating_threshold = operating_threshold
        self.target_calibration_slope = target_calibration_slope
        self.prevalence = prevalence

        # Optimized parameters (set by fit())
        self.noise_correlation: float = 0.7
        self.noise_scale: float = 0.3
        self.label_noise_strength: float = 1.0
        self.threshold: float = operating_threshold or 0.5
        self._fitted = False
        self._fit_report: Optional[Dict] = None

        # Platt scaling parameters (set by fit())
        self._platt_a: float = 1.0  # identity by default
        self._platt_b: float = 0.0

    def predict(
        self,
        risk_scores: np.ndarray,
        rng: np.random.Generator,
        true_labels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Generate predictions with controlled performance.

        Args:
            risk_scores: True underlying risk/probability scores.
            rng: Random generator (use scenario's prediction stream).
            true_labels: Binary outcomes (optional, improves realism
                via label-dependent noise if provided).

        Returns:
            Predicted scores in [0, 1].
        """
        if not self._fitted:
            warnings.warn(
                "ControlledMLModel.predict() called before fit(). "
                "Using default noise parameters — performance targets "
                "will NOT be achieved. Call fit() first.",
                UserWarning,
                stacklevel=2,
            )
        raw = self._generate_scores(
            risk_scores, rng, true_labels,
            self.noise_correlation, self.noise_scale,
            self.label_noise_strength,
        )
        return self._apply_platt(raw)

    def _apply_platt(self, raw_scores: np.ndarray) -> np.ndarray:
        """Apply Platt scaling calibration.

        Maps raw scores through sigmoid(a*logit(s) + b) to produce
        calibrated probabilities. When a=1, b=0 (default before
        fit), this is the identity transform.
        """
        eps = 1e-7
        clipped = np.clip(raw_scores, eps, 1 - eps)
        raw_logits = np.log(clipped / (1 - clipped))
        calibrated_logits = (
            self._platt_a * raw_logits + self._platt_b
        )
        calibrated = 1.0 / (1.0 + np.exp(-calibrated_logits))
        return np.clip(calibrated, 0, 1)

    def predict_binary(
        self,
        risk_scores: np.ndarray,
        rng: np.random.Generator,
        true_labels: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate predictions and binary labels.

        Returns:
            (scores, labels) where labels = scores >= threshold.
        """
        scores = self.predict(risk_scores, rng, true_labels)
        labels = (scores >= self.threshold).astype(int)
        return scores, labels

    def fit(
        self,
        true_labels: np.ndarray,
        risk_scores: np.ndarray,
        rng: np.random.Generator,
        n_iterations: int = 10,
        correlation_grid: Optional[np.ndarray] = None,
        scale_grid: Optional[np.ndarray] = None,
    ) -> Dict:
        """Optimize noise parameters to achieve performance targets.

        Grid searches over (noise_correlation, noise_scale) with inner
        threshold sweep, averaging metrics over multiple iterations.

        Args:
            true_labels: Binary ground truth (0/1).
            risk_scores: True underlying risk scores.
            rng: Random generator.
            n_iterations: Seeds to average over for stability.

        Returns:
            Fit report with achieved metrics and feasibility info.
        """
        if correlation_grid is None:
            correlation_grid = np.linspace(0.3, 0.99, 15)
        if scale_grid is None:
            scale_grid = np.linspace(0.01, 0.7, 15)

        # Detect prevalence
        prev = self.prevalence or float(true_labels.mean())
        self.prevalence = prev

        # Check feasibility for classification/threshold_ppv modes
        feasibility = None
        if self.mode in ("classification", "threshold_ppv"):
            feasibility = check_target_feasibility(
                prev, self.target_ppv, self.target_sensitivity,
            )
            if not feasibility["feasible"]:
                warnings.warn(
                    f"Target PPV={self.target_ppv:.2f} at "
                    f"sensitivity={self.target_sensitivity:.2f} "
                    f"may be infeasible at prevalence={prev:.3f}. "
                    f"Max PPV at 95% specificity: "
                    f"{feasibility['max_ppv_at_spec_95']:.3f}. "
                    f"Required specificity: "
                    f"{feasibility['required_specificity']:.3f}.",
                    UserWarning,
                    stacklevel=2,
                )

        # Grid search over (correlation, scale, label_noise_strength)
        label_strengths = np.linspace(0.5, 3.0, 6)
        best_score = float("inf")
        best_params = (0.7, 0.3, 1.0, 0.5)

        for corr in correlation_grid:
            for scale in scale_grid:
                for lns in label_strengths:
                    total_score = 0.0
                    for _ in range(n_iterations):
                        scores = self._generate_scores(
                            risk_scores, rng, true_labels,
                            corr, scale, lns,
                        )
                        score, best_t = self._evaluate_params(
                            true_labels, scores,
                        )
                        total_score += score

                    avg = total_score / n_iterations
                    if avg < best_score:
                        best_score = avg
                        best_params = (corr, scale, lns, best_t)

        self.noise_correlation = best_params[0]
        self.noise_scale = best_params[1]
        self.label_noise_strength = best_params[2]
        self.threshold = best_params[3]

        # Fit Platt scaling for calibration
        raw_scores = self._generate_scores(
            risk_scores, rng, true_labels,
            self.noise_correlation, self.noise_scale,
            self.label_noise_strength,
        )
        self._fit_platt_scaling(raw_scores, true_labels)
        self._fitted = True

        # Generate final metrics report (with calibrated scores)
        final_scores = self._apply_platt(raw_scores)
        report = self._build_report(
            true_labels, final_scores, feasibility,
        )
        self._fit_report = report

        logger.info(
            "ControlledMLModel fit: mode=%s, AUC=%.3f, "
            "sens=%.3f, PPV=%.3f, threshold=%.3f",
            self.mode,
            report["achieved_auc"],
            report["achieved_sensitivity"],
            report["achieved_ppv"],
            self.threshold,
        )

        return report

    def _evaluate_params(
        self, true_labels: np.ndarray, scores: np.ndarray,
    ) -> Tuple[float, float]:
        """Score a set of predictions against targets. Returns (score, threshold)."""
        if self.mode == "discrimination":
            auc = auc_score(true_labels, scores)
            cal, _, _ = calibration_slope(true_labels, scores)
            score = (
                abs(auc - self.target_auc)
                + 0.3 * abs(cal - self.target_calibration_slope)
            )
            return score, self.operating_threshold or 0.5

        elif self.mode == "classification":
            # Search over thresholds for best PPV + sensitivity
            best_t_score = float("inf")
            best_t = 0.5
            for t in np.linspace(0.05, 0.95, 19):
                m = confusion_matrix_metrics(true_labels, scores, t)
                dist = (
                    abs(m["sensitivity"] - self.target_sensitivity)
                    + 1.5 * abs(m["ppv"] - self.target_ppv)
                )
                # Penalize extreme sensitivity
                if m["sensitivity"] > 0.95:
                    dist += (m["sensitivity"] - 0.95) * 5
                if dist < best_t_score:
                    best_t_score = dist
                    best_t = t
            return best_t_score, best_t

        else:  # threshold_ppv
            t = self.operating_threshold or 0.5
            m = confusion_matrix_metrics(true_labels, scores, t)
            auc = auc_score(true_labels, scores)
            score = (
                abs(auc - self.target_auc)
                + 1.5 * abs(m["ppv"] - self.target_ppv)
            )
            return score, t

    def _generate_scores(
        self,
        risk_scores: np.ndarray,
        rng: np.random.Generator,
        true_labels: Optional[np.ndarray],
        correlation: float,
        scale: float,
        label_noise_strength: float = 1.0,
    ) -> np.ndarray:
        """4-component noise injection from pop-ml-simulator.

        1. Correlated noise: blends true risk with random noise
        2. Label-dependent noise: positive cases get boost (scaled
           by label_noise_strength for AUC control)
        3. Independent noise: unexplained model variance
        4. Sigmoid calibration: maps to probability space
        """
        n = len(risk_scores)
        base = np.clip(risk_scores, 0, 1)

        # Component 1: correlated noise mixing
        noise = rng.normal(0, scale, n)
        blended = correlation * base + (1 - correlation) * noise

        # Component 2: label-dependent noise
        # label_noise_strength > 1 increases class separation (higher AUC)
        # label_noise_strength < 1 decreases class separation (lower AUC)
        pos_mean = 0.05 * label_noise_strength
        neg_mean = -0.025 * label_noise_strength
        pos_std = 0.05 * label_noise_strength
        neg_std = 0.05 * label_noise_strength

        if true_labels is not None:
            label_noise = np.where(
                true_labels == 1,
                rng.normal(pos_mean, pos_std, n),
                rng.normal(neg_mean, neg_std, n),
            )
        else:
            label_noise = rng.normal(0, 0.05, n)

        # Component 3: independent noise
        independent_noise = rng.normal(0, 0.1, n)

        # Combine components 2+3, scaled
        blended += (label_noise + independent_noise) * scale

        # Component 4: sigmoid calibration to [0, 1]
        scores = 1.0 / (1.0 + np.exp(-4.0 * (blended - 0.5)))
        return np.clip(scores, 0, 1)

    def _fit_platt_scaling(
        self,
        raw_scores: np.ndarray,
        true_labels: np.ndarray,
    ) -> None:
        """Fit Platt scaling: P(y=1|s) = sigmoid(a*s + b).

        Maps raw prediction scores to calibrated probabilities
        by fitting a logistic regression. Preserves ranking (AUC)
        while fixing calibration slope to ~1.0.
        """
        from scipy.optimize import minimize

        # Transform raw scores to log-odds for fitting
        eps = 1e-7
        clipped = np.clip(raw_scores, eps, 1 - eps)
        raw_logits = np.log(clipped / (1 - clipped))

        def neg_log_likelihood(params):
            a, b = params
            p = 1.0 / (1.0 + np.exp(-(a * raw_logits + b)))
            p = np.clip(p, eps, 1 - eps)
            return -np.mean(
                true_labels * np.log(p)
                + (1 - true_labels) * np.log(1 - p)
            )

        result = minimize(
            neg_log_likelihood, [1.0, 0.0],
            method="Nelder-Mead",
        )
        self._platt_a = float(result.x[0])
        self._platt_b = float(result.x[1])
        logger.debug(
            "Platt scaling: a=%.3f, b=%.3f",
            self._platt_a, self._platt_b,
        )

    def _build_report(
        self,
        true_labels: np.ndarray,
        scores: np.ndarray,
        feasibility: Optional[Dict],
    ) -> Dict:
        """Build comprehensive fit report."""
        m = confusion_matrix_metrics(true_labels, scores, self.threshold)
        auc = auc_score(true_labels, scores)
        cal, _, _ = calibration_slope(true_labels, scores)

        report = {
            "mode": self.mode,
            "noise_correlation": self.noise_correlation,
            "noise_scale": self.noise_scale,
            "threshold": self.threshold,
            "achieved_auc": auc,
            "achieved_sensitivity": m["sensitivity"],
            "achieved_specificity": m["specificity"],
            "achieved_ppv": m["ppv"],
            "achieved_npv": m["npv"],
            "achieved_f1": m["f1"],
            "achieved_calibration_slope": cal,
            "flag_rate": m["flag_rate"],
            "prevalence": self.prevalence,
        }

        if feasibility:
            report["theoretical_max_ppv_at_spec_95"] = (
                feasibility["max_ppv_at_spec_95"]
            )
            report["required_specificity"] = (
                feasibility["required_specificity"]
            )
            report["target_feasible"] = feasibility["feasible"]

        return report
