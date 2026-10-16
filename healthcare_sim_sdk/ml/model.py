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
    auc_exact,
    calibration_slope,
    check_target_feasibility,
    confusion_matrix_metrics,
    required_auc_for_operating_point,
    validate_operating_point,
)

logger = logging.getLogger(__name__)

# Gate tolerance: a target is "achieved" when the relative miss is within
# 20%, floored in absolute terms so small targets are not held to a
# fraction of a percentage point (20% of a 0.02 PPV target is 0.004,
# which is inside sampling noise).
DEFAULT_TOLERANCE_REL = 0.20
DEFAULT_TOLERANCE_ABS = 0.02


def gate_tolerance(
    target: float,
    rel: float = DEFAULT_TOLERANCE_REL,
    abs_floor: float = DEFAULT_TOLERANCE_ABS,
) -> float:
    """Allowed absolute deviation from `target`."""
    return max(rel * abs(target), abs_floor)


class OperatingPointError(ValueError):
    """Raised when a fit cannot reach its configured operating point.

    Carries the fit report so callers can inspect what was achieved and
    what would be required.
    """

    def __init__(self, message: str, report: Optional[Dict] = None):
        super().__init__(message)
        self.report = report or {}


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
        target_auc: Optional[float] = None,
        target_sensitivity: float = 0.80,
        target_ppv: float = 0.30,
        operating_threshold: Optional[float] = None,
        target_calibration_slope: float = 1.0,
        prevalence: Optional[float] = None,
        binding: Optional[str] = None,
        binding_value: Optional[float] = None,
        goal_value: Optional[float] = None,
        allow_infeasible: bool = False,
        tolerance_rel: float = DEFAULT_TOLERANCE_REL,
        tolerance_abs: float = DEFAULT_TOLERANCE_ABS,
        min_positives_for_gate: int = 30,
    ):
        valid_modes = (
            "discrimination", "classification", "threshold_ppv",
            "operating_point",
        )
        if mode not in valid_modes:
            raise ValueError(
                f"mode must be one of {valid_modes}, got '{mode}'"
            )
        self.mode = mode
        self.binding = binding
        self.binding_value = binding_value
        self.goal_value = goal_value
        self.allow_infeasible = allow_infeasible
        self.tolerance_rel = tolerance_rel
        self.tolerance_abs = tolerance_abs
        # Below this many positive cases the empirical Bayes ceiling (and
        # any AUC measured on the fit labels) is too noisy to judge a
        # target against: the gates warn instead of raising.
        self.min_positives_for_gate = min_positives_for_gate
        self._required_auc: Optional[float] = None

        # target_auc defaults to None so operating_point mode can tell
        # "user specified an AUC" from "user left it alone". Every other
        # mode keeps the historical 0.83 default.
        if mode == "operating_point":
            self._validate_operating_point_config(target_auc)
            self.target_auc = target_auc
        else:
            self.target_auc = 0.83 if target_auc is None else target_auc
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
        self._last_correlation_grid: Optional[np.ndarray] = None
        self._last_scale_grid: Optional[np.ndarray] = None
        self._population_auc_ceiling: Optional[float] = None
        self._n_positives: Optional[int] = None
        self._n_entities: Optional[int] = None
        self._gate_underpowered = False

        # Platt scaling parameters (set by fit())
        self._platt_a: float = 1.0  # identity by default
        self._platt_b: float = 0.0

    def _validate_operating_point_config(
        self, target_auc: Optional[float],
    ) -> None:
        """Enforce the degrees-of-freedom rule for operating_point mode.

        The old 'classification' mode let callers pin BOTH sensitivity and
        PPV. At a known prevalence those two pin specificity through Bayes,
        so the pair names a single point in ROC space rather than a target
        region -- reachable only if some attainable ROC passes through it.
        That is why it silently failed.

        This mode is well posed by construction:
          * exactly ONE binding constraint (ppv | sensitivity) -> fixes the
            threshold once a curve exists;
          * exactly ONE quality spec (target_auc, or a goal for the other
            metric, which implies an AUC) -> fixes the curve.
        """
        if self.binding not in ("ppv", "sensitivity"):
            raise ValueError(
                "operating_point mode requires binding='ppv' or "
                f"binding='sensitivity'; got {self.binding!r}"
            )
        if self.binding_value is None:
            raise ValueError(
                "operating_point mode requires binding_value (the value "
                f"of {self.binding} to hold exactly)"
            )
        if not 0.0 < self.binding_value < 1.0:
            raise ValueError(
                f"binding_value must lie in (0, 1); got "
                f"{self.binding_value}"
            )

        has_goal = self.goal_value is not None
        has_auc = target_auc is not None
        if has_goal and has_auc:
            other = "sensitivity" if self.binding == "ppv" else "ppv"
            raise ValueError(
                "operating_point mode is over-determined: specify EITHER "
                f"target_auc OR goal_value (the {other} goal), not both. "
                "Holding a binding constraint at a known prevalence means "
                "the other metric is a consequence of the AUC, so fixing "
                "both leaves no degrees of freedom."
            )
        if not has_goal and not has_auc:
            other = "sensitivity" if self.binding == "ppv" else "ppv"
            raise ValueError(
                "operating_point mode is under-determined: specify EITHER "
                f"target_auc OR goal_value (the {other} goal)."
            )
        if has_goal and not 0.0 < self.goal_value < 1.0:
            raise ValueError(
                f"goal_value must lie in (0, 1); got {self.goal_value}"
            )

    def _place_threshold_for_binding(
        self, true_labels: np.ndarray, scores: np.ndarray,
        binding: Optional[str] = None,
        binding_value: Optional[float] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Put the threshold exactly on the binding constraint.

        Sweeps every achievable cut rather than a fixed grid: for a rare
        outcome the useful region is a narrow band near the top of the
        score distribution, which a coarse linspace steps straight over.

        Returns (threshold, metrics_at_that_threshold).
        """
        binding = binding or self.binding
        binding_value = (
            self.binding_value if binding_value is None else binding_value
        )
        order = np.argsort(-scores)
        y = true_labels[order]
        s = scores[order]
        n_pos = max(int(y.sum()), 1)

        tp = np.cumsum(y)
        n_flagged = np.arange(1, len(y) + 1)
        ppv_curve = tp / n_flagged
        sens_curve = tp / n_pos

        if binding == "ppv":
            # Largest flagged set whose PPV still meets the target: the
            # most sensitivity obtainable without breaking the constraint.
            ok = np.where(ppv_curve >= binding_value)[0]
            idx = int(ok[-1]) if len(ok) else 0
        else:
            ok = np.where(sens_curve >= binding_value)[0]
            idx = int(ok[0]) if len(ok) else len(y) - 1

        # Threshold sits just below the last included score so that
        # `scores >= threshold` reproduces this exact flagged set.
        cut = float(s[idx])
        below = s[s < cut]
        threshold = (
            float((cut + below[0]) / 2.0) if len(below) else cut
        )
        return threshold, {
            "ppv": float(ppv_curve[idx]),
            "sensitivity": float(sens_curve[idx]),
            "flag_rate": float((idx + 1) / len(y)),
        }

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
            true_labels: Accepted for backward compatibility and
                IGNORED. A simulated model must not see the outcome it
                is predicting; scores depend on risk_scores only, so
                calling with or without labels yields identical output.

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
            risk_scores, rng, None,
            self.noise_correlation, self.noise_scale,
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

        The score generator is label-free: it sees only the true risk
        scores, never the outcomes. Fitting and prediction therefore run
        the SAME generator, so what the fit report promises is what
        predict() delivers. (Before this, fitting injected a label-
        dependent boost that predict() dropped whenever it was called
        without outcomes -- the deployed model was weaker than reported
        -- and kept whenever it was called with them, letting a
        simulated model beat an oracle that knows every true risk.)

        Steps:
          1. Bayes ceiling: the AUC of the true risks themselves. A
             label-free model cannot exceed it, so every mode with an AUC
             requirement is gated against it before searching.
          2. Grid search (correlation x scale) for the curve that hits
             the required AUC, then a 1-D refinement of the scale.
          3. Platt scaling on one draw of label-free scores.
          4. Threshold placement (operating_point / classification) on
             that draw.
          5. The report is measured on a FRESH, independent noise draw,
             i.e. exactly what predict() will produce on this population.

        Args:
            true_labels: Binary ground truth (0/1).
            risk_scores: True underlying risk scores.
            rng: Random generator.
            n_iterations: Noise draws to average per grid point.

        Returns:
            Fit report with achieved (held-out draw) metrics.
        """
        if correlation_grid is None:
            correlation_grid = np.linspace(0.3, 0.99, 15)
        if scale_grid is None:
            scale_grid = np.linspace(0.01, 0.7, 15)
        # Retained so the report can tell a converged optimum from one
        # that merely stopped at the edge of the search box.
        self._last_correlation_grid = np.asarray(correlation_grid)
        self._last_scale_grid = np.asarray(scale_grid)

        true_labels = np.asarray(true_labels)
        prev = self.prevalence or float(true_labels.mean())
        self.prevalence = prev

        # The population's Bayes ceiling: the AUC of an oracle that knows
        # every entity's TRUE risk. Labels are stochastic draws from those
        # risks, so even perfect knowledge cannot order outcomes perfectly.
        # A model that sees only risk can match this at best; noise only
        # lowers it. Exact rank AUC, so narrow low-risk bands are not
        # under-resolved.
        self._population_auc_ceiling = float(
            auc_exact(true_labels, risk_scores)
        )
        self._n_positives = int(np.sum(true_labels))
        self._n_entities = int(len(true_labels))

        # With only one outcome class present, AUC (and the ceiling) is
        # undefined, so there is nothing to search or calibrate against.
        # Keep the label-free generator at its defaults, leave Platt at
        # identity, and say so rather than fitting to noise.
        if not 0 < self._n_positives < len(true_labels):
            return self._fit_degenerate(true_labels, risk_scores, rng)

        # Work out the AUC this mode requires, gating it against the
        # ceiling. Every mode is gated, not just operating_point.
        op_plan = None
        feasibility = None
        if self.mode == "operating_point":
            op_plan = self._plan_operating_point(prev)
            self._required_auc = op_plan["required_auc"]
        elif self.mode == "classification":
            feasibility = check_target_feasibility(
                prev, self.target_ppv, self.target_sensitivity,
            )
            self._required_auc = self._plan_classification(prev)
        else:  # discrimination, threshold_ppv
            if self.mode == "threshold_ppv":
                feasibility = check_target_feasibility(
                    prev, self.target_ppv, self.target_sensitivity,
                )
            self._required_auc = float(self.target_auc)
            self._check_population_ceiling(self._required_auc)

        # Search the label-free generator for the required AUC.
        best = self._search_curve(
            true_labels, risk_scores, rng, n_iterations,
            correlation_grid, scale_grid,
        )
        self.noise_correlation, self.noise_scale = best
        self.label_noise_strength = 0.0   # retained attribute; unused

        # Platt scaling on one draw of label-free scores.
        fit_raw = self._generate_scores(
            risk_scores, rng, None,
            self.noise_correlation, self.noise_scale,
        )
        self._fit_platt_scaling(fit_raw, true_labels)
        self._fitted = True
        fit_scores = self._apply_platt(fit_raw)

        # Thresholds live on the CALIBRATED scale, because that is the
        # scale predict()/predict_binary() return.
        if self.mode == "operating_point":
            self.threshold, _ = self._place_threshold_for_binding(
                true_labels, fit_scores, self.binding, self.binding_value,
            )
        elif self.mode == "classification":
            self.threshold, _ = self._place_threshold_for_binding(
                true_labels, fit_scores, "ppv", self.target_ppv,
            )
        else:
            self.threshold = self.operating_threshold or 0.5

        # Report on an independent draw: what predict() will deliver.
        report_scores = self._apply_platt(self._generate_scores(
            risk_scores, rng, None,
            self.noise_correlation, self.noise_scale,
        ))
        report = self._build_report(
            true_labels, report_scores, feasibility,
        )
        report["required_auc"] = self._required_auc
        report["report_basis"] = "independent_noise_draw"
        report["label_free"] = True
        if op_plan is not None:
            report.update(self._operating_point_report(op_plan, report))
        self._fit_report = report

        logger.info(
            "ControlledMLModel fit: mode=%s, AUC=%.3f (ceiling %.3f), "
            "sens=%.3f, PPV=%.3f, threshold=%.3f",
            self.mode,
            report["achieved_auc"],
            self._population_auc_ceiling,
            report["achieved_sensitivity"],
            report["achieved_ppv"],
            self.threshold,
        )

        if self.mode == "operating_point":
            self._enforce_operating_point(report)
        else:
            self._check_auc_delivered(report)

        return report

    def _fit_degenerate(
        self,
        true_labels: np.ndarray,
        risk_scores: np.ndarray,
        rng: np.random.Generator,
    ) -> Dict:
        """Fit path for a single-class population (AUC undefined)."""
        warnings.warn(
            f"ControlledMLModel.fit() received {self._n_positives} "
            f"positives in {len(true_labels)} entities; AUC is undefined, "
            f"so default noise parameters are kept and no performance "
            f"target is checked.",
            UserWarning,
            stacklevel=3,
        )
        self._required_auc = None
        self.label_noise_strength = 0.0
        self._platt_a, self._platt_b = 1.0, 0.0
        self.threshold = self.operating_threshold or 0.5
        self._fitted = True
        scores = self._apply_platt(self._generate_scores(
            risk_scores, rng, None,
            self.noise_correlation, self.noise_scale,
        ))
        report = self._build_report(true_labels, scores, None)
        report.update({
            "required_auc": None,
            "report_basis": "independent_noise_draw",
            "label_free": True,
            "degenerate_population": True,
        })
        self._fit_report = report
        return report

    def _plan_classification(self, prev: float) -> Optional[float]:
        """Required AUC for the legacy (sensitivity, PPV) pair.

        At a known prevalence the pair names one point in ROC space, so
        it implies a minimum AUC -- the same solve operating_point uses
        with binding='ppv'. Gate that AUC against the population ceiling.
        """
        check = validate_operating_point(prev, self.target_ppv)
        solve = None
        if check["valid"]:
            solve = required_auc_for_operating_point(
                prevalence=prev, binding="ppv",
                binding_value=self.target_ppv,
                goal_value=self.target_sensitivity,
            )
        if solve is None or solve["required_auc"] is None:
            msg = (
                f"Classification targets sensitivity="
                f"{self.target_sensitivity:.3f} at PPV="
                f"{self.target_ppv:.3f} are unreachable at prevalence "
                f"{prev:.4f} by any model."
            )
            if not self.allow_infeasible:
                raise OperatingPointError(
                    msg + " Lower one target, or set allow_infeasible="
                    "True to fit the best available curve.",
                    report={"solve": solve},
                )
            warnings.warn(msg, UserWarning, stacklevel=3)
            return self._population_auc_ceiling
        required = float(solve["required_auc"])
        self._check_population_ceiling(required)
        if (self.allow_infeasible and self._population_auc_ceiling
                is not None and required > self._population_auc_ceiling):
            return self._population_auc_ceiling
        return required

    def _search_curve(
        self,
        true_labels: np.ndarray,
        risk_scores: np.ndarray,
        rng: np.random.Generator,
        n_iterations: int,
        correlation_grid: np.ndarray,
        scale_grid: np.ndarray,
    ) -> Tuple[float, float]:
        """Find (correlation, scale) whose label-free AUC hits the target.

        Coarse grid first, then bisection on scale at the chosen
        correlation: AUC falls monotonically as scale grows, so the
        grid's step size no longer limits accuracy.
        """
        target = float(self._required_auc)
        n = len(risk_scores)
        # Common random numbers: one fixed set of draws for every
        # candidate, so differences between candidates are signal.
        draws = [
            (rng.normal(0, 1, n), rng.normal(0, 1, n))
            for _ in range(max(int(n_iterations), 5))
        ]

        def mean_auc(corr: float, scale: float) -> float:
            total = 0.0
            for z_corr, z_ind in draws:
                s = self._scores_from_noise(
                    risk_scores, z_corr, z_ind, corr, scale,
                )
                total += auc_exact(true_labels, s)
            return total / len(draws)

        best_err = float("inf")
        best = (0.7, 0.3)
        for corr in correlation_grid:
            for scale in scale_grid:
                err = abs(mean_auc(corr, scale) - target)
                if err < best_err:
                    best_err = err
                    best = (float(corr), float(scale))

        corr, scale = best
        grid = np.sort(np.asarray(scale_grid, dtype=float))
        if len(grid) >= 2 and best_err > 1e-3:
            step = float(grid[1] - grid[0])
            lo = max(float(grid[0]), scale - step)
            hi = min(float(grid[-1]), scale + step)
            auc_lo, auc_hi = mean_auc(corr, lo), mean_auc(corr, hi)
            # Bisection only if the target is bracketed on this interval.
            if (auc_lo - target) * (auc_hi - target) < 0:
                for _ in range(12):
                    mid = 0.5 * (lo + hi)
                    auc_mid = mean_auc(corr, mid)
                    if (auc_mid - target) * (auc_lo - target) > 0:
                        lo, auc_lo = mid, auc_mid
                    else:
                        hi, auc_hi = mid, auc_mid
                scale = lo if abs(auc_lo - target) <= abs(
                    auc_hi - target) else hi
        return corr, scale

    def _check_auc_delivered(self, report: Dict) -> None:
        """Warn (non-operating_point modes) when the held-out AUC misses.

        operating_point raises instead; legacy modes warn so existing
        scenario runs surface the gap without stopping.
        """
        required = report.get("required_auc")
        if required is None:
            return
        gap = report["achieved_auc"] - required
        report["auc_within_tolerance"] = abs(gap) <= self.tolerance_abs
        report["gate_underpowered"] = (
            int(self._n_positives or 0) < self.min_positives_for_gate
        )
        if report["gate_underpowered"]:
            return      # already warned; a miss on so few events is noise
        if not report["auc_within_tolerance"]:
            warnings.warn(
                f"ControlledMLModel ({self.mode}) delivered AUC "
                f"{report['achieved_auc']:.4f} on an independent draw "
                f"against a required {required:.4f} (tolerance "
                f"{self.tolerance_abs:.3f}). Population Bayes ceiling: "
                f"{self._population_auc_ceiling:.4f}.",
                UserWarning,
                stacklevel=3,
            )

    def _plan_operating_point(self, prev: float) -> Dict:
        """Translate (binding, goal) into the AUC the curve must reach."""
        ppv_value = (
            self.binding_value if self.binding == "ppv"
            else self.goal_value
        )
        if ppv_value is not None:
            check = validate_operating_point(prev, ppv_value)
            if not check["valid"]:
                raise OperatingPointError(
                    f"Degenerate operating point: {check['reason']}"
                )

        if self.target_auc is not None:
            self._check_population_ceiling(float(self.target_auc))
            return {
                "required_auc": float(self.target_auc),
                "required_auc_source": "target_auc",
                "goal_reachable": True,
                "solve": None,
            }

        solve = required_auc_for_operating_point(
            prevalence=prev,
            binding=self.binding,
            binding_value=self.binding_value,
            goal_value=self.goal_value,
        )
        if solve["required_auc"] is None:
            other = "sensitivity" if self.binding == "ppv" else "PPV"
            raise OperatingPointError(
                f"Operating point is unreachable at any AUC below "
                f"{solve['auc_bounds'][1]}: holding {self.binding}="
                f"{self.binding_value:.4f} at prevalence {prev:.4f} "
                f"yields at most {other}="
                f"{solve['attainable_at_auc_hi']:.4f}, short of the goal "
                f"{self.goal_value:.4f}.\n"
                f"Options: lower the goal to <= "
                f"{solve['attainable_at_auc_hi']:.4f}, or relax the "
                f"binding {self.binding} target.",
                report={"solve": solve},
            )
        plan = {
            "required_auc": solve["required_auc"],
            "required_auc_source": "derived_from_goal",
            "goal_reachable": True,
            "solve": solve,
        }
        self._check_population_ceiling(plan["required_auc"])
        return plan

    def _check_population_ceiling(self, required_auc: float) -> None:
        """Bail out before searching for an AUC the population cannot host.

        The ceiling is known as soon as labels and risks arrive, so there
        is no reason to spend a full grid search discovering it. Skipped
        when allow_infeasible is set -- that caller has asked to proceed
        and have the gap recorded.
        """
        ceiling = self._population_auc_ceiling
        if ceiling is None:
            return
        # The ceiling is ESTIMATED from finite data, so it carries
        # sampling noise -- with few positives its standard error can
        # reach several points of AUC. Refuse only when the requirement
        # clears it by more than the tolerance AND by two standard errors
        # (Hanley-McNeil), so the verdict does not flip from seed to seed.
        se = self._ceiling_standard_error()
        margin = max(self.tolerance_abs, 2.0 * se)
        if required_auc <= ceiling + margin:
            return
        if self.allow_infeasible:
            return          # the gate will record it after fitting
        n_pos = int(self._n_positives or 0)
        if n_pos < self.min_positives_for_gate:
            # One or ten positives cannot pin a ceiling; refusing to run
            # on that evidence would be as wrong as trusting it.
            warnings.warn(
                f"Mode '{self.mode}' requires AUC {required_auc:.4f}; the "
                f"population ceiling estimate is {ceiling:.4f} but rests "
                f"on only {n_pos} positive case(s) (gate needs "
                f"{self.min_positives_for_gate}), so it is not enforced. "
                f"The model is being fit on too few events to verify its "
                f"configured discrimination.",
                UserWarning,
                stacklevel=4,
            )
            self._gate_underpowered = True
            return
        relax = (
            f", relax the binding {self.binding} target"
            if self.binding else ""
        )
        noise_note = (
            f"\n(Ceiling estimated from {n_pos} positive cases; with few "
            f"positives this estimate is itself noisy.)"
            if n_pos and n_pos < 100 else ""
        )
        raise OperatingPointError(
            f"Mode '{self.mode}' requires AUC {required_auc:.4f}, which "
            f"exceeds this population's Bayes ceiling of {ceiling:.4f} "
            f"(standard error {se:.4f})."
            f"{noise_note}\n"
            f"That ceiling is the AUC of an oracle knowing every entity's "
            f"true risk; outcomes are stochastic draws from those risks, "
            f"so no model of any kind can do better on this population.\n"
            f"Options: lower the target{relax}, or increase risk "
            f"heterogeneity (for beta_distributed_risks, a LOWER "
            f"`concentration` widens the risk spread and raises this "
            f"ceiling).\n"
            f"Set allow_infeasible=True to fit anyway and have the gap "
            f"recorded in the report.",
            report={
                "required_auc": required_auc,
                "population_auc_ceiling": ceiling,
            },
        )

    def _ceiling_standard_error(self) -> float:
        """Hanley-McNeil standard error of the estimated ceiling AUC."""
        a = self._population_auc_ceiling
        n1 = int(self._n_positives or 0)
        n0 = int(self._n_entities or 0) - n1
        if a is None or not np.isfinite(a) or n1 < 1 or n0 < 1:
            return 0.0
        q1 = a / (2 - a)
        q2 = 2 * a * a / (1 + a)
        var = (a * (1 - a) + (n1 - 1) * (q1 - a * a)
               + (n0 - 1) * (q2 - a * a)) / (n1 * n0)
        return float(np.sqrt(max(var, 0.0)))

    def _operating_point_report(
        self, op_plan: Dict, report: Dict,
    ) -> Dict:
        """Operating-point specific fields appended to the fit report."""
        other = "sensitivity" if self.binding == "ppv" else "ppv"
        achieved_binding = report[f"achieved_{self.binding}"]
        achieved_other = report[f"achieved_{other}"]
        extra = {
            "binding": self.binding,
            "binding_value": self.binding_value,
            "achieved_binding": achieved_binding,
            "goal_metric": other,
            "goal_value": self.goal_value,
            "achieved_goal_metric": achieved_other,
            "required_auc": op_plan["required_auc"],
            "required_auc_source": op_plan["required_auc_source"],
            "binding_within_tolerance": abs(
                achieved_binding - self.binding_value
            ) <= gate_tolerance(
                self.binding_value, self.tolerance_rel, self.tolerance_abs
            ),
        }
        if self.goal_value is not None:
            extra["goal_within_tolerance"] = abs(
                achieved_other - self.goal_value
            ) <= gate_tolerance(
                self.goal_value, self.tolerance_rel, self.tolerance_abs
            )
        # A search that stops on its own grid edge has not converged --
        # the optimum is probably outside the box. Check BOTH ends of
        # BOTH grids: chasing a high AUC pins correlation at its maximum
        # and can pin scale at its minimum, so a one-sided test misses
        # half the cases.
        pinned = []
        for name, value, grid in (
            ("noise_correlation", self.noise_correlation,
             self._last_correlation_grid),
            ("noise_scale", self.noise_scale, self._last_scale_grid),
        ):
            if grid is None or len(grid) < 2:
                continue
            lo, hi = float(np.min(grid)), float(np.max(grid))
            step = (hi - lo) / (len(grid) - 1)
            if value <= lo + step * 0.5:
                pinned.append(f"{name}={value:.4f} (grid minimum {lo:.4f})")
            elif value >= hi - step * 0.5:
                pinned.append(f"{name}={value:.4f} (grid maximum {hi:.4f})")
        extra["auc_search_boundary_pinned"] = bool(pinned)
        extra["auc_search_pinned_params"] = pinned
        ceiling = self._population_auc_ceiling
        extra["population_ceiling_exceeded"] = bool(
            ceiling is not None
            and op_plan["required_auc"] > ceiling + self.tolerance_abs
        )
        extra["n_positives"] = self._n_positives
        return extra

    def _enforce_operating_point(self, report: Dict) -> None:
        """Fail loudly when the achieved point misses the configuration.

        Two-sided: overshoot is also a miss. ControlledMLModel exists to
        simulate a model with SPECIFIED characteristics, so landing above
        a target means the simulation is not running what was configured.
        """
        misses = []
        checks = [(self.binding, self.binding_value, "binding")]
        if self.goal_value is not None:
            other = "sensitivity" if self.binding == "ppv" else "ppv"
            checks.append((other, self.goal_value, "goal"))
        # The curve spec is a target too. Without this, binding on PPV
        # (always satisfiable, since the threshold is placed exactly on
        # it) means a target_auc-only config can miss its AUC by any
        # margin and still report success -- the very failure this mode
        # exists to prevent.
        if report.get("required_auc") is not None:
            checks.append(("auc", report["required_auc"], "curve"))

        for metric, target, role in checks:
            achieved = report[f"achieved_{metric}"]
            if metric == "auc":
                # AUC lives on (0.5, 1.0]; a relative tolerance would be
                # meaningless here (20% of 0.999 is 0.2, wider than the
                # entire useful range). Use the absolute floor.
                tol = self.tolerance_abs
            else:
                tol = gate_tolerance(
                    target, self.tolerance_rel, self.tolerance_abs
                )
            if abs(achieved - target) > tol:
                misses.append(
                    f"  {role} {metric}: configured {target:.4f}, "
                    f"achieved {achieved:.4f} "
                    f"({'over' if achieved > target else 'under'} by "
                    f"{abs(achieved - target):.4f}; tolerance {tol:.4f})"
                )

        if not misses:
            return

        report["accepted_infeasible"] = bool(self.allow_infeasible)
        report["gate_misses"] = misses

        # Same symptom, three causes, different remedies. Report the AUC
        # shortfall either way, since it is the quantity that explains the
        # miss, then name the cause.
        cause = (
            f"Achieved AUC {report['achieved_auc']:.4f} against a "
            f"required {report['required_auc']:.4f}."
        )
        ceiling = report.get("population_auc_ceiling")

        # Cause 1 (dominant): the POPULATION cannot host this model. An
        # oracle knowing every true risk tops out at `ceiling`, so no
        # amount of fitting reaches the requirement. This outranks a
        # pinned grid -- the search pins BECAUSE it is chasing something
        # unreachable, so widening the grid would waste the user's time.
        if (ceiling is not None
                and report["required_auc"]
                > ceiling + self.tolerance_abs):
            cause += (
                f"\nThat requirement exceeds this population's Bayes "
                f"ceiling of {ceiling:.4f} -- the AUC of an oracle that "
                f"knows every entity's true risk. No model of any kind "
                f"can reach it on this population, because outcomes are "
                f"stochastic draws from risk and this risk distribution "
                f"is not spread out enough to separate them further.\n"
                f"Options: lower the goal, relax the binding target, or "
                f"increase risk heterogeneity in the population (for "
                f"beta_distributed_risks, a LOWER `concentration` widens "
                f"the risk spread and raises this ceiling)."
            )
        elif report.get("auc_search_boundary_pinned"):
            cause += (
                "\nThe search stopped on its own grid boundary ("
                + "; ".join(report["auc_search_pinned_params"])
                + "), so the optimum is likely outside the search box. "
                "Fix this first: widen correlation_grid / scale_grid or "
                "raise n_iterations, then re-assess."
            )
        else:
            cause += (
                " The search converged inside its grid, so the risk "
                "distribution itself does not support this operating "
                "point.\nOptions: lower the goal, relax the binding "
                "target, or raise target_auc if a stronger model is "
                "plausible."
            )

        message = (
            "Operating point not achieved.\n"
            + "\n".join(misses)
            + f"\n{cause}\n"
            "Set allow_infeasible=True to proceed anyway; the accepted "
            "gap is recorded in the fit report."
        )

        if self.allow_infeasible:
            warnings.warn(message, UserWarning, stacklevel=2)
            return
        raise OperatingPointError(message, report=report)

    def _generate_scores(
        self,
        risk_scores: np.ndarray,
        rng: np.random.Generator,
        true_labels: Optional[np.ndarray],
        correlation: float,
        scale: float,
        label_noise_strength: float = 0.0,
    ) -> np.ndarray:
        """Label-free noise injection: scores depend on risk only.

        1. Correlated noise: blends true risk with random noise.
        2. Independent noise: unexplained model variance.
        3. Sigmoid squash to [0, 1] (Platt scaling calibrates later).

        `true_labels` and `label_noise_strength` are accepted for
        backward compatibility and ignored. The earlier generator added
        a label-dependent shift that let the simulated model read the
        outcome; it inflated AUC past the population's Bayes ceiling
        when labels were passed and vanished when they were not.
        """
        n = len(risk_scores)
        z_corr = rng.normal(0, 1, n)
        z_ind = rng.normal(0, 1, n)
        return self._scores_from_noise(
            risk_scores, z_corr, z_ind, correlation, scale,
        )

    @staticmethod
    def _scores_from_noise(
        risk_scores: np.ndarray,
        z_corr: np.ndarray,
        z_ind: np.ndarray,
        correlation: float,
        scale: float,
    ) -> np.ndarray:
        """Deterministic score map given standard-normal noise draws.

        Split out so the search can reuse one set of draws across every
        candidate (common random numbers): AUC then varies smoothly with
        the parameters instead of jumping with each fresh draw.
        """
        base = np.clip(risk_scores, 0, 1)
        blended = (
            correlation * base
            + (1 - correlation) * scale * z_corr
            + 0.1 * scale * z_ind
        )
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
        auc = auc_exact(true_labels, scores)
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
            "population_auc_ceiling": self._population_auc_ceiling,
            "population_auc_ceiling_se": self._ceiling_standard_error(),
            "n_positives": self._n_positives,
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
