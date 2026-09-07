"""Bounded synthetic design certification for the existing segmented ITS fit.

Certificates assess a configured estimator under explicit synthetic mechanisms.
They do not validate a clinical estimand, a deployment's assumptions, or an
analysis design that was not represented in the configuration.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import norm, t as student_t

from healthcare_sim_sdk.core.rng import RNGPartitioner
from healthcare_sim_sdk.experiments.analysis.its import segmented_regression
from healthcare_sim_sdk.experiments.uncertainty import (
    binomial_performance, summarize_values,
)


@dataclass(frozen=True)
class ITSDesign:
    """One synthetic observation design; outcome units are arbitrary continuous units.

    Each period samples ``denominator`` individuals around a population mean.
    ``denominator_variation`` changes sample size sinusoidally, not the mean.
    ``informative_censoring`` removes the largest outcomes after the break,
    creating selection bias relative to the full-population target estimand.
    """
    name: str
    n_pre: int = 36
    n_post: int = 24
    autocorrelation: float = 0.0
    hac_maxlags: int = 0
    denominator: int = 80
    denominator_variation: float = 0.0
    informative_censoring: float = 0.0
    individual_sd: float = 1.0
    temporal_sd: float = 0.1
    pre_slope: float = 0.0
    seasonal_amplitude: float = 0.0
    seasonal_period: int = 12

    def validate(self):
        if not self.name or self.n_pre < 4 or self.n_post < 4:
            raise ValueError("Each named design needs at least four pre/post periods")
        integer_fields = (self.n_pre, self.n_post, self.hac_maxlags,
                          self.denominator, self.seasonal_period)
        if any(type(v) is not int for v in integer_fields):
            raise ValueError("Periods, denominator and HAC lags must be integers")
        if not 0 <= self.hac_maxlags < self.n_pre + self.n_post:
            raise ValueError("HAC lags must be nonnegative and below the series length")
        if self.denominator < 4 or self.seasonal_period < 2:
            raise ValueError("Require denominator >= 4 and seasonal_period >= 2")
        if not -0.99 < self.autocorrelation < 0.99:
            raise ValueError("Autocorrelation must lie strictly between -.99 and .99")
        if not 0 <= self.denominator_variation < .95:
            raise ValueError("Denominator variation must be in [0, .95)")
        if not 0 <= self.informative_censoring < .9:
            raise ValueError("Informative censoring must be in [0, .9)")
        if self.individual_sd <= 0 or self.temporal_sd < 0:
            raise ValueError("Individual SD must be positive; temporal SD nonnegative")
        if not all(np.isfinite(v) for v in asdict(self).values()
                   if isinstance(v, (int, float))):
            raise ValueError("Design parameters must be finite")


@dataclass(frozen=True)
class DesignCertificationConfig:
    designs: tuple[ITSDesign, ...]
    repetitions: int = 100
    seed: int = 20260905
    planted_effect: float = -.5
    alpha: float = .05
    max_type_i: float = .10
    min_coverage: float = .90
    min_power: float = .80
    max_abs_bias: float = .10

    def validate(self):
        if not self.designs or len({d.name for d in self.designs}) != len(self.designs):
            raise ValueError("Designs must be nonempty with unique names")
        for design in self.designs:
            design.validate()
        if type(self.repetitions) is not int or self.repetitions < 2:
            raise ValueError("At least two independent repetitions are required")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("Seed must be a nonnegative integer")
        if not np.isfinite(self.planted_effect) or self.planted_effect == 0:
            raise ValueError("planted_effect must be finite and nonzero")
        if not all(0 < v < 1 for v in (
                self.alpha, self.max_type_i, self.min_coverage, self.min_power)):
            raise ValueError("Probability criteria must lie strictly between 0 and 1")
        if not np.isfinite(self.max_abs_bias) or self.max_abs_bias <= 0:
            raise ValueError("max_abs_bias must be positive and finite")


@dataclass(frozen=True)
class SyntheticITSData:
    observations: np.ndarray
    population_mean: np.ndarray
    denominators: np.ndarray
    observed_denominators: np.ndarray
    true_level_change: float


def generate_design_data(
    design: ITSDesign, effect: float, seed: int,
) -> SyntheticITSData:
    """Generate an independent cohort per period, retaining population truth.

    The target level change is the intervention's planted additive shift in
    the full-population mean. AR(1) shocks have stationary zero mean. Censoring
    changes who is observed; it never changes the planted target truth.
    """
    design.validate()
    if not np.isfinite(effect):
        raise ValueError("Effect must be finite")
    rng = RNGPartitioner(seed).create_streams()
    n = design.n_pre + design.n_post
    time = np.arange(n)
    post = time >= design.n_pre
    population_mean = (design.pre_slope * time + effect * post
                       + design.seasonal_amplitude * np.sin(
                           2 * np.pi * time / design.seasonal_period))
    denominators = np.maximum(4, np.rint(design.denominator * (
        1 + design.denominator_variation * np.sin(
            2 * np.pi * time / design.seasonal_period)))).astype(int)
    shocks = np.empty(n)
    shocks[0] = rng.temporal.normal(0, design.temporal_sd)
    innovation_sd = design.temporal_sd * np.sqrt(1 - design.autocorrelation ** 2)
    for k in range(1, n):
        shocks[k] = (design.autocorrelation * shocks[k - 1]
                     + rng.temporal.normal(0, innovation_sd))
    observations = np.empty(n)
    observed_denominators = denominators.copy()
    for k in range(n):
        values = (population_mean[k] + shocks[k]
                  + rng.outcomes.normal(0, design.individual_sd, denominators[k]))
        if post[k] and design.informative_censoring:
            keep = max(1, int(np.floor(
                len(values) * (1 - design.informative_censoring))))
            values = np.sort(values)[:keep]
            observed_denominators[k] = keep
        observations[k] = values.mean()
    return SyntheticITSData(observations, population_mean, denominators,
                            observed_denominators, float(effect))


def _upper_gate(metric, limit):
    if metric["ci95_high"] <= limit:
        return "PASS"
    if metric["ci95_low"] > limit:
        return "FAIL"
    return "INCONCLUSIVE"


def _lower_gate(metric, limit):
    if metric["ci95_low"] >= limit:
        return "PASS"
    if metric["ci95_high"] < limit:
        return "FAIL"
    return "INCONCLUSIVE"


def _bias_gate(metric, limit):
    low, high = metric["mean_ci95_low"], metric["mean_ci95_high"]
    if low >= -limit and high <= limit:
        return "PASS"
    if low > limit or high < -limit:
        return "FAIL"
    return "INCONCLUSIVE"


def _overall(states):
    if "FAIL" in states:
        return "FAIL"
    return "PASS" if all(s == "PASS" for s in states) else "INCONCLUSIVE"


def run_design_certification(config: DesignCertificationConfig) -> dict[str, Any]:
    """Run every configured null/effect repetition with explicit truth and precision.

    The Type I metric is any two-sided rejection under the null. Power additionally
    requires the estimated sign to match the planted effect. Coverage uses the
    estimator's OLS Student-t interval or HAC asymptotic-normal interval. Gate
    decisions use Monte Carlo intervals, not the point estimates alone.
    """
    config.validate()
    config_dict = asdict(config)
    cells = []
    seed_rng = RNGPartitioner(config.seed).create_streams().population
    used_seeds = set()
    for design in config.designs:
        arms = {}
        for arm, truth in (("null", 0.0), ("effect", config.planted_effect)):
            rows: list[dict[str, Any]] = []
            errors: list[dict[str, Any]] = []
            for repetition in range(config.repetitions):
                seed = int(seed_rng.integers(0, 2**32 - 1))
                while seed in used_seeds:
                    seed = int(seed_rng.integers(0, 2**32 - 1))
                used_seeds.add(seed)
                try:
                    data = generate_design_data(design, truth, seed)
                    fit = segmented_regression(
                        data.observations, design.n_pre, hac_maxlags=design.hac_maxlags)
                    critical = (norm.ppf(1 - config.alpha / 2)
                                if design.hac_maxlags else student_t.ppf(
                                    1 - config.alpha / 2,
                                    design.n_pre + design.n_post - 4))
                    estimate, se = fit.level_change, fit.level_change_se
                    lower, upper = estimate - critical * se, estimate + critical * se
                    if (not np.isfinite([estimate, se, lower, upper,
                                         fit.level_change_pvalue]).all()
                            or se < 0 or not 0 <= fit.level_change_pvalue <= 1):
                        raise ValueError("Estimator returned invalid inference")
                    rows.append({
                        "repetition": repetition, "seed": seed, "truth": truth,
                        "estimate": estimate, "se": se, "ci_lower": float(lower),
                        "ci_upper": float(upper), "pvalue": fit.level_change_pvalue,
                        "rejected": bool(fit.level_change_pvalue < config.alpha),
                        "correct_direction": bool(estimate * truth > 0),
                        "covered": bool(lower <= truth <= upper),
                        "denominator_min": int(data.denominators.min()),
                        "denominator_max": int(data.denominators.max()),
                        "observed_denominator_min": int(
                            data.observed_denominators.min()),
                        "observed_denominator_max": int(
                            data.observed_denominators.max()),
                    })
                except Exception as exc:
                    errors.append({"repetition": repetition, "seed": seed,
                                   "error": f"{type(exc).__name__}: {exc}"})
            # Retain failed repetitions; never calculate a certified rate from
            # the selected subset of fits that happened to succeed.
            arm_result = {"truth": truth, "expected_repetitions": config.repetitions,
                          "completed_repetitions": len(rows), "failures": errors,
                          "repetitions": rows, "status": "FAIL" if errors else None}
            if not errors:
                rejection = binomial_performance(
                    sum(r["rejected"] for r in rows), len(rows))
                coverage = binomial_performance(
                    sum(r["covered"] for r in rows), len(rows))
                bias = summarize_values(r["estimate"] - truth for r in rows)
                criteria = {"coverage": _lower_gate(coverage, config.min_coverage),
                            "absolute_bias": _bias_gate(bias, config.max_abs_bias)}
                arm_result.update(coverage=coverage, bias=bias)
                if arm == "null":
                    arm_result["type_i_error"] = rejection
                    criteria["type_i_error"] = _upper_gate(rejection, config.max_type_i)
                else:
                    power = binomial_performance(sum(
                        r["rejected"] and r["correct_direction"] for r in rows),
                        len(rows))
                    arm_result["power"] = power
                    criteria["power"] = _lower_gate(power, config.min_power)
                arm_result["criteria"] = criteria
                arm_result["status"] = _overall(list(criteria.values()))
            arms[arm] = arm_result
        cells.append({"design": asdict(design), "arms": arms,
                      "status": _overall([a["status"] for a in arms.values()])})
    return {
        "schema_version": 1,
        "dependencies": {name: version(name) for name in
                         ("numpy", "scipy", "statsmodels")},
        "scope": ("Synthetic configured-design evidence; "
                  "not clinical design certification"),
        "estimator": "healthcare_sim_sdk.experiments.analysis.its.segmented_regression",
        "estimator_sha256": hashlib.sha256(
            Path(__file__).with_name("its.py").read_bytes()).hexdigest(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config": config_dict,
        "config_sha256": hashlib.sha256(json.dumps(
            config_dict, sort_keys=True, allow_nan=False).encode()).hexdigest(),
        "status": _overall([cell["status"] for cell in cells]),
        "cells": cells,
        "interval_definitions": {
            "estimator": "1-alpha Student-t (OLS) or asymptotic-normal (HAC) interval",
            "binomial_metrics": "Exact Clopper-Pearson 95% Monte Carlo interval",
            "bias": "Student-t 95% Monte Carlo interval for mean estimate minus truth",
            "gates": "PASS only if entire MC interval meets the configured criterion",
        },
        "unsupported": [
            {"design": "DiD", "status": "NOT_RUN",
             "reason": "Repository exports DiD panels but provides no DiD estimator"},
            {"design": ("clustered sites, staggered rollout, "
                        "delayed or missing timepoints"),
             "status": "NOT_RUN", "reason": "Not modeled in this bounded generator"},
        ],
        "limitations": [
            "Continuous outcomes with independent individuals within periods; "
            "not event rates",
            "Censoring removes high outcomes after rollout; "
            "missing-timepoint imputation is untested",
            "Thresholds are user-configured operating requirements, "
            "not universal validation standards",
            "Pointwise MC bounds do not provide a simultaneous guarantee "
            "across design cells",
            "Only configured parameter combinations were run; "
            "no interpolation is certified",
        ],
    }


def write_design_evidence(result: dict, output_dir: Path | str) -> dict[str, str]:
    """Save strict JSON and a reviewer-facing summary referencing every cell."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "design_evidence.json"
    md_path = target / "design_evidence.md"
    json_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    lines = ["# Synthetic analysis-design evidence", "", result["scope"], "",
             f"Overall status: **{result['status']}**. Estimator: segmented ITS.", "",
             "[Full configurations, repetition seeds, planted truths and results]"
             "(design_evidence.json)",
             "", "| Design | Arm | Status | Type I / power (95% MC interval) | "
             "Coverage (95% MC interval) | Mean bias |",
             "| --- | --- | --- | --- | --- | --- |"]

    def rate(metric):
        return (f"{metric['rate']:.3f} [{metric['ci95_low']:.3f}, "
                f"{metric['ci95_high']:.3f}]")

    for cell in result["cells"]:
        for arm, evidence in cell["arms"].items():
            if evidence["failures"]:
                values = ("Not summarized: failed repetitions | "
                          "Not summarized | Not summarized")
            else:
                metric = evidence.get("type_i_error", evidence.get("power"))
                values = (f"{rate(metric)} | {rate(evidence['coverage'])} | "
                          f"{evidence['bias']['mean']:.4f}")
            name = cell["design"]["name"].replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {name} | {arm} | {evidence['status']} | {values} |")
    cfg = result["config"]
    lines += [
        "", "## Requirements", "",
        f"Alpha={cfg['alpha']}; maximum Type I={cfg['max_type_i']}; "
        f"minimum coverage={cfg['min_coverage']}; minimum power={cfg['min_power']}; "
        f"maximum absolute bias={cfg['max_abs_bias']} outcome units. "
        f"{cfg['repetitions']} repetitions per arm per design.",
        "", "## Interpretation", "",
        "PASS requires the entire Monte Carlo interval to satisfy each configured "
        "criterion. INCONCLUSIVE includes inadequate repetition precision. FAIL "
        "includes confidently missed criteria or failed fits; failed repetitions "
        "are never silently excluded.", "",
        "The null asks whether the estimator invents an intervention effect. The "
        "effect arm tests recovery of the planted full-population additive level "
        "change. Informative censoring changes the observed population, so high "
        "power can coexist with unacceptable bias and false-positive rates.", "",
        "## Untested scope", "",
    ]
    lines += [f"- {item['design']}: {item['status']} — {item['reason']}"
              for item in result["unsupported"]]
    lines += ["", "## Limitations", ""]
    lines += [f"- {item}" for item in result["limitations"]]
    md_path.write_text("\n".join(lines) + "\n")
    return {"json": str(json_path.resolve()), "markdown": str(md_path.resolve())}
