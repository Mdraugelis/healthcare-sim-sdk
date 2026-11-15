"""
Runner script for Paper 03: Kaiser AAM — Escobar et al. NEJM 2020.

Scenario class lives in ``scenario.py``. This module is the CLI /
verification harness around it.

Usage:
    cd <repo-root>
    python3 -m healthcare_sim_sdk.scenarios.paper03_kaiser_aam.run_evaluation
"""

import json

import hydra
import numpy as np
from omegaconf import DictConfig

from healthcare_sim_sdk.core.scenario import TimeConfig
from healthcare_sim_sdk.experiments.lifecycle import finalize_experiment
from healthcare_sim_sdk.experiments.runner import (
    check, run_dir, runner_settings,
)
from healthcare_sim_sdk.core.engine import BranchedSimulationEngine, CounterfactualMode
from healthcare_sim_sdk.scenarios.paper03_kaiser_aam.scenario import KaiserAAMScenario


def run_main(settings):
    print("\n=== Paper 03: Kaiser AAM (Escobar et al. 2020) ===")
    print("Calibrating to: c-stat=0.845, mortality 14.4%->9.8%")

    tc = TimeConfig(
        n_timesteps=settings["n_timesteps"],
        timestep_duration=4 / 24 / 365,
        timestep_unit="4h",
        prediction_schedule=list(range(settings["n_timesteps"])),
    )

    n = settings["n_patients"]
    scenario = KaiserAAMScenario(
        time_config=tc,
        seed=settings["seed"],
        **settings["scenario"],
    )

    engine = BranchedSimulationEngine(
        scenario, counterfactual_mode=CounterfactualMode.BRANCHED
    )
    results = engine.run(n)

    last_t = tc.n_timesteps - 1
    factual_mort_rate = 0.0
    cf_mort_rate = 0.0
    if last_t in results.outcomes:
        factual_mort_rate = (
            float(results.outcomes[last_t].secondary["mortality"].sum()) / n
        )
    if last_t in results.counterfactual_outcomes:
        cf_mort_rate = (
            float(results.counterfactual_outcomes[last_t].secondary["mortality"].sum())
            / n
        )

    mortality_delta = cf_mort_rate - factual_mort_rate

    print("\n--- RESULTS ---")
    print(f"N patients: {n}")
    print(
        f"Factual mortality rate (with AAM): {factual_mort_rate:.4f} ({factual_mort_rate * 100:.2f}%)"
    )
    print(
        f"Counterfactual mortality rate (no AAM): {cf_mort_rate:.4f} ({cf_mort_rate * 100:.2f}%)"
    )
    print(
        f"Mortality reduction: {mortality_delta * 100:.2f} pp [paper reports 4.6pp: 14.4%→9.8%]"
    )

    if scenario._model._fit_report:
        r = scenario._model._fit_report
        print("\nML Model fit report:")
        print(f"  Achieved AUC: {r.get('achieved_auc', 'N/A'):.3f} [target 0.845]")

    # Verification
    print("\n--- VERIFICATION ---")
    checks = {}
    sample_scores = results.predictions.get(5)
    if sample_scores:
        checks["no_nan_inf"] = not np.any(np.isnan(sample_scores.scores))
        checks["scores_in_unit_interval"] = bool(
            np.all(sample_scores.scores >= 0) and np.all(sample_scores.scores <= 1)
        )
    checks["mortality_plausible"] = 0.001 < factual_mort_rate < 0.30
    checks["intervention_reduces_mortality"] = mortality_delta >= 0
    print(f"  No NaN/Inf: {checks.get('no_nan_inf')}")
    print(f"  Scores in [0,1]: {checks.get('scores_in_unit_interval')}")
    print(f"  Mortality plausible: {checks['mortality_plausible']}")
    print(
        f"  Intervention reduces mortality: {checks['intervention_reduces_mortality']}"
    )
    print(
        f"  SANITY CHECKS (calibration exercise; these checks do not test the published effect size): "
        f"{'all passed' if all(checks.values()) else 'some failed'} ({sum(checks.values())}/{len(checks)})"
    )

    return {
        "factual_mortality_rate": factual_mort_rate,
        "cf_mortality_rate": cf_mort_rate,
        "mortality_delta_pp": mortality_delta * 100,
        "checks": checks,
    }


def _finalize(settings, summary, name):
    """Save, verify and register the exercise through the lifecycle."""
    checks = [
        check(key, f"Sanity check: {key.replace('_', ' ')}", bool(passed),
              category="sanity")
        for key, passed in summary["checks"].items()]
    metrics = {k: v for k, v in summary.items() if k != "checks"}
    finalize_experiment(
        run_dir(),
        {"scenario": name, "experiment_name": f"{name}_evaluation",
         "exercise": "calibration", **settings},
        metrics, validation_checks=checks, require_validated=True,
        notes="Calibration exercise: inputs were set from the published "
              "result; the checks are sanity checks only.")


@hydra.main(version_base=None, config_path="configs",
            config_name="evaluation")
def main(cfg: DictConfig) -> None:
    settings = runner_settings(cfg)
    summary = run_main(settings)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    _finalize(settings, summary, "paper03_kaiser_aam")
    print(f"\nResults saved to: {run_dir()}")


if __name__ == "__main__":
    main()
