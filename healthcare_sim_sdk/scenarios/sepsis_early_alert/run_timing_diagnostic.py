"""Diagnostic: measure ML detection timing vs baseline detection.

For patients who develop sepsis, compare:
- Factual branch: when were they first treated? (ML or baseline)
- CF branch: when were they first treated? (baseline only)
- What is the distribution of (CF treatment time - factual treatment time)?

If positive, ML IS providing earlier treatment.
If zero, ML detection timing = baseline detection timing.

Usage:
    python run_timing_diagnostic.py
"""

import logging

import numpy as np
import yaml
from pathlib import Path

from healthcare_sim_sdk.core.engine import (
    BranchedSimulationEngine,
    CounterfactualMode,
)
from healthcare_sim_sdk.core.scenario import TimeConfig
from healthcare_sim_sdk.scenarios.sepsis_early_alert.scenario import (
    N_STATE_ROWS,
    ROW_ONSET_TIMESTEP,
    ROW_STAGE,
    ROW_TREATED,
    ROW_TREATMENT_TIMER,
    STAGE_SEPSIS,
    STAGE_DECEASED,
    SepsisConfig,
    SepsisEarlyAlertScenario,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

CONFIGS_DIR = Path(__file__).parent / "configs"


def run_diagnostic(max_hours: float, capacity: int, seed: int = 42):
    """Run TREWS and extract per-patient treatment timing."""
    path = CONFIGS_DIR / "trews_replication.yaml"
    with open(path) as f:
        raw = yaml.safe_load(f)

    n_patients = raw["n_patients"]
    n_timesteps = raw["n_timesteps"]

    tc = TimeConfig(
        n_timesteps=n_timesteps,
        timestep_duration=4 / 8760,
        timestep_unit="4hr_block",
        prediction_schedule=list(range(n_timesteps)),
    )
    cfg = SepsisConfig(
        n_patients=n_patients,
        sepsis_incidence=raw["sepsis_incidence"],
        risk_concentration=raw["risk_concentration"],
        mean_los_timesteps=raw["mean_los_timesteps"],
        los_std_timesteps=raw["los_std_timesteps"],
        ar1_rho=raw["ar1_rho"],
        ar1_sigma=raw["ar1_sigma"],
        model_auc=raw["model_auc"],
        alert_threshold_percentile=raw["alert_threshold_percentile"],
        initial_response_rate=raw["initial_response_rate"],
        fatigue_coefficient=raw["fatigue_coefficient"],
        floor_response_rate=raw["floor_response_rate"],
        treatment_effectiveness=raw["treatment_effectiveness"],
        kumar_half_life_hours=raw.get("kumar_half_life_hours", 0.0),
        max_treatment_effectiveness=raw.get(
            "max_treatment_effectiveness", 0.50,
        ),
        rapid_response_capacity=capacity,
        prog_at_risk=raw["prog_at_risk"],
        prog_sepsis=raw["prog_sepsis"],
        prog_severe=raw["prog_severe"],
        mort_sepsis=raw["mort_sepsis"],
        mort_severe=raw["mort_severe"],
        mort_shock=raw["mort_shock"],
        baseline_detection_enabled=True,
        baseline_detect_alpha=raw.get("baseline_detect_alpha", 2.0),
        baseline_detect_beta=raw.get("baseline_detect_beta", 5.0),
        baseline_detect_max_hours=max_hours,
    )

    sc = SepsisEarlyAlertScenario(
        time_config=tc, seed=seed, config=cfg,
    )
    engine = BranchedSimulationEngine(
        sc, CounterfactualMode.BRANCHED,
    )
    results = engine.run(n_patients)

    final_t = n_timesteps - 1

    # Get final state from both branches
    f_out = results.outcomes[final_t]
    cf_out = results.counterfactual_outcomes[final_t]

    f_treated = f_out.secondary["treated"]
    cf_treated = cf_out.secondary["treated"]
    f_onset = f_out.secondary["onset_timestep"]
    cf_onset = cf_out.secondary["onset_timestep"]

    # Find patients who developed sepsis on either branch
    # Use CF onset as the "natural" onset (no ML influence)
    cf_septic = cf_onset >= 0
    f_septic = f_onset >= 0

    # For treatment timing, we need treatment_timer from the final
    # timestep. treatment_start_t = final_t - treatment_timer.
    # But we need to track this per-timestep to find FIRST treatment.
    # Let's scan through timesteps to find first treatment time.

    n = n_patients
    f_first_treat_t = np.full(n, -1.0)  # -1 = never treated
    cf_first_treat_t = np.full(n, -1.0)

    for t in range(n_timesteps):
        f_t = results.outcomes[t].secondary["treated"]
        cf_t = results.counterfactual_outcomes[t].secondary["treated"]

        # Record first treatment time
        newly_f = (f_t == 1) & (f_first_treat_t < 0)
        f_first_treat_t[newly_f] = t

        newly_cf = (cf_t == 1) & (cf_first_treat_t < 0)
        cf_first_treat_t[newly_cf] = t

    # Focus on patients who were septic on CF branch (natural disease)
    septic_mask = cf_septic
    n_septic = int(septic_mask.sum())

    logger.info("")
    logger.info("=" * 70)
    logger.info(
        "TIMING DIAGNOSTIC: max_hours=%d, capacity=%d",
        max_hours, capacity,
    )
    logger.info("=" * 70)
    logger.info("Total patients: %d", n_patients)
    logger.info("Patients who developed sepsis (CF branch): %d", n_septic)
    logger.info("")

    if n_septic == 0:
        logger.info("No septic patients found.")
        return

    # Onset times (should be similar on both branches for most patients)
    onset_cf = cf_onset[septic_mask]
    onset_f = f_onset[septic_mask]

    # Treatment times for septic patients
    f_tt = f_first_treat_t[septic_mask]
    cf_tt = cf_first_treat_t[septic_mask]

    # --- Treatment coverage ---
    f_ever_treated = f_tt >= 0
    cf_ever_treated = cf_tt >= 0

    logger.info("TREATMENT COVERAGE (among septic patients):")
    logger.info(
        "  Factual:        %d / %d (%.1f%%)",
        f_ever_treated.sum(), n_septic,
        f_ever_treated.sum() / n_septic * 100,
    )
    logger.info(
        "  Counterfactual: %d / %d (%.1f%%)",
        cf_ever_treated.sum(), n_septic,
        cf_ever_treated.sum() / n_septic * 100,
    )
    logger.info("")

    # --- Onset-to-treatment delay ---
    # For treated patients, compute delay from onset to treatment
    both_treated = f_ever_treated & cf_ever_treated
    n_both = int(both_treated.sum())

    if n_both > 0:
        f_delay = f_tt[both_treated] - onset_cf[both_treated]
        cf_delay = cf_tt[both_treated] - onset_cf[both_treated]
        timing_advantage = cf_delay - f_delay  # positive = ML was earlier

        logger.info(
            "ONSET-TO-TREATMENT DELAY "
            "(among %d patients treated on both branches):",
            n_both,
        )
        logger.info("  Factual (ML + baseline):")
        logger.info(
            "    Mean:   %.1f timesteps (%.1f hours)",
            f_delay.mean(), f_delay.mean() * 4,
        )
        logger.info(
            "    Median: %.1f timesteps (%.1f hours)",
            np.median(f_delay), np.median(f_delay) * 4,
        )
        logger.info(
            "    Min/Max: %.0f / %.0f timesteps",
            f_delay.min(), f_delay.max(),
        )
        logger.info("")
        logger.info("  Counterfactual (baseline only):")
        logger.info(
            "    Mean:   %.1f timesteps (%.1f hours)",
            cf_delay.mean(), cf_delay.mean() * 4,
        )
        logger.info(
            "    Median: %.1f timesteps (%.1f hours)",
            np.median(cf_delay), np.median(cf_delay) * 4,
        )
        logger.info(
            "    Min/Max: %.0f / %.0f timesteps",
            cf_delay.min(), cf_delay.max(),
        )
        logger.info("")
        logger.info("  ML TIMING ADVANTAGE (CF delay - F delay):")
        logger.info(
            "    Mean:   %.1f timesteps (%.1f hours)",
            timing_advantage.mean(), timing_advantage.mean() * 4,
        )
        logger.info(
            "    Median: %.1f timesteps (%.1f hours)",
            np.median(timing_advantage),
            np.median(timing_advantage) * 4,
        )

        # Distribution of timing advantage
        earlier = (timing_advantage > 0).sum()
        same = (timing_advantage == 0).sum()
        later = (timing_advantage < 0).sum()
        logger.info("")
        logger.info("  Distribution:")
        logger.info(
            "    ML earlier: %d (%.1f%%)",
            earlier, earlier / n_both * 100,
        )
        logger.info(
            "    Same time:  %d (%.1f%%)",
            same, same / n_both * 100,
        )
        logger.info(
            "    ML later:   %d (%.1f%%)",
            later, later / n_both * 100,
        )
        logger.info("")

        # Histogram of timing advantage
        logger.info("  Timing advantage histogram (timesteps):")
        bins = [-10, -5, -3, -1, 0, 1, 3, 5, 10, 20]
        counts, edges = np.histogram(timing_advantage, bins=bins)
        for i in range(len(counts)):
            bar = "#" * min(counts[i], 50)
            logger.info(
                "    [%3d,%3d): %4d %s",
                int(edges[i]), int(edges[i + 1]),
                counts[i], bar,
            )

    # --- Patients treated ONLY on factual (ML-only benefit) ---
    f_only = f_ever_treated & ~cf_ever_treated
    n_f_only = int(f_only.sum())
    logger.info("")
    logger.info(
        "Patients treated ONLY on factual (not by baseline): %d",
        n_f_only,
    )

    # --- Patients treated BEFORE onset on factual ---
    f_before_onset = f_ever_treated & (f_tt[..., ] < onset_cf)
    # Need to handle this carefully with the mask
    f_tt_septic = f_tt[f_ever_treated]
    onset_septic = onset_cf[f_ever_treated]
    before_onset = (f_tt_septic < onset_septic) & (f_tt_septic >= 0)
    logger.info(
        "Patients treated BEFORE sepsis onset (factual): %d / %d",
        before_onset.sum(), f_ever_treated.sum(),
    )

    # --- Kumar effectiveness comparison ---
    if n_both > 0:
        half_life = cfg.kumar_half_life_hours
        max_eff = cfg.max_treatment_effectiveness
        if half_life > 0:
            # Clamp delay to >= 0 (patients treated before onset
            # get max effectiveness, not exponential blowup)
            f_delay_hrs = np.maximum(f_delay, 0) * 4.0
            cf_delay_hrs = np.maximum(cf_delay, 0) * 4.0
            f_eff = max_eff * np.power(0.5, f_delay_hrs / half_life)
            cf_eff = max_eff * np.power(
                0.5, cf_delay_hrs / half_life,
            )
            logger.info("")
            logger.info(
                "KUMAR EFFECTIVENESS "
                "(among %d patients treated on both):",
                n_both,
            )
            logger.info(
                "  Factual mean effectiveness:  %.1f%%",
                f_eff.mean() * 100,
            )
            logger.info(
                "  CF mean effectiveness:       %.1f%%",
                cf_eff.mean() * 100,
            )
            logger.info(
                "  ML advantage:                %.1f pp",
                (f_eff.mean() - cf_eff.mean()) * 100,
            )

    logger.info("")
    logger.info("=" * 70)


def main():
    configs = [
        (24, 10, "Default TREWS config"),
        (24, 50, "Higher capacity"),
        (72, 10, "Long baseline delay"),
        (72, 50, "Long delay + higher capacity"),
    ]

    for max_hours, capacity, label in configs:
        logger.info("\n>>> %s <<<", label)
        run_diagnostic(max_hours, capacity)


if __name__ == "__main__":
    main()
