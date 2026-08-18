"""Calibrate reminder_effectiveness to match published effect sizes.

Binary search over reminder_effectiveness until the simulated
post-intervention no-show rate matches the target within tolerance.

Usage:
    # Chong calibration
    python -m healthcare_sim_sdk.scenarios.noshow_targeted_reminders.calibrate \
        --setting chong

    # Rosen calibration
    python -m healthcare_sim_sdk.scenarios.noshow_targeted_reminders.calibrate \
        --setting rosen
"""

import argparse
import time
from typing import Dict, Tuple

import numpy as np

from healthcare_sim_sdk.scenarios.noshow_targeted_reminders.scenario import (
    CallerConfig,
    NoShowTargetedReminderScenario,
)
from healthcare_sim_sdk.core.engine import (
    BranchedSimulationEngine,
    CounterfactualMode,
)
from healthcare_sim_sdk.core.scenario import TimeConfig
from healthcare_sim_sdk.ml.performance import auc_score


# -- Settings ---------------------------------------------------------------

CHONG_SETTINGS = dict(
    n_patients=3000,
    n_days=90,
    base_noshow_rate=0.193,
    model_auc=0.74,
    targeting_mode="top_fraction",
    targeting_fraction=0.25,
    call_success_rate=0.80,
    target_noshow_rate=0.159,
    n_providers=8,
    slots_per_provider=12,
)

ROSEN_SETTINGS = dict(
    n_patients=5000,
    n_days=180,
    base_noshow_rate=0.36,
    model_auc=0.72,
    targeting_mode="top_k",
    targeting_fraction=0.25,  # unused for top_k
    call_success_rate=0.60,
    target_noshow_rate=0.33,
    n_providers=8,
    slots_per_provider=12,
    race_ethnicity={
        "White": {"prob": 0.55, "noshow_mult": 0.86},
        "Black": {"prob": 0.30, "noshow_mult": 1.17},
        "Hispanic": {"prob": 0.10, "noshow_mult": 1.05},
        "Asian": {"prob": 0.02, "noshow_mult": 0.80},
        "Other": {"prob": 0.03, "noshow_mult": 1.00},
    },
)


def run_simulation(
    settings: Dict,
    reminder_effectiveness: float,
    seed: int = 42,
) -> Tuple[float, float, float, Dict]:
    """Run one simulation, return (factual_rate, cf_rate, auc, race_rates)."""
    n_days = settings["n_days"]
    n_patients = settings["n_patients"]

    cc = CallerConfig(
        n_providers=settings["n_providers"],
        slots_per_provider_per_day=settings["slots_per_provider"],
        call_capacity_per_day=settings.get("call_capacity_per_day", 24),
        call_success_rate=settings["call_success_rate"],
        reminder_effectiveness=reminder_effectiveness,
    )
    tc = TimeConfig(
        n_timesteps=n_days,
        timestep_duration=1 / 365,
        timestep_unit="day",
        prediction_schedule=list(range(n_days)),
    )

    kwargs = dict(
        time_config=tc,
        seed=seed,
        n_patients=n_patients,
        base_noshow_rate=settings["base_noshow_rate"],
        model_type="predictor",
        model_auc=settings["model_auc"],
        targeting_mode=settings["targeting_mode"],
        targeting_fraction=settings.get("targeting_fraction", 0.25),
        caller_config=cc,
    )
    if "race_ethnicity" in settings:
        kwargs["race_ethnicity"] = settings["race_ethnicity"]

    sc = NoShowTargetedReminderScenario(**kwargs)
    results = BranchedSimulationEngine(
        sc, CounterfactualMode.BRANCHED,
    ).run(n_patients)

    # Compute rates
    f_noshows = sum(
        results.outcomes[t].events.sum() for t in range(n_days)
    )
    f_total = sum(
        len(results.outcomes[t].events) for t in range(n_days)
    )
    cf_noshows = sum(
        results.counterfactual_outcomes[t].events.sum()
        for t in range(n_days)
    )
    cf_total = sum(
        len(results.counterfactual_outcomes[t].events)
        for t in range(n_days)
    )

    f_rate = f_noshows / max(f_total, 1)
    cf_rate = cf_noshows / max(cf_total, 1)

    # AUC
    all_pred, all_act = [], []
    for t in range(n_days):
        if t in results.predictions and (t + 1) < n_days:
            p = results.predictions[t].scores
            a = results.outcomes[t + 1].events
            if len(p) == len(a):
                all_pred.append(p)
                all_act.append(a)
    pred_arr = np.concatenate(all_pred) if all_pred else np.array([])
    act_arr = np.concatenate(all_act) if all_act else np.array([])
    achieved_auc = (
        float(auc_score(act_arr, pred_arr))
        if len(pred_arr) > 0 else 0.0
    )

    # Race-level rates
    race_rates = {}
    for t in range(n_days):
        for branch, outcomes in [
            ("factual", results.outcomes[t]),
            ("counterfactual", results.counterfactual_outcomes[t]),
        ]:
            races = outcomes.secondary.get(
                "race_ethnicity", np.array([])
            )
            events = outcomes.events
            for race in np.unique(races):
                mask = races == race
                key = f"{branch}_{race}"
                if key not in race_rates:
                    race_rates[key] = {"noshows": 0.0, "total": 0}
                race_rates[key]["noshows"] += events[mask].sum()
                race_rates[key]["total"] += mask.sum()

    race_rate_pcts = {
        k: v["noshows"] / max(v["total"], 1)
        for k, v in race_rates.items()
    }

    return float(f_rate), float(cf_rate), achieved_auc, race_rate_pcts


def calibrate(
    settings: Dict,
    eff_low: float = 0.05,
    eff_high: float = 0.95,
    tolerance_pp: float = 0.5,
    max_iterations: int = 12,
    seed: int = 42,
) -> float:
    """Binary search for reminder_effectiveness matching target rate."""
    target = settings["target_noshow_rate"]
    print(f"\nCalibrating to target no-show rate: {target:.1%}")
    print(f"Baseline rate: {settings['base_noshow_rate']:.1%}")
    print(f"Search range: [{eff_low:.2f}, {eff_high:.2f}]")
    print(f"Tolerance: {tolerance_pp:.1f} pp")
    print()
    print(f"{'Iter':>4s}  {'Eff':>6s}  {'F-rate':>7s}  "
          f"{'CF-rate':>7s}  {'Abs red':>8s}  {'Gap pp':>7s}  "
          f"{'AUC':>6s}  {'Time':>5s}")
    print("-" * 65)

    best_eff = None
    best_gap = float("inf")

    for i in range(max_iterations):
        eff_mid = (eff_low + eff_high) / 2
        t0 = time.time()
        f_rate, cf_rate, auc, _ = run_simulation(
            settings, eff_mid, seed=seed,
        )
        elapsed = time.time() - t0

        gap_pp = (f_rate - target) * 100
        abs_red_pp = (cf_rate - f_rate) * 100

        print(f"{i+1:>4d}  {eff_mid:>6.3f}  {f_rate:>6.1%}  "
              f"{cf_rate:>6.1%}  {abs_red_pp:>7.1f}pp  "
              f"{gap_pp:>+6.1f}pp  {auc:>6.3f}  {elapsed:>4.0f}s")

        if abs(gap_pp) < tolerance_pp:
            print(f"\n  Converged at effectiveness={eff_mid:.4f}")
            return eff_mid

        if abs(gap_pp) < best_gap:
            best_gap = abs(gap_pp)
            best_eff = eff_mid

        # If factual rate too high, need more effectiveness
        if f_rate > target:
            eff_low = eff_mid
        else:
            eff_high = eff_mid

    print(f"\n  Best after {max_iterations} iterations: "
          f"effectiveness={best_eff:.4f} (gap={best_gap:.1f}pp)")
    return best_eff


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate reminder effectiveness",
    )
    parser.add_argument(
        "--setting", type=str, required=True,
        choices=["chong", "rosen"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tolerance", type=float, default=0.5,
        help="Tolerance in percentage points",
    )
    parser.add_argument(
        "--capacity", type=int, default=None,
        help="Override call_capacity_per_day",
    )
    args = parser.parse_args()

    if args.setting == "chong":
        settings = dict(CHONG_SETTINGS)
    else:
        settings = dict(ROSEN_SETTINGS)

    if args.capacity is not None:
        settings["call_capacity_per_day"] = args.capacity

    t0 = time.time()
    best_eff = calibrate(
        settings,
        tolerance_pp=args.tolerance,
        seed=args.seed,
    )
    total = time.time() - t0

    print(f"\n{'='*65}")
    print(f"CALIBRATION RESULT ({args.setting.upper()})")
    print(f"{'='*65}")
    print(f"  reminder_effectiveness: {best_eff:.4f}")
    print(f"  Total time: {total:.0f}s")

    # Final validation run with calibrated value
    print(f"\nFinal validation run...")
    f_rate, cf_rate, auc, race_rates = run_simulation(
        settings, best_eff, seed=args.seed,
    )
    print(f"  Factual no-show rate:        {f_rate:.1%}")
    print(f"  Counterfactual no-show rate:  {cf_rate:.1%}")
    print(f"  Absolute reduction:           "
          f"{(cf_rate - f_rate)*100:.1f} pp")
    print(f"  Relative reduction:           "
          f"{(cf_rate - f_rate)/cf_rate*100:.1f}%")
    print(f"  Achieved AUC:                 {auc:.3f}")

    if args.setting == "rosen":
        print(f"\n  Race-level rates:")
        for race in ["White", "Black", "Hispanic"]:
            f_key = f"factual_{race}"
            cf_key = f"counterfactual_{race}"
            if f_key in race_rates and cf_key in race_rates:
                f_r = race_rates[f_key]
                cf_r = race_rates[cf_key]
                print(f"    {race:>10s}: factual={f_r:.1%}, "
                      f"CF={cf_r:.1%}, "
                      f"reduction={(cf_r - f_r)*100:.1f}pp")


if __name__ == "__main__":
    main()
