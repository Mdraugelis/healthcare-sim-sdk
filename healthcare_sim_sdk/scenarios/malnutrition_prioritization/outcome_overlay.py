"""Outcome overlay for the LOS & readmission experiment (exp7).

Re-applies the two-channel effect functions (the SAME module-level
functions step() uses -- single source of truth) to the per-patient
arrays saved by the queue stage, across the full efficacy grid,
without re-simulating. The queue (assessment days, tiers, diagnosis)
is efficacy-invariant by construction; outcomes are deterministic
given the creation-time coins.

Factorization: in the overlay, readmission depends only on
(eff_detect_rel, eff_timing_rel) and LOS only on (eff_detect_los,
eff_timing_los), so the dossier's 288-combo grid collapses to
24 readmission + 12 LOS combos per seed.

Exactness self-test (run on EVERY seed): at efficacy zero the overlay
must reproduce the saved readmission and LOS arrays bit-exactly, per
arm. Any mismatch aborts the overlay.

Disclosed leaks (quantified by the anchor cells, --anchors):
  L1  eff_los > 0 shortens stays, which in a full sim would slightly
      change occupancy and the queue; the overlay freezes the eff=0
      queue.
  L2  readmit0 is evaluated at the eff=0 discharge day; an earlier
      discharge under eff_los > 0 could see less severity progression.
  L3  window membership is fixed at the eff=0 discharge day.

Usage (from repo root):
  python -m healthcare_sim_sdk.scenarios.malnutrition_prioritization\
.outcome_overlay --root outputs/malnutrition/exp7_los_readmit_queue/example
  ... --anchors      # adds the 5 leak-targeted full-sim anchor checks
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .scenario import (
    MalnutritionConfig,
    los_reduction,
    readmission_probability,
    timing_days_early,
)

# Dossier 06 efficacy grid.
GRID_DETECT_REL = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25)
GRID_TIMING_REL = (0.0, 0.01, 0.02, 0.03)
GRID_DETECT_LOS = (0.0, 0.5, 1.0, 1.5)
GRID_TIMING_LOS = (0.0, 0.05, 0.10)

# Reporting window: discharges in [30, 180) sim-days (burn-in
# excluded); counts scale by 90/150 to a per-90-day equivalent.
WINDOW = (30.0, 180.0)


def _load(npz_path: Path) -> Dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _continuous_los(d: Dict[str, np.ndarray], arm: str) -> np.ndarray:
    """Recover the continuous creation-time LOS for discharged
    patients: repeated -1.0 decrements are fp-exact, so
    L = (discharge - admit + 1) + leftover, leftover in (-1, 0]."""
    disch = d[f"discharge_day_{arm}"].astype(np.float64)
    admit = d["admit_day"].astype(np.float64)
    leftover = d[f"los_remaining_{arm}"]
    return (disch - admit + 1.0) + leftover


def overlay_arm(d: Dict[str, np.ndarray], arm: str,
                eff_detect_rel: float, eff_timing_rel: float,
                eff_detect_los: float, eff_timing_los: float,
                cfg: MalnutritionConfig):
    """Recompute (readmit, los, discharge_day) for one arm at the
    given efficacy, mirroring step()'s arithmetic exactly."""
    dx = d[f"diagnosed_{arm}"].astype(np.float64)
    t_dx = d[f"assess_day_{arm}"].astype(np.float64)
    onset = d["onset_day"].astype(np.float64)
    admit = d["admit_day"].astype(np.float64)
    discharged = d[f"status_{arm}"] == 2

    days_early = timing_days_early(
        t_dx, onset, admit, cfg.timing_d_ref_days)

    # LOS channel (diagnosed & discharged only; undiagnosed stays and
    # censored patients keep their saved values).
    los_new = d[f"los_{arm}"].astype(np.float64).copy()
    disch_new = d[f"discharge_day_{arm}"].astype(np.float64).copy()
    m = (dx == 1.0) & discharged
    if m.any() and (eff_detect_los > 0 or eff_timing_los > 0):
        big_l = _continuous_los(d, arm)[m]
        rem_at_dx = big_l - (t_dx[m] - admit[m])
        stay_so_far = t_dx[m] - admit[m]
        guard = np.maximum(cfg.min_stay_days - stay_so_far, 0.0)
        runway = np.maximum(rem_at_dx - guard, 0.0)
        credit = los_reduction(
            1.0, days_early[m], runway,
            eff_detect_los, eff_timing_los)
        rem_after = np.maximum(rem_at_dx - credit, 0.0)
        disch_new[m] = t_dx[m] + np.maximum(
            np.ceil(rem_after), 1.0) - 1.0
        los_new[m] = disch_new[m] - admit[m]

    # Readmission channel (L2: readmit0 frozen at the eff=0
    # discharge-day baseline).
    p = readmission_probability(
        d[f"readmit0_{arm}"], dx, days_early,
        eff_detect_rel, eff_timing_rel)
    readmit_new = np.where(
        discharged, d["u_readmit"] < p, d[f"readmit_{arm}"])
    return readmit_new.astype(bool), los_new, disch_new


def exactness_test(d: Dict[str, np.ndarray],
                   cfg: MalnutritionConfig) -> None:
    """eff=0 edge: overlay must reproduce the saved arrays bit-exactly
    in BOTH arms; and the LOS path must be an exact no-op at
    eff_los=0 even with the readmission knobs on."""
    for arm in ("f", "c"):
        r0, l0, _ = overlay_arm(d, arm, 0.0, 0.0, 0.0, 0.0, cfg)
        if not np.array_equal(r0, d[f"readmit_{arm}"].astype(bool)):
            raise AssertionError(
                f"overlay eff=0 readmission mismatch (arm {arm})")
        if not np.array_equal(l0, d[f"los_{arm}"].astype(np.float64)):
            raise AssertionError(
                f"overlay eff=0 LOS mismatch (arm {arm})")
        _, l1, _ = overlay_arm(d, arm, 0.25, 0.03, 0.0, 0.0, cfg)
        if not np.array_equal(l1, d[f"los_{arm}"].astype(np.float64)):
            raise AssertionError(
                f"overlay eff_los=0 LOS not a no-op (arm {arm})")


def seed_ates(d: Dict[str, np.ndarray],
              cfg: MalnutritionConfig) -> Dict[str, Any]:
    """One seed -> window-cohort strata + ATE grids.

    Cohort: malnourished-at-discharge with the eff=0 discharge day in
    the reporting window (fixed membership, L3). Sign convention:
    counterfactual - factual, benefit positive.
    """
    lo, hi = WINDOW
    in_window_f = ((d["discharge_day_f"] >= lo)
                   & (d["discharge_day_f"] < hi))
    in_window_c = ((d["discharge_day_c"] >= lo)
                   & (d["discharge_day_c"] < hi))
    # At eff=0 the discharge days are arm-identical (verified by the
    # exactness test); use the F copy.
    cohort = d["mal_at_discharge_f"] & in_window_f
    assert np.array_equal(in_window_f, in_window_c)
    n = int(cohort.sum())

    a_f, a_c = d["assessed_f"], d["assessed_c"]
    strata = {
        "n_cohort": n,
        "both": int((cohort & a_f & a_c).sum()),
        "rescued": int((cohort & a_f & ~a_c).sum()),
        "displaced": int((cohort & ~a_f & a_c).sum()),
        "never": int((cohort & ~a_f & ~a_c).sum()),
        "dx_f": int((cohort & d["diagnosed_f"]).sum()),
        "dx_c": int((cohort & d["diagnosed_c"]).sum()),
    }

    readmit_grid: Dict[str, float] = {}
    for ed in GRID_DETECT_REL:
        for et in GRID_TIMING_REL:
            rf, _, _ = overlay_arm(d, "f", ed, et, 0.0, 0.0, cfg)
            rc, _, _ = overlay_arm(d, "c", ed, et, 0.0, 0.0, cfg)
            readmit_grid[f"{ed}|{et}"] = float(
                100 * (rc[cohort].mean() - rf[cohort].mean()))

    los_grid: Dict[str, float] = {}
    for ed in GRID_DETECT_LOS:
        for et in GRID_TIMING_LOS:
            _, lf, _ = overlay_arm(d, "f", 0.0, 0.0, ed, et, cfg)
            _, lc, _ = overlay_arm(d, "c", 0.0, 0.0, ed, et, cfg)
            los_grid[f"{ed}|{et}"] = float(
                lc[cohort].mean() - lf[cohort].mean())

    return {"strata": strata, "readmit_ate_pp": readmit_grid,
            "los_ate_days": los_grid}


def _agg(per_seed: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for combo in per_seed[0][key]:
        vals = np.array([s[key][combo] for s in per_seed])
        out[combo] = {
            "mean": float(vals.mean()),
            "ci_lo": float(np.percentile(vals, 2.5)),
            "ci_hi": float(np.percentile(vals, 97.5)),
            "se": float(vals.std(ddof=1) / np.sqrt(len(vals))),
        }
    return out


def run_overlay(root: Path,
                cfg: MalnutritionConfig) -> Dict[str, Any]:
    summary = json.loads((root / "summary.json").read_text())
    results: Dict[str, Any] = {
        "experiment": summary["experiment"],
        "campus": summary["campus"],
        "window": list(WINDOW),
        "sign_convention": "counterfactual - factual (benefit "
                           "positive); shares suppressed when "
                           "|ATE| < 2x seed-SE",
        "cells": {},
    }
    for i, label in enumerate(summary["cells"]):
        cell_dir = root / f"cell_{i:02d}"
        per_seed = []
        for npz in sorted(cell_dir.glob("seed_*.npz")):
            d = _load(npz)
            exactness_test(d, cfg)
            per_seed.append(seed_ates(d, cfg))
        strata_mean = {
            k: float(np.mean([s["strata"][k] for s in per_seed]))
            for k in per_seed[0]["strata"]}
        results["cells"][label] = {
            "n_seeds": len(per_seed),
            "strata_mean": strata_mean,
            "readmit_ate_pp": _agg(per_seed, "readmit_ate_pp"),
            "los_ate_days": _agg(per_seed, "los_ate_days"),
        }
        print(f"  cell {label}: {len(per_seed)} seeds, "
              f"exactness PASS; cohort ~{strata_mean['n_cohort']:.0f} "
              f"(rescued {strata_mean['rescued']:.0f}, "
              f"displaced {strata_mean['displaced']:.0f})")
    return results


# -- Leak-targeted anchors (full sim vs overlay, paired seeds) --------

ANCHORS = [
    # (label, cell overrides, efficacy, endpoint tolerance floors)
    ("literature-central @30/0.30",
     {"rdn_assessments_per_day": 30, "true_prevalence": 0.30},
     (0.20, 0.01, 0.5, 0.05)),
    ("max-LOS leak @30/0.30",
     {"rdn_assessments_per_day": 30, "true_prevalence": 0.30},
     (0.20, 0.01, 1.5, 0.10)),
    ("pure-readmit (no LOS feedback) @30/0.30",
     {"rdn_assessments_per_day": 30, "true_prevalence": 0.30},
     (0.25, 0.03, 0.0, 0.0)),
    ("congested @20/0.40",
     {"rdn_assessments_per_day": 20, "true_prevalence": 0.40},
     (0.20, 0.01, 0.5, 0.05)),
    ("uncongested @50/0.25",
     {"rdn_assessments_per_day": 50, "true_prevalence": 0.25},
     (0.20, 0.01, 0.5, 0.05)),
]
ANCHOR_SEEDS = 30
FLOOR_READMIT_PP = 0.10
FLOOR_LOS_DAYS = 0.02


def run_anchors(base_cfg_kwargs: Dict[str, Any],
                horizon: int) -> List[Dict[str, Any]]:
    """Run the 5 anchors: full sim at the anchor efficacy vs overlay
    on a fresh eff=0 queue run, SAME seeds (paired). The sim-side ATE
    uses the sim's own outcomes; the tolerance is
    max(seed-CI-halfwidth/4, floor)."""
    from .runner_common import reduce_patient_arrays, run_once

    out = []
    for label, cell, eff in ANCHORS:
        ed_r, et_r, ed_l, et_l = eff
        sim_r, sim_l, ovl_r, ovl_l = [], [], [], []
        for seed in range(1, ANCHOR_SEEDS + 1):
            kw = {**base_cfg_kwargs, **cell}
            cfg_eff = MalnutritionConfig(
                **kw, eff_detect_rel=ed_r, eff_timing_rel=et_r,
                eff_detect_los=ed_l, eff_timing_los=et_l)
            cfg_zero = MalnutritionConfig(
                **kw, eff_detect_rel=0.0, eff_timing_rel=0.0,
                eff_detect_los=0.0, eff_timing_los=0.0)
            _, res = run_once(cfg_eff, horizon, seed)
            H = len(res.outcomes) - 1
            f = res.outcomes[H].secondary
            c = res.counterfactual_outcomes[H].secondary
            lo, hi = WINDOW
            coh_f = ((f["mal_at_discharge"] == 1)
                     & (f["discharge_day"] >= lo)
                     & (f["discharge_day"] < hi))
            coh_c = ((c["mal_at_discharge"] == 1)
                     & (c["discharge_day"] >= lo)
                     & (c["discharge_day"] < hi))
            sim_r.append(100 * (c["readmit"][coh_c].mean()
                                - f["readmit"][coh_f].mean()))
            sim_l.append(float(c["los"][coh_c].mean()
                               - f["los"][coh_f].mean()))
            _, res0 = run_once(cfg_zero, horizon, seed)
            d = reduce_patient_arrays(res0)
            exactness_test(d, cfg_zero)
            cohort = (d["mal_at_discharge_f"]
                      & (d["discharge_day_f"] >= lo)
                      & (d["discharge_day_f"] < hi))
            rf, lf, _ = overlay_arm(d, "f", ed_r, et_r, ed_l, et_l,
                                    cfg_zero)
            rc, lc, _ = overlay_arm(d, "c", ed_r, et_r, ed_l, et_l,
                                    cfg_zero)
            ovl_r.append(100 * (rc[cohort].mean()
                                - rf[cohort].mean()))
            ovl_l.append(float(lc[cohort].mean()
                               - lf[cohort].mean()))
        rec: Dict[str, Any] = {"anchor": label, "efficacy": eff}
        for nm, sim, ovl, floor in (
                ("readmit_pp", sim_r, ovl_r, FLOOR_READMIT_PP),
                ("los_days", sim_l, ovl_l, FLOOR_LOS_DAYS)):
            diff = np.array(sim) - np.array(ovl)
            ci_half = 1.96 * np.array(sim).std(ddof=1) / np.sqrt(
                len(sim))
            tol = max(ci_half / 4, floor)
            rec[nm] = {
                "sim_mean": float(np.mean(sim)),
                "overlay_mean": float(np.mean(ovl)),
                "paired_gap": float(diff.mean()),
                "tolerance": float(tol),
                "pass": bool(abs(diff.mean()) <= tol),
            }
        rec["pass"] = rec["readmit_pp"]["pass"] and \
            rec["los_days"]["pass"]
        status = "PASS" if rec["pass"] else "FAIL"
        print(f"  anchor [{status}] {label}: readmit gap "
              f"{rec['readmit_pp']['paired_gap']:+.3f}pp "
              f"(tol {rec['readmit_pp']['tolerance']:.3f}), LOS gap "
              f"{rec['los_days']['paired_gap']:+.4f}d "
              f"(tol {rec['los_days']['tolerance']:.4f})")
        out.append(rec)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="queue-stage output dir (…/exp7…/<campus>)")
    ap.add_argument("--anchors", action="store_true",
                    help="also run the 5 leak-targeted anchor checks")
    args = ap.parse_args()
    root = Path(args.root)
    cfg = MalnutritionConfig()

    print(f"[outcome_overlay] {root}")
    results = run_overlay(root, cfg)

    if args.anchors:
        print("running leak-targeted anchors "
              f"({ANCHOR_SEEDS} paired seeds each)...")
        summary = json.loads((root / "summary.json").read_text())
        results["anchors"] = run_anchors(
            {}, int(summary.get("horizon_days", 180)))
        results["anchors_all_pass"] = all(
            a["pass"] for a in results["anchors"])

    out = root / "overlay_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
