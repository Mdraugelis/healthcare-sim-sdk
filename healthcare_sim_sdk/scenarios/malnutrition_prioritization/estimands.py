"""Estimands and D-table emitters for the LOS & readmission
experiment (exp7 overlay stage).

Consumes the queue-stage npz arrays and emits estimands.json plus the
dossier deliverables D1 / D1-abs / D3 (markdown). Sign convention
everywhere: counterfactual - factual, benefit positive (stated in
every caption). Shares are suppressed ("--") when
|ATE_total| < 2x seed-SE.

Estimands (per efficacy cell, per seed, aggregated over seeds):
  1   ATE on the malnourished-at-discharge window cohort (the D1
      grids come from outcome_overlay; re-derived here per stratum).
  2   Rescued LATE: per-patient paired contrast (exact under the
      shared creation-time coins) among assessed-in-F-only patients.
  2b  Displaced contrast: assessed-in-C-only patients (the harm side
      of reordering; C - F is negative when F hurts them).
  3   Naive early-vs-late observational contrast inside the factual
      arm -- the biased analysis a pilot must avoid; admission-clock
      and oracle onset-clock variants.
  4   GLIM-observed (coded) cohort ascertainment bias: outcome rates
      on the coded cohort vs the true malnourished cohort.

Usage (from repo root):
  python -m healthcare_sim_sdk.scenarios.malnutrition_prioritization\
.estimands --root outputs/malnutrition/exp7_los_readmit_queue/example
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .outcome_overlay import (
    GRID_DETECT_LOS,
    GRID_DETECT_REL,
    GRID_TIMING_LOS,
    GRID_TIMING_REL,
    WINDOW,
    _load,
    overlay_arm,
)
from .scenario import MalnutritionConfig

# D3 literature-central cell (pinned; dossier 06/07).
CENTRAL = {"eff_detect_rel": 0.20, "eff_timing_rel": 0.01,
           "eff_detect_los": 0.5, "eff_timing_los": 0.05}
PER90 = 90.0 / (WINDOW[1] - WINDOW[0])
NAIVE_EARLY_CUTOFF_DAYS = 2.0


def _cohort(d: Dict[str, np.ndarray]) -> np.ndarray:
    lo, hi = WINDOW
    return (d["mal_at_discharge_f"]
            & (d["discharge_day_f"] >= lo)
            & (d["discharge_day_f"] < hi))


def seed_estimands(d: Dict[str, np.ndarray],
                   cfg: MalnutritionConfig) -> Dict[str, Any]:
    """All estimands for one seed at the literature-central cell."""
    ed_r, et_r = CENTRAL["eff_detect_rel"], CENTRAL["eff_timing_rel"]
    ed_l, et_l = CENTRAL["eff_detect_los"], CENTRAL["eff_timing_los"]
    coh = _cohort(d)
    rf, lf, _ = overlay_arm(d, "f", ed_r, et_r, ed_l, et_l, cfg)
    rc, lc, _ = overlay_arm(d, "c", ed_r, et_r, ed_l, et_l, cfg)
    dr = rc.astype(float) - rf.astype(float)   # per-patient, paired
    dl = lc - lf

    a_f, a_c = d["assessed_f"], d["assessed_c"]
    both = coh & a_f & a_c
    rescued = coh & a_f & ~a_c
    displaced = coh & ~a_f & a_c

    def _mean(x, m):
        return float(x[m].mean()) if m.any() else float("nan")

    out: Dict[str, Any] = {
        # 1. ATE (cohort) -- matches the overlay grid cell.
        "ate_readmit_pp": 100 * float(dr[coh].mean()),
        "ate_los_days": float(dl[coh].mean()),
        # 2. Rescued LATE (paired, exact under shared coins).
        "n_rescued": int(rescued.sum()),
        "late_rescued_readmit_pp": 100 * _mean(dr, rescued),
        "late_rescued_los_days": _mean(dl, rescued),
        # 2b. Displaced (C - F < 0 means the reordering hurt them).
        "n_displaced": int(displaced.sum()),
        "late_displaced_readmit_pp": 100 * _mean(dr, displaced),
        "late_displaced_los_days": _mean(dl, displaced),
        # Timing margin inside the both-stratum.
        "n_both": int(both.sum()),
        "both_readmit_pp": 100 * _mean(dr, both),
        "days_earlier_mean": _mean(
            (d["assess_day_c"] - d["assess_day_f"]).astype(float),
            both),
    }

    # 3. Naive early-vs-late inside the FACTUAL arm (the biased
    #    pilot analysis): late-minus-early readmission among
    #    assessed cohort patients; positive reads as "early
    #    assessment helps" but is confounded by who gets seen early.
    ass = coh & a_f
    admit = d["admit_day"].astype(float)
    onset = d["onset_day"].astype(float)
    tta_adm = d["assess_day_f"].astype(float) - np.maximum(admit, 0.0)
    tta_onset = (d["assess_day_f"].astype(float)
                 - np.maximum(onset, admit))
    for tag, tta in (("admission", tta_adm), ("oracle", tta_onset)):
        early = ass & (tta <= NAIVE_EARLY_CUTOFF_DAYS)
        late = ass & (tta > NAIVE_EARLY_CUTOFF_DAYS)
        out[f"naive_{tag}_readmit_pp"] = 100 * (
            _mean(rf.astype(float), late)
            - _mean(rf.astype(float), early))
        out[f"naive_{tag}_los_days"] = (
            _mean(lf, late) - _mean(lf, early))
        out[f"naive_{tag}_n_early"] = int(early.sum())

    # 4. Ascertainment bias: outcome rates on the CODED (observed)
    #    cohort vs the true malnourished cohort, factual arm.
    lo, hi = WINDOW
    disch_w = (d["discharge_day_f"] >= lo) & (d["discharge_day_f"] < hi)
    coded = disch_w & d["coded_f"]
    out["obs_coded_readmit_rate"] = _mean(rf.astype(float), coded)
    out["true_mal_readmit_rate"] = _mean(rf.astype(float), coh)
    out["obs_coded_los_mean"] = _mean(lf, coded)
    out["true_mal_los_mean"] = _mean(lf, coh)
    out["n_coded"] = int(coded.sum())
    return out


def _agg_scalar(rows: List[Dict[str, Any]], k: str) -> Dict[str, Any]:
    vals = np.array([r[k] for r in rows], dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return {"mean": None}
    return {"mean": float(vals.mean()),
            "ci_lo": float(np.percentile(vals, 2.5)),
            "ci_hi": float(np.percentile(vals, 97.5)),
            "se": float(vals.std(ddof=1) / np.sqrt(len(vals)))}


# -- D-table emitters -------------------------------------------------

def _share_row(total: Dict[str, Any], det: float, tim: float
               ) -> str:
    """detect%/timing%/interaction% or -- under the 2xSE rule."""
    t = total["mean"]
    if abs(t) < 2 * total["se"] or t == 0:
        return "--"
    inter = t - det - tim
    return (f"{100*det/t:.0f}%/{100*tim/t:.0f}%/"
            f"{100*inter/t:+.0f}%")


def emit_d1(cell: Dict[str, Any], endpoint: str,
            rows_grid, cols_grid, unit: str, fmt: str) -> str:
    """D1 channel-decomposition matrix for one endpoint."""
    g = cell[endpoint]
    lines = [
        "| detect \\ timing | " + " | ".join(
            str(c) for c in cols_grid) + " |",
        "|" + "---|" * (len(cols_grid) + 1),
    ]
    for r in rows_grid:
        cells = []
        det = g[f"{r}|0.0"]["mean"]
        for c in cols_grid:
            tot = g[f"{r}|{c}"]
            tim = g[f"0.0|{c}"]["mean"]
            cells.append(
                f"{tot['mean']:{fmt}} [{tot['ci_lo']:{fmt}},"
                f" {tot['ci_hi']:{fmt}}]<br>"
                + _share_row(tot, det, tim))
        lines.append(f"| **{r}** | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        f"*{unit}; counterfactual − factual, benefit positive; "
        "cell = total ATE mean [95% CI] over seeds, then "
        "detect%/timing%/interaction% shares (-- when "
        "|ATE| < 2× seed-SE).*")
    return "\n".join(lines)


def emit_d1_abs(cell: Dict[str, Any], est: Dict[str, Any]) -> str:
    s = cell["strata_mean"]
    scale = PER90
    lines = [
        "| quantity | per 90 days (generic default site) |",
        "|---|---|",
        f"| malnourished discharges (window) | "
        f"{s['n_cohort']*scale:.0f} |",
        f"| assessed in both arms | {s['both']*scale:.0f} |",
        f"| **rescued** (assessed only with ML) | "
        f"{s['rescued']*scale:.0f} |",
        f"| displaced (lost assessment under ML) | "
        f"{s['displaced']*scale:.0f} |",
        f"| never assessed in either arm | {s['never']*scale:.0f} |",
        f"| mean days earlier (both-stratum) | "
        f"{est['days_earlier_mean']['mean']:.2f} d |",
        f"| rescued LATE, readmission | "
        f"{est['late_rescued_readmit_pp']['mean']:+.2f} pp |",
        f"| rescued LATE, LOS | "
        f"{est['late_rescued_los_days']['mean']:+.3f} d |",
        f"| displaced contrast, readmission (C−F) | "
        f"{est['late_displaced_readmit_pp']['mean']:+.2f} pp |",
        "",
        "*Counts scaled ×90/150 from the [30, 180) reporting window; "
        "campus rows are produced only from local (gitignored) "
        "configs at the literature-central cell.*",
    ]
    return "\n".join(lines)


def emit_d3(cell: Dict[str, Any]) -> str:
    ed, et = CENTRAL["eff_detect_rel"], CENTRAL["eff_timing_rel"]
    g = cell["readmit_ate_pp"]
    tot = g[f"{ed}|{et}"]
    det = g[f"{ed}|0.0"]["mean"]
    l_tot = cell["los_ate_days"][
        f"{CENTRAL['eff_detect_los']}|{CENTRAL['eff_timing_los']}"]
    share = (f"{100*det/tot['mean']:.0f}%"
             if abs(tot["mean"]) >= 2 * tot["se"] else
             "an indeterminate share (ATE below 2x seed-SE)")
    return (
        f"At the literature-central cell (detection {ed:.0%} "
        f"relative readmission / {CENTRAL['eff_detect_los']} d LOS; "
        f"timing {et:.0%}/day / {CENTRAL['eff_timing_los']} d/day), "
        "the simulation suggests, under these assumptions, a "
        f"malnourished-cohort readmission ATE of "
        f"{tot['mean']:+.2f} pp [{tot['ci_lo']:+.2f}, "
        f"{tot['ci_hi']:+.2f}] with {share} attributable to the "
        "detection channel, and an LOS ATE of "
        f"{l_tot['mean']:+.3f} d [{l_tot['ci_lo']:+.3f}, "
        f"{l_tot['ci_hi']:+.3f}].\n\n"
        "*Pre-registration smell-check: a single headline cell "
        "invites anchoring; the full D1 grid always travels with "
        "this sentence.*")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--cell", default=None,
                    help="cell label for the D tables (default: "
                         "capacity 30 / prevalence 0.30)")
    args = ap.parse_args()
    root = Path(args.root)
    cfg = MalnutritionConfig()

    overlay = json.loads((root / "overlay_results.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    label = args.cell or "rdn_assessments_per_day=30,true_prevalence=0.3"
    idx = summary["cells"].index(label)
    cell_dir = root / f"cell_{idx:02d}"

    rows = []
    for npz in sorted(cell_dir.glob("seed_*.npz")):
        rows.append(seed_estimands(_load(npz), cfg))
    est = {k: _agg_scalar(rows, k) for k in rows[0]}

    cell = overlay["cells"][label]
    md = [
        f"# LOS & readmission deliverables -- cell {label}",
        "",
        "Simulation-under-assumptions; counterfactual − factual, "
        "benefit positive.",
        "",
        "## D1 -- readmission (pp)",
        emit_d1(cell, "readmit_ate_pp",
                GRID_DETECT_REL, GRID_TIMING_REL,
                "percentage points", "+.2f"),
        "",
        "## D1 -- LOS (days)",
        emit_d1(cell, "los_ate_days",
                GRID_DETECT_LOS, GRID_TIMING_LOS, "days", "+.3f"),
        "",
        "## D1-abs -- absolute counts and per-stratum effects",
        emit_d1_abs(cell, est),
        "",
        "## D3 -- literature-central headline",
        emit_d3(cell),
        "",
        "## Estimand 3 -- naive early-vs-late (the biased analysis)",
        f"Naive late−early readmission contrast, admission clock: "
        f"{est['naive_admission_readmit_pp']['mean']:+.2f} pp; "
        f"oracle onset clock: "
        f"{est['naive_oracle_readmit_pp']['mean']:+.2f} pp. "
        f"True cohort ATE: {est['ate_readmit_pp']['mean']:+.2f} pp. "
        "The gap is confounding (who gets seen early), not effect.",
        "",
        "## Estimand 4 -- ascertainment bias (coded vs true cohort)",
        f"Observed (coded) readmission "
        f"{est['obs_coded_readmit_rate']['mean']:.3f} vs true "
        f"malnourished {est['true_mal_readmit_rate']['mean']:.3f}; "
        f"observed LOS {est['obs_coded_los_mean']['mean']:.2f} d vs "
        f"true {est['true_mal_los_mean']['mean']:.2f} d.",
    ]
    (root / "estimands.json").write_text(json.dumps(
        {"cell": label, "central": CENTRAL, "estimands": est},
        indent=2))
    (root / "d_tables.md").write_text("\n".join(md))
    print(f"wrote {root / 'estimands.json'} and "
          f"{root / 'd_tables.md'}")


if __name__ == "__main__":
    main()
