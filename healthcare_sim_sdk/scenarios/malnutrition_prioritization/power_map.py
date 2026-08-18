"""Power map (Table D2) for the LOS & readmission pilot designs.

Designs compared at a 90-day pilot horizon:
  - ITS (I=1): a single site CANNOT identify a stepped-wedge -- the
    treatment step is collinear with calendar time. Reported as an
    interrupted-time-series APPROXIMATION with an explicit secular-
    trend caveat; never silently labeled stepped-wedge.
  - SW I=2/3/6: Hussey & Hughes (2007) closed-form variance for a
    standard stepped wedge (T = I+1 periods, one cluster crossing
    per step), cluster-period mean model.
  - Tier-2 embedded RCT: among ML-flagged-but-unworked patients
    (the ml_flag_day pool), randomize surfacing. Pool size and
    control-arm fates are MEASURED from the queue stage; the pool's
    complier decomposition (detection vs timing-only) quantifies the
    critique that a persistent list mostly randomizes TIMING.

Effect sizes come from the overlay D1 grids (timing pinned 0 across
rows, captioned). Individual-level variance: p(1-p) for readmission,
measured LOS variance for LOS. ICC swept {0.01, 0.03, 0.05}; the D2
markdown shows ICC=0.03, the JSON carries all three.
alpha=0.05 two-sided; PASS = power >= 0.80.

Usage (from repo root):
  python -m healthcare_sim_sdk.scenarios.malnutrition_prioritization\
.power_map --root outputs/malnutrition/exp7_los_readmit_queue/example
"""

import argparse
import json
from math import erf, sqrt
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .outcome_overlay import (
    GRID_DETECT_LOS,
    GRID_DETECT_REL,
    WINDOW,
    _load,
    overlay_arm,
)
from .scenario import MalnutritionConfig

ALPHA_Z = 1.959963984540054          # z_{1 - 0.05/2}
ICC_GRID = (0.01, 0.03, 0.05)
ICC_SHOWN = 0.03
PILOT_DAYS = 90.0
PER90 = PILOT_DAYS / (WINDOW[1] - WINDOW[0])
SW_SITES = (2, 3, 6)
CENTRAL_TIMING_REL = 0.0             # D2 rows sweep detect, timing=0


def _phi(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def power_from_var(theta: float, var: float) -> float:
    if var <= 0 or theta == 0:
        return 0.0
    return _phi(abs(theta) / sqrt(var) - ALPHA_Z)


def hussey_hughes_var(i_sites: int, m: float, sigma2_ind: float,
                      icc: float) -> float:
    """H&H (2007) variance of the treatment effect for a standard
    stepped wedge: T = I+1 periods, cluster i treated from period
    i+1 on. sigma_e^2 = sigma2_ind / m (cluster-period mean)."""
    t_per = i_sites + 1
    x = np.zeros((i_sites, t_per))
    for i in range(i_sites):
        x[i, i + 1:] = 1.0
    u = x.sum()
    w = (x.sum(axis=0) ** 2).sum()
    v = (x.sum(axis=1) ** 2).sum()
    sigma_e2 = sigma2_ind / m
    tau2 = icc * sigma2_ind / (1.0 - icc)
    num = i_sites * sigma_e2 * (sigma_e2 + t_per * tau2)
    den = ((i_sites * u - w) * sigma_e2
           + (u ** 2 + i_sites * t_per * u - t_per * w
              - i_sites * v) * tau2)
    return float(num / den) if den > 0 else float("inf")


def its_var(m_total: float, sigma2_ind: float) -> float:
    """I=1 before/after approximation: two means of m_total/2 each.
    NOT a stepped wedge -- secular trend is not separable."""
    half = max(m_total / 2.0, 1.0)
    return 2.0 * sigma2_ind / half


def tier2_stats(npz_paths: List[Path],
                cfg: MalnutritionConfig,
                eff_detect_rel: float) -> Dict[str, Any]:
    """Measured Tier-2 pool: size, complier split, and the paired
    readmission contrast among pool members at the given detection
    efficacy (timing 0)."""
    pool_n, det_comp, tim_comp, contrasts, p_ctrl = [], [], [], [], []
    lo, hi = WINDOW
    for p in npz_paths:
        d = _load(p)
        coh = (d["mal_at_discharge_f"]
               & (d["discharge_day_f"] >= lo)
               & (d["discharge_day_f"] < hi))
        pool = coh & (d["ml_flag_day"] >= 0)
        if not pool.any():
            continue
        # Complier decomposition (the Tier-2 critique, quantified):
        # detection compliers change ASSESSMENT status; timing-only
        # compliers are assessed in both arms, merely earlier in F.
        det = pool & d["assessed_f"] & ~d["assessed_c"]
        tim = (pool & d["assessed_f"] & d["assessed_c"]
               & (d["assess_day_f"] < d["assess_day_c"]))
        rf, _, _ = overlay_arm(d, "f", eff_detect_rel,
                               CENTRAL_TIMING_REL, 0.0, 0.0, cfg)
        rc, _, _ = overlay_arm(d, "c", eff_detect_rel,
                               CENTRAL_TIMING_REL, 0.0, 0.0, cfg)
        pool_n.append(pool.sum())
        det_comp.append(det.sum() / pool.sum())
        tim_comp.append(tim.sum() / pool.sum())
        contrasts.append(float(
            rc[pool].astype(float).mean()
            - rf[pool].astype(float).mean()))
        p_ctrl.append(float(rc[pool].astype(float).mean()))
    n90 = float(np.mean(pool_n)) * PER90
    delta = float(np.mean(contrasts))
    p0 = float(np.mean(p_ctrl))
    # Two-proportion power, n90/2 per randomized arm.
    n_arm = max(n90 / 2.0, 1.0)
    var = (p0 * (1 - p0) + (p0 - delta) * (1 - p0 + delta)) / n_arm
    return {
        "pool_per_90d": n90,
        "detection_complier_frac": float(np.mean(det_comp)),
        "timing_only_complier_frac": float(np.mean(tim_comp)),
        "paired_contrast_pp": 100 * delta,
        "control_rate": p0,
        "power": power_from_var(delta, var),
    }


def build_map(root: Path, label: str,
              cfg: MalnutritionConfig) -> Dict[str, Any]:
    overlay = json.loads((root / "overlay_results.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    idx = summary["cells"].index(label)
    cell_dir = root / f"cell_{idx:02d}"
    npz_paths = sorted(cell_dir.glob("seed_*.npz"))
    cell = overlay["cells"][label]

    # Per-site scale: window cohort scaled to the 90-day pilot.
    n90 = cell["strata_mean"]["n_cohort"] * PER90

    # Measured outcome scale parameters (first seeds suffice).
    d0 = _load(npz_paths[0])
    coh0 = (d0["mal_at_discharge_f"]
            & (d0["discharge_day_f"] >= WINDOW[0])
            & (d0["discharge_day_f"] < WINDOW[1]))
    p_readmit = float(d0["readmit_c"][coh0].astype(float).mean())
    sigma2_readmit = p_readmit * (1 - p_readmit)
    sigma2_los = float(np.var(
        d0["los_c"][coh0].astype(float), ddof=1))

    result: Dict[str, Any] = {
        "cell": label,
        "cohort_per_site_per_90d": n90,
        "readmit_control_rate": p_readmit,
        "los_variance": sigma2_los,
        "icc_grid": list(ICC_GRID),
        "alpha": 0.05, "pass_threshold": 0.80,
        "endpoints": {},
    }
    for endpoint, grid_key, rows_grid, sigma2 in (
            ("readmission", "readmit_ate_pp", GRID_DETECT_REL,
             sigma2_readmit),
            ("los", "los_ate_days", GRID_DETECT_LOS, sigma2_los)):
        table: Dict[str, Any] = {}
        for eff in rows_grid:
            if eff == 0.0:
                continue
            ate = cell[grid_key][f"{eff}|0.0"]["mean"]
            theta = ate / 100.0 if endpoint == "readmission" else ate
            row: Dict[str, Any] = {"ate": ate}
            row["its_i1"] = {
                "power": power_from_var(theta, its_var(n90, sigma2)),
                "caveat": "ITS approximation -- I=1 cannot identify "
                          "a stepped wedge; secular trend not "
                          "separable",
            }
            for i_sites in SW_SITES:
                per_period = (n90 * i_sites) / (i_sites + 1)
                m = per_period / i_sites
                row[f"sw_i{i_sites}"] = {
                    icc: power_from_var(
                        theta,
                        hussey_hughes_var(i_sites, m, sigma2, icc))
                    for icc in ICC_GRID}
            table[str(eff)] = row
        result["endpoints"][endpoint] = table

    # Tier-2 (readmission only; detection is its estimand).
    result["tier2"] = {
        str(eff): tier2_stats(npz_paths, cfg, eff)
        for eff in GRID_DETECT_REL if eff > 0}
    return result


def emit_d2(res: Dict[str, Any], endpoint: str) -> str:
    tab = res["endpoints"][endpoint]
    hdr = ("| detect eff | ATE | ITS I=1* | SW I=2 | SW I=3 "
           "| SW I=6 |" + (" Tier-2 RCT |" if endpoint ==
                           "readmission" else ""))
    lines = [hdr, "|" + "---|" * (hdr.count("|") - 1)]

    def mark(p: float) -> str:
        return f"**{p:.2f} PASS**" if p >= 0.80 else f"{p:.2f}"

    for eff, row in tab.items():
        cells = [f"{row['ate']:+.2f}",
                 mark(row["its_i1"]["power"]) + "*"]
        for i_sites in SW_SITES:
            cells.append(mark(row[f"sw_i{i_sites}"][ICC_SHOWN]))
        if endpoint == "readmission":
            cells.append(mark(res["tier2"][eff]["power"]))
        lines.append(f"| {eff} | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"*Rows sweep the detection efficacy with timing pinned 0; "
        f"ICC {ICC_SHOWN} shown (grid {list(ICC_GRID)} in JSON); "
        "alpha=0.05 two-sided; PASS = power >= 0.80; 90-day pilot; "
        "counterfactual − factual, benefit positive.*",
        "",
        "*\\*I=1 is an interrupted-time-series APPROXIMATION -- a "
        "single site cannot identify a stepped wedge (treatment "
        "collinear with time); the secular CDI coding trend is not "
        "separable at I=1.*",
    ]
    if endpoint == "readmission":
        t2 = res["tier2"][list(res["tier2"])[-1]]
        lines += [
            "",
            f"*Tier-2 pool: ~{t2['pool_per_90d']:.0f} "
            "flagged-unworked malnourished patients per 90 d; "
            f"detection compliers {t2['detection_complier_frac']:.0%}"
            f", timing-only {t2['timing_only_complier_frac']:.0%} -- "
            "under a persistent list the embedded RCT randomizes "
            "mostly timing, so its detection power is bounded by the "
            "complier fraction.*",
        ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--cell",
                    default="rdn_assessments_per_day=30,"
                            "true_prevalence=0.3")
    args = ap.parse_args()
    root = Path(args.root)
    res = build_map(root, args.cell, MalnutritionConfig())
    (root / "power_map.json").write_text(json.dumps(res, indent=2))
    md = [
        f"# Table D2 -- detectability by design (cell {args.cell})",
        "",
        "## Readmission (pp)",
        emit_d2(res, "readmission"),
        "",
        "## LOS (days)",
        emit_d2(res, "los"),
    ]
    (root / "d2_power.md").write_text("\n".join(md))
    print(f"wrote {root / 'power_map.json'} and "
          f"{root / 'd2_power.md'}")


if __name__ == "__main__":
    main()
