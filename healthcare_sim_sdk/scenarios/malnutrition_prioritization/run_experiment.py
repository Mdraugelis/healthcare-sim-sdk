"""Hydra-driven experiment runner for the malnutrition prioritization
scenario.

The sweep GRID lives in configs/experiment/*.yaml (repo convention:
grids in YAML, never hardcoded Python loops); seeds are looped inside
each cell per the refined plan. Site facts compose from gitignored
configs/campus/<site>.yaml files (template: campus/example.yaml).

Usage (from repo root):
  python -m healthcare_sim_sdk.scenarios.malnutrition_prioritization\
.run_experiment experiment=exp2_capacity campus=example runner.n_seeds=30

Every experiment emits an Always-On Verification block before any
result, calls finalize_experiment() per sweep cell, and frames all
findings as simulation-under-assumptions.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from ...experiments.lifecycle import finalize_experiment
from .runner_common import (
    SUBGROUP_KEYS,
    always_on_verification,
    config_from_dict,
    format_verification,
    run_cell,
    summarize,
)

SUMMARY_KEYS = [
    "missed_rate_f", "missed_rate_c", "delta_missed_pp",
    "median_tta_f", "median_tta_c", "delta_tta_days",
    "harm_total_f", "harm_total_c", "delta_harm",
    "coded_rate_f", "coded_rate_c", "delta_coded_pp",
    "ml_sourced_assessments", "censored_truepos_frac",
]


def _build_cells(cfg: DictConfig) -> List[Dict[str, Any]]:
    """Expand the YAML sweep block into per-cell scenario overrides."""
    from typing import cast
    base = cast(Dict[str, Any],
                OmegaConf.to_container(cfg.scenario, resolve=True))
    sweep = cast(Dict[str, Any],
                 OmegaConf.to_container(cfg.sweep, resolve=True)) \
        if cfg.sweep is not None else None
    if not sweep:
        return [{"label": "single", "overrides": {}, "config": base}]
    cells = []
    for v in sweep["values"]:
        if isinstance(v, dict):        # e.g. operating_point cells
            overrides = dict(v)
            label = ",".join(f"{k}={val}" for k, val in v.items())
        else:
            overrides = {sweep["param"]: v}
            label = f"{sweep['param']}={v}"
        cells.append({
            "label": label,
            "overrides": overrides,
            "config": {**base, **overrides},
        })
    return cells


def _equity_table(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-subgroup missed-rate deltas (H4): mean over seeds + max gap."""
    table: Dict[str, Any] = {}
    for key in SUBGROUP_KEYS:
        groups = sorted({
            k.split(f"missed_{key}_")[1].rsplit("_", 1)[0]
            for r in rows for k in r
            if k.startswith(f"missed_{key}_") and k.endswith("_f")})
        entries: Dict[str, Any] = {}
        for g in groups:
            fk, ck = f"missed_{key}_{g}_f", f"missed_{key}_{g}_c"
            deltas = [100 * (r[ck] - r[fk]) for r in rows
                      if fk in r and ck in r]
            if len(deltas) < max(3, len(rows) // 2):
                continue          # group too rare to estimate
            entries[g] = {
                "delta_missed_pp_mean": float(np.mean(deltas)),
                "delta_missed_pp_ci": [
                    float(np.percentile(deltas, 2.5)),
                    float(np.percentile(deltas, 97.5))],
                "missed_f_mean": float(np.mean(
                    [r[fk] for r in rows if fk in r])),
                "missed_c_mean": float(np.mean(
                    [r[ck] for r in rows if ck in r])),
            }
        if entries:
            means = [float(e["delta_missed_pp_mean"])
                     for e in entries.values()]
            table[key] = {
                "groups": entries,
                "max_gap_pp": float(max(means) - min(means)),
            }
    return table


def _experiment_analysis(name: str, cells, cell_rows,
                         cell_summaries, cfg) -> Dict[str, Any]:
    """Per-experiment headline analysis."""
    extra: Dict[str, Any] = {}
    if name in ("exp3_operating_point", "exp3_policy"):
        harms = [s["harm_total_f"]["mean"] for s in cell_summaries]
        best = int(np.argmin(harms))
        ref = next(
            (i for i, c in enumerate(cells)
             if "rank_all" in c["label"]), 0)
        extra["harm_optimal_cell"] = cells[best]["label"]
        extra["deployed_reference_cell"] = cells[ref]["label"]
        extra["harm_at_optimum"] = harms[best]
        extra["harm_at_deployed"] = harms[ref]
    elif name == "exp5_erosion":
        decays = [c["overrides"][
            "unflagged_assessment_decay_per_week"] for c in cells]
        deltas = [s["delta_missed_pp"]["mean"]
                  for s in cell_summaries]
        tip = None
        for i in range(1, len(decays)):
            if deltas[i - 1] > 0 >= deltas[i]:
                # Linear interpolation of the zero crossing.
                d0, d1 = decays[i - 1], decays[i]
                y0, y1 = deltas[i - 1], deltas[i]
                tip = d0 + (d1 - d0) * (y0 / (y0 - y1))
                break
        extra["tipping_point_decay_per_week"] = tip
        extra["deltas_by_decay"] = dict(zip(map(str, decays),
                                            deltas))
    elif name == "exp6_coding":
        pct90 = float(cfg.scenario.cohort_benchmark_pct90)
        pct50 = float(cfg.scenario.cohort_benchmark_pct50)
        flags = {}
        for c, s in zip(cells, cell_summaries):
            rate = s["coded_rate_f"]["mean"]
            flags[c["label"]] = {
                "coded_rate_f": rate,
                "coded_rate_c": s["coded_rate_c"]["mean"],
                "above_cohort_pct90": bool(rate > pct90),
                "cohort_pct50": pct50,
                "cohort_pct90": pct90,
            }
        extra["audit_band_position"] = flags
    elif name == "exp4_equity":
        extra["equity"] = _equity_table(cell_rows[0])
    return extra


def _results_md(name, campus, cells, cell_summaries, extra,
                verification, n_seeds) -> str:
    lines = [
        f"# {name} -- campus {campus.upper()}",
        "",
        "All findings are what **the simulation suggests, under "
        "these assumptions** -- not real-world evidence. The model "
        "is frozen at its published operating characteristics; "
        "[O]-tagged workflow inputs (RDN capacity, baseline "
        "time-to-consult, worklist order, true prevalence) are "
        "placeholders pending local data.",
        "",
        format_verification(verification),
        "",
        f"## Results ({n_seeds} seeds per cell, mean [95% CI])",
        "",
        "| cell | missed F | missed C | delta missed (pp) "
        "| delta TTA (d) | delta harm | coded F | coded C |",
        "|" + "---|" * 8,
    ]
    for c, s in zip(cells, cell_summaries):
        def fmt(k, scale=1.0, pct=False):
            if k not in s:
                return "--"
            m = s[k]
            unit = "%" if pct else ""
            return (f"{m['mean']*scale:.1f}{unit} "
                    f"[{m['ci_lo']*scale:.1f}, "
                    f"{m['ci_hi']*scale:.1f}]")
        lines.append(
            f"| {c['label']} "
            f"| {fmt('missed_rate_f', 100, True)} "
            f"| {fmt('missed_rate_c', 100, True)} "
            f"| {fmt('delta_missed_pp')} "
            f"| {fmt('delta_tta_days')} "
            f"| {fmt('delta_harm')} "
            f"| {fmt('coded_rate_f', 100, True)} "
            f"| {fmt('coded_rate_c', 100, True)} |")
    if extra:
        lines += ["", "## Experiment-specific analysis", "",
                  "```json",
                  json.dumps(extra, indent=2), "```"]
    return "\n".join(lines)


@hydra.main(version_base=None, config_path="configs",
            config_name="base")
def main(cfg: DictConfig) -> None:
    name = cfg.runner.experiment_name
    campus = cfg.get("campus_name", "example")
    horizon = int(cfg.runner.horizon_days)
    seeds = list(range(
        int(cfg.runner.seed_start),
        int(cfg.runner.seed_start) + int(cfg.runner.n_seeds)))
    out_root = (Path(cfg.runner.output_root) / name / campus)
    out_root.mkdir(parents=True, exist_ok=True)

    cells = _build_cells(cfg)
    print(f"[{name} / {campus}] {len(cells)} cell(s) x "
          f"{len(seeds)} seeds x {horizon} days")

    # Always-On Verification on the center cell, first seed.
    center = cells[len(cells) // 2]
    v = always_on_verification(
        config_from_dict(center["config"]), horizon, seeds[0])
    print(format_verification(v))
    if not v["all_pass"]:
        print("VERIFICATION FAILED -- aborting before results.")
        raise SystemExit(1)

    cell_rows, cell_summaries = [], []
    for i, cell in enumerate(cells):
        scen_cfg = config_from_dict(cell["config"])
        rows = run_cell(scen_cfg, horizon, seeds)
        cell_rows.append(rows)
        summary = summarize(rows, SUMMARY_KEYS)
        cell_summaries.append(summary)
        print(f"  cell {cell['label']}: "
              f"delta missed "
              f"{summary['delta_missed_pp']['mean']:+.2f} pp "
              f"[{summary['delta_missed_pp']['ci_lo']:+.2f}, "
              f"{summary['delta_missed_pp']['ci_hi']:+.2f}]")
        cell_dir = out_root / f"cell_{i:02d}"
        cell_dir.mkdir(parents=True, exist_ok=True)
        metrics = {k: v_["mean"] for k, v_ in summary.items()}
        metrics["n_seeds"] = len(seeds)
        finalize_experiment(
            cell_dir,
            {"scenario": "malnutrition_prioritization",
             "experiment_name": f"{name}/{campus}/{cell['label']}",
             "seed": seeds[0],
             **cell["config"]},
            metrics)

    extra = _experiment_analysis(
        name, cells, cell_rows, cell_summaries, cfg)

    md = _results_md(name, campus, cells, cell_summaries, extra,
                     v, len(seeds))
    (out_root / "results.md").write_text(md)
    (out_root / "summary.json").write_text(json.dumps({
        "experiment": name,
        "campus": campus,
        "n_seeds": len(seeds),
        "cells": [c["label"] for c in cells],
        "summaries": cell_summaries,
        "analysis": extra,
        "verification": v,
    }, indent=2, default=float))
    print(f"\nwrote {out_root / 'results.md'}")


if __name__ == "__main__":
    main()
