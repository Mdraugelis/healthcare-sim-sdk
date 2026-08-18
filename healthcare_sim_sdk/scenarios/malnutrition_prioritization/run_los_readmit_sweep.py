"""Queue stage for the LOS & readmission experiment (exp7).

Simulates the three-path worklist ONCE per (capacity, prevalence)
cell at efficacy ZERO -- the queue (who is assessed when, in which
arm, via which tier) does not depend on the efficacy knobs -- and
saves per-patient arrays (npz per seed) for `outcome_overlay.py`,
which re-applies the two-channel effect functions across the full
efficacy grid as vectorized post-processing.

Disclosed approximation: at eff_los > 0 the true sim's LOS reductions
would slightly change occupancy and hence the queue; the overlay
freezes the eff=0 queue. Leak-targeted anchor cells (full sim vs
overlay) quantify this.

Usage (from repo root):
  python -m healthcare_sim_sdk.scenarios.malnutrition_prioritization\
.run_los_readmit_sweep experiment=exp7_los_readmit_queue campus=example
"""

import json
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from ...experiments.lifecycle import finalize_experiment
from .run_experiment import _build_cells
from .runner_common import (
    always_on_verification,
    config_from_dict,
    format_verification,
    reduce_patient_arrays,
    reduce_results,
    run_once,
    summarize,
)

QUEUE_SUMMARY_KEYS = [
    "missed_rate_f", "missed_rate_c", "delta_missed_pp",
    "median_tta_f", "median_tta_c", "delta_tta_days",
    "censored_truepos_frac",
]


@hydra.main(version_base=None, config_path="configs",
            config_name="base")
def main(cfg: DictConfig) -> None:
    name = cfg.runner.experiment_name
    campus = cfg.get("campus_name", "example")
    horizon = int(cfg.runner.horizon_days)
    seeds = list(range(
        int(cfg.runner.seed_start),
        int(cfg.runner.seed_start) + int(cfg.runner.n_seeds)))
    out_root = Path(cfg.runner.output_root) / name / campus
    out_root.mkdir(parents=True, exist_ok=True)

    cells = _build_cells(cfg)
    print(f"[{name} / {campus}] queue stage: {len(cells)} cell(s) x "
          f"{len(seeds)} seeds x {horizon} days")

    # Always-On Verification on the center cell, first seed.
    center = config_from_dict(cells[len(cells) // 2]["config"])
    v = always_on_verification(center, horizon, seeds[0])
    print(format_verification(v))
    if not v["all_pass"]:
        print("VERIFICATION FAILED -- aborting before results.")
        raise SystemExit(1)

    cell_summaries = []
    for i, cell in enumerate(cells):
        scen = config_from_dict(cell["config"])
        # Hard guard: the queue stage is only valid at efficacy zero.
        assert (scen.eff_detect_rel == 0.0
                and scen.eff_timing_rel == 0.0
                and scen.eff_detect_los == 0.0
                and scen.eff_timing_los == 0.0), (
            "exp7 queue stage must run with all efficacy knobs at 0; "
            "efficacy is applied by outcome_overlay.py")
        cell_dir = out_root / f"cell_{i:02d}"
        cell_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for seed in seeds:
            _, res = run_once(scen, horizon, seed)
            arrays = reduce_patient_arrays(res)
            np.savez_compressed(
                cell_dir / f"seed_{seed:04d}.npz",
                **arrays)  # type: ignore[arg-type]
            row = reduce_results(res)   # release res immediately
            row["seed"] = seed
            rows.append(row)
        summary = summarize(rows, QUEUE_SUMMARY_KEYS)
        cell_summaries.append(summary)
        print(f"  cell {cell['label']}: queue saved "
              f"({len(seeds)} seeds); missed F "
              f"{summary['missed_rate_f']['mean']:.3f} vs C "
              f"{summary['missed_rate_c']['mean']:.3f}")
        metrics = {k: m["mean"] for k, m in summary.items()}
        metrics["n_seeds"] = len(seeds)
        finalize_experiment(
            cell_dir,
            {"scenario": "malnutrition_prioritization",
             "experiment_name": f"{name}/{campus}/{cell['label']}",
             "seed": seeds[0],
             **cell["config"]},
            metrics)

    (out_root / "summary.json").write_text(json.dumps({
        "experiment": name,
        "campus": campus,
        "n_seeds": len(seeds),
        "horizon_days": horizon,
        "cells": [c["label"] for c in cells],
        "cell_overrides": [c["overrides"] for c in cells],
        "summaries": cell_summaries,
        "verification": v,
    }, indent=2))
    print(f"wrote {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
