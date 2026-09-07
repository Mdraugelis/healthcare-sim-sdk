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
import shutil
from typing import Any, Dict

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from ...experiments.lifecycle import finalize_experiment
from .run_experiment import _build_cells
from .review import create_packet, run_reviewed_cell
from .runner_common import (
    config_from_dict,
    reduce_patient_arrays,
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

    packet = create_packet(out_root / "review",
                           OmegaConf.to_container(cfg, resolve=True), cells, seeds)
    v: Dict[str, Any] = {"checks": [], "all_pass": True}

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

        def write_arrays(seed, res):
            relative = f"cell_{i:02d}_seed_{seed:04d}.npz"
            destination = packet.output_dir / relative
            np.savez_compressed(destination, **reduce_patient_arrays(res))
            shutil.copyfile(destination, cell_dir / f"seed_{seed:04d}.npz")
            return [relative]

        rows = run_reviewed_cell(packet, f"cell_{i:02d}", scen, horizon,
                                 seeds, result_writer=write_arrays)
        for seed in seeds:
            v["checks"].extend(packet.runs[(f"cell_{i:02d}", seed)]["checks"])
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

    summary_data = {
        "experiment": name,
        "campus": campus,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "horizon_days": horizon,
        "cells": [c["label"] for c in cells],
        "cell_overrides": [c["overrides"] for c in cells],
        "summaries": cell_summaries,
        "verification": v,
    }
    (out_root / "summary.json").write_text(json.dumps(
        summary_data, indent=2, allow_nan=False,
    ))
    (packet.output_dir / "queue_summary.json").write_text(json.dumps(
        summary_data, indent=2, allow_nan=False,
    ))
    packet.add_artifact("queue_summary.json")
    packet.write()
    print(f"wrote {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
