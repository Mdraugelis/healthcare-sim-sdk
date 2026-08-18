"""Experiment lifecycle: the standard post-simulation pipeline.

Every runner calls these functions after simulation completes.
Replaces the ad-hoc save/register/report/validate logic that
was previously duplicated across individual runners.

Usage:
    from healthcare_sim_sdk.experiments.lifecycle import (
        finalize_experiment, register_sweep,
    )

    # After simulation:
    finalize_experiment(output_dir, config, metrics, results)

    # After --multirun sweep:
    summary = register_sweep(sweep_dir)
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from healthcare_sim_sdk.experiments.catalog import ExperimentCatalog

logger = logging.getLogger(__name__)


def save_experiment(
    output_dir: Path,
    config: Dict[str, Any],
    metrics: Dict[str, Any],
    results: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save experiment artifacts to output_dir.

    Writes:
      - config.json (the resolved config)
      - metrics.json (key metrics for catalog/comparison)
      - results.json (optional full results)

    Returns output_dir for chaining.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    if results is not None:
        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

    logger.info("Saved experiment to %s", output_dir)
    return output_dir


def register_experiment(
    output_dir: Path,
    catalog: Optional[ExperimentCatalog] = None,
) -> None:
    """Register a completed experiment in the catalog.

    Reads config.json and metrics.json from output_dir.
    Idempotent — re-registering an existing timestamp updates it.
    """
    output_dir = Path(output_dir)
    config = _load_json(output_dir / "config.json")
    metrics = _load_json(output_dir / "metrics.json")

    if not config:
        logger.warning(
            "No config.json in %s — skipping registration",
            output_dir,
        )
        return

    if catalog is None:
        catalog = ExperimentCatalog()

    catalog.register(output_dir, config, metrics)
    logger.info(
        "Registered %s in catalog",
        config.get("timestamp", output_dir.name),
    )


def register_sweep(
    sweep_dir: Path,
    catalog: Optional[ExperimentCatalog] = None,
) -> Dict[str, Any]:
    """Register all cells of a Hydra --multirun sweep.

    Walks sweep_dir for subdirectories containing metrics.json.
    Registers each cell individually.
    Returns a sweep_summary dict aggregating across cells.
    Writes sweep_summary.json to sweep_dir.
    """
    sweep_dir = Path(sweep_dir)
    if catalog is None:
        catalog = ExperimentCatalog()

    cells = []
    for metrics_path in sorted(sweep_dir.rglob("metrics.json")):
        cell_dir = metrics_path.parent
        # Skip if this is the sweep root itself
        if cell_dir == sweep_dir:
            continue
        register_experiment(cell_dir, catalog)
        metrics = _load_json(metrics_path)
        if metrics:
            cells.append({
                "cell_dir": str(cell_dir.relative_to(sweep_dir)),
                **metrics,
            })

    summary = {
        "sweep_dir": str(sweep_dir),
        "n_cells": len(cells),
        "cells": cells,
    }

    with open(sweep_dir / "sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(
        "Registered %d cells from %s", len(cells), sweep_dir,
    )
    return summary


def finalize_experiment(
    output_dir: Path,
    config: Dict[str, Any],
    metrics: Dict[str, Any],
    results: Optional[Dict[str, Any]] = None,
    notes: str = "",
) -> Path:
    """Full lifecycle: save -> register.

    This is the recommended single call for runners to use
    after simulation completes.
    """
    output_dir = Path(output_dir)

    save_experiment(output_dir, config, metrics, results)
    register_experiment(output_dir)

    return output_dir


def _load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file, returning empty dict if missing."""
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)
