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
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from healthcare_sim_sdk.experiments.catalog import ExperimentCatalog
from healthcare_sim_sdk.experiments.provenance import (
    PROVENANCE_FILE, write_provenance,
)

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

    validation = _load_json(output_dir / "validation.json")
    if validation.get("rejected"):
        logger.warning(
            "%s failed required validation — not registered",
            output_dir,
        )
        return

    provenance = _load_json(output_dir / PROVENANCE_FILE)

    if catalog is None:
        catalog = ExperimentCatalog()

    catalog.register(
        output_dir, config, metrics,
        validated=validation.get("validated") if validation else None,
        code_revision=provenance.get("code_revision"),
    )
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
    *,
    validation_checks=None,
    required_checks=(),
    require_validated: bool = False,
) -> Path:
    """Save, record provenance, validate, then register if allowed.

    Saving is not certification. Order of operations:

    1. Save config.json, metrics.json and results.json.
    2. Write provenance.json: code revision, config hash, seeds,
       Python and dependency versions.
    3. Run the generic artifact checks plus the supplied checks and
       write validation.json.
    4. Register in the catalog, with ``validated`` set from the gate.

    A run whose artifacts fail the generic checks is never registered.
    With ``require_validated=True``, a run whose required checks fail or
    did not run is never registered either: its files are kept for
    diagnosis, validation.json is marked ``rejected``, and ValueError is
    raised. Without it, the run is registered as unverified, and only
    ``ExperimentCatalog.list_verified()`` returns verified runs.
    """
    output_dir = Path(output_dir)

    from .validate import (
        Check, CheckStatus, checks_passed, ensure_required_checks,
        validate_generic,
    )
    config = dict(config)
    config.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    save_experiment(output_dir, config, metrics, results)
    write_provenance(output_dir, config)

    artifact_checks = validate_generic(output_dir)
    artifacts_ok = checks_passed(artifact_checks)
    checks = list(artifact_checks)
    supplied_checks = (list(validation_checks)
                       if validation_checks is not None else [])
    if not supplied_checks:
        checks.extend([
            Check("simulation_validation", "No simulation checks supplied",
                  "run-specific checks", None, "declared scope", False,
                  status=CheckStatus.NOT_RUN, category="simulation"),
            Check("analysis_validation", "No analysis checks supplied",
                  "run-specific checks", None, "declared scope", False,
                  status=CheckStatus.NOT_RUN, category="analysis"),
        ])
    else:
        checks.extend(supplied_checks)
    checks = ensure_required_checks(checks, required_checks)
    valid = checks_passed(checks)
    rejected = require_validated and not valid
    will_register = artifacts_ok and not rejected

    (output_dir / "validation.json").write_text(json.dumps({
        "validated": valid,
        "rejected": rejected,
        "registered": will_register,
        "scope": "Only the explicitly supplied checks",
        "checks": [asdict(check) for check in checks],
    }, indent=2, default=str, allow_nan=False))

    if will_register:
        register_experiment(output_dir)
    elif not artifacts_ok:
        logger.warning(
            "%s failed artifact checks — not registered", output_dir,
        )
    if rejected:
        raise ValueError(
            "Experiment validation incomplete or failed; not registered. "
            f"Diagnostic files kept in {output_dir}"
        )

    return output_dir


def _load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file, returning empty dict if missing."""
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)
