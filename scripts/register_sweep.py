#!/usr/bin/env python3
"""Register all cells of a completed Hydra --multirun sweep.

Usage:
    python scripts/register_sweep.py outputs/sweep_20260404_120000/
"""

import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from healthcare_sim_sdk.experiments.lifecycle import (  # noqa: E402
    register_sweep,
)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python scripts/register_sweep.py "
            "<sweep_dir>"
        )
        sys.exit(1)

    sweep_dir = Path(sys.argv[1])
    if not sweep_dir.exists():
        print(f"Directory not found: {sweep_dir}")
        sys.exit(1)

    summary = register_sweep(sweep_dir)
    print(
        f"Registered {summary['n_cells']} cells "
        f"from {sweep_dir}"
    )
