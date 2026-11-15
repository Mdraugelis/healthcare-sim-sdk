"""Public and private test profiles (issue #8).

Tests that need institution-private files (gitignored, never in the repo)
carry the ``private`` marker.

- Public profile (CI, default gate): ``pytest -m "not private"``. These tests
  are deselected and reported as such, not counted as skips.
- Private profile: ``SIM_SDK_PRIVATE_TESTS=1 pytest -m private``. The skip is
  removed, so a missing private file fails the run instead of passing quietly.

A plain ``pytest`` run keeps the old behavior: private tests skip when their
files are absent.
"""

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENV = "SIM_SDK_PRIVATE_TESTS"
MALNUTRITION_MODEL = (
    "healthcare_sim_sdk/scenarios/malnutrition_prioritization/configs/"
    "model_frozen.yaml",
    "healthcare_sim_sdk/scenarios/malnutrition_prioritization/configs/"
    "model_targets.yaml",
)


def requires_private_files(*paths: str):
    """Mark a test or class as private; skip only outside the private run."""
    missing = [p for p in paths if not (ROOT / p).exists()]
    marks = [pytest.mark.private]
    if os.environ.get(ENV) != "1":
        marks.append(pytest.mark.skipif(
            bool(missing),
            reason=f"private files absent: {', '.join(missing)}"))

    def apply(obj):
        for mark in marks:
            obj = mark(obj)
        return obj
    return apply
