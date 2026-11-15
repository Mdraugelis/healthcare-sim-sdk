"""Every experiment runner, launched the way a person launches it, leaves a
saved, verified and registered record (config/invariants.yaml,
runner_contract). Each runs once with settings small enough for a test.

The static rule (sdk-lint SDK007) checks that a runner *names* Hydra and
finalize_experiment(); this checks that the run actually goes through them.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PKG = "healthcare_sim_sdk.scenarios"

SMALL_CLINIC = ["n_days=5", "n_patients=300"]
RUNNERS = [
    ("noshow_overbooking.run_evaluation", SMALL_CLINIC),
    ("noshow_overbooking.run_burden_analysis", SMALL_CLINIC),
    ("noshow_overbooking.run_governance_eval",
     SMALL_CLINIC + ["model_aucs=[0.8]", "thresholds=[0.3]"]),
    ("noshow_overbooking.run_threshold_optimizer", SMALL_CLINIC),
    ("noshow_targeted_reminders.run_evaluation",
     ["n_days=5", "n_patients=400", "effectiveness_sweep=[0.35]",
      "auc_sweep=[0.74]"]),
    ("noshow_targeted_reminders.run_threshold_sweep",
     ["n_days=5", "n_patients=400", "base_noshow_rates=[0.13]",
      "auc_targets=[0.8]", "thresholds=[0.1,0.2]"]),
    ("nurse_retention.run_evaluation", ["n_weeks=8", "n_nurses=300"]),
    ("sepsis_early_alert.run_replication", ["studies=[trews_replication]"]),
    ("sepsis_early_alert.run_trews_replication", ["n_seeds=2"]),
    ("sepsis_early_alert.run_baseline_sweep",
     ["max_hours_values=[12]", "capacity_values=[25]"]),
    ("sepsis_early_alert.run_capacity_sweep", ["capacities=[25]"]),
    ("paper01_epic_esm.run_evaluation", ["scenario.n_patients=1500"]),
    ("paper03_kaiser_aam.run_evaluation", ["n_patients=1500"]),
    ("paper04_insight_rct.run_evaluation",
     ["n_patients=600", "variance_seeds=2"]),
    ("paper05_composer.run_evaluation", ["n_patients=1500"]),
]


def launch(module, overrides, cwd):
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    return subprocess.run(
        [sys.executable, "-m", f"{PKG}.{module}", *overrides],
        cwd=cwd, env=env, capture_output=True, text=True, timeout=600)


@pytest.mark.parametrize("module, overrides", RUNNERS,
                         ids=[m for m, _ in RUNNERS])
def test_runner_saves_verifies_and_registers(module, overrides, tmp_path):
    done = launch(module, overrides, tmp_path)
    assert done.returncode == 0, done.stderr[-2000:]

    records = sorted((tmp_path / "outputs").rglob("validation.json"))
    assert records, "the run wrote no validation record"
    for record in records:
        run = record.parent
        validation = json.loads(record.read_text())
        assert validation["validated"] is True, validation["checks"]
        assert validation["registered"] is True
        supplied = [c for c in validation["checks"]
                    if c["category"] not in ("structural", "completeness")]
        assert supplied, "no run-specific check was recorded"
        assert json.loads((run / "provenance.json").read_text())
        config = json.loads((run / "config.json").read_text())
        assert "seed" in config
        # Hydra kept its own record of the composed settings for the run.
        assert list(tmp_path.rglob(".hydra/config.yaml"))

    catalog = json.loads((tmp_path / "catalog.json").read_text())
    assert len(catalog) == len(records)
    assert all(entry.get("validated") is True for entry in catalog)


def test_setting_comes_from_yaml_and_override_is_recorded(tmp_path):
    """An override on the command line reaches the saved configuration."""
    done = launch("noshow_overbooking.run_burden_analysis",
                  SMALL_CLINIC + ["seed=7", "max_individual_overbooks=3"],
                  tmp_path)
    assert done.returncode == 0, done.stderr[-2000:]
    (config_file,) = (tmp_path / "outputs").rglob("config.json")
    config = json.loads(config_file.read_text())
    assert (config["seed"], config["max_individual_overbooks"]) == (7, 3)


def test_unknown_setting_is_refused(tmp_path):
    """Hydra rejects a key that is not in the runner's YAML, so a typo
    cannot silently fall back to a default."""
    done = launch("noshow_overbooking.run_evaluation",
                  SMALL_CLINIC + ["n_dayz=3"], tmp_path)
    assert done.returncode != 0
    assert "n_dayz" in done.stderr
    assert not (tmp_path / "catalog.json").exists()


def test_failed_check_is_kept_but_never_registered(tmp_path):
    """A run whose required check fails is saved for diagnosis, marked
    rejected, left out of the catalog, and exits non-zero."""
    done = launch("sepsis_early_alert.run_trews_replication",
                  ["n_seeds=2", "max_reduction_std_pp=0.0"], tmp_path)
    assert done.returncode != 0
    (record,) = (tmp_path / "outputs").rglob("validation.json")
    validation = json.loads(record.read_text())
    assert validation["rejected"] is True
    assert validation["registered"] is False
    assert (record.parent / "metrics.json").exists()
    catalog = tmp_path / "catalog.json"
    assert not catalog.exists() or json.loads(catalog.read_text()) == []
