"""Unit tests for experiment lifecycle module."""

import json

from healthcare_sim_sdk.experiments.lifecycle import (
    finalize_experiment,
    register_experiment,
    register_sweep,
    save_experiment,
)
from healthcare_sim_sdk.experiments.catalog import ExperimentCatalog


def test_save_experiment_creates_expected_files(tmp_path):
    """save_experiment writes config.json and metrics.json."""
    out = tmp_path / "run_001"
    config = {"timestamp": "20260404_001", "seed": 42}
    metrics = {"utilization": 0.90, "collision_rate": 0.30}

    save_experiment(out, config, metrics)

    assert (out / "config.json").exists()
    assert (out / "metrics.json").exists()
    assert not (out / "results.json").exists()

    with open(out / "config.json") as f:
        assert json.load(f)["seed"] == 42
    with open(out / "metrics.json") as f:
        assert json.load(f)["utilization"] == 0.90


def test_save_experiment_with_results(tmp_path):
    """save_experiment writes results.json when provided."""
    out = tmp_path / "run_002"
    save_experiment(
        out,
        {"timestamp": "t1"},
        {"auc": 0.85},
        results={"data": [1, 2, 3]},
    )
    assert (out / "results.json").exists()


def test_register_experiment_appears_in_catalog(tmp_path):
    """Round-trip: save then register, verify in catalog."""
    out = tmp_path / "run_003"
    catalog_path = tmp_path / "catalog.json"
    config = {
        "timestamp": "20260404_003",
        "experiment_name": "test",
        "scenario": "noshow",
        "seed": 42,
    }
    metrics = {"utilization": 0.88}

    save_experiment(out, config, metrics)

    catalog = ExperimentCatalog(catalog_path)
    register_experiment(out, catalog)

    found = catalog.find("003")
    assert found is not None
    assert found["scenario"] == "noshow"
    assert found["metrics"]["utilization"] == 0.88


def test_register_is_idempotent(tmp_path):
    """Calling register twice doesn't duplicate."""
    out = tmp_path / "run_004"
    catalog_path = tmp_path / "catalog.json"
    config = {"timestamp": "20260404_004"}
    metrics = {"auc": 0.80}

    save_experiment(out, config, metrics)

    catalog = ExperimentCatalog(catalog_path)
    register_experiment(out, catalog)
    register_experiment(out, catalog)

    entries = catalog.list_experiments()
    assert len(entries) == 1


def test_register_sweep_finds_all_cells(tmp_path):
    """register_sweep walks subdirectories for metrics.json."""
    sweep_dir = tmp_path / "sweep_001"

    # Create 3 mock sweep cells
    for i in range(3):
        cell_dir = sweep_dir / f"cell_{i}"
        cell_dir.mkdir(parents=True)
        config = {
            "timestamp": f"20260404_cell{i}",
            "scenario": "test",
        }
        metrics = {"utilization": 0.80 + i * 0.05}
        with open(cell_dir / "config.json", "w") as f:
            json.dump(config, f)
        with open(cell_dir / "metrics.json", "w") as f:
            json.dump(metrics, f)

    catalog_path = tmp_path / "catalog.json"
    catalog = ExperimentCatalog(catalog_path)
    summary = register_sweep(sweep_dir, catalog)

    assert summary["n_cells"] == 3
    assert len(catalog.list_experiments()) == 3
    assert (sweep_dir / "sweep_summary.json").exists()


def test_finalize_experiment_full_lifecycle(tmp_path):
    """finalize_experiment saves + registers in one call."""
    out = tmp_path / "run_005"
    catalog_path = tmp_path / "catalog.json"

    # Pre-create catalog so finalize can find it
    # (In real usage, catalog.json is in CWD)
    import os
    orig_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        finalize_experiment(
            out,
            config={
                "timestamp": "20260404_005",
                "scenario": "test",
                "seed": 99,
            },
            metrics={"auc": 0.75},
        )

        assert (out / "config.json").exists()
        assert (out / "metrics.json").exists()

        catalog = ExperimentCatalog(catalog_path)
        assert catalog.find("005") is not None
    finally:
        os.chdir(orig_cwd)


def test_catalog_list_by_scenario(tmp_path):
    """list_by_scenario filters correctly."""
    catalog_path = tmp_path / "catalog.json"
    catalog = ExperimentCatalog(catalog_path)

    for i, scenario in enumerate(["noshow", "noshow", "stroke"]):
        out = tmp_path / f"run_{i}"
        out.mkdir()
        config = {
            "timestamp": f"t{i}",
            "scenario": scenario,
        }
        with open(out / "config.json", "w") as f:
            json.dump(config, f)
        with open(out / "metrics.json", "w") as f:
            json.dump({}, f)
        catalog.register(out, config, {})

    assert len(catalog.list_by_scenario("noshow")) == 2
    assert len(catalog.list_by_scenario("stroke")) == 1
