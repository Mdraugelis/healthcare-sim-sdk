"""The save step's registration gate and run provenance."""

import json

import pytest

from healthcare_sim_sdk.experiments.catalog import ExperimentCatalog
from healthcare_sim_sdk.experiments.lifecycle import (
    finalize_experiment, register_sweep,
)
from healthcare_sim_sdk.experiments.provenance import find_seeds
from healthcare_sim_sdk.experiments.validate import Check, CheckStatus


def _passing(name="simulation_validation"):
    return Check(name, "supplied", "pass", "pass", "exact", True,
                 category="simulation")


def _failing(name="simulation_validation"):
    return Check(name, "supplied", "pass", "fail", "exact", False,
                 category="simulation")


def test_required_failure_is_never_registered(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "run"
    with pytest.raises(ValueError, match="not registered"):
        finalize_experiment(out, {"seed": 1, "timestamp": "t1"},
                            {"events": 3}, validation_checks=[_failing()],
                            require_validated=True)
    record = json.loads((out / "validation.json").read_text())
    assert record["validated"] is False
    assert record["rejected"] is True
    assert record["registered"] is False
    # Diagnostic files are kept.
    for name in ("config.json", "metrics.json", "provenance.json"):
        assert (out / name).is_file()
    catalog = ExperimentCatalog(tmp_path / "catalog.json")
    assert catalog.find("t1") is None


def test_missing_required_check_is_never_registered(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "run"
    with pytest.raises(ValueError):
        finalize_experiment(out, {"seed": 1, "timestamp": "t2"},
                            {"events": 3}, validation_checks=[_passing()],
                            required_checks=["equity_audit"],
                            require_validated=True)
    assert not (tmp_path / "catalog.json").exists()


def test_required_success_is_registered_as_verified(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "run"
    finalize_experiment(out, {"seed": 1, "timestamp": "t3"}, {"events": 3},
                        validation_checks=[_passing()],
                        require_validated=True)
    catalog = ExperimentCatalog(tmp_path / "catalog.json")
    entry = catalog.find("t3")
    assert entry["validated"] is True
    assert entry["validation_status"] == "verified"
    assert [e["timestamp"] for e in catalog.list_verified()] == ["t3"]


def test_unchecked_run_is_registered_as_unverified(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "run"
    finalize_experiment(out, {"seed": 1, "timestamp": "t4"}, {"events": 3})
    catalog = ExperimentCatalog(tmp_path / "catalog.json")
    entry = catalog.find("t4")
    assert entry["validated"] is False
    assert entry["validation_status"] == "unverified"
    assert catalog.list_verified() == []


def test_failed_checks_without_requirement_stay_unverified(tmp_path,
                                                           monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "run"
    finalize_experiment(out, {"seed": 1, "timestamp": "t5"}, {"events": 3},
                        validation_checks=[_failing()])
    entry = ExperimentCatalog(tmp_path / "catalog.json").find("t5")
    assert entry["validated"] is False


def test_provenance_is_recorded(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "run"
    config = {"seed": 7, "timestamp": "t6",
              "model": {"fit_seed": 11}, "sweep": {"seeds": [1, 2]}}
    finalize_experiment(out, config, {"events": 3})
    prov = json.loads((out / "provenance.json").read_text())
    assert set(prov) >= {"code_revision", "config_sha256", "seeds",
                         "python", "dependencies"}
    assert prov["seeds"] == {"seed": 7, "model.fit_seed": 11,
                             "sweep.seeds": [1, 2]}
    assert prov["dependencies"]["numpy"]
    assert len(prov["config_sha256"]) == 64
    entry = ExperimentCatalog(tmp_path / "catalog.json").find("t6")
    assert entry["code_revision"] == prov["code_revision"]


def test_find_seeds_handles_lists():
    assert find_seeds({"arms": [{"seed": 1}, {"seed": 2}]}) == [
        ("arms[0].seed", 1), ("arms[1].seed", 2)]


def test_sweep_registration_skips_rejected_cells(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sweep = tmp_path / "sweep"
    finalize_experiment(sweep / "0", {"seed": 1, "timestamp": "a"},
                        {"events": 1}, validation_checks=[_passing()],
                        require_validated=True)
    with pytest.raises(ValueError):
        finalize_experiment(sweep / "1", {"seed": 2, "timestamp": "b"},
                            {"events": 1}, validation_checks=[_failing()],
                            require_validated=True)
    catalog = ExperimentCatalog(tmp_path / "sweep_catalog.json")
    register_sweep(sweep, catalog)
    assert catalog.find("a") is not None
    assert catalog.find("b") is None


def test_legacy_entries_report_no_validation_record(tmp_path):
    out = tmp_path / "legacy"
    out.mkdir()
    (out / "config.json").write_text(json.dumps({"timestamp": "old"}))
    (out / "metrics.json").write_text("{}")
    catalog = ExperimentCatalog(tmp_path / "catalog.json")
    catalog.register(out, {"timestamp": "old"}, {})
    entry = catalog.find("old")
    assert entry["validated"] is None
    assert entry["validation_status"] == "no validation record"


def test_not_applicable_needs_rationale_to_pass_the_gate(tmp_path,
                                                         monkeypatch):
    monkeypatch.chdir(tmp_path)
    bare = Check("simulation_validation", "na", "", "", "",
                 status=CheckStatus.NOT_APPLICABLE)
    with pytest.raises(ValueError):
        finalize_experiment(tmp_path / "r1", {"seed": 1, "timestamp": "n1"},
                            {"events": 1}, validation_checks=[bare],
                            require_validated=True)
    justified = Check("simulation_validation", "na", "", "", "",
                      status=CheckStatus.NOT_APPLICABLE,
                      applicability="no intervention in this run",
                      rationale="counterfactual-only baseline")
    finalize_experiment(tmp_path / "r2", {"seed": 1, "timestamp": "n2"},
                        {"events": 1}, validation_checks=[justified],
                        require_validated=True)
    assert ExperimentCatalog(tmp_path / "catalog.json").find("n2")
