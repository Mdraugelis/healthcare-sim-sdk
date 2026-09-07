"""Actual-run packet evidence and intentionally corrupted outcomes."""

import json

import numpy as np
import pytest

from healthcare_sim_sdk.experiments.review import verify_packet
from healthcare_sim_sdk.scenarios.malnutrition_prioritization.review import (
    create_packet, run_reviewed_cell,
)
from healthcare_sim_sdk.scenarios.malnutrition_prioritization.runner_common import (
    config_from_dict, run_once, verify_results,
)


def config():
    return dict(initial_census=30, admissions_per_day=4,
                rdn_assessments_per_day=3, ml_enabled=False)


def test_each_actual_seed_has_checks_and_traces(tmp_path):
    cfg = config()
    p = create_packet(tmp_path, cfg, [{"config": cfg}], [1, 2])
    rows = run_reviewed_cell(p, "cell_00", config_from_dict(cfg), 14, [1, 2])
    p.write()
    assert len(rows) == len(p.runs) == 2
    assert verify_packet(tmp_path)[0]["status"] == "PASS"
    assert all(any(c["name"] == "capacity" for c in r["checks"]) for r in p.runs.values())
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["machine_status"] == "NOT_RUN"  # no actual study design supplied
    trace = json.loads((tmp_path / "cell_00_seed_1_traces.json").read_text())
    assert {r["branch"] for r in trace} == {"factual", "counterfactual"}
    assert any(c["candidate_count"] == 0 for c in p.runs[("cell_00", 1)]["cases"])


def test_public_demo_binds_its_full_registry_before_running(tmp_path):
    from healthcare_sim_sdk.scenarios.malnutrition_prioritization.run_review_demo import (
        run_demo,
    )
    run_demo(tmp_path)
    checks = {c["name"]: c["status"] for c in verify_packet(tmp_path)}
    assert checks["saved_identity_consistency"] == "PASS"
    assert "FAIL" not in checks.values()
    saved = json.loads((tmp_path / "evidence.json").read_text())
    provenance = next(c for c in saved["experiment_checks"]
                      if c["name"] == "packet_provenance")
    assert provenance["status"] == "PASS"


@pytest.mark.parametrize("fault", ["capacity", "nan", "inf", "missing_period", "duplicate_id", "event", "missing_prediction", "action_count", "id_order", "event_reassignment"])
def test_actual_run_faults_detected(fault):
    cfg = config_from_dict(config())
    _, res = run_once(cfg, 14, 1)
    if fault == "capacity":
        res.counterfactual_outcomes[3].metadata["assessed_today"] = 99
    elif fault in ("nan", "inf"):
        res.counterfactual_outcomes[3].secondary["harm"][0] = getattr(np, fault)
    elif fault == "missing_period":
        del res.counterfactual_outcomes[3]
    elif fault == "duplicate_id":
        res.outcomes[3].entity_ids[0] = res.outcomes[3].entity_ids[1]
    elif fault == "id_order":
        res.outcomes[3].entity_ids = res.outcomes[3].entity_ids[::-1]
    elif fault == "missing_prediction":
        del res.predictions[3]
    elif fault == "action_count":
        res.outcomes[3].secondary["assess_day"][:20] = 3
    elif fault == "event_reassignment":
        out = next(o for o in res.outcomes.values() if o.events.sum())
        yes = np.flatnonzero(out.events == 1)[0]
        no = np.flatnonzero(out.events == 0)[0]
        out.events[yes], out.events[no] = 0, 1  # aggregate stays unchanged
    else:
        res.outcomes[3].events[0] += 1
    assert not verify_results(cfg, res)["all_pass"]


def test_missing_model_preserves_failed_run(tmp_path):
    cfg = {**config(), "ml_enabled": True,
           "model_params_path": str(tmp_path / "missing_model.yaml")}
    packet = create_packet(tmp_path, cfg, [{"config": cfg}], [1])
    with pytest.raises(FileNotFoundError):
        run_reviewed_cell(packet, "cell_00", config_from_dict(cfg), 14, [1])
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["machine_status"] == "FAIL"
    assert manifest["failed_runs"]


def test_model_identity_covers_each_cell(tmp_path):
    from healthcare_sim_sdk.experiments.review import file_digest
    first, second = tmp_path / "first.yaml", tmp_path / "second.yaml"
    first.write_text("first model")
    second.write_text("second model")
    cells = [{"config": {**config(), "ml_enabled": True,
                         "model_params_path": str(path)}}
             for path in [first, second]]
    packet = create_packet(tmp_path / "review", {}, cells, [1])
    packet.write()
    saved = json.loads((tmp_path / "review/config.json").read_text())
    assert saved["resolved_cells"][1]["model_sha256"] == file_digest(second)
    identity = packet.identity
    second.write_text("changed second model")
    other = create_packet(tmp_path / "other", {}, cells, [1])
    assert other.identity != identity


@pytest.mark.parametrize("fault", ["config", "horizon", "seed_schedule", "model_bytes"])
def test_run_plan_mismatch_is_rejected_and_preserved(tmp_path, fault):
    cfg = config()
    if fault == "model_bytes":
        model = tmp_path / "frozen.yaml"
        model.write_text("original model bytes")
        cfg.update(ml_enabled=True, model_params_path=str(model))
    packet = create_packet(tmp_path / "review", {"horizon_days": 14},
                           [{"config": cfg}], [1, 2])
    actual = config_from_dict(cfg)
    if fault == "config":
        actual.admissions_per_day += 1
    elif fault == "model_bytes":
        model.write_text("altered model bytes")
    with pytest.raises(ValueError, match="differs|changed"):
        run_reviewed_cell(
            packet, "cell_00", actual, 15 if fault == "horizon" else 14,
            [1] if fault == "seed_schedule" else [1, 2],
        )
    saved = json.loads((packet.output_dir / "manifest.json").read_text())
    assert saved["machine_status"] == "FAIL"
    assert saved["failed_runs"] == [["cell_00", 1]]
    assert (packet.output_dir / "cell_00_seed_1_run_manifest.json").exists()


def test_model_bytes_changing_during_run_are_rejected(tmp_path, monkeypatch):
    from healthcare_sim_sdk.scenarios.malnutrition_prioritization import review
    model = tmp_path / "model.yaml"
    model.write_text("original model bytes")
    cfg = {**config(), "ml_enabled": True, "model_params_path": str(model)}
    packet = create_packet(tmp_path / "review", {}, [{"config": cfg}], [1])

    def changed_during_run(*args):
        model.write_text("changed during execution")
        return None, None  # verification must never consume this result

    monkeypatch.setattr(review, "run_once", changed_during_run)
    with pytest.raises(ValueError, match="model bytes changed"):
        run_reviewed_cell(packet, "cell_00", config_from_dict(cfg), 14, [1])
    saved = json.loads((packet.output_dir / "manifest.json").read_text())
    assert saved["machine_status"] == "FAIL"
    actual = json.loads(
        (packet.output_dir / "cell_00_seed_1_run_manifest.json").read_text()
    )
    assert actual["before_run"]["model_sha256"] != actual["after_run"]["model_sha256"]
