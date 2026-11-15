"""Review packet for targeted reminders: evidence is real and faults show."""

import json

import pytest
from omegaconf import OmegaConf

from healthcare_sim_sdk.experiments.review import verify_packet
from healthcare_sim_sdk.scenarios.noshow_targeted_reminders import review
from healthcare_sim_sdk.scenarios.noshow_targeted_reminders.run_review_demo import (
    DEFAULT_CONFIG, run_demo,
)


@pytest.fixture
def small_config(tmp_path):
    cfg = OmegaConf.to_container(OmegaConf.load(DEFAULT_CONFIG), resolve=True)
    cfg.update(horizon_days=8, n_patients=400, seeds=[1, 2],
               cells=[cfg["cells"][1]])
    path = tmp_path / "demo.yaml"
    OmegaConf.save(OmegaConf.create(cfg), path)
    return path


def _claims(out):
    data = json.loads((out / "evidence.json").read_text())
    return {c["id"]: c["status"] for c in data["claims"]}, data


def test_demo_packet_is_intact_and_honest(tmp_path, small_config):
    out = tmp_path / "packet"
    run_demo(out, small_config)
    claims, data = _claims(out)
    assert claims["process_integrity"] == "PASS"
    assert claims["inspectability"] == "PASS"
    # Deliberately unrun claims stay visible and keep the packet from PASS.
    assert claims["model_validation"] == "NOT_RUN"
    assert claims["study_design"] == "NOT_RUN"
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["machine_status"] == "NOT_RUN"
    assert manifest["missing_runs"] == []
    status = {c["name"]: c["status"] for c in verify_packet(out)}
    assert status["artifact_integrity"] == "PASS"
    assert status["run_accounting"] == "PASS"
    for run in data["runs"]:
        assert {c["category"] for c in run["cases"]} == {
            "helped", "reached_no_effect", "missed_noshow"}


def test_tampered_trace_is_detected(tmp_path, small_config):
    out = tmp_path / "packet"
    run_demo(out, small_config)
    trace = next((out / "traces").glob("*.jsonl"))
    trace.write_text(trace.read_text() + "{}\n")
    status = {c["name"]: c["status"] for c in verify_packet(out)}
    assert status["artifact_integrity"] == "FAIL"


def test_broken_counterfactual_fails_process_integrity(
        tmp_path, small_config, monkeypatch):
    original = review.build_scenario

    def wrong_seed(cell, defaults, n, horizon, seed, with_predictions=True,
                   effectiveness_override=None):
        if not with_predictions:
            seed += 1  # the "no-AI world" no longer shares random numbers
        return original(cell, defaults, n, horizon, seed, with_predictions,
                        effectiveness_override)

    monkeypatch.setattr(review, "build_scenario", wrong_seed)
    out = tmp_path / "packet"
    run_demo(out, small_config)
    claims, data = _claims(out)
    assert claims["process_integrity"] == "FAIL"
    statuses = {c["name"]: c["status"] for c in data["runs"][0]["checks"]}
    assert statuses["counterfactual_is_no_ai_world"] == "FAIL"
