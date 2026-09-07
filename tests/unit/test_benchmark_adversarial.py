"""Independent attacks on custody, disclosure order and complete submissions.

These tests exercise the real signed protocol. A deterministic in-process
adapter stands in for OS execution where needed; OS isolation is tested
separately. No published target is needed to demonstrate these attacks.
"""

from copy import deepcopy
from pathlib import Path

import pytest

from healthcare_sim_sdk.experiments.benchmark import execution, protocol
from healthcare_sim_sdk.experiments.benchmark.contracts import (
    canonical, digest, read_json, validate_case,
)
from healthcare_sim_sdk.experiments.benchmark.shield_rt import run_case


def _case():
    root = Path(__file__).resolve().parents[2]
    case = read_json(root / "examples/benchmarks/shield_rt_pilot.json")
    case["case_id"] = "adversarial_synthetic"
    case["study"]["id"] = "synthetic_trial"
    case["study"]["title"] = "Synthetic integrity check"
    case["exposure"] = "declared_unseen"
    case["run_plan"]["seeds"] = [17, 19]
    case["sources"] = {"synthetic": {
        "kind": "assumed", "citation": "Invented test inputs only",
        "available_at": "2026-09-07", "derived_from": [],
    }}
    for entry in case["inputs"].values():
        entry["source_id"] = "synthetic"
    case["inputs"]["n_control"]["value"] = 20
    case["inputs"]["n_intervention"]["value"] = 21
    return case


@pytest.fixture
def custody(tmp_path):
    case = _case()
    reference = {
        "case_id": case["case_id"], "study_id": case["study"]["id"],
        "exposure": "synthetic",
        "targets": {
            "risk_difference": {"value": 0, "source": "invented"},
            "control_risk": {"value": 0.2, "source": "invented"},
            "intervention_risk": {"value": 0.2, "source": "invented"},
        },
    }
    vault, session = tmp_path / "custodian-vault", tmp_path / "runner-session"
    protocol.seal_reference(reference, vault)
    protocol.freeze_case(case, session, vault)
    return case, session, vault


def _synthetic_execution(case, output_dir, *, require_isolation=True):
    return {
        "runs": [run_case(case, seed) for seed in case["run_plan"]["seeds"]],
        "execution": {
            "status": "PASS", "mode": "macos_sandbox_exec",
            "source_hashes": execution.source_manifest(),
            "environment": execution.runtime_manifest(),
        },
    }


def _replace(path, value):
    if path.exists():
        path.chmod(0o600)
    path.write_text(canonical(value) + "\n")


def test_plausible_forged_results_cannot_replace_supervised_execution(custody):
    case, session, vault = custody
    # Every seed, numerical value and public hash can be copied or calculated by
    # the runner. Only the custodian's execution receipt establishes custody.
    forged = _synthetic_execution(case, session / "not-executed")
    forged["identity"] = read_json(session / "frozen_manifest.json")
    _replace(session / "run_result.json", forged)
    with pytest.raises(ValueError):
        protocol.commit_submission(session, vault)


def test_results_rewritten_after_supervised_execution_are_rejected(
    custody, monkeypatch,
):
    _, session, vault = custody
    monkeypatch.setattr(execution, "run_isolated", _synthetic_execution)
    protocol.execute_registered(session, vault)
    rewritten = read_json(session / "run_result.json")
    rewritten["runs"][0]["predictions"]["risk_difference"] = 0.123456
    _replace(session / "run_result.json", rewritten)
    with pytest.raises(ValueError):
        protocol.commit_submission(session, vault)


def test_concurrent_rewrite_cannot_be_signed_as_the_supervisors_output(
    custody, monkeypatch,
):
    _, session, vault = custody
    monkeypatch.setattr(execution, "run_isolated", _synthetic_execution)
    original_write = protocol._write
    attacked = []

    def concurrent_write(path, value):
        original_write(path, value)
        if Path(path).name == "run_result.json":
            # Schedule an attacker write after the supervisor closes its output
            # but before it signs custody. The trusted in-memory result remains
            # untouched; the session owner can overwrite this file on disk.
            changed = deepcopy(value)
            changed["runs"][0]["predictions"]["risk_difference"] = 0.123456
            _replace(Path(path), changed)
            attacked.append(True)

    monkeypatch.setattr(protocol, "_write", concurrent_write)
    with pytest.raises(ValueError):
        protocol.execute_registered(session, vault)
        protocol.commit_submission(session, vault)
    assert attacked


def test_frozen_case_and_self_consistent_manifest_cannot_be_rewritten(custody):
    _, session, vault = custody
    changed = read_json(session / "frozen_case.json")
    changed["inputs"]["baseline_risk"]["value"] = {
        "distribution": "fixed", "value": 0.01,
    }
    manifest = read_json(session / "frozen_manifest.json")
    manifest["case_sha256"] = digest(changed)
    _replace(session / "frozen_case.json", changed)
    _replace(session / "frozen_manifest.json", manifest)
    with pytest.raises(ValueError):
        protocol.verify_session(session, vault)


def test_rewrite_after_receipt_verification_cannot_enter_submission(
    custody, monkeypatch,
):
    _, session, vault = custody
    monkeypatch.setattr(execution, "run_isolated", _synthetic_execution)
    protocol.execute_registered(session, vault)
    original_verify = protocol.verify_session
    attacked = []

    def verify_then_race(session_dir, vault_dir):
        verified = original_verify(session_dir, vault_dir)
        rewritten = read_json(session / "run_result.json")
        rewritten["runs"][0]["predictions"]["risk_difference"] = 0.123456
        _replace(session / "run_result.json", rewritten)
        attacked.append(True)
        return verified

    monkeypatch.setattr(protocol, "verify_session", verify_then_race)
    with pytest.raises(ValueError):
        protocol.commit_submission(session, vault)
    assert attacked


def test_reveal_before_committed_submission_is_rejected(custody):
    _, session, vault = custody
    with pytest.raises(ValueError):
        protocol.reveal(session, vault)
    assert not (session / "revealed_reference.json").exists()


def test_revealed_vault_cannot_be_restarted_as_declared_unseen(
    custody, monkeypatch,
):
    case, session, vault = custody
    monkeypatch.setattr(execution, "run_isolated", _synthetic_execution)
    protocol.execute_registered(session, vault)
    protocol.commit_submission(session, vault)
    protocol.reveal(session, vault)
    changed = deepcopy(case)
    changed["inputs"]["relative_effect"]["value"] = {
        "distribution": "fixed", "value": 0,
    }
    with pytest.raises(ValueError):
        protocol.freeze_case(changed, session.parent / "optimized-restart", vault)


@pytest.mark.parametrize("attack", ["missing", "reordered", "duplicate", "extra"])
def test_seed_selection_cannot_be_committed(custody, monkeypatch, attack):
    _, session, vault = custody

    def incomplete(case, output_dir, **kwargs):
        result = _synthetic_execution(case, output_dir, **kwargs)
        runs = result["runs"]
        if attack == "missing":
            runs.pop()
        elif attack == "reordered":
            runs.reverse()
        elif attack == "duplicate":
            runs[1] = deepcopy(runs[0])
        else:
            runs.append(run_case(case, 23))
        return result

    monkeypatch.setattr(execution, "run_isolated", incomplete)
    with pytest.raises(ValueError):
        protocol.execute_registered(session, vault)
        protocol.commit_submission(session, vault)


@pytest.mark.parametrize("linked", [True, False])
def test_withheld_target_cannot_be_delivered_through_source_ledger(linked):
    case = _case()
    case["sources"]["target"] = {
        "kind": "target_outcome", "citation": "withheld study endpoint",
        "available_at": "2020-01-01", "derived_from": [],
        "target_ids": ["risk_difference"],
    }
    if linked:
        case["sources"]["bridge"] = {
            "kind": "external", "citation": "apparently external fitted prior",
            "available_at": "2020-01-01", "derived_from": ["target"],
        }
        case["sources"]["synthetic"]["derived_from"] = ["bridge"]
    with pytest.raises(ValueError):
        validate_case(case)


@pytest.mark.parametrize("attack", ["unknown_endpoint", "wrong_support", "primary"])
def test_unexecutable_or_redefined_adapter_estimands_fail_before_freeze(attack):
    case = _case()
    if attack == "unknown_endpoint":
        case["endpoints"].append({
            "id": "mortality", "units": "risk", "lower_bound": 0, "upper_bound": 1,
        })
    elif attack == "wrong_support":
        case["endpoints"][0]["lower_bound"] = -100
    else:
        case["scoring"]["primary_endpoint"] = "control_risk"
    with pytest.raises(ValueError):
        validate_case(case)
