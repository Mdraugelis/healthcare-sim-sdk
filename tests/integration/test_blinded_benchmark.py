"""Full benchmark custody/evaluation against an actual copied Python worker."""

from copy import deepcopy
from pathlib import Path

import pytest

from healthcare_sim_sdk.experiments.benchmark import execution, protocol
from healthcare_sim_sdk.experiments.benchmark.contracts import (
    digest, read_json, validate_case,
)


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def registered(tmp_path):
    case = read_json(ROOT / "examples/benchmarks/shield_rt_pilot.json")
    case["run_plan"]["seeds"] = list(range(8))
    reference = read_json(
        ROOT / "examples/benchmarks/shield_rt_published_reference.json")
    vault, session = tmp_path / "vault", tmp_path / "session"
    protocol.seal_reference(reference, vault)
    protocol.freeze_case(case, session, vault)
    return case, reference, vault, session


def test_real_worker_full_lifecycle_preserves_human_boundary(registered, monkeypatch):
    case, reference, vault, session = registered
    monkeypatch.setattr(execution, "_sandbox_backend", lambda: None)
    result = protocol.execute_registered(session, vault, require_isolation=False)
    assert [run["seed"] for run in result["runs"]] == case["run_plan"]["seeds"]
    assert all(run["status"] == "PASS" for run in result["runs"])
    assert result["execution"]["mode"] == "unisolated_development"
    assert result["execution"]["status"] == "NOT_RUN"
    assert not (session / "revealed_reference.json").exists()
    protocol.commit_submission(session, vault)
    protocol.reveal(session, vault)
    evaluation = protocol.score(session, vault)
    assert evaluation["protocol_integrity"] == "PASS"
    assert evaluation["independent_validation"]["status"] == "NOT_APPLICABLE"
    assert evaluation["input_adequacy"]["status"] == "INCONCLUSIVE"
    assert evaluation["human_review"] == "NOT_RUN"
    proof = read_json(session / "reveal_proof.json")
    manifest = read_json(session / "frozen_manifest.json")
    assert digest({"reference": reference, "nonce": proof["nonce"]}) == (
        manifest["reference_commitment"])
    assert protocol.verify_session(session, vault)["state"] == "SCORED"
    reviews = list((session / "review").glob("review_*.md"))
    assert len(reviews) == 6
    human = session / "review/human_review.json"
    human.write_text('{"reviewer_note":"Needs independent efficacy evidence"}\n')
    assert protocol.verify_session(session, vault)["status"] == "PASS"
    machine = session / "review/evaluation.json"
    machine.write_text('{}\n')
    with pytest.raises(ValueError, match="artifact"):
        protocol.verify_session(session, vault)


def test_terminal_failures_stay_visible_and_can_be_reviewed(registered, monkeypatch):
    case, _, vault, session = registered

    def failed_worker(case, output_dir, **kwargs):
        return {
            "runs": [{"seed": s, "status": "FAIL", "error": "synthetic failure"}
                     for s in case["run_plan"]["seeds"]],
            "execution": {"mode": "test_stub", "status": "FAIL",
                          "source_hashes": execution.source_manifest(),
                          "environment": execution.runtime_manifest()},
        }

    monkeypatch.setattr(execution, "run_isolated", failed_worker)
    protocol.execute_registered(session, vault)
    protocol.commit_submission(session, vault)
    protocol.reveal(session, vault)
    result = protocol.score(session, vault)
    assert result["status"] == "FAIL"
    assert result["endpoints"] == {}
    assert result["n_submitted_runs"] == len(case["run_plan"]["seeds"])
    assert result["completeness"]["status"] == "FAIL"


@pytest.mark.parametrize("mutate", [
    lambda c: c.update(secret_target=.123),
    lambda c: c["sources"]["illustrative_assumptions"].update(
        available_at="2027-01-01"),
    lambda c: c["inputs"]["n_control"].update(value=True),
    lambda c: c["inputs"]["baseline_risk"]["value"].update(high=float("nan")),
    lambda c: c["sources"]["illustrative_assumptions"].update(
        derived_from=["illustrative_assumptions"]),
])
def test_invalid_input_contract_refused_before_execution(mutate):
    case = deepcopy(read_json(ROOT / "examples/benchmarks/shield_rt_pilot.json"))
    mutate(case)
    with pytest.raises(ValueError):
        validate_case(case)


def test_nonfinite_and_duplicate_json_are_rejected(tmp_path):
    path = tmp_path / "invalid.json"
    for text in ('{"x":NaN}', '{"x":1,"x":2}'):
        path.write_text(text)
        with pytest.raises(ValueError):
            read_json(path)
