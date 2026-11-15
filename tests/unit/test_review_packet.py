"""Fault-injection and golden examples for reviewer confidence."""

import copy
import json

import pytest

from healthcare_sim_sdk.experiments.review import (
    ReviewPacket, evidence, file_digest, run_identity, verify_packet,
    write_certificate,
)
from healthcare_sim_sdk.experiments.uncertainty import (
    binomial_performance, summarize_metrics, summarize_values,
)


def registry():
    return {"decision_question": "Can we trust this example?", "claims": [{
        "id": "accounting", "question": "Do totals reconcile?",
        "estimand": "event count", "population": "all entities",
        "denominator": "all", "comparison": "raw vs summary",
        "horizon": "3 steps", "acceptance": "exact equality",
        "reviewers": ["clinical", "operations", "statistics", "governance", "engineering"],
        "evidence": ["reconcile"],
    }], "metrics": [{"key": "events", "unit": "events"}]}


def packet(tmp_path, expected=(("a", 1),)):
    return ReviewPacket(tmp_path / "packet", registry(), {"seed": 1},
                        expected, source_root=tmp_path)


def test_missing_evidence_never_green_and_roles_visible(tmp_path):
    p = packet(tmp_path)
    p.add_run("a", 1, metrics={"events": 3}, checks=[])
    overview = p.write()
    assert "NOT_RUN" in overview.read_text()
    assert "required machine evidence" not in overview.read_text()
    for role in ("clinical", "operations", "statistics", "governance", "engineering"):
        assert (p.output_dir / f"review_{role}.md").is_file()
    assert all(x["status"] == "NOT_RUN" for x in json.loads(
        (p.output_dir / "human_review.json").read_text()).values())


def test_missing_duplicate_and_unexpected_seed(tmp_path):
    p = packet(tmp_path, (("a", 1), ("a", 2)))
    p.add_run("a", 1, metrics={"events": 3}, checks=[evidence("reconcile", "PASS")])
    with pytest.raises(ValueError, match="duplicate"):
        p.add_run("a", 1)
    with pytest.raises(ValueError, match="Unexpected"):
        p.add_run("a", 3)
    p.write()
    m = json.loads((p.output_dir / "manifest.json").read_text())
    assert m["missing_runs"] == [["a", 2]]
    assert m["machine_status"] == "NOT_RUN"


def test_report_tampering_detected_and_human_notes_preserved(tmp_path):
    p = packet(tmp_path)
    p.add_run("a", 1, metrics={"events": 3}, checks=[evidence("reconcile", "PASS")])
    p.write()
    assert verify_packet(p.output_dir)[0]["status"] == "PASS"
    notes = p.output_dir / "human_review.json"
    notes.write_text('{"clinical":{"notes":"Reviewer concern"}}')
    p.write()
    assert "Reviewer concern" in notes.read_text()
    (p.output_dir / "review_packet.md").write_text("Fabricated benefit")
    assert verify_packet(p.output_dir)[0]["status"] == "FAIL"


def test_config_and_model_change_invalidate_certificate(tmp_path):
    model = tmp_path / "model.yaml"
    model.write_text("parameter: 1")
    identity = run_identity({"seed": 1}, model, tmp_path)
    p = packet(tmp_path)
    report = tmp_path / "certificate_results.json"
    report.write_text('{"verified":true}')
    cert = write_certificate(tmp_path / "certificate.json", identity,
                             [evidence("fit", "PASS")], [report.name])
    p.add_certificate("original", cert, identity)
    changed = run_identity({"seed": 2}, model, tmp_path)
    p.add_certificate("changed_config", cert, changed)
    model.write_text("parameter: 2")
    p.add_certificate("changed_model", cert, run_identity({"seed": 1}, model, tmp_path))
    report.write_text('{"verified":false}')
    p.add_certificate("tampered_report", cert, identity)
    assert [c["status"] for c in p.experiment_checks] == [
        "PASS", "INCONCLUSIVE", "INCONCLUSIVE", "INCONCLUSIVE"]


def test_source_change_changes_identity(tmp_path):
    package = tmp_path / "healthcare_sim_sdk"
    package.mkdir()
    source = package / "example.py"
    source.write_text("value = 1")
    first = run_identity({}, source_root=tmp_path)
    source.write_text("value = 2")
    assert run_identity({}, source_root=tmp_path) != first


def test_failed_execution_retained(tmp_path):
    p = packet(tmp_path)
    p.add_run("a", 1, error="simulated failure")
    p.write()
    m = json.loads((p.output_dir / "manifest.json").read_text())
    assert m["machine_status"] == "FAIL"
    assert m["failed_runs"] == [["a", 1]]


def test_invalid_cases_and_artifact_paths_rejected(tmp_path):
    p = packet(tmp_path)
    with pytest.raises(ValueError):
        p.add_run("a", 1, metrics={"rate": float("nan")})
    with pytest.raises(ValueError, match="Artifact"):
        p.add_run("a", 1, artifacts=["../secret"])
    with pytest.raises(ValueError, match="rationale"):
        evidence("capacity", "NOT_APPLICABLE")
    with pytest.raises(ValueError, match="Duplicate evidence"):
        p.add_run("a", 1, checks=[evidence("x", "PASS"), evidence("x", "PASS")])


def test_human_review_is_not_machine_approval(tmp_path):
    p = packet(tmp_path)
    p.add_run("a", 1, metrics={"events": 3}, checks=[evidence("reconcile", "PASS")])
    p.write()
    m = json.loads((p.output_dir / "manifest.json").read_text())
    assert m["machine_status"] == "PASS"
    assert m["human_review"] == "NOT_RUN"
    assert "pending" in (p.output_dir / "review_clinical.md").read_text()


def test_claim_definitions_required(tmp_path):
    r = copy.deepcopy(registry())
    del r["claims"][0]["denominator"]
    with pytest.raises(ValueError, match="denominator"):
        ReviewPacket(tmp_path, r, {}, [("a", 1)])


def test_distribution_and_mean_precision_distinguished():
    s = summarize_values(range(100))
    assert s["mean"] == 49.5
    assert s["seed_p025"] == pytest.approx(2.475)
    assert s["mean_ci95_low"] > s["seed_p025"]
    assert s["mean_ci95_high"] < s["seed_p975"]
    assert summarize_values([1])["mean_ci95_low"] is None
    assert summarize_values([1])["uncertainty_status"] == "INCONCLUSIVE"


def test_summaries_do_not_drop_bad_seeds():
    with pytest.raises(ValueError, match="Missing required"):
        summarize_metrics([{"x": 1}, {}], ["x"])
    with pytest.raises(ValueError, match="finite"):
        summarize_metrics([{"x": float("nan")}], ["x"])
    with pytest.raises(ValueError, match="seed"):
        summarize_metrics([{"seed": 1, "x": 1}], ["x"], [1, 2])


def test_zero_events_have_nonzero_uncertainty():
    s = binomial_performance(0, 30)
    assert s["mcse"] == 0
    assert 0.10 < s["ci95_high"] < 0.12
    assert s["boundary_note"]
    assert binomial_performance(24, 30)["mcse"] == pytest.approx(0.0730296743)


def test_recorded_artifacts_are_hashed(tmp_path):
    p = packet(tmp_path)
    trace = p.output_dir / "trace.json"
    trace.write_text("[]")
    p.add_run("a", 1, metrics={"events": 1}, checks=[evidence("reconcile", "PASS")], artifacts=[trace.name])
    p.write()
    manifest = json.loads((p.output_dir / "manifest.json").read_text())
    assert manifest["artifacts"][trace.name] == file_digest(trace)


def test_registry_requirement_cannot_be_made_optional(tmp_path):
    r = registry()
    r["claims"][0]["evidence"] = ["reconcile", "heldout"]
    p = ReviewPacket(tmp_path, r, {}, [("a", 1)])
    p.add_run("a", 1, metrics={"events": 1}, checks=[
        evidence("reconcile", "PASS"), evidence("heldout", "NOT_RUN", required=False),
    ])
    p.write()
    assert json.loads((tmp_path / "manifest.json").read_text())["machine_status"] == "NOT_RUN"


def test_missing_registry_metric_blocks_and_partial_seed_summary_is_explicit(tmp_path):
    p = packet(tmp_path, (("a", 1), ("a", 2)))
    p.add_run("a", 1, checks=[evidence("reconcile", "PASS")])
    p.write()
    assert "incomplete |" in (p.output_dir / "review_packet.md").read_text()
    assert json.loads((p.output_dir / "manifest.json").read_text())["machine_status"] == "NOT_RUN"
    p.add_run("a", 2, metrics={"events": 4}, checks=[evidence("reconcile", "PASS")])
    p.write()
    assert json.loads((p.output_dir / "manifest.json").read_text())["machine_status"] == "NOT_RUN"


def test_artifacts_referenced_by_evidence_and_cases_are_bound(tmp_path):
    p = packet(tmp_path)
    trace = p.output_dir / "trace.json"
    trace.write_text("[]")
    p.add_run("a", 1, metrics={"events": 1}, checks=[evidence("reconcile", "PASS", artifacts=[trace.name])],
              cases=[{"category": "sample", "artifact": trace.name}])
    p.write()
    assert trace.name in json.loads((p.output_dir / "manifest.json").read_text())["artifacts"]
    trace.write_text("[1]")
    with pytest.raises(ValueError, match="changed after recording"):
        p.write()


def test_absent_or_reserved_link_cannot_be_evidence(tmp_path):
    p = packet(tmp_path)
    with pytest.raises(ValueError, match="Missing evidence artifact"):
        p.add_run("a", 1, checks=[evidence("reconcile", "PASS", artifacts=["missing.json"])])
    with pytest.raises(ValueError, match="reserved"):
        p.add_artifact("config.json")


def test_experiment_artifact_is_hashed_and_linked(tmp_path):
    p = packet(tmp_path)
    analysis = p.output_dir / "analysis.json"
    analysis.write_text('{"estimate": 2}')
    p.add_artifact(analysis.name)
    p.write()
    assert "[analysis.json](analysis.json)" in (p.output_dir / "review_packet.md").read_text()
    assert analysis.name in json.loads((p.output_dir / "manifest.json").read_text())["artifacts"]


@pytest.mark.parametrize("payload", [
    [], {"checks": [{"name": "fake", "status": "UNKNOWN"}], "artifacts": {}},
    {"checks": [{"name": "fake", "status": "NOT_APPLICABLE"}], "artifacts": {}},
    {"checks": "PASS", "artifacts": {}},
])
def test_malformed_certificate_records_fail(tmp_path, payload):
    p = packet(tmp_path)
    path = tmp_path / "certificate.json"
    path.write_text(json.dumps(payload))
    p.add_certificate("bad", path)
    assert p.experiment_checks[-1]["status"] == "FAIL"


def test_certificate_mutation_before_packet_write_invalidates_reuse(tmp_path):
    p = packet(tmp_path)
    report = tmp_path / "report.json"
    report.write_text('{"result": 1}')
    cert = write_certificate(tmp_path / "certificate.json", p.identity,
                             [evidence("fit", "PASS")], [report.name])
    p.add_certificate("cert", cert)
    assert p.experiment_checks[-1]["status"] == "PASS"
    report.write_text('{"result": 2}')
    p.write()
    assert next(c for c in p.experiment_checks if c["name"] == "cert")["status"] == "INCONCLUSIVE"


def test_packet_identity_binds_registry_and_expected_seeds(tmp_path):
    first = packet(tmp_path / "first", (("a", 1),))
    second = packet(tmp_path / "second", (("a", 1), ("a", 2)))
    assert first.identity["expected_runs_sha256"] != second.identity["expected_runs_sha256"]
    original = first.identity["registry_sha256"]
    first.registry["claims"][0]["acceptance"] = "changed criterion"
    first.write()
    assert original != first.identity["registry_sha256"] or any(
        c["name"] == "packet_provenance" and c["status"] == "INCONCLUSIVE" for c in first.experiment_checks)


def test_manifest_cannot_omit_required_artifacts(tmp_path):
    p = packet(tmp_path)
    p.write()
    path = p.output_dir / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["artifacts"] = {"config.json": manifest["artifacts"]["config.json"]}
    path.write_text(json.dumps(manifest))
    assert verify_packet(p.output_dir)[0]["status"] == "FAIL"


def test_manifest_false_green_and_deleted_seed_are_detected(tmp_path):
    p = packet(tmp_path, (("a", 1), ("a", 2)))
    p.add_run("a", 1, metrics={"events": 1}, checks=[evidence("reconcile", "PASS")])
    p.write()
    path = p.output_dir / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest.update(machine_status="PASS", expected_runs=[["a", 1]], missing_runs=[])
    path.write_text(json.dumps(manifest))
    assert any(c["status"] == "FAIL" for c in verify_packet(p.output_dir))


@pytest.mark.parametrize("raw", ["[]", '{"artifacts": NaN}', '{"artifacts": 1e999}', '{broken'])
def test_invalid_manifest_returns_failed_check(tmp_path, raw):
    (tmp_path / "manifest.json").write_text(raw)
    assert verify_packet(tmp_path)[0]["status"] == "FAIL"


def test_failed_machine_evidence_is_not_human_approval(tmp_path):
    p = packet(tmp_path)
    p.write()
    assert any(c["status"] == "NOT_RUN" for c in verify_packet(p.output_dir))
    assert json.loads((p.output_dir / "manifest.json").read_text())["human_review"] == "NOT_RUN"


def test_optional_registry_metric_does_not_block(tmp_path):
    r = registry()
    r["metrics"][0]["required"] = False
    p = ReviewPacket(tmp_path, r, {}, [("a", 1)])
    p.add_run("a", 1, checks=[evidence("reconcile", "PASS")])
    p.write()
    assert json.loads((tmp_path / "manifest.json").read_text())["machine_status"] == "PASS"


def test_duplicate_seed_without_expected_list_cannot_inflate_precision():
    with pytest.raises(ValueError, match="duplicate seed"):
        summarize_metrics([{"seed": 1, "x": 2}, {"seed": 1, "x": 4}], ["x"])
    with pytest.raises(ValueError, match="seed"):
        summarize_metrics([{"seed": 1, "x": 2}, {"x": 4}], ["x"])


def test_numeric_overflow_cannot_return_passing_uncertainty():
    import numpy as np
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="overflowed"):
            summarize_values([1e308, 1e308])
    with pytest.raises(ValueError, match="integer"):
        binomial_performance(True, 2)


def test_case_highlights_are_short_with_complete_hashed_appendix(tmp_path):
    p = packet(tmp_path, (("a", 1), ("a", 2)))
    for seed in (1, 2):
        p.add_run("a", seed, metrics={"events": 1},
                  checks=[evidence("reconcile", "PASS")], cases=[
                      {"category": "harm", "status": "NOT_APPLICABLE" if seed == 1 else "PASS",
                       "narrative": f"harm narrative seed {seed}", "candidate_count": seed - 1},
                      {"category": "help", "status": "PASS", "narrative": f"help narrative seed {seed}"},
                  ])
    p.write()
    overview = (p.output_dir / "review_packet.md").read_text()
    assert "harm narrative seed 2" in overview
    assert "harm narrative seed 1" not in overview
    assert "help narrative seed 1" in overview
    assert "help narrative seed 2" not in overview
    assert "Unavailable/unresolved in 1 recorded run(s)" in overview
    appendix = (p.output_dir / "case_appendix.md").read_text()
    assert all(f"{kind} narrative seed {seed}" in appendix
               for kind in ("harm", "help") for seed in (1, 2))
    statistics = (p.output_dir / "review_statistics.md").read_text()
    assert "case_appendix.md" in statistics and "harm narrative" not in statistics
    manifest = json.loads((p.output_dir / "manifest.json").read_text())
    assert "case_appendix.md" in manifest["artifacts"]


def test_result_tables_use_resolved_cell_labels(tmp_path):
    p = ReviewPacket(tmp_path, registry(), {"resolved_cells": [
        {"cell": "a", "label": "Higher assessment capacity"}]}, [("a", 1)])
    p.add_run("a", 1, metrics={"events": 3}, checks=[evidence("reconcile", "PASS")])
    p.write()
    assert "Higher assessment capacity (a)" in (tmp_path / "review_packet.md").read_text()
