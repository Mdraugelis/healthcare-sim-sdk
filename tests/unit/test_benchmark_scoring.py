"""Independent numerical examples and false-confidence regression checks."""

from copy import deepcopy
import json

import numpy as np
import pytest

from healthcare_sim_sdk.experiments.benchmark.report import render_report
from healthcare_sim_sdk.experiments.benchmark.scoring import (
    empirical_crps, evaluate, interval_score,
)


@pytest.fixture
def evidence():
    case = {
        "case_id": "demo", "study": {"id": "study-demo"},
        "exposure": "outcome_exposed_development", "information_track": "predeployment",
        "endpoints": [{"id": "risk_difference", "units": "risk difference",
                       "lower_bound": -1, "upper_bound": 1}],
        "scoring": {"primary_endpoint": "risk_difference", "interval_alpha": .1,
                    "absolute_error_tolerance": .05, "minimum_skill": 0},
        "run_plan": {"seeds": [1, 2]},
        "inputs": {"effect": {"value": .2, "source_id": "independent"}},
        "sources": {"independent": {"kind": "external", "citation": "historical source",
                                    "available_at": "2020-01-01", "derived_from": []}},
    }
    submission = {"runs": [
        {"seed": seed, "status": "PASS", "predictions": {"risk_difference": value},
         "baselines": {"risk_difference": 0.0}, "metrics": {},
         "checks": [{"name": "conservation", "status": "PASS", "detail": "exact"}]}
        for seed, value in ((1, .1), (2, .3))
    ]}
    reference = {
        "case_id": "demo", "study_id": "study-demo", "exposure": "published",
        "targets": {"risk_difference": {"value": .2, "source": "sealed paper"}},
    }
    return case, submission, reference


def test_empirical_crps_hand_calculation():
    # E|X-1|=1; half E|X-X'|=.5, including two zero diagonal pairs.
    assert empirical_crps([0, 2], 1) == pytest.approx(.5)
    assert empirical_crps([2, 2, 2], 5) == pytest.approx(3)
    assert empirical_crps([2, 2], 2) == 0


def test_crps_against_independent_quadratic_definition():
    samples = np.array([-3.0, -.4, .3, 1.2, 5.0])
    truth = sum(abs(x - .2) for x in samples) / 5
    truth -= sum(abs(x - z) for x in samples for z in samples) / 50
    assert empirical_crps(samples[::-1], .2) == pytest.approx(truth)
    assert empirical_crps(samples + 10000, 10000.2) == pytest.approx(truth)


def test_interval_score_penalizes_width_and_misses():
    assert interval_score(0, 2, 1, .1) == 2
    assert interval_score(-10, 10, 1, .1) == 20
    assert interval_score(0, 2, 3, .1) == 22
    assert interval_score(0, 2, -1, .1) == 22
    with pytest.raises(ValueError):
        interval_score(2, 0, 1)


def test_mean_mcse_and_predictive_interval_are_distinct(evidence):
    result = evaluate(*evidence)
    row = result["endpoints"]["risk_difference"]
    assert row["mean"] == pytest.approx(.2)
    assert row["mcse_mean"] == pytest.approx(.1)
    assert row["predictive_interval"]["lower"] == pytest.approx(.11)
    assert row["predictive_interval"]["upper"] == pytest.approx(.29)
    assert row["crps"] == pytest.approx(.05)
    assert row["baseline_crps"] == pytest.approx(.2)
    assert row["crps_skill"] == pytest.approx(.75)
    assert result["numerical_fit"]["status"] == "PASS"
    assert result["independent_validation"]["status"] == "NOT_APPLICABLE"
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("fault", [
    "failed", "missing", "extra", "nan", "range", "bool",
    "duplicate_seed", "missing_seed", "failed_check",
])
def test_invalid_evidence_is_not_dropped_or_zero_filled(evidence, fault):
    case, submission, reference = evidence
    run = submission["runs"][1]
    if fault == "failed":
        run["status"] = "FAIL"
    elif fault == "missing":
        run["predictions"] = {}
    elif fault == "extra":
        run["predictions"]["unexpected"] = 0
    elif fault in {"nan", "range", "bool"}:
        run["predictions"]["risk_difference"] = {
            "nan": float("nan"), "range": 2, "bool": True}[fault]
    elif fault == "duplicate_seed":
        run["seed"] = 1
    elif fault == "missing_seed":
        submission["runs"].pop()
    else:
        run["checks"][0]["status"] = "FAIL"
    result = evaluate(case, submission, reference)
    assert result["status"] == "FAIL"
    assert result["endpoints"] == {}
    assert result["n_submitted_runs"] == len(submission["runs"])
    assert len(result["retained_runs"]) == len(submission["runs"])
    json.dumps(result, allow_nan=False)


def test_missing_reference_and_wrong_identity_fail(evidence):
    case, submission, reference = evidence
    reference["targets"] = {}
    reference["case_id"] = "another-case"
    result = evaluate(case, submission, reference)
    assert result["status"] == "FAIL"
    assert result["numerical_fit"]["status"] == "NOT_RUN"


def test_zero_baseline_skill_is_undefined(evidence):
    for run in evidence[1]["runs"]:
        run["baselines"]["risk_difference"] = .2
    result = evaluate(*evidence)
    assert result["endpoints"]["risk_difference"]["crps_skill"] is None
    assert result["numerical_fit"]["status"] == "INCONCLUSIVE"


def test_secondary_baselines_optional_but_unknown_keys_fail(evidence):
    case, submission, reference = evidence
    case["endpoints"].append({"id": "control_risk", "units": "probability",
                              "lower_bound": 0, "upper_bound": 1})
    reference["targets"]["control_risk"] = {"value": .3, "source": "paper"}
    for run in submission["runs"]:
        run["predictions"]["control_risk"] = .3
    result = evaluate(*evidence)
    assert result["numerical_fit"]["status"] == "PASS"
    assert result["endpoints"]["control_risk"]["crps_skill"] is None
    submission["runs"][0]["baselines"]["unknown"] = 0
    assert evaluate(*evidence)["status"] == "FAIL"


@pytest.mark.parametrize("kind", ["assumed", "development"])
def test_assumed_or_development_inputs_are_inconclusive(evidence, kind):
    evidence[0]["sources"]["independent"]["kind"] = kind
    result = evaluate(*evidence)
    assert result["input_adequacy"]["status"] == "INCONCLUSIVE"
    assert result["numerical_fit"]["status"] == "PASS"


def test_transitive_target_leakage_and_unused_source(evidence):
    case, submission, reference = evidence
    case["sources"]["leaked"] = {
        "kind": "target_outcome", "citation": "paper", "available_at": "2020-01-01",
        "target_ids": ["risk_difference"], "derived_from": [],
    }
    assert evaluate(*evidence)["input_adequacy"]["status"] == "PASS"
    case["sources"]["independent"]["derived_from"] = ["leaked"]
    result = evaluate(*evidence)
    assert result["input_adequacy"]["status"] == "FAIL"
    assert result["status"] == "FAIL"
    case["information_track"] = "conditional"
    case["released_targets"] = ["risk_difference"]
    assert evaluate(*evidence)["input_adequacy"]["status"] == "FAIL"
    case["sources"]["leaked"]["target_ids"] = ["historical_control"]
    case["released_targets"] = ["historical_control"]
    assert evaluate(*evidence)["input_adequacy"]["status"] == "PASS"


def test_unseen_declaration_does_not_certify_blinding(evidence):
    evidence[0]["exposure"] = "declared_unseen"
    result = evaluate(*evidence)
    assert result["blinding"]["status"] == "INCONCLUSIVE"
    assert result["independent_validation"]["status"] == "INCONCLUSIVE"


def test_missing_sources_and_cycles_visible(evidence):
    case = evidence[0]
    case["sources"]["independent"]["derived_from"] = ["missing"]
    assert evaluate(*evidence)["input_adequacy"]["status"] == "INCONCLUSIVE"
    case["sources"]["independent"]["derived_from"] = ["independent"]
    assert evaluate(*evidence)["input_adequacy"]["status"] == "FAIL"


def test_preregistered_failure_not_rescued_by_interval_overlap(evidence):
    evidence[0]["scoring"]["minimum_skill"] = .9
    result = evaluate(*evidence)
    interval = result["endpoints"]["risk_difference"]["predictive_interval"]
    assert interval["contains_reference"]
    assert result["numerical_fit"]["status"] == "FAIL"


def test_render_preserves_notes_and_exact_identity(evidence, tmp_path):
    evaluation = evaluate(*evidence)
    paths = render_report(tmp_path, *evidence, evaluation)
    assert len([key for key in paths if key.startswith("review_")]) == 5
    human = tmp_path / "human_review.json"
    first = json.loads(human.read_text())
    assert all(role["status"] == "NOT_RUN" for role in first["roles"].values())
    first["roles"]["clinical"]["notes"] = "Human observation: retain exactly."
    human.write_text(json.dumps(first, indent=3))
    before = human.read_bytes()
    modified = deepcopy(evidence)
    modified[1]["identity"] = {"source_sha256": "new-content"}
    render_report(tmp_path, *modified, evaluate(*modified))
    assert human.read_bytes() == before
    overview = (tmp_path / "review_packet.md").read_text()
    assert "re-review is required" in overview
    assert "new-content" in overview
    assert "Evaluator-only" in overview


def test_renderer_retains_invalid_numeric_evidence(evidence, tmp_path):
    evidence[1]["runs"][0]["predictions"]["risk_difference"] = float("nan")
    render_report(tmp_path, *evidence, evaluate(*evidence))
    records = json.loads((tmp_path / "source_records.json").read_text())
    invalid = records["submission"]["runs"][0]["predictions"]["risk_difference"]
    assert invalid == {"invalid_numeric": "nan"}


def test_report_summarizes_large_identity_maps(evidence, tmp_path):
    submission = evidence[1]
    submission["identity"] = {
        "session_id": "session-example", "case_sha256": "case-digest",
        "source_hashes": {"module.py": "source-digest"},
    }
    submission["execution"] = {
        "status": "PASS", "mode": "os-isolated", "self_test": {"status": "PASS"},
        "environment": {"python": "3.14.0 (build details)",
                        "dependencies": {"numpy": "2.3.0"},
                        "dependency_hashes": {f"large-file-{i}": "x" * 64
                                              for i in range(2000)}},
    }
    result = evaluate(*evidence)
    result["protocol_integrity"] = "PASS"
    result["checks"] = [
        {"name": f"check-{i}", "status": "PASS", "detail": "passed"}
        for i in range(3300)
    ]
    result["retained_runs"] = [
        {"index": i, "seed": i, "status": "PASS"} for i in range(256)
    ]
    render_report(tmp_path, *evidence, result)
    for path in tmp_path.glob("*.md"):
        content = path.read_text()
        assert "large-file-" not in content
        assert "session-example" in content
        assert "Python 3.14.0; NumPy 2.3.0" in content
        assert "| protocol_integrity | PASS |" in content
        assert "source_records.json" in content
        assert len(content) < 16000
    overview = (tmp_path / "review_packet.md").read_text()
    assert "0.2000" in overview and "[0.1100, 0.2900]" in overview
    assert "3300 checks recorded: PASS 3300" in overview
    assert "first 25 in submission order" in overview
    assert "| 255 |" not in overview
    saved = json.loads((tmp_path / "evaluation.json").read_text())
    assert len(saved["checks"]) == 3300
    assert len(saved["retained_runs"]) == 256
    records = json.loads((tmp_path / "source_records.json").read_text())
    assert len(records["submission"]["execution"]["environment"]
               ["dependency_hashes"]) == 2000
