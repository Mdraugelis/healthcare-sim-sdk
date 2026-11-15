"""Fault injection for the validation gate, independent of scenario models."""

import json
import subprocess
import sys
from dataclasses import asdict

import pytest

from healthcare_sim_sdk.experiments.validate import (
    Check, CheckStatus, MetricRule, checks_passed, collect_validation,
    ensure_required_checks, format_appendix, load_json_object, main, validate_generic,
)


@pytest.fixture
def run_dir(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"seed": 7, "timestamp": "20260905_120000"}))
    (tmp_path / "metrics.json").write_text(json.dumps({"throughput": 150, "auc": 0.8}))
    return tmp_path


def test_legacy_check_and_serialization():
    check = Check("name", "description", 1, 1, "exact", True, "detail")
    assert check.status == CheckStatus.PASS and check.passed
    assert json.loads(json.dumps(asdict(check)))["status"] == "PASS"
    check = Check("name", "description", 1, None, "exact", True, status="NOT_RUN")
    assert check.status == CheckStatus.NOT_RUN and not check.passed


@pytest.mark.parametrize("status", ["FAIL", "NOT_RUN", "INCONCLUSIVE", "NOT_APPLICABLE"])
def test_unresolved_required_checks_block(status):
    check = Check("required", "", 1, None, "", status=status)
    assert not checks_passed([check])
    assert "BLOCKED" in format_appendix({}, [check])
    assert status in format_appendix({}, [check])


def test_not_applicable_requires_explanation():
    check = Check("no counterfactual", "", "", "", "", status="NOT_APPLICABLE",
                  applicability="single-trajectory design", rationale="No paired-effect claim is made")
    assert checks_passed([check])
    assert not check.passed
    assert not checks_passed([])
    assert not checks_passed([Check("optional", "", "", "", "", True, required=False)])


def test_missing_required_check_cannot_be_green(run_dir):
    checks = validate_generic(run_dir, required_checks=["Held-out deployed scoring fidelity"])
    missing = next(check for check in checks if check.name == "Held-out deployed scoring fidelity")
    assert missing.status == CheckStatus.NOT_RUN
    assert not checks_passed(checks)
    optional = Check("Held-out deployed scoring fidelity", "", "", "", "", status="NOT_RUN", required=False)
    assert not checks_passed(ensure_required_checks([optional], [optional.name]))


@pytest.mark.parametrize("raw", [
    '{"auc": NaN}', '{"auc": Infinity}', '{"auc": -Infinity}',
    '{"auc": 1e999}', '{"nested": [0, {"auc": NaN}]}',
    '[0.8]', 'null', '"metrics"', '{"auc": 0.9, "auc": 0.1}', '{broken', '{}',
])
def test_invalid_metric_artifacts_block_instead_of_crashing(run_dir, raw):
    (run_dir / "metrics.json").write_text(raw)
    assert not checks_passed(validate_generic(run_dir))
    assert main([str(run_dir)]) == 1


@pytest.mark.parametrize("raw", ['[]', 'null', '"config"', '{"seed": true, "timestamp": "t"}',
                                 '{"seed": -1, "timestamp": "t"}', '{"seed": 1, "timestamp": 42}'])
def test_invalid_config_schema_blocks(run_dir, raw):
    (run_dir / "config.json").write_text(raw)
    assert not checks_passed(validate_generic(run_dir))


def test_missing_files_show_dependent_checks_not_run(tmp_path):
    checks = validate_generic(tmp_path)
    assert not checks_passed(checks)
    assert next(c for c in checks if c.name == "Seed recorded").status == CheckStatus.NOT_RUN


def test_explicit_metric_bounds_and_nested_presence(run_dir):
    (run_dir / "metrics.json").write_text('{"event_rate": 150, "subgroup": {"sensitivity": 1.1}}')
    # No arbitrary rate bounds inferred from a name that may represent events/person-year.
    assert checks_passed(validate_generic(run_dir))
    rules = [MetricRule("subgroup.sensitivity", minimum=0, maximum=1), MetricRule("denominator")]
    checks = validate_generic(run_dir, metric_rules=rules)
    by_name = {check.name: check for check in checks}
    assert by_name["Metric: subgroup.sensitivity"].status == CheckStatus.FAIL
    assert by_name["Metric: denominator"].status == CheckStatus.NOT_RUN
    assert not checks_passed(checks)


def test_metric_boolean_is_not_numeric(run_dir):
    (run_dir / "metrics.json").write_text('{"sensitivity": true}')
    assert not checks_passed(validate_generic(run_dir, [MetricRule("sensitivity")]))


def test_optional_missing_metric_is_visible_without_blocking(run_dir):
    checks = validate_generic(run_dir, [MetricRule("optional", required=False)])
    assert checks[-1].status == CheckStatus.NOT_RUN
    assert checks_passed(checks)


def test_scenario_validator_exception_and_empty_result_block(run_dir):
    def broken(output_dir, config):
        raise RuntimeError("injected estimator failure")
    assert not checks_passed(collect_validation(run_dir, broken))
    assert not checks_passed(collect_validation(run_dir, lambda *_: []))


def test_invalid_config_does_not_execute_scenario_validator(run_dir):
    (run_dir / "config.json").write_text('[]')
    calls = []
    checks = collect_validation(run_dir, lambda *_: calls.append(True))
    assert not calls
    assert any(c.name == "Scenario validator" and c.status == CheckStatus.NOT_RUN for c in checks)


def test_scenario_required_evidence_is_enforced_after_collection(run_dir):
    def validator(*_):
        return [Check("oracle", "", 5, 5, "exact", True)]

    assert checks_passed(collect_validation(run_dir, validator, required_checks=["oracle"]))


def test_strict_json_public_helper_rejects_overflow(run_dir):
    path = run_dir / "bad.json"
    path.write_text('{"value": 1e999}')
    with pytest.raises(ValueError, match="Nonfinite"):
        load_json_object(path)


def test_module_cli_exit_codes(run_dir):
    command = [sys.executable, "-m", "healthcare_sim_sdk.experiments.validate", str(run_dir)]
    assert subprocess.run(command, capture_output=True).returncode == 0
    missing = subprocess.run(command + ["--require-check", "analysis calibration"],
                             capture_output=True, text=True)
    assert missing.returncode == 1 and "NOT_RUN" in missing.stdout
    (run_dir / "metrics.json").write_text('{"auc": NaN}')
    invalid = subprocess.run(command, capture_output=True, text=True)
    assert invalid.returncode == 1 and "BLOCKED" in invalid.stdout


def test_cli_metric_rule_file(run_dir):
    path = run_dir / "rules.json"
    path.write_text('{"rules": [{"path": "auc", "minimum": 0.9, "maximum": 1}]}')
    assert main([str(run_dir), "--metric-rules", str(path)]) == 1


def test_markdown_escapes_table_contents():
    text = format_appendix({}, [Check("A|B", "", "a\nb", "1|2", "", True)])
    assert "A\\|B" in text and "a<br>b" in text and "1\\|2" in text
