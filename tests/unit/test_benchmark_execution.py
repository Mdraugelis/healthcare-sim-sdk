"""OS boundary orchestration: real enforcement is exercised by a macOS smoke."""
import json
import os
from pathlib import Path
import subprocess

import pytest

from healthcare_sim_sdk.experiments.benchmark import execution as ex


def _case():
    return {"run_plan": {"adapter": "shield_rt_v1", "seeds": [13, 17]},
            "inputs": {}}


def test_no_backend_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "_sandbox_backend", lambda: None)
    with pytest.raises(ex.IsolationError, match="required"):
        ex.run_isolated(_case(), tmp_path)


def test_bundle_is_an_explicit_allowlist(tmp_path, monkeypatch):
    root, bundle = tmp_path / "repo", tmp_path / "bundle"
    for name in (*ex.BUNDLE_SOURCES, ex.EXECUTION_SOURCE):
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# test source\n")
    hidden = root / "research/reference.json"
    hidden.parent.mkdir()
    hidden.write_text('{"secret":123}')
    monkeypatch.setattr(ex, "_repository_root", lambda: root)
    hashes = ex._stage_bundle(bundle)
    files = {str(p.relative_to(bundle)) for p in bundle.rglob("*") if p.is_file()}
    assert files == set(ex.BUNDLE_SOURCES) | set(ex.SYNTHETIC_INITIALIZERS)
    assert not (bundle / "research").exists()
    assert set(hashes) == files
    for name in ex.SYNTHETIC_INITIALIZERS:
        assert (bundle / name).read_bytes() == ex.INITIALIZER_BYTES


@pytest.mark.parametrize("update", [
    {"run_plan": {"adapter": "arbitrary_module", "seeds": [1]}},
    {"run_plan": {"adapter": "shield_rt_v1", "seeds": [1, 1]}},
    {"reference": {"outcome": .2}},
])
def test_invalid_plan_or_reference_rejected(tmp_path, update):
    case = _case() | update
    with pytest.raises(ValueError):
        ex.run_isolated(case, tmp_path)


def _fake_worker(command, **kwargs):
    assert "-I" in command and "-S" in command
    bundle = Path(kwargs["cwd"])
    writable = Path(kwargs["env"]["TMPDIR"])
    assert "PYTHONPATH" not in kwargs["env"]
    hashes = json.loads((bundle / "bundle_manifest.json").read_text())
    case = json.loads((bundle / "case.json").read_text())
    result = {"self_test": {"status": "PASS", "outside_read_denied": True,
                            "outside_write_denied": True, "network_denied": True},
              "bundle_hashes": hashes,
              "runs": [{"seed": seed, "status": "PASS", "predictions": {}}
                       for seed in case["run_plan"]["seeds"]]}
    (writable / "worker_result.json").write_text(json.dumps(result))
    return subprocess.CompletedProcess(command, 0)


def test_records_identity_and_every_seed(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "_sandbox_backend", lambda: "/mock/sandbox-exec")
    monkeypatch.setattr(ex.subprocess, "run", _fake_worker)
    result = ex.run_isolated(_case(), tmp_path)
    assert result["execution"]["status"] == "PASS"
    assert [r["seed"] for r in result["runs"]] == [13, 17]
    assert result["execution"]["source_hashes"] == ex.source_manifest()
    assert result["execution"]["environment"] == ex.runtime_manifest()


def test_timeout_retains_completed_and_marks_remaining(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "_sandbox_backend", lambda: "/mock/sandbox-exec")

    def timeout(command, **kwargs):
        _fake_worker(command, **kwargs)
        p = Path(kwargs["env"]["TMPDIR"]) / "worker_result.json"
        data = json.loads(p.read_text())
        data["runs"] = data["runs"][:1]
        p.write_text(json.dumps(data))
        raise subprocess.TimeoutExpired(command, 1)
    monkeypatch.setattr(ex.subprocess, "run", timeout)
    result = ex.run_isolated(_case(), tmp_path)
    assert result["runs"][0]["status"] == "PASS"
    assert result["runs"][1]["status"] == "FAIL"
    assert result["execution"]["status"] == "FAIL"


def test_probe_failure_invalidates_worker_claims(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "_sandbox_backend", lambda: "/mock/sandbox-exec")

    def bad_probe(command, **kwargs):
        result = _fake_worker(command, **kwargs)
        p = Path(kwargs["env"]["TMPDIR"]) / "worker_result.json"
        data = json.loads(p.read_text())
        data["self_test"]["status"] = "FAIL"
        p.write_text(json.dumps(data))
        return result
    monkeypatch.setattr(ex.subprocess, "run", bad_probe)
    result = ex.run_isolated(_case(), tmp_path)
    assert result["execution"]["status"] == "FAIL"
    assert all(r["status"] == "FAIL" for r in result["runs"])


def test_development_opt_out_never_claims_isolation(tmp_path, monkeypatch):
    monkeypatch.setattr(ex.subprocess, "run", _fake_worker)
    result = ex.run_isolated(_case(), tmp_path, require_isolation=False)
    assert result["execution"]["mode"] == "unisolated_development"
    assert result["execution"]["status"] == "NOT_RUN"


def test_profile_does_not_grant_repo_or_home_data_access(tmp_path):
    env = ex.runtime_manifest()
    profile = ex._macos_profile(tmp_path / "bundle", tmp_path / "write", env)
    assert '(deny network*)' in profile
    assert f'(subpath "{ex._repository_root()}")' not in profile
    assert f'(subpath "{Path.home()}")' not in profile
    assert '(subpath "/Users")' not in profile
    assert '(subpath "/")' not in profile
    for path in env["dependency_paths"]:
        assert f'(subpath "{path}")' not in profile
        assert f'(literal "{path}")' in profile
    assert '(allow file-write* (subpath' in profile


def test_source_symlink_cannot_stage_outside_content(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    source = root / ex.BUNDLE_SOURCES[0]
    source.parent.mkdir(parents=True)
    secret = tmp_path / "reference.json"
    secret.write_text('{"private":true}')
    source.symlink_to(secret)
    monkeypatch.setattr(ex, "_repository_root", lambda: root)
    with pytest.raises(ex.IsolationError, match="symlink"):
        ex._stage_bundle(tmp_path / "bundle")


def test_claimed_pass_without_denial_evidence_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "_sandbox_backend", lambda: "/mock/sandbox-exec")

    def incomplete(command, **kwargs):
        result = _fake_worker(command, **kwargs)
        path = Path(kwargs["env"]["TMPDIR"]) / "worker_result.json"
        data = json.loads(path.read_text())
        data["self_test"] = {"status": "PASS"}
        path.write_text(json.dumps(data))
        return result

    monkeypatch.setattr(ex.subprocess, "run", incomplete)
    result = ex.run_isolated(_case(), tmp_path)
    assert result["execution"]["status"] == "FAIL"
    assert all(record["status"] == "FAIL" for record in result["runs"])


def test_changed_dependency_bytes_fail_runtime_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "_sandbox_backend", lambda: "/mock/sandbox-exec")
    environment = ex.runtime_manifest()
    environments = iter([environment, environment | {"dependency_hashes": {}}])
    monkeypatch.setattr(ex, "runtime_manifest", lambda: next(environments))
    monkeypatch.setattr(ex.subprocess, "run", _fake_worker)
    result = ex.run_isolated(_case(), tmp_path)
    assert result["execution"]["status"] == "FAIL"
    assert result["execution"]["environment_stable"] is False


def test_nonfinite_worker_payload_is_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "_sandbox_backend", lambda: "/mock/sandbox-exec")

    def nonfinite(command, **kwargs):
        path = Path(kwargs["env"]["TMPDIR"]) / "worker_result.json"
        path.write_text('{"runs": [], "extra": NaN}')
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(ex.subprocess, "run", nonfinite)
    result = ex.run_isolated(_case(), tmp_path)
    assert result["execution"]["status"] == "FAIL"
    assert "Nonfinite" in result["execution"]["error"]
    assert [r["seed"] for r in result["runs"]] == [13, 17]


def test_runtime_hashes_package_bytes_but_not_caches(tmp_path, monkeypatch):
    site = tmp_path / "site-packages"
    package = site / "numpy"
    package.mkdir(parents=True)
    source = package / "__init__.py"
    source.write_text("version = 1\n")
    cache = package / "__pycache__/__init__.pyc"
    cache.parent.mkdir()
    cache.write_bytes(b"cache one")
    original = ex.sysconfig.get_path
    monkeypatch.setattr(ex.sysconfig, "get_path", lambda key: str(site)
                        if key in ("purelib", "platlib") else original(key))
    before = ex.runtime_manifest()
    assert str(source) in before["dependency_hashes"]
    assert str(cache) not in before["dependency_hashes"]
    source.write_text("version = 2\n")
    after = ex.runtime_manifest()
    assert before["dependency_hashes"] != after["dependency_hashes"]


def test_real_worker_preserves_adapter_errors_per_seed(tmp_path):
    result = ex.run_isolated(_case(), tmp_path, require_isolation=False)
    assert [r["seed"] for r in result["runs"]] == [13, 17]
    assert all(r["status"] == "FAIL" for r in result["runs"])
    assert all("ValueError" in r["error"] for r in result["runs"])


@pytest.mark.skipif(
    os.environ.get("SDK_BENCHMARK_OS_SMOKE") != "1",
    reason="Opt-in real OS sandbox test; nested host sandboxes may deny launch",
)
def test_real_macos_boundary_and_adapter(tmp_path):
    if ex._sandbox_backend() is None:
        pytest.skip("macOS sandbox-exec unavailable")
    case = _case()
    for name, value in (
        ("n_control", 50), ("n_intervention", 50),
        ("baseline_risk", {"distribution": "fixed", "value": .2}),
        ("relative_effect", {"distribution": "fixed", "value": .3}),
        ("uptake", {"distribution": "fixed", "value": 1.0}),
    ):
        case["inputs"][name] = {"value": value, "source_id": "synthetic"}
    result = ex.run_isolated(case, tmp_path)
    assert result["execution"]["status"] == "PASS", result["execution"]
    assert result["execution"]["self_test"] == {
        "status": "PASS", "outside_read_denied": True,
        "outside_write_denied": True, "network_denied": True,
    }
    assert all(record["status"] == "PASS" for record in result["runs"])
