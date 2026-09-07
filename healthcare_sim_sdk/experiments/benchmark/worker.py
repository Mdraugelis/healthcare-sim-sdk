"""Standalone worker copied into an allowlisted bundle; no evaluator imports."""
from __future__ import annotations

import errno
import hashlib
import json
from pathlib import Path
import socket
import sys


def _write(path, result):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, sort_keys=True, allow_nan=False))
    temporary.replace(path)


def _self_test(sentinel):
    checks = {}
    try:
        sentinel.read_bytes()
        checks["outside_read_denied"] = False
    except OSError as exc:
        checks["outside_read_denied"] = exc.errno in (errno.EACCES, errno.EPERM)
    try:
        sentinel.with_name("write-probe").write_text("probe")
        checks["outside_write_denied"] = False
    except OSError as exc:
        checks["outside_write_denied"] = exc.errno in (errno.EACCES, errno.EPERM)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            sock.connect(("127.0.0.1", 9))
        checks["network_denied"] = False
    except OSError as exc:
        # Refused/unavailable networking does not prove a sandbox denial.
        checks["network_denied"] = exc.errno in (errno.EACCES, errno.EPERM)
    return {"status": "PASS" if all(checks.values()) else "FAIL", **checks}


def _failure(seed, exc):
    return {"seed": seed, "status": "FAIL", "predictions": {},
            "baselines": {}, "metrics": {}, "checks": [],
            "error": f"{type(exc).__name__}: {exc}"}


def main():
    bundle, output, sentinel = (Path(p).resolve() for p in sys.argv[1:4])
    dependencies = json.loads(sys.argv[4])
    isolated = sys.argv[5] == "isolated"
    # -I -S disables startup .pth/editable hooks. Add only explicit roots.
    sys.path[:0] = [str(bundle), *dependencies]
    result_path = output / "worker_result.json"
    result = {"runs": [], "self_test": _self_test(sentinel)}
    _write(result_path, result)
    if isolated and result["self_test"]["status"] != "PASS":
        return 2
    expected = json.loads((bundle / "bundle_manifest.json").read_text())
    actual = {name: hashlib.sha256((bundle / name).read_bytes()).hexdigest()
              for name in expected}
    if actual != expected:
        raise ValueError("Copied source does not match bundle manifest")
    result["bundle_hashes"] = actual
    case = json.loads((bundle / "case.json").read_text())
    if case["run_plan"]["adapter"] != "shield_rt_v1":
        raise ValueError("Unsupported static adapter")
    try:
        from healthcare_sim_sdk.experiments.benchmark.shield_rt import run_case
    except Exception as exc:
        result["runs"] = [_failure(seed, exc) for seed in case["run_plan"]["seeds"]]
        _write(result_path, result)
        return 1
    for seed in case["run_plan"]["seeds"]:
        try:
            record = run_case(case, seed)
            if not isinstance(record, dict) or record.get("seed") != seed:
                raise ValueError("Adapter must return the requested seed record")
            json.dumps(record, allow_nan=False)
        except Exception as exc:
            record = _failure(seed, exc)
        result["runs"].append(record)
        _write(result_path, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
