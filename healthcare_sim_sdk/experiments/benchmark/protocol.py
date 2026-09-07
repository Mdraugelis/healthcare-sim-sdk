"""Custodian-controlled freeze, submission, reveal and scoring transitions.

The SDK execution process receives no vault path or key. Custodian HMAC
receipts detect rewriting by a runner without that key; they are not public
signatures or protection from a dishonest custodian/account administrator.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import hmac
import os
from pathlib import Path
import secrets

from .contracts import (canonical, digest, finite, read_json, validate_case,
                        validate_reference)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write(path, value):
    """Create immutable-by-protocol content; identical crash retries are safe."""
    path = Path(path)
    data = canonical(value) + "\n"
    try:
        with path.open("x") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        path.chmod(0o400)
    except FileExistsError:
        if path.read_text() != data:
            raise ValueError(f"Refusing to replace existing artifact: {path.name}")


def _atomic(path, value):
    tmp = path.with_name(path.name + "." + secrets.token_hex(6) + ".tmp")
    with tmp.open("x") as handle:
        handle.write(canonical(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.chmod(0o600)
    os.replace(tmp, path)


@contextmanager
def _lock(folder):
    lock = Path(folder) / ".transition.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ValueError("Another transition is active; inspect stale locks") from exc
    try:
        os.close(fd)
        yield
    finally:
        lock.unlink()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluator_manifest():
    return {p.name: _sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


def seal_reference(reference, vault_dir):
    """Custodian command. Store references outside the SDK repository."""
    reference = validate_reference(reference)
    vault = Path(vault_dir).resolve()
    repo = Path(__file__).resolve().parents[3]
    if vault.is_relative_to(repo):
        raise ValueError("Reference vault must be outside the SDK repository")
    _check_vault_location(vault)
    vault.mkdir(mode=0o700, parents=True, exist_ok=False)
    vault.chmod(0o700)
    nonce = secrets.token_hex(32)
    sealed = {"reference": reference, "nonce": nonce}
    commitment = digest(sealed)
    _write(vault / "sealed_reference.json", sealed)
    _write(vault / "custodian_key.json", {"key": secrets.token_hex(32)})
    (vault / "sessions").mkdir(mode=0o700)
    _write(vault / "commitment.json", {
        "reference_commitment": commitment, "sealed_at": _now(),
        "case_id": reference["case_id"], "study_id": reference["study_id"],
    })
    return read_json(vault / "commitment.json")


def _vault(vault):
    vault = Path(vault).resolve()
    sealed = read_json(vault / "sealed_reference.json")
    commitment = read_json(vault / "commitment.json")
    if digest(sealed) != commitment["reference_commitment"]:
        raise ValueError("Sealed reference changed")
    return vault, sealed, commitment


def _check_vault_location(vault):
    from .execution import runtime_manifest
    environment = runtime_manifest()
    readable = [environment["base_prefix"], "/System/Library", "/usr/lib",
                "/usr/share/locale", "/usr/share/zoneinfo"]
    readable.extend(environment.get("dependency_read_roots", []))
    # A vault placed inside a trusted runtime directory would defeat the
    # worker's intended read boundary even though it is outside the repository.
    if any(vault.is_relative_to(Path(root).resolve()) for root in readable):
        raise ValueError("Reference vault must be outside worker-readable runtimes")


def _mac(vault, value):
    key = bytes.fromhex(read_json(vault / "custodian_key.json")["key"])
    return hmac.new(key, canonical(value).encode(), hashlib.sha256).hexdigest()


def _record(vault, session):
    sid = read_json(session / "session.json")["session_id"]
    if not isinstance(sid, str) or len(sid) != 32:
        raise ValueError("Invalid session id")
    if any(c not in "0123456789abcdef" for c in sid):
        raise ValueError("Invalid session id")
    record_path = vault / "sessions" / (sid + ".json")
    record = read_json(record_path)
    previous = None
    for index, event in enumerate(record["events"]):
        body = {k: v for k, v in event.items() if k != "mac"}
        if (event["sequence"] != index or event["previous"] != previous
                or not hmac.compare_digest(event["mac"], _mac(vault, body))):
            raise ValueError("Custodian receipt history was altered")
        previous = digest(event)
    if not record["events"]:
        raise ValueError("Empty custodian history")
    return record_path, record


def _append(vault, path, record, state, artifacts):
    events = record["events"]
    body = {"sequence": len(events), "state": state, "timestamp": _now(),
            "previous": digest(events[-1]) if events else None,
            "artifacts": artifacts}
    event = {**body, "mac": _mac(vault, body)}
    _atomic(path, {"events": [*events, event]})
    return event


def _artifacts(session, names):
    return {name: _sha(session / name) for name in names}


def _expected_artifacts(session, objects):
    result = {}
    for name, value in objects.items():
        expected = hashlib.sha256((canonical(value) + "\n").encode()).hexdigest()
        path = session / name
        if path.is_symlink() or _sha(path) != expected:
            raise ValueError(f"Artifact changed before custody receipt: {name}")
        result[name] = expected
    return result


def _read_pinned(session, vault, name):
    _, record = _record(vault, session)
    expected = None
    for event in record["events"]:
        expected = event["artifacts"].get(name, expected)
    value = read_json(session / name)
    actual = hashlib.sha256((canonical(value) + "\n").encode()).hexdigest()
    if expected is None or not hmac.compare_digest(actual, expected):
        raise ValueError(f"Read does not match custodian receipt: {name}")
    return value


def verify_session(session_dir, vault_dir):
    """Verify against the custodian's record, not runner-owned manifests."""
    session = Path(session_dir).resolve()
    vault, _, commitment = _vault(vault_dir)
    _, record = _record(vault, session)
    for event in record["events"]:
        for name, expected in event["artifacts"].items():
            path = session / name
            if (not path.is_file() or path.is_symlink()
                    or _sha(path) != expected):
                raise ValueError(f"Frozen artifact changed or missing: {name}")
    manifest = read_json(session / "frozen_manifest.json")
    if manifest["reference_commitment"] != commitment["reference_commitment"]:
        raise ValueError("Reference commitment mismatch")
    return {"status": "PASS", "state": record["events"][-1]["state"],
            "last_receipt": record["events"][-1], "manifest": manifest}


def freeze_case(case, session_dir, vault_dir):
    """Custodian command: bind inputs, code, scoring and reference before run."""
    from .execution import runtime_manifest, source_manifest
    validate_case(case)
    vault, sealed, commitment = _vault(vault_dir)
    _check_vault_location(vault)
    validate_reference(sealed["reference"], case)
    session = Path(session_dir).resolve()
    if session.is_relative_to(vault) or vault.is_relative_to(session):
        raise ValueError("Runner session and reference vault must be separate")
    if list((vault / "sessions").glob("*.json")):
        raise ValueError("Reference vault is single-use; preserve its first evaluation")
    session.mkdir(parents=True, exist_ok=False)
    sid = secrets.token_hex(16)
    manifest = {
        "schema_version": 1, "session_id": sid, "frozen_at": _now(),
        "case_sha256": digest(case), "source_hashes": source_manifest(),
        "environment": runtime_manifest(),
        "evaluator_hashes": evaluator_manifest(),
        "reference_commitment": commitment["reference_commitment"],
        "expected_seeds": case["run_plan"]["seeds"],
        "exposure": case["exposure"],
    }
    session_identity = {"session_id": sid}
    _write(session / "session.json", session_identity)
    _write(session / "frozen_case.json", case)
    _write(session / "frozen_manifest.json", manifest)
    with _lock(vault):
        if list((vault / "sessions").glob("*.json")):
            raise ValueError("Reference vault has already registered a session")
        receipt = _append(vault, vault / "sessions" / (sid + ".json"),
                          {"events": []}, "INPUTS_FROZEN",
                          _expected_artifacts(session, {
                              "session.json": session_identity,
                              "frozen_case.json": case,
                              "frozen_manifest.json": manifest}))
    _write(session / "freeze_receipt.json", receipt)
    return receipt


def execute(session_dir, *, require_isolation=True):
    """Unattested helper; the supervisor must certify its output for submission.

    This function neither accepts nor opens a vault. Public run commands use
    execute_registered so a submitter cannot invent self-attested run files.
    """
    from .execution import run_isolated, runtime_manifest, source_manifest
    session = Path(session_dir).resolve()
    with _lock(session):
        if (session / "run_result.json").exists():
            raise ValueError("Run already exists; preserve the original attempt")
        case = validate_case(read_json(session / "frozen_case.json"))
        manifest = read_json(session / "frozen_manifest.json")
        if (digest(case) != manifest["case_sha256"]
                or source_manifest() != manifest["source_hashes"]
                or runtime_manifest() != manifest["environment"]):
            raise ValueError("Code, inputs or runtime changed after freeze")
        result = run_isolated(case, session / "execution",
                              require_isolation=require_isolation)
        result["identity"] = manifest
        _write(session / "run_result.json", result)
        return result


def execute_registered(session_dir, vault_dir, *, require_isolation=True):
    """Trusted custodian supervisor; the child SDK gets no vault/key access."""
    session = Path(session_dir).resolve()
    vault, _, _ = _vault(vault_dir)
    with _lock(vault):
        status = verify_session(session, vault)
        if status["state"] != "INPUTS_FROZEN":
            raise ValueError("A registered execution can run only once")
        result = execute(session, require_isolation=require_isolation)
        # The supervisor receives the child output directly and pins it before
        # returning control to a submitter. Later edits cannot be certified.
        expected = hashlib.sha256((canonical(result) + "\n").encode()).hexdigest()
        if _sha(session / "run_result.json") != expected:
            raise ValueError("Worker result changed before custodian receipt")
        path, record = _record(vault, session)
        _append(vault, path, record, "EXECUTED",
                {"run_result.json": expected})
        return result


def _validate_runs(case, result):
    if not isinstance(result, dict) or not isinstance(result.get("runs"), list):
        raise ValueError("Submission must retain all run records")
    runs = result["runs"]
    expected = case["run_plan"]["seeds"]
    if ([r.get("seed") for r in runs] != expected
            or any(type(r.get("seed")) is not int for r in runs)):
        raise ValueError("Missing, reordered, duplicate or unexpected seeds")
    endpoints = {e["id"]: e for e in case["endpoints"]}
    for run in runs:
        if run.get("status") not in ("PASS", "FAIL"):
            raise ValueError("Run must record PASS or FAIL")
        if run["status"] == "FAIL":
            if not run.get("error") and not run.get("detail"):
                raise ValueError("Failed run must record its failure")
            continue
        if set(run.get("predictions", {})) != set(endpoints):
            raise ValueError("Prediction endpoint mismatch")
        for key, value in run["predictions"].items():
            endpoint = endpoints[key]
            if not endpoint["lower_bound"] <= finite(value) <= endpoint["upper_bound"]:
                raise ValueError("Prediction outside registered support")
        baseline = run.get("baselines", {})
        if case["scoring"]["primary_endpoint"] not in baseline:
            raise ValueError("Primary reference-model prediction is missing")
        for key, value in baseline.items():
            if key not in endpoints:
                raise ValueError("Undeclared baseline endpoint")
            lower, upper = endpoints[key]["lower_bound"], endpoints[key]["upper_bound"]
            if not lower <= finite(value) <= upper:
                raise ValueError("Baseline outside endpoint support")


def commit_submission(session_dir, vault_dir):
    """Custodian command. Commit every result before any outcome reveal."""
    session = Path(session_dir).resolve()
    vault, _, _ = _vault(vault_dir)
    with _lock(vault):
        status = verify_session(session, vault)
        if status["state"] != "EXECUTED":
            raise ValueError("Commit requires a custodian-supervised execution")
        result = _read_pinned(session, vault, "run_result.json")
        case = _read_pinned(session, vault, "frozen_case.json")
        _validate_runs(case, result)
        execution = result.get("execution", {})
        manifest = status["manifest"]
        if (execution.get("source_hashes") != manifest["source_hashes"]
                or execution.get("environment") != manifest["environment"]
                or result.get("identity") != manifest
                or evaluator_manifest() != manifest["evaluator_hashes"]):
            raise ValueError("Submission code/runtime identity does not match freeze")
        submission = {**result, "submission_sha256": digest(result)}
        _write(session / "submission.json", submission)
        path, record = _record(vault, session)
        receipt = _append(vault, path, record, "PREDICTIONS_FROZEN",
                          _expected_artifacts(session, {
                              "run_result.json": result,
                              "submission.json": submission}))
        _write(session / "submission_receipt.json", receipt)
        return receipt


def reveal(session_dir, vault_dir):
    """Custodian command: disclosure is forbidden before frozen submission."""
    session = Path(session_dir).resolve()
    vault, sealed, _ = _vault(vault_dir)
    with _lock(vault):
        status = verify_session(session, vault)
        if status["state"] != "PREDICTIONS_FROZEN":
            raise ValueError("Reveal requires an intact frozen prediction submission")
        reference = sealed["reference"]
        validate_reference(reference, read_json(session / "frozen_case.json"))
        _write(session / "revealed_reference.json", reference)
        # Nonce becomes public only at reveal, allowing verification of the
        # pre-run commitment without enabling brute-force guessing beforehand.
        proof = {"nonce": sealed["nonce"]}
        _write(session / "reveal_proof.json", proof)
        path, record = _record(vault, session)
        _append(vault, path, record, "REVEALED", _expected_artifacts(
            session, {"revealed_reference.json": reference,
                      "reveal_proof.json": proof}))
    return reference


def score(session_dir, vault_dir):
    """Evaluator command: score and render, preserving pending human review."""
    from .report import render_report
    from .scoring import evaluate
    session = Path(session_dir).resolve()
    vault, _, _ = _vault(vault_dir)
    with _lock(vault):
        status = verify_session(session, vault)
        if status["state"] != "REVEALED":
            raise ValueError("Scoring requires revealed, intact evidence")
        if evaluator_manifest() != status["manifest"]["evaluator_hashes"]:
            raise ValueError("Evaluator changed after preregistration")
        case = _read_pinned(session, vault, "frozen_case.json")
        submission = _read_pinned(session, vault, "submission.json")
        reference = _read_pinned(session, vault, "revealed_reference.json")
        evaluation = evaluate(case, submission, reference)
        evaluation["protocol_integrity"] = "PASS"
        evaluation["frozen_identity"] = status["manifest"]
        evaluation["submission_sha256"] = submission["submission_sha256"]
        _write(session / "evaluation.json", evaluation)
        render_report(session / "review", case, submission, reference, evaluation)
        names = ["evaluation.json"]
        names.extend(str(p.relative_to(session))
                     for p in sorted((session / "review").rglob("*"))
                     if p.is_file() and p.name != "human_review.json")
        path, record = _record(vault, session)
        _append(vault, path, record, "SCORED", _artifacts(session, names))
    return evaluation
