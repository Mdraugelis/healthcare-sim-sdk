"""Minimal copied benchmark worker, with fail-closed OS isolation.

The sandbox protects the process boundary, not an analyst's prior knowledge.
Only already-curated case inputs may be supplied. Published reference evidence
has no argument or mount in this API. An unisolated development run requires an
explicit opt-out and never reports a passing isolation status.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import sysconfig
import tempfile

BUNDLE_SOURCES = (
    "healthcare_sim_sdk/__init__.py",
    "healthcare_sim_sdk/core/engine.py",
    "healthcare_sim_sdk/core/scenario.py",
    "healthcare_sim_sdk/core/rng.py",
    "healthcare_sim_sdk/core/results.py",
    "healthcare_sim_sdk/experiments/benchmark/shield_rt.py",
    "healthcare_sim_sdk/experiments/benchmark/worker.py",
)
EXECUTION_SOURCE = "healthcare_sim_sdk/experiments/benchmark/execution.py"
SYNTHETIC_INITIALIZERS = (
    "healthcare_sim_sdk/core/__init__.py",
    "healthcare_sim_sdk/experiments/__init__.py",
    "healthcare_sim_sdk/experiments/benchmark/__init__.py",
)
INITIALIZER_BYTES = b"# Minimal isolated benchmark package.\n"
MAX_TIMEOUT_SECONDS = 180


class IsolationError(RuntimeError):
    """The required process isolation could not be established."""


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_manifest() -> dict:
    """Identity of the actual allowlisted source and sandbox implementation."""
    root = _repository_root()
    return {name: _hash(root / name)
            for name in (*BUNDLE_SOURCES, EXECUTION_SOURCE)}


def runtime_manifest() -> dict:
    """Stable runtime description for freezing and comparing submissions."""
    dependencies: dict[str, str | None] = {}
    for name in ("numpy",):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    executable = Path(sys.executable).resolve()
    framework_binary = (Path(sys.base_prefix)
                        / "Resources/Python.app/Contents/MacOS/Python")
    worker_executable = (framework_binary.resolve()
                         if sys.platform == "darwin" and framework_binary.is_file()
                         else executable)
    dependency_paths = sorted({sysconfig.get_path("purelib"),
                               sysconfig.get_path("platlib")})
    dependency_roots = []
    for parent in dependency_paths:
        for name in ("numpy", "numpy.libs"):
            candidate = Path(parent) / name
            if not candidate.is_dir():
                continue
            if not candidate.resolve().is_relative_to(Path(parent).resolve()):
                raise IsolationError("Dependency root symlink escapes site-packages")
            dependency_roots.append(str(candidate.resolve()))
    dependency_roots = sorted(set(dependency_roots))
    for directory in dependency_roots:
        if any(path.is_symlink() and not path.resolve().is_relative_to(directory)
               for path in Path(directory).rglob("*")):
            raise IsolationError("Dependency symlink escapes allowlisted package")
    dependency_hashes = {
        str(path): _hash(path)
        for directory in dependency_roots
        for path in sorted(Path(directory).rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
        and path.suffix not in (".pyc", ".pyo")
    }
    return {
        "python": sys.version,
        "executable": str(executable),
        "executable_sha256": _hash(executable),
        "worker_executable": str(worker_executable),
        "worker_executable_sha256": _hash(worker_executable),
        "platform": {"system": platform.system(), "release": platform.release(),
                     "machine": platform.machine()},
        "dependencies": dependencies,
        "dependency_paths": dependency_paths,
        "dependency_read_roots": dependency_roots,
        "dependency_hashes": dependency_hashes,
        "root_directory_scope": "Literal directory handle; no recursive data grant",
        "base_prefix": str(Path(sys.base_prefix).resolve()),
        "synthetic_initializers": {
            name: hashlib.sha256(INITIALIZER_BYTES).hexdigest()
            for name in SYNTHETIC_INITIALIZERS
        },
    }


def _stage_bundle(destination: Path) -> dict:
    root = _repository_root()
    for name in BUNDLE_SOURCES:
        source = root / name
        if not source.resolve().is_relative_to(root.resolve()):
            raise IsolationError(f"Source symlink escapes repository: {name}")
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    for name in SYNTHETIC_INITIALIZERS:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(INITIALIZER_BYTES)
    return {name: _hash(destination / name)
            for name in (*BUNDLE_SOURCES, *SYNTHETIC_INITIALIZERS)}


def _sandbox_backend():
    if sys.platform == "darwin" and Path("/usr/bin/sandbox-exec").is_file():
        return "/usr/bin/sandbox-exec"
    return None


def _macos_profile(bundle: Path, writable: Path, environment: dict) -> str:
    # Grant data access to exact runtime roots, never their home/repo parents.
    runtime_roots = {
        bundle.resolve(), writable.resolve(),
        Path(environment["base_prefix"]),
        Path(sys.base_prefix),
        Path("/System/Library"), Path("/usr/lib"),
        Path("/System/Volumes/Preboot/Cryptexes/OS/System/Library/dyld"),
        Path("/usr/share/locale"), Path("/usr/share/zoneinfo"),
        *(Path(p).resolve() for p in environment["dependency_read_roots"]),
    }
    literals = {
        # dyld opens the root directory as a traversal handle. This literal
        # grants root-directory listing, never data beneath its children.
        Path("/"),
        Path(environment["worker_executable"]), Path("/dev/null"),
        Path("/dev/random"), Path("/dev/urandom"),
        Path("/private/etc/localtime"), Path("/etc/localtime"),
        # Import discovery lists the containing directory, but reads only the
        # NumPy package roots granted above; other installed code is denied.
        *(Path(p).resolve() for p in environment["dependency_paths"]),
    }

    def quote(path):
        return json.dumps(str(path))
    reads = " ".join(f"(subpath {quote(p)})" for p in sorted(runtime_roots))
    reads += " " + " ".join(f"(literal {quote(p)})" for p in sorted(literals))
    # Traversal metadata permits stat(), not reading directory listings/content.
    parents = {p for root in runtime_roots | literals for p in root.parents}
    metadata = " ".join(f"(literal {quote(p)})" for p in sorted(parents))
    return "\n".join([
        "(version 1)", "(deny default)", "(deny network*)",
        "(allow sysctl-read)",
        f"(allow process-exec (literal {quote(environment['worker_executable'])}))",
        f"(allow file-read* {reads})",
        f"(allow file-read-metadata {metadata})",
        f"(allow file-write* (subpath {quote(writable.resolve())}))",
    ])


def _failure(seed, message):
    return {"seed": seed, "status": "FAIL", "predictions": {},
            "baselines": {}, "metrics": {}, "checks": [], "error": message}


def _strict_object(path: Path) -> dict:
    def reject(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")
    result = json.loads(path.read_text(), parse_constant=reject)
    if not isinstance(result, dict):
        raise ValueError("Worker result must be an object")
    return result


def run_isolated(case: dict, output_dir: Path, *, require_isolation=True) -> dict:
    """Run a frozen case in a fresh worker, retaining every declared seed.

    A missing backend, denied sandbox launch, failed isolation probe, or changed
    source is an explicit failure. No silent development fallback is performed.
    Timeout is one bounded budget for this batch; completed records survive and
    uncompleted seeds receive explicit failures. The final JSON contains no
    reference-study outcome payload.
    """
    if not isinstance(case, dict):
        raise ValueError("case must be a dictionary")
    forbidden = {"reference", "sealed_reference", "reference_outcomes",
                 "observed_outcomes"} & case.keys()
    if forbidden:
        raise ValueError(f"Reference evidence is forbidden: {sorted(forbidden)}")
    plan = case.get("run_plan", {})
    if plan.get("adapter") != "shield_rt_v1":
        raise ValueError("Only the static shield_rt_v1 adapter is supported")
    seeds = plan.get("seeds")
    if (not isinstance(seeds, list) or not seeds
            or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("seeds must be nonempty, unique nonnegative integers")
    case_bytes = json.dumps(case, sort_keys=True, allow_nan=False).encode()
    backend = _sandbox_backend() if require_isolation else None
    if require_isolation and backend is None:
        raise IsolationError("No supported OS sandbox; isolated execution required")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sources, environment = source_manifest(), runtime_manifest()
    mode = "macos_sandbox_exec" if backend else "unisolated_development"
    with tempfile.TemporaryDirectory(prefix="sdk-benchmark-") as staging:
        stage = Path(staging).resolve()
        bundle = stage / "bundle"
        bundle.mkdir()
        bundle_hashes = _stage_bundle(bundle)
        # Commit actual copied bytes, not a second independent source read.
        if any(bundle_hashes[n] != sources[n] for n in BUNDLE_SOURCES):
            raise IsolationError("Source changed while staging the bundle")
        (bundle / "case.json").write_bytes(case_bytes)
        (bundle / "bundle_manifest.json").write_text(json.dumps(bundle_hashes))
        forbidden_dir = stage / "forbidden"
        forbidden_dir.mkdir()
        sentinel = forbidden_dir / "sentinel.txt"
        sentinel.write_text("isolation probe: no study information")
        with tempfile.TemporaryDirectory(prefix=".worker-", dir=output_dir) as work:
            writable = Path(work)
            command = [environment["worker_executable"], "-I", "-S",
                       str(bundle / BUNDLE_SOURCES[-1]), str(bundle),
                       str(writable), str(sentinel),
                       json.dumps(environment["dependency_paths"]),
                       "isolated" if backend else "development"]
            profile = _macos_profile(bundle, writable, environment) if backend else ""
            if backend:
                profile_path = stage / "sandbox.sb"
                profile_path.write_text(profile)
                command = [backend, "-f", str(profile_path), *command]
            child_env = {"PATH": "/usr/bin:/bin", "HOME": str(writable),
                         "TMPDIR": str(writable), "LANG": "en_US.UTF-8",
                         "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
            error = None
            with (writable / "stdout.log").open("wb") as stdout, \
                    (writable / "stderr.log").open("wb") as stderr:
                try:
                    completed = subprocess.run(
                        command, cwd=bundle, env=child_env, stdin=subprocess.DEVNULL,
                        stdout=stdout, stderr=stderr, timeout=MAX_TIMEOUT_SECONDS,
                        check=False,
                    )
                    if completed.returncode:
                        error = f"Worker exited with code {completed.returncode}"
                except subprocess.TimeoutExpired:
                    error = f"Worker exceeded {MAX_TIMEOUT_SECONDS}s batch timeout"
                except OSError as exc:
                    error = f"Worker launch failed: {type(exc).__name__}: {exc}"
            worker = {}
            result_path = writable / "worker_result.json"
            if result_path.is_file():
                try:
                    worker = _strict_object(result_path)
                except (ValueError, OSError) as exc:
                    error = f"Invalid worker result: {exc}"
            elif error is None:
                error = "Worker did not produce a result"
            tests = worker.get("self_test", {})
            isolated = bool(
                backend and isinstance(tests, dict)
                and tests.get("status") == "PASS"
                and all(tests.get(name) is True for name in (
                    "outside_read_denied", "outside_write_denied", "network_denied"
                ))
            )
            if backend and not isolated:
                error = error or "Required OS isolation self-test did not pass"
            records = worker.get("runs", [])
            if not isinstance(records, list):
                records, error = [], "Worker runs must be a list"
            by_seed: dict[int, dict] = {}
            for record in records:
                if (not isinstance(record, dict) or record.get("seed") not in seeds
                        or record["seed"] in by_seed):
                    error = "Unexpected or duplicate worker seed record"
                    by_seed = {}
                    break
                by_seed[record["seed"]] = record
            # An isolation failure invalidates even purported completed records.
            if backend and not isolated:
                by_seed = {}
            runs = [by_seed.get(seed, _failure(seed, error or "Seed not completed"))
                    for seed in seeds]
            source_stable = source_manifest() == sources
            environment_stable = runtime_manifest() == environment
            if worker.get("bundle_hashes") != bundle_hashes and not error:
                error = "Worker did not confirm copied source identity"
            if not source_stable:
                error = "Source changed during execution"
            if not environment_stable:
                error = "Runtime dependency identity changed during execution"
            execution = {
                "mode": mode,
                "status": "PASS" if isolated and not error else "FAIL" if backend
                else "NOT_RUN",
                "source_hashes": sources, "bundle_hashes": bundle_hashes,
                "case_sha256": hashlib.sha256(case_bytes).hexdigest(),
                "environment": environment, "self_test": tests,
                "source_stable": source_stable,
                "environment_stable": environment_stable,
                "sandbox_profile_sha256": hashlib.sha256(profile.encode()).hexdigest()
                if profile else None,
                "timeout_seconds": MAX_TIMEOUT_SECONDS,
                "error": error,
                "stderr_tail": (writable / "stderr.log").read_text(
                    errors="replace")[-4000:],
                "scope": "OS worker isolation; not proof of analyst or model blindness",
            }
    result = {"runs": runs, "execution": execution}
    (output_dir / "execution.json").write_text(json.dumps(
        result, indent=2, allow_nan=False))
    return result
