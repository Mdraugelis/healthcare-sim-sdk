"""Run provenance: what produced a saved experiment.

Records the code revision, the resolved configuration's hash, every seed
found in the configuration, and the interpreter and dependency versions.
Nothing here affects simulation behavior or random-number consumption.
"""

import hashlib
import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROVENANCE_FILE = "provenance.json"

DEPENDENCIES = (
    "healthcare-sim-sdk", "numpy", "scipy", "hydra-core", "omegaconf",
    "statsmodels", "pandas",
)

_PACKAGE_DIR = Path(__file__).resolve().parents[1]


def _git(args: List[str]) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", *args], cwd=_PACKAGE_DIR, capture_output=True,
            text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _installed_commit() -> Optional[str]:
    """Commit recorded by pip/uv for a VCS install (no git checkout present)."""
    try:
        text = metadata.distribution("healthcare-sim-sdk").read_text(
            "direct_url.json")
    except metadata.PackageNotFoundError:
        return None
    if not text:
        return None
    try:
        return json.loads(text).get("vcs_info", {}).get("commit_id")
    except ValueError:
        return None


def code_revision() -> Dict[str, Any]:
    """Return the commit that produced the run and whether the tree was dirty.

    ``dirty`` is None when no git checkout is available. A dirty tree means
    the commit alone does not reproduce the run.
    """
    commit = _git(["rev-parse", "HEAD"])
    if commit:
        status = _git(["status", "--porcelain", "--untracked-files=no"])
        return {"commit": commit, "dirty": bool(status) if status is not None
                else None, "source": "git"}
    commit = _installed_commit()
    if commit:
        return {"commit": commit, "dirty": None, "source": "installed package"}
    return {"commit": None, "dirty": None, "source": "unavailable"}


def dependency_versions() -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {}
    for name in DEPENDENCIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def find_seeds(config: Any, prefix: str = "") -> List[Tuple[str, Any]]:
    """Every config entry whose key is or ends with 'seed'/'seeds'."""
    found: List[Tuple[str, Any]] = []
    if isinstance(config, dict):
        for key, value in config.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            lowered = str(key).lower()
            if lowered.endswith("seed") or lowered.endswith("seeds"):
                found.append((path, value))
            else:
                found.extend(find_seeds(value, path))
    elif isinstance(config, list):
        for index, value in enumerate(config):
            found.extend(find_seeds(value, f"{prefix}[{index}]"))
    return found


def build_provenance(config_path: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    config_bytes = Path(config_path).read_bytes()
    return {
        "code_revision": code_revision(),
        "config_file": Path(config_path).name,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "seeds": {path: value for path, value in find_seeds(config)},
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "dependencies": dependency_versions(),
    }


def write_provenance(output_dir: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    record = build_provenance(output_dir / "config.json", config)
    (output_dir / PROVENANCE_FILE).write_text(
        json.dumps(record, indent=2, default=str))
    return record
