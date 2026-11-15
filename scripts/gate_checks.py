#!/usr/bin/env python3
"""Checks used by scripts/quality_gate.sh. Standard library only.

  gate_checks.py same-wheel DIST_A DIST_B [SHA_OUT]   identical wheel bytes
  gate_checks.py wheel-contents DIST SNAPSHOT          YAML set and no private files
  gate_checks.py smoke                                 run from outside the checkout
"""

import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile
from fnmatch import fnmatch
from pathlib import Path

PRIVATE_PATTERNS = (
    "*.private.*", "*/model_frozen.yaml", "*/model_targets.yaml",
    "*/configs/campus/*",
)


def _wheel(dist: str) -> Path:
    wheels = sorted(Path(dist).glob("*.whl"))
    if len(wheels) != 1:
        sys.exit(f"expected one wheel in {dist}, found {len(wheels)}")
    return wheels[0]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same_wheel(dist_a: str, dist_b: str, out: str = "") -> int:
    a, b = _wheel(dist_a), _wheel(dist_b)
    sa, sb = _sha(a), _sha(b)
    print(f"{a.name}\n  build 1 {sa}\n  build 2 {sb}")
    if out:
        Path(out).write_text(sa + "\n")
    if sa != sb:
        za, zb = zipfile.ZipFile(a), zipfile.ZipFile(b)
        for name in sorted(set(za.namelist()) | set(zb.namelist())):
            ia = za.getinfo(name) if name in za.namelist() else None
            ib = zb.getinfo(name) if name in zb.namelist() else None
            if (ia is None or ib is None or ia.CRC != ib.CRC
                    or ia.date_time != ib.date_time):
                print(f"  differs: {name}")
        return 1
    return 0


def wheel_contents(dist: str, snapshot: str) -> int:
    names = zipfile.ZipFile(_wheel(dist)).namelist()
    root = Path(snapshot)
    tracked = sorted(
        p.relative_to(root).as_posix()
        for p in (root / "healthcare_sim_sdk").rglob("*.y*ml")
        if not fnmatch(p.relative_to(root).as_posix(), "*/configs/campus/*"))
    in_wheel = sorted(n for n in names if n.endswith((".yaml", ".yml")))
    private = sorted(n for n in names
                     if any(fnmatch(n, pat) for pat in PRIVATE_PATTERNS))
    missing = sorted(set(tracked) - set(in_wheel))
    extra = sorted(set(in_wheel) - set(tracked))
    print(f"YAML in wheel: {len(in_wheel)}; tracked public YAML: {len(tracked)}")
    for label, items in (("missing", missing), ("unexpected", extra),
                         ("private", private)):
        for item in items:
            print(f"  {label}: {item}")
    return 1 if missing or extra or private else 0


def smoke() -> int:
    """Installed wheel only: import, packaged config, a real Hydra runner."""
    import healthcare_sim_sdk

    pkg = Path(healthcare_sim_sdk.__file__).resolve().parent
    if "site-packages" not in pkg.parts:
        sys.exit(f"imported from {pkg}, not an installed wheel")
    print(f"installed package: {pkg}")

    from importlib.resources import files
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(files("healthcare_sim_sdk.scenarios.noshow_targeted_reminders")
                         / "configs" / "review_demo.yaml")
    assert cfg.horizon_days > 0
    print("packaged YAML config: loaded")

    from healthcare_sim_sdk.experiments.analysis.its import segmented_regression
    assert callable(segmented_regression)
    print("ITS module: imported")

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        cmd = [sys.executable, "-m",
               "healthcare_sim_sdk.experiments.issue_register_validation",
               "kind=trews", "capacity=10", "max_hours=6", "seed=42",
               f"hydra.run.dir={run_dir}", "hydra.job.chdir=true"]
        done = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True,
                              check=False)
        if done.returncode != 0:
            print(done.stdout[-2000:], done.stderr[-2000:])
            return 1
        metrics = json.loads((run_dir / "metrics.json").read_text())
        provenance = json.loads((run_dir / "provenance.json").read_text())
        assert metrics, "runner saved no metrics"
        print(f"Hydra runner from wheel: {len(metrics)} metrics; code revision "
              f"source '{provenance['code_revision']['source']}'")
    return 0


if __name__ == "__main__":
    command, *args = sys.argv[1:] or ["--help"]
    if command == "same-wheel":
        sys.exit(same_wheel(*args))
    if command == "wheel-contents":
        sys.exit(wheel_contents(*args))
    if command == "smoke":
        sys.exit(smoke())
    print(__doc__)
    sys.exit(2)
