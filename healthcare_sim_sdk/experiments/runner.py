"""Shared pieces of a scenario runner.

Every runner has the same shape (AGENTS.md, "Changing code"):

    @hydra.main(version_base=None, config_path="configs",
                config_name="<runner>")
    def main(cfg: DictConfig) -> None:
        settings = runner_settings(cfg)      # every knob comes from YAML
        out = run_dir()                      # Hydra's directory for this run
        ...simulate...
        finalize_experiment(out, settings, metrics, results,
                            validation_checks=checks)

Settings and grids live in the scenario's ``configs/<runner>.yaml``; none
is hard-coded in Python. ``scripts/sdk_lint.py`` rule SDK007 enforces the
shape, and ``config/invariants.yaml`` lists the scripts exempt from the
final step, each with a reason.
"""

from pathlib import Path
from typing import Any, Dict, Iterable, List

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from .validate import Check, CheckStatus


def runner_settings(cfg: DictConfig) -> Dict[str, Any]:
    """The resolved run configuration as a plain dict, ready to save."""
    settings = OmegaConf.to_container(cfg, resolve=True)
    settings.pop("hydra", None)
    return settings


def run_dir() -> Path:
    """The directory Hydra assigned to this run (``hydra.run.dir``, or the
    cell's directory under ``--multirun``)."""
    return Path(HydraConfig.get().runtime.output_dir)


def check(name: str, description: str, passed: bool, *, expected: Any = True,
          actual: Any = None, tolerance: str = "exact",
          category: str = "simulation", detail: str = "") -> Check:
    """One verification result for ``finalize_experiment``."""
    return Check(
        name, description, expected, passed if actual is None else actual,
        tolerance, bool(passed), detail,
        status=CheckStatus.PASS if passed else CheckStatus.FAIL,
        category=category)


def finite_metrics_check(metrics: Dict[str, Any],
                         keys: Iterable[str] = ()) -> Check:
    """Structural check: the named metrics (default: all numeric ones) are
    finite numbers."""
    import math

    names: List[str] = list(keys) or [
        k for k, v in metrics.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)]
    bad = [k for k in names
           if not isinstance(metrics.get(k), (int, float))
           or not math.isfinite(metrics[k])]
    return check(
        "metrics_finite", "Reported metrics are finite numbers", not bad,
        expected="all finite", actual=bad or "all finite",
        detail=f"checked {len(names)} metric(s)")


def build(config_class, settings: Dict[str, Any], **overrides):
    """Build a runner's config dataclass from YAML settings.

    Every field must come from the YAML (or ``overrides``), so the
    dataclass defaults are never what a run silently used. Unknown
    settings are ignored: one YAML file may feed several dataclasses.
    """
    import dataclasses

    values = {**settings, **overrides}
    names = [f.name for f in dataclasses.fields(config_class) if f.init]
    missing = [n for n in names if n not in values]
    if missing:
        raise KeyError(
            f"{config_class.__name__}: the runner's YAML config lacks "
            f"{', '.join(missing)}")
    return config_class(**{n: values[n] for n in names})
