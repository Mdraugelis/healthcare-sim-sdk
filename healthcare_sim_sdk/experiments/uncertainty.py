"""Explicit simulation uncertainty, with completeness checked before reduction."""

from typing import Any, Dict, Iterable

import numpy as np
from scipy.stats import beta, t  # type: ignore[import-untyped]


def summarize_values(values: Iterable[float]) -> Dict[str, Any]:
    """Separate the seed distribution from Monte Carlo error of its mean.

    The t interval describes Monte Carlo precision assuming independent
    repetitions with finite variance. It is not an interval for a clinical
    study estimator or for uncertain input assumptions.
    """
    x: np.ndarray = np.asarray(list(values), dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Summary requires nonempty, finite scalar values")
    n = len(x)
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else None
    mcse = sd / np.sqrt(n) if sd is not None else None
    half = float(t.ppf(0.975, n - 1) * mcse) if n > 1 else None
    if any(value is not None and not np.isfinite(value)
           for value in (mean, sd, mcse, half)):
        raise ValueError(
            "Summary arithmetic overflowed; no finite uncertainty estimate")
    return {
        "mean": mean, "n_seeds": n, "seed_sd": sd,
        "seed_p025": float(np.percentile(x, 2.5)),
        "seed_p975": float(np.percentile(x, 97.5)),
        "mcse_mean": float(mcse) if mcse is not None else None,
        "mean_ci95_low": mean - half if half is not None else None,
        "mean_ci95_high": mean + half if half is not None else None,
        "uncertainty_status": "PASS" if n > 1 else "INCONCLUSIVE",
        "interval_definition": (
            "seed_p025/p975: empirical run distribution; mean_ci95: "
            "t interval for Monte Carlo mean, conditional on inputs"
        ),
    }


def summarize_metrics(rows, keys, expected_seeds=None):
    """Never silently drop missing metrics or repetitions."""
    if not rows:
        raise ValueError("No repetitions to summarize")
    if any("seed" in row for row in rows):
        actual = [row.get("seed") for row in rows]
        if any(seed is None for seed in actual) or len(set(actual)) != len(actual):
            raise ValueError("Missing or duplicate seed in repetitions")
    if expected_seeds is not None:
        expected = list(expected_seeds)
        actual = [r.get("seed") for r in rows]
        if len(set(expected)) != len(expected):
            raise ValueError("Expected seeds must be unique")
        if len(actual) != len(expected) or set(actual) != set(expected):
            raise ValueError("Missing, duplicate, or unexpected seed")
    result = {}
    for key in keys:
        if any(key not in row for row in rows):
            raise ValueError(f"Missing required metric: {key}")
        result[key] = summarize_values(row[key] for row in rows)
    return result


def binomial_performance(hits: int, repetitions: int) -> Dict[str, Any]:
    """Estimate detection/power/coverage with exact two-sided 95% bounds."""
    if (isinstance(hits, bool) or isinstance(repetitions, bool)
            or not isinstance(hits, int) or not isinstance(repetitions, int)
            or repetitions <= 0 or not 0 <= hits <= repetitions):
        raise ValueError("Require integer 0 <= hits <= repetitions, n > 0")
    rate = hits / repetitions
    lo = float(beta.ppf(0.025, hits, repetitions - hits + 1)) if hits else 0.0
    hi = (float(beta.ppf(0.975, hits + 1, repetitions - hits))
          if hits < repetitions else 1.0)
    return {
        "hits": hits, "repetitions": repetitions, "rate": rate,
        "mcse": float(np.sqrt(rate * (1 - rate) / repetitions)),
        "ci95_low": lo, "ci95_high": hi,
        "interval_definition": "Clopper-Pearson exact binomial 95% interval",
        "boundary_note": (
            "Plug-in MCSE is zero at a boundary; use the exact interval"
            if hits in (0, repetitions) else ""
        ),
    }
