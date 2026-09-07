#!/usr/bin/env python3
"""Check exported teaching JavaScript against independent numerical references.

Run from any directory with the repository's Python environment. This does not
run clinical simulations, browser interactions, or the production test suite.
"""

from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import numpy as np
from scipy.special import expit, logit
from scipy.stats import norm, rankdata

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

# Script execution from another directory requires the repository path first.
from healthcare_sim_sdk.experiments.analysis.its import (  # noqa: E402
    segmented_regression,
)
from healthcare_sim_sdk.ml.performance import theoretical_ppv  # noqa: E402
from healthcare_sim_sdk.population.temporal_dynamics import (  # noqa: E402
    annual_risk_to_hazard,
    hazard_to_timestep_probability,
)

BUNDLED_NODE = (
    Path.home() / ".cache/codex-runtimes/codex-primary-runtime"
    / "dependencies/node/bin/node"
)
APP = ROOT / "docs/textbook/assets/app.js"
NODE_DRIVER = r"""
const fs = require('fs');
const math = require(process.argv[1]);
const calls = JSON.parse(fs.readFileSync(0, 'utf8'));
function finite(value) {
  if (typeof value === 'number' && !Number.isFinite(value)) {
    throw new Error('Nonfinite numerical output');
  }
  if (value && typeof value === 'object') Object.values(value).forEach(finite);
}
const result = {};
for (const call of calls) {
  try {
    const value = math[call.method](...call.args);
    finite(value);
    result[call.id] = {value};
  } catch (error) {
    result[call.id] = {error: String(error.message)};
  }
}
process.stdout.write(JSON.stringify(result));
"""


def compact(value):
    """Keep numeric evidence readable without dropping failure diagnostics."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, list) and len(value) > 16:
        return {"length": len(value), "first_eight": value[:8]}
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", type=Path)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "docs/textbook/qa/numerical_results.json",
    )
    args = parser.parse_args()
    node = args.node or (BUNDLED_NODE if BUNDLED_NODE.is_file() else None)
    node = node or shutil.which("node")
    if not node:
        parser.error("Node is unavailable; provide --node /absolute/path/to/node")

    calls = []

    def request(name, method, *values):
        calls.append({"id": name, "method": method, "args": values})

    for name, values in {
        "bayes_default": [.05, .8, .9],
        "bayes_no_flags": [0, .8, 1],
        "bayes_no_cases": [0, .8, .9],
        "bayes_empty": [.2, .8, .9, 0],
        "bayes_all_cases": [1, 1, 0],
    }.items():
        request(name, "bayes", *values)
    risk_cases = [
        (0., 1.), (.05, 1 / 52), (.05, 4 / 52), (.36, .5),
        (.9, 1.), (.9, 1 / 52), (1e-12, 1 / 52), (.5, 0.),
    ]
    for i, values in enumerate(risk_cases):
        request(f"risk_{i}", "risk", *values)
    for rho in [-1, -.5, 0, .8, 1]:
        request(f"coupling_{rho}", "coupling", 10, 10, rho, 25)
    request("coupling_unequal", "coupling", 3, 4, 1, 25)
    for sd, n in [(4, 25), (4, 100), (10, 5), (0, 25)]:
        request(f"mc_{sd}_{n}", "mc", sd, n)

    risks = [.05, .1, .2, .3, .4, .55, .65, .75, .85, .95]
    labels = [0, 0, 1, 0, 0, 1, 0, 1, 1, 1]
    for slope, offset in [(1, 0), (.2, -2), (3, 2), (1.7, -.8)]:
        request(f"cal_{slope}_{offset}", "calibrate", risks, slope, offset)
        predictions = expit(offset + slope * logit(risks)).tolist()
        request(f"auc_{slope}_{offset}", "auc", predictions, labels)
        request(f"brier_{slope}_{offset}", "expectedBrier", risks, predictions)
    request("auc_ties", "auc", [.1, .2, .2, .9], [0, 1, 0, 1])
    request("auc_no_positive", "auc", [.1, .2], [0, 0])
    for k, reserved in [(0, 0), (1, 0), (3, 1), (2, 6), (6, 0), (6, 6)]:
        for order in ["risk", "benefit"]:
            request(f"capacity_{k}_{reserved}_{order}", "capacity", k, reserved, order)

    its_cases = [
        (-3, -.1, 0, 0, 0, 24), (-3, -.1, 5, 0, 0, 24),
        (0, 0, 0, 0, 0, 12), (8, .4, -8, 0, 0, 36),
        (-3, -.1, 2, 1.25, .7, 24), (8, .4, -8, 2, .9, 12),
        (-8, -.4, 8, 2, .9, 36),
    ]
    for i, values in enumerate(its_cases):
        request(f"its_{i}", "its", *values)
    request("singular", "solve", [[1, 1], [2, 2]], [1, 2])
    request("monitor_null", "monitor", 0, False, .5, 4)
    request("monitor_abrupt", "monitor", 1.5, False, .5, 4)
    request("monitor_boundary", "monitor", 4, False, .5, 4)
    request("monitor_ramp", "monitor", 1.5, True, .5, 4)
    alarm_cases = [(0, 52), (.0027, 52), (.05, 1), (.05, 20), (1e-12, 1000)]
    for i, values in enumerate(alarm_cases):
        request(f"false_alarm_{i}", "falseAlarm", *values)

    process = subprocess.run(
        [str(node), "-e", NODE_DRIVER, str(APP)],
        input=json.dumps(calls, allow_nan=False), text=True,
        capture_output=True, check=True, timeout=30,
    )
    results = json.loads(process.stdout)
    checks = []

    def result(name):
        if "error" in results[name]:
            raise RuntimeError(f"{name}: {results[name]['error']}")
        return results[name]["value"]

    def check(name, actual, expected, *, atol=1e-11, rtol=1e-10, note=""):
        record = {
            "id": name, "actual": compact(actual), "expected": compact(expected),
        }
        try:
            if expected is None or isinstance(expected, (str, bool)):
                assert actual == expected, f"{actual!r} != {expected!r}"
            elif (isinstance(expected, list) and expected
                  and isinstance(expected[0], str)):
                assert actual == expected, f"{actual!r} != {expected!r}"
            else:
                np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
            record["status"] = "PASS"
        except AssertionError as error:
            record.update(status="FAIL", detail=str(error))
        if note:
            record["note"] = note
        checks.append(record)

    b = result("bayes_default")
    check("bayes_counts", [b[k] for k in ["tp", "fn", "fp", "tn"]],
          [400, 100, 950, 8550])
    check("bayes_ppv_independent", b["ppv"], 400 / 1350)
    check("bayes_ppv_sdk", b["ppv"], theoretical_ppv(
        prevalence=.05, sensitivity=.8, specificity=.9,
    ))
    for key in ["bayes_no_flags", "bayes_empty"]:
        check(key, result(key)["ppv"], None)
    check("bayes_zero_ppv", result("bayes_no_cases")["ppv"], 0.)
    check("bayes_unit_ppv", result("bayes_all_cases")["ppv"], 1.)
    for i, (risk, duration) in enumerate(risk_cases):
        # High-precision survival powers avoid mirroring the JS log1p/expm1 path.
        with localcontext() as context:
            context.prec = 50
            expected = float(
                1 - (1 - Decimal(str(risk))) ** Decimal(str(duration))
            )
        check(f"risk_reference_{i}", result(f"risk_{i}"), expected, atol=1e-28)
        sdk = hazard_to_timestep_probability(
            annual_risk_to_hazard(np.array([risk])), duration,
        )[0]
        check(f"risk_sdk_{i}", result(f"risk_{i}"), float(sdk), atol=2e-16,
              note="Absolute tolerance allows cancellation in the SDK's 1-exp form.")
    check("risk_hand_half_year", result("risk_3"), .2)
    for rho in [-1, -.5, 0, .8, 1]:
        expected = 200 - 200 * rho
        value = result(f"coupling_{rho}")
        check(f"coupling_variance_{rho}", value["variance"], expected)
        check(f"coupling_mcse_{rho}", value["mcse"], math.sqrt(expected / 25))
    check("coupling_unequal_perfect", result("coupling_unequal")["variance"], 1.)
    for sd, n in [(4, 25), (4, 100), (10, 5), (0, 25)]:
        value = result(f"mc_{sd}_{n}")
        check(f"mcse_{sd}_{n}", value["mcse"], sd / math.sqrt(n))
        check(f"normal_halfwidth_{sd}_{n}", value["half"],
              norm.ppf(.975) * sd / math.sqrt(n))

    q = np.array(risks)
    labels_array = np.array(labels)
    ranks = rankdata(q)
    auc_reference = (ranks[labels_array == 1].sum() - 5 * 6 / 2) / 25
    for slope, offset in [(1, 0), (.2, -2), (3, 2), (1.7, -.8)]:
        p = expit(offset + slope * logit(q))
        check(f"calibration_reference_{slope}_{offset}",
              result(f"cal_{slope}_{offset}"), p)
        check(f"auc_invariance_{slope}_{offset}",
              result(f"auc_{slope}_{offset}"), auc_reference)
        # Separate irreducible variance and distortion via a proper-score identity.
        expected = float(np.mean(q * (1 - q)) + np.mean((p - q) ** 2))
        check(f"brier_decomposition_{slope}_{offset}",
              result(f"brier_{slope}_{offset}"), expected)
    check("auc_hand_ties", result("auc_ties"), 3.5 / 4)
    check("auc_missing_class", result("auc_no_positive"), None)

    gains = {"A": .1, "B": .02, "C": .05, "D": .18, "E": .144, "F": .06}
    orders = {"risk": ["B", "D", "E", "A", "F", "C"],
              "benefit": ["D", "E", "A", "F", "C", "B"]}
    for k, reserved in [(0, 0), (1, 0), (3, 1), (2, 6), (6, 0), (6, 6)]:
        available = max(k - reserved, 0)
        baseline = list(gains)[:available]
        for order, ranked_ids in orders.items():
            value = result(f"capacity_{k}_{reserved}_{order}")
            chosen = ranked_ids[:available]
            tag = f"capacity_{k}_{reserved}_{order}"
            check(tag + "_ids", [p["id"] for p in value["ranked"]], chosen)
            check(tag + "_equal_budget",
                  [len(value["baseline"]), len(value["ranked"])],
                  [available, available])
            check(tag + "_gains",
                  [value["baselineGain"], value["rankedGain"], value["delta"]],
                  [sum(gains[x] for x in baseline), sum(gains[x] for x in chosen),
                   sum(gains[x] for x in chosen) - sum(gains[x] for x in baseline)])

    fixture = np.array([.2, -.7, .5, .1, -.4, .9, -.3, .4, -.8, .6, -.2, .1])
    time = np.arange(48)
    for i, (level, slope, confound, amplitude, rho, breakpoint) in enumerate(its_cases):
        value = result(f"its_{i}")
        design = np.column_stack([
            np.ones(48), time, time >= breakpoint,
            np.maximum(0, time - breakpoint),
        ])
        # Convolution independently checks the JS recursive filtering implementation.
        innovations = np.resize(fixture, 48) * amplitude
        noise = np.convolve(innovations, rho ** np.arange(48))[:48]
        expected_y = design @ np.array([30, .2, level + confound, slope]) + noise
        beta = np.linalg.lstsq(design, expected_y, rcond=None)[0]
        check(f"its_fixture_{i}", [r["y"] for r in value["rows"]], expected_y)
        check(f"its_lstsq_{i}", value["beta"], beta)
        check(f"its_fit_{i}", [r["fit"] for r in value["rows"]], design @ beta)
        fitted = segmented_regression(expected_y, breakpoint)
        check(f"its_sdk_{i}", value["beta"], [
            fitted.intercept, fitted.pre_slope,
            fitted.level_change, fitted.slope_change,
        ])
        if amplitude == 0:
            check(f"its_planted_truth_{i}", value["beta"],
                  [30, .2, level + confound, slope])
    check("its_singular_rejected", results["singular"].get("error"),
          "Singular teaching design")

    null = result("monitor_null")
    check("cusum_null_hand_path", [r["c"] for r in null[:12]],
          [0, 0, .2, 0, 0, 0, 0, 0, .4, 0, 0, 0])
    abrupt = result("monitor_abrupt")
    check("cusum_abrupt_hand_path", [r["c"] for r in abrupt[30:39]],
          [1.5, 2.3, 4.2, .5, 1.6, 2.2, 3.4, 3.8, 5.5])
    check("cusum_first_alarm", next(r["t"] for r in abrupt if r["alarm"]), 32)
    check("point_rule_no_alarm_at_moderate_shift",
          any(r["point"] for r in abrupt), False)
    boundary = result("monitor_boundary")
    check("cusum_strict_threshold", [
        boundary[30]["c"], boundary[30]["alarm"], boundary[31]["alarm"],
    ], [4., False, True])
    ramp = result("monitor_ramp")
    check("ramp_endpoints", [ramp[t]["z"] for t in [30, 44, 45]], [.6, 2.4, 1.])
    for i, (alpha, n_checks) in enumerate(alarm_cases):
        with localcontext() as context:
            context.prec = 50
            expected = float(1 - (1 - Decimal(str(alpha))) ** n_checks)
        check(f"repeated_false_alarm_{i}", result(f"false_alarm_{i}"),
              expected, atol=1e-25)

    report = {
        "status": "FAIL" if any(c["status"] == "FAIL" for c in checks) else "PASS",
        "check_count": len(checks), "node_calls": len(calls),
        "app_sha256": hashlib.sha256(APP.read_bytes()).hexdigest(),
        "checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "node": subprocess.check_output([str(node), "--version"], text=True).strip(),
        "scope": ("Pure JavaScript teaching functions, declared slider domains "
                  "and selected mathematical boundaries."),
        "limitations": [
            "No browser layout, controls, persistence or accessibility execution.",
            "No clinical validity, stochastic calibration, estimator coverage "
            "or monitoring-performance certification.",
            "The ITS and monitoring fixtures are deterministic; the checks "
            "do not turn them into random samples.",
            "These functions are teaching helpers, not a validated public API "
            "for arbitrary invalid input.",
        ],
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "check_count": len(checks),
                      "output": str(args.output)}, allow_nan=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
