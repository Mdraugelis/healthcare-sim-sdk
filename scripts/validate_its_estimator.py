#!/usr/bin/env python3
"""Statistical validation run for ``experiments/analysis/its``.

Runs the full sim-guide verification protocol against the ITS
estimator and prints a structured report. Used to establish
empirical bounds before locking in tests.

Usage::

    python scripts/validate_its_estimator.py

The script is deterministic (fixed seeds) so the output is
reproducible. Every number printed has the MC noise budget
explicit so reviewers can judge whether the bound is load-bearing
or just a sanity band.
"""

from __future__ import annotations

import time

import numpy as np

from healthcare_sim_sdk.experiments.analysis.its import (
    power_across_seeds,
    segmented_regression,
)

# ---------------- Level 2: statistical sanity ----------------


def run_type_i_rate_calibration(
    n_reps: int = 2000, seed: int = 2024,
) -> dict:
    """Type I rate under the null at nominal α=0.05.

    With direction matching required, expected rate is α/2 = 0.025.
    4-sigma SD on a Bernoulli(0.025) over n_reps=2000 is
    sqrt(2000 * 0.025 * 0.975) / 2000 ≈ 0.0035, so the 4-sigma
    band is approximately [0.011, 0.039].
    """
    rng = np.random.default_rng(seed)
    n = 60
    noise_sd = 0.004
    series_list = [
        0.048 + rng.normal(0, noise_sd, n) for _ in range(n_reps)
    ]
    rate = power_across_seeds(
        series_list, break_index=36,
        expected_direction="decrease", alpha=0.05,
    )
    ci_half = 4.0 * float(np.sqrt(0.025 * 0.975 / n_reps))
    band = (max(0.0, 0.025 - ci_half), 0.025 + ci_half)
    return {
        "name": "Type I rate (direction-matched, α=0.05)",
        "n_reps": n_reps,
        "observed": rate,
        "expected": 0.025,
        "band_4sigma": band,
        "pass": band[0] <= rate <= band[1],
    }


def run_ci_coverage(
    n_reps: int = 1000, seed: int = 2024,
) -> dict:
    """95% Wald CI coverage on planted-effect series.

    Construct CI as level_change ± 1.96 * level_change_se and
    count seeds where the planted truth is inside. Expected
    coverage = 0.95. 4-sigma SD = sqrt(n_reps * 0.95 * 0.05) /
    n_reps ≈ 0.0069, so 4σ band ≈ [0.922, 0.978].
    """
    rng = np.random.default_rng(seed)
    planted = -0.010
    noise_sd = 0.004
    hits = 0
    for _ in range(n_reps):
        pre = 0.048 + rng.normal(0, noise_sd, 36)
        post = 0.048 + planted + rng.normal(0, noise_sd, 24)
        series = np.concatenate([pre, post])
        result = segmented_regression(series, break_index=36)
        lo = result.level_change - 1.96 * result.level_change_se
        hi = result.level_change + 1.96 * result.level_change_se
        if lo <= planted <= hi:
            hits += 1
    rate = hits / n_reps
    ci_half = 4.0 * float(np.sqrt(0.95 * 0.05 / n_reps))
    band = (0.95 - ci_half, min(1.0, 0.95 + ci_half))
    return {
        "name": "95% Wald CI coverage",
        "n_reps": n_reps,
        "observed": rate,
        "expected": 0.95,
        "band_4sigma": band,
        "pass": band[0] <= rate <= band[1],
    }


def run_null_bias(
    n_reps: int = 2000, seed: int = 2024,
) -> dict:
    """Null bias: mean of level_change across null seeds.

    Under H0 the estimator is unbiased → mean should be ~0. The
    theoretical SD of level_change on a 60-point series with
    noise SD 0.004 at break_index=36 is approximately
    sqrt(1/36 + 1/24) * 0.004 ≈ 0.00167; SD of the sample mean
    across n_reps=2000 is ~3.7e-5. 4σ band ≈ [-0.00015, 0.00015].
    """
    rng = np.random.default_rng(seed)
    n = 60
    noise_sd = 0.004
    estimates = []
    for _ in range(n_reps):
        series = 0.048 + rng.normal(0, noise_sd, n)
        result = segmented_regression(series, break_index=36)
        estimates.append(result.level_change)
    arr = np.array(estimates)
    mean_est = float(arr.mean())
    se_mean = float(arr.std(ddof=1) / np.sqrt(n_reps))
    band = (-4.0 * se_mean, 4.0 * se_mean)
    return {
        "name": "Null bias on level_change",
        "n_reps": n_reps,
        "observed": mean_est,
        "expected": 0.0,
        "band_4sigma": band,
        "empirical_se_of_estimator": float(arr.std(ddof=1)),
        "pass": band[0] <= mean_est <= band[1],
    }


# ---------------- Level 3: conservation laws / monotonicity ----------------


def run_power_vs_effect_size(
    n_reps: int = 400, seed: int = 2024,
) -> dict:
    """Power must be monotone non-decreasing in planted effect size."""
    rng = np.random.default_rng(seed)
    effects = [0.0, -0.003, -0.006, -0.010, -0.015]
    powers = []
    noise_sd = 0.004
    for eff in effects:
        series_list = []
        for _ in range(n_reps):
            pre = 0.048 + rng.normal(0, noise_sd, 36)
            post = 0.048 + eff + rng.normal(0, noise_sd, 24)
            series_list.append(np.concatenate([pre, post]))
        powers.append(power_across_seeds(
            series_list, break_index=36,
            expected_direction="decrease",
        ))
    # Check strict monotonicity with small MC slack
    slack = 0.03
    monotone = all(
        powers[i + 1] >= powers[i] - slack
        for i in range(len(powers) - 1)
    )
    return {
        "name": "Power monotone in effect size",
        "effects": effects,
        "powers": powers,
        "slack": slack,
        "pass": monotone,
    }


def run_power_vs_sample_size(
    n_reps: int = 400, seed: int = 2024,
) -> dict:
    """Power must be monotone non-decreasing in n_post at fixed effect."""
    rng = np.random.default_rng(seed)
    n_posts = [12, 18, 24]
    powers = []
    planted = -0.010
    noise_sd = 0.004
    for n_post in n_posts:
        series_list = []
        for _ in range(n_reps):
            pre = 0.048 + rng.normal(0, noise_sd, 36)
            post = 0.048 + planted + rng.normal(0, noise_sd, n_post)
            series_list.append(np.concatenate([pre, post]))
        powers.append(power_across_seeds(
            series_list, break_index=36,
            expected_direction="decrease",
        ))
    slack = 0.03
    monotone = all(
        powers[i + 1] >= powers[i] - slack
        for i in range(len(powers) - 1)
    )
    return {
        "name": "Power monotone in n_post",
        "n_posts": n_posts,
        "powers": powers,
        "slack": slack,
        "pass": monotone,
    }


def run_hac_correctness_ar1(
    n_reps: int = 1500, seed: int = 2024,
) -> dict:
    """Under AR(1) ρ=0.5 null, OLS Type I is inflated; HAC is lower.

    Generates AR(1) null series (no planted effect, just correlated
    noise around 0.048) and compares OLS vs HAC rejection rates at
    α=0.05 (two-sided, ignoring direction). Expected: OLS rate
    clearly above 0.05, HAC rate closer to 0.05 and strictly below
    OLS.
    """
    rng = np.random.default_rng(seed)
    n = 60
    rho = 0.5
    innov_sd = 0.002

    ols_rejects = 0
    hac_rejects = 0
    for _ in range(n_reps):
        eps = np.empty(n)
        eps[0] = rng.normal(0, innov_sd)
        for i in range(1, n):
            eps[i] = rho * eps[i - 1] + rng.normal(0, innov_sd)
        series = 0.048 + eps
        ols = segmented_regression(
            series, break_index=36, hac_maxlags=0,
        )
        hac = segmented_regression(
            series, break_index=36, hac_maxlags=4,
        )
        if ols.level_change_pvalue < 0.05:
            ols_rejects += 1
        if hac.level_change_pvalue < 0.05:
            hac_rejects += 1

    ols_rate = ols_rejects / n_reps
    hac_rate = hac_rejects / n_reps
    return {
        "name": "HAC correctness under AR(1) ρ=0.5",
        "n_reps": n_reps,
        "ols_rate": ols_rate,
        "hac_rate": hac_rate,
        "ols_inflated": ols_rate > 0.075,
        "hac_below_ols": hac_rate < ols_rate,
        "pass": (ols_rate > 0.075) and (hac_rate < ols_rate),
    }


# ---------------- Level 5: boundary / calibration ----------------


def run_mde_calibration(
    n_reps: int = 1500, seed: int = 2024,
) -> dict:
    """Empirical SE and MDE for the segmented-regression estimator
    at the perinatal §5 parameters.

    Protocol §5 derives its 24m MDE = 0.011 absolute (22.9% RRR)
    from a simple two-rate comparison SE

        SE(diff) ≈ sqrt(p_cf*(1-p_cf)/n_pre + p_f*(1-p_f)/n_post)

    with a 1.4x autocorrelation penalty added on top. That is NOT
    the SE the segmented-regression estimator produces: segmented
    regression jointly estimates beta0/beta1/beta2/beta3, and the
    additional slope parameters inflate SE(beta2) beyond the simple
    two-rate SE.

    This check computes the estimator's empirical SD of beta2
    across seeds, compares it against the protocol's closed-form,
    and reports the empirical MDE at 80% power (``2.8*empirical_SE``).
    It does NOT assert that the protocol's formula is correct — the
    comparison is reported as a finding.
    """
    rng = np.random.default_rng(seed)
    p_cf = 0.048
    planted = -0.011
    monthly_n = 350
    monthly_noise_sd = float(
        np.sqrt(p_cf * (1 - p_cf) / monthly_n),
    )
    n_pre_months, n_post_months = 36, 24

    beta2_list = []
    se_list = []
    series_list = []
    for _ in range(n_reps):
        pre = p_cf + rng.normal(
            0, monthly_noise_sd, n_pre_months,
        )
        post = (
            p_cf + planted
            + rng.normal(0, monthly_noise_sd, n_post_months)
        )
        series = np.concatenate([pre, post])
        series_list.append(series)
        result = segmented_regression(series, break_index=36)
        beta2_list.append(result.level_change)
        se_list.append(result.level_change_se)

    empirical_sd_beta2 = float(np.std(beta2_list, ddof=1))
    mean_reported_se = float(np.mean(se_list))

    n_pre_events = n_pre_months * monthly_n
    n_post_events = n_post_months * monthly_n
    p_f = p_cf + planted
    protocol_se_raw = float(
        np.sqrt(
            p_cf * (1 - p_cf) / n_pre_events
            + p_f * (1 - p_f) / n_post_events,
        ),
    )
    protocol_se_inflated = protocol_se_raw * 1.4

    power = power_across_seeds(
        series_list, break_index=36,
        expected_direction="decrease",
    )

    empirical_mde_at_80 = 2.8 * empirical_sd_beta2
    empirical_rrr_at_80 = -empirical_mde_at_80 / p_cf

    return {
        "name": (
            "Empirical MDE vs protocol §5 closed-form "
            "(segmented regression)"
        ),
        "n_reps": n_reps,
        "planted_absolute": planted,
        "planted_rrr": -planted / p_cf,
        "empirical_sd_beta2": empirical_sd_beta2,
        "mean_reported_se": mean_reported_se,
        "protocol_se_raw": protocol_se_raw,
        "protocol_se_inflated": protocol_se_inflated,
        "empirical_power_at_planted": power,
        "empirical_mde_at_80pct": empirical_mde_at_80,
        "empirical_rrr_at_80pct": empirical_rrr_at_80,
        "se_ratio_vs_inflated": (
            empirical_sd_beta2 / protocol_se_inflated
        ),
        # No pass/fail: this is a reported finding.
        "pass": True,
    }


# ---------------- Level 4: case walkthroughs ----------------


def run_case_walkthroughs(seed: int = 2024) -> list:
    """Five constructed series traced through the estimator."""
    rng = np.random.default_rng(seed)
    cases = []

    # Case 1: null — flat series, noise only
    series = 0.048 + rng.normal(0, 0.003, 60)
    r = segmented_regression(series, break_index=36)
    cases.append({
        "label": "1. Flat null",
        "description": (
            "60 months of noise around 0.048, no planted effect. "
            "Direction should be 'none', level_change_pvalue > 0.05."
        ),
        "pre_mean": float(series[:36].mean()),
        "post_mean": float(series[36:].mean()),
        "level_change": r.level_change,
        "level_change_se": r.level_change_se,
        "level_change_pvalue": r.level_change_pvalue,
        "direction": r.direction,
        "pre_slope": r.pre_slope,
        "pre_slope_pvalue": r.pre_slope_pvalue,
    })

    # Case 2: small effect near the MDE
    pre = 0.048 + rng.normal(0, 0.003, 36)
    post = 0.037 + rng.normal(0, 0.003, 24)  # ~23% RRR
    series = np.concatenate([pre, post])
    r = segmented_regression(series, break_index=36)
    cases.append({
        "label": "2. Small effect at protocol MDE (23% RRR)",
        "description": (
            "Planted -0.011 level shift (0.048 → 0.037). "
            "This is right at the protocol's 24m MDE; p should "
            "be marginal."
        ),
        "pre_mean": float(series[:36].mean()),
        "post_mean": float(series[36:].mean()),
        "level_change": r.level_change,
        "level_change_se": r.level_change_se,
        "level_change_pvalue": r.level_change_pvalue,
        "direction": r.direction,
        "pre_slope": r.pre_slope,
        "pre_slope_pvalue": r.pre_slope_pvalue,
    })

    # Case 3: large effect (ceiling case)
    pre = 0.085 + rng.normal(0, 0.004, 36)
    post = 0.042 + rng.normal(0, 0.004, 24)  # ~50% RRR
    series = np.concatenate([pre, post])
    r = segmented_regression(series, break_index=36)
    cases.append({
        "label": "3. Ceiling case (50% RRR at A=0.085)",
        "description": (
            "Protocol boundary ceiling. Expected p << 0.001, "
            "decrease direction, level_change ≈ -0.043."
        ),
        "pre_mean": float(series[:36].mean()),
        "post_mean": float(series[36:].mean()),
        "level_change": r.level_change,
        "level_change_se": r.level_change_se,
        "level_change_pvalue": r.level_change_pvalue,
        "direction": r.direction,
        "pre_slope": r.pre_slope,
        "pre_slope_pvalue": r.pre_slope_pvalue,
    })

    # Case 4: AR(1) null — this is where HAC matters
    eps = np.empty(60)
    eps[0] = rng.normal(0, 0.002)
    for i in range(1, 60):
        eps[i] = 0.6 * eps[i - 1] + rng.normal(0, 0.002)
    series = 0.048 + eps
    r_ols = segmented_regression(
        series, break_index=36, hac_maxlags=0,
    )
    r_hac = segmented_regression(
        series, break_index=36, hac_maxlags=4,
    )
    cases.append({
        "label": "4. AR(1) null (ρ=0.6)",
        "description": (
            "No planted effect, but correlated residuals make "
            "segments look like a shift. OLS SE is naive and may "
            "produce a false positive; HAC should widen it."
        ),
        "pre_mean": float(series[:36].mean()),
        "post_mean": float(series[36:].mean()),
        "ols": {
            "level_change": r_ols.level_change,
            "level_change_se": r_ols.level_change_se,
            "level_change_pvalue": r_ols.level_change_pvalue,
            "direction": r_ols.direction,
        },
        "hac": {
            "level_change": r_hac.level_change,
            "level_change_se": r_hac.level_change_se,
            "level_change_pvalue": r_hac.level_change_pvalue,
            "direction": r_hac.direction,
        },
    })

    # Case 5: pre-trend with no level shift — β1 catches it, β2 doesn't
    pre = 0.020 + 0.0005 * np.arange(36) + rng.normal(0, 0.0005, 36)
    post = (
        0.020 + 0.0005 * 36
        + 0.0005 * np.arange(24)
        + rng.normal(0, 0.0005, 24)
    )
    series = np.concatenate([pre, post])
    r = segmented_regression(series, break_index=36)
    cases.append({
        "label": "5. Pre-trend, no break",
        "description": (
            "Linear upward trend through the whole series. "
            "β1 (pre_slope) should be large and significant; "
            "β2 (level_change) and β3 (slope_change) should both "
            "be near zero. This is the validity check for H4-style "
            "confounding."
        ),
        "pre_mean": float(series[:36].mean()),
        "post_mean": float(series[36:].mean()),
        "level_change": r.level_change,
        "level_change_se": r.level_change_se,
        "level_change_pvalue": r.level_change_pvalue,
        "direction": r.direction,
        "pre_slope": r.pre_slope,
        "pre_slope_pvalue": r.pre_slope_pvalue,
        "slope_change": r.slope_change,
        "slope_change_pvalue": r.slope_change_pvalue,
    })

    return cases


# ---------------- Report ----------------


def main() -> None:
    print("=" * 72)
    print("ITS ESTIMATOR VALIDATION RUN")
    print("=" * 72)

    t0 = time.time()

    statistical_checks = [
        run_type_i_rate_calibration(),
        run_ci_coverage(),
        run_null_bias(),
    ]
    conservation_checks = [
        run_power_vs_effect_size(),
        run_power_vs_sample_size(),
        run_hac_correctness_ar1(),
    ]
    calibration_checks = [
        run_mde_calibration(),
    ]
    cases = run_case_walkthroughs()

    elapsed = time.time() - t0

    # Level 2
    print("\nLevel 2 — Statistical Sanity")
    print("-" * 72)
    for c in statistical_checks:
        status = "PASS" if c["pass"] else "FAIL"
        print(f"[{status}] {c['name']}")
        print(f"       observed = {c['observed']:.5f}")
        print(f"       expected = {c['expected']:.5f}")
        print(
            f"       4σ band  = "
            f"[{c['band_4sigma'][0]:.5f}, "
            f"{c['band_4sigma'][1]:.5f}]"
        )
        print(f"       n_reps   = {c['n_reps']}")
        if "empirical_se_of_estimator" in c:
            print(
                f"       empirical SE of estimator = "
                f"{c['empirical_se_of_estimator']:.5f}"
            )

    # Level 3
    print("\nLevel 3 — Conservation Laws")
    print("-" * 72)
    for c in conservation_checks:
        status = "PASS" if c["pass"] else "FAIL"
        print(f"[{status}] {c['name']}")
        if "powers" in c and "effects" in c:
            for e, p in zip(c["effects"], c["powers"]):
                print(f"       effect={e:+.4f}  power={p:.3f}")
        if "powers" in c and "n_posts" in c:
            for nps, p in zip(c["n_posts"], c["powers"]):
                print(f"       n_post={nps}  power={p:.3f}")
        if "ols_rate" in c:
            print(
                f"       OLS Type I rate  = {c['ols_rate']:.3f} "
                f"(nominal 0.05, AR(1) inflates)"
            )
            print(
                f"       HAC Type I rate  = {c['hac_rate']:.3f} "
                f"(should be < OLS)"
            )
            print(f"       OLS inflated?   = {c['ols_inflated']}")
            print(f"       HAC below OLS?  = {c['hac_below_ols']}")

    # Level 5 — calibration (reported as a finding)
    print("\nLevel 5 — Empirical MDE vs Protocol §5 Closed-Form")
    print("-" * 72)
    for c in calibration_checks:
        print(f"[FINDING] {c['name']}")
        print(
            f"    Planted effect        = "
            f"{c['planted_absolute']:+.5f} "
            f"({100*c['planted_rrr']:.1f}% RRR)"
        )
        print(
            f"    Empirical SD(beta2)   = "
            f"{c['empirical_sd_beta2']:.5f} "
            f"(over {c['n_reps']} seeds)"
        )
        print(
            f"    Mean reported SE      = "
            f"{c['mean_reported_se']:.5f} "
            f"(should match empirical SD)"
        )
        print(
            f"    Protocol §5 SE raw    = "
            f"{c['protocol_se_raw']:.5f} "
            f"(two-rate formula)"
        )
        print(
            f"    Protocol §5 SE ×1.4   = "
            f"{c['protocol_se_inflated']:.5f} "
            f"(with autocorr penalty)"
        )
        print(
            f"    Ratio emp/inflated    = "
            f"{c['se_ratio_vs_inflated']:.2f}x"
        )
        print("")
        print(
            f"    Empirical power @ planted = "
            f"{c['empirical_power_at_planted']:.3f}"
        )
        print("    Protocol target power     = 0.80")
        print("")
        print(
            f"    TRUE empirical MDE @ 80% power = "
            f"{c['empirical_mde_at_80pct']:.5f} absolute "
            f"({100*c['empirical_rrr_at_80pct']:.1f}% RRR)"
        )
        print("    Protocol-stated 24m MDE        = "
              "0.01100 absolute (22.9% RRR)")

    # Level 4 — case walkthroughs
    print("\nLevel 4 — Case Walkthroughs")
    print("-" * 72)
    for case in cases:
        print(f"\n{case['label']}")
        print(f"  {case['description']}")
        print(
            f"  pre_mean = {case['pre_mean']:.5f}  "
            f"post_mean = {case['post_mean']:.5f}"
        )
        if "ols" in case:
            for variant in ("ols", "hac"):
                d = case[variant]
                print(
                    f"  {variant.upper():3s}: level_change="
                    f"{d['level_change']:+.5f}  "
                    f"SE={d['level_change_se']:.5f}  "
                    f"p={d['level_change_pvalue']:.3f}  "
                    f"dir={d['direction']}"
                )
        else:
            print(
                f"  level_change={case['level_change']:+.5f}  "
                f"SE={case['level_change_se']:.5f}  "
                f"p={case['level_change_pvalue']:.3f}  "
                f"dir={case['direction']}"
            )
            if "pre_slope_pvalue" in case:
                print(
                    f"  pre_slope={case['pre_slope']:+.5f}  "
                    f"p={case['pre_slope_pvalue']:.3f}"
                )
            if "slope_change_pvalue" in case:
                print(
                    f"  slope_change="
                    f"{case['slope_change']:+.5f}  "
                    f"p={case['slope_change_pvalue']:.3f}"
                )

    # Summary
    print("\n" + "=" * 72)
    print("VERIFICATION SUMMARY")
    print("=" * 72)
    all_checks = (
        statistical_checks + conservation_checks + calibration_checks
    )
    passed = sum(1 for c in all_checks if c["pass"])
    total = len(all_checks)
    print(f"Statistical + conservation + calibration: {passed}/{total}")
    print(f"Case walkthroughs:                         {len(cases)} traced")
    print(f"Total runtime:                             {elapsed:.1f}s")
    if passed < total:
        print("\nFAILURES:")
        for c in all_checks:
            if not c["pass"]:
                print(f"  - {c['name']}")


if __name__ == "__main__":
    main()
