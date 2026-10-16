"""Rebuild the four reviewed narrative reports from complete validation sweeps.

Raw outputs stay in gitignored outputs/. Refuse partial sweeps. Commands and
fixed seed grids are documented in docs/issue_register_validation.md.
"""

from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from scipy.stats import t as student_t

from healthcare_sim_sdk.scenarios.nurse_retention.monitoring.tier2_cusum import (
    Tier2CUSUM,
    calibrate_h,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"


def cells(name, expected):
    paths = sorted((OUT / name).glob("*/metrics.json"))
    if len(paths) != expected:
        raise ValueError(f"{name}: expected {expected} cells, found {len(paths)}")
    return [
        (
            json.loads((p.parent / "config.json").read_text()),
            json.loads(p.read_text()),
            p.parent,
        )
        for p in paths
    ]


def mean_ci(values):
    x = np.asarray(values, float)
    half = student_t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return f"{x.mean():.2f} ({x.mean() - half:.2f}, {x.mean() + half:.2f})"


def alarm_weeks(run, h, raw=False):
    chart = Tier2CUSUM(
        "raw_turnover" if raw else "risk_adjusted_departures", h_multiplier=h
    )
    found = []
    for row in run["weekly_history"]:
        value = row["unit_turnover_rate"] if raw else row["risk_adjusted_departures"]
        if value is not None:
            found.extend(chart.update(row["week"], value))
    return [e.week for e in found]


def monitoring_report():
    cal = cells("issue_register_calibration_100", 100)
    calibration_runs = [
        json.loads((p / "monitoring_run.json").read_text()) for _, _, p in cal
    ]
    settings = OmegaConf.load(
        ROOT / "healthcare_sim_sdk/experiments/configs/cusum_calibration.yaml"
    )
    calibration = calibrate_h(
        [
            [r["risk_adjusted_departures"] for r in run["weekly_history"]]
            for run in calibration_runs
        ],
        list(settings.h_candidates),
        float(settings.max_horizon_alarm_probability),
    )
    h = calibration["h"]
    (OUT / "cusum_calibration_verified.json").write_text(
        json.dumps(calibration, indent=2)
    )
    primary = cells("issue_register_monitoring_verified", 180)
    large = cells("issue_register_monitoring_6800_verified", 60)
    groups = defaultdict(list)
    for config, metric, path in primary + large:
        run = json.loads((path / "monitoring_run.json").read_text())
        assert run["metadata"]["tier3_mode"] == "cits_with_control"
        assert (
            run["metadata"]["tier4_label_source"]
            == "realized_forward_factual_departures"
        )
        groups[(metric["n_nurses"], metric["regime"])].append((metric, run))
    # Fresh no-AI holdouts after expanding the calibration precision. No
    # chart parameter is chosen using these 30 unseen validation seeds.
    fresh_null = cells("issue_register_null_holdout", 30)
    groups[(1000, "null_program")] = [
        (m, json.loads((p / "monitoring_run.json").read_text()))
        for _, m, p in fresh_null
    ]
    lines = [
        "# Monitoring validation after the algorithm-review fixes",
        "",
        "**Re-run: 16 October 2026. This report replaces the pre-fix report.**",
        "",
        "The simulation suggests, under these assumptions, the operating characteristics below. "
        "The previous claims of zero Tier 3 false positives, universal early model drift, and linear power growth are withdrawn.",
        "",
        "The primary design has 1,000 nurses, 104 weekly steps, and predictions every two weeks. "
        "Each alternative regime uses seeds 1000–1029 (30 runs); the null row uses fresh holdouts 2000–2029. A separate control site uses seed + 1,000,000, "
        "SOC and no AI adoption. Its realized observations, rather than the oracle counterfactual, enter Tier 3. "
        "Model AUC is 0.75 on the fixed initial training reference; drift falls to 0.60. "
        "Deployed discrimination may change as the cohort and intervention change outcomes.",
        "",
        "## Detection by regime",
        "",
        "These are any-alarm fractions over the full 104-week horizon, including repeated looks. "
        "They are not per-test significance levels. Departures prevented are the oracle cumulative C−F contrast.",
        "",
        "| Regime | Runs | Mean prevented (95% CI of mean) | Tier 1 any | Tier 2 any | Tier 3 any | Tier 4 any |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for (n, regime), values in sorted(groups.items()):
        if n != 1000:
            continue
        counts = [
            sum(
                bool(
                    run["detection_events"]
                    and any(e["tier"] == tier for e in run["detection_events"])
                )
                for _, run in values
            )
            for tier in (1, 3, 4)
        ]
        t2 = sum(bool(alarm_weeks(run, h)) for _, run in values)
        lines.append(
            f"| {regime} | {len(values)} | {mean_ci([m['departures_prevented'] for m, _ in values])} | {counts[0]}/30 | {t2}/30 | {counts[1]}/30 | {counts[2]}/30 |"
        )
    null = groups[(1000, "null_program")]
    raw_alarms = sum(bool(alarm_weeks(run, 4, raw=True)) for _, run in null)
    adjusted = sum(bool(alarm_weeks(run, h)) for _, run in null)
    initial = np.mean(
        [
            np.mean([r["unit_turnover_rate"] for r in run["weekly_history"][:8]])
            for _, run in null
        ]
    )
    final = np.mean(
        [
            np.mean([r["unit_turnover_rate"] for r in run["weekly_history"][-26:]])
            for _, run in null
        ]
    )
    lines += [
        "",
        "## Tier 2: risk adjustment and calibrated limits",
        "",
        f"Raw-turnover CUSUM (k=0.5, h=4) alarms in {raw_alarms}/30 no-AI holdout runs. "
        f"Mean raw turnover declines from {initial:.4f} in the first eight weeks to {final:.4f} in the last 26. "
        "This is closed-cohort risk selection, not an AI effect. Null factual outcomes equal the oracle counterfactual here, "
        "so this also checks the raw chart on the counterfactual branch.",
        "",
        "The replacement charts a standardized observed-minus-predicted departure residual. Expected weekly counts use "
        "the last prospectively available calibrated scores, converted from the two-week horizon and restricted to the "
        "current observed risk set. No true risk or counterfactual is passed to this chart.",
        "",
        f"On independent calibration seeds 9000–9099, the smallest candidate whose Wilson 95% upper confidence bound meets a target of at most 10% "
        f"any-alarm probability over 104 weeks is **h={h:g}**, with k=0.5 and eight baseline observations. "
        f"Calibration rate: {calibration['calibration_rates'][h]:.0%}; holdout rate: **{adjusted}/30 ({adjusted / 30:.1%})**. "
        "This is a finite-horizon simulation calibration, not an estimate of an uncensored average run length. "
        "One hundred calibration runs and 30 holdout runs leave substantial uncertainty. Wide limits sacrifice sensitivity; "
        "risk adjustment does not guarantee a stationary residual process when model calibration changes.",
        "",
        "| Candidate h | Calibration any-alarm fraction | 95% upper bound |",
        "|---:|---:|---:|",
    ]
    lines += [
        f"| {key:g} | {value:.0%} | {calibration['calibration_upper_95'][key]:.1%} |"
        for key, value in calibration["calibration_rates"].items()
    ]
    lines += [
        "",
        "## Tier 3: observed control and direct population-size comparison",
        "",
        "The control is a separate no-AI site with independently observed noise. Oracle mode remains available only as "
        "an explicitly labeled benchmark and uses paired difference-series inference. Identical-series silence is not "
        "evidence of deployment specificity.",
        "",
        "The table is the final quarterly endpoint test: p<0.05 with a negative factual-minus-control estimate in the "
        "alternative; any direction under the null. This is empirical rejection probability for that endpoint estimand, "
        "not power for cumulative departures prevented. It is estimated directly at each population size.",
        "",
        "| Nurses | Alternative negative-direction rejections | Null two-sided rejections |",
        "|---:|---:|---:|",
    ]
    for n in (1000, 6800):
        hits = []
        for regime in ("calibrated_success", "null_program"):
            vals = groups[(n, regime)]
            hits.append(
                sum(
                    (
                        run["tier3_estimates"][-1]["p_value"] < 0.05
                        and (
                            regime == "null_program"
                            or run["tier3_estimates"][-1]["effect_estimate"] < 0
                        )
                    )
                    for _, run in vals
                )
            )
        lines.append(
            f"| {n:,} | {hits[0]}/30 ({hits[0] / 30:.1%}) | {hits[1]}/30 ({hits[1] / 30:.1%}) |"
        )
    paired = groups[(1000, "calibrated_success")]
    auc_differences = []
    for _, run in paired:
        f = run["metadata"]["tier4_paired_factual_rolling_stats"][-1]["realized_auc"]
        c = run["metadata"]["tier4_oracle_rolling_stats"][-1]["realized_auc"]
        if f is not None and c is not None:
            auc_differences.append(c - f)
    lines += [
        "",
        "Power cannot grow linearly with population size. Even the usual square-root standard-error scaling is "
        "only a heuristic here: closed-cohort selection changes the endpoint contrast, and finite independent sites "
        "can differ at baseline. These rejection rates do not establish unbiasedness or a validated deployment study design.",
        "",
        "## Tier 4: realized labels and performative change",
        "",
        "Only fully mature forecasts are scored: a prediction at week t is compared with observed departures in "
        "weeks t+1 through t+2, and becomes available at t+2. Fit labels are ignored. Right-censored forecasts at the end "
        "are omitted. Logistic recalibration regresses outcomes on logit(p), fitting both intercept and slope. "
        "The reference pools at least 12 weeks and 100 departures; it extends until the event requirement is met. "
        "Four consecutive evaluations represent eight weeks at this cadence. Overlapping windows make alarms dependent.",
        "",
        f"On the identical scores and common active risk set, the final-window oracle-minus-factual AUC difference "
        f"in the calibrated-success regime is **{mean_ci(auc_differences)}** across 30 runs. "
        "Positive values mean that the program's realized factual labels produce lower measured discrimination. "
        "This comparison is an oracle diagnostic; only factual labels feed the deployed drift monitor. "
        "A performance alarm is therefore not, by itself, proof that the model changed.",
        "",
        "**Operating limit:** Tier 4 still alarms in every no-AI holdout. Correct logistic "
        "recalibration and a larger reference do not make a closed cohort stationary. These "
        "default model-drift thresholds are not validated for operational use. Tier 3's "
        "endpoint test also has poor negative-direction detection despite a positive "
        "cumulative effect. Neither method should be described as deployment-ready from this study.",
        "",
        "## Reproduction and limits",
        "",
        "See [the validation protocol](../../docs/issue_register_validation.md). Raw artifacts are ephemeral under "
        "`outputs/issue_register_*_verified/`; this report contains reviewed aggregates. These are simulated populations, "
        "not clinical or workforce evidence. The cohort is closed; the control-site exchangeability, risk forecasts, "
        "effect sizes, calendar horizon, and operating thresholds all require local validation. The chart limits are "
        "calibrated at 1,000 nurses; the separate 6,800-nurse runs evaluate Tier 3 rather than claiming those limits transfer.",
    ]
    (REPORTS / "monitoring_validation/monitoring_validation.md").write_text(
        "\n".join(lines) + "\n"
    )
    return h


def retention_report():
    data = cells("issue_register_retention_verified", 360)
    groups = defaultdict(list)
    for c, m, _ in data:
        groups[(c["auc"], c["capacity"])].append(m)
    lines = [
        "# Nurse retention: model quality and manager capacity — revised",
        "",
        "**Re-run: 16 October 2026. Replaces the pre-label-free report.**",
        "",
        "The simulation suggests, under these assumptions, the following cumulative departure contrasts. Each cell has "
        "30 seeds (1000–1029), 1,000 nurses, and 52 weekly steps. All model fits use a larger independent bootstrap "
        "reference from the initial risk population, with two-week labels. Subsequent refits keep that reference; "
        "survivor selection does not redefine the configured training quality.",
        "",
        "| Reference AUC target | Capacity per manager | Mean departures prevented, C−F (95% CI of mean) | Mean delivered reference AUC | Mean fit positives |",
        "|---:|---:|---:|---:|---:|",
    ]
    for (auc, cap), rows in sorted(groups.items()):
        lines.append(
            f"| {auc:.2f} | {cap} | {mean_ci([m['prevented'] for m in rows])} | {np.mean([m['achieved_auc'] for m in rows]):.3f} | {np.mean([m['fit_events'] for m in rows]):.0f} |"
        )
    lines += [
        "",
        "The old AUC 0.80 and 0.85 cells are withdrawn: this risk distribution does not support those targets "
        "reliably. They are replaced by attainable reference targets rather than described as delivered model quality. "
        "Configured quality refers to the training reference, not guaranteed discrimination in a selected post-intervention cohort.",
        "",
        "**Estimand limit:** both branches receive SOC; the factual branch also allocates AI check-ins. The current "
        "scenario does not enforce a single shared SOC-plus-AI time budget. These contrasts therefore measure the "
        "configured AI augmentation, not pure targeting benefit at equal total manager hours. The earlier equal-time "
        "interpretation is withdrawn. Confidence intervals describe the mean across synthetic sites; they are not "
        "prediction intervals for an individual real deployment.",
        "",
        "See [reproduction commands](../docs/issue_register_validation.md). Raw cells are under "
        "`outputs/issue_register_retention_verified/`.",
    ]
    (REPORTS / "nurse_retention_sweep.md").write_text("\n".join(lines) + "\n")


def reminder_report():
    data = cells("issue_register_reminder", 60)
    groups = defaultdict(list)
    for c, m, _ in data:
        groups[c["reminder_study"]].append(m)
    lines = [
        "# Targeted reminders: revised proof-point checks",
        "",
        "**Re-run: 16 October 2026. Replaces the pre-label-free report.**",
        "",
        "The original configured study parameters are retained, including their previously calibrated intervention "
        "effect sizes. Models now fit on an independent reference sampled from the patient-risk distribution instead "
        "of one small daily schedule. Each study uses 30 seeds (1000–1029); no effect parameter is re-tuned to recover "
        "a published number. This tests the configured simulation, not the truth of the clinical effect.",
        "",
        "| Study configuration | Mean no-AI no-show rate | Mean factual no-show rate | Reduction, pp (95% CI of mean) | Delivered reference AUC | Gap narrowing, Black−White pp (95% CI) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, rows in sorted(groups.items()):
        lines.append(
            f"| {name} | {np.mean([m['counterfactual_rate'] for m in rows]):.1%} | {np.mean([m['factual_rate'] for m in rows]):.1%} | {mean_ci([m['reduction_pp'] for m in rows])} | {np.mean([m['achieved_auc'] for m in rows]):.3f} | {mean_ci([m['gap_narrowing_pp'] for m in rows])} |"
        )
    lines += [
        "",
        "Positive gap narrowing means a larger absolute reduction among Black than White patients. Every "
        "appointment is paired to its no-AI counterpart, and the runner asserts that reminders do not increase "
        "any appointment’s no-show outcome. That monotonicity does not guarantee that a racial gap narrows in "
        "each finite seed, nor that increasing AUC improves every equity metric. Previous single-seed assertions "
        "of guaranteed disparity reduction are withdrawn. The Rosen demographic multipliers and effect sizes "
        "are assumed/calibrated inputs, not independently validated site facts.",
        "",
        "Chong runs 3,000 patients over 90 days; Rosen runs 5,000 over 180 days, using the tracked replication YAMLs. "
        "Rates pool resolved appointment slots, including the initial untreated schedule. See "
        "[reproduction commands](../docs/issue_register_validation.md). Raw cells: `outputs/issue_register_reminder/`.",
    ]
    (REPORTS / "noshow_targeted_reminders_proof_points.md").write_text(
        "\n".join(lines) + "\n"
    )


def trews_report():
    data = cells("issue_register_trews", 28)
    lookup = {(int(c["capacity"]), int(c["max_hours"])): m for c, m, _ in data}
    lines = [
        "# TREWS-style baseline detection and capacity sweep — revised",
        "",
        "**Re-run: 16 October 2026. Replaces the pre-label-free report.**",
        "",
        "This reproduces the original 28-cell grid with label-free scoring and an independent reference fit. "
        "Each cell uses seed 42, 5,000 admissions, 42 four-hour steps, model AUC target 0.82, top-7% alerts, "
        "and the unchanged TREWS replication assumptions. This is a sensitivity illustration with one seed per "
        "cell; it provides no Monte Carlo confidence interval and does not establish a faithful TREWS replication.",
        "",
        "Entries are final cumulative whole-admission mortality differences, C−F, in percentage points. "
        "They are neither sepsis-patient mortality differences nor a sum of repeated mortality occupancy.",
        "",
        "| Capacity per four-hour step | 6h max | 12h max | 18h max | 24h max | 36h max | 48h max | 72h max |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cap in (10, 25, 50, 100):
        lines.append(
            "| "
            + str(cap)
            + " | "
            + " | ".join(
                f"{lookup[cap, h]['reduction_pp']:.2f}"
                for h in (6, 12, 18, 24, 36, 48, 72)
            )
            + " |"
        )
    lines += [
        "",
        "The baseline clinical delay is Beta(2,5) × the column maximum; SOC runs on both branches. Kumar "
        "decay links treatment timing to efficacy. The model scores risk level and has no feature-level "
        "knowledge of proximity to sepsis onset. Therefore any numerical agreement with a published mortality "
        "effect would not validate the clinical detection mechanism. The historical replication conclusion and "
        "its exact pre-fix numbers are withdrawn. Sepsis `events` are patient-time occupancy; use `incident_onset` "
        "for case counts.",
        "",
        "See [reproduction commands](../docs/issue_register_validation.md). "
        "Raw cells: `outputs/issue_register_trews/`.",
    ]
    (REPORTS / "trews_baseline_detection_sweep.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    h = monitoring_report()
    retention_report()
    reminder_report()
    trews_report()
    print(f"Updated four reports; calibrated h={h:g}")
