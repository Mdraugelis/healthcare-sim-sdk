# Monitoring validation after the algorithm-review fixes

**Re-run: 16 October 2026. This report replaces the pre-fix report.**

The simulation suggests, under these assumptions, the operating characteristics below. The previous claims of zero Tier 3 false positives, universal early model drift, and linear power growth are withdrawn.

The primary design has 1,000 nurses, 104 weekly steps, and predictions every two weeks. Each alternative regime uses seeds 1000–1029 (30 runs); the null row uses fresh holdouts 2000–2029. A separate control site uses seed + 1,000,000, SOC and no AI adoption. Its realized observations, rather than the oracle counterfactual, enter Tier 3. Model AUC is 0.75 on the fixed initial training reference; drift falls to 0.60. Deployed discrimination may change as the cohort and intervention change outcomes.

## Detection by regime

These are any-alarm fractions over the full 104-week horizon, including repeated looks. They are not per-test significance levels. Departures prevented are the oracle cumulative C−F contrast.

| Regime | Runs | Mean prevented (95% CI of mean) | Tier 1 any | Tier 2 any | Tier 3 any | Tier 4 any |
|---|---:|---:|---:|---:|---:|---:|
| calibrated_success | 30 | 27.27 (25.65, 28.89) | 0/30 | 0/30 | 12/30 | 30/30 |
| capacity_collapse | 30 | 25.63 (24.04, 27.22) | 30/30 | 1/30 | 11/30 | 30/30 |
| gradual_decay | 30 | 6.10 (5.30, 6.90) | 0/30 | 1/30 | 18/30 | 30/30 |
| model_drift | 30 | 21.87 (20.05, 23.68) | 0/30 | 3/30 | 12/30 | 30/30 |
| null_program | 30 | 0.00 (0.00, 0.00) | 0/30 | 0/30 | 8/30 | 30/30 |
| partial_adoption | 30 | 13.43 (11.06, 15.81) | 0/30 | 1/30 | 8/30 | 30/30 |

## Tier 2: risk adjustment and calibrated limits

Raw-turnover CUSUM (k=0.5, h=4) alarms in 30/30 no-AI holdout runs. Mean raw turnover declines from 0.0101 in the first eight weeks to 0.0052 in the last 26. This is closed-cohort risk selection, not an AI effect. Null factual outcomes equal the oracle counterfactual here, so this also checks the raw chart on the counterfactual branch.

The replacement charts a standardized observed-minus-predicted departure residual. Expected weekly counts use the last prospectively available calibrated scores, converted from the two-week horizon and restricted to the current observed risk set. No true risk or counterfactual is passed to this chart.

On independent calibration seeds 9000–9099, the smallest candidate whose Wilson 95% upper confidence bound meets a target of at most 10% any-alarm probability over 104 weeks is **h=96**, with k=0.5 and eight baseline observations. Calibration rate: 3%; holdout rate: **0/30 (0.0%)**. This is a finite-horizon simulation calibration, not an estimate of an uncensored average run length. One hundred calibration runs and 30 holdout runs leave substantial uncertainty. Wide limits sacrifice sensitivity; risk adjustment does not guarantee a stationary residual process when model calibration changes.

| Candidate h | Calibration any-alarm fraction | 95% upper bound |
|---:|---:|---:|
| 4 | 74% | 81.6% |
| 6 | 51% | 60.6% |
| 8 | 42% | 51.8% |
| 12 | 32% | 41.7% |
| 16 | 28% | 37.5% |
| 24 | 21% | 30.0% |
| 32 | 13% | 21.0% |
| 48 | 8% | 15.0% |
| 64 | 5% | 11.2% |
| 96 | 3% | 8.5% |

## Tier 3: observed control and direct population-size comparison

The control is a separate no-AI site with independently observed noise. Oracle mode remains available only as an explicitly labeled benchmark and uses paired difference-series inference. Identical-series silence is not evidence of deployment specificity.

The table is the final quarterly endpoint test: p<0.05 with a negative factual-minus-control estimate in the alternative; any direction under the null. This is empirical rejection probability for that endpoint estimand, not power for cumulative departures prevented. It is estimated directly at each population size.

| Nurses | Alternative negative-direction rejections | Null two-sided rejections |
|---:|---:|---:|
| 1,000 | 0/30 (0.0%) | 2/30 (6.7%) |
| 6,800 | 0/30 (0.0%) | 1/30 (3.3%) |

Power cannot grow linearly with population size. Even the usual square-root standard-error scaling is only a heuristic here: closed-cohort selection changes the endpoint contrast, and finite independent sites can differ at baseline. These rejection rates do not establish unbiasedness or a validated deployment study design.

## Tier 4: realized labels and performative change

Only fully mature forecasts are scored: a prediction at week t is compared with observed departures in weeks t+1 through t+2, and becomes available at t+2. Fit labels are ignored. Right-censored forecasts at the end are omitted. Logistic recalibration regresses outcomes on logit(p), fitting both intercept and slope. The reference pools at least 12 weeks and 100 departures; it extends until the event requirement is met. Four consecutive evaluations represent eight weeks at this cadence. Overlapping windows make alarms dependent.

On the identical scores and common active risk set, the final-window oracle-minus-factual AUC difference in the calibrated-success regime is **0.02 (0.02, 0.03)** across 30 runs. Positive values mean that the program's realized factual labels produce lower measured discrimination. This comparison is an oracle diagnostic; only factual labels feed the deployed drift monitor. A performance alarm is therefore not, by itself, proof that the model changed.

**Operating limit:** Tier 4 still alarms in every no-AI holdout. Correct logistic recalibration and a larger reference do not make a closed cohort stationary. These default model-drift thresholds are not validated for operational use. Tier 3's endpoint test also has poor negative-direction detection despite a positive cumulative effect. Neither method should be described as deployment-ready from this study.

## Reproduction and limits

See [the validation protocol](../../docs/issue_register_validation.md). Raw artifacts are ephemeral under `outputs/issue_register_*_verified/`; this report contains reviewed aggregates. These are simulated populations, not clinical or workforce evidence. The cohort is closed; the control-site exchangeability, risk forecasts, effect sizes, calendar horizon, and operating thresholds all require local validation. The chart limits are calibrated at 1,000 nurses; the separate 6,800-nurse runs evaluate Tier 3 rather than claiming those limits transfer.
