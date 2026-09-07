# Nurse retention monitoring evidence

The monitoring runner now writes a review packet for the exact configured run:

```bash
python -m healthcare_sim_sdk.scenarios.nurse_retention.monitoring_validation \
  regime=capacity_collapse tier3_mode=its_only
```

Open `outputs/monitoring_validation/capacity_collapse_seed1000/review/review_packet.md`.
The packet includes clinical, operations, statistics, governance and engineering
views, a configuration/source identity, artifact hashes, machine evidence and
human review notes. A single-run packet explicitly leaves repeated-run monitor
certification and human workflow review pending.

## Interpret the comparison

The current scenario assigns each manager a separate SOC check-in budget and an
additional AI check-in budget on scoring weeks. Reviewers must approve whether
that additional-resource comparison reflects their question. It is not an
allocation-only comparison under equal total manager time. Dashboard adherence
now includes both potential budgets on scoring weeks. Recorded adoption uses
actual manager behavior; unknown historical adoption remains unknown.

`cits_with_cf` estimates an oracle factual-versus-counterfactual rate difference
at the current endpoint. `its_only` consumes factual data alone and reports its
turnover trend. Without a pre-period or valid control, that trend cannot identify
a causal program effect. Observable-only harness runs do not access the
counterfactual payload; optional ground-truth retention must be requested
explicitly with `include_counterfactual_ground_truth=True`.

## Define events before evaluating alerts

Regime YAML files declare `monitoring_claims`: claim ID, tier, metric, direction,
change week, and eligible start. The runner supplies the inclusive horizon
`n_weeks - 1`. Linear schedules use the first discrete week whose configured
value actually differs from baseline: week 27 for efficacy decay and week 21 for
AUC drift in the supplied regimes.

`monitoring.evaluation.evaluate_claim()` reports pre-change alarms separately,
requires the right metric and direction after the change and eligibility start,
and records detection delay and censored nondetections. Missing observations
are inconclusive. A horizon ending before the monitor becomes eligible is
`NOT_RUN`. No-change claims count full-horizon false alarms in either direction.

The returned `evaluation_status` describes whether the event evaluation is
complete. It does not say the monitor successfully detected a failure. Inspect
`detected`, `pre_change_alert_count`, `detection_delay_weeks` and
`right_censored`. The old `tier*_first_detection_week` fields remain available
with an explicit legacy label: they are simply the first alert of any kind.

`summarize_claim_evaluations()` requires matching claim definitions and regime,
and rejects repeated seeds. It reports detection or false-alert probability,
Monte Carlo standard error, an exact one-sided 95% binomial upper bound,
incomplete runs, censored nondetections, and delay conditional on detection.
Use seeds independent of monitoring tuning. The utility does not itself certify
that tuning and validation were independent.

## Distinguish model fidelity questions

The score generator uses synthetic prediction-window labels in generating its
scores. Its calibration fit and per-scoring-window diagnostic therefore do not
establish performance against subsequently realized departures.

The runner additionally checks the frozen final scorer on an independently
generated initial population, using the same `predict()` path and a separate
partitioned seed (default simulation seed + 10,000,000). It never refits or
consumes the original run's RNG streams. This checks a synthetic-label scoring
mechanism on that independent cohort; it does not certify every population or
time-varying period of the simulation.

Tier 4 now uses actual factual departure events in the future window
`score_week + 1` through `score_week + prediction_interval`. The label becomes
available only when the entire window has matured. Terminal incomplete windows
remain `NOT_RUN`, and insufficient positive/negative outcomes produce
`INCONCLUSIVE`. AUC checks report their absolute tolerance and minimum class
counts; that engineering tolerance is not a confidence interval. Rank-based AUC
includes ties, and zero is retained as a valid score.

Calibration slope and intercept now come from logistic regression of observed
outcomes on the logit score. The previous linear regression did not estimate
that calibration quantity. The persistence rule counts consecutive observed
prediction windows, explicitly labeled as observations rather than calendar
weeks; its old parameter name remains a compatibility alias.

## Tests and interpretation limits

Fixed-length temporal random priorities are drawn for all nurses before SOC
eligibility masking. Departures, tenure, cooldown and capacity differences no
longer shift subsequent temporal RNG consumption. Tests deliberately diverge
those states and verify RNG-state equality, shared AR(1) noise, exact null-effect
branch outcomes and separate-budget capacity accounting.

Monitoring tests include hand-worked event windows, wrong-direction alarms,
late/incomplete outcomes, falsified counterfactual payloads, incorrect fit
claims, perfect calibration data and artifact tampering. Actual packet case
stories include a seeded random nurse and explicitly selected favorable,
adverse and neutral categories. They link to the actual event/score/action
records, disclose absent categories, and do not reconstruct unrecorded initial
state or decision ranks.

An adoption chart trained on a consistently low-adoption baseline can remain
quiet despite poor adherence. Detecting an absolute shortfall requires a
separately specified operational target. Full-regime validation sweeps,
pre-specified latency/probability acceptance targets, scenario certification
across supported ranges and human workflow confirmation remain necessary before
claiming deployment monitoring is adequate.
