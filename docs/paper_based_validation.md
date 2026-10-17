# Paper-Based Validation and Quasi-Experimental Testing

**Date:** 17 October 2026

**Status:** Proposed approach

The SDK should be allowed to predict a different outcome from a published study
when the discrepancy is explainable within independently justified assumptions.
The simulation must still be internally consistent and numerically faithful to
its declared model. Assumption ranges and evaluation criteria must be fixed
before the published outcome is revealed.

This approach separates two questions:

1. **External prediction:** Can independently justified inputs predict a held-out
   published result, with useful uncertainty bounds?
2. **Evaluation design:** Can an analysis of simulated observational data recover
   the simulator's hidden causal effect?

Passing software tests and repeating runs support implementation verification.
Independent agreement with published outcomes requires the additional safeguards
below. The [completed issue-register reruns](issue_register_validation.md) remain
a record of those implementation checks and configured simulation results.

## 1. Reclassify the existing paper reproductions

A match obtained by fitting an intervention parameter to the published outcome
is calibration. That same outcome cannot independently validate the fitted
parameter. The current [Chong configuration](../healthcare_sim_sdk/scenarios/noshow_targeted_reminders/configs/chong_replication.yaml)
explicitly calibrates reminder effectiveness to the reported intervention rate.
The [configuration labeled Rosen](../healthcare_sim_sdk/scenarios/noshow_targeted_reminders/configs/rosen_replication.yaml)
also calibrates effectiveness and call capacity against reported benefits.

The 30-entry research audit should become a registry of distinct evidence roles:

| Paper type | Appropriate role |
|---|---|
| Randomized intervention study | Predict held-out intervention outcomes while preserving the randomized design |
| Observational deployment or quasi-experimental study | Reconstruct its comparison, timing, adjustment, and confounding assumptions |
| Prediction or diagnostic validation study | Check discrimination, calibration, operating points, and burden; do not infer an unmeasured clinical benefit |
| Methods paper | Supply mathematical identities, estimator benchmarks, and verification requirements |
| Review, framework, or qualitative study | Inform assumptions, mechanisms, and reporting requirements, with explicit limits on transfer |

Do not force every audit entry into a clinical-outcome experiment. The existing
audit also combines two reminder papers in one entry; the new registry should
identify each paper, version, population, and comparison separately.

## 2. Separate sourced parameters from assumptions

Every parameter needs an evidence label and an experiment role. A reported
estimate is evidence about a particular population and period, with measurement
uncertainty. Its applicability to a different setting requires justification.

| Evidence label | Meaning | Permitted use |
|---|---|---|
| Reported in this study | Directly documented in the primary paper, supplement, protocol, or analysis code | Input when its cohort, time horizon, and role match the benchmark |
| Sourced elsewhere | Documented in an independent study | Input with explicit transport assumptions and uncertainty |
| Mathematically derived | Calculated from permitted inputs using a stated formula | Input with the derivation and its assumptions recorded |
| Assumed | Not identified by available evidence | Prespecified range, distribution, or alternative mechanism |
| Calibrated to an outcome | Estimated by matching an observed result | Calibration evidence; cannot validate against that same result |
| Held-out outcome | Reserved for comparison after the model is frozen | Evaluator only; excluded from simulator configuration and tuning |

For each entry, record the SDK field or mechanism, units, eligible population,
time horizon, exact source location, uncertainty, dependence on other parameters,
and whether it is available to the modeler. An apparent derivation that uses the
held-out outcome belongs in the calibration category.

### Initial source checks

These selected studies illustrate the required extraction. They are an initial
check of core examples, not a completed re-extraction of all 30 audit entries.

| Study | Reported information that can inform inputs | Assumptions still required | Outcomes to withhold |
|---|---|---|---|
| [Tarabichi et al., targeted reminders](https://pmc.ncbi.nlm.nih.gov/articles/PMC10150669/) | 5,840 eligible appointments; 2,858 augmented-care and 2,982 standard-care appointments; predicted-risk threshold at least 15%; calls 3–5 business days beforehand; automated reminders in both arms; pretrial sensitivity 0.76 and specificity 0.55 | Benefit attributable to contact, risk distribution within eligible appointments, repeated-appointment dependence, and operational capacity | Overall and subgroup no-show reductions |
| [Hong et al., SHIELD-RT](https://pubmed.ncbi.nlm.nih.gov/32886536/) | 963 screened treatment courses; 311 high-risk courses randomized, with 157 control and 154 intervention; risk threshold above 10%; weekly versus twice-weekly evaluation; AUC 0.851 in the standard-care population | Effect of additional evaluation, course-duration variation, treatment-effect heterogeneity, and score distribution | Acute-care rates and their risk ratio |
| [Escobar et al., Kaiser AAM](https://pmc.ncbi.nlm.nih.gov/articles/PMC7787261/) | Nineteen rollout hospitals; hourly scores predicting 12-hour deterioration; alert score 5 corresponds to risk at least 8%; remote nurse review followed by ward response; mortality within 30 days of the first alert | Response probabilities and delays, rescue effectiveness, site differences, and concurrent changes in care | Adjusted mortality effect |
| [Lång et al., MASAI 2023 analysis](https://pubmed.ncbi.nlm.nih.gov/37541274/) | Randomized screening design; Transpara score categories 1–9 route to single reading and category 10 to double reading | Cancer prevalence including undetected disease, reader-error dependence, and mapping commercial score categories to latent risk | Cancer detection, recalls, false positives, and reading workload |
| [Adams et al., TREWS](https://www.nature.com/articles/s41591-022-01894-0) | Five hospitals; primary cohort of 6,877 patients with sepsis alerted before antibiotics; comparison based on confirmation within three hours versus later or no confirmation | Treatment-delay effects, severity-dependent confirmation, clinician behavior, and residual confounding | Adjusted mortality association for that comparison |

The paper matching the figures used in the scenario labeled “Rosen/VA” is
Tarabichi et al. at MetroHealth. It reports information that the audit marked as
missing, including a threshold and contact rates. Contact outcomes were collected
for a later subset; using those rates as inputs would make a prediction
conditional on observed delivery. SHIELD-RT's reported AUC also differs from the
audit's 0.81 assumption. Bibliographic identity and parameter provenance need
correction before these cases can support independent reproduction claims.

### Constraints on direct SDK inputs

- **Keep cohorts aligned.** A model operating point from a pretrial validation
  population cannot automatically be combined with an outcome prevalence from a
  selected high-risk intervention cohort.
- **Keep score meanings aligned.** A commercial category, percentile, and
  calibrated probability are different quantities. Their mapping needs evidence
  or an explicit assumption.
- **Keep outcomes and underlying disease distinct.** A screen-detected cancer
  rate is not the prevalence of all cancers, including missed cancers.
- **Keep time horizons aligned.** A course-level risk needs a justified
  conversion before it becomes a weekly or hourly event probability.
- **Use the reported model evidence.** Missing AUC does not erase a reported
  operating point. A fixed binary decision may be studied from that evidence;
  ranking and capacity-limited targeting require additional score-distribution
  assumptions.

## 3. Run a blinded external prediction experiment

### Specify the target study

Before configuration, fix eligibility, assignment, usual care, time zero,
follow-up, outcome definition, and the effect being estimated. Preserve
randomization for an RCT. For an observational study, reconstruct the actual
comparison and its adjustment assumptions. This specification follows the
[target-trial framework](https://pubmed.ncbi.nlm.nih.gov/26994063/).

For TREWS, confirmation within three hours versus later or no confirmation is a
different comparison from deploying AI versus not deploying AI. The benchmark
must state which question it addresses.

### Separate inputs, observations, and grading

Maintain three distinct packets:

1. **Input packet:** permitted population, model, workflow, and natural-history
   evidence, plus prespecified assumptions.
2. **Observed-data packet:** simulated records and assigned exposures that a
   real evaluator could observe.
3. **Grading packet:** held-out published outcomes and, for internal method
   experiments, latent risks and simulator counterfactuals.

A separate custodian controls the grading packet. Withhold the terminal outcome
and every statistic or coefficient that reconstructs it. Check existing scenario
code, comments, reports, and configuration files for outcome-derived values;
separating files alone does not remove leakage.

If control-arm outcomes establish the baseline, label the exercise a conditional
prediction of the intervention arm. If observed contact or adoption rates are
inputs, identify the prediction as conditional on that delivery process.

The current papers and their outcomes have already been examined and used in
configuration. Rebuilding them now can provide a retrospectively masked
benchmark. A credible new blinded test needs a previously untouched paper,
cohort, site, or later study period, with the prediction frozen before its target
outcome is released to the modeling process.

### Freeze uncertainty and the analysis plan

Before unblinding, record:

- The input evidence, independently justified assumption ranges or distributions,
  correlations among inputs, and alternative plausible mechanisms.
- The outcome, estimator, eligible cohort, sampling design, adjustment variables,
  and treatment of missingness, repeated observations, and censoring.
- Required internal checks, model-performance tolerances, discrepancy criteria,
  and a limit on how broad an acceptable prediction may be.
- Code revision, configuration, environment, seed scheme, and repetition rule.

Do not infer a clinical treatment effect from AUC and capacity alone. If efficacy
has no independent evidence, the appropriate conclusion may be that the outcome
is underdetermined. Fitting efficacy to the held-out result changes the exercise
into calibration.

### Simulate the study and unblind once

Generate studies with the paper's enrollment units, assignment, calendar periods,
clustering, follow-up, and analysis. An actual study's control arm should consist
of simulated observed units; an oracle branch must not be presented to the
analyst as an independently observed control site.

Vary uncertain parameters and mechanisms separately from random seeds. Form a
predictive distribution of the same statistic the paper reports from studies of
the corresponding size. Account for sampling uncertainty once; do not add the
paper's sampling error again if the simulated study design already represents it.

After internal verification passes, reveal the published result once. Report
its discrepancy from the prediction and the prediction's width. Any subsequent
tuning produces a new calibrated model requiring a fresh holdout.

## 4. Test quasi-experimental analyses against hidden causal truth

Use the simulator to test whether an evaluation method can recover a known
effect when it sees only observable records. This is a separate experiment from
predicting a paper's clinical result. Deliberately chosen zero and nonzero
effects are test conditions, not independently established clinical parameters.

Kaiser AAM is a useful first case. Represent hospital-specific rollout dates,
ordinary care, patient severity, calendar trends, and review and response
processes. Preserve calendar time across the rollout and patient time within an
admission. Do not compress a multiyear deployment comparison into one short
patient trajectory.

Reconstruct the published adjusted comparison as one analysis. Prespecify a
controlled interrupted time series or an appropriate staggered-rollout
difference-in-differences analysis as a separate candidate design. Give each
analysis the observed-data packet while keeping latent states and paired
counterfactuals in the grading packet.

Test at least these conditions:

| Condition | Purpose |
|---|---|
| No program effect with common secular improvement | Check false attribution to the intervention |
| Known beneficial effect with balanced rollout | Check bias, precision, and detection under favorable identification conditions |
| Rollout related to patient severity or hospital readiness | Check vulnerability to confounding |
| Concurrent care improvement or different site trends | Check whether the comparison can separate overlapping changes |
| Limited adherence, capacity, or delayed response | Check the operational conditions under which the design remains useful |

Measure bias against the simulator's corresponding causal effect, interval
coverage, false-positive probability, and power. Keep the published adjusted
association separate from simulator causal truth: reproducing an observational
association does not by itself establish that the same causal effect generated it.

## 5. Make internal correctness a prerequisite

A run that violates its declared mathematics or accounting is ineligible for
paper comparison. Required gates for this proposed benchmark include:

| Area | Required check |
|---|---|
| Entity and event accounting | Reconcile arrivals, active entities, absorbing outcomes, exits, appointment IDs, and repeated observations; enforce valid state transitions |
| Numerical integrity | Correct shapes and units, finite values, bounded probabilities, and consistent hazard-to-probability conversion |
| Resource accounting | Respect declared capacity and budget across ordinary care and AI activity, including displacement or additional funded capacity |
| Model fidelity | Deliver required performance within a prespecified measurement tolerance on an appropriate independent reference; do not silently substitute infeasible requirements |
| Causal and temporal integrity | Prevent future-outcome leakage, preserve branch isolation and required RNG coupling, and use only available information in predictions and decisions |
| Mathematical benchmarks | Recover analytically solvable cases and valid null boundaries; apply monotonicity and untreated-identity checks only where their assumptions hold |
| Case walkthroughs | Trace representative entities through risk, prediction, action, resource use, and observed outcomes |

Warnings that a required paper condition was not delivered must block a claim
that the run reproduces that condition. A different, explicitly labeled scenario
may still be informative. Bounds on assumptions cannot excuse arithmetic errors,
inconsistent cohort definitions, or violation of a declared resource budget.

These checks should preserve the SDK's existing five-method scenario contract,
opaque engine state, and five RNG streams. Put domain assumptions and checks in
scenario configuration, scenario logic, and validation infrastructure.

## 6. Choose repetitions by the required precision

The number of papers and the number of Monte Carlo repetitions answer different
questions. Repeating one configuration estimates behavior under that model; it
does not create another independent empirical validation case.

For an estimated probability with true value approximately p, its Monte Carlo
standard error from R independent repetitions is approximately:

```text
MCSE = sqrt(p * (1 - p) / R)
```

For a 5% false-positive probability, 30 repetitions give an MCSE of about 4
percentage points; 2,000 give about 0.49 percentage points. These are illustrative
precision calculations, not a requirement to use 2,000 repetitions for every
scenario. Choose the repetition count for the quantity and precision needed,
and report its Monte Carlo error. The planning framework in
[Morris, White, and Crowther](https://pubmed.ncbi.nlm.nih.gov/30652356/) separates
aims, data-generating mechanisms, estimands, methods, and performance measures.

Use separate variation for stochastic outcomes, uncertainty in sourced and
assumed parameters, and alternative structural mechanisms. A narrow confidence
interval around the simulator's mean is not a narrow predictive interval for a
real deployment.

## 7. Report results without forcing a match

| Result | Interpretation |
|---|---|
| Internal verification failure | Correct the implementation or inconsistent configuration before external interpretation |
| Compatible and informative | The held-out result is compatible with the frozen prediction, whose precision meets the prespecified usefulness criterion |
| Compatible but uninformative | Assumption bounds are too broad to provide a meaningful test |
| Outside prespecified bounds | Diagnose the discrepancy; preserve the original result and do not widen bounds afterward to declare success |
| Underdetermined | Independent evidence does not identify the mechanism sufficiently to make the proposed prediction |
| Calibrated | Outcome information was used to estimate inputs; require a different holdout for independent validation |

An explanation for a mismatch should identify the assumptions responsible,
their sensitivity, and whether agreement would require values outside the
prespecified evidence bounds. It can motivate a revised model without converting
the original unsuccessful prediction into a successful one.

## 8. Start with two complementary benchmarks

**Tarabichi provides the first randomized workflow benchmark.** Correct the paper
identity and cohort mapping, preserve the appointment-level assignment and
existing automated reminders, and separate model targeting, contact, and benefit.

**AAM provides the first quasi-experimental benchmark.** Reconstruct the rollout,
time origin, usual-care comparison, and analysis; test candidate estimators
against hidden simulator effects under explicit threats to identification.

For each benchmark, produce an evidence ledger, an assumption ledger, a frozen
experiment specification, an internal verification report, and a held-out
comparison. Extend this process across the collection only after those first
cases establish a repeatable standard.

The governing rule is: **an external outcome mismatch can be scientifically
explainable; an internal inconsistency fails verification.**
