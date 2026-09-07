# Verification and human review implementation

Date: 2026-09-05. Baseline: `648c56ce330981449435a10201f2116f2f111c5c`.
Implementation branch: `codex/verification-review-packets`.
At the time of this implementation review, the refreshed `origin/main` matched
this baseline. This record describes the verification work completed on that
baseline; later benchmark and documentation changes have their own records.

The proposed verification structure is implemented with working malnutrition
and nurse-retention review packets, a reusable packet API, actual-engine traces,
strict evidence gates and independent statistical design tests. These changes
make the evidence for a simulation reviewable. They do not turn synthetic
results into clinical evidence or supply a human approval.

## Start your review

| Artifact | Why to open it |
|---|---|
| [Malnutrition review packet](../outputs/review_demo/malnutrition/review_packet.md) | Nine actual cell/seed runs covering usual care, ranked worklists and severe capacity constraints; five role views and helped/harmed case examples |
| [Retention monitoring packet](../outputs/retention_assurance_capacity_review/review/review_packet.md) | A 52-week capacity-loss example with observable-only monitoring, actual label maturity and event-specific detection |
| [Statistical design evidence](../reports/analysis_design_demo/design_evidence.md) | 1,200 planted-truth fits showing where ITS inference passes, fails or lacks Monte Carlo precision |
| [Review workflow guide](review_packets.md) | Status meanings, reviewer responsibilities, reproducibility and adoption by additional scenarios |
| [Executed notebook](../reports/sanity_checks_executed.ipynb) | Actual-engine no-show and stroke trajectories with outcome reconciliation |

Generated examples are local ignored outputs; source, public configuration,
tests and documentation are durable repository changes. The commands below
recreate the examples without institutional patient data.

## What changed and why

| Earlier weakness | Implemented behavior | Practical effect |
|---|---|---|
| Passing structural files could be mistaken for validated analysis | Explicit PASS, FAIL, NOT_RUN, INCONCLUSIVE and justified NOT_APPLICABLE; missing required evidence and invalid JSON cannot pass | Reviewers see the evidence gap before relying on a result |
| A separate center-cell check stood in for an entire sweep | Malnutrition runners check every actual seed/cell and both branches; missing/failed runs are recorded | The packet refers to the results being interpreted |
| Certificates could outlive code/model/config changes | Exact source/config/model/environment identities and artifact checks; per-run pre/post model and configuration binding | A stale or incompatible certificate cannot silently validate a new experiment |
| Human review mixed generic checklists and selected examples | Five role pages sharing a claim registry, assumptions, selected cases and a complete case appendix | Reviewers have specific questions and traceable evidence while retaining a common record |
| Analysis exports could select wrong outcomes or treatment periods | Correct secondary outcome selection, entity-ID alignment, per-entity treatment starts and explicit missing observations | Downstream estimators receive the intended outcome and exposure timing |
| Appointment day zero reused IDs and appeared twice | Initial day is scheduled once and first resolves on the next day | N appointment timesteps yield N−1 resolved days; raw outcomes reconcile with counters |
| Retention branches could consume different temporal draws | Fixed-length random priorities before eligibility masking, plus explicit SOC action records | Paired branch comparisons preserve their intended shared randomness |
| Monitoring could count pre-change alarms as successful detection | Declared tier/metric/direction/change/eligibility rules; separate false alarms, delays and censored nondetections | A monitor is judged against the event it was meant to detect |
| Predicted synthetic labels were used as future outcomes | Tier 4 uses actual fully matured departure windows; frozen score checks disclose label provenance and holdout scope | Model-generation fidelity is distinct from observed future-outcome performance |
| Calibration and fit metrics were easy to overinterpret | Logistic calibration and independent exact-rank AUC; sepsis fit/predict conditioning is consistent | Numerical diagnostics refer to their declared quantity and prediction target |
| Seed percentiles were called confidence intervals and missing values could disappear | Separate seed spread, MCSE and mean interval; exact binomial uncertainty; missing files/undefined estimands rejected | Simulation variability, simulation precision and study uncertainty remain distinct |
| Boundary rules made universal causal assumptions | Mechanism-specific eligibility, capacity, null and monotonicity claims | An impossible generic expectation no longer masquerades as a correctness test |
| Architectural expectations were partly documentary | YAML-backed AST guards, negative fixtures and GitHub Actions test/JUnit artifacts | Contract regressions and skipped evidence are visible in automation |

## What the examples show

The malnutrition demo completes all nine planned runs. Its usual-care identity
cell produces zero paired benefit, its ranked-worklist cell has both helpful
and harmful individual examples, and its capacity-stress cell leaves no
assessments available to ML ranking after higher-priority work. These are
conditional behavior demonstrations. Three seeds do not establish stable
clinical effect estimates, equity or model calibration.

The retention example models a capacity drop at week 30. A qualifying Tier 1
alert appears at week 36, a six-week delay. Earlier unrelated alerts are recorded
separately. One run cannot certify a monitor's detection probability or
false-alert rate. Terminal prediction windows and sparse realized outcomes
remain explicitly incomplete.

The analysis-design demonstration completes six designs, each with 100 null
and 100 planted-effect repetitions. Autocorrelation and informative selection
produce failed requirements even where apparent power is high. Some otherwise
reasonable designs are inconclusive at this replication count. This is an
intentional demonstration of failure detection, not a failed software run.
It covers continuous-outcome segmented ITS, not a clinical study, DiD,
clustered deployment or staggered rollout.

## Verification record

All **782 tests in the final collected inventory have passing recorded results**
across the full regression run and final targeted runs. No current test is
missing, failed or skipped in that reconciliation. The
[per-test evidence ledger](../reports/implementation_validation_summary.json)
maps every current test to its JUnit result.

| Check | Result and evidence |
|---|---|
| Full regression run | [730 passed; one obsolete equity assertion failed](../reports/implementation_test_results.txt), 17m59s. This original log is retained; it is not rewritten as an all-green run. |
| Final changed-feature/fault-injection run | [232 passed](../reports/implementation_final_focused.txt), including both equity replacements, missing-evidence cases and integrity/provenance faults. |
| Final demo integration regression | [19 passed](../reports/implementation_final_demo_regression.txt), including a full nine-run public demo with a consistent registry identity. |
| Final inventory reconciliation | [782 current tests collected](../reports/implementation_final_inventory.txt); all map to a passing result from the three retained JUnit records. This is selective rerun evidence, not a second full-suite execution. |
| Architectural guards and diff whitespace | Passed. |
| Main runner and queue/overlay smoke | Two seeds per runner; the 180-day queue output passed overlay exactness and expected-seed checks. |
| Packet consistency and applicability | [Both example packets pass artifact, saved-identity, accounting, referenced-artifact, evidence-consistency and current-identity checks](../reports/implementation_packet_integrity.json). Recorded model bytes match. Required clinical/design evidence correctly remains NOT_RUN or INCONCLUSIVE. |
| Notebook | Seven code cells executed successfully, including traces and reconciliation. |
| Synthetic ITS design evidence | All 1,200 fits completed; adverse designs deliberately FAIL declared requirements. |

Whole-repository static checks are **not clean**. At the project's declared
88-column style, lint decreased from 204 baseline diagnostics to 199, with no
new diagnostics. Mypy decreased from 77 to 69: 64 pre-existing diagnostics
remain, 13 were removed, and five new diagnostics concern unavailable SciPy
type stubs at new imports. No new implementation typing errors were found.
The [baseline comparison](../reports/implementation_quality_comparison.json)
and [independent review](../reports/implementation_integration_review.md)
preserve these limits. `.flake8` now makes the already-declared 88-column
setting effective for flake8 itself.

The rebuilt notebook executed all seven code cells with the checked-in
headless runner. Standard Jupyter kernel execution was unavailable because the
sandbox disallowed socket binding; the successful execution method is recorded
in the notebook. CI configuration is implemented, but this local task does not
prove a remote Actions run or change repository branch-protection settings.

An older reminder-equity test asserted that a higher-AUC model must improve a
racial disparity more in a single simulated run. The corrected day-zero
schedule exposed that unsupported assumption. Its replacement derives expected
targeting effects by hand and checks actual engine behavior over prespecified
paired repetitions. No favorable seed was selected and the model was not tuned
to make the old claim pass. See [the migration notes](analysis_exports_and_traces.md).

## Reproduce the evidence

```bash
.venv/bin/python -m healthcare_sim_sdk.scenarios.malnutrition_prioritization.run_review_demo
.venv/bin/python -m healthcare_sim_sdk.scenarios.nurse_retention.monitoring_validation regime=capacity_collapse population.n_nurses=500 n_weeks=52 model_validation_n_nurses=5000 tier3_mode=its_only hydra.run.dir=outputs/retention_assurance_capacity_review
.venv/bin/python scripts/certify_analysis_design.py --repetitions 100
.venv/bin/python scripts/check_invariants.py
.venv/bin/python -m pytest tests -ra
```

The statistical demonstration intentionally exits 1 when adverse designs fail
its stated acceptance requirements; inspect its evidence instead of treating
that exit as a simulator exception.

## Decisions reserved for review

- Confirm clinical mechanisms, effect assumptions and local usual-care workflows.
- Confirm whether retention's separate SOC and AI budgets represent the intended comparison; they add resources rather than hold total manager time fixed.
- Specify the intended real-world estimand, design, rollout, outcome availability and prespecified acceptance requirements, then validate that design with held-out repetitions.
- Produce institution-specific model/scenario certificates for the actual supported range; illustrative demo parameters are not those certificates.
- Assess subgroup evidence and actual burdens; representative cases cannot certify equity.
- Enter a real reviewer decision against the packet identity. Machine results leave human review pending.

Additional scenarios can adopt the common API, but they do not receive a
clinical claim registry or certification automatically. Existing seeded outputs
affected by appointment timing, retention random coupling or sepsis score
conditioning should be regenerated before comparison with these packets.
