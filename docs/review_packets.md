# Review packets for simulation and analysis

The SDK now produces evidence for a specific run and decision question, with
five review views. A packet is a structured case for human assessment; passing
code checks does not establish clinical validity or approve a deployment.

## Start with the public demonstration

From the repository root, after installing the development dependencies:

```bash
.venv/bin/python -m healthcare_sim_sdk.scenarios.malnutrition_prioritization.run_review_demo
```

Open `outputs/review_demo/malnutrition/review_packet.md`. This is a public
synthetic example with three workflow/capacity cells, three seeds per cell and
35 days. It writes its illustrative model parameters alongside the evidence;
it does not load institutional model targets or patient data. These nine runs
demonstrate the review workflow, not sufficient Monte Carlo precision or
external model validation.

The packet links to these views:

| Reviewer | Main questions | Evidence to inspect |
|---|---|---|
| Clinical | Does the modeled population, usual care and treatment mechanism represent the proposed use? | Assumptions, helped and harmed cases, clinical outcomes and unresolved clinical certification |
| Operations | Are eligibility, action timing and resource limits realistic? Who loses access? | Actual assessments, capacity, missed cases, queue measures and displacement examples |
| Statistics | What is the estimand, denominator and comparison? Is the analysis adequately tested? | Complete cell/seed accounting, outcome summaries, uncertainty and independent design evidence |
| Governance and equity | Who benefits and who bears burdens? Which groups are too sparse to assess? | Subgroup availability, false-positive workups, selected case limitations and intended-use assumptions |
| Engineering | Can these exact results be reproduced and trusted as artifacts? | Code/config/model fingerprints, checks, trace IDs, expected runs and artifact hashes |

The roles share one evidence record. They do not create five independent
validations. Case stories are explicitly selected from actual engine histories:
seeded random examples plus available mechanism examples. Absence of a case
category is disclosed. Examples illustrate behavior and do not estimate its
frequency. Where historical outputs lack a decision detail, the trace says so
instead of reconstructing an unrecorded explanation.

## Read statuses literally

| Status | Meaning |
|---|---|
| PASS | The declared, applicable check ran and met its specified criterion. |
| FAIL | Executed evidence contradicted a requirement or an execution/integrity check failed. |
| NOT_RUN | Required evidence is absent or the necessary observation window has not completed. |
| INCONCLUSIVE | Evidence exists but is too weak, sparse, imprecise or incompatible to settle the claim. |
| NOT_APPLICABLE | The claim does not apply in this scope; a rationale is mandatory. |

Required claims cannot be satisfied by an empty check list, optional checks,
missing metrics, missing seeds or a stale certificate. Diagnostic ranges are
separate from structural gates: a deliberately stressed workflow can miss a
calibration target without implying a broken simulator. The registry declares
the evidence and acceptance criteria for each decision claim.

`human_review.json` starts with pending reviewers and is never automatically
signed. Existing notes are preserved on regeneration. Notes are editable and
excluded from machine artifact hashes; human approval requires a real reviewer
and should reference the packet identity. A generated PASS is not human review.

## What is saved

- `config.json`: resolved inputs, with per-cell scenario settings and model hashes for the malnutrition runner.
- `registry.json`: decision question, estimands, populations, denominators, comparisons, horizons, required evidence and reviewers.
- `evidence.json`: each actual run, metrics, checks, selected cases, failures and claim statuses.
- `manifest.json`: identity, expected/completed/missing/failed runs, machine status and artifact hashes.
- `review_clinical.md`, `review_operations.md`, `review_statistics.md`, `review_governance.md`, `review_engineering.md`: role views of the evidence.
- Referenced trace and analysis files, with hashes; `human_review.json` remains separate.

Fingerprints cover repository source contents (including uncommitted source),
resolved configuration, model bytes where supplied, and recorded environment
versions. Ignored institutional data is not discovered or copied by source
hashing. Certificates have exact applicability; changing a model or a relevant
configuration requires new evidence. Integrity hashes detect inconsistency;
they are not digital signatures against a malicious editor who rewrites the
whole evidence package.

Verify a saved packet with the Python API:

```python
from healthcare_sim_sdk.experiments.review import verify_packet
checks = verify_packet("outputs/review_demo/malnutrition")
for check in checks:
    print(check["name"], check["status"])
```

Artifact/identity/accounting consistency is distinct from
`declared_machine_evidence`: a complete, intact packet can correctly say that
clinical or statistical evidence is NOT_RUN. Supply `current_identity` when
checking whether an older packet applies to a particular current run.

## Existing runners

The malnutrition `run_experiment` and `run_los_readmit_sweep` runners now check
every actual cell/seed on both branches and write a `review/` directory. The
queue runner also hashes saved per-patient arrays and records exact expected
seed IDs. Overlay/estimand summaries reject missing files or undefined
repetitions rather than silently selecting the available data. Old queue
manifests without exact seed lists must be regenerated.

The malnutrition Stage A and Stage B tools write certificates alongside their
existing reports. Stage A applies to the stated score-generation cohort and
targets; Stage B applies to its declared scenario configuration and seed suite.
Neither certifies all settings of a new sweep. Private target-dependent work
must be performed in the configured institutional environment.

The retention monitoring runner also creates role packets. See
[monitoring evidence](monitoring_assurance.md) for label maturity, observable
versus oracle comparisons, event-specific detection, independent model checks
and the separate SOC/AI resource budgets that reviewers must assess.

The generic `ReviewPacket` API is available for other scenarios. Those scenarios
do not automatically acquire claim registries or clinical certification by
importing the SDK. Generic lifecycle outputs now explicitly distinguish
structural artifact checks from missing simulation/analysis validation.

## Statistical evidence and regression protection

Seed summaries separately report empirical 2.5th/97.5th percentiles, standard
deviation, Monte Carlo standard error and a t interval for the Monte Carlo mean.
These are conditional on the supplied assumptions. A single seed has no
estimated Monte Carlo interval. Binomial performance uses exact bounds, so zero
false alarms among a finite number of runs never means a certified zero risk.

The [analysis-design harness](analysis_design_certification.md) exercises the
existing segmented ITS estimator against independent planted truth. It includes
null effects, signal, autocorrelation, short observation windows, changing
denominators and informative selection. The adverse demonstration is expected
to fail some design requirements; a harness that reports this failure is
working. Its results do not certify a clinical study, DiD, clustered sites or
staggered deployment.

[Export and trace migration notes](analysis_exports_and_traces.md) describe
corrected outcome selection, staggered treatment timing, missing observations,
stable appointment IDs and actual-engine tracing. The notebook can be executed
with `scripts/execute_notebook_headless.py` when Jupyter socket creation is
unavailable.

The GitHub Actions workflow runs architectural guards and the public test suite
and preserves JUnit failures/skips. Installing this workflow does not configure
remote branch protection or prove that a remote run passed. Configuration or
private-data-dependent skips remain visible rather than being counted as passes.

## Recommended human review

1. Read the overview's scope and unresolved claims before any effect estimate.
2. Inspect your role page, then compare random and mechanism-selected case traces.
3. Review the shared analysis, especially missing groups, failed seeds and uncertainty.
4. Record questions, assumption changes and a decision against the exact packet identity.
5. Regenerate relevant evidence after model, code, workflow or design changes.

The implementation makes uncertainty and missing evidence reviewable. Local
clinical plausibility, operational feasibility and deployment decisions still
require the designated people and evidence appropriate to their intended use.
