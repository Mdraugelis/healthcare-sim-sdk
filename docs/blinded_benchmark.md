# Frozen-input evaluation: SHIELD-RT pilot

The benchmark separates permitted simulation inputs from observed study
results. A trusted coordinator freezes the inputs and execution environment,
runs the SDK in a restricted child process, records its output, and only then
permits a separate evaluator to reveal and compare the study results.

**The supplied SHIELD-RT case is outcome-exposed development work.** It tests
the evaluation machinery and a simplified trial adapter. It is not an
independently blinded clinical validation. Published outcomes already influenced
earlier project work; neither a new vault nor new seeds can erase that history.

## Implementation plan and delivered components

| Step | Delivered behavior |
|---|---|
| Specify the case | Strict JSON contract, explicit inputs and source dependencies, endpoint/support checks, preregistered seeds and scoring criteria |
| Separate the reference | Custodian-only vault outside the repository and worker-readable runtime roots; random nonce prevents guessing a small reference space from its public commitment |
| Freeze execution | Source, input, evaluator, interpreter and NumPy identities recorded before execution |
| Run the SDK | Minimal copied code bundle, static adapter, no evaluator imports; macOS read/write/network restrictions verified by actual probes |
| Commit evidence | Trusted supervisor pins each batch's output immediately; HMAC-protected custody history blocks fabricated, replaced or incomplete submissions |
| Reveal and score | Explicit transitions; empirical CRPS, interval score, absolute error and null-reference skill; failed evidence cannot silently become a zero |
| Support human review | Five role views, exact identity, full run inventory and pending human notes, with numerical fit separate from evidence adequacy |
| Challenge the machinery | Analytic scenario checks, numerical scoring oracles, full lifecycle tests, adversarial tampering/race tests and a real OS isolation test |

The pipeline currently supports one case per vault and the `shield_rt_v1`
adapter. General suite orchestration, additional study adapters and Linux
isolation are future work. Unsupported platforms fail closed when isolation
is required; there is no silent fallback.

## What the pilot actually simulates

The new adapter lives in
`healthcare_sim_sdk/experiments/benchmark/shield_rt.py`. It uses the existing
`BaseScenario` interface and `BranchedSimulationEngine` with
`CounterfactualMode.NONE`. The study's two randomized arms exist inside one
trial population; an additional paired counterfactual is not substituted for
the trial-arm contrast.

- One timestep is one complete treatment course, not a fixed number of weeks.
- Courses are already in the high-risk eligible cohort. The adapter does not
  recreate the screening classifier or repeat a threshold decision.
- It assigns exactly the declared number of courses to each arm.
- Baseline risk, treatment relative effect and uptake are independent,
  explicitly declared fixed or uniform parameter distributions. They are drawn
  once per simulated trial, not separately for each patient's latent risk.
- Control outcomes reflect standard care. In the intervention arm, uptake
  determines which courses receive the additional evaluation's effect.
- Each course has one binary acute-care outcome. Repeated measurement does
  not redraw that outcome or compound treatment.
- Negative effectiveness permits harm, provided the entire joint support
  remains a valid probability distribution. Full prevention and zero uptake
  are also supported.
- The primary estimator is intervention risk minus control risk. A separately
  sampled independent-arm null-effect reference uses the same risk context
  and arm sizes. It is a predictive distribution, not a zero-width point at
  zero.

With fixed baseline probability `p`, relative effect `e`, and uptake `u`, the
analytic expected risk difference is `-p * e * u`. Tests check the simulator
against that independent calculation and exact boundary cases. The engine is
useful here for trial-level variation and uncertain-input propagation; the
expected value itself does not require a simulator.

This adapter does not model visit schedules, clinician capacity, radiation
physiology, individual case mix, repeated courses in the same person,
correlated parameter priors, or treatment-dependent follow-up. It therefore
does not certify all details of the original trial. Those limits are preserved
in the case specification and review packet.

## Supplied inputs and evaluator reference

`examples/benchmarks/shield_rt_pilot.json` supplies 256 fixed seeds and explicit
input provenance. It conditions on reported allocation sizes and uses broad
**illustrative**, independent distributions for baseline risk, efficacy and
uptake. They are software demonstration assumptions, not estimates from
independent clinical studies. No parameter is searched or updated to improve
agreement with the target.

`examples/benchmarks/shield_rt_published_reference.json` is evaluator-side
material. It records the rounded during-treatment event proportions from the
[original SHIELD-RT report](https://pubmed.ncbi.nlm.nih.gov/32886536/), with the
endpoint and rounding stated explicitly. It is copied to the separate vault
before running and is never included in the worker bundle. The existing
repository already contains the published answers, so source isolation and
the exposure declaration are both necessary.

The error tolerance and minimum skill are declared demo criteria, not clinical
acceptance standards. An appropriate reviewer must specify meaningful
criteria and independent input evidence before a confirmatory evaluation.

The original `scenarios/paper06_shield_rt/scenario.py` is preserved as historical
calibration code. Its outcome-derived effectiveness is not imported by this
adapter or included in the worker's allowed source files.

## Run the complete evaluation

Run commands from the repository root with the development environment
installed. Choose **new** session and vault directories. A vault accepts only
one registered evaluation and cannot be reused after inspecting results.

The coordinator commands can read the vault. The separate SDK child process
cannot: it receives no vault path, reference object, nonce or custody key.
For an independent evaluation, the trusted coordinator and its vault must be
controlled by the designated custodian, not by the modeler.

```sh
.venv/bin/python -m healthcare_sim_sdk.experiments.benchmark seal \
  --reference examples/benchmarks/shield_rt_published_reference.json \
  --vault /private/tmp/shield-rt-custodian

.venv/bin/python -m healthcare_sim_sdk.experiments.benchmark freeze \
  --case examples/benchmarks/shield_rt_pilot.json \
  --session outputs/benchmarks/shield_rt \
  --vault /private/tmp/shield-rt-custodian

.venv/bin/python -m healthcare_sim_sdk.experiments.benchmark run \
  --session outputs/benchmarks/shield_rt \
  --vault /private/tmp/shield-rt-custodian

.venv/bin/python -m healthcare_sim_sdk.experiments.benchmark commit \
  --session outputs/benchmarks/shield_rt \
  --vault /private/tmp/shield-rt-custodian

.venv/bin/python -m healthcare_sim_sdk.experiments.benchmark reveal \
  --session outputs/benchmarks/shield_rt \
  --vault /private/tmp/shield-rt-custodian

.venv/bin/python -m healthcare_sim_sdk.experiments.benchmark score \
  --session outputs/benchmarks/shield_rt \
  --vault /private/tmp/shield-rt-custodian

.venv/bin/python -m healthcare_sim_sdk.experiments.benchmark verify \
  --session outputs/benchmarks/shield_rt \
  --vault /private/tmp/shield-rt-custodian
```

Open `outputs/benchmarks/shield_rt/review/review_packet.md` afterward. Scoring
returns exit zero when the requested operation completed, even when numerical
agreement fails. Read the saved statuses. Execution failures return exit one;
refused/invalid protocol operations return exit two.

The state sequence is:

```text
INPUTS_FROZEN → EXECUTED → PREDICTIONS_FROZEN → REVEALED → SCORED
```

`EXECUTED` means the trusted supervisor has already recorded the result hash;
the following commit checks its complete roster and registered endpoints.
All declared seed failures can be committed and reviewed as failures. Missing,
duplicate, reordered or invented seeds are refused. The first completed
attempt is preserved. An interrupted pre-receipt operation may require
custodian inspection; the software does not silently delete locks or rewrite
historical evidence to recover.

For developer-only demonstrations on platforms without isolation, `run` accepts
`--allow-unisolated-development`. That explicitly reports isolation `NOT_RUN`
and never establishes blinding. Do not use that mode as evidence of outcome
secrecy. A restrictive parent environment may also prevent starting the macOS
sandbox; the command fails visibly rather than silently weakening it.

## Input and custody rules

Every input points to a source with a kind, citation, availability date and
source dependencies. The contract rejects missing sources, cycles, post-cutoff
sources, unknown fields and direct or transitive use of a withheld endpoint.
An unused target-bearing source cannot be hidden in the visible ledger.
It also rejects scoring an endpoint explicitly released as an input.

The framework defines before-deployment and conditional information tracks.
The present SHIELD-RT adapter scores both arms, so it cannot condition on a
released control event rate while also scoring that same rate. Its supplied
pilot instead conditions on allocation sizes. Additional adapters/contracts
are needed for other conditional targets and their shared-control covariance.

Declared provenance is not proof that a citation is correct or an assumption
is justified. It requires curator review. The checks cannot erase knowledge
from a person or a trained model, detect every target-derived number falsely
labeled as independent, or prevent someone making a different vault outside
the protocol and concealing the earlier result. Independent custody and an
evaluation registry remain necessary for stronger claims.

The vault's HMAC receipts protect against changes by a submitter without the
custody key. They are not publicly verifiable digital signatures or protection
from a malicious custodian or operating-system administrator. Local file
permissions and read-only artifacts are supporting controls; the worker's OS
sandbox and the separately held receipt history are the relevant boundaries.

On macOS the worker allows only its copied source, the required Python/NumPy
runtime, specific system facilities, and a temporary output directory. Root
and dependency-parent directory handles are permitted where the loader needs
them, without recursive access to their children. It probes an outside file
read, outside write and network attempt, requiring actual permission denials.
Environment fingerprints include runtime binary and NumPy bytes. This proves
the tested process boundary, not the analyst's epistemic blindness.

## Reading the results

The packet separates completeness, numerical fit, declared exposure, input
adequacy, independent validation and human review. A numerical PASS requires
the preregistered primary absolute-error and skill criteria; it does not
approve a clinical model. Zero baseline CRPS makes relative skill undefined
and inconclusive, rather than an invented value.

Empirical CRPS and interval score assess the submitted distribution against
the observed study statistic. Predictive intervals reflect the modeled trial
and input variation. MCSE reflects only the precision of the simulated mean.
The evaluator does not add publication sampling variance a second time to
matched-size synthetic trials. One observed study cannot establish interval
coverage. The current inputs' independent uniform distributions do not claim
to quantify all clinical or structural uncertainty.

The five views address clinical mechanisms, operations, statistics, governance
and engineering. `human_review.json` begins pending and is never auto-signed.
Existing human notes are preserved; machine evidence is verified separately.

## Recorded development pilot: 2026-09-07

The supplied case completed all **256 registered runs**, with no seed failures
and a passing macOS isolation probe. The final local session is
`outputs/benchmarks/shield_rt_2026-09-07_v3/`. Generated sessions and custody
material are not tracked in Git; the commands above reproduce the workflow.

| Quantity | Recorded result |
|---|---|
| Mean predicted control / intervention risks | 24.72% / 22.03% |
| Mean predicted intervention-minus-control risk difference | −2.69 percentage points |
| Published rounded risk difference | −10.00 percentage points |
| Absolute error / preregistered tolerance | 7.31 / 5.00 percentage points |
| 90% predictive interval for the risk difference | −12.70 to +7.66 percentage points |
| Monte Carlo standard error of the predicted mean | 0.39 percentage points |
| CRPS / null-reference CRPS | 0.04626 / 0.07669 |
| Relative CRPS skill | 0.39677 |
| Numerical fit | **FAIL**: absolute error exceeds the declared tolerance |
| Input adequacy | **INCONCLUSIVE**: illustrative inputs lack independent clinical support |
| Independent validation / blinding | **NOT_APPLICABLE**: outcome-exposed development |
| Human review | **NOT_RUN**: pending reviewer assessment |

All 3,338 machine consistency checks passed. Those checks concern the evidence
and execution record; they do not overturn the numerical-fit failure. The
predictive interval includes the published difference, but interval inclusion
alone is not the acceptance rule. No inputs or seeds were tuned after reveal.
Earlier local output versions are retained; subsequent report compaction and
a type annotation preserved every seed's numerical result. They are not
independent replications.

The focused benchmark suite recorded **98 passing tests**, including the
opt-in OS isolation test. A broader unit/integration run recorded **544 passed
and one skipped** (the opt-in OS test in that run). This is bounded test
evidence, not a fresh execution of the entire repository suite. The final
type annotation was followed by 32 passing adapter tests and the full
256-run isolated pilot.

## Verification commands

```sh
.venv/bin/python -m pytest tests/unit/test_benchmark* \
  tests/integration/test_blinded_benchmark.py -q
.venv/bin/python scripts/check_invariants.py
SDK_BENCHMARK_OS_SMOKE=1 .venv/bin/python -m pytest \
  tests/unit/test_benchmark_execution.py -q
```

The OS smoke is opt-in and requires a host that permits applying the restrictive
macOS sandbox. Ordinary CI records that test as skipped when it cannot run;
the skip is not an isolation pass. Synthetic lifecycle tests separately use an
explicit development-mode worker and do not claim OS enforcement.

The next scientific step is to replace demonstration assumptions with
independently reviewed evidence, validate the missing study mechanisms and
use genuinely fresh observations. No amount of successful software testing
converts this familiar published study into an unseen holdout.
