# Healthcare Intervention Simulation SDK: Design Principles

> A design principles document for the `healthcare-sim-sdk`. Explains why the SDK is built the way it is, what it enables, what it deliberately excludes, and the lessons we learned building and validating it against 30 published healthcare AI deployment papers.

---

## Preface: The Unobservable Counterfactual Problem

Healthcare AI systems promise to improve patient care by identifying high-risk individuals for early intervention. Evaluating these systems presents a fundamental challenge: in real-world deployments, we never observe the counterfactual — what would have happened without the AI intervention. When the AI predicts an unwanted outcome and triggers an action intended to prevent it, the very success of the action obscures our ability to measure its impact.

**Example 1: Stroke Prevention.** An AI system predicts which patients are at high risk of stroke within the next 12 months. When the model flags a high-risk patient, clinicians intervene — anticoagulants, blood pressure management, specialized clinic enrollment. If the patient doesn't have a stroke, we face an attribution problem: did the intervention prevent it, or was the patient never going to have one? The intervention's success masks the AI's accuracy and impact.

**Example 2: Breast Cancer Screening.** An AI system identifies patients at elevated risk for breast cancer, prompting earlier mammograms. When those patients receive timely screening, we can observe whether cancer is detected. But attribution is hard: did the AI improve early detection, or were improvements driven by concurrent initiatives — a general awareness campaign, additional scheduling staff, reminder systems that improved screening rates across all patients? Confounding factors make it difficult to isolate the AI's specific contribution.

**Example 3: Sepsis Early Warning.** A health system deploys an AI model to detect sepsis onset before clinical deterioration. But clinicians catch sepsis through routine care whether or not the AI is deployed — vitals monitoring, lactate orders, clinical suspicion. The relevant causal question is not "AI vs. no treatment" but "AI *plus* standard care vs. standard care alone." Without modeling the baseline, the AI's measured benefit is inflated.

Traditional RCTs can address these problems but are expensive, time-consuming, and ethically complex. You cannot run an RCT before deciding whether to deploy. This creates a specific gap that simulation fills: a controlled environment where the counterfactual is known by construction, and where policies, methods, and vendor claims can be stress-tested before real patients are affected.

This document explains the design principles of the Healthcare Intervention Simulation SDK — a toolkit for pre-deployment evaluation of clinical ML systems.

---

## Project Purpose

### What This SDK Is

- **A pre-deployment stress test with known ground truth.** When you run a simulation, you know the true causal effect, which enables method validation, power analysis, and assumption stress-testing.
- **A toolkit for building custom scenarios via a 5-method contract.** Each scenario encodes the domain physics, ML model behavior, intervention mechanics, and outcome measurement for a specific clinical use case.
- **A framework for validated evaluation design.** It lets you test whether a proposed causal inference method (ITS, DiD, RDD) can actually recover a known effect in your specific setting before you commit to that method in a real evaluation.
- **An honest counterfactual engine.** The SDK's branched simulation generates parallel factual and counterfactual trajectories synchronized at the stochastic level, not compared post-hoc.

### What This SDK Is Not

- **Not evidence of clinical effectiveness.** Every result is conditional on the assumptions baked into the scenario. The simulation suggests, under these assumptions.
- **Not a substitute for prospective study design.** It informs the design; it does not replace the study.
- **Not a continuous-time physiologic simulator.** Intraoperative hemodynamics, sub-minute pharmacokinetics, and vasopressor titration are outside the SDK's discrete-time architecture (see the HYPE study in the audit).
- **Not a qualitative implementation modeler.** Connectivity outages, trust erosion, workflow disruption, and organizational dysfunction are not expressible in a `step()` function (see the Beede study in the audit).
- **Not a model training or evaluation tool.** The SDK simulates model *performance characteristics*, not model internals.

### Primary and Secondary Goals

**Primary goal:** Evaluate healthcare AI deployment questions before real commitments are made.

**Secondary goals:**

1. Train operations and research teams on causal inference methods and AI deployment design
2. Compare the performance of different causal methods (ITS, DiD, RDD, equity analysis) against known ground truth
3. Validate evaluation designs — if a proposed analytic plan can't recover a known effect in simulation, it won't find it in production
4. Support equity audits by surfacing demographic differentials in intervention benefit
5. Stress-test vendor claims against the specific patient mix and operational constraints of a target health system

---

## Core Design Principles

The SDK is organized around ten principles. Each is a design commitment, not a feature. Every line of code in the SDK serves one or more of these.

### 1. Targeted Fidelity

High fidelity where it matters for causal inference; simplified abstractions elsewhere.

The SDK maintains mathematical rigor for:

- Population risk distributions (heterogeneity drives intervention effects)
- Temporal risk dynamics (AR(1) drift creates the time series that ITS and DiD operate on)
- Intervention mechanics (multiplicative effects are analytically tractable)
- Known ground truth at every level (enables method validation)

It deliberately simplifies:

- Individual clinical workflows
- Physiological detail below the discrete timestep granularity
- Patient characteristics that don't affect causal estimates

This targeted approach allows population-scale simulation (5,000 to 100,000+ entities over multi-year horizons) in seconds to minutes, which enables the multi-seed variance analysis and parameter sweeps that make the SDK useful in practice.

### 2. Known Ground Truth

Unlike real studies, the simulation knows the true causal effect by construction. This enables:

- **Method validation:** Can your preferred causal inference method recover the true effect?
- **Power analysis:** What sample size do you need for a given effect size?
- **Robustness testing:** How sensitive are your estimates to assumption violations?
- **Perfect counterfactuals:** What would have happened without the intervention?

When a method fails to recover the true effect in simulation, that is a *finding about the method*, not a failure of the simulation. This is the SDK's most important epistemological contribution: it separates method failures from effect failures.

### 3. Branched Counterfactual Validity

The counterfactual is not a post-hoc comparison of two runs with different seeds. It is a parallel trajectory generated in the same stochastic stream as the factual, mathematically synchronized via RNG stream partitioning. Both branches experience identical background events — the same patients get sick at the same times, the same temporal drift occurs — and the branches diverge only where the intervention actually changes state.

This matters because a naive "run it twice, once with and once without treatment" approach introduces statistical noise from independent stochastic evolution. Differences between runs can be driven by the intervention *or* by the accident of different random draws. The branched engine eliminates that ambiguity: the only source of difference is the intervention.

Implementation: `BranchedSimulationEngine` maintains two state trajectories, and `RNGPartitioner.fork()` produces independent but identically-seeded stream sets for the counterfactual branch. See Section "Core Mechanisms" for details.

### 4. Step Purity

The `step()` method — the function that evolves state from one timestep to the next — must be a pure function of `(state, t, rng.temporal)`. No side effects. No external mutable state. No reading from RNG streams other than `temporal`. No attribute access on `self` beyond reading configuration.

This is what makes branch synchronization mathematically valid. If `step()` were allowed to read or write shared state, the factual and counterfactual branches could desynchronize silently. A test called `test_branched_vs_none_identity` verifies this property directly: running the factual branch of a branched simulation must produce identical outcomes to running the same scenario with no counterfactual branch, on every timestep. If the test fails, `step()` is impure.

The purity contract is machine-enforced. It is what distinguishes the SDK from an ad hoc simulation script.

### 5. Unit-of-Analysis Agnosticism

The SDK's core engine never inspects state internals. It never calls `getattr`, `isinstance`, or subscripts state variables. State is declared as generic type `S` and can be anything a scenario needs:

- **Stroke Prevention:** scalar patient risks + boolean treatment flags
- **No-Show Overbooking:** nested dataclass with patients dict, provider schedules, waitlists
- **Sepsis Early Alert:** 16-row NumPy array per patient-admission
- **SHIELD-RT:** `n_patients × 3` matrix per RT course
- **No-Show Targeted Reminders:** per-patient reminder histories with demographic stratification

This polymorphism is not accidental. It is the result of refusing to define a "common state schema" that would have forced domain-specific concepts into a one-size-fits-all structure. The 5-method contract is polymorphic by design: scenario authors define the state representation that fits their domain, and the engine carries it opaquely.

If the SDK had enforced a common state type, the no-show and sepsis scenarios would not share the same engine. They would have been three different simulators.

### 6. Counterfactual Honesty

This is the hardest-won principle. The counterfactual should reflect realistic standard-of-care, not "no treatment."

When we first built the sepsis scenario, the counterfactual branch assumed nobody detected sepsis without the AI. That produced a 0.16pp whole-population mortality reduction — wildly inflating the AI's benefit, because in reality clinicians catch sepsis through routine care whether or not the AI is deployed. The AI's measured benefit was really "AI + standard care" vs. "no care at all," which is not the deployment decision anyone is making.

The fix: model standard-of-care sepsis detection as a per-patient delay drawn from `Beta(2, 5) × 24h` (right-skewed, mean ~6.9h, matching CMS SEP-1 compliance). Baseline detection runs on *both* branches. The AI only gets credit for improvement *over* baseline care.

With this correction, the TREWS replication produces 4.27pp mortality reduction among septic patients (95% CI: 3.30-5.56 across 30 seeds), with the published 3.3pp falling within the CI. The TREWS paper moved from PARTIALLY_REPRODUCED to REPRODUCED in the 30-paper audit.

This is more than a parameter adjustment. It is a principle: **the counterfactual in a well-designed healthcare simulation must reflect what would actually happen without the AI, including competent standard care.** Any scenario where "no AI" means "no treatment" is asking the wrong question.

A related principle: **time matters.** We also added Kumar time-dependent treatment effectiveness (halves every 6 hours after sepsis onset, per Kumar et al. 2006) so that earlier detection is genuinely more valuable than late detection. The ML system's timing advantage over baseline care, amplified by Kumar decay, is the actual mechanism for clinical benefit.

### 7. Machine-Checked Invariants

Architectural guarantees are encoded in `.claude/invariants.yaml` and enforced by pre-commit hooks. No reliance on human discipline:

- **Protected files:** `core/scenario.py`, `core/engine.py`, `core/rng.py`, `core/results.py` require explicit review. Scenario developers should never need to touch these.
- **Banned patterns:** bare `np.random` and `import random` are blocked everywhere except `core/rng.py` and tests. All randomness must flow through RNGPartitioner streams.
- **Structural checks:** the 5-method contract is verified — required abstract methods (`create_population`, `step`, `predict`, `intervene`, `measure`) must be implemented.
- **Counterfactual isolation:** the engine must not call `predict()` on the counterfactual branch.
- **State opacity:** the engine must not inspect state internals.

If a principle is worth stating, it is worth enforcing mechanically.

### 8. Evidence Generator, Not Decision Maker

The SDK produces analysis-ready data and honest assessment. It does not make go/no-go recommendations. The human, who knows the clinical context, institutional constraints, and patient population, decides.

This is encoded in the agent protocol at `.claude/agents/sim-guide.md`: the agent helps frame questions, configures scenarios, runs verification, and translates findings — but it never says "you should deploy this" or "you should not deploy this."

When stakeholders ask "should we deploy?", the answer is "here is what the simulation suggests under these assumptions; here is what would need to be true for the decision to go one way or the other; the decision is yours."

### 9. Verification Before Interpretation

Every simulation run, no exceptions, goes through a 4-level verification protocol before results are interpreted. A misconfigured simulation that looks plausible is worse than no simulation at all, because it produces false confidence.

The protocol (detailed in the "Verification Protocol" section):

1. **Structural integrity** — no NaN/Inf, population size constant, prediction scores in [0,1]
2. **Statistical sanity** — outcome rates within 4-sigma CLT bounds of targets, model AUC/PPV within tolerance
3. **Conservation laws** — null treatment identity, monotonicity, Bayes bounds
4. **Case-level walkthroughs** — trace 3-5 individual entities through the full simulation in narrative form

The `tests/bulletproof/` directory contains the conservation-law and purity tests that validate the SDK's core claims. Every scenario adds its own conservation tests.

### 10. Explicit Assumptions, Surfaced Not Hidden

The scenario code *is* the assumption list. Every parameter that affects the result is in the Python scenario file or its YAML config, both of which are committed to the repository and reviewed like code. Nothing material is buried in opaque data files or hidden defaults.

When a result looks surprising, the scenario file is where you go to find out why. When a result looks clean, the scenario file is where you explain what produced that cleanliness. Configuration is not a substitute for explanation, and "the simulation said so" is never an adequate justification on its own.

This principle is why the SDK uses Hydra YAML configs for parameter sweeps but keeps the domain logic in Python. Configuration controls variants; Python encodes invariants.

---

## The 5-Method Scenario Contract

Every scenario in the SDK inherits from `BaseScenario[S]` (generic over state type) and implements exactly five methods:

| Method | Signature | RNG Stream | Runs On |
|--------|-----------|------------|---------|
| `create_population` | `(n_entities: int) -> S` | `rng.population` | Once, at start |
| `step` | `(state: S, t: int) -> S` | `rng.temporal` | Every timestep, **both branches** |
| `predict` | `(state: S, t: int) -> Predictions` | `rng.prediction` | Prediction schedule, **factual only** |
| `intervene` | `(state: S, predictions: Predictions, t: int) -> (S, Interventions)` | `rng.intervention` | After predict, **factual only** |
| `measure` | `(state: S, t: int) -> Outcomes` | `rng.outcomes` | Every timestep, **both branches** |

Each method may read only from its assigned RNG stream. This is what enables branch synchronization — the factual branch consuming the `prediction` and `intervention` streams does not advance the `temporal` stream, so the counterfactual branch's `step()` calls draw from identical positions in the `temporal` stream and the two branches evolve in lockstep (modulo intervention-induced state changes).

### Why Exactly Five Methods

Not six, not three. This is a deliberate choice with specific rationale:

- **`create_population` must be separate from `step`** because initial population generation has different stochasticity (beta-distributed risks, demographic assignment) that only happens once. Folding it into `step(state=None, t=0)` would be an awkward special case.
- **`step` must be separate from `predict`** because `step` runs on both branches and `predict` runs only on the factual branch. If they were combined, the counterfactual branch would either skip prediction (losing symmetry) or run a prediction it never uses (wasting RNG draws).
- **`intervene` must be separate from `predict`** because it takes predictions as input and returns a modified state. A prediction without an intervention is a monitoring system, not a clinical AI.
- **`measure` must be separate from `intervene`** because it runs on both branches. Combining them would force the counterfactual branch to run a null intervention, which is both wasteful and error-prone.

Three methods (init, step, observe) would force scenarios to do too much inside `step`, violating purity. Seven methods (separating prediction-score generation from thresholding, or separating intervention selection from application) would be over-engineered for the population-level question the SDK answers.

### The Fitness Check

Before implementing any scenario, walk through these six questions. If three or more fail, the SDK is probably not the right tool.

1. **Is this a population-level intervention?** The SDK simulates cohorts over discrete time. It is not for individual clinical decision support or real-time alerting logic.
2. **Is there a predictive model involved?** The SDK's strength is simulating ML model behavior at controlled performance levels. Without a predictive model, the SDK still works but you're not using its strongest capability.
3. **Can you define the intervention as a state change?** `intervene()` modifies state based on predictions. Purely informational interventions need a behavioral-response model, which adds assumptions.
4. **Can you define a measurable outcome?** `measure()` must observe something concrete each timestep. Subjective outcomes (e.g., "clinician satisfaction") need a proxy.
5. **Is the causal question counterfactual?** The branched engine answers "what would have happened without the AI?" For "which of three interventions is best?", you need multiple runs.
6. **Is the process discrete-time?** Continuous-time dynamics (pharmacokinetics, intraoperative hemodynamics) are outside scope.

---

## Core Mechanisms

### RNG Stream Partitioning

NumPy's `SeedSequence` cryptographically generates child seeds from a master seed. The `RNGPartitioner` spawns five such child seeds, one per stream:

```python
STREAM_NAMES = ["population", "temporal", "prediction", "intervention", "outcomes"]
```

Each stream is an independent `np.random.Generator`. Drawing from the `prediction` stream does not advance the `temporal` stream. When the `BranchedSimulationEngine` creates the counterfactual branch, it calls `partitioner.fork()` to produce a new partitioner with the same master seed spawned into a fresh child sequence — ensuring the counterfactual branch has independent mutable generator state but identical starting positions.

This provides a mathematical guarantee, not a statistical approximation: the factual and counterfactual branches draw from identical RNG positions in the `temporal` stream at every timestep (provided `step()` is pure). Divergence between branches comes only from intervention-induced state changes.

### Heterogeneous Patient Risk

Real patient populations exhibit extreme risk heterogeneity — most patients have minimal risk while a small fraction drive the majority of events. A naive uniform-risk model would produce wildly unrealistic intervention effects, resource requirements, and model discrimination challenges.

The SDK provides `beta_distributed_risks()` — right-skewed Beta sampling with a concentration parameter that controls the degree of heterogeneity. Risks are scaled to ensure the population-mean incident rate exactly matches the target. Extension points exist for mixture distributions and empirical distributions, for teams that have trusted risk models from real data.

```python
raw_risks = rng.beta(alpha=0.5, beta_param, n_patients)
scaling_factor = annual_incident_rate / np.mean(raw_risks)
base_risks = np.clip(raw_risks * scaling_factor, 0, 0.99)
```

This is the foundation for realistic simulation. Without heterogeneity, there are no high-risk patients for the ML model to identify and no differential intervention benefit to measure.

### Time-Varying Risk via AR(1)

Patient risk is not static. Seasonal patterns, life events, disease progression, and healthcare interactions all create temporal variation. The SDK models this via a first-order autoregressive (AR(1)) process on a multiplicative risk modifier:

```
modifier_{t+1} = ρ · modifier_t + (1 - ρ) · 1.0 + ε_t
```

with `ρ ∈ [0.80, 0.95]` for persistence, `ε_t ~ N(0, σ²)` for innovation, and bounds `[0.5, 2.0]`. The current-risk at each timestep is `base_risk × modifier × demographic_multiplier`.

This form is a deliberate choice for several reasons:

- **Analytically tractable** — AR(1) has closed-form moments and a clear persistence interpretation
- **Matches observed clinical variability** — ρ ≈ 0.9 produces realistic week-to-week correlation
- **Enables ITS and DiD** — creates trending time series with structure those methods can exploit
- **Bounded and centered** — the mean modifier stays near 1.0, so the population-level incidence rate remains close to the target

Optional seasonal effects can be added via `step_with_season()`, which superimposes a sinusoidal amplitude for use cases like flu season or summer heat waves.

### ML Model Simulation (Not Emulation)

The SDK does **not** train ML models. It simulates model *performance characteristics* — AUC, PPV, sensitivity, calibration — by injecting controlled noise into the true risk signal.

This is the right abstraction for pre-deployment questions. When evaluating a vendor tool, the question is never "what model will we build?" It is "what if the model achieves AUC 0.82 in our setting?" Training a real model would answer a different question: what *does* the model achieve, after you've invested in feature engineering and training data. The SDK exists to ask the first question without paying for the second.

The `ControlledMLModel` provides three modes:

- **`discrimination`** — target AUC only, useful for ranking-based interventions
- **`classification`** — target PPV + sensitivity at an optimized threshold, useful for binary alert systems
- **`threshold_ppv`** — target AUC + PPV at a fixed operating threshold

Implementation details:

1. The model requires `fit()` before `predict()` to optimize a 4-component noise injection (correlation, scale, label-noise strength, calibration offset) via 2D grid search against the target metric.
2. Platt scaling (`_platt_a`, `_platt_b`) achieves calibration so that scores can be interpreted as probabilities.
3. Feasibility is checked via Bayes bounds: given a population prevalence, the maximum achievable PPV at a given sensitivity has an analytic upper bound. If the target is infeasible, the model warns rather than silently failing.
4. `auc_score()` uses `np.lexsort` for stable ROC curve integration under tied values (a fix from the 30-paper audit after a boundary-condition test failed under perfect/inverted models).

When an audit paper doesn't report calibration metrics (0 out of 11 deployment papers did), the SDK can't validate against them, but it can at least report that the scenario's calibration assumption is now visible in the config.

### Intervention Mechanics and Capacity Constraints

The default intervention form is multiplicative risk reduction: a treated patient's event probability becomes `base_prob × (1 - effectiveness)`. This is analytically tractable, maps to real intervention effect sizes reported in the literature, and is additive on the log-hazard scale, which matches most clinical evidence.

But real interventions are almost always capacity-constrained. A rapid response team has finite bandwidth. A clinic has a fixed number of appointment slots per day. A nurse can only follow up with so many patients. The SDK makes capacity constraints first-class:

- **Hard capacity cap** — `rapid_response_capacity=N` means at most N patients per timestep receive the intervention, prioritized by prediction score
- **Response rate decay** — `clinician_response_rate = initial × exp(-fatigue × cumulative_false_alerts)` models alert fatigue
- **Reach rate** — fraction of targeted patients actually contacted (e.g., phone reminders: 60-80% reach)

The TREWS 2D sweep (baseline detection delay × capacity) showed flat rows across baseline delay and monotonic response to capacity. The binding constraint was operational bandwidth, not model performance. **Capacity, not AUC, is the most common real-world bottleneck.** Scenarios that don't model capacity will systematically overstate intervention benefit.

### Baseline Care and Time-Dependent Effectiveness

Two mechanisms added to the sepsis scenario that generalize as principles:

**Baseline clinical detection.** A per-patient delay drawn from `Beta(α, β) × max_hours` (configurable). When `t - onset_t >= detect_delay`, standard-of-care treatment is applied — on both branches. The distribution is right-skewed to match real clinical practice, where most cases are caught quickly but a long tail reflects atypical presentations and after-hours gaps.

**Kumar time-dependent treatment effectiveness.** Effectiveness decays exponentially with delay from event onset:

```
effectiveness(delay_hours) = max_effectiveness × 0.5^(delay_hours / half_life)
```

With a 6-hour half-life (per Kumar et al. 2006), immediate treatment gets 50% effectiveness, 6-hour delay gets 25%, 12-hour delay gets 12.5%. This makes the timing of detection genuinely consequential, amplifying the value of any mechanism that detects events earlier.

Together, these two mechanisms enable an honest counterfactual comparison: "what if we hadn't deployed the AI, given that standard care still catches most cases eventually?" This is the comparison health systems actually face — and it is the one the SDK now supports.

These mechanisms are in the sepsis scenario today. A task in Appendix B is to generalize them so that any scenario can opt into baseline-care semantics.

### Vectorization as a Performance Principle

No patient-level Python loops for population-level computation. The SDK uses NumPy broadcasting everywhere — pre-computing time-varying hazards, applying intervention multipliers element-wise, generating all random draws in single calls.

This is not just a performance optimization. It enables:

- **Large populations** — 5,000 to 100,000+ entities per simulation
- **Multi-seed variance analysis** — 30 replications in minutes instead of hours
- **Parameter sweeps** — 2D sweeps with 28+ configurations
- **Real-time interactive exploration** — changing a parameter and re-running in seconds

A patient-level loop implementation of the sepsis scenario would take ~8 hours for a 30-seed run. The vectorized implementation takes ~10 minutes. That difference is the difference between "run this once and trust it" and "run it 30 times and report a confidence interval."

---

## Analysis-Ready Data, Not Turnkey Causal Inference

The SDK produces the data shapes each causal method needs; it does not prescribe the analysis. This is a deliberate choice. The SDK's job is to provide known ground truth and realistic simulation; the analyst's job is to evaluate whether their preferred method recovers that truth on that data.

The `AnalysisDataset` class offers four export formats, each with unit-of-analysis metadata so downstream analyses know what they're looking at:

| Method | Format | Used For |
|--------|--------|----------|
| `to_time_series()` | Per-timestep aggregated rates | Interrupted Time Series (ITS) |
| `to_panel()` | Entity-timestep long-format panel | Difference-in-Differences (DiD), fixed-effects regression |
| `to_entity_snapshots(t)` | Entity-level cross-section at time `t` | Regression Discontinuity Design (RDD), requires continuous score + threshold |
| `to_subgroup_panel(subgroup_key)` | Stratified panel (race, insurance, age, etc.) | Equity analysis, Oaxaca-Blinder decomposition |

Each format includes unit-of-analysis metadata: `"patient"`, `"appointment_slot"`, `"patient_admission"`, `"rt_course"`. Downstream analyses can use this to compute the right standard errors (e.g., clustering on patient for multi-slot panels).

### Why Not Turnkey?

The SDK could ship with a `run_its_analysis()` function that takes results and returns an effect estimate. We chose not to, for two reasons:

1. **Analytic method choice is domain-specific.** The right bandwidth for an RDD, the right control for parallel trends in a DiD, the right way to handle autocorrelation in an ITS — these depend on the scenario and the analyst's judgment, not on the SDK.
2. **Baking in one implementation would freeze method choices.** New methods emerge; existing methods evolve. Keeping analysis out of the SDK means the SDK doesn't go stale when the statistics literature moves.

---

## Verification Protocol

Every simulation run, no exceptions, goes through this protocol before results are interpreted. This is the enforcement mechanism for the "Verification Before Interpretation" principle.

### Level 1: Structural Integrity

- No NaN or Inf in any output array
- Population size constant across timesteps
- All prediction scores in [0, 1]
- Confusion matrix identity: TP + FP + TN + FN = N
- Entity IDs set on all outcomes

### Level 2: Statistical Sanity

- Observed event rate within 4-sigma CLT bounds of target rate
- Achieved model AUC within ±0.03 of target
- Achieved sensitivity/PPV within ±0.05 of target (if classification mode)
- Demographic proportions match specified distributions
- AR(1) variance matches theoretical expectation

### Level 3: Conservation Laws

- **Null treatment identity:** `effectiveness=0` → factual == counterfactual exactly
- **Monotonicity:** higher effectiveness → fewer factual events
- **Null threshold:** `threshold=1.0` → no treatments → factual == counterfactual
- **Bayes bound:** achieved PPV ≤ theoretical_PPV(prevalence, sensitivity, specificity)

### Level 4: Case-Level Walkthroughs

Trace 3-5 individual entities through the full simulation in narrative form. Present as stories, not arrays:

- **High-risk treated entity** — flagged by model, treated, outcome explained by reduced risk
- **Low-risk untreated entity** — not flagged, outcome consistent with untreated population rate
- **Counterfactual entity** — identical risk trajectory as factual pre-intervention
- **Boundary entity** — score near threshold, assignment deterministic given score

Case walkthroughs catch bugs that aggregate statistics miss. A scenario can have perfectly correct population-level rates while silently mis-treating individual entities.

### Level 5: Boundary Conditions (First Run of New Scenarios)

- `threshold=0` → all patients treated
- `threshold=1` → no patients treated
- `effectiveness=0` → factual == counterfactual exactly
- `effectiveness=1.0` → near-zero events in factual (for preventable events)
- `AUC=0.50` → no discriminative targeting value (treatment randomly assigned relative to true risk)

These tests are the ones that caught the `auc_score` tied-value bug fixed in the 30-paper audit.

---

## What This SDK Enables

Six concrete question types, each with an example drawn from validated work in this repository.

### 1. "How good does the model need to be?"

**Example:** The TREWS 30-seed replication. Running the sepsis scenario at AUC=0.82 with the calibrated TREWS config produces a 4.27pp mean mortality reduction among septic patients (95% CI: 3.30-5.56), consistent with the published 3.3pp adjusted reduction. The sweep across capacity × baseline delay showed that at this AUC, the binding constraint is not model performance but operational capacity — a finding that changes the deployment conversation from "build a better model" to "staff the response team adequately."

### 2. "How many resources should we apply?"

**Example:** The sepsis capacity sweep. Varying `rapid_response_capacity` from 10 to 200 per 4-hour block, the mortality reduction scales monotonically with capacity while AUC holds constant. At cap=10, only 2.4% of septic patients get earlier ML treatment than baseline; at cap=200, 28.3% do. The per-patient Kumar effectiveness advantage is 3-5pp for patients who benefit. These numbers inform staffing decisions before deployment.

### 3. "How should we design real-world evaluation?"

**Example:** Export the same scenario as time-series (for ITS), panel (for DiD), and entity snapshots (for RDD). Run each method and check which recovers the known true effect. If your preferred method can't find the effect in simulation — where you know the effect exists — it won't find it in production.

### 4. "Will this be equitable?"

**Example:** The Rosen no-show replication. ML-targeted phone reminders produce a 5.3pp no-show reduction for Black patients versus 3.2pp for White patients, narrowing the racial disparity gap by 2.1pp. The equity mechanism emerges automatically from the multiplicative effectiveness model applied to higher-baseline-risk patients — no per-race effectiveness parameters needed. This reproduces the Rosen et al. (2023) finding that ML-targeted interventions can narrow, rather than exacerbate, demographic gaps.

### 5. "Should we buy this vendor's tool?"

**Example:** A vendor claims their sepsis model achieves AUC 0.82 at a 7% alert rate. Run the claim through `sepsis_early_alert/configs/trews_replication.yaml` with your institution's patient volume and sepsis incidence. Estimate the mortality benefit under optimistic (TREWS-like capacity + confirmation rate) and pessimistic (ESM-like response + fatigue) assumptions. Compare against the alert fatigue benchmarks from Hussain et al. If the pessimistic scenario shows zero benefit, the vendor's claims depend on operational assumptions you may not satisfy.

### 6. "Will our monitoring plan detect the effect?"

**Example:** Before deploying, simulate the planned rollout with the known effect size you're trying to detect. Run the proposed ITS analysis on the simulated data. If it can't recover the effect, the real-world analysis won't either — and you need a different evaluation design before you commit.

---

## Validated Against Published Literature

The SDK has been stress-tested against 30 landmark healthcare AI deployment papers in a systematic reproducibility audit. The results establish three things:

**The SDK works.** Among papers with sufficient deployment data, the simulation engine correctly models alert burden, intervention direction, and in the cleanest cases, reproduces published effect sizes precisely.

| Reproducibility | Count | Papers |
|-----------------|-------|--------|
| **REPRODUCED** | 3 | Hong (SHIELD-RT), Adams (TREWS), Chong/Rosen (No-Show) |
| PARTIALLY_REPRODUCED | 4 | Wong (ESM), Escobar (AAM), Boussina (COMPOSER), Manz (Nudges), Lång (MASAI) |
| NOT_REPRODUCED | **0** | — |
| UNDERDETERMINED | 6 | Shimabukuro, Wijnberge, Henry, Edelson, Rajkomar, Obermeyer |

**Zero papers were NOT_REPRODUCED.** The SDK never produced a result that contradicted a paper's direction of effect. All discrepancies were magnitude gaps, calibration limitations, or missing parameters — not conceptual failures.

**Scientific reporting quality is the binding constraint, not SDK capability.** The single most common failure mode across the 30 papers was not a simulation architecture problem — it was that papers did not report the parameters needed to independently reproduce their findings. AUC is missing from 36% of deployment papers. Calibration metrics from 100%. Demographic stratification from 73%. The reproducibility crisis in clinical AI is not a lack of trials; it is a lack of reporting discipline in the trials that exist.

Full audit results: [`research/synthesis_report.md`](../research/synthesis_report.md) | 30 individual paper verdicts: [`research/paper_verdicts/`](../research/paper_verdicts/) | Deep-dive educational packet: [`research/educational_packet.md`](../research/educational_packet.md)

---

## Deliberate Scope Boundaries

What the SDK is NOT trying to do. These are design choices, not deficiencies:

- **Continuous-time physiology.** The engine is discrete-time by design. Sub-minute hemodynamics, pharmacokinetics, and vasopressor titration are outside scope. Paper 8 in the audit (Wijnberge HYPE) represents this class. The SDK documents this gap explicitly; teams with continuous-time needs should use a different tool.
- **Individual clinical decision support.** The SDK is population-level. It does not answer "should *this* patient get this drug?" — only "under a policy that treats patients above threshold X, what happens to outcomes?"
- **Multi-agent social dynamics.** One entity type per scenario. Simultaneous modeling of patients, providers, and payers requires simplifying assumptions that break the 5-method contract.
- **Qualitative implementation failure.** Connectivity outages, workflow disruption, trust erosion, nurse bypass behavior are not expressible in `step()`. Paper 10 in the audit (Beede Google DR) represents this class. These are legitimate research topics for a different methodology.
- **Model training or evaluation.** The SDK simulates model performance characteristics. It does not train models, does not evaluate real models on real data, and does not produce regulatory submission artifacts.
- **Turnkey causal inference analysis.** The SDK produces data shapes; the analyst runs the method and interprets the result.

A use case that fails multiple fit criteria is a signal to use a different tool, not to stretch the SDK.

---

## Agent-Assisted Workflow

The SDK is designed to be used with an AI assistant (Claude Code, Claude Cowork, or similar) as a collaborator. The agent protocol lives in [`.claude/agents/sim-guide.md`](../.claude/agents/sim-guide.md) and covers four phases:

- **Phase 1: Use-case fitness screening** — walk through the six fitness questions with the human; flag issues as they emerge; be direct when the SDK is not the right tool
- **Phase 2: Scenario development** — guide the 5-method design conversation, translate domain language into the contract
- **Phase 3: Verification** — run the 4-level protocol on every simulation, refuse to interpret until verification passes
- **Phase 4: Stakeholder translation** — translate findings into the right language for clinical leaders (operational impact), governance (equity audits, burden distribution), research teams (causal method validation), and procurement (vendor claim stress-testing)

The agent is a collaborator, not an autopilot. The human frames the question, reviews the scenario, runs verification, and owns the deployment decision. The agent thinks with the human; it does not think for them.

Any Claude Code session opening this repository picks up the protocol automatically.

---

## Appendix A: Lessons Learned Building This SDK

Four concrete lessons from the audit and TREWS work, each with a short story.

### The Baseline Care Lesson (TREWS)

Initial runs of the TREWS replication produced 0.16pp mortality reduction versus the published 3.3pp — a 20x gap. The root cause was not AUC, not capacity, not calibration. It was that the counterfactual branch assumed nobody detected sepsis without the AI. In reality, clinicians catch sepsis through routine care (vitals, lactate orders, clinical suspicion). The AI's value is the *time advantage* over that baseline, not the total detection benefit.

Fix: implement baseline clinical detection as a per-patient Beta-distributed delay, running on both branches. Add Kumar time-dependent treatment effectiveness to reward earlier detection. Result: 4.27pp mean reduction, 95% CI [3.30, 5.56] across 30 seeds, published value within CI.

**Lesson:** The counterfactual in a well-designed healthcare simulation must reflect competent standard of care. Anywhere the SDK is asked "what happens without the AI," the honest answer includes whatever clinicians do anyway.

### The Capacity-Over-AUC Lesson (TREWS Sweep)

A 2D sweep across baseline delay × capacity showed flat rows — baseline delay had no effect on ML value-add — and monotonic capacity response. At capacity=10 (rapid response team model), 2.4% of septic patients benefited from earlier ML treatment. At capacity=200 (decentralized bedside model, matching the real TREWS deployment), 28.3% benefited.

Reading the Adams et al. paper revealed the reason: TREWS was decentralized with no dedicated staff — every bedside provider evaluated alerts in Epic. The initial config of `capacity=10` modeled the wrong operational structure.

**Lesson:** AUC alone does not determine clinical impact. Operational bandwidth does. Any evaluation of an ML deployment that ignores capacity constraints will systematically overstate the benefit. Any vendor claim that does not disclose its assumed workflow model is underspecified.

### The Reporting Quality Lesson (30-Paper Audit)

Across 30 papers, the SDK never produced a result that contradicted a paper's direction of effect (0 NOT_REPRODUCED). In every case where full reproduction failed, the failure was traced to missing parameters in the published paper: AUC (missing in 36% of deployment papers), calibration (100%), demographic stratification (73%), clinician response rate (all papers), per-patient intervention effectiveness (almost all).

The single paper reproduced to within 2% of its published target (SHIELD-RT) is the only one that reported a binary outcome, a clean AUC range, an explicit threshold, and a randomized counterfactual.

**Lesson:** The reproducibility crisis in clinical AI is not a lack of trials. It is a lack of reporting discipline in the trials that exist. When papers report what they need to report, the SDK reproduces them precisely. This is a finding about the state of the field, not about the SDK.

### The Unit-of-Analysis Polymorphism Lesson (Stroke vs No-Show vs Sepsis)

The first two scenarios built for the SDK — stroke prevention and no-show overbooking — had almost nothing in common structurally. Stroke used scalar per-patient risks. No-show used nested dataclasses with patients, provider schedules, and waitlists. The temptation at the time was to define a common state schema that both could inherit from. We resisted.

When the sepsis scenario was added later, it used 16-row NumPy arrays per patient-admission — yet another structure. The SDK accepted it without modification. The 5-method contract is generic over state type, and the core engine never inspects state internals. Scenarios carry their own state opaquely.

Had we defined a common schema, we would have forced the sepsis scenario to compromise its design or required engine modifications to accommodate it. The polymorphism is what lets the same engine serve scenarios that look nothing like each other.

**Lesson:** When you don't know the future scenarios, make the core engine refuse to look inside state. The cost (slightly more boilerplate in scenario code) is trivial compared to the benefit (any domain model can plug in).

---

## Appendix B: What's Missing That We Should Apply

A candid list of items we know we should add, drawn from the reference document and the 30-paper audit. These are open tasks, not commitments. The SDK ships today with what it has; these are candidates for the next batch of work.

### Causal Method Support

- **Parallel trends diagnostic for DiD** — port from the reference implementation. Given a panel, compute the slope of treated and control pre-intervention and flag significant differences.
- **Durbin-Watson autocorrelation check for ITS** — port from the reference. Compute the DW statistic on residuals of the segmented regression model.
- **Synthetic Control export format** — add alongside ITS/DiD/RDD. Requires multi-unit staggered intervention timing, which the SDK doesn't yet model at the engine level.
- **Power analysis tooling** — given a true effect size and a method, estimate the sample size needed to detect it at specified α and power.

### Model Simulation Extensions

- **Adversarial noise mode for `ControlledMLModel`** — enable AUC targets below 0.70 at low prevalence. Required to correctly model the Epic ESM's real-world AUC of 0.63. The current noise model converges to AUC ≥ 0.75 at prevalence < 7%, regardless of target.
- **Detection-timing correlation mode** — make model scores correlate with proximity to disease onset, not just current risk level. Would enable full reproduction of TREWS's "detect before clinical suspicion" mechanism.

### Fairness and Governance

- **Triple fairness reporting** — compute equal outcomes, equal performance, and equal allocation for every scenario run, with flags on the inherent tradeoffs between them. Papers 21 (Rajkomar fairness) and 22 (Chen/Pierson) establish this as a regulatory and governance requirement.
- **Proxy variable risk check** — Phase 1 design check that flags whether the outcome variable is a direct measure of health need or a utilization proxy. Obermeyer (Paper 15) showed this catches the failure mode before deployment.

### Analysis and Sensitivity Discipline

- **Sensitivity analysis bands** — explicit optimistic / realistic / pessimistic variants for every simulation, using alert fatigue benchmarks from Hussain et al. (Paper 30) as the default pessimistic settings.
- **Pre-registration templates** — specify what will be measured before running the simulation, to prevent post-hoc hypothesis rescue.
- **Multi-cohort staggered deployment engine** — would enable full stepped-wedge ITS reproduction for Escobar (AAM 21-hospital rollout) and COMPOSER (multi-year BSTS).

### Scenario Generalization

- **Generalize baseline care and Kumar decay** — these mechanisms live in the sepsis scenario today. Lift them to the `BaseScenario` level so any scenario can opt into baseline-care semantics without reimplementing them.
- **Capacity-constrained intervention helper** — common pattern across sepsis, no-show, and SHIELD-RT scenarios. Could be a reusable utility.

Each item is an open task. The SDK is useful today; these are the candidates for when we next cut a minor version.

---

*This document describes the SDK as of the 30-paper audit reconciliation (April 2026). The current state of code and scenarios is the source of truth; where this document and the code disagree, the code wins and the document should be updated.*
