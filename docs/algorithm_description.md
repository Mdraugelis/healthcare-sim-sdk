# Algorithm Description Document

**Healthcare Intervention Simulation SDK**

| | |
|---|---|
| Document | Algorithm Description Document (ADD) |
| Version | 0.1, draft for review |
| Date | 15 October 2026 |
| Code baseline | Branch `fix/label-free-model-scores` (PR #5) at commit `6db58a6`. Where `main` differs, the text says so. |
| Companion notebooks | Planned, not yet written. Section 15 specifies them. |
| Status of numbers | Every number in this document was produced by running the SDK at the baseline above. The notebooks in Section 15 will reproduce each one. |

---

## Contents

1. [Scope](#1-scope)
2. [System overview](#2-system-overview)
3. [A1. Random number stream partitioning](#3-a1-random-number-stream-partitioning)
4. [A2. State representation and cloning](#4-a2-state-representation-and-cloning)
5. [A3. Branched counterfactual engine and coupling](#5-a3-branched-counterfactual-engine-and-coupling)
6. [A4. Step purity and draw-count invariance](#6-a4-step-purity-and-draw-count-invariance)
7. [A5. Population, event rates and temporal dynamics](#7-a5-population-event-rates-and-temporal-dynamics)
8. [A6. Controlled ML model simulation](#8-a6-controlled-ml-model-simulation)
9. [A7. Intervention impact](#9-a7-intervention-impact)
10. [A8. Outcomes, estimands and uncertainty](#10-a8-outcomes-estimands-and-uncertainty)
11. [A9. Evaluation design by simulation](#11-a9-evaluation-design-by-simulation)
12. [A10. Monitoring design: control charts](#12-a10-monitoring-design-control-charts)
13. [A11. Verification pillars](#13-a11-verification-pillars)
14. [Issues register](#14-issues-register)
15. [Companion notebooks](#15-companion-notebooks)
16. [Design principles this document adds](#16-design-principles-this-document-adds)
17. [References](#17-references)

---

## 1. Scope

### 1.1 Purpose

This document describes the algorithms inside the SDK precisely enough that a reader could re-implement them, test them, and know their limits. `docs/design_principles.md` explains why the SDK is built the way it is. This document explains how each part works, what it assumes, how it is verified, and where it currently falls short.

Each algorithm chapter follows the same template:

| Part | Content |
|---|---|
| Need | The problem the algorithm solves and what goes wrong without it |
| Inputs and outputs | What the algorithm consumes and produces |
| Method | The mathematics |
| Procedure | Step-by-step logic, matching the code |
| Assumptions and limits | What must hold for the output to be valid |
| Verification | How correctness is checked today, and gaps |
| Worked example | Numbers produced by running the SDK |
| Implementation | Files and functions |
| Open issues | Links to the issues register (Section 14) |

### 1.2 Audience

Analysts who configure scenarios and interpret results, developers who write scenarios, and reviewers who need to judge whether a result can be trusted.

### 1.3 Critical review

This document was written alongside a critical read of the code. Problems found are recorded in Section 14, with the evidence and a proposed fix for each. Several of them affect reported results. The most important are:

- Two scenarios break the coupling between the factual and counterfactual branches partway through a run (I-01).
- The monitoring validation's null-regime claims do not hold (I-04, I-05, I-06).
- Re-running a scenario object silently decouples its branches (I-03).

### 1.4 Related documents

| Document | Relationship |
|---|---|
| `docs/design_principles.md` | Rationale. This document supersedes its "Core Mechanisms" section on points of detail. |
| `docs/analytical_first_pass.md` | Closed-form screening that should precede any simulation. Section 9.6 links to it. |
| `.claude/agents/sim-guide.md` | Agent protocol, including the verification checklist that Section 13 formalizes. |
| `.claude/invariants.yaml` | Machine-readable invariants. Section 6.5 states which ones are actually enforced. |

### 1.5 Notation

| Symbol | Meaning |
|---|---|
| $i$ | Entity index (patient, nurse, appointment slot) |
| $t$ | Timestep index, $t = 0, \dots, T-1$ |
| $\Delta$ | Timestep length in years (`TimeConfig.timestep_duration`) |
| $r_i$ | Entity's annual risk (probability of the event within one year) |
| $h_i$ | Hazard rate, per year |
| $p_{i,t}$ | Probability of the event in timestep $t$ |
| $p$ | Prevalence or mean risk in the population |
| $F$, $C$ | Factual branch (AI deployed) and counterfactual branch (no AI) |
| $Y^F_{i,t}$, $Y^C_{i,t}$ | Outcome of entity $i$ at time $t$ on each branch |
| $U$ | A standard uniform random variable |
| $\Phi$, $\Phi^{-1}$ | Standard normal CDF and its inverse |
| $\sigma(x)$ | Logistic function $1/(1+e^{-x})$ |
| AUC | Area under the ROC curve |
| PPV | Positive predictive value, $P(Y=1 \mid \text{flag})$ |
| sens | Sensitivity, $P(\text{flag} \mid Y=1)$ |
| FPR | False positive rate, $1-\text{specificity}$ |

---

## 2. System overview

### 2.1 The problem

When a model flags a patient and a clinician acts, a good outcome hides the evidence. Nobody observes what would have happened without the flag. The SDK builds a synthetic population in which both worlds are computed: one with the AI deployed and one without it. The difference between them is the true effect, known exactly. That known effect is used to answer pre-deployment questions:

- How good must the model be?
- How much staff capacity is needed?
- Will a planned evaluation detect the effect?
- Will a monitoring plan raise false alarms?

### 2.2 Processing chain

```mermaid
flowchart TD
    A[create_population<br/>population stream] --> B[clone_state]
    B --> CF0[Counterfactual state]
    A --> F0[Factual state]
    subgraph loop[Each timestep t]
      F0 --> FS[step<br/>factual temporal stream]
      CF0 --> CS[step<br/>counterfactual temporal stream]
      FS --> P{t in prediction<br/>schedule?}
      P -- yes --> PR[predict<br/>prediction stream] --> IV[intervene<br/>intervention stream] --> FM
      P -- no --> FM[measure factual<br/>outcomes stream]
      CS --> CM[measure counterfactual<br/>outcomes stream]
    end
    FM --> R[SimulationResults]
    CM --> R
```

The counterfactual branch never calls `predict()` or `intervene()`. It answers one question: what happens without the AI, with standard care still in place.

### 2.3 Algorithm inventory

| ID | Algorithm | Main code | Existing notebook |
|---|---|---|---|
| A1 | Random number stream partitioning | `core/rng.py` | 01 |
| A2 | State representation and cloning | `core/scenario.py` (`clone_state`) | 02 |
| A3 | Branched counterfactual engine and coupling | `core/engine.py` | 03 |
| A4 | Step purity and draw-count invariance | `core/scenario.py` contract, `tests/integration/test_step_purity.py` | 04 |
| A5 | Population, event rates, temporal dynamics | `population/risk_distributions.py`, `population/temporal_dynamics.py` | none |
| A6 | Controlled ML model simulation | `ml/model.py`, `ml/performance.py`, `ml/probability_model.py` | 05 and 06 cover legacy classes only |
| A7 | Intervention impact | Each scenario's `intervene()` and `step()` | 08, 09 (partial) |
| A8 | Outcomes, estimands, uncertainty | `core/results.py`, scenario `measure()` | 07 |
| A9 | Evaluation design by simulation | `experiments/analysis/its.py`, malnutrition `power_map.py` | `its_estimator_demonstration` |
| A10 | Monitoring design: control charts | `scenarios/nurse_retention/monitoring/` | none |
| A11 | Verification pillars | `tests/`, scenario verification runners | `sanity_checks` (partial) |

---

## 3. A1. Random number stream partitioning

### 3.1 Need

A simulation must be reproducible: the same seed gives the same result. A branched simulation also needs something stronger. Random draws that represent the same real-world event must line up across the two branches. If the factual branch's model-noise draws advanced the same generator that drives disease progression, the counterfactual branch (which makes no model draws) would see different disease progression. The difference between branches would then mix the AI's effect with arbitrary noise.

### 3.2 Inputs and outputs

- Input: one integer master seed.
- Output: five independent generators, one per process.

### 3.3 Method

`RNGPartitioner` wraps `numpy.random.SeedSequence(master_seed)` and spawns five child sequences, one per stream name. Each child seeds its own `numpy.random.Generator` (PCG64). `SeedSequence` guarantees that children are statistically independent and reproducible.

`fork()` builds a new `SeedSequence` from the same master seed. Its first spawn yields children with the same entropy as the original partitioner's first spawn, so the forked generators start in exactly the same states as the original ones did. They are separate objects, so consuming one does not move the other.

### 3.4 Stream assignment

| Stream | Used by | Runs on | Must stay synchronized across branches? |
|---|---|---|---|
| `population` | `create_population` | Once, before cloning | No. The counterfactual branch receives a clone and never draws. |
| `temporal` | `step` | Both branches | **Yes** |
| `prediction` | `predict` | Factual only | No |
| `intervention` | `intervene` | Factual only | No |
| `outcomes` | `measure` | Both branches | **Yes** |

Only the two shared streams need synchronization. Factual-only streams may consume any number of draws.

### 3.5 Procedure

1. `BaseScenario.__init__` creates `RNGPartitioner(seed)` and calls `create_streams()` once. The result is stored as `self.rng`.
2. At the start of a branched run, the engine calls `create_population` using the factual streams, then clones the state.
3. The engine calls `self._partitioner.fork().create_streams()` to get the counterfactual streams.
4. Before each hook call, the engine sets `scenario.rng` to the stream set of the branch it is about to run.

Scenario code must therefore read `self.rng.<stream>` inside each method call. Caching a generator on `self` at construction time would bypass the swap. No scenario currently does this.

### 3.6 Assumptions and limits

- The partitioner must spawn exactly once before forking. A second `create_streams()` on the same partitioner yields different children. No code does this today.
- `seed=None` becomes seed 42, not a fresh random seed (I-19).
- Re-running the same scenario object breaks synchronization (I-03, below).

### 3.7 Verification

Notebook 01 demonstrates reproducibility, stream independence, cross-stream correlation near zero, and fork equivalence. Synchronization across a real run is not tested anywhere. Section 5.6 proposes a stream-position audit that would test it.

### 3.8 Worked example: re-running a scenario object

A stroke scenario at zero intervention effectiveness must produce identical factual and counterfactual outcomes, because the AI changes nothing. Run with seed 7 and 2,000 patients for 26 weeks:

| Run | Entity-weeks where factual ≠ counterfactual | Factual identical to a fresh object's run? |
|---|---|---|
| First `engine.run()` | 0 | Yes |
| Second `engine.run()` on the same scenario object | 118 | No |

On the second run, the factual branch keeps using streams already consumed by the first run, while the engine forks fresh counterfactual streams from the master seed. The branches start out of step. No current runner re-uses a scenario object, so no reported result is affected. Nothing prevents it, though, and notebooks or sweeps could easily do it.

### 3.9 Implementation

`healthcare_sim_sdk/core/rng.py` (`RNGStreams`, `RNGPartitioner.create_streams`, `RNGPartitioner.fork`); `core/engine.py` lines 73–113.

### 3.10 Open issues

I-03, I-19.

---

## 4. A2. State representation and cloning

### 4.1 Need

The counterfactual branch must evolve its own copy of the population. If the two branches shared mutable objects, an intervention on one branch would leak into the other. The engine must also work with any state shape: a NumPy array for sepsis, nested dataclasses for no-show clinics.

### 4.2 Method

State is a generic type `S`. The engine never inspects it. `BaseScenario.clone_state(state)` returns an independent copy. The default is `copy.deepcopy`. Array-backed scenarios override it with `state.copy()`, which is faster and sufficient.

Cloning happens:

- once, right after `create_population`, in BRANCHED mode;
- once per prediction timestep, before `intervene`, in SNAPSHOT mode.

### 4.3 Rules for scenario authors

1. Every quantity that evolves during a run belongs in `state`. Scenario attributes (`self.*`) hold configuration and caches only.
2. Treat the incoming state as read-only. Copy it first (`new_state = state.copy()`), then modify and return the copy.
3. Any helper object that holds evolving values must live inside the state, not on `self`.

### 4.4 Assumptions and limits

The sepsis family keeps a system-wide alert-fatigue counter on `self._cumulative_false_alerts`. Only `intervene()` reads and writes it, and that runs on the factual branch only, so a single run is correct. The counter is never reset and is not part of the state. A second run on the same object starts with the first run's fatigue (I-14). Fitted-model caches (`self._model_fitted`) behave the same way.

`AR1Process` in `population/temporal_dynamics.py` stores its current values in `self.current`. A scenario that kept one on `self` and stepped it inside `step()` would share it across both branches. Both branches would advance the same process, which is a purity violation (I-12). No scenario uses the class today. Stroke and sepsis store their AR(1) modifiers in the state array, which is correct.

### 4.5 Verification

Notebook 02 shows mutation isolation for several state types and the speed difference between `deepcopy` and `np.copy`.

### 4.6 Implementation

`core/scenario.py` (`BaseScenario.clone_state`); overrides in `scenarios/stroke_prevention/scenario.py` and `scenarios/sepsis_early_alert/scenario.py`.

### 4.7 Open issues

I-12, I-14.

---

## 5. A3. Branched counterfactual engine and coupling

### 5.1 Need

The SDK's central claim is that it knows the true effect. That requires a counterfactual trajectory computed from the same population, under the same background randomness, with only the AI removed. Comparing two independent runs would also give an unbiased effect on average, but every comparison would carry the full outcome noise of both runs.

### 5.2 Method

The engine maintains two state trajectories from one shared initial population. At each timestep:

1. `step` on the factual state, with the factual streams.
2. `step` on the counterfactual state, with the counterfactual streams.
3. If $t$ is in the prediction schedule: `predict` then `intervene` on the factual state only.
4. `measure` on both states, each with its own branch's outcomes stream.

Because the two branches' temporal and outcomes streams start in identical states, the $k$-th draw on one branch equals the $k$-th draw on the other. If both branches consume draws in the same order and number, every random event that the AI did not touch resolves the same way on both branches. This is the method of common random numbers.

### 5.3 Why coupling matters: the mathematics

Most outcomes are drawn as $Y = \mathbb{1}[U < p]$. Suppose on one entity-step the factual probability is $p^F$, the counterfactual probability is $p^C$, and $p^F \le p^C$.

With a shared uniform $U$:

$$D = Y^C - Y^F = \mathbb{1}[p^F \le U < p^C] \sim \text{Bernoulli}(p^C - p^F)$$

$$\mathbb{E}[D] = p^C - p^F, \qquad \text{Var}(D) = (p^C - p^F)(1 - p^C + p^F)$$

With independent uniforms, the mean is the same but:

$$\text{Var}(D) = p^C(1-p^C) + p^F(1-p^F)$$

For an untreated entity, $p^F = p^C$. With coupling its contribution to the effect is exactly zero. With independent draws it contributes noise of about $2p$. In a population where few entities are treated, coupling removes almost all of the noise from the effect estimate.

This holds per entity-step while the two state paths agree except for the treatment. Once outcomes feed back into state (an absorbing event such as death or departure), the paths diverge for that entity. Coupling still holds for every other entity.

### 5.4 Counterfactual modes

| Mode | What runs | Counterfactual recorded | Coupled? | Use |
|---|---|---|---|---|
| BRANCHED | Two full trajectories | Every timestep | Yes, if A4 holds | Default for all effect estimates |
| SNAPSHOT | One trajectory; clone before each `intervene`, measure the clone | Prediction timesteps only | **No.** Both `measure` calls draw from the same outcomes stream in sequence, so the clone and the real state use different uniforms. | Legacy. Same-step effects only, no downstream propagation. |
| NONE | One trajectory | Never | Not applicable | Purity testing, and runs where no comparison is needed |

### 5.5 Definition of the counterfactual

The counterfactual is "no AI, standard care continues". Standard care must be modeled inside `step()` so that it runs on both branches. The sepsis scenario's baseline clinical detection and nurse retention's standard-of-care check-ins are examples. A counterfactual with no standard care inflates the AI's benefit; `design_principles.md` §6 documents how that happened in the first sepsis scenario.

### 5.6 Verification: what is tested and what is not

| Test | Proves | Does not prove | Status |
|---|---|---|---|
| BRANCHED factual = NONE trajectory (`assert_purity`) | The counterfactual branch does not disturb the factual branch | That the branches stay coupled | Exists for stroke, sepsis, both no-show scenarios, malnutrition. None of the seven paper replications. |
| Zero-effect identity: effectiveness = 0 ⇒ factual ≡ counterfactual | The intervention is the only source of difference **when it changes nothing** | Coupling once the intervention does change state | Partial: some scenarios, as tests or in notebooks |
| **Stream-position audit** (proposed) | After every step, both branches have consumed the same number of draws from the shared streams | — | Not implemented. Proposed in I-01. |
| **Untreated-entity identity** (proposed) | In scenarios without interaction between entities, never-treated entities have identical outcomes on both branches | — | Not implemented |

The audit was run for this document by replaying the engine loop and comparing the bit-generator positions of the two branches after each step:

| Scenario | Temporal stream | Outcomes stream |
|---|---|---|
| Stroke prevention | Coupled for all 52 steps | Coupled |
| Sepsis early alert | Coupled for all 42 steps | Coupled |
| Epic ESM (paper 01) | Coupled for all 84 steps | Coupled |
| No-show targeted reminders | Coupled for all 30 steps | Coupled |
| **Nurse retention** | **Decouples at week 4** | Coupled |
| **No-show overbooking** | **Decouples at day 1** | Coupled |

Section 6 explains the cause.

### 5.7 Worked example: coupling and ground truth

Stroke scenario, 5,000 patients, 52 weeks, intervention effectiveness 0.5, treatment threshold 0.005 on the calibrated score, seed 3.

**Expected versus realized effect.** Because both branches' event probabilities are known at every step, the expected number of events prevented can be computed exactly as $\sum_{i,t} (p^C_{i,t} - p^F_{i,t})$.

| Quantity | Value |
|---|---|
| Expected events prevented | 84.9 |
| Realized events prevented (one run) | 72 |
| Patients ever treated | 902 |
| Patients whose outcome differs between branches | 67 |
| Untreated patients whose outcome differs | **0** |

The zero in the last row is coupling working as designed. Only treated patients can contribute to the effect.

**Variance reduction.** The same population and model, with 200 replicates that vary only the outcome draws:

| Design | Mean events prevented | Standard deviation |
|---|---|---|
| Coupled (shared outcome draws) | 84.9 | 9.6 |
| Decoupled (independent outcome draws per branch) | 86.9 | 20.6 |

The coupled mean matches the expected 84.9. Its standard deviation is close to the theoretical floor $\sqrt{84.9} = 9.2$. Variance ratio: 4.6. To reach the coupled design's precision, a decoupled design would need about 4.6 times as many replicates.

### 5.8 Implementation

`core/engine.py` (`BranchedSimulationEngine._run_branched`, `_run_snapshot`, `_run_simple`).

### 5.9 Open issues

I-01, I-02, I-03, I-13.

---

## 6. A4. Step purity and draw-count invariance

### 6.1 Need

Coupling (Section 5) only works if each branch's `step()` and `measure()` depend on nothing except their own inputs, and consume the shared streams identically. Two separate conditions are needed. The SDK currently states only the first.

### 6.2 Condition 1: purity

`step(state, t)` must be a function of `state`, `t`, and the temporal stream only:

- no reads of mutable scenario attributes;
- no writes to anything except the returned state;
- no draws from any other stream.

Reading immutable configuration from `self` is allowed. A time-varying parameter is allowed if its value is a pure function of `t`; `TimeVaryingParameter.value_at(t)` in nurse retention is an example.

### 6.3 Condition 2: draw-count invariance

**The number of draws `step()` and `measure()` take from a shared stream must not depend on state.**

A function can be pure and still violate this. For example, `rng.random(n_active)` is a pure function of state, but `n_active` differs between branches once the AI has changed who is active. From that point on, the two branches read from different positions in the stream. Every later random event on the counterfactual branch is drawn independently of its factual counterpart.

Effects stay unbiased, because each branch is still a valid sample path. What is lost is the pairing. Effect estimates become as noisy as two independent runs, and per-entity comparisons stop meaning anything.

Safe patterns:

| Need | Unsafe | Safe |
|---|---|---|
| Random draw per eligible entity | `rng.random(mask.sum())` | `u = rng.random(n)` then use `u[mask]` |
| Random order within a group | `rng.permutation(group)` | `u = rng.random(n)` once per step, then `group[np.argsort(u[group])]` |
| Conditional extra draw | `if overbooked: rng.random()` | Draw for every slot, use only where needed |
| Outcome for an entity that must pair exactly across branches | Draw at measure time | Draw a per-entity coin at population creation and store it in state (as malnutrition does) |

The last pattern is the strongest. A coin drawn once at creation and stored in state makes the pairing independent of any stream position.

### 6.4 The two violations found

**Nurse retention.** Standard-of-care allocation runs in `step()`, on both branches. For each manager, it shuffles eligible nurses with `rng.permutation(new_hires)` and `rng.permutation(others)`, and draws `rng.random(1)` when a team is empty. Team sizes depend on who has departed and who is in cooldown, which differs between branches once AI check-ins take effect. The code comment says "fixed RNG consumption"; that is not true. The audit shows the temporal stream decoupling at week 4. All later departures are drawn independently on the two branches.

**No-show overbooking.** `step()` draws two extra variates (`rng.normal` and `rng.random`) for each overbooked slot. Only the factual branch has overbooked slots. The temporal stream decouples at day 1. From then on, whether a given patient shows up is drawn independently on each branch, including in the per-patient burden and equity analyses.

### 6.5 What is enforced today

| Invariant (from `invariants.yaml`) | Enforced by | Gap |
|---|---|---|
| Five abstract methods | `scripts/check_invariants.py` (AST) | None |
| Five stream names | `check_invariants.py` | None |
| No bare `np.random` | `check_no_bare_random.py` | None |
| No stdlib `import random` | Nothing | Banned in `invariants.yaml`; the hook's patterns cover NumPy only |
| Engine does not inspect state | `check_invariants.py`: catches `getattr`, `isinstance`, `hasattr` on state variables | `invariants.yaml` also forbids subscript and attribute access; those are not checked |
| Counterfactual branch never calls `predict` | Code structure only | No automated check |
| Step purity | Per-scenario identity tests | Seven paper scenarios have no tests. `design_principles.md` calls purity "machine-enforced"; it is test-enforced where tests exist. |
| Draw-count invariance | Nothing | Not stated in any document before this one |

### 6.6 Verification

Notebook 04 demonstrates the purity test and what happens when purity or stream discipline is violated. It does not cover draw-count invariance. Section 15 adds that.

### 6.7 Open issues

I-01, I-02, I-12, I-17, I-22.

---

## 7. A5. Population, event rates and temporal dynamics

### 7.1 Need

Targeting only has value if risk varies across people. The shape of the risk distribution determines how many events the highest-risk patients carry, how well any model can rank patients (the Bayes ceiling, Section 8.4), and how many patients a fixed capacity can reach. Event rates must also convert correctly between the time scales used in configuration (annual risk) and in simulation (per-step probability).

### 7.2 Method: heterogeneous risk

`beta_distributed_risks(n, p, α, rng)` draws

$$r_i \sim \text{Beta}\!\left(\alpha,\ \alpha\left(\tfrac{1}{p} - 1\right)\right)$$

which has mean $p$ and variance

$$\text{Var}(r) = \frac{p^2 (1-p)}{\alpha + p}.$$

It then rescales the sample to have mean exactly $p$ and clips values at 0.99.

The parameter called `concentration` is the Beta $\alpha$, not the Beta precision $\alpha + \beta$. **Lower concentration means more spread.** Most patients then have very low risk and a few have very high risk.

The rescaling removes sampling error in the mean. If clipping binds, the realized mean falls slightly below $p$. At $p = 0.05$ and concentration ≥ 0.3, no values reach the clip.

### 7.3 Worked example: what concentration does

Annual risk $p = 0.05$, 200,000 patients, seed 2026. The Bayes ceiling (Section 8.4) is computed with one-year outcomes drawn from each patient's risk.

| Concentration $\alpha$ | SD of risk | 10th pct | Median | 90th pct | 99th pct | Expected events in the top 10% | Bayes ceiling |
|---|---|---|---|---|---|---|---|
| 4.0 | 0.024 | 0.022 | 0.046 | 0.083 | 0.122 | 20% | 0.643 |
| 1.0 | 0.048 | 0.0056 | 0.036 | 0.114 | 0.216 | 32% | 0.752 |
| 0.5 | 0.066 | 0.0009 | 0.024 | 0.136 | 0.303 | 42% | 0.828 |
| 0.3 | 0.082 | 0.00006 | 0.013 | 0.153 | 0.388 | 51% | 0.874 |

Reading the table: at concentration 4.0, risk is fairly uniform. The riskiest tenth of patients carry only 20% of events, and no model can exceed AUC 0.64. At concentration 0.3, the riskiest tenth carry half of all events and the ceiling is 0.87.

**Concentration must be chosen to match the AUC the scenario claims.** SHIELD-RT configures the published AUC of 0.81 on a Beta(2, 7) population. That population's ceiling is about 0.70, so the configuration asks for more than the population can support.

### 7.4 Method: converting risk to per-step probability

Annual risk is converted under a constant-hazard assumption:

$$h_i = -\ln(1 - r_i), \qquad p_{i,t} = 1 - e^{-h_i \Delta}$$

Composing the step probability back over a year recovers the annual risk exactly: $1 - (1-p_{i,t})^{1/\Delta} = r_i$.

Worked example, annual risk 5%, weekly steps ($\Delta = 1/52$):

| Quantity | Value |
|---|---|
| Hazard $h$ | 0.05129 per year |
| Weekly probability $1 - e^{-h/52}$ | 0.000986 |
| Naive weekly probability $0.05/52$ | 0.000962 (2.5% too low) |
| Weekly probability composed over 52 weeks | 0.0500 |

Model labels use the same conversion with $\Delta$ equal to the label window. Stroke, for example, predicts strokes within a four-week window, so its labels use $\Delta = 4/52$.

### 7.5 Method: temporal risk variation (AR(1))

Each entity's risk is scaled by a modifier $X_{i,t}$ that follows

$$X_{i,t} = \rho\, X_{i,t-1} + (1-\rho)\,\mu_t + \varepsilon_{i,t}, \qquad \varepsilon_{i,t} \sim N(0, \sigma^2)$$

clipped to $[0.5, 2.0]$. The current risk is $r_i \cdot X_{i,t}$ multiplied by any demographic or treatment factors.

Without clipping and with constant $\mu$, the stationary mean is $\mu$, the stationary standard deviation is $\sigma / \sqrt{1-\rho^2}$, and the lag-1 autocorrelation is $\rho$.

Simulated check, 20,000 entities, 200 burn-in steps then 200 measured steps:

| $\rho$ | $\sigma$ | Theoretical SD | Simulated SD | Simulated lag-1 autocorrelation | Share of values at a bound |
|---|---|---|---|---|---|
| 0.90 | 0.10 | 0.229 | 0.223 | 0.881 | 0.90% |
| 0.95 | 0.05 | 0.160 | 0.160 | 0.931 | 0.05% |

Clipping trims the SD slightly at $\rho = 0.9$. The simulated autocorrelations sit below $\rho$ because sample autocorrelation over a 200-step series is biased low by roughly $(1 + 3\rho)/T$: 0.019 at both settings, which accounts for the gap.

### 7.6 Seasonality: the parameter is not the realized amplitude

Stroke (and `AR1Process.step_with_season`) adds seasonality by moving the AR(1) target: $\mu_t = 1 + A\cos(2\pi t / P)$. The AR(1) process smooths this forcing. In steady state, the modifier's seasonal swing has amplitude

$$A \cdot \frac{1-\rho}{\left|1 - \rho\, e^{-i\omega}\right|}, \qquad \omega = 2\pi / P$$

and lags the forcing by $\arg(1 - \rho e^{-i\omega}) / \omega$ timesteps.

For stroke's defaults ($A = 0.1$, $\rho = 0.95$, $P = 52$ weeks), the realized seasonal swing is **±3.9%, not ±10%**, and it peaks **9 weeks** after the configured peak. Simulation confirms both numbers. A user who sets `seasonal_amplitude=0.1` expecting ±10% gets less than half of it (I-11).

### 7.7 Outcome types

Scenarios record two different kinds of outcome in `Outcomes.events`, and they mean different things when summed over time.

| Type | Example | `events[i]` at time $t$ means | Sum over $t$ means |
|---|---|---|---|
| Incident event | Stroke: 1 if a stroke occurred this week | A new event | Number of events |
| State occupancy | Sepsis family: 1 if the patient is in sepsis or worse at $t$ | Being in a state | Patient-time spent in the state |

Effect summaries that sum `events` over time measure different things in different scenarios. The sepsis family's "events prevented" is patient-time in sepsis prevented, not cases prevented (I-10). Each scenario should document what `events` means.

### 7.8 Verification

The bulletproof statistical-sanity tests check beta means within 4-sigma CLT bounds, variance ordering by concentration, hazard round-trips, and AR(1) stationary mean, variance and autocorrelation. No test checks the seasonal response.

### 7.9 Implementation

`population/risk_distributions.py` (`beta_distributed_risks`); `population/temporal_dynamics.py` (`annual_risk_to_hazard`, `hazard_to_timestep_probability`, `AR1Process`); AR(1) state rows in `scenarios/stroke_prevention/scenario.py` and `scenarios/sepsis_early_alert/scenario.py`.

### 7.10 Open issues

I-10, I-11, I-12.

---

## 8. A6. Controlled ML model simulation

### 8.1 Need

Pre-deployment questions take the form "what if the model achieves AUC 0.82 here?" Training a real model would answer a different question. The SDK instead generates scores whose performance matches a specification. That specification must be achievable on the synthetic population, and the scores must behave like a real model's scores: calibrated, and with no knowledge of the outcome.

### 8.2 How the metrics are linked

At prevalence $p$:

$$\text{PPV} = \frac{\text{sens}\cdot p}{\text{sens}\cdot p + \text{FPR}\cdot(1-p)} \qquad \text{(Bayes' theorem)}$$

The SDK models the ROC curve as equal-variance binormal:

$$\text{sens} = \Phi\!\left(\Phi^{-1}(\text{FPR}) + d'\right), \qquad d' = \sqrt{2}\;\Phi^{-1}(\text{AUC})$$

Holding PPV fixed defines a straight line through the origin of ROC space:

$$\text{FPR} = k\cdot\text{sens}, \qquad k = \frac{p\,(1-\text{PPV})}{\text{PPV}\,(1-p)}$$

The curve crosses that line once. Consequence: **at fixed prevalence, any two of AUC, PPV and sensitivity determine the third.** Specifying all three overdetermines the model.

PPV plus sensitivity gives the AUC required, in closed form:

$$\text{FPR} = k \cdot \text{sens}, \qquad \text{AUC}_{\text{req}} = \Phi\!\left(\frac{\Phi^{-1}(\text{sens}) - \Phi^{-1}(\text{FPR})}{\sqrt{2}}\right)$$

A PPV at or below prevalence describes a model worse than random and is rejected.

### 8.3 Worked examples from the SDK's functions

| Prevalence | Given | Result |
|---|---|---|
| 5% | AUC 0.80, PPV 0.20 | Sensitivity 0.453 |
| 5% | AUC 0.80, sensitivity 0.50 | PPV 0.184 |
| 13% | PPV 0.25, sensitivity 0.80 | AUC needed 0.803 |
| 5% | PPV 0.30, sensitivity 0.70 | AUC needed 0.909 |
| 0.55% (stroke, 4-week window) | PPV 0.15, sensitivity 0.80 | AUC needed **0.976** |

The binormal model is an approximation. The code documents its error on the SDK's beta populations: sensitivity error 0.004–0.011 at AUC ≥ 0.83, rising to about 0.09 near AUC 0.68. That is accurate enough to judge feasibility. The fitted model is always judged on its actual output.

### 8.4 The Bayes ceiling

Outcomes are random draws from each entity's risk. Even an oracle that scores every entity by its true risk cannot rank outcomes perfectly. A patient at 2% risk sometimes has the event, and a patient at 30% risk usually does not. The oracle's AUC, true risk scored against realized outcomes, is the population's **Bayes ceiling**. Within the simulation, true risk is all there is to know before the draw, so no model without access to the outcome can exceed it.

The ceiling is estimated from a finite sample. Its standard error follows Hanley and McNeil (1982):

$$\text{SE}^2 = \frac{A(1-A) + (n_1 - 1)(Q_1 - A^2) + (n_0 - 1)(Q_2 - A^2)}{n_1 n_0}, \quad Q_1 = \frac{A}{2-A}, \quad Q_2 = \frac{2A^2}{1+A}$$

where $A$ is the estimated ceiling and $n_1$, $n_0$ are the numbers of positives and negatives.

The ceiling depends mainly on risk spread (Section 7.3) and only weakly on event rate.

### 8.5 Score generator

For entity $i$ with true risk $r_i$, correlation $c$, noise scale $k$, and standard normal draws $z_{1i}$, $z_{2i}$:

$$x_i = c\cdot r_i + (1-c)\,k\,z_{1i} + 0.1\,k\,z_{2i}, \qquad s_i^{\text{raw}} = \sigma\big(4(x_i - 0.5)\big)$$

The generator sees only $r_i$. It never sees the outcome. `predict()` accepts `true_labels` for backward compatibility and ignores them, so scores are identical with or without labels.

On `main`, the generator added a label-dependent shift during fitting. `predict()` dropped it when called without outcomes, which is how 10 of 12 scenarios call it, and kept it when called with them. PR #5 removed the shift; Section 8.10 gives the before/after numbers.

### 8.6 Fitting procedure

```
fit(labels, risks, rng):
  1. ceiling ← exact rank AUC of (labels, risks); n₁ ← number of positives
  2. if labels are all one class: warn, keep defaults, return
  3. AUC_req ← target_auc                       (discrimination, threshold_ppv)
             ← solve(binding, goal)  or target   (operating_point)
             ← AUC implied by (PPV, sens)        (classification)
  4. ceiling gate:
       if AUC_req > ceiling + max(tol, 2·SE):
         if n₁ < 30: warn (estimate too noisy to judge)
         elif not allow_infeasible: raise OperatingPointError
  5. draws ← max(n_iterations, 5) sets of (z₁, z₂), reused for every candidate
  6. grid search over c ∈ [0.30, 0.99] × k ∈ [0.01, 0.70] (15 × 15):
       minimize |mean AUC over draws − AUC_req|
  7. bisection on k within ± one grid step of the best k (12 iterations, if bracketed)
  8. Platt scaling on one fresh draw: P(y=1|s) = σ(a·logit(s) + b), maximum likelihood
  9. threshold:  operating_point → exact cut on the binding metric (calibrated scale)
                 classification  → exact cut on PPV (calibrated scale)
                 otherwise       → operating_threshold or 0.5
 10. report all metrics on another fresh draw (what predict() will deliver)
 11. gates: operating_point raises on a delivered miss; other modes warn
```

Using the same draws for every candidate in step 5 (common random numbers, again) means candidates differ only through their parameters, so the search compares signal rather than noise.

### 8.7 Modes

| Mode | You set | The model derives | Use |
|---|---|---|---|
| `discrimination` | AUC | Nothing about the threshold. The scenario picks its own cut (percentile, top-K). | Ranking and capacity-limited targeting. Most scenarios. |
| `operating_point` | One binding metric (PPV or sensitivity) and either AUC or a goal for the other metric | Threshold placed exactly on the binding metric | Alert systems where a threshold matters. Recommended. |
| `classification` (legacy) | PPV and sensitivity | The AUC they imply; threshold placed on PPV | Equivalent to operating_point with a PPV binding |
| `threshold_ppv` (legacy) | AUC, PPV and a fixed threshold | Nothing | Overdetermined. AUC is enforced; PPV is reported, not enforced. |

### 8.8 Gates

| Gate | Condition | Action |
|---|---|---|
| Ceiling | Required AUC > ceiling + max(0.02, 2·SE) | Raise; warn instead if fewer than 30 positives, or if `allow_infeasible` |
| Delivered metrics | Delivered AUC or binding/goal metric misses tolerance (both directions) | operating_point raises; other modes warn |
| Degenerate PPV | PPV ≤ prevalence | Raise |
| Single-class population | No positives or no negatives | Warn, keep defaults, calibration at identity |

### 8.9 Assumptions and limits

- **Scores depend on current risk only.** They have no feature-level structure, no subgroup-specific accuracy, and no knowledge of time to onset. The malnutrition scenario carries its own subgroup-differential scorer. TREWS-style "detects before clinical suspicion" timing is not representable.
- **Models are fit on the population at the first prediction call.** In the sepsis family that population has 9–42 positive patients, because labels mark who is septic at $t = 0$. A configured AUC cannot be verified on so few events. The gate warns instead of refusing (I-21).
- **Calibration targets the fit population.** If prevalence drifts during a run, the scores stay calibrated to the original population.
- **Arguments a mode does not use are silently ignored** (I-18).
- **A refit on the same model object reuses the first fit's prevalence** (I-18).

### 8.10 Worked example: configured versus delivered AUC

5,000 patients, 5% prevalence, Bayes ceiling 0.91. The model is fit on one population, then scored on an independent one through `predict()` with no labels. Mean of three seeds.

| Configured AUC | `main`, delivered | `main`, outcomes passed in | Branch, delivered | Calibration slope, `main` → branch |
|---|---|---|---|---|
| 0.630 | 0.670 | 0.693 | 0.611 | 0.82 → 0.90 |
| 0.720 | 0.685 | 0.705 | 0.702 | 0.84 → 0.94 |
| 0.760 | 0.717 | 0.740 | 0.747 | 0.86 → 0.97 |
| 0.780 | 0.711 | 0.763 | 0.768 | 0.77 → 0.97 |
| 0.810 | 0.768 | 0.794 | 0.803 | 0.87 → 0.99 |
| 0.830 | 0.771 | 0.823 | 0.826 | 0.82 → 0.99 |
| 0.845 | 0.757 | 0.835 | 0.847 | 0.77 → 1.00 |

The branch's largest miss is 0.019. On `main`, the fit report agreed with the target to within 0.01 in six of seven rows. (At 0.63 the old search could not reach the target at all.) That agreement is why nobody noticed: the report described a model `predict()` never ran.

### 8.11 Verification

`tests/unit/test_label_free_fidelity.py` (27 tests) scores fresh populations through `predict()` and never relies on the fit report. 21 of the 27 fail on `main`. `tests/unit/test_operating_point.py` covers the over- and under-determined configurations and the gate messages.

### 8.12 Implementation

`ml/model.py` (`ControlledMLModel`); `ml/performance.py` (`auc_exact`, `required_auc_for_operating_point`, `sensitivity_at_ppv`, `ppv_at_sensitivity`, `validate_operating_point`); `ml/probability_model.py` (`ControlledProbabilityModel`, same principles in log-odds space).

### 8.13 Open issues

I-18, I-21.

---

## 9. A7. Intervention impact

### 9.1 Need

A flag changes nothing by itself. Between the model's score and a changed outcome sit a threshold, a human who may or may not respond, finite staff capacity, a treatment whose benefit depends on timing, and standard care that would have caught some cases anyway. Leaving any of these out overstates benefit. `design_principles.md` records capacity, not AUC, as the binding constraint in the TREWS sweep.

### 9.2 The causal chain

For entity $i$ at a prediction time $t$:

```
score s_i  →  flagged?  →  response?  →  within capacity?  →  treated  →  effect e(τ)  →  hazard reduced
```

| Link | General form | Example |
|---|---|---|
| Flag | $s_i \ge$ threshold; or top percentile among active entities; or top-K | Sepsis: top 10% of active patients |
| Response | Bernoulli with rate $r_t$ | Sepsis: $r_t = \max\!\left(r_{\text{floor}},\ r_0\, e^{-\phi F_t}\right)$, $F_t$ = system-wide cumulative false alerts |
| Capacity | At most $C$ treated per step, highest score first | Sepsis: rapid response capacity per 4-hour block |
| Effect | $h' = h\,(1 - e)$ on the treated entity's hazard or transition probability | Stroke: persistent risk multiplier |
| Effect over time | Permanent, decaying, or timing-dependent | Nurse retention: $e \cdot 2^{-\tau/T_{1/2}}$ per week since check-in. Sepsis (Kumar): $e_{\max}\cdot 0.5^{d / T_{1/2}}$, where $d$ is hours from onset to treatment; treatment before onset gets $e_{\max}$. |
| Standard care | Runs inside `step()` on both branches | Sepsis baseline detection delay drawn per patient; nurse retention standard-of-care check-ins. The effective benefit is $\max(e_{\text{AI}}, e_{\text{SOC}})$. |

### 9.3 Ground truth from the chain

Because every link is explicit, the expected effect is known exactly from the two branches' probabilities (Section 5.7): $\sum_{i,t} (p^C_{i,t} - p^F_{i,t})$. For screening before any simulation, the chain collapses to a product:

$$\text{events prevented} \approx N \cdot p \cdot \text{sens} \cdot r \cdot \kappa \cdot e$$

where $\kappa$ is the share of responded flags that fit within capacity. `docs/analytical_first_pass.md` describes when this product is enough and when the full simulation is needed. Simulation is needed whenever timing, fatigue, capacity queues or score-dependent efficacy matter.

### 9.4 Interference between entities

Capacity makes entities interfere. Treating one patient can use the slot another would have received. Alert fatigue depends on system-wide false alerts. Per-entity effects are therefore not independent, and the stable-unit assumption behind entity-level estimators (for example an entity-level DiD) does not hold. Effects should be estimated at the level where the intervention operates: the unit, the clinic, the time block.

### 9.5 Interventions that do not change risk

Overbooking changes the schedule, not any patient's risk. Its outcomes are operational: collisions, wait time, overtime. The same engine applies; the estimand changes (Section 10).

### 9.6 Assumptions and limits

- Effect size $e$ is an input, usually from literature. The simulation propagates it; it does not validate it.
- Threshold semantics must match the score scale. Scores from `ControlledMLModel` are calibrated probabilities. MASAI uses a fixed cut of 0.70, written to mean "AI score above 7 of 10", but its calibrated scores peak near 0.63, so nobody is routed (I-23).
- A scenario whose outcomes do not depend on the model's scores cannot test the model. Manz decides outcomes from true risk (I-23).

### 9.7 Verification

Monotonicity tests in `tests/bulletproof/test_conservation_laws.py` check that higher effectiveness gives fewer events and that a higher threshold gives fewer treated. Zero-effect identity is checked in some scenarios. Capacity is checked in the sepsis conservation tests.

### 9.8 Implementation

Each scenario's `intervene()` and `step()`. Generic pieces worth extracting are listed in `design_principles.md` Appendix B: baseline care, Kumar decay, a capacity helper.

### 9.9 Open issues

I-01, I-14, I-23.

---

## 10. A8. Outcomes, estimands and uncertainty

### 10.1 Need

A result is only interpretable if three things are stated: what was measured, what effect was estimated, and how uncertain it is. Without them, numbers from different scenarios cannot be compared, and a single run looks more certain than it is.

### 10.2 Estimands

Sign convention: benefit is positive, $C - F$. This is the convention the malnutrition scenario uses and states in every caption.

| Estimand | Definition | Needs coupling? |
|---|---|---|
| Total effect | $\sum_{i,t} (Y^C_{i,t} - Y^F_{i,t})$ | No (coupling only reduces its variance) |
| Average effect per entity | Total effect $/ N$ | No |
| Effect on the treated (ATT) | Mean of $\sum_t (Y^C_{i,t} - Y^F_{i,t})$ over entities ever treated | No |
| Per-entity contrast ("rescued", "displaced") | $Y^C_i - Y^F_i$ for individual entities | **Yes.** Without coupling the contrast is mostly noise. |
| Expected effect | $\sum_{i,t} (p^C_{i,t} - p^F_{i,t})$ | No. Computable only inside the simulation. |

Every analysis should state which estimand it reports, what `events` means in the scenario (Section 7.7), and the sign convention.

### 10.3 Two sources of uncertainty

1. **Outcome noise within a population.** Which events happen, given everyone's risk. Coupling reduces it (Section 5.7).
2. **Population variation between seeds.** Which patients the synthetic site happens to have, and how many of them are high-risk. Coupling cannot reduce it. It represents real uncertainty about a site like this one.

From the stroke example:

| Variation | Standard deviation of events prevented |
|---|---|
| Within one population, coupled (200 outcome replicates) | 9.6 |
| Between 30 seeds, coupled (population and outcomes both vary) | 20.6 |

Most of the uncertainty about one site's result comes from which patients it has, not from outcome noise.

**Reporting rule.** Run at least 30 seeds. Report the mean with a confidence interval, $\bar{x} \pm t_{0.975,S-1}\, s/\sqrt{S}$, for "the average site like this". Report the spread, $\bar{x} \pm t_{0.975,S-1}\, s$, for "what one deployment might see". These answer different questions, and the second is usually the one a decision needs.

### 10.4 Analysis exports

`SimulationResults.to_analysis()` returns an `AnalysisDataset` with four exports:

| Export | Shape | Caveat |
|---|---|---|
| `to_time_series()` | Per-step totals of `events` | `treatment_indicator` is 1 at every step with an `intervene()` call, not a post-period indicator. Build the post indicator yourself. |
| `to_panel()` | Entity × step long format | `treated` = ever treated **and** $t \ge$ the first intervention time anywhere in the run. An entity first treated in week 30 is marked treated from the run's first intervention time (week 0 in stroke), not from week 30 (I-09). |
| `to_entity_snapshots(t)` | Cross-section at $t$ with scores | For RDD around a threshold |
| `to_subgroup_panel()` | Panel with group labels | Labels from `secondary` or `metadata` |

All four, and `get_outcome_series()`, accept an `outcome_key` argument that is never used. They always return `events`. A request for `"mortality"` silently returns `events` (I-08).

### 10.5 Open issues

I-08, I-09, I-10.

---

## 11. A9. Evaluation design by simulation

### 11.1 Need

A real deployment has no counterfactual branch. It is evaluated with quasi-experimental methods: interrupted time series, controlled ITS, difference-in-differences, regression discontinuity, stepped wedge. Each can be biased or underpowered in a given setting. The SDK's most valuable use is to test the planned method on synthetic data where the true effect is known, before committing to it.

### 11.2 The rule that makes this valid: oracle versus observable

| Data | Available in a real deployment? | Role in a design study |
|---|---|---|
| Factual branch outcomes | Yes | Input to the estimator |
| Real comparison units (other sites, wards, time periods) | Sometimes | Input to the estimator |
| Counterfactual branch | **Never** | Ground truth for grading the estimator |
| True risks, true labels, per-entity effects | **Never** | Ground truth and diagnostics |

An estimator that uses the counterfactual branch as its control (`cits_with_control` in its oracle configuration, nurse-retention Tier 3) is a benchmark, not a deployable method. Its performance says nothing about what a real evaluation would achieve.

### 11.3 Procedure

1. **Define the estimand and its truth.** For example: average weekly events prevented in the post period, computed from $C - F$.
2. **Simulate the real deployment schedule.** A pre-period without the AI, then switch-on or a staggered rollout. Use the prediction schedule to control when the AI starts.
3. **Generate $S$ seeds under the alternative** (the planned effectiveness) **and $S$ under the null** (effectiveness 0).
4. **Apply the candidate estimator to observable data only.**
5. **Grade it:**
   - bias = mean(estimate − truth);
   - RMSE;
   - confidence-interval coverage;
   - power = share of alternative seeds with $p < \alpha$ in the expected direction;
   - type I error = share of null seeds with $p < \alpha$.
6. **Sweep the design knobs:** population size, horizon, switch-on time, aggregation level, number of sites.
7. **Choose the smallest design that meets power and type I targets.** Report the design and its operating characteristics.

Sweeps use Hydra YAML grids and `--multirun`, and runners call `finalize_experiment()`, per `CLAUDE.md`.

### 11.4 Estimators available

| Estimator | Code | Notes |
|---|---|---|
| Segmented regression ITS: $Y_t = \beta_0 + \beta_1 t + \beta_2\,\text{post}_t + \beta_3 (t - T^*)\,\text{post}_t + \varepsilon_t$ | `experiments/analysis/its.py` `segmented_regression` | Plain OLS by default; empirical type I 0.0505 at nominal 0.05 on 60-point iid series. Newey–West HAC is anti-conservative in small samples (type I ≈ 0.095 at lag 1, 0.163 at lag 4). |
| Power across seeds | `power_across_seeds` | Counts $p < \alpha$ **and** the expected sign |
| Controlled ITS, stacked groups: $Y = b_0 + b_1 t + b_2 g + b_3 g t$ | `cits_with_control` | With a real control series: deployable. With the counterfactual branch as control: oracle only. Its OLS standard error treats the two series as independent. Coupled branches are positively correlated, so the standard error is conservative in that configuration (I-15). |
| Slope only | `its_slope_only` | Floor estimator when there is no pre-period or control |
| Difference-in-differences, RDD | Use `to_panel()`, `to_entity_snapshots()` with your own model | Mind the `to_panel` treated definition (I-09) and interference (Section 9.4) |
| Stepped wedge (closed form, Hussey and Hughes 2007) | malnutrition `power_map.py` | A single site cannot identify a stepped wedge; it reduces to ITS |

### 11.5 Worked example: is a one-site, one-year ITS adequate?

Stroke scenario. 52 weekly steps. The AI switches on at week 26 (prediction schedule 26, 30, …, 50). Segmented regression on the factual weekly event counts, break at week 26. 40 seeds per cell. The truth is the mean post-period weekly events prevented, $C - F$, from the counterfactual branch. The ITS estimate of the same quantity is $-\beta_2$, the drop in level at the break.

| Population | Effectiveness | True events prevented per week | ITS estimate, mean $-\beta_2$ | Share with $p < 0.05$ for $\beta_2$ and $\beta_2 < 0$ |
|---|---|---|---|---|
| 5,000 | 0.5 | 1.66 | 1.43 | **22.5%** (power) |
| 5,000 | 0 | 0 | −0.17 | 2.5% (type I, one-sided) |
| 20,000 | 0.5 | 6.39 | 7.04 | **80.0%** (power) |
| 20,000 | 0 | 0 | 1.19 | 0.0% (type I, one-sided) |

Reading it: at 5,000 patients this design would miss a real effect about four times in five. Reaching 80% power takes about 20,000 patients over one year. Type I error is in range. With 40 seeds, a rate near 80% carries a standard error of about 6 percentage points, so these are design-level answers, not precise ones.

**Bias check against the oracle.** Power and type I error are not enough. The same 40 seeds were run with the scenario's seasonality switched off. Because the seeds are paired, the difference between the two rows of each population isolates the effect of seasonality.

| Population | Seasonality | Bias under the null (SE) | Bias under the alternative (SE) |
|---|---|---|---|
| 5,000 | On ($A = 0.1$) | −0.17 (0.22) | −0.23 (0.20) |
| 5,000 | Off | −0.41 (0.22) | −0.45 (0.20) |
| 20,000 | On ($A = 0.1$) | **+1.19 (0.29)** | +0.65 (0.31) |
| 20,000 | Off | +0.28 (0.30) | −0.11 (0.30) |

Bias is the ITS estimate of events prevented per week minus the oracle truth.

Seasonality adds about +0.24 events per week of spurious benefit at 5,000 patients and +0.91 at 20,000; it grows in proportion to population. At 20,000 the null-run bias is 4.1 standard errors from zero, about a fifth of the true effect in the alternative. At 5,000 it is hidden by noise. The −0.41 offsets at 5,000 are shared seed noise, since they appear with and without seasonality.

This happens even though the realized seasonal swing is only ±3.9% (Section 7.6). Type I error stayed near zero because the OLS standard error absorbs the seasonal misfit. A design study that checked only power and type I error would have approved a biased estimator. Checking bias against the oracle caught it. In a real evaluation the remedies are a seasonal term in the model, a pre-period covering a full cycle, or a concurrent control series.

### 11.6 Relationship to the analytical-first pass

If the design question depends only on point estimates (expected effect, variance of a weekly count), closed-form power is enough and faster. `docs/analytical_first_pass.md` gives the protocol. The simulation is needed when the effect ramps up as treated patients accumulate, when counts are small or autocorrelated, or when the deployment has staggered timing or interference.

---

## 12. A10. Monitoring design: control charts

### 12.1 Need

After deployment, a monitoring plan should detect when the program stops working, without raising so many false alarms that people stop looking. The two properties trade off. A simulation with known regimes (working, null, decaying, capacity collapse) can measure both properties before the plan is adopted.

### 12.2 Methods implemented (nurse retention, `monitoring/`)

| Tier | Method | Statistic and rule |
|---|---|---|
| 1 | Shewhart individuals and moving range (XmR) | $\hat\sigma = \overline{MR}/1.128$; limits $\bar{x} \pm 3\hat\sigma$; WE1 = one point outside the limits; WE2 = 8 consecutive points on one side of the centre line |
| 2 | Two-sided CUSUM (Page 1954) | $C^+_t = \max(0,\ C^+_{t-1} + x_t - \mu_0 - K)$; $C^-_t = \max(0,\ C^-_{t-1} - x_t + \mu_0 - K)$; alarm when either exceeds $H$. $K = 0.5\hat\sigma$, $H = 4\hat\sigma$; $\mu_0$, $\hat\sigma$ from the first 8 weeks. The arm that fires is reset. |
| 3 | Rolling controlled ITS | `cits_with_control` refit quarterly; oracle mode uses the counterfactual branch |
| 4 | Model drift | Rolling 12-week AUC, calibration slope, Brier score; alarm on AUC < 0.65 for 4 weeks or a calibration slope 20% off its initial value |

### 12.3 The key property: in-control run length

A chart's false alarm behavior is summarized by ARL₀, the average number of steps until a false alarm when nothing has changed. Textbook values assume the in-control mean and standard deviation are known and observations are independent. Neither holds here: limits come from 8 baseline weeks, and weekly turnover inherits autocorrelation from the AR(1) risk modifiers.

Simulated two-sided CUSUM, $k = 0.5$, $h = 4$, 2,000 replicates:

| In-control process | Median run length | Mean run length | Probability of a false alarm within 52 weeks |
|---|---|---|---|
| Independent, parameters known | 124 | 170 | 26% |
| Independent, $\mu_0$ and $\sigma$ estimated from 8 points | 32 | 212 | 60% |
| AR(1) with $\rho = 0.5$, estimated from 8 points | 10 | 23 | **92%** |

The first row matches the textbook ARL₀ of about 168 for this design (Montgomery). The other rows show why estimated limits and autocorrelation matter. The mean in row 2 is high because a few replicates with an overestimated $\sigma$ alarm very late, or not within the 2,000-step horizon.

### 12.4 What this means for the monitoring validation report

`reports/monitoring_validation/monitoring_validation.md` reports Tier 2 firing in **30 of 30 null-regime seeds, at a median of week 20**. That is consistent with the table above. In the null regime the AI has no effect, so these are false alarms. The report attributes Tier 2's firing to "the beneficial effect of the AI", which cannot apply there (I-04).

The report's headline "0% false positive rate in the null regime" refers to Tier 3 only. In the null regime the factual and counterfactual series are bit-identical, so an estimator that compares them cannot fire. That silence is guaranteed by construction and says nothing about specificity in practice (I-05).

Tier 4 scores the model against labels drawn at prediction time from the prediction stream. Those draws are independent of the departures that later happen. A deployed monitor can only score against realized outcomes. Realized outcomes are suppressed among the high-scoring nurses the program treated, so a deployed AUC drifts downward when the program works, even if the model has not changed. The validation used a label production cannot see and did not model this effect (I-06). The report also predates PR #5: nurse retention passed outcomes into `predict()`, and its configured AUC of 0.80 sits above the population ceiling of about 0.71 (I-07).

### 12.5 Procedure for choosing chart parameters

1. Define regimes: null, working, the failure modes to catch.
2. For each candidate (chart type, $k$, $h$, baseline length, aggregation), simulate $S$ seeds per regime.
3. Compute in-control run length on the null regime and detection delay on each failure regime, **using observable data only**.
4. Pick $h$ to meet a target in-control run length (for example, one false alarm per two years), then compare detection delay.
5. For autocorrelated data, chart residuals from a fitted AR model or use a longer baseline. Do not rely on textbook constants.

### 12.6 Open issues

I-04, I-05, I-06, I-07, I-20.

---

## 13. A11. Verification pillars

### 13.1 Need

A misconfigured simulation that looks plausible is worse than none, because it produces false confidence. Verification must show that the code does what the scenario claims before anyone interprets a result. Every test must be able to fail.

### 13.2 The five levels

| Level | Checks | Catches | Automated today |
|---|---|---|---|
| 1. Structural integrity | No NaN/Inf; population conserved; scores in [0,1]; TP+FP+TN+FN = N; entity IDs set | Indexing and bookkeeping bugs | Stroke, sepsis, malnutrition, reminders tests |
| 2. Statistical sanity | Event rates within 4σ of configuration; **delivered** model metrics within tolerance on an independent population; AR(1) moments; demographic shares | Mis-parameterization; models that do not deliver what they claim | Partial. The model check on `main` was circular (Case 1). |
| 3. Conservation and monotonicity | Zero effect ⇒ factual ≡ counterfactual; higher effectiveness ⇒ fewer events; higher threshold ⇒ fewer flags; delivered PPV within Bayes bounds; **delivered AUC ≤ population ceiling**; **branch coupling holds** | Wiring errors; impossible results | Partial; the last two are new |
| 4. Case walkthroughs | 3–5 entities traced in narrative form: a treated high-risk entity, an untreated low-risk one, its counterfactual twin, one near the threshold | Errors that aggregates hide | Malnutrition only |
| 5. Boundary conditions | Threshold 0 and 1; AUC ≈ 0.5; effectiveness 0 and 1 | Edge-case logic | Malnutrition stage B; some bulletproof tests |

### 13.3 Rules for tests people can trust

1. **Test the deployed path.** Score what the scenario actually runs, on data the fit never saw.
2. **Every check must be able to fail.** A check like `x >= 0` on an absolute value, or one that sets a total to N and then asserts it equals N, verifies nothing. The Epic ESM runner has both (I-16).
3. **Fix tolerances before running.** Tolerances chosen after seeing results are not checks.
4. **Gate interpretation on verification.** `finalize_experiment()` saves and registers results without running any checks. Malnutrition's stage-B gate is the model to copy.
5. **Run the suite automatically.** There is no CI today. The pre-commit hooks check structure, not tests.

### 13.4 Case 1: the label leak

**What happened.** On `main`, the statistical-sanity test fit the model, then compared the fit report's AUC with the target at a tolerance of 0.08. The fit report was computed on scores that included the outcome shift, so it always agreed. The test passed while the deployed model missed its target by up to 0.09 AUC.

**What would have caught it:**

- **Level 2, done right.** Score an independent population through `predict()` without labels. The new tests do this; 21 of 27 fail on `main`.
- **Level 3, the ceiling.** In stroke and nurse retention, delivered AUC exceeded the population's Bayes ceiling (0.87 against 0.78 in a controlled test). That is impossible for a model without the outcome, so a conservation check fails immediately.

### 13.5 Case 2: the coupling audit

**What happened.** The BRANCHED-versus-NONE identity test passed for nurse retention and no-show overbooking, and both looked healthy. The stream-position audit (Section 5.6) shows their branches decoupling at week 4 and day 1. The identity test checks a different property (Section 5.6). Effects stay unbiased, but they are noisier than reported, and per-entity comparisons in those scenarios are not paired.

**The new check.** After each step, assert that both branches' temporal and outcomes generators are at the same position. This takes a few lines in a test helper and runs in seconds.

### 13.6 Case 3: the null regime

A monitoring tier that fires in every null seed is a false-alarm generator (Section 12.4). A tier that cannot fire in the null regime, because it compares two identical series, has not been tested. Null regimes must be run on observable data with realistic noise.

---

## 14. Issues register

Severity: **High** means a reported result or conclusion is affected. **Medium** means results are weaker or a likely future error is unguarded. **Low** means a documentation or interface problem.

| ID | Area | Finding | Evidence | Severity | Proposed fix |
|---|---|---|---|---|---|
| I-01 | Coupling | Nurse retention and no-show overbooking violate draw-count invariance; their branches decouple at week 4 and day 1 | Stream-position audit (§5.6). Nurse retention: `rng.permutation` on state-dependent team sizes in SOC allocation. Overbooking: extra draws for overbooked slots in `step()` | High | Draw fixed-size vectors and select by mask; order by `argsort` of $n$ uniforms; per-entity coins drawn at creation |
| I-02 | Testing | The purity identity test proves non-interference, not coupling | §5.6 | Medium | Add the stream-position audit and an untreated-entity identity test to the scenario test template |
| I-03 | Engine | Re-running a scenario object decouples its branches and changes its factual results | 118 discrepancies at zero effect on the second run (§3.8) | Medium | Reset `self.rng` from the partitioner at the start of `run()`, or make scenarios single-use and raise on reuse |
| I-04 | Monitoring | Tier 2 CUSUM alarms in 30/30 null seeds (median week 20); limits estimated from 8 weeks of autocorrelated data | Validation report table; §12.3 simulation | High | Calibrate $h$ by simulation for a target in-control run length; longer baseline; residual charting |
| I-05 | Monitoring | Tier 3's null-regime "zero false positives" is guaranteed by construction (identical series) | Null regime ⇒ factual ≡ counterfactual | High | Validate Tier 3 with a real control series and observable noise; label oracle mode as a benchmark |
| I-06 | Monitoring | Tier 4 scores against labels a deployment cannot observe; post-deployment AUC drop from the program's own effect is not modeled | `nurse_retention/scenario.py` `predict()` draws labels from the prediction stream | High | Score against realized factual outcomes; quantify the performative drop using the counterfactual branch |
| I-07 | Reports | Monitoring validation and other pre-fix reports were produced with the outcome leak; nurse retention's AUC 0.80 exceeds its ~0.71 ceiling | PR #5 reruns | High | Re-run and revise all reports after PR #5 merges |
| I-08 | Analysis API | `outcome_key` is accepted and ignored by every export; requests for other outcomes silently return `events` | `core/results.py` | Medium | Implement lookup in `secondary`, or raise when a key other than `"events"` is passed |
| I-09 | Analysis API | `to_panel()` marks ever-treated entities as treated from the first intervention anywhere, not from their own first treatment | `core/results.py` lines 205–211 | Medium | Use each entity's first treatment time; document both variants |
| I-10 | Outcomes | Sepsis-family `events` are state occupancy; summed effects are patient-time, not cases | `sepsis_early_alert/scenario.py` `measure()` | Medium | Document per scenario; add an incident-onset outcome in `secondary` |
| I-11 | Population | `seasonal_amplitude` sets the AR(1) target, not risk; stroke's 0.1 gives ±3.9% with a 9-week lag | §7.6, analytic and simulated | Medium | Document the transfer; or apply seasonality multiplicatively to risk |
| I-12 | Purity | `AR1Process` keeps state on the object; using it from `self` inside `step()` would break purity | `temporal_dynamics.py` | Low | Make it functional (state in, state out) or deprecate |
| I-13 | Engine | SNAPSHOT mode's counterfactual outcomes are not coupled and cover prediction steps only | `engine.py` `_run_snapshot` | Low | Document; or retire SNAPSHOT |
| I-14 | State | Sepsis-family fatigue counter and model caches live on `self` and persist across runs | `self._cumulative_false_alerts` | Low | Move into state, or reset at run start |
| I-15 | Estimators | `cits_with_control` treats coupled branches as independent (conservative SE) and is oracle-only in that use | `its.py` | Medium | Paired difference-series estimator for oracle use; documentation |
| I-16 | Verification | Epic ESM runner's checks cannot fail (`total_accounted = n` then `== n`; `abs(x) >= 0`); boundary checks print and assert nothing | `paper01_epic_esm/run_evaluation.py` lines 136–173 | Medium | Replace with real assertions; fail the run on violation |
| I-17 | Invariants | Purity and draw-count invariance are test-only; seven paper scenarios have no tests; the hooks cover less than `invariants.yaml` states (no subscript/attribute check on state, no `import random` check) | §6.5 | Medium | Add the identity and audit tests to every scenario; extend the hooks |
| I-18 | Model | Unused arguments are silently ignored in every mode; a refit on the same object reuses the first fit's prevalence | `ml/model.py` | Low | Raise on unused arguments; recompute prevalence on each fit |
| I-19 | RNG | `seed=None` means 42, not a fresh seed | `core/scenario.py` | Low | Document, or draw a seed and record it |
| I-20 | Monitoring | CUSUM docstring says both arms reset; the code resets only the arm that fired | `tier2_cusum.py` | Low | Align the docstring |
| I-21 | Model | Sepsis-family and no-show models fit at the first prediction on 9–42 events; configured AUC is unverifiable | PR #5 reruns | High | Fit on a larger reference cohort drawn from the same population |
| I-22 | Docs | `design_principles.md` presents stream-per-method plus purity as sufficient for synchronization, and purity as machine-enforced | §6.3, §6.5 | Low | Update the text after review of this document |
| I-23 | Scenarios | MASAI's 0.70 cut on calibrated scores flags nobody; Manz outcomes ignore model scores | PR #5 reruns | Medium | Percentile cut for MASAI; decide whether Manz should route outcomes through scores |
| I-24 | Reports | Monitoring report says Tier 3 power "would be ~7x higher" at 6,800 nurses; power cannot scale that way (the standard error shrinks by about √6.8 ≈ 2.6) | Report §2 | Low | Re-estimate by simulation at 6,800 |

---

## 15. Companion notebooks

Each notebook demonstrates one chapter, reproduces the chapter's worked-example numbers, and ends with checks that fail loudly if the numbers do not reproduce. Existing notebooks 01–04 and 07 are kept and extended. Notebooks 05 and 06 demonstrate legacy classes and are superseded by 12.

| Notebook | Chapter | Demonstrates | Acceptance checks |
|---|---|---|---|
| 01 (extend) | A1 | Existing content, plus the re-run hazard of §3.8 | Second-run discrepancy reproduced, then removed once I-03 is fixed |
| 02 (keep) | A2 | Mutation isolation; clone cost | Existing |
| 03 (extend) | A3 | Mode comparison, plus SNAPSHOT's uncoupled outcomes | SNAPSHOT counterfactual draws differ from factual draws for untouched entities |
| 04 (extend) | A4 | Existing purity content, plus draw-count invariance: an unsafe and a safe version of the same scenario; the stream-position audit | Unsafe version desyncs; safe version stays coupled for all steps |
| 11 (new) | A5 | Beta populations (§7.3 table); hazard conversion (§7.4); AR(1) moments (§7.5); seasonal transfer (§7.6) | Table values within Monte Carlo error; seasonal amplitude 0.039 ± 0.002, lag 9 ± 1 weeks |
| 12 (new) | A6 | Pick-two geometry; required AUC; Bayes ceiling and its SE; fit procedure; delivered vs configured (§8.10); gates firing | Delivered AUC within 0.03 of target on a fresh population; gate raises above ceiling |
| 13 (new) | A7 | Causal chain in a minimal scenario: flag → response → capacity → effect; fatigue; Kumar timing; standard care on both branches | Expected vs realized prevented events agree within 3 SD; zero effect ⇒ identity |
| 14 (new) | A3, A8 | Coupling and ground truth (§5.7): expected effect, untreated identity, variance ratio; uncertainty decomposition (§10.3) | Untreated differing = 0; variance ratio > 3 |
| 15 (new) | A9 | ITS design study (§11.5), including the seasonal bias check; extended to a staggered multi-site rollout with DiD and controlled ITS against a real control | Power, type I and bias tables reproduced; seasonal bias at 20,000 patients more than 3 SE from zero; oracle CITS flagged as non-deployable |
| 16 (new) | A10 | CUSUM run lengths (§12.3); calibrating $h$ by simulation; null and failure regimes on observable data | Known-parameter ARL₀ ≈ 168; calibrated chart meets its target in-control run length |
| 17 (new) | A11 | The three cases of §13: the leak caught by Levels 2 and 3, the coupling audit, a null regime done right | Each check fails on the defective version and passes on the corrected one |

---

## 16. Design principles this document adds

These principles are not in `design_principles.md` today. Each was needed to explain an algorithm or an issue above.

1. **Coupling is a requirement, not a side effect.** Shared streams must be consumed identically on both branches. Purity is necessary but not sufficient; draw-count invariance is the other half (§6.3).
2. **Separate oracle from observable, and grade on bias.** The counterfactual branch, true risks and true labels exist only to grade methods. Any estimator or monitor proposed for deployment must run on observable data only (§11.2). Grade it on bias against the oracle, not only on power and type I error: the ITS example in §11.5 passes both and is still biased.
3. **State the estimand before running.** Say what `events` means, which effect is reported, and the sign convention (§10.2).
4. **Report both kinds of uncertainty.** Within-population noise and between-population variation answer different questions (§10.3).
5. **Test the deployed path, with checks that can fail.** Self-reports and tautologies give false confidence (§13.3).
6. **Monitoring is performative.** A model that drives an intervention changes the outcomes it is later scored against. Monitors must account for this or they will misread success as drift (§12.4).
7. **Scenario objects are single-use** until the engine resets streams at the start of each run (§3.8).
8. **Parameters must mean what they say.** Configured values that pass through dynamics (seasonality through AR(1), annual risk through hazards, thresholds through calibration) must be documented by their realized effect, not their input value (§7.6, §9.6).

---

## 17. References

- Bernal JL, Cummins S, Gasparrini A (2017). Interrupted time series regression for the evaluation of public health interventions: a tutorial. *Int J Epidemiol* 46(1):348–355.
- Hanley JA, McNeil BJ (1982). The meaning and use of the area under a receiver operating characteristic (ROC) curve. *Radiology* 143(1):29–36.
- Hussey MA, Hughes JP (2007). Design and analysis of stepped wedge cluster randomized trials. *Contemp Clin Trials* 28(2):182–191.
- Jensen WA, Jones-Farmer LA, Champ CW, Woodall WH (2006). Effects of parameter estimation on control chart properties: a literature review. *J Qual Technol* 38(4):349–364.
- Kumar A, et al. (2006). Duration of hypotension before initiation of effective antimicrobial therapy is the critical determinant of survival in human septic shock. *Crit Care Med* 34(6):1589–1596.
- Law AM, Kelton WD. *Simulation Modeling and Analysis*. McGraw-Hill. Chapter on variance-reduction techniques (common random numbers).
- Montgomery DC. *Introduction to Statistical Quality Control*, 7th ed. Wiley. Chapters 6 and 9.
- Newey WK, West KD (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica* 55(3):703–708.
- Page ES (1954). Continuous inspection schemes. *Biometrika* 41(1/2):100–115.
- Perdomo J, Zrnic T, Mendler-Dünner C, Hardt M (2020). Performative prediction. *Proceedings of ICML*.
- Platt J (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. *Advances in Large Margin Classifiers*.
- Wagner AK, Soumerai SB, Zhang F, Ross-Degnan D (2002). Segmented regression analysis of interrupted time series studies in medication use research. *J Clin Pharm Ther* 27(4):299–309.
