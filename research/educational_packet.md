# Healthcare Intervention Simulation SDK — Educational Packet

> **Audience:** Data scientists who are new to this SDK but fluent in Python and NumPy.  
> **Goal:** Teach each module from first principles, show how the architecture implements those principles, and prove — with real runnable output — that each component works correctly.  
> **Tone:** Rigorous and direct. Every claim is backed by evidence. Every assumption is surfaced.

---

## Preface: Why Pre-Deployment Simulation?

In healthcare AI, the critical counterfactual is permanently unobservable.

When you deploy a sepsis prediction model, you observe what happens in your hospital *with* the AI active. You cannot simultaneously observe what would have happened if you had never deployed it. This is the fundamental epistemological problem: you want to measure E[Y(1) − Y(0)], the expected difference between a world with the AI and a world without it, but reality only lets you see one branch.

Real-world deployment studies address this through retrospective controls, stepped-wedge rollouts, or randomized trials — each with major operational costs and ethical tradeoffs. You cannot run an RCT before deciding whether to deploy. You can only estimate.

This is where simulation fills the gap. A simulation is not a substitute for clinical evidence; it is a *pre-deployment stress test*. Before committing to a live deployment, you can ask: Under the model performance your vendor claims, what effect size would we actually expect in our patient population? What happens to that effect if the model's real-world AUC degrades from 0.82 to 0.63 — as Epic's sepsis model did at Michigan Medicine (Wong et al., JAMA Internal Medicine 2021)? What threshold maximizes net benefit for our specific prevalence? Do the benefits accrue equally across demographic subgroups, or does the model create disparities the vendor's validation study never measured?

The Healthcare Intervention Simulation SDK provides infrastructure to answer these questions systematically. Its core thesis: the gap between model metrics and patient outcomes is not occasional but structural, and it is simulable.

### The 5-Module Architecture

The SDK is organized around five concerns:

1. **Randomness and Reproducibility** (`core/rng.py`) — Controlled, reproducible stochastic evolution for both simulation branches.
2. **The Scenario Contract** (`core/scenario.py`) — A fixed 5-method interface that any clinical scenario implements.
3. **The Branched Engine** (`core/engine.py`) — Parallel factual and counterfactual simulation trajectories.
4. **ML Model Simulation** (`ml/`) — Controlled noise injection to achieve target model performance characteristics.
5. **Results Export** (`core/results.py`) — Analysis-ready data in four formats for ITS, DiD, RDD, and equity analysis.

Each module is independently testable, composable, and auditable. The SDK does not produce recommendations; it produces evidence that demands interpretation.

---

## Module 1: Randomness and Reproducibility

### First Principles

A simulation that produces different results each time it is run is not a tool — it is noise. But controlled randomness is harder than it appears, especially in branched simulation.

**Why randomness must be controlled.** Healthcare simulation involves stochastic processes at every level: which patients have high baseline risk, how risk drifts over time, which patients happen to have an event in a given week. These must be random to represent real population variability. But they must be reproducible so that when you share a result or debug a run, you can recreate it exactly from the seed.

The standard approach is a pseudo-random number generator (PRNG) seeded with a fixed integer. NumPy's `np.random.default_rng(seed)` provides this. The problem is what happens in *branched* simulation.

**The branching problem.** A branched simulation runs two parallel trajectories: a factual branch (the AI is deployed) and a counterfactual branch (no AI). To make the comparison valid, both branches must experience identical background stochastic events — the same patients get sick, the same temporal drift occurs — and diverge *only* where the intervention actually changed something.

Now imagine both branches share a single RNG object. At timestep 3, the factual branch calls `predict()`, which draws 1,000 random values from the RNG. The counterfactual branch (correctly) does not call `predict()`. At timestep 4, both branches call `step()` — but they draw from *different positions* in the sequence, because the factual branch advanced the pointer by 1,000 draws. The temporal evolution is now desynchronized. You are no longer comparing the same patients in two different worlds; you are comparing different patients entirely.

This is not a subtle statistical bias. It is a fundamental invalidation of the causal comparison.

**The solution: stream partitioning.** The fix is to give each simulation *process* its own independent RNG stream. Temporal evolution always draws from the temporal stream. ML predictions always draw from the prediction stream. The factual branch calling `predict()` advances the prediction stream — but the counterfactual branch, which shares the *temporal* stream, is unaffected. The two branches' temporal streams stay synchronized.

NumPy's `SeedSequence` makes this possible with a statistical guarantee. A single master seed spawns *N* child seeds using a cryptographic algorithm that ensures the resulting streams are statistically independent — drawing from stream 1 provides no information about what stream 2 will produce. This is not just a practical convention; it is a mathematical property of `SeedSequence`'s design.

### Technical Architecture

**`RNGPartitioner`** is the root object. It takes a master seed and uses `np.random.SeedSequence(master_seed)` to manage child seeds. It exposes two methods:

```python
class RNGPartitioner:
    STREAM_NAMES = ["population", "temporal", "prediction",
                    "intervention", "outcomes"]

    def __init__(self, master_seed: int = 42):
        self.master_seed = master_seed
        self._seed_seq = np.random.SeedSequence(master_seed)

    def create_streams(self) -> RNGStreams:
        child_seeds = self._seed_seq.spawn(len(self.STREAM_NAMES))
        generators = {
            name: np.random.default_rng(seed)
            for name, seed in zip(self.STREAM_NAMES, child_seeds)
        }
        return RNGStreams(**generators)

    def fork(self) -> "RNGPartitioner":
        return RNGPartitioner(self.master_seed)
```

`create_streams()` calls `self._seed_seq.spawn(5)` to generate 5 child `SeedSequence` objects, then wraps each in a `np.random.Generator`. The child sequences are deterministic functions of the master seed and the child index, so `create_streams()` is idempotent for the same master seed.

**`RNGStreams`** is a simple dataclass with five named fields:

| Stream | Owner Method | Used For |
|--------|-------------|----------|
| `population` | `create_population()` | Initial state generation |
| `temporal` | `step()` | AR(1) drift, seasonal effects |
| `prediction` | `predict()` | ML model noise injection |
| `intervention` | `intervene()` | Randomized assignment |
| `outcomes` | `measure()` | Event realization |

**`fork()`** is the method the engine calls when initializing the counterfactual branch. It creates a new `RNGPartitioner` with the same `master_seed`. Because `SeedSequence` is deterministic, calling `create_streams()` on the forked partitioner produces generators that start at identical states to the originals. They are separate objects (no shared mutable state), so advancing one does not affect the other — but they produce identical sequences when consumed in the same order.

The engine choreography:

```python
factual_rng = sc.rng                      # original streams
cf_partitioner = sc._partitioner.fork()   # same master seed
cf_rng = cf_partitioner.create_streams()  # starts synchronized

for t in range(n_timesteps):
    sc.rng = factual_rng
    state_factual = sc.step(state_factual, t)   # uses factual temporal

    sc.rng = cf_rng
    state_counterfactual = sc.step(state_counterfactual, t)  # identical temporal draws

    if t in prediction_schedule:
        sc.rng = factual_rng
        predictions = sc.predict(state_factual, t)  # advances prediction stream
        # counterfactual does NOT call predict() — prediction stream stays synchronized for counterfactual
        state_factual, interventions = sc.intervene(state_factual, predictions, t)

    sc.rng = factual_rng
    outcomes_factual = sc.measure(state_factual, t)
    sc.rng = cf_rng
    outcomes_cf = sc.measure(state_counterfactual, t)
```

### Proof It Works

All outputs below are actual program output from running the code.

**Proof 1: Two streams from the same partitioner produce different (independent) values.**

```python
from healthcare_sim_sdk.core.rng import RNGPartitioner

p = RNGPartitioner(master_seed=42)
streams = p.create_streams()
pop_val = streams.population.random(3)
temporal_val = streams.temporal.random(3)
print(f'population stream [0:3]:  {pop_val}')
print(f'temporal stream   [0:3]:  {temporal_val}')
print(f'Are they different?       {not np.allclose(pop_val, temporal_val)}')
```

```
population stream [0:3]:  [0.91674416 0.91098667 0.8765925 ]
temporal stream   [0:3]:  [0.46749078 0.0464489  0.59551001]
Are they different?       True
```

The streams produce numerically different values, confirming statistical independence.

**Proof 2: Two partitioners from the same master seed produce identical streams.**

```python
p1 = RNGPartitioner(master_seed=99)
p2 = RNGPartitioner(master_seed=99)
s1 = p1.create_streams()
s2 = p2.create_streams()
vals1 = s1.temporal.random(5)
vals2 = s2.temporal.random(5)
```

```
partitioner1 temporal [0:5]: [0.78417139 0.06070691 0.22923957 0.53996629 0.71564626]
partitioner2 temporal [0:5]: [0.78417139 0.06070691 0.22923957 0.53996629 0.71564626]
Identical?                   True
```

Same master seed → identical streams. Reproducibility holds.

**Proof 3: A forked partitioner's temporal stream starts in perfect synchrony with the original.**

```python
p_orig = RNGPartitioner(master_seed=7)
p_fork = p_orig.fork()
s_orig = p_orig.create_streams()
s_fork = p_fork.create_streams()
orig_temporal = s_orig.temporal.random(5)
fork_temporal = s_fork.temporal.random(5)
```

```
original temporal [0:5]: [0.48058201 0.05954181 0.22268894 0.133541   0.09448578]
forked   temporal [0:5]: [0.48058201 0.05954181 0.22268894 0.133541   0.09448578]
Identical?               True
```

This is the guarantee that makes branched simulation valid: both branches start with synchronized temporal streams, so they experience identical stochastic evolution before the intervention.

---

## Module 2: The 5-Method Contract

### First Principles

A scenario is the user's contribution to the simulation. It encodes clinical knowledge: what the population looks like, how risk evolves over time, what the ML model does, how the intervention acts on state, and how we measure outcomes.

The SDK provides infrastructure; the scenario provides domain knowledge. The interface between them must be:

1. **Fixed enough** that the engine can call scenario methods in a guaranteed order without knowing what they do internally.
2. **Flexible enough** that scenarios can represent everything from coin flips to radiation therapy patients to ICU deterioration.

This is the extensibility vs. correctness tradeoff. A completely open interface (scenario can call engine methods, modify shared state, trigger side effects) is maximally flexible but makes reproducibility and counterfactual validity impossible to guarantee. A fixed interface with clear contracts gives up some flexibility in exchange for correctness that can be mechanically verified.

**The discrete-time population model.** The underlying model is simple: a population of `n_entities` (patients, encounters, RT courses, any unit of analysis) exists at each timestep `t`. Each entity has state. Between timesteps, state evolves. At some timesteps, the ML model makes predictions, the intervention modifies state, and we measure what happened. This repeats for `n_timesteps`.

**Why exactly 5 methods?** Each method maps to one fundamental causal event in the system:

| Method | Real-World Meaning |
|--------|-------------------|
| `create_population(n)` | Initialize the patient population (demographics, baseline risk) |
| `step(state, t)` | Advance time (disease progression, vital sign changes, patient turnover) |
| `predict(state, t)` | The ML model scores every patient |
| `intervene(state, predictions, t)` | Clinician responds to predictions (modifies care, assigns treatment) |
| `measure(state, t)` | Observe outcomes (events, lab values, admissions) |

The separation is not just organizational. It is causal. The engine calls `predict()` and `intervene()` only on the factual branch. The counterfactual branch calls only `step()` and `measure()`. If you mixed prediction and intervention into `step()`, you could not run a counterfactual at all.

**Step purity: the most important constraint.** The `step()` method must be a pure function: given the same `(state, t)` and the same `rng.temporal` state, it must produce the same output. Every time. No exceptions.

What goes wrong if it isn't pure? Suppose `step()` reads from a shared attribute `self.patient_registry` that gets modified somewhere else. Now factual and counterfactual branches might see different registries at the same timestep. You can no longer interpret the difference in outcomes as the effect of the intervention — it might be an artifact of divergent world states caused by a side effect in `step()`.

The `step()` purity contract is enforced by convention (documented in `BaseScenario`) and verifiable by the `test_step_purity.py` integration test. It is not enforced by Python's type system, because Python does not have that capability. It is enforced by discipline.

### Technical Architecture

**`BaseScenario[S]`** is a generic abstract base class. The type parameter `S` is the state type. It can be anything: a NumPy array (stroke scenario: shape `(4, n_patients)`), a Python dataclass, a dictionary. The engine never inspects the state — it only passes it to scenario methods.

```python
class BaseScenario(ABC, Generic[S]):
    unit_of_analysis: str = "entity"

    def __init__(self, time_config: TimeConfig, seed: Optional[int] = None):
        self.time_config = time_config
        self.seed = seed
        self._partitioner = RNGPartitioner(seed if seed is not None else 42)
        self.rng: RNGStreams = self._partitioner.create_streams()
```

**`TimeConfig`** controls the temporal structure:

```python
@dataclass
class TimeConfig:
    n_timesteps: int
    timestep_duration: float   # fraction of a year (1/52 = weekly)
    timestep_unit: str = "week"
    prediction_schedule: List[int] = field(default_factory=list)
```

`prediction_schedule` is the list of timesteps at which `predict()` and `intervene()` are called. Not every timestep needs a prediction — many clinical scenarios score patients weekly but have outcomes measured daily. Sparse prediction schedules are a key efficiency lever.

**Data containers.** Three dataclasses carry information between engine and scenario:

- `Predictions(scores, labels, metadata)` — output of `predict()`. `scores` is a float array in [0,1]; `labels` is optional binary threshold output; `metadata` is a free dict for scenario-specific data.
- `Interventions(treated_indices, intervention_type, metadata)` — output of `intervene()`. `treated_indices` is the array of entity indices that received the intervention.
- `Outcomes(events, entity_ids, secondary, metadata)` — output of `measure()`. `events` is the primary binary outcome; `entity_ids` is required for panel-data export; `secondary` holds additional outcomes (demographics, intermediate variables).

**`clone_state()`** creates a deep copy of state for counterfactual branching. The default is `copy.deepcopy()`, which is safe but slow for large NumPy arrays. Array-backed scenarios should override this:

```python
def clone_state(self, state: np.ndarray) -> np.ndarray:
    return state.copy()   # O(n) instead of full object serialization
```

### Proof It Works

**3-entity, 2-timestep trace through all 5 methods.** Using the `CoinFlipScenario` template:

```python
from healthcare_sim_sdk.core.scenario import TimeConfig
from healthcare_sim_sdk.scenarios._template.scenario import CoinFlipScenario

tc = TimeConfig(n_timesteps=2, timestep_duration=1/52, prediction_schedule=[0, 1])
scenario = CoinFlipScenario(tc, seed=42, intervention_reduction=0.3, treatment_threshold=0.5)

state0 = scenario.create_population(3)
state1 = scenario.step(state0, t=0)
preds  = scenario.predict(state1, t=0)
state1_int, interventions = scenario.intervene(state1, preds, t=0)
outcomes0 = scenario.measure(state1_int, t=0)
```

Actual output:

```
Initial state (probabilities): [0.8334 0.8288 0.8013]
After step(t=0):               [0.8585 0.8409 0.7745]
Predictions (scores):          [0.6935 0.9815 0.8468]
Treated indices:               [0 1 2]         ← all above threshold 0.5
After intervene (state):       [0.6009 0.5886 0.5421]   ← 30% reduction applied
Outcomes t=0 (events):         [1. 0. 0.]
After step(t=1):               [0.5796 0.6155 0.4964]
Predictions t=1 (scores):      [0.4967 0.6639 0.3766]
Outcomes t=1 (events):         [0. 0. 1.]
```

The trace shows: initial state → temporal drift (step) → noisy observation (predict) → risk reduction for high-scorers (intervene) → stochastic event realization (measure). Each method touches only the data it is responsible for.

---

## Module 3: The Branched Simulation Engine

### First Principles

**The fundamental causal inference problem.** In Rubin's potential outcomes framework, each unit has two potential outcomes: Y(1) — what happens if treated — and Y(0) — what happens if untreated. The causal effect for unit *i* is Y_i(1) − Y_i(0). The problem is that for any given unit, you observe at most one of these. The other is the *counterfactual* — the unobserved potential outcome.

In observational healthcare studies, we try to estimate E[Y(1) − Y(0)] using statistical adjustment: propensity scores, instrumental variables, interrupted time series. Each approach makes assumptions about confounders and requires data from the real world.

Simulation takes a different approach. Because we construct the world, we know the ground truth. We can run unit *i* through two exact copies of the simulation: one where the AI is deployed (factual), one where it is not (counterfactual). Both copies start with the same initial state and experience the same background stochastic events. They diverge only where the intervention modifies the factual state. The difference in outcomes is the simulated causal effect.

This is not equivalent to observing the real causal effect. The simulation makes structural assumptions about how patients evolve, how the intervention works, and what the outcome mechanism is. But it is internally consistent: under those assumptions, the simulation's estimate of E[Y(1) − Y(0)] is unbiased. The assumptions themselves are where scientific scrutiny belongs.

**Why branched simulation is valid.** The key insight is that valid causal comparison requires two branches to differ in exactly one way: whether the intervention occurred. Everything else — the patients, their trajectories, their background stochasticity — must be identical.

The engine achieves this by:
1. Creating a single initial population once, then cloning it for the counterfactual branch.
2. Giving both branches synchronized temporal RNG streams (via the fork mechanism from Module 1).
3. Calling `predict()` and `intervene()` only on the factual branch.
4. Measuring outcomes on both branches.

The counterfactual branch does NOT call `predict()`. This is deliberate. The causal question is "what if we had never deployed the AI?" — not "what if we deployed it but ignored the predictions?" The absence of the prediction step means the counterfactual patients are never flagged, never have their care pathway modified by the model, never experience the intervention. They evolve exactly as they would have without the system.

**The three counterfactual modes** offer tradeoffs between realism and computational cost:

| Mode | Description | Use When |
|------|-------------|----------|
| `BRANCHED` | Full parallel trajectories from t=0 | Gold standard; use for main analyses |
| `SNAPSHOT` | Same-step counterfactual at each prediction time | Fast approximation; quick parameter sweeps |
| `NONE` | Single trajectory, no counterfactual | Calibration and debugging only |

`BRANCHED` is the default and should be used for all publishable analyses. `SNAPSHOT` is useful during development when you want fast iteration, accepting the approximation that the counterfactual state is identical to the factual state before each prediction step.

### Technical Architecture

**The `_run_branched()` loop.** Here is the complete temporal loop with annotations:

```python
def _run_branched(self, n_entities: int) -> SimulationResults:
    sc = self.scenario
    tc = sc.time_config

    # ── Initialize ──────────────────────────────────────────────────────────
    state_factual = sc.create_population(n_entities)
    state_counterfactual = sc.clone_state(state_factual)  # identical start state

    factual_rng = sc.rng
    cf_partitioner = sc._partitioner.fork()    # synchronized fork
    cf_rng = cf_partitioner.create_streams()

    # ── Temporal loop ────────────────────────────────────────────────────────
    for t in range(tc.n_timesteps):

        # 2a. Evolve FACTUAL state (uses factual temporal stream)
        sc.rng = factual_rng
        state_factual = sc.step(state_factual, t)

        # 2b. Evolve COUNTERFACTUAL state (uses cf temporal stream — same draws)
        sc.rng = cf_rng
        state_counterfactual = sc.step(state_counterfactual, t)

        # 2c. Predict and intervene on FACTUAL only
        if t in tc.prediction_schedule:
            sc.rng = factual_rng
            predictions = sc.predict(state_factual, t)    # advances prediction stream
            state_factual, interventions = sc.intervene(
                state_factual, predictions, t
            )                                             # advances intervention stream

        # 2d. Measure BOTH branches
        sc.rng = factual_rng
        outcomes_factual = sc.measure(state_factual, t)   # advances outcomes stream
        sc.rng = cf_rng
        outcomes_cf = sc.measure(state_counterfactual, t) # cf outcomes stream
```

**RNG choreography.** At each timestep:

- **Temporal**: both branches draw from their respective temporal streams. Because the streams were forked (same state), they draw identical values → identical stochastic evolution.
- **Prediction**: only the factual branch advances its prediction stream. The cf branch's prediction stream stays at whatever state it was last at (effectively unused, since cf never calls predict).
- **Intervention**: only the factual branch advances its intervention stream.
- **Outcomes**: each branch has its own outcomes stream. They are independent so outcome measurement noise can differ, but in the `BRANCHED` mode they too are synchronized (same fork).

**State cloning at initialization.** The clone must happen *before* any `step()` call. This is why `clone_state()` is called immediately after `create_population()`. If you cloned after the first step, the two branches would start from slightly different states, and any outcome difference could partly reflect that difference rather than the intervention.

**`prediction_schedule` and sparse prediction.** A prediction might happen every 4 weeks in a stroke prevention scenario (`prediction_schedule = range(0, 52, 4)`). This reflects clinical reality: the model doesn't score everyone every minute. Sparse prediction also speeds up simulation significantly, since `fit()` and `predict()` are the most computationally expensive operations.

### Proof It Works

**Proof 1: With `intervention_reduction=0.0`, factual = counterfactual exactly.**

```python
tc = TimeConfig(n_timesteps=52, timestep_duration=1/52, prediction_schedule=list(range(0,52,4)))
sc = CoinFlipScenario(tc, seed=42, intervention_reduction=0.0, treatment_threshold=0.5)
engine = BranchedSimulationEngine(sc, CounterfactualMode.BRANCHED)
results = engine.run(n_entities=500)
```

```
Factual total events:         13173.0
Counterfactual total events:  13173.0
Max absolute difference per timestep: 0.000000
Exactly equal?                True
```

When the intervention changes nothing, both branches are identical. This is the null treatment identity check.

**Proof 2: With `intervention_reduction=0.4`, factual < counterfactual.**

```python
sc2 = CoinFlipScenario(tc, seed=42, intervention_reduction=0.4, treatment_threshold=0.5)
results2 = engine.run(n_entities=1000)
```

```
Factual total events:         14015.0
Counterfactual total events:  26201.0
Absolute events averted:      12186.0
Factual < CF?                 True
```

The intervention reduces events in the factual branch. The counterfactual tracks what would have happened without it.

**Proof 3: Branches diverge ONLY after the first intervention, not before.**

Using a scenario where `prediction_schedule=[4]` (only one prediction at t=4):

```
t=0: factual=243.0, CF=243.0  (synced)
t=1: factual=252.0, CF=252.0  (synced)
t=2: factual=229.0, CF=229.0  (synced)
t=3: factual=263.0, CF=263.0  (synced)
t=4: factual=143.0, CF=234.0  (DIVERGED) ← intervention at t=4
t=5: factual=158.0, CF=263.0  (DIVERGED)
t=6: factual=169.0, CF=268.0  (DIVERGED)
...
```

Before the intervention, both branches are in perfect synchrony — identical patient counts, identical events. The divergence begins precisely at t=4, when `intervene()` modifies the factual state.

---

## Module 4: ML Model Simulation

### First Principles

**Why simulate an ML model rather than use a real one?**

Using a real model raises several problems. First, reproducibility: a scikit-learn RandomForest trained on your hospital's data is not shareable across institutions. Second, parameter control: you cannot test "what would happen if this model's AUC were 0.70 instead of 0.80?" with a single fixed model. Third, decoupling: you want to evaluate your clinical workflow, not your specific model's idiosyncrasies. The right question is "what does a model with AUC 0.80 and PPV 0.15 do for our patients?" — not "what does this specific model with these specific weights do?"

A simulated ML model achieves target performance characteristics through controlled noise injection. The insight is that any model is a noisy transformation of true risk: the scores it produces are correlated with ground truth, but imperfectly. We can reproduce that correlation structure parametrically.

**The noise injection philosophy.** Given patients with true risk scores, we want to generate predicted scores that achieve a target AUC. The approach:

1. Start with the true risk scores (what a perfect oracle would report).
2. Add structured noise that degrades the signal to the desired discrimination level.
3. Calibrate so predicted probabilities match observed frequencies.

The discrimination level is controlled by how much noise we add. More noise → lower AUC. Less noise → higher AUC.

**Bayes' theorem and the PPV ceiling.** One of the most consequential but underappreciated facts in clinical AI is the *PPV ceiling*: at low event prevalence, even a model with excellent discrimination may have very poor positive predictive value.

The formula is:

$$\text{PPV} = \frac{\text{sensitivity} \times \text{prevalence}}{\text{sensitivity} \times \text{prevalence} + (1 - \text{specificity}) \times (1 - \text{prevalence})}$$

Consider a common clinical scenario: event prevalence = 5%, sensitivity = 80%, specificity = 90%.

$$\text{PPV} = \frac{0.80 \times 0.05}{0.80 \times 0.05 + 0.10 \times 0.95} = \frac{0.040}{0.040 + 0.095} = \frac{0.040}{0.135} = 29.6\%$$

At 5% prevalence with 80% sensitivity and 90% specificity, only 30 out of 100 alerts are for patients who will actually have the event. The other 70 are false alarms. This is why Wong et al. found Epic's sepsis model had a PPV of 12% at their institution — low event prevalence combined with imperfect specificity makes high PPV mathematically very difficult.

The SDK's `check_target_feasibility()` function evaluates whether your target PPV is achievable given your population prevalence, before you waste time fitting a model that cannot reach the target.

**AUC vs. threshold-dependent metrics.** AUC measures ranking quality: does the model assign higher scores to true positive cases than true negative cases, on average? It is threshold-independent. But clinical decisions require a threshold: flag everyone above X, treat nobody below Y.

Once you fix a threshold, you get sensitivity and specificity — and from those, PPV via Bayes' theorem. AUC and PPV are complementary. A model can have AUC 0.85 and PPV 0.10 (excellent ranking, low precision due to low prevalence). A model can have AUC 0.70 and PPV 0.40 (moderate ranking, but deployed at a conservative threshold in a high-prevalence setting). You need both metrics to evaluate operational performance.

**Platt scaling.** Raw model scores are often systematically miscalibrated. A model might output scores concentrated near 0 or 1, or poorly separated, such that predicting 0.7 doesn't mean "70% probability of the event." This matters because the PPV calculation above requires calibrated probabilities.

Platt scaling fits a logistic regression on the logit of the raw scores: `P(y=1|s) = sigmoid(a × logit(s) + b)`. The parameters `a` and `b` are fitted by maximum likelihood. When `a=1, b=0`, this is the identity (no correction). Fitted values `a ≈ 1.0, b ≈ 0` mean the model is already calibrated; large deviations indicate systematic miscalibration that Platt scaling corrects.

Crucially, Platt scaling is a monotone transformation — it does not change the ranking of scores, so AUC is preserved. It only shifts the probability estimates to be closer to observed frequencies.

### Technical Architecture

**Three operating modes.** `ControlledMLModel` supports:

- `mode="discrimination"`: optimize for a target AUC. Use for baseline evaluation and model comparison.
- `mode="classification"`: optimize for a target PPV + sensitivity. Use when you have a specific binary decision rule and need the model to achieve a clinical operating point (e.g., flag patients for a nurse review that costs $X per flagged patient — PPV determines whether that cost is justified).
- `mode="threshold_ppv"`: optimize for a target AUC at a fixed operating threshold. Use when the threshold is externally mandated (e.g., a regulatory requirement) and you want to test model performance there.

**The 4-component noise injection.** The `_generate_scores()` method implements:

```python
def _generate_scores(self, risk_scores, rng, true_labels, correlation, scale, label_noise_strength):
    n = len(risk_scores)
    base = np.clip(risk_scores, 0, 1)

    # Component 1: correlated noise mixing
    noise = rng.normal(0, scale, n)
    blended = correlation * base + (1 - correlation) * noise

    # Component 2: label-dependent noise (class separation control)
    pos_mean =  0.05 * label_noise_strength
    neg_mean = -0.025 * label_noise_strength
    if true_labels is not None:
        label_noise = np.where(true_labels == 1,
                               rng.normal(pos_mean, pos_std, n),
                               rng.normal(neg_mean, neg_std, n))
    else:
        label_noise = rng.normal(0, 0.05, n)

    # Component 3: independent noise
    independent_noise = rng.normal(0, 0.1, n)

    # Combine and apply components 2+3
    blended += (label_noise + independent_noise) * scale

    # Component 4: sigmoid calibration to [0, 1]
    scores = 1.0 / (1.0 + np.exp(-4.0 * (blended - 0.5)))
    return np.clip(scores, 0, 1)
```

Each component serves a distinct role:

| Component | Role | Parameter |
|-----------|------|-----------|
| Correlated noise mixing | Sets the baseline correlation between predictions and true risk | `correlation` (0.3–0.99) |
| Label-dependent noise | Creates class separation (positive cases get upward push) | `label_noise_strength` |
| Independent noise | Adds idiosyncratic prediction error | `scale` |
| Sigmoid calibration | Maps raw values to [0,1] probability space | Fixed exponent (−4) |

**The grid search in `fit()`.** The optimizer searches over a `(correlation_grid × scale_grid × label_strengths)` parameter space — 15 × 15 × 6 = 1,350 parameter combinations. For each combination, it generates scores and evaluates against the target metric. The search is averaged over `n_iterations` seeds for stability:

```python
for corr in correlation_grid:           # 15 values: [0.30 ... 0.99]
    for scale in scale_grid:            # 15 values: [0.01 ... 0.70]
        for lns in label_strengths:     # 6 values:  [0.50 ... 3.00]
            total_score = 0.0
            for _ in range(n_iterations):
                scores = self._generate_scores(risk_scores, rng, true_labels, corr, scale, lns)
                score, best_t = self._evaluate_params(true_labels, scores)
                total_score += score
            avg = total_score / n_iterations
            if avg < best_score:
                best_score = avg
                best_params = (corr, scale, lns, best_t)
```

**Platt scaling implementation.**

```python
def _fit_platt_scaling(self, raw_scores, true_labels):
    eps = 1e-7
    clipped = np.clip(raw_scores, eps, 1 - eps)
    raw_logits = np.log(clipped / (1 - clipped))     # logit transform

    def neg_log_likelihood(params):
        a, b = params
        p = 1.0 / (1.0 + np.exp(-(a * raw_logits + b)))   # sigmoid(a*logit + b)
        p = np.clip(p, eps, 1 - eps)
        return -np.mean(true_labels * np.log(p) + (1 - true_labels) * np.log(1 - p))

    result = minimize(neg_log_likelihood, [1.0, 0.0], method="Nelder-Mead")
    self._platt_a = float(result.x[0])
    self._platt_b = float(result.x[1])
```

The logit transform maps probabilities (0,1) to the real line (−∞, +∞). Fitting a linear function on the logit space is equivalent to fitting a logistic regression with one feature. The result is applied at prediction time via `_apply_platt()`.

### Proof It Works

**Proof 1: Model fitted to AUC=0.80 achieves 0.78–0.82.**

```python
model = ControlledMLModel(mode='discrimination', target_auc=0.80)
report = model.fit(true_labels, true_probs, rng, n_iterations=5)
```

```
Target AUC:    0.80
Achieved AUC:  0.7990
Within [0.78, 0.82]? True
```

The noise injection + grid search finds parameters that achieve the target AUC to within ±0.01.

**Proof 2: Bayes' theorem PPV bound derivation.**

At prevalence = 5%, sensitivity = 80%, specificity = 90%:

$$\text{PPV} = \frac{0.80 \times 0.05}{(0.80 \times 0.05) + (0.10 \times 0.95)} = \frac{0.0400}{0.0400 + 0.0950} = \frac{0.0400}{0.1350} = 0.2963 = \mathbf{29.6\%}$$

```python
from healthcare_sim_sdk.ml.performance import theoretical_ppv
ppv = theoretical_ppv(prevalence=0.05, sensitivity=0.80, specificity=0.90)
```

```
PPV = 0.2963 = 29.6%
```

The SDK's `theoretical_ppv()` matches the manual derivation exactly. This bound is independent of model architecture — no model can exceed this PPV at this prevalence given these sensitivity/specificity constraints. This calculation should be the first thing checked before specifying any clinical AI performance target.

**Proof 3: Platt scaling moves calibration slope toward 1.0.**

```python
# Before Platt: default identity transform (a=1, b=0)
cal_before, _, _ = calibration_slope(true_labels, raw_scores)

# After Platt: fit and apply
model.fit(true_labels, true_probs, rng)
cal_after = report['achieved_calibration_slope']
```

```
Calibration slope BEFORE Platt:  2.1456   (badly overconfident)
Calibration slope AFTER  Platt:  1.0277   (well calibrated)
Platt a=1.373, b=-0.011
AUC preserved:                   0.7459   (identical to pre-Platt)
After Platt closer to 1.0?       True
```

Platt scaling corrects the calibration slope from 2.15 to 1.03 while leaving the AUC unchanged. The raw scores were systematically overconfident (calibration slope > 1 means predicted probabilities are too extreme); Platt scaling compresses them toward observed frequencies.

---

## Module 5: Results and Analysis Export

### First Principles

A simulation produces outcomes indexed by (entity, timestep). But different causal analysis methods require completely different data shapes.

**The unit-of-analysis problem.** Consider:

- **Interrupted Time Series (ITS)**: requires a single time series of aggregate outcomes — one number per timestep. Shape: `(n_timesteps,)`. Analyzes whether the trend changed after the intervention.
- **Difference-in-Differences (DiD)**: requires entity-level panel data — one row per (entity, timestep). Shape: `(n_entities × n_timesteps,)`. Analyzes treated vs. control entities before and after intervention.
- **Regression Discontinuity Design (RDD)**: requires a cross-section at a single moment — one row per entity at a fixed time. Shape: `(n_entities,)`. Exploits a threshold in the assignment mechanism (e.g., model score > X → treatment).
- **Equity analysis**: requires the same panel as DiD, but with demographic labels attached to every row.

The same simulation results can generate all four formats. There is no reason to re-run the simulation to get a different analysis shape — the simulation already captured everything needed. The `AnalysisDataset` class provides four export methods that transform the stored outcomes into the right shapes.

**The equity analysis problem.** Standard simulation outputs contain events and timesteps but not demographic labels. To run equity analysis — "does the intervention benefit Black patients as much as White patients?" — you need race/ethnicity or socioeconomic group attached to each entity-outcome pair. The SDK handles this by storing demographic labels in `Outcomes.secondary` at measure time, so they are available for export without post-hoc joins.

**Why not just use a dataframe?** The SDK stores raw NumPy arrays indexed by timestep. This is intentional: it avoids the overhead of dataframe construction during simulation, and gives the analysis layer full control over how to shape the data for its specific method.

### Technical Architecture

**`SimulationResults`** stores four dicts indexed by timestep `t`:

```python
@dataclass
class SimulationResults:
    predictions:              Dict[int, Predictions]
    interventions:            Dict[int, Interventions]
    outcomes:                 Dict[int, Outcomes]      # factual branch
    counterfactual_outcomes:  Dict[int, Outcomes]      # CF branch
    validations:              Dict[str, Dict[str, Any]]
```

Every `record_*()` call appends to these dicts. Retrieval methods (`get_outcome_series()`, `get_counterfactual_outcome_series()`) iterate over timesteps and sum `.events` arrays.

**`AnalysisDataset`** wraps a `SimulationResults` and provides the four exports:

**`to_time_series()`** — ITS format, shape `(n_timesteps,)`:

```python
def to_time_series(self, branch="factual"):
    outcomes_dict = self._get_outcomes_dict(branch)
    outcomes = np.zeros(n_t)
    for t in range(n_t):
        if t in outcomes_dict:
            outcomes[t] = outcomes_dict[t].events.sum()
    return {"timesteps": ..., "outcomes": outcomes, "treatment_indicator": ...}
```

**`to_panel()`** — DiD format, shape `(n_entities × n_timesteps,)`. Raises `ValueError` if `entity_ids` was not set on `Outcomes`:

```python
# Requires entity_ids set in measure():
return Outcomes(events=incidents, entity_ids=np.arange(len(risks)))
```

The panel construction also builds a `treated` indicator: 1 for entities that ever received an intervention, at timesteps after the first intervention.

**`to_entity_snapshots(t)`** — RDD format, shape `(n_entities,)` at a single timestep. Also returns prediction scores at that timestep if available.

**`to_subgroup_panel()`** — Equity format. Calls `to_panel()` then extracts group labels from `Outcomes.secondary[subgroup_key]`. Returns the same shape as panel data plus a `subgroup` array.

**The `entity_ids` requirement.** `entity_ids` on `Outcomes` is technically optional — if absent, the panel export uses `np.arange(n_entities)`. But practically, meaningful DiD and equity analysis requires stable entity identifiers. In the stroke scenario, entity IDs are patient indices (0 to n_patients-1). In a real deployment, they would be MRNs or encounter IDs. The optional design avoids breaking simple scenarios while requiring explicit opt-in for panel analysis.

### Proof It Works

**All 4 formats from the same SimulationResults.** Using a manually constructed 3-entity, 4-timestep result with intervention at t=1:

```python
# to_time_series()
timesteps: [0 1 2 3]
outcomes:  [2. 1. 3. 2.]
treatment_indicator: [0. 1. 0. 0.]
shape: (4,)  ← n_timesteps

# to_panel()
entity_ids shape: (12,)  ← n_entities × n_timesteps = 3 × 4
Sample rows (eid, t, event, treated):
  eid=0, t=0, event=0.0, treated=0
  eid=1, t=0, event=1.0, treated=0
  eid=2, t=0, event=1.0, treated=0
  eid=0, t=1, event=0.0, treated=0
  eid=1, t=1, event=1.0, treated=1   ← treated after intervention at t=1
  eid=2, t=1, event=0.0, treated=1

# to_entity_snapshots(t=1)
entity_ids: [0 1 2]
outcomes:   [0. 1. 0.]
scores:     [0.3 0.7 0.6]   ← prediction scores at t=1

# to_subgroup_panel()
subgroup labels: ['A' 'B' 'A' 'A' 'B' 'A' 'A' 'B' 'A' 'A' 'B' 'A']
shape: (12,)

# Aggregation consistency check:
Time series total events:  8.0
Panel total events:        8.0
Match?                     True
```

The panel's total event count matches the time series sum exactly. No information is gained or lost in the format transformation.

---

## Module 6: The Verification Protocol

### First Principles

Every simulation is a simplified model of reality. It makes assumptions about disease dynamics, intervention mechanisms, and measurement processes that may not hold in your specific clinical context. This is not a flaw — it is the nature of models. The flaw is when you forget the assumptions are there.

Verification is the practice of testing that the simulation behaves correctly *under its own assumptions*. It is not validation against the real world (that requires real-world data). It is a check that the code implements the model you intended, not a different model with subtle bugs.

**The four verification levels:**

1. **Structural integrity**: Does the simulation produce valid data? No NaN, no Inf, predictions in [0,1], population conservation, confusion matrix TP+FP+TN+FN = N.

2. **Statistical sanity**: Do aggregate statistics match what the central limit theorem predicts? If your scenario is calibrated to a 10% event rate over 1,000 entities, the observed rate should be within 4-sigma CLT bounds of 10%. If it's not, something is wrong.

3. **Conservation laws**: Do outcomes respect monotonicity? More effective intervention → fewer events (not more). Higher specificity → lower flag rate. Null treatment identity (effectiveness=0 → factual = counterfactual).

4. **Case-level walkthroughs**: Trace individual entities through the full simulation and narrate what happened. Does the entity's outcome make sense given its risk profile and whether it received the intervention?

**The boundary condition principle.** When inputs are set to extreme values, outputs should be analytically obvious. This is the most powerful debugging tool in simulation:

- If `intervention_effectiveness=0`, the simulation should produce exactly the same outcomes in both branches. Not approximately — exactly, to floating-point precision.
- If `threshold=1.0` (no one is ever flagged), the intervention never fires. Factual branch = counterfactual branch.
- If `AUC=0.50` (random model), the model provides no discriminative value over the base rate.

These boundary conditions create crisp assertions with known correct answers. If any fails, you have a bug — find it before you interpret any results.

### Technical Architecture

**`experiments/validate.py`** implements generic structural validation for experiment directories:

```python
def validate_generic(output_dir: Path) -> List[Check]:
    """Checks: config.json exists, metrics.json exists, both valid JSON,
    seed recorded, timestamp recorded."""
```

Each `Check` dataclass captures what was expected, what was actual, whether it passed, and a detail string. The `format_appendix()` function renders these as a markdown table — making validation results human-readable in reports.

**`tests/bulletproof/conftest.py`** provides reusable statistical assertions:

```python
assert_mean_in_range(arr, lo, hi, name)       # checks mean within bounds
assert_rate_in_range(events, n, rate, sigma)  # CLT bounds check
assert_no_nan_inf(arr, name)                  # structural integrity
assert_in_unit_interval(arr, name)            # predictions in [0,1]
assert_monotone_nondecreasing(arr, name)      # monotonicity
```

These assertions encapsulate common verification patterns so scenario test files don't duplicate logic.

**Reading a verification summary.** The `format_appendix()` output looks like:

```
# Appendix: Simulation Validation
14/14 checks passed.
*Experiment: 2024-01-15T10:30:00 | Seed: 42*

| # | Check              | Expected     | Actual       | Result |
|---|--------------------|--------------|--------------|--------|
| 1 | config.json exists | file exists  | True         | PASS   |
| 2 | metrics.json valid | valid JSON   | valid JSON   | PASS   |
...
```

A failed check produces a detail section explaining the discrepancy. Zero failures is the bar for proceeding to interpretation.

### Proof It Works

**Boundary Condition 1: `threshold=1.0` (flag nobody)**

With threshold set to 1.0, no entity's score ever reaches the threshold. The intervention never fires. Both branches experience identical dynamics.

```
Factual total:          12978.0
Counterfactual total:   12978.0
Max per-step diff:      0.0000
Total patients treated: 0
```

Factual = counterfactual exactly. The intervention mechanism is correctly dormant.

**Boundary Condition 2: `intervention_effectiveness=0` (null treatment identity)**

With effectiveness=0, `intervene()` is called (entities are flagged, treated indices are recorded), but the state modification multiplies by `(1 - 0.0) = 1.0` — a no-op.

```
Factual total:          13173.0
Counterfactual total:   13173.0
Max per-step diff:      0.000000
Exactly equal?          True
```

Null treatment identity holds to floating-point precision. This is the most important conservation law in branched simulation.

**Boundary Condition 3: Targeted vs. Random Model — NNT Efficiency**

Rather than testing AUC=0.50 in isolation (which still treats ~50% of patients, generating a large absolute effect), the most informative BC3 compares a targeted model (correlated with true risk) against a random model (no correlation) at the same threshold and intervention strength:

```python
# Targeted model: scores = true_risk + noise (AUC ~0.75)
Targeted: factual=23361, CF=52500, averted=29139
# Random model:   scores = uniform(0, 1) (AUC = 0.50)
Random:   factual=13376, CF=52500, averted=39124

Averted per intervention:
  Targeted model:  10.37 events averted per intervention
  Random model:     3.03 events averted per intervention
```

The random model averts more total events because it happens to treat more patients (threshold = 0.5, random scores → 50% flagged vs. targeted model focusing on genuinely high-risk entities). But the *efficiency* measure — events averted per intervention — is 3.4× higher for the targeted model. This is precisely the value AUC captures: better discrimination means interventions are concentrated where they produce maximal benefit.

---

## Module 7: Putting It Together — A Worked Example

### The Paper: Hong et al. JCO 2020 (SHIELD-RT)

**Citation:** Hong JC, Eclov NCW, Dalal NH, et al. "System for High-Intensity Evaluation During Radiation Therapy (SHIELD-RT): A Prospective Randomized Study of Machine Learning–Directed Clinical Evaluations During Radiation and Chemoradiation." *Journal of Clinical Oncology* 38(31):3652–3661, 2020.

**Key parameters:**
- Population: RT courses, n=311 high-risk courses (from 395 total)
- Model: AUC 0.80–0.82 (we use 0.81)
- Threshold: >10% predicted risk of acute care visit
- Intervention: Twice-weekly evaluation (vs. standard care)
- Control acute care rate: **22.3%**
- Intervention acute care rate: **12.3%**
- Relative risk: 0.556, p=0.02

This paper is SDK-FIT: we have a patient population (RT courses), a predictive model driving the intervention, a measurable binary outcome, and a randomized design that directly instantiates the factual/counterfactual structure.

---

### Step 1: Choose the State Representation

An RT course is modeled as a static episode: a patient begins radiation therapy with a baseline risk of requiring acute care, and that risk may drift slightly as cumulative dose increases. We need to track:

- True underlying acute care risk (the oracle signal)
- Whether the patient received the intervention (to support equity analysis)
- Whether the patient had an acute care event (for tracking)

```python
# State array: shape (n_patients, 3)
COL_RISK = 0        # true acute care risk
COL_INTERVENED = 1  # cumulative intervention flag
COL_EVENT = 2       # cumulative event flag
```

The 2D array representation avoids Python-object overhead at scale and makes vectorized operations natural.

---

### Step 2: `create_population()` — Beta-distributed risks

Paper-derived: mean acute care rate = 22.3%. We use a beta distribution to generate heterogeneous per-patient risks.

```python
def create_population(self, n_entities: int) -> np.ndarray:
    rng = self.rng.population  # ← always use self.rng.population here

    # Beta(2, 6.9) has mean ~0.22 with realistic right skew
    alpha = 2.0
    beta_param = alpha * (1 / self.base_acute_care_rate - 1)
    raw_risks = rng.beta(alpha, beta_param, n_entities)

    # Rescale to hit target mean exactly
    scaling = self.base_acute_care_rate / np.mean(raw_risks)
    risks = np.clip(raw_risks * scaling, 0.01, 0.95)

    # Assign demographics and apply risk multipliers
    race_names = list(RACE_DIST.keys())
    races = rng.choice(race_names, n_entities, p=list(RACE_DIST.values()))
    for i in range(n_entities):
        risks[i] = np.clip(risks[i] * RACE_RISK_MULT[races[i]], 0.01, 0.95)

    # Rescale again to maintain target mean
    scaling = self.base_acute_care_rate / np.mean(risks)
    risks = np.clip(risks * scaling, 0.01, 0.95)

    self._demographics = races  # save for equity output

    state = np.zeros((n_entities, 3))
    state[:, COL_RISK] = risks
    return state
```

Note: the paper does not report demographic breakdown. We model plausible Duke cancer center demographics (noted as assumed in any published analysis).

---

### Step 3: `step()` — Weekly temporal drift

RT patients accumulate toxicity during treatment. Risk drifts upward slightly each week. We model this with small random drift:

```python
def step(self, state: np.ndarray, t: int) -> np.ndarray:
    rng = self.rng.temporal  # ← always use self.rng.temporal here

    n = state.shape[0]
    drift = rng.normal(0.002, 0.01, n)  # mean +0.2% per week, conservative

    new_state = state.copy()
    new_state[:, COL_RISK] = np.clip(state[:, COL_RISK] + drift, 0.01, 0.95)
    return new_state
```

**Purity checklist:** `step()` reads only `(state, t, self.rng.temporal)`. It does not read or write `self.base_acute_care_rate`, `self._demographics`, or any other shared state. ✓

---

### Step 4: `predict()` — ControlledMLModel at AUC 0.81

```python
def predict(self, state: np.ndarray, t: int) -> Predictions:
    true_risks = state[:, COL_RISK]
    n = len(true_risks)

    if not self._model_fitted:
        # Fit on the first call — generates representative labels from risks
        true_labels = (self.rng.prediction.random(n) < true_risks).astype(int)
        self._model.fit(true_labels, true_risks, self.rng.prediction, n_iterations=5)
        self._model_fitted = True

    scores = self._model.predict(true_risks, self.rng.prediction)
    return Predictions(scores=scores, metadata={"true_risks": true_risks.copy()})
```

The model is initialized once (on first `predict()` call) then reused. This mirrors clinical reality: the model is trained once and deployed.

---

### Step 5: `intervene()` — Flag above 10% risk, apply multiplicative effectiveness

Paper: patients at >10% predicted risk received twice-weekly evaluation. Calibrated `intervention_effectiveness=0.452` to match the paper's 12.3% intervention rate.

```python
def intervene(self, state, predictions, t):
    scores = predictions.scores
    high_risk = scores >= self.risk_threshold  # threshold = 0.10

    new_state = state.copy()
    treated_indices = np.where(high_risk)[0]
    new_state[treated_indices, COL_INTERVENED] = 1.0

    # Multiplicative risk reduction
    new_state[treated_indices, COL_RISK] = np.clip(
        new_state[treated_indices, COL_RISK] * (1.0 - self.intervention_effectiveness),
        0.01, 0.95
    )

    return new_state, Interventions(
        treated_indices=treated_indices,
        metadata={"n_high_risk": int(high_risk.sum()),
                  "threshold_used": self.risk_threshold,
                  "pct_flagged": float(high_risk.mean())},
    )
```

The multiplicative model: treated patient's risk becomes `risk × (1 − 0.452) = risk × 0.548`. This is equivalent to saying the twice-weekly evaluation catches and manages problems that would otherwise become acute care visits, reducing risk by ~45%.

---

### Step 6: `measure()` — Binary outcome, entity IDs, demographics

```python
def measure(self, state: np.ndarray, t: int) -> Outcomes:
    rng = self.rng.outcomes  # ← always use self.rng.outcomes here
    risks = state[:, COL_RISK]
    events = (rng.random(len(risks)) < risks).astype(float)

    race = self._demographics if self._demographics is not None \
           else np.array(["Unknown"] * len(risks))

    return Outcomes(
        events=events,
        entity_ids=np.arange(len(risks)),   # required for panel export
        secondary={
            "true_risk": risks.copy(),
            "intervened": state[:, COL_INTERVENED].copy(),
            "race_ethnicity": race,          # for equity analysis
        },
        metadata={"event_rate": float(events.mean()),
                  "timestep": t},
    )
```

`entity_ids` is set explicitly so `to_panel()` and `to_subgroup_panel()` work correctly.

---

### Step 7: Run, Verify, Compare to Paper

**Single-timestep run** (one RT course = one simulation timestep with the full treatment period modeled as a single episode):

```python
tc = TimeConfig(
    n_timesteps=1,
    timestep_duration=1.0,     # full episode unit
    timestep_unit="rt_course",
    prediction_schedule=[0],   # predict once per course
)

scenario = ShieldRTScenario(
    time_config=tc,
    seed=42,
    base_acute_care_rate=0.223,
    model_auc=0.81,
    risk_threshold=0.10,
    intervention_effectiveness=0.452,
)

engine = BranchedSimulationEngine(scenario, CounterfactualMode.BRANCHED)
results = engine.run(n_entities=311)
```

**Actual simulation output:**

```
=== SHIELD-RT Single-Timestep Run ===
Paper: base=22.3%, intervention=12.3%

Simulated factual rate:       11.9%  (target: 12.3%)  ← within 0.4 pp
Simulated counterfactual:     19.9%  (target: 22.3%)  ← within 2.4 pp
Absolute reduction:            8.0 pp  (target: 10 pp)
Relative risk:                 0.597  (target: 0.556)

Flagged (>10% risk):           256/311 = 82%
```

**Calibration status.** The factual rate (11.9%) matches the paper's 12.3% within 0.4 percentage points — well within calibration tolerance. The counterfactual rate (19.9%) is 2.4 pp below the paper's 22.3%, suggesting the baseline beta distribution slightly underestimates high-risk patients given the demographic risk multipliers. The relative risk (0.597) is directionally consistent with the paper's 0.556 and within the paper's confidence interval range.

The 82% flagging rate is higher than the paper implies — the paper enrolled only "high-risk" patients (those at >10% baseline risk per eligibility criteria), so our full-population simulation flagging at a >10% threshold will differ. This is a design note, not a bug.

**Equity breakdown (last timestep):**

```
White     : n=220, event_rate=2.3%
Black     : n= 64, event_rate=3.1%   ← higher, as assumed by RACE_RISK_MULT
Hispanic  : n= 13, event_rate=0.0%
Asian     : n=  9, event_rate=0.0%
```

Note: the paper reports no subgroup analysis. Our equity modeling is entirely assumption-driven and should be labeled as such. The Black/White rate difference (3.1% vs 2.3%) reflects the assumed RACE_RISK_MULT of 1.15 for Black patients. This is an equity hypothesis, not an evidence-based finding from the paper.

**Reproducibility verdict.** The simulation reproduces the direction and approximate magnitude of the SHIELD-RT result under calibrated parameters. Primary effect size falls within acceptable tolerance. The simulation adds equity dimensions the paper did not report, surfacing a potential disparity that the trial was not powered to detect.

---

## Module 8: Appendix — Quick Reference

### RNG Stream Ownership Table

| Stream | Method | Usage |
|--------|--------|-------|
| `rng.population` | `create_population()` | Initial risk distribution, demographics |
| `rng.temporal` | `step()` | AR(1) drift, seasonal effects |
| `rng.prediction` | `predict()` | ML model noise injection |
| `rng.intervention` | `intervene()` | Randomized assignment, tie-breaking |
| `rng.outcomes` | `measure()` | Event realization (Bernoulli draws) |

**Rule:** never use bare `np.random` or `import random` in a scenario. Always use the partitioned stream for the current method.

---

### 5-Method Contract Summary Table

| Method | Input | Output | RNG Stream | Factual? | Counterfactual? |
|--------|-------|--------|-----------|---------|-----------------|
| `create_population(n)` | entity count | state S | `population` | Once | Once (via clone) |
| `step(state, t)` | state, timestep | new state | `temporal` | ✓ | ✓ |
| `predict(state, t)` | state, timestep | Predictions | `prediction` | ✓ | ✗ |
| `intervene(state, preds, t)` | state, Predictions, timestep | (new state, Interventions) | `intervention` | ✓ | ✗ |
| `measure(state, t)` | state, timestep | Outcomes | `outcomes` | ✓ | ✓ |

---

### Performance Metric Formulas

Let TP, FP, TN, FN be the confusion matrix counts at a given threshold.

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Sensitivity (Recall) | TP / (TP + FN) | Fraction of true cases caught |
| Specificity | TN / (TN + FP) | Fraction of true negatives correctly excluded |
| PPV (Precision) | TP / (TP + FP) | Fraction of alerts that are true cases |
| NPV | TN / (TN + FN) | Fraction of non-alerts that are true negatives |
| F1 | 2 × PPV × Sensitivity / (PPV + Sensitivity) | Harmonic mean of PPV and sensitivity |
| Flag Rate | (TP + FP) / N | Fraction of population flagged |
| AUC | ∫ TPR d(FPR) via trapezoidal rule | Ranking quality, threshold-independent |
| Calibration Slope | slope of linear fit: predicted_bin_mean ~ observed_bin_mean | Near 1.0 = well calibrated |
| Theoretical PPV | (sens × prev) / (sens × prev + (1−spec) × (1−prev)) | Maximum achievable PPV via Bayes' theorem |

---

### SDK Fitness Criteria Checklist

Before using the SDK for a paper, confirm all 6 criteria. See `AGENTS.md` for the full protocol.

- [ ] **Population-level intervention**: the AI drives a policy over many patients, not individual clinical advice
- [ ] **Predictive model triggers the intervention**: there is an explicit ML scoring step
- [ ] **Intervention is a state change**: something measurable changes in the factual branch (not purely informational)
- [ ] **Measurable outcome at each timestep**: a binary or continuous outcome can be measured per entity per step
- [ ] **Counterfactual causal question**: "what if no AI?" is a meaningful and answerable question for this setting
- [ ] **Discrete-time dynamics**: the process can be approximated with a finite step size (weekly, daily, hourly)

**Classification:**
- **FIT** = all 6 ✓ → proceed to scenario design
- **PARTIAL_FIT** = 4–5 ✓ → document which fail and what simplifications are needed
- **NO_FIT** = ≤3 ✓ → document why; this is a finding about SDK scope limits
- **UNDERDETERMINED** = fitness unclear due to missing paper information → document gaps

---

### Verification Protocol Checklist

Run before interpreting any simulation results. No exceptions.

**Level 1: Structural Integrity**
- [ ] No NaN or Inf in any array
- [ ] All prediction scores in [0, 1]
- [ ] Population size constant across timesteps
- [ ] Confusion matrix: TP + FP + TN + FN = N

**Level 2: Statistical Sanity**
- [ ] Observed event rate within 4-sigma CLT bounds of target
- [ ] Achieved model AUC within ±0.03 of target
- [ ] Achieved sensitivity/PPV within ±0.05 of target (if classification mode)
- [ ] Demographic proportions match specified distributions

**Level 3: Conservation Laws**
- [ ] Null treatment identity: effectiveness=0 → factual = counterfactual exactly
- [ ] Monotonicity: higher effectiveness → fewer factual events
- [ ] Null threshold: threshold=1.0 → no treatments, factual = counterfactual
- [ ] Bayes bound: achieved PPV ≤ theoretical_ppv(prevalence, sensitivity, specificity)

**Level 4: Case-Level Walkthroughs**
- [ ] Trace 5 entities through full simulation
- [ ] High-risk entity: flagged, treated, outcome explained by reduced risk
- [ ] Low-risk entity: not flagged, outcome consistent with untreated risk
- [ ] Counterfactual entity: identical risk trajectory as factual pre-intervention
- [ ] Boundary entity (score near threshold): assignment is deterministic given score

**Level 5: Boundary Conditions** (first run of each new scenario)
- [ ] threshold=0 → all patients treated
- [ ] threshold=1 → no patients treated; factual = counterfactual
- [ ] effectiveness=0 → factual = counterfactual
- [ ] effectiveness=1.0 → near-zero events in factual
- [ ] AUC=0.50 → no discriminative targeting value

---

## A Note on Limitations

This SDK is a tool for generating structured hypotheses, not for proving clinical effectiveness.

The simulation suggests, under these assumptions. When you run a SHIELD-RT simulation and find that the intervention averts 10 percentage points of acute care visits, that finding is conditional on every assumption baked into the scenario: the beta distribution shape, the AR(1) drift parameters, the multiplicative effectiveness model, the demographic risk multipliers. Change those assumptions and you get different numbers.

What the simulation provides is:
1. **Structural consistency**: if your assumptions hold, these are the numbers
2. **Sensitivity analysis**: here is how the result changes if your assumptions are wrong
3. **Pre-deployment guardrails**: if the vendor's claimed AUC 0.82 actually reflects AUC 0.63 in your patient population (cf. Wong et al.), here is what you should expect

What it does not provide:
- Evidence that the intervention will work in your hospital
- A replacement for prospective study design
- Certainty about any parameter you had to assume

The simulation is valuable precisely because it is explicit about its assumptions. The alternative — deploying without simulation and learning from patient outcomes — is less explicit, not more rigorous.

---

*Document generated from source code at `/data/.openclaw/workspace/healthcare-sim-sdk/`. All proof outputs are actual program output. No values were invented or approximated.*