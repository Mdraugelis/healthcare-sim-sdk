# Task Checklist: NB01–NB04 Implementation

## NB01 — RNG Partitioning
**Module:** `sdk/core/rng.py` | **Notebook:** `notebooks/01_rng_partitioning.ipynb`

- [x] **#1** Implement RNGStreams dataclass and RNGPartitioner
  - `RNGStreams`: 5 named `np.random.Generator` fields (population, temporal, prediction, intervention, outcomes)
  - `RNGPartitioner.__init__(master_seed)`, `create_streams()` via `SeedSequence.spawn(5)`, `fork()` for branching
- [x] **#2** Create RNG partitioning proof notebook *(blocked by #1)*
  - Reproducibility, stream independence, fork equivalence, cross-stream correlation heatmap
- [x] **#3** Write RNG unit tests → `tests/unit/test_rng.py` *(blocked by #1)*
  - Reproducibility, stream independence, fork equivalence, all 5 streams present, different seeds differ

---

## NB02 — State & Clone
**Module:** `sdk/core/scenario.py` (clone utilities) | **Notebook:** `notebooks/02_state_and_clone.ipynb`

- [x] **#4** Implement clone_state with mutation isolation
  - Default `clone_state(state)` via `copy.deepcopy`; override pattern for optimized implementations
  - Must handle: numpy arrays, dataclasses, nested dicts, mixed structures
- [x] **#5** Create state & clone proof notebook *(blocked by #4)*
  - Deep copy correctness, mutation isolation, memory profiling at scale, custom override pattern
- [x] **#6** Write state management unit tests → `tests/unit/test_state.py` *(blocked by #4)*
  - Mutation isolation on arrays, dataclasses, nested dicts; custom override; deep nested mutation

---

## NB03 — Discrete-Time Engine
**Modules:** `sdk/core/engine.py`, `sdk/core/scenario.py` | **Notebook:** `notebooks/03_discrete_time_engine.ipynb`

- [x] **#7** Implement BaseScenario ABC and data classes *(blocked by #1, #4)*
  - `TimeConfig`, `Predictions`, `Interventions`, `Outcomes` dataclasses
  - `BaseScenario(ABC, Generic[S])`: 5 abstract methods, `unit_of_analysis` attribute, `self.rng` set by engine
- [x] **#8** Implement BranchedSimulationEngine *(blocked by #1, #4, #7)*
  - `CounterfactualMode` enum (BRANCHED, SNAPSHOT, NONE)
  - Engine loop: step→predict→intervene→measure with RNG context swapping per branch
- [x] **#9** Create toy counter scenario for engine testing *(blocked by #7, #8)*
  - Trivial `CounterScenario(BaseScenario)`: step increments, predict returns counter, intervene adds 10, measure records
- [x] **#10** Create discrete-time engine proof notebook *(blocked by #8, #9)*
  - Hook ordering, clock management, all 3 modes, factual vs counterfactual divergence plot
- [x] **#11** Write engine unit tests → `tests/unit/test_engine.py` *(blocked by #8, #9)*
  - Hook ordering, clock, prediction_schedule, all 3 modes, toy counter expected values

---

## NB04 — Step Purity Contract
**Module:** `tests/integration/test_step_purity.py` | **Notebook:** `notebooks/04_step_purity_contract.ipynb`

- [x] **#12** Implement BRANCHED-vs-NONE identity test helper *(blocked by #8, #9)*
  - `assert_purity(scenario_class, seed, n_steps, **kwargs)`: runs BRANCHED and NONE, asserts factual == NONE element-wise
  - THE critical test — proves RNG partitioning, fork, purity, engine swapping, and clone_state together
- [x] **#13** Create impure and RNG-violating scenarios for failure demos *(blocked by #12)*
  - `ImpureScenario`: step reads/writes shared `self.counter` (not in state)
  - `RNGViolationScenario`: step draws from `self.rng.intervention` instead of `self.rng.temporal`
  - Both must FAIL the identity test
- [x] **#14** Create step purity contract proof notebook *(blocked by #12, #13)*
  - Correct scenario passes, broken scenarios fail with visible divergence points

---

## Dependency Graph

```
#1 (RNG impl) ──┬──→ #2 (RNG notebook)
                 ├──→ #3 (RNG tests)
                 │
#4 (clone impl) ─┼──→ #5 (clone notebook)
                 │├──→ #6 (clone tests)
                 ││
                 ├┘
                 ▼
#7 (BaseScenario) ──→ #8 (Engine) ──→ #9 (Toy scenario)
                                       │
                              ┌────────┤
                              ▼        ▼
                      #10 (engine NB)  #11 (engine tests)
                              │
                              ▼
                      #12 (purity helper) ──→ #13 (failure demos) ──→ #14 (purity NB)
```

**Parallel entry points:** #1 and #4 can be worked simultaneously.

## Verification

After all tasks complete:
```bash
pytest tests/unit/test_rng.py tests/unit/test_state.py tests/unit/test_engine.py
pytest tests/integration/test_step_purity.py
jupyter nbconvert --execute notebooks/01_rng_partitioning.ipynb
jupyter nbconvert --execute notebooks/02_state_and_clone.ipynb
jupyter nbconvert --execute notebooks/03_discrete_time_engine.ipynb
jupyter nbconvert --execute notebooks/04_step_purity_contract.ipynb
flake8 sdk/core/
mypy sdk/core/
```
