# Development Plan: Incremental Proof Points for SDK Invariants

Following the same pattern as [pop-ml-simulator/notebooks](https://github.com/Mdraugelis/pop-ml-simulator/tree/main/notebooks) — each notebook builds one piece of the SDK, tests it in isolation, then produces a working `.py` module that lands in the `sdk/` package.

## The 7 Invariants

| # | Invariant | First Proved In |
|---|-----------|-----------------|
| 1 | Reproducibility Infrastructure (RNG Partitioning) | NB01 |
| 2 | Discrete-Time Engine | NB03 |
| 3 | ML Model Interface | NB05, NB06 |
| 4 | Branched Counterfactual Engine | NB03, NB04 |
| 5 | State Management | NB02 |
| 6 | Analysis-Ready Data Interface | NB07 |
| 7 | Experiment Management | NB13 |

---

## Phase 1: Core SDK (Weeks 1-5)

### NB01 — `01_rng_partitioning.ipynb`

**Proves:** Invariant #1 — Reproducibility Infrastructure
**Builds:** `sdk/core/rng.py` (RNGPartitioner, RNGStreams)
**Depends on:** Nothing — this is the root

Demonstrate that `SeedSequence.spawn()` produces statistically independent child generators across the 5 named streams (population, temporal, prediction, intervention, outcomes).

**Tests:**
- Same master seed → identical stream outputs across runs
- Consuming one stream does not affect another: draw 1000 values from `intervention` between two `temporal` draws; `temporal` output is unchanged
- `fork()` produces a new partitioner whose streams match the original when consumed identically
- Cross-stream correlation matrix shows zero correlation

**Key assertion:**
```python
# Stream independence under asymmetric consumption
draws_a = partitioner_a.streams.temporal  # also consumed 1000 intervention draws
draws_b = partitioner_b.streams.temporal  # consumed zero intervention draws
assert np.array_equal(draws_a, draws_b)
```

This is the foundation. If streams aren't independent, branched counterfactuals are meaningless.

---

### NB02 — `02_state_and_clone.ipynb`

**Proves:** Invariant #5 — State Management
**Builds:** Clone utilities in `sdk/core/scenario.py`
**Depends on:** Nothing (independent of NB01)

Demonstrate that `clone_state` produces mutation-isolated copies for any state type the SDK must support.

**Tests:**
- Deep copy correctness on numpy arrays, dataclasses, nested dicts, mixed structures
- Mutating clone does not affect original
- Memory profiling: `deepcopy` vs `np.copy` at 10k, 50k, 100k entities
- Custom `clone_state()` override pattern for optimized copying

**Key assertion:**
```python
cloned = clone_state(original)
cloned.risks[0] = 999.0
assert original.risks[0] != 999.0  # mutation isolation
```

---

### NB03 — `03_discrete_time_engine.ipynb`

**Proves:** Invariants #2 (Engine) and #4 (Branched Counterfactuals)
**Builds:** `sdk/core/engine.py`, `sdk/core/scenario.py` (BaseScenario, TimeConfig, Predictions, Interventions, Outcomes)
**Depends on:** NB01 (RNGPartitioner), NB02 (clone_state)

Use a **trivial toy scenario** (counter that increments each step; predict returns the counter; intervene adds 10; measure records it) to isolate engine behavior from domain complexity.

**Tests:**
- Hook ordering: step → predict → intervene → measure, verified via method call log
- Clock: `t` advances 0 to `n_timesteps - 1`; `prediction_schedule` controls which steps trigger predict/intervene
- `CounterfactualMode.NONE`: single trajectory
- `CounterfactualMode.BRANCHED`: two trajectories sharing step() physics, diverging at intervene()
- `CounterfactualMode.SNAPSHOT`: same-step counterfactual snapshots only
- Engine swaps `scenario.rng` between factual and counterfactual stream sets at each step

**Key assertion:**
```python
# Toy counter scenario, BRANCHED mode, predictions at t=5
# Factual:        10 steps + 1 intervention × 10 = 20
# Counterfactual: 10 steps + 0 interventions     = 10
assert factual_counter == 20
assert counterfactual_counter == 10
```

---

### NB04 — `04_step_purity_contract.ipynb`

**Proves:** Cross-cutting validation of #1, #2, #4 together
**Builds:** `tests/integration/test_step_purity.py`
**Depends on:** NB01, NB02, NB03

This notebook introduces **the single most important test in the SDK**: run the same scenario with `BRANCHED` and `NONE` modes, same seed. The factual branch from BRANCHED must be identical to the NONE trajectory.

**Tests:**
- Stochastic scenario (random draws in step) passes the BRANCHED-vs-NONE identity test
- Deliberately impure scenario (step reads shared `self.counter`) desynchronizes — demonstrate the failure
- RNG discipline violation (step draws from `self.rng.intervention` instead of `self.rng.temporal`) — demonstrate the failure

**Key assertion:**
```python
results_branched = engine.run(scenario, mode=BRANCHED, seed=42)
results_none     = engine.run(scenario, mode=NONE, seed=42)
assert np.array_equal(
    results_branched.factual_outcomes,
    results_none.outcomes
)
```

If this passes, it simultaneously proves: RNG partitioning works, stream forking works, step purity holds, the engine swaps RNG contexts correctly, and clone_state produces true independent copies.

---

### NB05 — `05_ml_binary_classifier.ipynb`

**Proves:** Invariant #3 — ML Model Interface (binary classification)
**Builds:** `sdk/ml/binary_classifier.py`, `sdk/ml/performance.py`
**Depends on:** NB01 (prediction RNG stream)

Extract `ControlledBinaryClassifier` from the existing [MLPredictionSimulator](https://github.com/Mdraugelis/pop-ml-simulator/blob/main/src/pop_ml_simulator/ml_simulation.py). Given known true labels and risk scores, generate predictions that hit target PPV and sensitivity.

**Tests:**
- `optimize()` finds noise parameters to achieve target performance
- Measured PPV and sensitivity within 5% relative error of targets
- Changing prediction seed changes predictions but not temporal evolution
- Performance utilities: confusion matrix, ROC, calibration

**Key assertion:**
```python
classifier = ControlledBinaryClassifier(target_ppv=0.15, target_sensitivity=0.80)
scores, labels = classifier.predict(true_labels, risk_scores)
assert abs(measured_ppv - 0.15) / 0.15 < 0.05
assert abs(measured_sensitivity - 0.80) / 0.80 < 0.05
```

---

### NB06 — `06_ml_probability_model.ipynb`

**Proves:** Invariant #3 — ML Model Interface (probability estimation)
**Builds:** `sdk/ml/probability_model.py`
**Depends on:** NB01, NB05 (performance utilities)

Build `ControlledProbabilityModel` — targets AUC and calibration slope rather than PPV/sensitivity at a threshold. This is the ML utility the no-show scenario needs.

**Tests:**
- Given true probabilities, generate predictions hitting target AUC and calibration slope
- Hosmer-Lemeshow test passes (p > 0.05)
- Same interface pattern works for both stroke-like and no-show-like domains

**Key assertion:**
```python
model = ControlledProbabilityModel(target_auc=0.78, target_calibration_slope=1.0)
predictions = model.predict(true_probabilities)
assert abs(measured_auc - 0.78) < 0.02
assert abs(measured_calibration_slope - 1.0) < 0.1
```

---

### NB07 — `07_results_and_analysis_dataset.ipynb`

**Proves:** Invariant #6 — Analysis-Ready Data Interface
**Builds:** `sdk/core/results.py` (SimulationResults, AnalysisDataset)
**Depends on:** NB03 (engine produces SimulationResults)

Run the toy scenario from NB03 through the engine and validate all 4 export methods.

**Tests:**
- `to_time_series()` → `{"timesteps", "outcomes", "treatment_indicator"}` with correct shapes
- `to_panel()` → entity-level panel, shape `(n_entities × n_timesteps,)`; entity_ids, timesteps, outcomes, treated arrays all correct
- `to_entity_snapshots(t)` → cross-sectional data at single timestep
- `to_subgroup_panel()` → extends panel with group labels
- Both `branch="factual"` and `branch="counterfactual"` work
- Error when `entity_ids` is None and `to_panel()` is called

**Key assertion:**
```python
panel = analysis.to_panel()
assert len(panel["entity_ids"]) == n_entities * n_timesteps
assert set(panel["entity_ids"]) == set(range(n_entities))
```

---

### NB08 — `08_stroke_scenario_integration.ipynb`

**Proves:** ALL Phase 1 invariants composed together
**Builds:** `sdk/population/risk_distributions.py`, `sdk/population/temporal_dynamics.py`, `scenarios/stroke_prevention/scenario.py`
**Depends on:** NB01–NB07

The full integration test. Implement `StrokePreventionScenario` using every SDK component. Extract `beta_distributed_risks` and `AR1Process` from the existing pop-ml-simulator, wiring them to partitioned RNG streams.

**Tests:**
- 10,000 patients, 52 weeks, branched mode — runs end-to-end
- **The purity test (NB04) on the real stroke scenario** — BRANCHED factual == NONE trajectory
- All 4 AnalysisDataset exports produce valid data
- Factual incident rate < counterfactual incident rate (intervention works)
- Population incident rate matches `annual_incident_rate` within statistical tolerance

**Key assertions:**
```python
# Purity
assert np.array_equal(branched_factual_outcomes, none_outcomes)
# Intervention effect
assert factual_total_incidents < counterfactual_total_incidents
```

---

## Phase 2: Second Scenario Validation (Weeks 6-9)

### NB09 — `09_noshow_scenario.ipynb`

**Proves:** SDK generality — same engine works for a fundamentally different domain
**Builds:** `scenarios/noshow_overbooking/scenario.py`
**Depends on:** NB01–NB07, NB06 (ControlledProbabilityModel)

The acid test for generality. NoShowOverbookingScenario has a completely different state type (appointment slots + patient histories vs scalar risk arrays), `unit_of_analysis = "appointment"`, custom `clone_state()`, and multiple outcome dimensions (show/no-show, collision rate, wait time, overbooking burden).

**Tests:**
- The same `engine.run()` call works without engine modifications
- Purity test passes: BRANCHED factual == NONE
- Compounding effects visible: factual branch shows increasing per-patient overbooking burden over time; counterfactual shows zero
- `to_subgroup_panel()` produces valid equity analysis data

**Key assertions:**
```python
# Same engine, different domain
assert np.array_equal(branched_factual_outcomes, none_outcomes)
# Compounding
assert factual_overbooking_burden[-1] > factual_overbooking_burden[0]
assert counterfactual_overbooking_burden.sum() == 0
```

---

### NB10 — `10_scenario_template.ipynb`

**Proves:** SDK usability — the 5-method contract is implementable from a template
**Builds:** `scenarios/_template/scenario.py`
**Depends on:** NB08, NB09

Walk through creating a minimal scenario from the template (e.g., "coin flip with bias" where intervention reduces flip probability). Demonstrates the implementation checklist: define state type, implement 5 methods, declare `unit_of_analysis`, use correct RNG streams.

**Key assertion:** Template scenario passes BRANCHED-vs-NONE purity test out of the box.

---

## Phase 3: Analysis + Experiments (Weeks 10-12)

### NB11 — `11_its_analysis.ipynb`

**Proves:** Invariant #6 in practice — ITS workflow end-to-end
**Builds:** `sdk/analysis/its.py`
**Depends on:** NB07 (AnalysisDataset), NB08 (stroke data)

Implement segmented regression on `to_time_series()` output. Detect the intervention effect as a level change and/or slope change. Show same ITS code works on both stroke and no-show data.

**Key assertion:** ITS model detects statistically significant level change at intervention point, with effect size consistent with configured `intervention_effectiveness`.

---

### NB12 — `12_panel_and_equity_exports.ipynb`

**Proves:** Invariant #6 — DiD, RDD, and equity analysis workflows
**Builds:** `sdk/analysis/panel_utils.py`
**Depends on:** NB07, NB08, NB09

Demonstrate the "hand-off" pattern: SDK produces analysis-ready data, analyst uses external packages (`statsmodels`, `linearmodels`, `rdrobust`) for estimation.

**Key assertion:** `to_panel()` output fed to a basic DiD estimator recovers the configured intervention effect within a 95% CI.

---

### NB13 — `13_hydra_experiment_management.ipynb`

**Proves:** Invariant #7 — Experiment Management
**Builds:** `sdk/config/base_config.py`, `sdk/config/hydra_utils.py`
**Depends on:** NB08 (stroke scenario with config)

Parameter sweeps via Hydra: vary `intervention_effectiveness` from 0.1 to 0.5, run each, compare. Seed sweeps to measure variance.

**Key assertion:** Effectiveness sweep produces monotonically increasing lives saved. Seed sweep variance is consistent with expected statistical variation, not RNG artifacts.

---

## Phase 4: Documentation (Weeks 13-14)

### NB14 — `14_build_your_first_scenario.ipynb`

**Proves:** Documentation completeness — a reader can follow the tutorial and build a working scenario
**Builds:** Tutorial only (e.g., a hospital readmission scenario as teaching example)
**Depends on:** All prior notebooks

Step-by-step walkthrough referencing which earlier notebook proved each concept. Includes RNG discipline checklist, step purity checklist, and the BRANCHED-vs-NONE regression test.

---

## The One Test That Rules Them All

The step purity regression test (NB04) is repeated in NB08, NB09, and NB10. It is the single most important validation:

> Run the same scenario with `CounterfactualMode.BRANCHED` and `CounterfactualMode.NONE`, same seed. The factual branch from BRANCHED must be element-wise identical to the NONE trajectory.

If this passes, it simultaneously proves:
- RNG partitioning (streams are independent)
- Stream forking (branches get the same seed state)
- Step purity (no shared mutable state leaks)
- Engine RNG context swapping
- clone_state mutation isolation

If this fails, the simulation's causal semantics are broken.

---

## Module Output Map

| Notebook | SDK Module(s) Produced |
|----------|----------------------|
| NB01 | `sdk/core/rng.py` |
| NB02 | Clone utilities in `sdk/core/scenario.py` |
| NB03 | `sdk/core/engine.py`, `sdk/core/scenario.py` |
| NB04 | `tests/integration/test_step_purity.py` |
| NB05 | `sdk/ml/binary_classifier.py`, `sdk/ml/performance.py` |
| NB06 | `sdk/ml/probability_model.py` |
| NB07 | `sdk/core/results.py` |
| NB08 | `sdk/population/risk_distributions.py`, `sdk/population/temporal_dynamics.py`, `scenarios/stroke_prevention/scenario.py` |
| NB09 | `scenarios/noshow_overbooking/scenario.py` |
| NB10 | `scenarios/_template/scenario.py` |
| NB11 | `sdk/analysis/its.py` |
| NB12 | `sdk/analysis/panel_utils.py` |
| NB13 | `sdk/config/base_config.py`, `sdk/config/hydra_utils.py` |
| NB14 | Documentation only |

## Dependency Graph

```
NB01 (RNG) ──────────────┬──────────────────────────────┐
                          │                              │
NB02 (State) ────────┐   │                              │
                      │   │                              │
NB03 (Engine) ◄───────┴───┘                              │
      │                                                  │
NB04 (Purity) ◄── NB03                                  │
                                                         │
NB05 (Binary ML) ◄──────────────────────────────────────┘
      │
NB06 (Prob ML) ◄── NB05

NB07 (Results) ◄── NB03

NB08 (Stroke) ◄── NB01–NB07     ← Phase 1 integration

NB09 (No-Show) ◄── NB01–NB07    ← Phase 2 acid test

NB10 (Template) ◄── NB08, NB09  ← Usability proof

NB11 (ITS) ◄── NB07, NB08       ← Phase 3 analysis
NB12 (Panel) ◄── NB07–NB09
NB13 (Hydra) ◄── NB08

NB14 (Tutorial) ◄── ALL         ← Phase 4 onboarding
```
