# Analysis export semantics and inspectable traces

Exports preserve the selected outcome, entity identity, measurement timing, and AI assignment history. A successfully exported panel is analysis input; it does not establish that a chosen causal estimator identifies the desired effect.

## Outcome selection and missing periods

`outcome_key="events"` selects `Outcomes.events`. Any other key selects the per-entity array in `Outcomes.secondary`; missing keys, mismatched lengths, and invalid branches raise `ValueError`.

```python
analysis = results.to_analysis()
cost_series = analysis.to_time_series(outcome_key="cost")
cost_panel = analysis.to_panel(outcome_key="cost")
cost_snapshot = analysis.to_entity_snapshots(t=4, outcome_key="cost")
cost_by_group = analysis.to_subgroup_panel(
    outcome_key="cost", subgroup_key="race_ethnicity"
)
```

Time series include `observed`. Missing measurements are `NaN`, with `observed=False`. Recorded empty event arrays sum to zero unless the scenario explicitly sets `Outcomes.metadata["outcome_observed"] = False`; that marker requires an empty array and represents an unobserved measurement. Panel exports omit periods without entity observations. With no time configuration, the horizon is inferred from recorded timesteps.

This changes the old missing-period behavior, which silently reported zero outcomes. Consumers must choose an explicit missing-data policy instead of summing or averaging an incomplete series as though it were complete.

## Entity IDs, scores, and assignments

`Interventions.treated_indices` contains **positions**, not entity IDs. The exporter maps those positions using `Predictions.metadata["entity_ids"]`, or the appointment scenarios' existing `"slot_ids"` key. Without either, it uses same-step factual outcome IDs and requires lengths to agree. A scenario whose prediction and outcome populations differ must provide prediction entity IDs explicitly. IDs must be unique within a timestep and stable across time for longitudinal interpretation.

Snapshot scores match the same timestep's predictions by entity ID. A measured entity absent from those predictions receives `NaN` rather than another entity's score. Counterfactual snapshots contain no prediction scores because the engine does not predict on that branch.

Panel `treated` is an AI-assignment indicator with explicit semantics:

| Setting | Meaning |
| --- | --- |
| `treatment_semantics="ever_treated"` (default) | Assigned at or before this row's timestep, starting at **this entity's own first assignment** |
| `treatment_semantics="current_step"` | Assigned at exactly this row's timestep |

Neither setting infers biological effect duration, eligibility, action completion, or standard clinical care. Both return zero for counterfactual AI assignments. A snapshot counterfactual can still inherit effects of earlier factual treatment; zero assignments do not imply an untreated biological state in that mode.

Time-series `treatment_indicator` instead reports intervention calls by default (`"scheduled"`), including calls with no assigned entities. Use `"any_treated"` to require a nonempty intervention. The indicator is zero on the counterfactual branch.

## Appointment timing migration

The simple overbooking, realistic overbooking, and targeted reminder scenarios preserve the initial day-zero schedule for prediction and action. `step(0)` leaves those appointments unresolved. Day one resolves day zero; day two resolves day one. Measurements include only resolved slots.

This prevents initial day-zero appointment IDs from being generated and measured twice. Day-zero event arrays are empty and explicitly unobserved. An `N`-timestep run resolves `N-1` scheduled days; add a resolution timestep when outcomes for the final scheduled day are required. Cumulative resolution/no-show counters reconcile exactly to the raw measured appointments. A single patient's many appointments remain distinct entities because the unit of analysis is `appointment`.

## Trace the actual engine

`TracingScenario` observes normal scenario calls. It uses the existing five-method interface and delegates optional validation, cloning, and RNG assignment. The engine remains unaware of scenario state.

```python
from healthcare_sim_sdk.core.engine import BranchedSimulationEngine
from healthcare_sim_sdk.experiments.trace import TracingScenario

# The projector understands this scenario's state and selects a small sample.
wrapped = TracingScenario(
    fresh_scenario,
    projector=lambda state: {"selected_risks": state[3, selected_ids]},
    stages=("population", "intervene_before", "intervene_after", "measure_after"),
)
results = BranchedSimulationEngine(wrapped).run(n_entities)
trace_rows = wrapped.to_rows()
```

Projectors receive defensive copies and must be deterministic and free of side effects or random draws. Choose review samples with a separate reviewer RNG; record the selection rule, seed, candidate count, and unavailable categories. The default output records contain counts only. Supply `output_projector(kind, output)` to select detailed facts from actual prediction, intervention, or outcome outputs. State facts and output facts are serialized when recorded, so subsequent state changes cannot rewrite evidence.

Each immutable record has a run-local `trace_id`, timestep, branch, stage, and JSON facts. Store trace files beside the experiment manifest to make references unambiguous. The default cap of 10,000 records raises an error instead of silently truncating evidence. Each wrapper is single-use; use a fresh scenario for another run. In snapshot mode, measurements sharing a single RNG are conservatively labeled `snapshot_unresolved`; use branched mode for fully labeled paired stories.

The [sanity-check notebook](../examples/sanity_checks.ipynb) demonstrates sampled and mechanism-selected no-show/stroke stories, exact equality with fresh unobserved engine runs, and trace-to-outcome reconciliation. These public-default illustrations are not evidence for another experiment's configuration.

```sh
.venv/bin/python scripts/execute_notebook_headless.py
```

The command starts fresh Python execution, runs every cell in order, and writes figures/text to `reports/sanity_checks_executed.ipynb`. It supports ordinary Python cells without Jupyter sockets, magics, or widgets; the saved notebook explicitly records this execution method. Use Jupyter/nbconvert when front-end behavior also needs verification.

## Equity regression interpretation

The reminder equity regression no longer asserts that one seeded run with a higher configured AUC must narrow a racial no-show gap more than a low-AUC run. Aggregate discrimination does not determine which subgroup receives the limited calls, whether contact succeeds, or the realized subgroup event difference. A low-AUC scorer is also not an explicitly randomized allocation policy. Correcting appointment timing changed that single realization, exposing the unsupported test premise.

The replacement isolates an auditable mechanism: four synthetic patients have risks .8/.6 in one group and .4/.2 in the other; capacity is two calls, contact succeeds, and a reminder halves risk. Known-risk targeting reduces the expected group gap by .35. Uniformly choosing two of four has six equally likely allocations and reduces the expected gap by .10. These are configured synthetic assumptions, not properties intrinsic to demographic groups.

A second regression runs the actual reminder intervention, resolution, measurement, and branched engine across the prespecified seeds 0–511. It compares paired gap reductions against the hand-worked difference of .25 and reports the Monte Carlo mean, MCSE, and 95% t interval through JUnit properties. This verifies allocation and outcome mechanics under the declared fixture; it makes no general monotonicity claim about model AUC or clinical equity benefit.
