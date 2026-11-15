# Analysis export semantics

`SimulationResults.to_analysis()` returns an `AnalysisDataset` with four exports: `to_time_series`, `to_panel`, `to_entity_snapshots` and `to_subgroup_panel`. This page states what each column means. An export that succeeds is analysis input; it does not show that an estimator identifies the effect you want.

## What changed from the previous behavior

| Area | Before | Now |
|---|---|---|
| Unrecorded timesteps in a time series | Reported as 0 | `NaN`, with a new `observed` column set to `False` |
| Recorded but empty outcome arrays | 0 | 0 (unchanged), unless the scenario sets `Outcomes.metadata["outcome_observed"] = False` |
| `get_outcome_series()` | Missing steps were 0 | Missing steps are `NaN` |
| Panel treatment | `treatment_definition="since_first"` or `"ever"` | Same names still work. New name `treatment_semantics`: `"ever_treated"` (same as `since_first`), `"ever"`, and new `"current_step"` |
| Time-series treatment indicator | Intervention called at that step | Same by default (`"scheduled"`); new `"any_treated"` requires at least one assigned entity |
| Snapshot scores | Taken positionally from that step's predictions | Matched to entities by ID; unmatched entities get `NaN`; counterfactual snapshots have no scores |
| Subgroup panel | Factual branch only | `branch=` accepted |
| Alignment errors | Some mismatches passed silently | Duplicate, missing or non-finite IDs, mismatched lengths, out-of-range or non-integer positions, and recorded timesteps outside the horizon raise `ValueError` |
| Unknown outcome key | `KeyError` | `UnknownOutcomeError`, which is both a `KeyError` and a `ValueError` |

No runner or report script on main calls these exports. The tracked reports cannot change. Tests and the example notebooks do call them; all pass.

## Outcome selection

`outcome_key="events"` selects `Outcomes.events`. Any other key selects that per-entity array in `Outcomes.secondary`. It must have the same length as `events`.

```python
analysis = results.to_analysis()
cost_series = analysis.to_time_series(outcome_key="cost")
cost_panel = analysis.to_panel(outcome_key="cost")
cost_snapshot = analysis.to_entity_snapshots(t=4, outcome_key="cost")
cost_by_group = analysis.to_subgroup_panel(
    outcome_key="cost", subgroup_key="race_ethnicity")
```

## Missing periods

A time series has one row per timestep in the horizon. A timestep with no recorded outcome is `NaN` with `observed=False`. Choose a missing-data policy before summing or averaging such a series; treating `NaN` as zero reproduces the old behavior and its bias.

Panel exports have one row per measured entity per recorded timestep, so missing periods add no rows.

## Entity identity and treatment

`Interventions.treated_indices` holds positions, not entity IDs. The exporter resolves them in this order:

1. `Interventions.metadata["treated_entity_ids"]`, when the scenario supplies it (used by the reminder scenario, whose outcomes lag predictions by a day).
2. `Predictions.metadata["entity_ids"]`, or the appointment scenarios' `"slot_ids"`.
3. The same step's factual outcome IDs, which must have the same length as the predictions.

IDs must be unique within a timestep, and stable across time for longitudinal use.

Panel `treated` records AI assignment only:

| Setting | Meaning |
|---|---|
| `"ever_treated"` (default; old name `since_first`) | Assigned at or before this row's timestep, from the entity's own first assignment |
| `"ever"` | Assigned at any time in the run, marked on every row |
| `"current_step"` | Assigned at exactly this row's timestep |

None of these says how long an effect lasts, whether the action was completed, or what usual care the entity received. All are zero on the counterfactual branch. In snapshot counterfactual mode the counterfactual can still carry effects of earlier factual treatment.
