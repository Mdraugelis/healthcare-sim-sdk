# stroke_prevention

**Status:** reference

## What it models
Patients with time-varying stroke risk. A model flags high-risk patients for anticoagulant treatment, which lowers their risk. The unit is a patient and a timestep is a week. `events` counts strokes in that week.

## Entry points
- No runner and no Hydra config. Build `StrokeConfig` and `StrokePreventionScenario` in code, as `examples/08_stroke_scenario_integration.ipynb` does.
- Defaults: AUC 0.75, and PPV held at 3.7 times the window base rate (`target_lift`).

## Verify
- `pytest tests/integration/test_stroke.py`
- `pytest tests/integration/test_all_scenario_contracts.py -k stroke`
- `pytest tests/bulletproof/test_conservation_laws.py tests/bulletproof/test_boundary_conditions.py`

## Private files
None.

## Known limits
- The model is fitted once per run on a reference cohort, which takes a few seconds. With too few events it raises; `allow_infeasible_model=True` runs with nobody flagged and says so in `model_status`.
- The legacy `target_ppv` and `target_sensitivity` pair needs an AUC near 0.98 and raises unless explicitly allowed. See `docs/algorithm_description.md` section 9.6.
- Treatment starts once per patient and persists. A stroke ends follow-up, so nobody has two.
- The number of patients treated varies widely across seeds, because the threshold is fitted per run. Section 5.7 of the same document has the numbers.
