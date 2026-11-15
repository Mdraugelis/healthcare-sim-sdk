# <scenario name>

**Status:** template

Copy this folder to start a scenario, then replace every `<...>` below. `scripts/check_docs.py` fails while the copy still says `template`, lacks one of the five headings, or names a file that does not exist. Allowed statuses: reference, operational, calibration exercise, institution-specific.

## What it models
One or two sentences. Name the unit of analysis and say what `events` counts. As shipped, this folder is `CoinFlipScenario`: a biased coin per entity, the smallest scenario that meets the five-method contract.

## Entry points
- Config: `configs/example_sweep.yaml`. Conventions are in `configs/README.md`.
- Runner: `python -m healthcare_sim_sdk.scenarios.<name>.<runner>`. A runner lives in this folder, starts through `@hydra.main` with its settings in `configs/`, and ends with `finalize_experiment()`. The shape is in `healthcare_sim_sdk/experiments/runner.py`.
- Walkthrough: `examples/10_scenario_template.ipynb`.

## Verify
- Add the scenario to `SCENARIOS` in `tests/integration/test_all_scenario_contracts.py`. That runs the purity and branch-coupling checks on it.
- `pytest tests/integration/test_<name>.py`
- On the first run, do the five-level verification in `docs/simulation_protocol.md`, including boundary conditions.

## Private files
None | list each ignored file and say what fails without it.

## Known limits
- Two to four bullets. Link each to the document that explains it.
