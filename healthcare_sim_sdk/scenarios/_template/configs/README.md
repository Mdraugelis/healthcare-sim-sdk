# Experiment Configuration

This directory contains Hydra YAML configs for running experiments.

## Quick Start

1. Copy `example_sweep.yaml` and rename for your experiment
2. Update parameters for your scenario
3. Run: `python run_my_scenario.py --config-name=my_config`
4. Sweep: `python run_my_scenario.py --multirun param=val1,val2`

## Conventions

- All experiment configs use Hydra, not Python argparse or dataclasses
- Set `hydra.run.dir` to `outputs/${now:%Y%m%d_%H%M%S}`
- Outputs are gitignored — never commit simulation results
- Call `finalize_experiment()` at the end of your runner to save, validate, report, and register in one step
- Sweeps can be batch-registered: `python scripts/register_sweep.py outputs/sweep_*/`

## After Running

Results are written to `outputs/` (gitignored). To work with them:

- **List experiments:** `python -m healthcare_sim_sdk.experiments.report --list`
- **Validate:** `python -m healthcare_sim_sdk.experiments.validate <output_dir>`
- **Register sweep:** `python scripts/register_sweep.py outputs/sweep_<timestamp>/`
