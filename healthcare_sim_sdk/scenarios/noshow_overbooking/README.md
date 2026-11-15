# No-Show Overbooking Scenario

Clinic no-show prediction and overbooking policy evaluation. The question is whether ML-guided overbooking captures more revenue / throughput than uniform overbooking rules, and whether it does so without concentrating burden on specific demographics.

## Which runner to use?

| Runner | When to run it |
|---|---|
| [`run_evaluation.py`](run_evaluation.py) | **Default entry point.** Baseline vs. ML predictor across a threshold sweep, writes structured outputs under `outputs/`. Start here. |
| [`run_threshold_optimizer.py`](run_threshold_optimizer.py) | Grid sweep, defined in YAML, over (noshow_rate × utilization × model × AUC × threshold). Use when you want the full ops-facing grid for access operations. |
| [`evaluation_harness.py`](evaluation_harness.py) | **Library, not a CLI.** Imported by notebooks and other runners for sweep orchestration. `run_evaluation_sweep()`, `CLINIC_PROFILES`, `summarize_results()`. |
| [`run_burden_analysis.py`](run_burden_analysis.py) | Bioethics-review output: how is overbooking burden distributed across patients and demographics? Answers "are the same people overbooked repeatedly?" |
| [`run_governance_eval.py`](run_governance_eval.py) | AI-governance phase-1 validation: AUC / PPV / calibration / subgroup fairness against the governance framework's acceptance criteria. One-shot report. |

## Scenario implementations

Two `BaseScenario` subclasses live in this directory — they are not redundant, they trade off fidelity for speed:

| Class | File | Notes |
|---|---|---|
| `NoShowOverbookingScenario` | [`scenario.py`](scenario.py) | Basic version. Used by `evaluation_harness.py` for sweep speed. |
| `RealisticNoShowScenario` | [`realistic_scenario.py`](realistic_scenario.py) | Adds visit-frequency-weighted scheduling, accumulating waitlist, AR(1) drift. Used by `run_evaluation.py`, `run_burden_analysis.py`, and `run_governance_eval.py`. |

A consolidation decision (keep both vs. retire the basic one) is tracked in issue #28.

## Configuration

Every runner starts through Hydra and reads `configs/<runner>.yaml`: `evaluation.yaml`, `governance_eval.yaml`, `burden_analysis.yaml`, `threshold_optimizer.yaml`. No setting is typed into Python. Override with `key=value`, for example `run_evaluation n_days=90 model_auc=0.87`.

## Outputs

All runners write under `outputs/noshow_overbooking/<runner>/<timestamp>/` (gitignored). Each run ends in `experiments/lifecycle.finalize_experiment()`, which saves it, checks it and registers it in the catalog only if the checks pass.
