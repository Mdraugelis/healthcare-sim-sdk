# Implementation Task: Unify Experiment Catalog with Hydra Output Structure

| Field | Value |
|-------|-------|
| Repo | healthcare-sim-sdk |
| Target branch | `feature/experiment-unification` |
| Priority | Medium — improves developer experience and reproducibility |
| Estimated effort | 4–6 hours |
| Depends on | `feature/invariant-guardrails` (gitignore patterns) |
| Created | 2026-04-04 |

---

## Problem Statement

The SDK currently has **two independent experiment systems** that don't talk to each other:

**System 1: ExperimentCatalog** (`experiments/catalog.py`, `experiments/report.py`, `experiments/validate.py`). A JSON-based index of past runs. Stores summary metadata in `catalog.json` and points to output directories containing `config.json`, `results.json`, `summary.json`. Used by `run_evaluation.py` and `run_burden_analysis.py`, which manually register results after each run. The catalog, report generator, and validator all assume a specific output directory layout.

**System 2: Hydra** (`run_threshold_optimizer.py` + `configs/threshold_optimizer.yaml`). Manages experiment configuration, parameter sweeps, and output directories. Writes to `outputs/{timestamp}/` for single runs and `outputs/sweep_{timestamp}/{param_combo}/` for `--multirun` sweeps. Each Hydra output directory gets a `.hydra/` subdirectory with the resolved config, overrides, and Hydra's own metadata. The threshold optimizer writes `metrics.json` to Hydra's output dir but **never calls `ExperimentCatalog.register()`**.

The result:

- Hydra runs are invisible to the catalog, report generator, and validator
- The older argparse runners (`run_evaluation.py`, `run_burden_analysis.py`, `run_governance_eval.py`) do register with the catalog but don't use Hydra's config management
- There's no standard for what a "completed experiment" looks like across runner types
- The `validate.py` module hardcodes no-show scenario internals and can't work with Hydra's output layout
- A user running `--multirun` sweeps gets 189 output directories with no way to catalog or compare them without writing custom scripts

This task unifies the two systems so there is **one experiment lifecycle** that works for both single runs and Hydra sweeps.

---

## Design Decisions

### Hydra is the configuration layer

All experiment configuration should flow through Hydra YAML configs. The existing `ExperimentConfig` and `BurdenConfig` dataclasses become Hydra structured configs or are replaced by YAML files. This aligns with the CLAUDE.md convention: "Experiment configuration uses Hydra (YAML configs, not Python dataclasses). Never hardcode sweep parameters in Python loops."

### ExperimentCatalog is the persistence layer

The catalog remains the system of record for completed experiments. But it needs to understand Hydra's output structure and support batch registration from sweeps.

### The standard experiment output directory

Every completed experiment — whether from a single run, a Hydra `--multirun` sweep cell, or an argparse runner — produces a directory with this layout:

```
outputs/{timestamp}/
├── .hydra/                  # Hydra auto-generated (config snapshot, overrides)
├── config.json              # Resolved experiment config (SDK-written)
├── metrics.json             # Key metrics (SDK-written)
├── results.json             # Full results (optional, for detailed analysis)
├── report.md                # Generated markdown report (optional)
├── validation_appendix.md   # Validation results (optional)
└── experiment.log           # Run log
```

For `--multirun` sweeps, each subdirectory follows this layout, and a top-level `sweep_summary.json` aggregates across all cells.

---

## Work Items

### 1. Create `experiments/lifecycle.py` — the glue layer

This module provides the standard post-simulation lifecycle that any runner calls after completing a simulation. It replaces the ad-hoc save/register/report/validate logic currently duplicated across `run_evaluation.py`, `run_burden_analysis.py`, and `run_governance_eval.py`.

**Functions to implement:**

```python
def save_experiment(
    output_dir: Path,
    config: dict,
    metrics: dict,
    results: dict | None = None,
    notes: str = "",
) -> Path:
    """Save experiment artifacts to output_dir.

    Writes:
      - config.json (the resolved config)
      - metrics.json (key metrics for catalog/comparison)
      - results.json (optional full results)

    Returns output_dir for chaining.
    """

def register_experiment(
    output_dir: Path,
    catalog: ExperimentCatalog | None = None,
) -> None:
    """Register a completed experiment in the catalog.

    Reads config.json and metrics.json from output_dir.
    Idempotent — re-registering an existing timestamp updates it.
    """

def register_sweep(
    sweep_dir: Path,
    catalog: ExperimentCatalog | None = None,
) -> dict:
    """Register all cells of a Hydra --multirun sweep.

    Walks sweep_dir for subdirectories containing metrics.json.
    Registers each cell individually.
    Returns a sweep_summary dict aggregating across cells.
    Writes sweep_summary.json to sweep_dir.
    """

def finalize_experiment(
    output_dir: Path,
    config: dict,
    metrics: dict,
    results: dict | None = None,
    validate: bool = True,
    report: bool = True,
    notes: str = "",
) -> Path:
    """Full lifecycle: save → validate → report → register.

    This is the recommended single call for runners to use.
    Handles everything after the simulation is complete.
    """
```

**Why a separate module instead of adding to catalog.py:** The catalog is a pure persistence layer (read/write JSON index). The lifecycle module orchestrates save, validate, report, and register — it depends on the catalog, not the other way around.

### 2. Refactor `ExperimentCatalog` to support Hydra outputs

The current catalog stores noshow-specific fields (`model_auc`, `baseline_auc`, `collision_rate_reduction`, etc.). This needs to become generic so it works for any scenario.

**Changes to `experiments/catalog.py`:**

```python
class ExperimentCatalog:
    def register(
        self,
        output_dir: Path,
        config: dict,
        metrics: dict,
        notes: str = "",
    ) -> None:
        """Register a completed experiment.

        Stores a standard set of fields plus all of metrics
        as a nested dict. This way any scenario can store
        its domain-specific metrics without changing the schema.
        """
        entry = {
            "timestamp": config.get("timestamp", ""),
            "experiment_name": config.get("experiment_name", ""),
            "scenario": config.get("scenario", "unknown"),
            "output_dir": str(output_dir),
            "seed": config.get("seed"),
            "n_entities": config.get("n_patients", config.get("n_entities")),
            "n_timesteps": config.get("n_days", config.get("n_timesteps")),
            "metrics": metrics,  # <-- store all metrics as nested dict
            "notes": notes,
            "registered_at": datetime.now().isoformat(),
        }
        # ... rest of registration logic
```

**Add a `list_by_scenario()` method** for filtering:

```python
def list_by_scenario(self, scenario: str) -> list[dict]:
    """List experiments for a specific scenario."""
    return [e for e in self._entries if e.get("scenario") == scenario]
```

**Add a `to_dataframe()` method** for analysis (optional, but useful):

```python
def to_dataframe(self) -> "pd.DataFrame":
    """Export catalog as a pandas DataFrame, flattening metrics."""
    # Flattens nested metrics dict into columns
    # e.g., metrics.utilization, metrics.auc, etc.
```

### 3. Wire the Hydra threshold optimizer to the lifecycle

Update `run_threshold_optimizer.py` to call `finalize_experiment()` (or at minimum `save_experiment()` + `register_experiment()`) at the end of each run.

**Current code (line 213-216):**
```python
out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
with open(out_dir / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
```

**Should become:**
```python
from healthcare_sim_sdk.experiments.lifecycle import save_experiment, register_experiment

out_dir = Path(HydraConfig.get().runtime.output_dir)
config_dict = OmegaConf.to_container(cfg, resolve=True)
config_dict["timestamp"] = out_dir.name  # Hydra timestamp dir name
config_dict["scenario"] = "noshow_overbooking"
config_dict["experiment_name"] = "threshold_optimizer"

save_experiment(out_dir, config_dict, metrics)
register_experiment(out_dir)
```

**Also add a Hydra callback or post-sweep script** for `--multirun`:

```python
# scripts/register_sweep.py
"""Register all cells of a completed Hydra sweep."""
from pathlib import Path
import sys
from healthcare_sim_sdk.experiments.lifecycle import register_sweep

if __name__ == "__main__":
    sweep_dir = Path(sys.argv[1])
    summary = register_sweep(sweep_dir)
    print(f"Registered {summary['n_cells']} cells from {sweep_dir}")
```

Usage after a sweep completes:
```bash
python scripts/register_sweep.py outputs/sweep_20260404_120000/
```

### 4. Make `validate.py` scenario-agnostic

The current `validate.py` is hardcoded to the noshow overbooking scenario — it imports `RealisticNoShowScenario`, `ClinicConfig`, `RACE_ETHNICITY`, etc. This needs to be refactored so validation can work for any scenario.

**Approach: split into generic + scenario-specific validators**

```
experiments/
├── validate.py              # Generic framework (structural, statistical)
├── catalog.py               # Persistence (unchanged interface)
├── lifecycle.py             # NEW: orchestration layer
└── report.py                # Report generation

scenarios/noshow_overbooking/
├── validate_noshow.py       # Noshow-specific checks (demographics, AR(1), clinic)
├── ...
```

**Generic `validate.py` should check:**
- config.json exists and is valid JSON
- metrics.json exists and is valid JSON
- Seed is recorded (reproducibility)
- Timestamp is recorded
- Output directory structure is complete

**Scenario-specific validators** implement a `validate(output_dir) -> list[Check]` function that the generic validator discovers and calls. The discovery mechanism can be simple: look for a `validate_{scenario_name}.py` file in the scenario directory, or have the scenario class expose a `get_validator()` method.

This is the largest refactor in this task. Keep the existing 11-category check logic intact — just move it from `experiments/validate.py` to `scenarios/noshow_overbooking/validate_noshow.py` and make it callable through the generic interface.

### 5. Refactor argparse runners to use the lifecycle

Update the three existing runners to use the lifecycle module instead of duplicating save/register/report logic:

**`run_evaluation.py`** — currently has its own `save_results()`, manual catalog registration, manual report generation, and manual validation. Replace lines 250-453 with:

```python
from healthcare_sim_sdk.experiments.lifecycle import finalize_experiment

# After run_experiment() completes:
finalize_experiment(
    output_dir=output_dir,
    config=asdict(config),
    metrics=experiment["summary"],
    results=experiment["results"],
    validate=True,
    report=True,
)
```

**`run_burden_analysis.py`** — same pattern. Replace the manual save/register/report block (lines 460-510).

**`run_governance_eval.py`** — same pattern.

The per-runner `save_results()` functions and inline catalog registration code can be removed.

### 6. Migrate argparse runners to Hydra configs

This is the final step: convert the argparse-based runners to Hydra-based configuration so there's one config approach across the SDK.

**Create config YAML files:**

```
scenarios/noshow_overbooking/configs/
├── threshold_optimizer.yaml    # Already exists
├── evaluation.yaml             # NEW: replaces ExperimentConfig dataclass
├── burden_analysis.yaml        # NEW: replaces BurdenConfig dataclass
└── governance_eval.yaml        # NEW: replaces GovernanceConfig dataclass
```

**Example: `evaluation.yaml`**

```yaml
# Structured evaluation: control + baseline + predictor sweep
#
# Single run:
#   python run_evaluation.py
#
# Override:
#   python run_evaluation.py model.auc=0.87 n_days=90

seed: 42
n_patients: 2000
n_days: 60

scenario: noshow_overbooking
experiment_name: noshow_evaluation

clinic:
  noshow_rate: 0.13
  n_providers: 6
  slots_per_provider: 12
  max_overbook_per_provider: 2
  waitlist_requests_per_day: 5

model:
  auc: 0.83
  baseline_threshold: 0.50

policy:
  max_individual_overbooks: 10
  overbooking_policy: threshold
  predictor_thresholds: [0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
  ar1_rho: 0.95
  ar1_sigma: 0.04

hydra:
  run:
    dir: outputs/${now:%Y%m%d_%H%M%S}
```

**Note:** The `ExperimentConfig` and `BurdenConfig` dataclasses should remain in the runner files as Hydra structured configs (using `@dataclass` with `ConfigStore`) or be removed entirely in favor of pure YAML. The choice depends on whether the runners need runtime config validation beyond what Hydra provides. Recommend starting with pure YAML + OmegaConf for simplicity.

### 7. Create a reference example in the template scenario

The `_template/` scenario should include a minimal Hydra config example that new scenario developers can copy. This is the "example that demonstrates Hydra test structure" that stays in the repo.

**Create `scenarios/_template/configs/example_sweep.yaml`:**

```yaml
# Example Hydra experiment configuration
# ========================================
# Copy this to your scenario's configs/ directory and customize.
#
# Single run:
#   python run_my_scenario.py
#
# Parameter sweep:
#   python run_my_scenario.py --multirun \
#     model.auc=0.65,0.75,0.85 \
#     policy.threshold=0.20,0.30,0.40

# -- Required fields (every experiment must have these) --
seed: 42
scenario: my_scenario_name
experiment_name: my_experiment

# -- Simulation parameters --
n_entities: 1000
n_timesteps: 52

# -- Model configuration --
model:
  auc: 0.80
  threshold: 0.30

# -- Policy configuration --
policy:
  # Your scenario-specific policy parameters here
  effectiveness: 0.50

# -- Hydra output configuration --
# These settings ensure outputs are organized and gitignored.
hydra:
  run:
    dir: outputs/${now:%Y%m%d_%H%M%S}
  sweep:
    dir: outputs/sweep_${now:%Y%m%d_%H%M%S}
    subdir: ${model.auc}_${policy.threshold}
```

**Create `scenarios/_template/configs/README.md`:**

```markdown
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
- Call `finalize_experiment()` at the end of your runner to
  save, validate, report, and register in one step
- Sweeps can be batch-registered: `python scripts/register_sweep.py outputs/sweep_*/`

## After Running

Results are written to `outputs/` (gitignored). To work with them:

- **List experiments:** `python -m healthcare_sim_sdk.experiments.report --list`
- **View report:** `python -m healthcare_sim_sdk.experiments.report <timestamp>`
- **Validate:** `python -m healthcare_sim_sdk.experiments.validate <timestamp>`
- **Register sweep:** `python scripts/register_sweep.py outputs/sweep_<timestamp>/`
- **Compare:** `python -m healthcare_sim_sdk.experiments.report --compare ts1 ts2`
```

### 8. Update CLAUDE.md and sim-guide.md

**Add to CLAUDE.md under Conventions:**

```markdown
- **Experiment lifecycle**: All runners call `finalize_experiment()` from `experiments/lifecycle.py` after simulation completes. This saves config and metrics, runs validation, generates reports, and registers in the experiment catalog. For Hydra `--multirun` sweeps, run `scripts/register_sweep.py` after the sweep completes.
- **Experiment outputs are ephemeral**: The `outputs/` directory is gitignored at all depths. Never commit simulation results. The repo contains only Hydra config YAML files (the experiment *structure*) and Python runner scripts (the experiment *logic*).
```

**Add to sim-guide.md, in the "Reference: SDK Validation Infrastructure" section:**

```markdown
**`experiments/lifecycle.py`** — `finalize_experiment(output_dir, config, metrics)` is the standard post-simulation call. Handles save, validate, report, and catalog registration. For sweeps, `register_sweep(sweep_dir)` batch-registers all cells.

**`scripts/register_sweep.py`** — CLI tool to register a completed Hydra `--multirun` sweep in the experiment catalog. Run after sweep completes: `python scripts/register_sweep.py outputs/sweep_<timestamp>/`.
```

**Add to sim-guide.md, in the Phase 4 "Suggesting Next Steps" section**, a note about parameter sweeps:

```markdown
When suggesting parameter sweeps, guide the human to define the sweep grid in a Hydra YAML config (not a Python loop) and use `--multirun`. After the sweep, use `register_sweep()` to catalog all cells, then compare results via the catalog.
```

---

## Implementation Order

The work items have dependencies. Recommended sequence:

```
1. experiments/lifecycle.py        (no dependencies, new module)
    ↓
2. Refactor ExperimentCatalog      (lifecycle.py uses catalog)
    ↓
3. Wire threshold optimizer        (uses lifecycle.py)
    ↓
4. Refactor validate.py            (lifecycle calls validate)
    ↓
5. Refactor argparse runners       (uses lifecycle.py)
    ↓
6. Migrate to Hydra configs        (replaces argparse in runners)
    ↓
7. Template example                (reference for new scenarios)
    ↓
8. Update CLAUDE.md + sim-guide    (documentation)
```

Items 1-3 can ship as one PR. Items 4-6 are a second PR. Items 7-8 can accompany either.

---

## File Inventory

| Action | File | Notes |
|--------|------|-------|
| CREATE | `healthcare_sim_sdk/experiments/lifecycle.py` | Orchestration layer (Item 1) |
| CREATE | `scripts/register_sweep.py` | CLI for batch sweep registration (Item 3) |
| CREATE | `scenarios/_template/configs/example_sweep.yaml` | Reference Hydra config (Item 7) |
| CREATE | `scenarios/_template/configs/README.md` | Config conventions doc (Item 7) |
| CREATE | `scenarios/noshow_overbooking/configs/evaluation.yaml` | Replaces ExperimentConfig (Item 6) |
| CREATE | `scenarios/noshow_overbooking/configs/burden_analysis.yaml` | Replaces BurdenConfig (Item 6) |
| CREATE | `scenarios/noshow_overbooking/configs/governance_eval.yaml` | Replaces GovernanceConfig (Item 6) |
| CREATE | `scenarios/noshow_overbooking/validate_noshow.py` | Scenario-specific validation (Item 4) |
| MODIFY | `healthcare_sim_sdk/experiments/catalog.py` | Generic metrics storage (Item 2) |
| MODIFY | `healthcare_sim_sdk/experiments/validate.py` | Extract generic framework (Item 4) |
| MODIFY | `scenarios/noshow_overbooking/run_threshold_optimizer.py` | Wire to lifecycle (Item 3) |
| MODIFY | `scenarios/noshow_overbooking/run_evaluation.py` | Wire to lifecycle + Hydra (Items 5-6) |
| MODIFY | `scenarios/noshow_overbooking/run_burden_analysis.py` | Wire to lifecycle + Hydra (Items 5-6) |
| MODIFY | `scenarios/noshow_overbooking/run_governance_eval.py` | Wire to lifecycle + Hydra (Items 5-6) |
| MODIFY | `CLAUDE.md` | Experiment conventions (Item 8) |
| MODIFY | `.claude/agents/sim-guide.md` | Lifecycle reference (Item 8) |

---

## Acceptance Criteria

### Core lifecycle (Items 1-3)

1. `finalize_experiment()` saves `config.json`, `metrics.json`, and registers in catalog — verified by round-tripping a dummy experiment
2. `register_sweep()` walks a Hydra `--multirun` output tree and registers every cell with `metrics.json`
3. Running `python run_threshold_optimizer.py` produces an output directory that appears in `catalog.json` — verified by `ExperimentCatalog().find(timestamp)`
4. Running `python run_threshold_optimizer.py --multirun` followed by `python scripts/register_sweep.py outputs/sweep_*/` registers all cells
5. `catalog.json` is gitignored (it's generated, user-specific)

### Validation refactor (Item 4)

6. The existing 11-category validation checks for noshow overbooking continue to work (output matches current `validation_appendix.md` format)
7. `validate.py` can be called on a Hydra output directory, not just an argparse output directory
8. A new scenario can implement validation by adding a `validate_{name}.py` without modifying `experiments/validate.py`

### Runner migration (Items 5-6)

9. `run_evaluation.py` works with Hydra config: `python run_evaluation.py` uses defaults from `evaluation.yaml`
10. All three argparse runners produce output directories that register successfully in the catalog
11. The `save_results()` functions in individual runners are removed — `lifecycle.py` handles it

### Reference example (Item 7)

12. `scenarios/_template/configs/` contains a working example config with Hydra sweep structure
13. The example config has comments explaining every section
14. New scenario developers can copy the template and have a working experiment infrastructure

### Documentation (Item 8)

15. `CLAUDE.md` documents the experiment lifecycle convention
16. `sim-guide.md` references `lifecycle.py` and `register_sweep.py` in the validation infrastructure section
17. A Claude Code agent working on a new scenario would know to use `finalize_experiment()` and Hydra configs

---

## Testing

### Unit tests (`tests/unit/test_lifecycle.py`)

- `test_save_experiment_creates_expected_files` — writes config.json and metrics.json
- `test_register_experiment_appears_in_catalog` — round-trip registration
- `test_register_sweep_finds_all_cells` — mock a sweep directory structure, verify all cells registered
- `test_finalize_experiment_full_lifecycle` — save + register in one call
- `test_register_is_idempotent` — calling register twice doesn't duplicate

### Integration tests (`tests/integration/test_experiment_lifecycle.py`)

- `test_threshold_optimizer_registers_in_catalog` — run the threshold optimizer with minimal params, verify catalog entry
- `test_evaluation_runner_with_hydra_config` — run evaluation with the new YAML config, verify output structure
- `test_sweep_directory_registration` — run a 2x2 multirun, register sweep, verify 4 entries in catalog

---

## Out of Scope

- **Hydra Optuna sweeper** — the SDK supports `--multirun` grid sweeps; Bayesian optimization via Optuna is a future enhancement
- **Remote experiment storage** — catalog.json is local; cloud storage (S3, MLflow) is a future integration point
- **Experiment comparison dashboards** — `catalog.compare()` returns raw data; interactive comparison is a separate task
- **Migration of existing catalog.json files** — any existing catalog entries from the old format should still load (add backwards-compatible parsing if needed)
