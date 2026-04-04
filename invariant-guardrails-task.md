# Implementation Task: SDK Invariant Protection

**Guardrails for the Healthcare Simulation SDK**

| Field | Value |
|-------|-------|
| Repo | healthcare-sim-sdk |
| Target branch | `feature/invariant-guardrails` |
| Priority | High — foundational for multi-user workflow |
| Estimated effort | 3–5 hours |
| Assigned to | *(Developer name)* |
| Created | 2026-04-04 |

---

## Context

The Healthcare Simulation SDK has five architectural invariants documented in `CLAUDE.md`. Today these are enforced only by documentation and code review. As we onboard users who will work through Claude Code and GitHub, we need mechanical enforcement so that invariant violations are caught automatically — before a PR is opened, not after.

This task creates three layers of protection plus a gitignore policy for experiment outputs:

- **Layer 1** — Pre-commit hooks that block commits violating invariants
- **Layer 2** — Pytest structural tests that verify the contract at import time
- **Layer 4** — Claude Code agent awareness via a machine-readable invariants file and enhanced CLAUDE.md instructions
- **Experiment output policy** — Ensure no experiment results are ever committed; keep one Hydra config example

---

## The Five Invariants (Reference)

All work in this task protects these invariants. Refer back to this list when implementing checks.

| # | Invariant | What It Means |
|---|-----------|---------------|
| 1 | 5-method scenario contract | `BaseScenario` defines exactly these abstract methods: `create_population`, `step`, `predict`, `intervene`, `measure`. No additions, removals, or renames. |
| 2 | Step purity | `step()` is a pure function of `(state, t, self.rng.temporal)`. No external side effects or shared mutable state. |
| 3 | RNG partitioning | All randomness flows through `RNGPartitioner` streams (`population`, `temporal`, `prediction`, `intervention`, `outcomes`). No bare `np.random` calls anywhere. |
| 4 | Generic state type | State can be any type. The engine never inspects state internals — no `getattr`, `isinstance`, or key access on state objects. |
| 5 | No predictions on counterfactual | The counterfactual branch never calls `predict()`. The causal question is: what if we hadn't deployed the AI? |

---

## Layer 1: Pre-Commit Hooks

These hooks run automatically on every `git commit`. They catch invariant violations before code reaches a branch or PR. Because Claude Code executes `git commit`, these hooks will also stop the agent from committing violating code.

### 1A. Install pre-commit framework

Add `pre-commit` to dev dependencies in `pyproject.toml`:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "flake8>=6.0",
    "pre-commit>=3.0",  # <-- ADD
    ...
]
```

Create `.pre-commit-config.yaml` at the repo root:

```yaml
repos:
  - repo: local
    hooks:
      - id: check-invariants
        name: SDK invariant checks
        entry: python scripts/check_invariants.py
        language: python
        always_run: true
        pass_filenames: false

      - id: no-bare-np-random
        name: No bare np.random calls
        entry: python scripts/check_no_bare_random.py
        language: python
        types: [python]
        pass_filenames: true

      - id: flake8
        name: flake8 lint
        entry: flake8
        language: python
        types: [python]
        args: [healthcare_sim_sdk/]
        pass_filenames: false

      - id: core-modification-warning
        name: Warn on core/ modifications
        entry: python scripts/check_core_modifications.py
        language: python
        always_run: true
        pass_filenames: false
```

### 1B. Script: `scripts/check_invariants.py`

This script imports the SDK and verifies the structural contract at the Python level. It runs on every commit regardless of which files changed.

**What it checks:**

- `BaseScenario` has exactly 5 abstractmethods: `create_population`, `step`, `predict`, `intervene`, `measure`
- `RNGStreams` dataclass has exactly 5 fields: `population`, `temporal`, `prediction`, `intervention`, `outcomes`
- `RNGPartitioner.STREAM_NAMES` matches the `RNGStreams` fields
- `BranchedSimulationEngine.__init__` does not call `getattr`/`isinstance`/`hasattr` on state

**Implementation guidance:**

```python
#!/usr/bin/env python3
"""Pre-commit hook: verify SDK architectural invariants."""

import inspect
import sys
from dataclasses import fields

EXPECTED_ABSTRACT = {
    "create_population", "step", "predict", "intervene", "measure"
}
EXPECTED_STREAMS = {
    "population", "temporal", "prediction", "intervention", "outcomes"
}

def check_scenario_contract():
    from healthcare_sim_sdk.core.scenario import BaseScenario
    abstract = {
        name for name, method in inspect.getmembers(BaseScenario)
        if getattr(method, "__isabstractmethod__", False)
    }
    if abstract != EXPECTED_ABSTRACT:
        added = abstract - EXPECTED_ABSTRACT
        removed = EXPECTED_ABSTRACT - abstract
        msg = "INVARIANT VIOLATION: 5-method scenario contract broken\n"
        if added:
            msg += f"  Added: {added}\n"
        if removed:
            msg += f"  Removed: {removed}\n"
        print(msg, file=sys.stderr)
        return False
    return True

def check_rng_streams():
    from healthcare_sim_sdk.core.rng import RNGStreams, RNGPartitioner
    stream_fields = {f.name for f in fields(RNGStreams)}
    if stream_fields != EXPECTED_STREAMS:
        print(f"INVARIANT VIOLATION: RNGStreams fields changed\n"
              f"  Expected: {EXPECTED_STREAMS}\n"
              f"  Got: {stream_fields}", file=sys.stderr)
        return False
    if set(RNGPartitioner.STREAM_NAMES) != EXPECTED_STREAMS:
        print("INVARIANT VIOLATION: RNGPartitioner.STREAM_NAMES "
              "out of sync with RNGStreams", file=sys.stderr)
        return False
    return True

def check_engine_state_opacity():
    """Verify engine.py does not inspect state internals."""
    import ast
    from pathlib import Path
    engine_path = Path("healthcare_sim_sdk/core/engine.py")
    tree = ast.parse(engine_path.read_text())
    STATE_NAMES = {"state_factual", "state_counterfactual",
                   "state", "state_snapshot"}
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id in (
                "getattr", "isinstance", "hasattr"
            ):
                if (node.args and isinstance(node.args[0], ast.Name)
                    and node.args[0].id in STATE_NAMES):
                    violations.append(
                        f"  Line {node.lineno}: {fn.id}({node.args[0].id}, ...)"
                    )
    if violations:
        print("INVARIANT VIOLATION: engine inspects state internals\n"
              + "\n".join(violations), file=sys.stderr)
        return False
    return True

if __name__ == "__main__":
    results = [
        check_scenario_contract(),
        check_rng_streams(),
        check_engine_state_opacity(),
    ]
    if all(results):
        print("All invariant checks passed.")
        sys.exit(0)
    else:
        sys.exit(1)
```

### 1C. Script: `scripts/check_no_bare_random.py`

Scans staged Python files for bare `np.random` calls. Only `rng.py` is allowed to reference `np.random` directly. This enforces Invariant #3 (RNG partitioning).

**What it checks:**

- Any occurrence of `np.random.` (e.g., `np.random.random()`, `np.random.default_rng()`, `np.random.seed()`) in files other than `core/rng.py`
- Also catches `import numpy.random` and `from numpy import random`

**Implementation guidance:**

```python
#!/usr/bin/env python3
"""Pre-commit hook: no bare np.random calls outside rng.py."""

import re
import sys
from pathlib import Path

ALLOWED_FILES = {
    Path("healthcare_sim_sdk/core/rng.py"),
}

PATTERNS = [
    re.compile(r"np\.random\."),
    re.compile(r"numpy\.random"),
    re.compile(r"from\s+numpy\s+import\s+random"),
]

def check_file(path: Path) -> list[str]:
    violations = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if line.lstrip().startswith("#"):
            continue
        for pattern in PATTERNS:
            if pattern.search(line):
                violations.append(f"  {path}:{i}: {line.strip()}")
    return violations

if __name__ == "__main__":
    files = [Path(f) for f in sys.argv[1:]]
    violations = []
    for f in files:
        if f in ALLOWED_FILES or not f.suffix == ".py":
            continue
        violations.extend(check_file(f))
    if violations:
        print("INVARIANT VIOLATION: bare np.random usage found\n"
              "All randomness must flow through RNGPartitioner.\n"
              + "\n".join(violations), file=sys.stderr)
        sys.exit(1)
    sys.exit(0)
```

### 1D. Script: `scripts/check_core_modifications.py`

This hook checks whether any files in `healthcare_sim_sdk/core/` are in the git staging area. If so, it prints a prominent warning but does **NOT** block the commit. The purpose is awareness, not prevention — branch protection and CODEOWNERS handle the merge gate.

**Behavior:**

- Runs `git diff --cached --name-only` to get staged files
- If any match `healthcare_sim_sdk/core/*`, prints a warning to stderr
- Always exits 0 (warning only, never blocks)

**Warning message to print:**

```
⚠️  CORE MODIFICATION DETECTED
──────────────────────────────
You are modifying SDK invariant files:
  - healthcare_sim_sdk/core/scenario.py
  - healthcare_sim_sdk/core/engine.py

These files define the SDK's architectural contract.
Changes here affect ALL scenarios and ALL users.

Before proceeding, consider:
  1. Is this a scenario-level change that belongs in
     your scenario directory instead?
  2. Should this be proposed as a design change to
     the main repo via RFC/discussion?
  3. Does this change break any of the 5 invariants
     documented in CLAUDE.md?

This commit will proceed, but core/ changes require
PR review from a maintainer.
```

**Implementation guidance:**

```python
#!/usr/bin/env python3
"""Pre-commit hook: warn on core/ modifications."""

import subprocess
import sys

CORE_PREFIX = "healthcare_sim_sdk/core/"

def main():
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        capture_output=True, text=True
    )
    staged = result.stdout.strip().splitlines()
    core_files = [f for f in staged if f.startswith(CORE_PREFIX)]
    if core_files:
        files_list = "\n".join(f"    - {f}" for f in core_files)
        print(f"""
  CORE MODIFICATION DETECTED
  You are modifying SDK invariant files:
{files_list}

  These files define the SDK architectural contract.
  Changes here affect ALL scenarios and ALL users.

  Before proceeding, consider:
    1. Is this a scenario-level change that belongs
       in your scenario directory instead?
    2. Should this be proposed as a design change
       to the main repo via RFC/discussion?
    3. Does this change break any of the 5 invariants
       documented in CLAUDE.md?

  This commit will proceed, but core/ changes
  require PR review from a maintainer.
""", file=sys.stderr)
    # Always exit 0: this is a warning, not a gate
    sys.exit(0)

if __name__ == "__main__":
    main()
```

### Layer 1: Acceptance Criteria

1. `pre-commit` framework is configured and installs with `pip install -e .[dev]`
2. `git commit` on an unmodified repo passes all hooks cleanly
3. Adding a 6th `abstractmethod` to `BaseScenario` causes `check_invariants.py` to fail the commit
4. Adding `np.random.random()` to a scenario file causes `check_no_bare_random.py` to fail the commit
5. Modifying `engine.py` triggers the core modification warning but does not block
6. All three scripts have clear, human-readable error messages that explain the invariant and what to do

---

## Layer 2: Structural Invariant Tests (pytest)

These tests verify the SDK's architectural contract at the Python import level. They complement the pre-commit hooks by running in CI (pytest) and catching issues that survive past the commit stage. They are fast, deterministic, and require no simulation runs.

**Location:** `tests/bulletproof/test_structural_invariants.py`

### 2A. Scenario Contract Tests

**`test_base_scenario_has_exactly_five_abstract_methods`**

Import `BaseScenario`, introspect for `__isabstractmethod__`, assert the set equals `{create_population, step, predict, intervene, measure}`. Fail with a descriptive message showing added/removed methods.

**`test_base_scenario_method_signatures_stable`**

For each of the 5 methods, inspect the signature (parameter names and annotations) and assert they match the documented contract. This catches subtle breaks like reordering parameters or changing return types.

Expected signatures:

```python
create_population(self, n_entities: int) -> S
step(self, state: S, t: int) -> S
predict(self, state: S, t: int) -> Predictions
intervene(self, state: S, predictions: Predictions, t: int) -> tuple[S, Interventions]
measure(self, state: S, t: int) -> Outcomes
```

**`test_optional_hooks_are_not_abstract`**

Verify that `validate_population`, `validate_results`, and `clone_state` exist on `BaseScenario` but are NOT abstract. They must have concrete default implementations.

**`test_data_classes_stable`**

Verify that `Predictions`, `Interventions`, and `Outcomes` dataclasses have exactly the documented fields with correct types. For example, `Predictions` must have fields: `scores` (`np.ndarray`), `labels` (`Optional[np.ndarray]`), `metadata` (`Dict[str, Any]`).

### 2B. RNG Contract Tests

**`test_rng_streams_has_exactly_five_fields`**

`RNGStreams` dataclass must have exactly these fields in this order: `population`, `temporal`, `prediction`, `intervention`, `outcomes`. All must be typed as `np.random.Generator`.

**`test_rng_partitioner_stream_names_match`**

`RNGPartitioner.STREAM_NAMES` list must equal the `RNGStreams` field names. These two definitions must stay in sync.

**`test_rng_fork_produces_synchronized_streams`**

Create a partitioner, fork it, create streams from both. Verify that consuming the temporal stream in the same order produces identical values. This is the foundation of the branched counterfactual design.

**`test_rng_streams_are_independent`**

Create streams, draw from temporal, verify that population stream state is unaffected. This confirms that consuming one stream does not advance another.

### 2C. Engine Contract Tests

**`test_engine_does_not_inspect_state`**

Parse `engine.py` with `ast` and walk the AST. Assert no calls to `getattr()`, `isinstance()`, or `hasattr()` where the first argument is a state variable (`state_factual`, `state_counterfactual`, `state`, `state_snapshot`). Also check there are no subscript operations (`state[...]`) or attribute access (`state.some_field`) on state variables, excluding the method calls via `self.scenario`.

**`test_counterfactual_branch_never_calls_predict`**

This can be tested by creating a mock scenario where `predict()` raises an exception and running in BRANCHED mode. If the counterfactual branch calls `predict()`, the test fails. This directly verifies Invariant #5.

**`test_branched_mode_calls_step_on_both_branches`**

Create a scenario that counts `step()` calls. Run in BRANCHED mode. Verify step is called exactly `2 × n_timesteps` (once per branch per step).

### 2D. Cross-Invariant Integration Test

**`test_minimal_scenario_round_trip`**

Create a trivially simple scenario (state is a single numpy array, step adds 1, predict returns uniform random, intervene is no-op, measure counts events). Run it through the engine in all three modes (BRANCHED, SNAPSHOT, NONE). Verify it completes without error. This is the canary — if the contract is intact, the simplest possible scenario works.

### Layer 2: Acceptance Criteria

1. All tests pass on the current codebase (baseline green)
2. Removing an abstract method from `BaseScenario` fails `test_base_scenario_has_exactly_five_abstract_methods`
3. Adding a field to `RNGStreams` fails `test_rng_streams_has_exactly_five_fields`
4. The mock-scenario predict-raises test verifies counterfactual branch isolation
5. Tests run in < 2 seconds total (no simulation runs, just introspection)
6. Tests are in `tests/bulletproof/test_structural_invariants.py` and use existing conftest fixtures where applicable

---

## Layer 4: Claude Code Agent Awareness

This layer ensures that when Claude Code (or any agent operating on the repo) is working, it has machine-readable knowledge of the invariants and receives clear guidance about what it should and should not modify. This works in concert with the `CLAUDE.md` instructions the agent already reads.

### 4A. Create `.claude/invariants.yaml`

This file is the single source of truth for invariant definitions. The pre-commit scripts and pytest tests should reference this file rather than hardcoding expected values. Claude Code agents will also read this file when the `CLAUDE.md` instructions tell them to.

```yaml
# SDK Architectural Invariants
# ============================================================
# This file is the machine-readable source of truth.
# Pre-commit hooks and pytest structural tests validate
# against these definitions.
# ============================================================

version: 1

protected_files:
  description: >
    Files that define the SDK contract. Modifications require
    maintainer review and should be rare. Scenario developers
    should never need to touch these.
  paths:
    - healthcare_sim_sdk/core/scenario.py
    - healthcare_sim_sdk/core/engine.py
    - healthcare_sim_sdk/core/rng.py
    - healthcare_sim_sdk/core/results.py

scenario_contract:
  class: BaseScenario
  module: healthcare_sim_sdk.core.scenario
  required_abstract_methods:
    - create_population
    - step
    - predict
    - intervene
    - measure
  optional_hooks:
    - validate_population
    - validate_results
    - clone_state

rng_contract:
  streams_class: RNGStreams
  partitioner_class: RNGPartitioner
  module: healthcare_sim_sdk.core.rng
  required_streams:
    - population
    - temporal
    - prediction
    - intervention
    - outcomes

banned_patterns:
  - pattern: "np\\.random\\."
    allowed_in:
      - healthcare_sim_sdk/core/rng.py
      - tests/
    message: >
      All randomness must flow through RNGPartitioner streams.
      Use self.rng.temporal, self.rng.population, etc.

  - pattern: "import random"
    allowed_in: []
    message: >
      Do not use Python stdlib random. Use RNGPartitioner streams.

engine_constraints:
  - name: state_opacity
    description: >
      The engine must never inspect state internals.
      No getattr, isinstance, hasattr, subscript, or
      attribute access on state variables.
    state_variable_names:
      - state_factual
      - state_counterfactual
      - state
      - state_snapshot

  - name: counterfactual_isolation
    description: >
      The counterfactual branch must never call predict().
      The causal question is: what if we had not deployed
      the AI at all?

fitness_criteria:
  description: >
    A use case fits the SDK if ALL of these are true.
    If any are false, flag it to the user before building.
  checks:
    - Population-level intervention (not individual decision support)
    - Predictive model drives the intervention
    - Intervention is a state change (not purely informational)
    - Measurable outcome at each timestep
    - Counterfactual causal question (what if no AI?)
    - Discrete-time dynamics (not fundamentally continuous)
```

### 4B. Update CLAUDE.md

Add the following section to the existing `CLAUDE.md`, after the "Architecture Invariants" section:

```markdown
## Invariant Enforcement

Machine-readable invariant definitions live in
`.claude/invariants.yaml`. Before modifying any file in
`healthcare_sim_sdk/core/`, read that file and verify your
change does not violate any listed invariant.

If your change REQUIRES modifying an invariant (e.g., adding
a 6th scenario method), this is a design-level decision:

1. Do NOT make the change in a scenario branch
2. Open a discussion/RFC on the main repo
3. Update `.claude/invariants.yaml` as part of the RFC
4. The pre-commit hooks and structural tests will need
   updating to match the new contract

If a simulation use case does not fit the SDK (see
`fitness_criteria` in invariants.yaml), say so directly.
The SDK is not the right tool for every problem.
```

### 4C. Update sim-guide.md Agent Protocol

Add a new section to `.claude/agents/sim-guide.md`, immediately after the "What You Must Never Do" section:

```markdown
## Invariant Awareness

Before writing or modifying any code, check whether your
change touches a protected file listed in
`.claude/invariants.yaml`. If it does:

1. STOP and explain to the human why this file is protected
2. Confirm the change is intentional and necessary
3. Explain that this will trigger a core modification warning
   on commit and will require maintainer PR review
4. Verify the change does not violate any invariant

When helping a human build a new scenario, if you find
yourself wanting to change the BaseScenario interface or
the engine behavior, this is a signal that either:

- The scenario design needs rethinking to fit the contract
- The use case genuinely does not fit the SDK (flag this)
- The SDK contract needs evolution (open an RFC, do not
  make the change in a scenario branch)
```

### 4D. Wire Scripts to Read `invariants.yaml`

Refactor the pre-commit scripts (1B, 1C) and the pytest structural tests (Layer 2) to read their expected values from `.claude/invariants.yaml` rather than hardcoding them. This ensures a single source of truth. If the YAML file is missing or malformed, the scripts should fail loudly.

> **Important:** The `invariants.yaml` file is the source of truth. If a developer needs to evolve the contract (e.g., add a method), they update `invariants.yaml` and the hooks/tests automatically accept the new contract. This is by design — the file is version-controlled and changes to it will be visible in PR review.

### Layer 4: Acceptance Criteria

1. `.claude/invariants.yaml` exists, is valid YAML, and documents all 5 invariants
2. Pre-commit scripts and pytest tests read from `invariants.yaml` (no hardcoded expected values except as fallback)
3. `CLAUDE.md` references the invariants file and explains the RFC process for contract changes
4. `sim-guide.md` includes the invariant awareness section
5. A Claude Code agent reading `CLAUDE.md` would know to check `invariants.yaml` before modifying `core/`

---

## Experiment Output Policy

Experiment results are ephemeral, user-specific, and often large. They must never be committed to the repo. The repo should contain only the Hydra config structure as a reference example.

### Current state

The `.gitignore` already has:

```gitignore
# Generated experiment/simulation outputs
experiments/
outputs/
*_dashboard.html
```

This covers the repo root, but Hydra's default `outputs/` directory is relative to the working directory. If a user runs a scenario from inside `scenarios/noshow_overbooking/`, the outputs land in `scenarios/noshow_overbooking/outputs/` — which is **not** caught by the current rules.

### Changes needed

**Update `.gitignore`** — add patterns that catch Hydra outputs regardless of where they're generated:

```gitignore
# Generated experiment/simulation outputs
experiments/
**/outputs/
**/multirun/
*_dashboard.html

# Hydra auto-generated files
**/.hydra/
**/hydra_output/
```

The `**/` prefix makes these patterns match at any directory depth, so `scenarios/noshow_overbooking/outputs/` and `scenarios/sepsis_early_alert/outputs/` are both caught.

**Keep the existing Hydra config example** — the file `healthcare_sim_sdk/scenarios/noshow_overbooking/configs/threshold_optimizer.yaml` is already tracked and should stay. It demonstrates the correct Hydra config structure including the `hydra.run.dir` and `hydra.sweep` settings. This is the reference example.

**Add a README to the example config directory** — create `healthcare_sim_sdk/scenarios/noshow_overbooking/configs/README.md`:

```markdown
# Hydra Experiment Configs

This directory contains Hydra configuration files for running
parameter sweeps and experiments.

## Reference Example

`threshold_optimizer.yaml` demonstrates the standard config structure:
- Simulation defaults (seed, population size, time horizon)
- Domain-specific parameter groups (clinic, model, policy)
- Hydra output directory configuration

## Convention

All experiment configs should follow this pattern:
- Define parameters as Hydra config groups, not Python loops
- Set `hydra.run.dir` to `outputs/${now:%Y%m%d_%H%M%S}`
- Set `hydra.sweep.subdir` to encode the parameter combination
- Run sweeps with `--multirun`, not Python for-loops

## Output

Experiment outputs are written to `outputs/` which is gitignored.
Results are ephemeral. To persist findings, use the SDK's
`AnalysisDataset` export or the `experiments/report.py` generator.
```

**Update `.gitignore` note in CLAUDE.md** — in the Conventions section, add:

```markdown
- Experiment outputs (`outputs/`, `multirun/`, `.hydra/`) are gitignored at all directory depths. Never commit simulation results. The repo contains config structure examples only.
```

### What stays in the repo vs. what's ignored

| In the repo (tracked) | Gitignored (never committed) |
|------------------------|------------------------------|
| `scenarios/*/configs/*.yaml` (Hydra configs) | `**/outputs/` (Hydra run outputs) |
| `experiments/*.py` (catalog, report, validate code) | `**/multirun/` (Hydra sweep outputs) |
| `examples/*.ipynb` (reference notebooks) | `**/.hydra/` (Hydra auto-generated logs) |
| Config README with conventions | Any `*_dashboard.html` |

### Experiment Output Acceptance Criteria

1. Running `python run_threshold_optimizer.py` from the noshow scenario dir generates outputs locally but `git status` shows no untracked files
2. Running a `--multirun` sweep similarly produces no git-visible artifacts
3. The existing `threshold_optimizer.yaml` remains tracked and unaffected
4. The `**/outputs/` pattern works at any directory depth (verify with `git check-ignore -v scenarios/noshow_overbooking/outputs/test.json`)

---

## File Inventory

Complete list of files to create or modify:

| Action | File | Notes |
|--------|------|-------|
| CREATE | `.pre-commit-config.yaml` | Pre-commit framework config |
| CREATE | `scripts/check_invariants.py` | Structural invariant checks (Layer 1B) |
| CREATE | `scripts/check_no_bare_random.py` | Bare np.random scanner (Layer 1C) |
| CREATE | `scripts/check_core_modifications.py` | Core modification warning (Layer 1D) |
| CREATE | `tests/bulletproof/test_structural_invariants.py` | Structural contract tests (Layer 2) |
| CREATE | `.claude/invariants.yaml` | Machine-readable invariant definitions (Layer 4A) |
| CREATE | `healthcare_sim_sdk/scenarios/noshow_overbooking/configs/README.md` | Hydra config conventions |
| MODIFY | `pyproject.toml` | Add `pre-commit` to dev dependencies |
| MODIFY | `CLAUDE.md` | Add invariant enforcement section + experiment output convention |
| MODIFY | `.claude/agents/sim-guide.md` | Add invariant awareness section (Layer 4C) |
| MODIFY | `.gitignore` | Add `**/outputs/`, `**/multirun/`, `**/.hydra/` patterns |

---

## Testing the Guardrails Themselves

After implementing all three layers, verify the guardrails work end-to-end with these manual tests. Each test should be run on a throwaway branch.

**Test 1: Add a 6th abstract method.** Add an abstract method `summarize()` to `BaseScenario`. Attempt to commit. Expected: `check_invariants.py` FAILS the commit with a clear message about the 5-method contract. Also: `pytest test_base_scenario_has_exactly_five_abstract_methods` fails.

**Test 2: Use bare `np.random` in a scenario.** Add `x = np.random.random(100)` to any scenario file. Attempt to commit. Expected: `check_no_bare_random.py` FAILS with `file:line` reference and explanation.

**Test 3: Modify `engine.py`.** Make any change to `engine.py` (e.g., add a comment). Attempt to commit. Expected: core modification warning is printed to stderr. Commit proceeds (warning only, not blocking).

**Test 4: Counterfactual branch isolation.** The pytest test with the mock scenario that raises in `predict()` should pass, confirming the counterfactual branch never calls `predict()`.

**Test 5: Baseline green.** On an unmodified repo, both `pre-commit run --all-files` and `pytest tests/bulletproof/test_structural_invariants.py` should pass with zero failures.

**Test 6: Experiment outputs ignored.** Run the threshold optimizer from the noshow scenario directory. Confirm `git status` shows nothing. Run `git check-ignore -v scenarios/noshow_overbooking/outputs/anything` and confirm it matches.

---

## Out of Scope (Future Work)

These items are deliberately excluded from this task but noted for the roadmap:

- **Layer 3: CODEOWNERS file and GitHub branch protection rules** — Mike will configure these directly in GitHub settings
- **Layer 5: Use-case fitness gate** (`fit_assessment.yaml` per scenario) — separate task once scenario onboarding workflow is established
- **GitHub Actions CI workflow** — separate task, but should run the same pre-commit hooks and pytest suite
- **mypy strict mode on `core/`** — would add type-level invariant enforcement
