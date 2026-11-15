# core

The engine, the scenario interface, results and the random-number partitioner. Every scenario depends on these files, so a change here changes all of them. Most work never needs to touch this folder.

## Before you edit

1. Read `config/invariants.yaml`. It lists the protected files and the contract.
2. Ask whether the change belongs in a scenario instead. If a scenario needs a sixth method or wants the engine to look inside state, either the scenario design needs rethinking or the use case does not fit the SDK.
3. Tell the person the file is protected and confirm the change is intended. A commit that touches this folder prints a warning and needs maintainer review.

## What is enforced

| Rule | Enforced by |
|---|---|
| Five abstract methods; five named streams, in a fixed order | `scripts/check_invariants.py`, reading the YAML |
| No randomness outside the partitioner | `scripts/sdk_lint.py`, rule SDK001 |
| Engine does not inspect state | `scripts/check_invariants.py` |
| Step purity; equal draw counts on both branches | `tests/integration/test_all_scenario_contracts.py`, on every scenario |
| Runners start through Hydra; experiments end in `finalize_experiment()` | `scripts/sdk_lint.py`, rule SDK007; `tests/integration/test_runner_lifecycle.py` runs each one |
| The YAML, the code and the documents agree | `tests/unit/test_invariants_file.py` |

Static checks cannot prove purity. Only the behavioral tests do.

## Changing an invariant

This is a design decision, not an edit.

1. Do not make the change on a scenario branch.
2. Open an issue against `main` that states the new contract and why.
3. Change `config/invariants.yaml`, the code, the tests and `AGENTS.md` in one pull request. The parity tests fail until all of them agree.

## Contracts that are easy to break

- A scenario object runs once. A second `run()` raises; build a new object to repeat a seed.
- `seed=None` means seed 42, not a random seed.
- Stream order fixes seeded results. Reordering `STREAM_NAMES` changes every run.
- The counterfactual branch never calls `predict()` or `intervene()`.
- Export semantics are part of the contract: see `docs/analysis_exports.md`.
- The full description is `docs/algorithm_description.md`, sections 3 to 6 and 10.
