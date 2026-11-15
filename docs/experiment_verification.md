# Saving, verifying and registering a run

`finalize_experiment()` in `healthcare_sim_sdk/experiments/lifecycle.py` is the one call every runner makes after a simulation. This page describes what it does and what a catalog entry claims.

## Order of operations

1. **Save.** `config.json`, `metrics.json`, and `results.json` when given.
2. **Record provenance.** `provenance.json` holds the code revision, the SHA-256 of `config.json`, every seed found in the configuration, the Python version, the platform, and the versions of the SDK and its main dependencies.
3. **Validate.** The generic artifact checks run, then the checks the runner supplies. The result goes to `validation.json`.
4. **Register.** The run enters the catalog only if the gate allows it.

None of these steps changes simulation behavior or random-number use.

## Check states

Each check has one of five states (`experiments/validate.py`):

| State | Meaning |
|---|---|
| `PASS` | The check ran and met its criterion. |
| `FAIL` | The check ran and missed its criterion. |
| `NOT_RUN` | The evidence was not produced. |
| `INCONCLUSIVE` | The check ran and could not decide. |
| `NOT_APPLICABLE` | The check does not apply. It counts only with a stated applicability and rationale. |

The gate passes only when every required check is `PASS` or a justified `NOT_APPLICABLE`, and no check is `FAIL`. An empty list of checks never passes. A check named in `required_checks` but not supplied is added as `NOT_RUN`.

If a runner supplies no checks, two `NOT_RUN` placeholders are added, one for simulation and one for analysis. Saving alone therefore never yields a verified run.

## The registration gate

| Situation | Registered? | Catalog `validated` |
|---|---|---|
| Artifact checks fail (missing or invalid JSON, no seed, no metrics) | No | — |
| `require_validated=True` and the gate is blocked | No. Files are kept, `validation.json` has `"rejected": true`, and `ValueError` is raised. | — |
| `require_validated=True` and the gate passes | Yes | `true` |
| `require_validated=False` and the gate is blocked | Yes | `false` |
| Entry registered before this change, or without a `validation.json` | Yes | `null` |

Each catalog entry carries `validated`, a readable `validation_status` (`verified`, `unverified`, or `no validation record`), and `code_revision`. Use `ExperimentCatalog.list_verified()` to read only verified runs. `register_sweep()` skips any cell whose `validation.json` is marked rejected.

## Supplying checks

```python
from healthcare_sim_sdk.experiments.lifecycle import finalize_experiment
from healthcare_sim_sdk.experiments.validate import Check

checks = [
    Check("population_conserved", "Entities in equal entities out",
          expected=n, actual=n_out, tolerance="exact",
          passed=(n == n_out), category="simulation"),
]
finalize_experiment(out_dir, config, metrics,
                    validation_checks=checks,
                    required_checks=["population_conserved"],
                    require_validated=True)
```

Use `require_validated=True` for any run whose result will be reported. Leave it off for exploratory runs; they are registered and labeled unverified.

## Reading `provenance.json`

`code_revision.dirty` is `true` when tracked files differed from the recorded commit at run time. A dirty run cannot be reproduced from the commit alone. `code_revision.source` is `git` for a checkout, `installed package` for a VCS install without a checkout, and `unavailable` otherwise.

## Command line

`python -m healthcare_sim_sdk.experiments.validate <output_dir> [--require-check NAME] [--metric-rules rules.json]` prints the validation appendix and exits 1 when the gate is blocked.
