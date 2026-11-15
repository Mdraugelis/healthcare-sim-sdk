# Review packets and case traces

A review packet gathers the evidence for one experiment and one decision question, and lays it out for five kinds of reviewer. It records what was checked, what passed, and what was never run. It does not approve anything. Human review is recorded separately and is never filled in by code.

## Try it

From the repository root, after `pip install -e ".[dev]"`:

```bash
python -m healthcare_sim_sdk.scenarios.noshow_targeted_reminders.run_review_demo
```

Open `outputs/review_demo/noshow_reminders/review_packet.md`. The demo runs the targeted-reminders scenario at two call capacities (12 and 24 calls a day), three seeds each, 30 days, 2,000 synthetic patients. It takes about 20 seconds. Cells and seeds are set in `scenarios/noshow_targeted_reminders/configs/review_demo.yaml`, and the output folder in `configs/review_demo_run.yaml`; the claims are in `configs/review_claims.yaml`.

The packet's overall machine status is `NOT_RUN`. That is deliberate. Two claims, held-out model validation and real-world study design, have no evidence in this demo, and a packet with missing required evidence cannot read as PASS.

## What a reviewer sees

| File | Contents |
|---|---|
| `review_packet.md` | Decision question, claim statuses, unresolved evidence, results by cell, highlighted cases |
| `review_clinical.md`, `review_operations.md`, `review_statistics.md`, `review_governance.md`, `review_engineering.md` | Each role's claims, assumptions to review, results and cases |
| `case_appendix.md` | Every case category for every run, including categories with no candidates |
| `evidence.json` | Every check for every run, with status, actual and expected values |
| `registry.json` | Claims: question, estimand, population, denominator, comparison, horizon, acceptance, reviewers, required evidence |
| `config.json` | The resolved configuration |
| `manifest.json` | Run identity, expected, completed, missing and failed runs, and a SHA-256 for every artifact |
| `human_review.json` | One pending entry per role; kept when the packet is regenerated |
| `traces/` | Per-run case traces |

## Statuses

Statuses mean the same thing here as in `docs/experiment_verification.md`: `PASS`, `FAIL`, `NOT_RUN`, `INCONCLUSIVE`, and `NOT_APPLICABLE` with a required rationale. A claim's status is the worst status among its required evidence. Evidence the registry names but no run supplied is added as `NOT_RUN`.

## Checks in the reminders demo

Each run records these, computed from that run:

| Check | What it compares |
|---|---|
| `counterfactual_is_no_ai_world` | The no-AI branch against a separate run with an empty prediction schedule. Must be identical. |
| `branches_share_schedule` | Slot IDs and patient groups on both branches, every day |
| `call_capacity_respected` | Calls per day against capacity; zero calls on the no-AI branch |
| `outcome_accounting` | Per-day outcomes against running totals on each branch |
| `zero_effect_identity` | A rerun with reminder effectiveness 0. Both branches must be identical. |
| `trace_reconciles` | Outcome totals in the trace against the recorded results |
| `no_program_rate_matches_config` | No-AI no-show rate against the configured rate, within four binomial standard errors |
| `case_traces` | One seeded random slot per category: helped, reached without effect, missed |

The base-rate check passes, but the no-AI rate sits below the configured 0.36 in every seed (about 34.5%). The scenario rescales patient probabilities to the target and then caps them at 0.80 (`create_population`). With concentration 0.3, enough patients exceed 0.80 that the cap lowers the patient mean to about 0.35. A four-standard-error check is too loose to catch a bias of this size; the packet makes it visible in `evidence.json`.

## Identity and integrity

`run_identity()` hashes the tracked source files, the resolved configuration, any model file supplied, the registry, the expected runs, and the Python and dependency versions. Changing any of these changes the packet identity.

Check a saved packet:

```python
from healthcare_sim_sdk.experiments.review import verify_packet
for check in verify_packet("outputs/review_demo/noshow_reminders"):
    print(check["name"], check["status"])
```

`artifact_integrity` fails if any file listed in the manifest changed. The hashes detect accidental or partial edits. They are not a signature: someone who rewrites the whole packet can rewrite the manifest too.

## Traces

`experiments/trace.py` provides `TracingScenario`, a wrapper that records what a scenario does at each stage of a run without changing it:

```python
from healthcare_sim_sdk.core.engine import BranchedSimulationEngine
from healthcare_sim_sdk.experiments.trace import TracingScenario

wrapped = TracingScenario(
    scenario,                              # a fresh scenario
    lambda state: {"day": state.day},      # what to record from state
    stages=("population", "step_after", "intervention", "outcome"),
)
results = BranchedSimulationEngine(wrapped).run(n_entities)
rows = wrapped.to_rows()
```

- The wrapper draws no random numbers and passes every call through. `tests/integration/test_trace_equivalence.py` checks that outcomes, scores, assignments and the end position of every random-number stream match a plain run, for the reminders, nurse-retention and sepsis scenarios.
- Each record carries a trace ID, timestep, branch (`factual`, `counterfactual`, or `shared` for the population), stage and JSON facts.
- In snapshot mode both measurements share one random-number stream, so their branch is labeled `snapshot_unresolved`. Use branched mode for paired stories.
- The default cap is 10,000 records. Exceeding it raises an error; records are never dropped.
- A wrapper and its scenario are single-use, like every scenario on main.

## Building a packet for another scenario

1. Write a claims registry like `noshow_targeted_reminders/configs/review_claims.yaml`.
2. Create `ReviewPacket(output_dir, registry, config, expected_runs)` with every `(cell, seed)` you intend to run.
3. For each run, compute checks with `experiments.review.evidence(...)` and call `packet.add_run(cell, seed, metrics=..., checks=..., cases=..., artifacts=...)`. Record failures with `error=`; never skip them.
4. Add experiment-level evidence with `packet.add_evidence(...)`, including `NOT_RUN` entries for anything you did not do.
5. Call `packet.write()`.

`experiments/uncertainty.py` summarizes metrics across seeds. It reports the seed range and the Monte Carlo standard error of the mean separately, and refuses missing or duplicate seeds.

## Related work

- Held-out model checks: `ml/fidelity.py` (`auc_fidelity`). Use it to supply evidence such as the reminders demo's `model_heldout_auc`.
- Analysis-design evidence: `docs/analysis_design_certification.md`.
- Frozen-input benchmark against a published study: `docs/blinded_benchmark.md`.
- Changes to how the results module exports treatment timing and missing periods are proposed separately and are not part of this feature.
