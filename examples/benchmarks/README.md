# SHIELD-RT evaluation inputs

- `shield_rt_pilot.json`: permitted inputs, provenance, declared exposure,
  endpoint definitions, fixed seeds and scoring rules. Mechanism distributions
  are explicitly illustrative and are not tuned to the published answer.
- `shield_rt_published_reference.json`: evaluator-only published observations.
  The coordinator seals these outside the repository. The SDK worker does not
  receive this file or any outcome-derived effectiveness parameter.

Follow the [evaluation guide](../../docs/blinded_benchmark.md) for the separate
seal, freeze, run, commit, reveal, score and verify commands. The supplied case
is an exposed development pilot, not an independently blinded validation.
