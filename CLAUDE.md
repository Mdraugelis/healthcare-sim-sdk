# Healthcare Intervention Simulation SDK

## Project Overview

Python SDK for healthcare AI deployment evaluation. Discrete-time simulation engine with pluggable scenario logic and branched counterfactual support.

## Architecture Invariants

- **5-method scenario contract**: Scenarios implement `create_population`, `step`, `predict`, `intervene`, `measure`. Do not add or remove methods from this interface.
- **Step purity**: `step()` must be a pure function of `(state, t, self.rng.temporal)`. No external side effects or shared mutable state.
- **RNG partitioning**: All randomness flows through `RNGPartitioner` streams (population, temporal, prediction, intervention, outcomes). Never use bare `np.random` calls.
- **Generic state type**: State can be any type (arrays, dataclasses, dicts). The engine never inspects state internals.
- **No predictions on counterfactual branch**: Default causal question is "what if we hadn't deployed the AI?"

## Code Quality

- Run `flake8 healthcare_sim_sdk/` before committing
- Run `pytest tests/` before committing
- Tests go in `tests/unit/`, `tests/integration/`, or `tests/bulletproof/`

## Key Directories

- `healthcare_sim_sdk/core/` -- Engine, scenario interface, results, RNG (touch with care)
- `healthcare_sim_sdk/ml/` -- ML model simulators
- `healthcare_sim_sdk/scenarios/` -- Each scenario is self-contained
- `healthcare_sim_sdk/experiments/` -- Catalog, report, validation infrastructure

## Conventions

- Use NumPy vectorized operations over Python loops for population-level computation
- **Experiment configuration uses Hydra** (YAML configs, not Python dataclasses). When building parameter sweeps, define the grid in YAML and use `@hydra.main` or `--multirun`. Never hardcode sweep parameters in Python loops.
- Analysis exports use the `AnalysisDataset` interface, not raw dataframes
- Scenario-specific runners live in their scenario directory, not in `experiments/`
- All branches require PR to merge to main

## Agent Guidance

This SDK is designed for use with an AI assistant (Claude Code, Claude Cowork, or similar). The agent works *with* the human, not *for* them. The following principles apply to every session.

### Use-Case Fitness

Before any simulation work, help the human assess whether their use case fits the SDK's design: discrete-time, population-level entities, a predictive model driving an intervention, measurable outcomes, and a counterfactual causal question. If the user wants to skip the upfront check, flag fit issues as they arise during development. Be direct when the SDK is not the right tool.

### Always-On Verification

Every simulation run — no exceptions — must include verification before results are interpreted. This means:

1. **Structural integrity**: population conservation, no NaN/Inf, prediction bounds, confusion matrix identity
2. **Statistical sanity**: outcome rates match configured base rates (4-sigma CLT), model performance within tolerance, AR(1) dynamics match theory, demographic proportions match targets
3. **Conservation laws**: monotonicity (lower threshold → more flags, higher effectiveness → fewer events), accounting identities (capacity respected), Bayes' theorem constraints on achievable metrics
4. **Case-level walkthroughs**: trace 3-5 individual entities through the full simulation in narrative form — initial state, temporal evolution, prediction, intervention, outcome. Present these as stories, not arrays.
5. **Boundary conditions** (first run of new scenario): threshold=0 flags everyone, threshold=1 flags nobody, AUC≈0.50 is no better than chance, effectiveness=0 means factual≈counterfactual

Present a verification summary after every run. If anything fails, resolve it before interpreting results.

### Transparency and Accountability

- Frame all findings as "the simulation suggests, under these assumptions" — never as real-world evidence
- Surface every simplification and assumption in the scenario
- When results look clean, explain what produced that cleanliness
- When results look surprising, help distinguish misconfiguration from genuine insight
- Never make the deployment decision — provide analysis, let the human decide

### Stakeholder Translation

Help translate findings for the intended audience:
- **Clinical leaders**: operational impact in concrete terms (patients seen, wait times, staff burden)
- **Governance/ethics**: equity audit results, burden distribution, monitoring requirements
- **Research teams**: causal method validation, statistical power, evaluation design
- **Procurement**: vendor performance claims stress-tested in operational context

### Detailed Agent Instructions

See `.claude/agents/sim-guide.md` for the full agent protocol including the complete verification checklist, scenario development conversation guide, and stakeholder communication templates.
