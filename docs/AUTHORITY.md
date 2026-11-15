# Authority map

Every markdown file in this repository is one of four things. This page says which, so a reader or an agent does not have to guess from a filename.

| Class | Meaning | How to use it |
|---|---|---|
| `current` | Binding now. Describes the code as it is. | Follow it. If it disagrees with the code, one of them is a bug. |
| `rationale` | Explains why the SDK is built this way. | Read for intent. Its numbers and examples may be dated. |
| `proposal` | Not implemented. | Do not treat as behavior. Do not cite as a capability. |
| `record` | What was true on a stated date. | Evidence and history. May name code that has since changed or gone. |

When two files disagree, the higher row in that table wins, and code plus tests win over all of them.

`scripts/check_docs.py` reads the table below. Every tracked markdown file must match a row (rule DOC006). A file matches the most specific row: an exact path, then a pattern, then the longest folder.

## The map

| Path | Class | Notes |
|---|---|---|
| `AGENTS.md` | current | Root guidance for every coding agent: invariants, data boundary, conventions, mandatory verification. |
| `CLAUDE.md` | current | Symlink to `AGENTS.md`. Claude Code and Cowork load it automatically. |
| `README.md` | current | Overview and quick start for people. |
| `CONTRIBUTING.md` | current | Setup, hooks and pull-request rules. |
| `CHANGELOG.md` | current | Release notes and breaking changes, newest first. |
| `.claude/` | current | Claude Code adapters. They register and point; they hold no rules of their own. See "Adapters" below. |
| `docs/AUTHORITY.md` | current | This page. |
| `docs/algorithm_description.md` | current | Sections 1 to 13 describe current behavior. Section 15 is a proposal. Appendices A and B are records. |
| `docs/analysis_design_certification.md` | current | The synthetic certification harness for analysis designs. |
| `docs/analysis_exports.md` | current | Export semantics: outcome selection, missing periods, entity identity, treatment. |
| `docs/analytical_first_pass.md` | current | When closed-form math is enough, and the worked protocol. |
| `docs/blinded_benchmark.md` | current | The frozen-input benchmark runner and its SHIELD-RT pilot. |
| `docs/data_boundary.md` | current | What must never enter tracked files, and the guards that enforce it. |
| `docs/experiment_verification.md` | current | Save, verify and register order; the registration gate. |
| `docs/lint_rules.md` | current | Lint rules, documentation checks, test profiles and the quality gate. |
| `docs/review_packets.md` | current | Review packets and case traces. |
| `docs/simulation_protocol.md` | current | The six-phase protocol for designing, running, verifying and interpreting a simulation. |
| `docs/design_principles.md` | rationale | Why the SDK is built this way. Proof-point numbers are historical. Appendix B is a proposal. |
| `docs/paper_based_validation.md` | proposal | A proposed approach to validation against published studies. |
| `docs/issue_register_validation.md` | record | Reproduction and test coverage for issues I-01 to I-29, reviewed 16 October 2026. |
| `docs/validation_methodology_alignment.md` | record | Assessment of validation methods, reviewed 17 October 2026. |
| `examples/` | current | Inputs and notes for runnable examples. |
| `healthcare_sim_sdk/` | current | Guidance that sits beside the code it describes. |
| `healthcare_sim_sdk/scenarios/nurse_retention/EXPERIMENT_DESIGN.md` | rationale | The design reasoning behind the nurse-retention experiments. |
| `reports/` | record | Dated reports. The four revised reports of 16 October 2026 supersede anything earlier. |
| `research/` | record | The April 2026 paper audit and teaching packet. Reproduction labels there are superseded; see "Published-Study Calibration Exercises" in `docs/design_principles.md`. |

## Sources and adapters

Three tool-neutral files are sources. Vendor-specific files are thin adapters that point at them, so no rule lives in two places.

| Source | Holds | Adapter | What the adapter does |
|---|---|---|---|
| `AGENTS.md` | Repo-wide rules | `CLAUDE.md` | Symlink. Claude Code and Cowork read `CLAUDE.md`. Where symlinks are unavailable, a file containing only `@AGENTS.md` is accepted. |
| `docs/simulation_protocol.md` | The simulation protocol | `.claude/agents/sim-guide.md` | Registers a `sim-guide` agent with Claude Code. Its body says to read and follow the protocol. |
| `config/invariants.yaml` | The machine-read contract | `.claude/invariants.yaml` | Compatibility symlink for tools that still read the old path. Kept until the tools that read the old path are updated, then removed. |
| `AGENTS.md` files under `healthcare_sim_sdk/` | Guidance beside the code | `.claude/rules/nested-agents.md` | Path-scoped rule. Claude Code does not read nested `AGENTS.md` files when a root `CLAUDE.md` exists, so this rule tells it to. |

`scripts/check_docs.py` checks each adapter (rules DOC003 and DOC008) and stops the old paths from being cited as sources again (DOC009).

## Mixed documents

Two documents hold more than one class. The map gives each its main class; the sections below are the exceptions.

| Document | Section | Class |
|---|---|---|
| `docs/algorithm_description.md` | 15, Companion notebooks | proposal |
| `docs/algorithm_description.md` | Appendix A, Defect history; Appendix B, Issues register | record |
| `docs/design_principles.md` | Appendix B, What's Missing That We Should Apply | proposal |

## Not markdown

- Code and tests are the final authority on behavior. `config/invariants.yaml` is `current`: the invariant hook reads it.
- GitHub issues track open problems and proposals. A closed issue is a record.
- Commit messages and pull-request descriptions are records.
- `outputs/` is never committed and has no authority.
