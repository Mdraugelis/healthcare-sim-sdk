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
| `CLAUDE.md` | current | Root guidance for agents: invariants, conventions, mandatory verification. Loaded automatically by Claude Code and Cowork. |
| `README.md` | current | Overview and quick start for people. |
| `CONTRIBUTING.md` | current | Setup, hooks and pull-request rules. |
| `.claude/agents/sim-guide.md` | current | The six-phase simulation protocol. Read on demand from the pointer in `CLAUDE.md`; not a registered subagent. |
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
| `docs/design_principles.md` | rationale | Why the SDK is built this way. Proof-point numbers are historical. Appendix B is a proposal. |
| `docs/paper_based_validation.md` | proposal | A proposed approach to validation against published studies. |
| `docs/issue_register_validation.md` | record | Reproduction and test coverage for issues I-01 to I-29, reviewed 16 October 2026. |
| `docs/validation_methodology_alignment.md` | record | Assessment of validation methods, reviewed 17 October 2026. |
| `examples/` | current | Inputs and notes for runnable examples. |
| `healthcare_sim_sdk/` | current | Guidance that sits beside the code it describes. |
| `healthcare_sim_sdk/scenarios/nurse_retention/EXPERIMENT_DESIGN.md` | rationale | The design reasoning behind the nurse-retention experiments. |
| `reports/` | record | Dated reports. The four revised reports of 16 October 2026 supersede anything earlier. |
| `research/` | record | The April 2026 paper audit and teaching packet. Reproduction labels there are superseded; see "Published-Study Calibration Exercises" in `docs/design_principles.md`. |

## Mixed documents

Two documents hold more than one class. The map gives each its main class; the sections below are the exceptions.

| Document | Section | Class |
|---|---|---|
| `docs/algorithm_description.md` | 15, Companion notebooks | proposal |
| `docs/algorithm_description.md` | Appendix A, Defect history; Appendix B, Issues register | record |
| `docs/design_principles.md` | Appendix B, What's Missing That We Should Apply | proposal |

## Not markdown

- Code and tests are the final authority on behavior.
- GitHub issues track open problems and proposals. A closed issue is a record.
- Commit messages and pull-request descriptions are records.
- `outputs/` is never committed and has no authority.
