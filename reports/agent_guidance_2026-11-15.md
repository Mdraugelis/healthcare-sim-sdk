# Agent guidance: before and after (issue #9)

**Date:** 15 November 2026. **Before:** `main` at `c5f4bf1`. **After:** the four changes for issue #9.

This is a record. It measures the change against the five principles in the issue and lists what is still open. Current rules are in [`AGENTS.md`](../AGENTS.md) and [`docs/AUTHORITY.md`](../docs/AUTHORITY.md).

## Summary

| Measure | Before | After |
|---|---|---|
| Root guidance loaded in every session | 112 lines (`CLAUDE.md`) | 48 lines (`AGENTS.md`); limit of 80 enforced |
| Documents with a stated class | 0 | Every tracked markdown file |
| Scenario folders with local guidance | 1 of 14 | 14 of 14, plus `core/` |
| Checks that read `invariants.yaml` | 0 | The invariant hook and 12 parity tests |
| Documentation rules in the gate | 0 | 9 (DOC001 to DOC009), 34 tests |
| Places the fitness criteria were listed, and whether they agreed | 3, all different | 3, held equal by a test |
| Registered Claude Code agents | 0 (`sim-guide.md` had no frontmatter) | 1 (`sim-guide`) |

## By principle

### Canonicality

| | |
|---|---|
| Before | No map of what is current. The README and `design_principles.md` said sessions load the protocol automatically; they load the root file only. `design_principles.md` described four phases and four levels; the protocol has six and five. |
| After | `docs/AUTHORITY.md` gives every markdown file one class: current, rationale, proposal or record. DOC006 fails when a file has none. DOC002 holds documents classed `current` to paths that exist. The loading claims are corrected. |
| Still open | Two documents are mixed and are split by section in prose, not by a check. `design_principles.md` still carries historical numbers under a warning. |

### Localization

| | |
|---|---|
| Before | One scenario folder had a README. Everything else an agent needed was in the root file or in long documents. |
| After | Each scenario folder has an `AGENTS.md` of at most 26 lines: status, what it models, entry points, how to verify, private files, known limits. `core/` has one for the invariant rules. The template carries the scaffold, and a copy left unfilled fails DOC004. |
| Still open | `ml/`, `experiments/`, `population/` and `tests/` have none; add one when a rule turns up that belongs only there. The checks confirm that paths, runner modules and test files exist. They do not confirm that the prose is right. |

### Verifiability

| | |
|---|---|
| Before | Invariant hooks, lint rules, the test suite and the five-level protocol existed. No check read `invariants.yaml`; the hook kept its own copy of the contract. |
| After | The hook reads `config/invariants.yaml`. Parity tests hold the code, the lint exemptions and the restated counts to it, including stream order. `scripts/check_docs.py` runs as a gate step and a pre-commit hook. Mandatory verification is still in the root file, as five named levels. |
| Still open | Runner commands in local guidance are checked to exist, not executed. |

### Interoperability

| | |
|---|---|
| Before | All shared guidance lived under Claude-specific names: `CLAUDE.md` and `.claude/`. |
| After | Three tool-neutral sources: `AGENTS.md`, `docs/simulation_protocol.md`, `config/invariants.yaml`. Four thin adapters point at them; the table is in `docs/AUTHORITY.md`. DOC003 and DOC008 check the adapters, and DOC009 stops the old paths from returning. |
| Still open | Loading is confirmed in Claude Code. Cowork and Codex are still to be checked; see the next section. A checkout without symlink support needs `CLAUDE.md` replaced by the one-line import `@AGENTS.md`, which DOC003 accepts. |

### Default-no

| | |
|---|---|
| Before | 112 lines loaded in every session, repeating the README's overview and directory list and three paragraphs of the analytical-first document. |
| After | 48 lines. The protocol (363 lines) is read on demand. The path-scoped rule loads only when a file under `healthcare_sim_sdk/` is read. Limits are enforced: 80 lines for root, 40 for a scenario file. |
| Still open | None known. |

## Loading smoke checks

These need a signed-in tool, so they cannot run in the gate. Each row is filled in when its check is run. One of four is done.

| Tool | How to check | Expected | Result |
|---|---|---|---|
| Claude Code | `scripts/smoke_claude_loading.sh` | `AGENTS.md` loaded at session start through `CLAUDE.md`; `.claude/rules/nested-agents.md` loaded when a scenario file is read | **Pass**, 15 November 2026, Claude Code 2.1.325 on macOS. Loaded at session start: `AGENTS.md`, reached through the `CLAUDE.md` symlink. Loaded when the scenario file was read: `.claude/rules/nested-agents.md` (`path_glob_match`). The user's own `~/.claude/CLAUDE.md` also loaded, as expected. |
| Claude Code, agent | `claude --agent sim-guide` | `@sim-guide` in the startup header | Not yet run |
| Cowork | Start a session with the repository folder connected | The root rules are in context without being asked for | Partly observed. The device bridge reads `CLAUDE.md` through the symlink and returns the content of `AGENTS.md` (15 November, macOS clone outside Dropbox). Loading at session start has not been observed on the symlink. On `c5f4bf1`, a session loaded the regular-file `CLAUDE.md` automatically and did not load the protocol. |
| Codex | Ask which instruction files it loaded from the root and from one scenario folder | Root `AGENTS.md`, and the scenario's `AGENTS.md` | Not yet run. Nested loading is assumed from the `AGENTS.md` convention, not verified. |

The path-scoped rule loads in Claude Code, so no `CLAUDE.md` symlinks are needed beside the nested `AGENTS.md` files. The root file's instruction to read local guidance applies in every tool regardless.

## Outside this repository

Three other workspaces name the old paths. The compatibility adapters keep them working. They should be updated to the new paths, after which `.claude/invariants.yaml` can be removed:

- the Catalyst repository's agent file, loading contract item 4;
- the Cowork `healthcare-sim-sdk` skill, step 2;
- the AI programs agent file, where it names the `sim-guide` agent.

## Checks on this change

Recorded in the pull requests: the quality gate on each of the four commits.
