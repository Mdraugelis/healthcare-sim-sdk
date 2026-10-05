# Evidence-First Testing Principles

A portable specification for AI-developed engineering projects.

These principles were worked out while building and reviewing the Healthcare
Intervention Simulation SDK. They are written here without SDK-specific
dependencies so that other projects, starting with the HL7 simulator that
will drive the HL7 data pipeline, can adopt them directly. Each principle
names the failure it exists to catch. A worked HL7 application follows the
principles, and an adoption checklist and a paste-ready agent instruction
block close the document.

The goal throughout is **justified confidence in a specific claim**, not a
badge that a system is beyond question. A test suite is a map from claims
to the error classes that would falsify them. Its value is what it can
detect, not how many assertions it contains.

---

## Part A. What a test is for

### P1. Confidence begins with a claim

Every check exists to establish a specific claim about the system:
"assessments respect capacity", "event histories reconcile per entity",
"the parser preserves patient identity across a merge". Write the claim
first, then the evidence that would support it, then the error it would
detect.

- **Why.** A long list of passing tests establishes a collection of local
  properties, not an assurance argument. Reviewers cannot tell what a green
  run proves unless each check is tied to a claim.
- **Apply.** Keep a claim registry (YAML or Markdown) alongside the code.
  Each claim states its scope, denominator, comparison, required evidence,
  acceptance criterion, and the reviewer role that owns it. Completeness
  means every required claim has applicable evidence or an explicit
  unresolved status.
- **Catches.** "All tests pass" while the decision-critical claim was never
  tested.

Distinguish *verification* (does the implementation follow the intended
model?) from *validation* (is the model accurate enough for its intended
use?). A clinically implausible effect can be implemented perfectly; a
plausible mechanism can be coded incorrectly. Neither word means universal
correctness.

### P2. Expected answers come from outside the implementation

A test's expected value must be derived independently: worked by hand, taken
from a published reference, or computed by a deliberately different method.

- **Why.** A test that calls the production function twice and asserts the
  values agree proves determinism, not correctness. If implementation and
  "reference" omit the same factor, the test passes and the result is wrong.
- **Apply.** Write golden examples small enough for a reviewer to check by
  hand. Use distinct IDs, differing outcomes, and staggered timing so that a
  positional join or a one-period shift visibly changes the answer. Two
  patients with risks 0.8 and 0.2, one treatment slot, treatment halves
  risk: expected events fall from 1.0 to 0.6. Check identity, probability,
  and outcome against that number.
- **Catches.** Shape-only tests (right length, right keys) that pass while
  exposure timing or the outcome column is wrong.

### P3. Layers detect different mistakes

Structure, accounting, mechanism, statistical behavior, and external review
each catch an error class the others miss. Coverage is a map from claims to
error classes, not a count.

| Layer | Example question | What it cannot supply alone |
|---|---|---|
| Structure | Are contract boundaries respected? | Domain plausibility |
| Accounting | Are all entities and events represented, with the right identities? | Correct effect size |
| Mechanism | Does a known input produce the intended change under controlled conditions? | Generalization |
| Statistical behavior | Does an estimator recover planted truth across replications? | Validity of the generating assumptions |
| External and human review | Are the assumptions fit for this decision? | Proof the code executed correctly |

- **Why.** A source-pattern checker catches a prohibited random call and
  misses a wrong denominator. A perfect total conceals events assigned to
  the wrong patient. A met calibration target coexists with a wrong
  treatment effect.
- **Apply.** For each claim, name which layer supplies its evidence. If a
  layer is missing for a required claim, the claim's status is NOT_RUN, not
  PASS.

---

## Part B. Invariants and architecture

### P4. If a principle is worth stating, it is worth enforcing mechanically

Architectural rules live in a machine-readable file and are enforced by
pre-commit hooks, AST guards, and CI. Human discipline is not the
enforcement mechanism.

- **Apply.** Maintain an `invariants.yaml` (or equivalent) declaring
  protected files, required interface methods, required randomness streams,
  banned patterns, and engine constraints. Have the checker *read* that file
  rather than hardcoding the rules, so definition and enforcement cannot
  drift. Changing an invariant is a design decision made by RFC, never in a
  feature branch.
- **Test the guards.** Keep intentionally invalid source fixtures for every
  prohibited pattern and assert that the checker rejects each one. A guard
  that has never been seen to fail is not known to work.
- **Catches.** Contract regressions that documentation alone would let
  through; a checker whose description is broader than its assertions.

### P5. Determinism and partitioned randomness make comparisons testable

Given a seed, a run produces byte-identical output. Randomness is drawn
from named, independent streams, one per process (population, temporal
dynamics, prediction, intervention, outcomes in the SDK), never from a
global generator.

- **Why.** Paired comparisons (factual vs counterfactual, baseline vs
  treated, clean vs corrupted) are only valid when the two arms consume the
  same draws for the same purposes. One extra draw in one arm silently
  desynchronizes everything after it.
- **Apply.** Ban bare global random calls by pattern. Test stream
  independence. Run the single most important identity test: the same
  scenario in paired mode and single mode with the same seed must match
  element-wise at every timestep. Then go further: after the two arms
  *diverge* (one has an extra departed entity, a different eligible set),
  verify generator state still aligns. Use fixed-shape draws followed by
  masking, or randomness keyed by entity and time, so state-dependent
  lengths cannot shift the stream.
- **Catches.** Shared mutable state leaking between arms; the wrong stream
  used inside a pure step; state-dependent draw counts that break pairing
  without breaking the mean.

### P6. Opaque state and narrow contracts

The core engine never inspects domain state. Domain logic lives behind a
small, fixed interface, and every scenario or adapter is self-contained.

- **Why.** The SDK accepted scalar risks, nested dataclasses, and 16-row
  arrays per admission through the same five-method contract without a
  single engine change. A shared schema would have forced each new domain
  to compromise or to modify the core.
- **Apply.** Enforce state opacity mechanically (no attribute, subscript,
  `isinstance`, or `getattr` access on state variables in core). Keep the
  contract's method count fixed; if a new feature seems to need a sixth
  method, the design needs rethinking or the tool does not fit.
- **Catches.** Engine changes that quietly couple the core to one domain's
  representation.

---

## Part C. Verifying every run

### P7. Verify before you interpret, every run, no exceptions

Every run, including quick analytical passes and demos, is verified before
any result is read. If the human asks to skip verification, explain why it
matters and run it anyway. A plausible-looking misconfigured run is worse
than no run, because it manufactures false confidence.

The five levels, run in order:

1. **Structural integrity.** Population and event accounting reconcile;
   arrays match declared units and IDs at every observed step; eligible
   values are finite; documented sentinels appear only in their declared
   fields; scores are in bounds; confusion-matrix cells sum to the number
   of aligned, eligible, observed labels.
2. **Statistical sanity.** Realized rates fall within 4-sigma CLT bounds of
   configured rates; achieved model metrics are within tolerance of targets
   with the gap reported; process dynamics match theory; subgroup
   proportions match targets.
3. **Conservation and monotonicity.** See P8.
4. **Case-level walkthroughs.** See P10.
5. **Boundary conditions** on the first run of any new configuration. See
   P9.

Report the result as a verification summary that a non-engineer can read:
what passed, what flagged, what it means, and what warrants attention.
Failed structural checks block interpretation entirely.

### P8. Conservation laws and metamorphic relations need stated assumptions

A metamorphic test changes an input and asserts a known relation between
outputs. The relation must actually follow from the mechanism, under the
coupling the test uses.

- **Why.** "Higher AUC reduces the disparity" does not hold on every finite
  seed; capacity, contact, and outcome noise also move the gap. "Stronger
  treatment reduces every downstream count" fails under competing risks or
  changing eligibility. A test asserting an unsupported relation is not a
  correctness test; it is a wrong expectation waiting to fail.
- **Apply.** Classify each check as one of: an *exact invariant*, an
  *expected direction across independent repetitions with uncertainty*, a
  *scenario assumption*, or an *informational diagnostic*. Attach its
  applicability, tolerance rationale, and exclusions. Assert pathwise
  monotonicity only where the mechanism guarantees it; otherwise assess
  direction across seeds with an interval.
- **Reading a failure.** A failed metamorphic test means one of three
  things: an implementation defect, an incorrect expected relation, or an
  unstated assumption. Examine all three before changing anything.
- **Catches.** Real logic errors (accounting identities, capacity
  overruns) and, just as importantly, tests that encode folklore.

### P9. Boundaries are claim-specific

At the extremes, specify exactly what must hold and what may not.

- **Apply.** Threshold zero flags every *eligible* entity; it does not treat
  everyone under a capacity limit. Threshold one flags nobody, unless the
  comparison is inclusive and scores can equal one. A random score is
  compared against an equal-resource random policy, because extra
  resources can help even at AUC 0.50. Zero efficacy equalizes the outcomes
  that the disabled channel governs; assessment burden, reordering, and
  capacity effects may remain. Maximum efficacy cannot rescue events
  outside its eligibility or timing window.
- **Report** a claim-by-claim table with PASS, FAIL, NOT_RUN, INCONCLUSIVE,
  or NOT_APPLICABLE and a rationale for every inapplicable entry.
- **Catches.** Universal boundary expectations that fail for a valid reason
  and get "fixed" by weakening the test.

### P10. Walk individual cases as stories

Trace three to five entities through the full path, from initial state
through each transformation to final outcome on every arm, and present
them in narrative form rather than as arrays.

- **Why.** Aggregates hide mechanism. A human builds intuition about
  whether the system captures reality by following one entity and asking
  whether each step makes sense.
- **Apply.** Draw a seeded random sample plus mechanism-selected cases:
  one helped, one harmed or displaced, one unchanged, one with incomplete
  follow-up. Disclose the selection rule, the candidate count, and any
  category that had no example. Generate stories from structured traces of
  the *actual reviewed run*, and if a separate trace runner is used, assert
  its output matches the engine exactly. Where a decision detail was not
  recorded, say so instead of reconstructing it.
- **Catches.** Illustrative stories from a different default run presented
  as evidence about the run under review.

---

## Part D. Testing the tests

### P11. Inject faults to prove the checker can see

For each verification path, corrupt a known-good artifact in a specific way
and assert that the check fails. Pair every fault with an unmodified
control and assert that the control passes.

- **Fault menu.** Reverse entity IDs without moving outcomes. Move one
  event between entities while preserving the total. Remove a period. Add
  one action beyond capacity. Shift treatment start by one period. Omit a
  seed. Duplicate an entity. Swap the model artifact after a run starts.
  Alter a single number in a report. Desynchronize one random draw.
- **Why.** A total-only reconciliation cannot see an event reassigned
  between patients. Fault injection is the only way to learn where the
  verification path is blind.
- **Apply.** Keep deliberately broken implementations in the test suite
  (an impure step, a step using the wrong stream) and assert the purity
  test rejects them. Parameterize faults so each case is explicit.
- **Catches.** A checker that accepts everything.

### P12. Name tests by what they actually accept

A test's name, docstring, and any report language must match its
acceptance rule.

- **Why.** A test named `auc_near_05` that accepts anything below 0.80
  guards against high discrimination, not closeness to chance. A validator
  that reports six of six checks passed on a metrics file containing a rate
  of 2.0 and a NaN describes artifact structure, not correctness.
- **Apply.** Distinguish artifact-structure checks from simulation, model,
  analysis, and review checks in the name and the report. Tighten the
  assertion or rename the test; never leave the two mismatched.
- **Catches.** Confidence borrowed from a test's name.

---

## Part E. Statistics and uncertainty

### P13. Statistical checks carry their own uncertainty

A stochastic test compares an observed statistic against an expectation
with a stated tolerance, and a certification gate passes only when the
whole Monte Carlo interval satisfies the requirement.

- **Apply.** Use 4-sigma CLT bounds for means and rates (false-positive
  rate below 0.007 percent per check, so repeated runs stay quiet). Use
  exact binomial (Clopper-Pearson) intervals for detection and false-alarm
  rates: zero false alarms in twenty runs has a one-sided 95 percent upper
  bound near 14 percent and is not evidence of a zero rate. Report the
  Monte Carlo standard error of every simulated mean. If the interval
  overlaps the requirement, the status is INCONCLUSIVE, not PASS.
- **Declare before reading.** Follow the ADEMP structure: aims,
  data-generating mechanisms, estimands, methods, performance measures,
  all written before results are examined.
- **Keep failed fits.** A repetition that failed to fit is recorded and
  fails its arm. Metrics are never summarized over only the successful
  repetitions.
- **Catches.** Flaky tests with arbitrary tolerances; power claims from
  thirty repetitions presented as precise; a design with informative
  censoring that shows excellent apparent power while failing bias.

### P14. Keep four kinds of uncertainty separate

1. Variation across simulated runs (seed spread; empirical percentiles).
2. Monte Carlo precision of the estimated mean (standard error, t
   interval; undefined for one seed).
3. The interval a study estimator would produce on observable data.
4. Uncertainty about input assumptions, which more seeds never reduce.

- **Why.** Calling the 2.5th and 97.5th seed percentiles a confidence
  interval conflates the first two. Rows missing the requested metric that
  silently drop out turn a two-seed summary into a one-seed summary without
  an error.
- **Apply.** Label every interval by kind. Reject missing seeds and
  undefined estimands instead of summarizing the data that happens to be
  present.
- **Catches.** Seed spread reported as evidence precision; missing results
  improving a summary.

### P15. Separate tuning from validation

Seeds, cohorts, and periods used to fit a model, choose a threshold, or tune
a monitor are never reused to demonstrate performance.

- **Why.** Reusing the same realization to tune and to certify makes the
  evidence optimistic. Calibrating to reproduce one target is not
  independent validation of that target.
- **Apply.** Hold out validation seeds and cohorts. Reserve additional
  quantities, populations, or periods for validation where possible. For
  parameters with no external evidence, show sensitivity and structural
  alternatives. Where an evaluation must be blind, freeze the inputs,
  record custody receipts, and fail closed when isolation cannot be proven.
- **Catches.** Self-fulfilling certifications.

---

## Part F. Evidence, provenance, and human review

### P16. Five evidence states, not a score

Every claim carries exactly one of:

| State | Meaning |
|---|---|
| PASS | The declared, applicable check ran and met its criterion. |
| FAIL | Executed evidence contradicted a requirement, or an execution or integrity check failed. |
| NOT_RUN | Required evidence is absent or its observation window has not completed. |
| INCONCLUSIVE | Evidence exists but is too weak, sparse, or incompatible to settle the claim. |
| NOT_APPLICABLE | The claim does not apply in this scope. A rationale is mandatory. |

- **Rules.** The states are not ordinal and are never averaged. Nine passes
  do not compensate for one failed identity check. A required claim cannot
  be satisfied by an empty check list, an optional check, a missing metric,
  a missing seed, or a stale certificate. Malformed or non-finite evidence
  files fail. A validation command returns a non-zero exit code when
  required checks fail, and a distinct code for INCONCLUSIVE.
- **Catches.** "Checks passed" appendices that meant only "files exist and
  parse".

### P17. Bind evidence to the exact thing being reviewed

Every run writes a manifest: source fingerprint (including uncommitted
changes), resolved configuration hash, model or artifact bytes hash,
environment versions, seed list, expected cells, completed cells, missing
and failed cells, checks executed, skips, and artifact hashes.

- **Why.** A certificate that outlives a code, model, or configuration
  change silently validates something it never examined. A verification
  run on the center cell and first seed says nothing about the other eight
  cells.
- **Apply.** Run cheap integrity checks on every actual cell and seed.
  Reuse expensive certification only when its declared applicability
  identity matches the current run. Never certify a sweep with a separately
  rerun cell. State the trust boundary: hashes detect change relative to a
  trusted record; they are not signatures and do not protect against
  someone who rewrites both artifact and manifest.
- **Catches.** Stale evidence; a passed check against older bytes presented
  as current.

### P18. Preserve the record, including failures and skips

Original logs are retained as they were produced. A run with one failure is
not rewritten as all-green after a fix; the fix produces a new record.

- **Apply.** Report skipped required evidence as prominently as failures.
  When tests are skipped because private inputs are unavailable, CI shows
  the skip. Do not add counts from overlapping runs. State explicitly when
  a rerun is selective rather than a full-suite execution.
- **Catches.** History that looks cleaner than what happened.

### P19. Human review is a separate, role-specific act

Machine evidence is prepared for named reviewer roles who each own
specific claims. A generated PASS is never a human approval, and approval
is never inferred from silence.

- **Roles.** Domain (are mechanisms and usual practice represented?),
  operations (are eligibility, timing, and resource limits realistic, and
  who loses access?), statistics (what is the estimand, denominator, and
  comparison?), governance and equity (who benefits, who bears burden,
  which groups are too sparse to assess?), engineering (can these exact
  results be reproduced and trusted as artifacts?).
- **Apply.** The roles share one evidence record; they do not create five
  independent validations. Human notes are mutable, excluded from machine
  hashes, and bound to the packet identity; a changed model or config
  invalidates applicability and requires a renewed decision. Disagreement
  between roles is recorded with scope and evidence, never averaged. An
  AI assistant prepares evidence and questions; it does not sign.
- **Catches.** Software accounting that passes standing in for an
  unresolved domain assumption.

### P20. Assumptions surfaced, findings conditional, decision human

The code is the assumption list. Every parameter that affects a result is
in a reviewed file, never a hidden default. Findings are framed as "under
these assumptions, the system suggests", never as real-world evidence. The
analysis names what it cannot answer.

- **Apply.** Before building, ask whether the question collapses to
  closed-form math; when it does, do that pass first and attach the same
  verification, plus an explicit list of what the pass cannot address.
  When a result looks clean, explain what produced the cleanliness. When
  it looks surprising, distinguish misconfiguration from insight. Ask what
  would change the decision, which assumption carries that sensitivity,
  and which observation would reduce the uncertainty.
- **Catches.** "The simulation said so" used as a justification.

---

## Worked application: the HL7 simulator

The simulator generates realistic HL7 v2 message traffic (ADT, ORM, ORU)
with configurable population, timing, content variation, and fault
injection, and drives the development of the data pipeline that parses,
validates, and lands those messages. The same principles apply; only the
entities change.

**Claims (P1).** Write these before the first line of generator code.

| Claim | Evidence | Owner |
|---|---|---|
| Emitted messages conform to the declared profile | Profile validation of every emitted message; hand-written golden messages | Engineering |
| Patient identity survives the pipeline, including A40 merges and A08 updates | Per-patient event-history reconciliation, generator ledger to landed tables | Data / integration |
| Arrival timing matches the configured process | Inter-arrival distribution within 4-sigma bounds; per-hour counts reconcile | Statistics |
| Malformed messages are rejected without dropping valid ones | Conservation: in = parsed + rejected + quarantined, every message accounted by control ID | Operations |
| Pipeline output is deterministic for a given seed | Byte-identical landed tables across two runs | Engineering |

**Golden examples (P2).** Hand-author a minimal ADT^A01, A03, A08, and A40
set for two patients with distinct MRNs, non-consecutive control IDs, and
staggered MSH-7 timestamps. Write the expected landed rows by hand. Assert
identity, timestamp, and field-level values, not row counts.

**Partitioned randomness (P5).** Separate streams for population
(demographics, MRNs), event timing (admits, transfers, discharges),
content variation (names, addresses, free-text), and fault injection
(corruption, duplication, reordering). A run with fault injection disabled
must produce the identical clean stream as a run that never had it,
because the clean streams never see the fault stream's draws.

**Conservation and monotonicity (P8).** Every control ID appears exactly
once in the union of parsed, rejected, and quarantined. Per-patient event
counts in landed tables equal the generator ledger. Raising the corruption
rate never decreases rejections across seeds (an expected direction with an
interval, since corruptions can land in fields the validator ignores).

**Boundaries (P9).** Corruption rate zero: landed output equals the golden
run exactly. Corruption rate one: every message rejected or quarantined,
none landed. Zero arrivals: all tables empty and the run completes. A
message with only MSH: rejected with the declared reason, not an exception.

**Case stories (P10).** Trace three patients through generator, transport,
parser, and landed tables: one uncomplicated admit-transfer-discharge, one
merge where the surviving MRN absorbs the prior history, and one whose
A08 was corrupted and quarantined. For each, show every message, its
control ID, what the pipeline did, and what the warehouse holds at the end.

**Fault injection (P11).** Swap PID-3 between two messages keeping counts
equal; duplicate a control ID; reorder two messages so MSH-7 runs
backward; truncate an OBX segment; change the MSH-2 encoding characters;
drop a required PV1 field. Each fault must be caught by a named check, and
each control message must pass.

**Evidence and identity (P16, P17).** Each simulator run writes a manifest
with generator source hash, profile and configuration hash, seed list,
expected and produced message counts, and landed-table hashes. Pipeline
validation reads that manifest; a pipeline validated against yesterday's
generator output is NOT_RUN for today's.

**Review (P19).** Interface analysts own the profile-conformance and
identity claims. Pipeline engineers own determinism and conservation. A
data-quality reviewer owns the fault taxonomy and what counts as
"rejected correctly". Their sign-off lives in `human_review.json`, never
in the test report.

---

## Adoption checklist for a new project

1. Write the claim registry before the first feature (P1).
2. Create `invariants.yaml` and a checker that reads it; add pre-commit and
   CI; add negative fixtures for every rule (P4).
3. Partition randomness into named streams; ban global random calls; add
   the paired-vs-single identity test and a divergent-state alignment test
   (P5).
4. Hand-author golden examples with distinct IDs and staggered timing
   before writing the code they test (P2).
5. Implement the five-level verification and its summary template; wire it
   to run after every run (P7).
6. Classify every conservation or metamorphic check by kind and record its
   assumptions (P8, P9).
7. Build case tracing from actual run records and present stories, with
   the selection rule disclosed (P10).
8. Write the fault menu and pair each fault with a control (P11).
9. Use CLT and exact binomial tolerances; report Monte Carlo standard
   error; separate tuning and validation seeds (P13, P14, P15).
10. Adopt the five evidence states, non-zero exit codes, and run manifests
    with identity binding (P16, P17).
11. Retain original logs; report skips beside failures (P18).
12. Define reviewer roles and a separate human-review file (P19).
13. Frame all findings conditionally; keep every assumption in reviewed
    files (P20).

---

## Paste-ready agent instructions

Copy this block into the CLAUDE.md of a project that adopts these
principles.

```markdown
## Testing principles (evidence-first)

- Every check establishes a named claim from the claim registry. Map
  claims to the error classes they detect, not to assertion counts.
- Expected values come from outside the implementation: hand-worked
  examples with distinct IDs and staggered timing. Never test a function
  against itself.
- Architectural rules live in `invariants.yaml` and are enforced by hooks
  and CI. Keep negative fixtures that prove each guard rejects a violation.
- All randomness flows through named, partitioned streams. No global
  random calls. Same seed, identical output.
- Verify before interpreting, every run, no exceptions: structural
  integrity, statistical sanity (4-sigma CLT), conservation and
  monotonicity with stated assumptions, 3-5 narrative case walkthroughs
  from the actual run, and boundary conditions on first runs. Present a
  verification summary.
- Inject faults to prove checks can fail; pair each fault with a passing
  control. Name tests by what they actually accept.
- Report statistical results with their own uncertainty (exact binomial,
  Monte Carlo standard error). A gate passes only when the whole interval
  meets the requirement; otherwise INCONCLUSIVE.
- Keep seed spread, Monte Carlo precision, study-estimator intervals, and
  assumption uncertainty separate and labeled. Never drop missing seeds.
- Every claim carries exactly one of PASS, FAIL, NOT_RUN, INCONCLUSIVE,
  NOT_APPLICABLE (with rationale). States are never averaged.
- Every run writes a manifest binding evidence to source, config,
  artifact, environment, seeds, and expected versus completed cells.
  Stale evidence is NOT_RUN.
- Retain original logs. Report skipped required evidence as prominently
  as failures.
- Human review is role-specific and recorded separately. A machine PASS
  is not approval; silence is not approval. The assistant prepares
  evidence and never signs.
- Frame findings as conditional on stated assumptions. Surface every
  assumption. The human makes the decision.
```

---

## References

- Sargent, R. G. (2011). Verification and validation of simulation
  models. *Proceedings of the Winter Simulation Conference.*
- Morris, T. P., White, I. R., and Crowther, M. J. (2019). Using
  simulation studies to evaluate statistical methods. *Statistics in
  Medicine.* (The ADEMP framework.)
- In this repository: `.claude/agents/sim-guide.md` (verification
  protocol), `.claude/invariants.yaml` (machine-readable invariants),
  `docs/review_packets.md` (evidence states and role views),
  `docs/analysis_design_certification.md` (Monte Carlo gates),
  `tests/integration/test_step_purity.py` (identity test with deliberate
  violations), `tests/bulletproof/conftest.py` (statistical assertion
  helpers).
