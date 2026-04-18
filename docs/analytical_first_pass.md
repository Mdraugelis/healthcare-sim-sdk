# Analytical-First Pass — Worked Protocol and Example

## Why This Document Exists

The Healthcare Intervention Simulation SDK is a Monte Carlo engine. It exists for problems that need entity-level fidelity — per-seed variance, operator dynamics, score-distribution-dependent decisions, time-to-action coupling. Many real decision questions about AI deployment do not need any of that. They are answerable in closed form with a page of Python.

This document describes the **Analytical-First Principle**: before building a scenario, evaluate whether the primary decision question can be answered with closed-form math. If it can, do the analytical pass first. The SDK is the wrong tool for a problem that a spreadsheet can answer.

The principle applies whether the user is a scenario author, a program manager designing a deployment evaluation, or an agent (`sim-guide`) helping a human explore a new use case.

---

## The Protocol

### Step 1 — Write the Decision Question as a Sentence

Before any code, write one sentence that starts with "We will deploy / not deploy / escalate / stop this AI if …". If the sentence cannot be written, stop and go back to the stakeholder. You are not ready to simulate anything.

### Step 2 — Identify the Causal Chain

For predictive-model-driven interventions, the chain is almost always:

```
factual_rate(t) = baseline_rate(t) × [1 − sensitivity × action_rate × efficacy]
counterfactual_rate(t) = baseline_rate(t)
```

where:

- **baseline_rate** is the event rate absent any AI
- **sensitivity** is the model's recall at the chosen operating point
- **action_rate** is the probability a flagged case actually gets acted on (alert fatigue, capacity, overnight staffing)
- **efficacy** is the probability that action, once taken, prevents the event

If your use case's chain is multiplicative and its links are point estimates, you are a candidate for the analytical pass.

### Step 3 — Check Fitness for Closed Form

Closed-form is sufficient when **all of the following are true**:

- The decision question is about *whether* an effect is detectable or *how big* it is, not about the dynamics that produce it.
- Power is governed by sample size, base rate, and effect size — not by per-seed variance, queueing, or capacity dynamics.
- Equity can be expressed as *deltas* on the same point estimates (Group B sensitivity is X pp lower; Group B baseline is X% higher).
- The outcome series' autocorrelation is roughly known (use a published derate — e.g. Cruz 2019's 0.85 for monthly hospital outcomes).
- Threshold selection is *not* on the table for this pass (sensitivity is taken as a fixed input, not derived from the score distribution).

If any fails, escalate to Phase 1 (SDK). See the "When You Must Escalate" list below.

### Step 4 — Write the Sweep

For a factorial design with factors A, B, C, D, E, the sweep is a handful of nested loops around `itertools.product` and the two-proportion z-test. A complete working example is in `__AI_Programs_Agent/simulation_experiments/2026-04-13-perigen-efm-early-warning/run_analytical_sweep.py` (~200 lines total, no SDK imports).

The two-proportion power formula (with autocorrelation derate) is:

```python
def power(p1, p2, n1, n2, alpha=0.05, derate=0.85):
    ALPHA_Z = 1.959964  # two-sided 0.05
    se_null = math.sqrt(p1 * (1 - p1) * (1 / n1 + 1 / n2))
    se_alt  = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    eff = abs(p1 - p2)
    z_need  = ALPHA_Z * se_null / se_alt
    z_shift = eff / se_alt
    return (1 - ncdf(z_need - z_shift) + ncdf(-z_need - z_shift)) * derate
```

### Step 5 — Run Validation

**Closed-form answers are still subject to the full verification protocol.** See `.claude/agents/sim-guide.md` Phase 2 "Doing the Analytical Pass Well" for the seven-step checklist. In brief:

1. Bayes constraint: PPV ≤ (sens × prev) / (sens × prev + (1 − spec) × (1 − prev))
2. Monotonicity: lower threshold → more flags; higher efficacy → fewer events; effectiveness = 0 → factual = counterfactual
3. Boundary spot-check: threshold = 0, threshold = 1, AUC = 0.5, effectiveness = 0
4. CLT sanity at configured baseline rate and planned sample size
5. Sensitivity sweep ±10–25% around each point estimate the decision rests on
6. Explicit "what this cannot answer" gap list
7. Counterfactual baseline computed explicitly, not dropped from the algebra

### Step 6 — Write the Gap List

Every Phase-0 results doc must name what it cannot answer. At minimum:

- Per-seed variance (can't estimate without Monte Carlo)
- Operator dynamics (alert fatigue half-life, capacity ceilings, queueing)
- Time-to-action coupling (does earlier alert → more lead time → more efficacy?)
- Score-distribution-dependent effects (threshold selection, ranking under capacity, calibration, subgroup calibration)
- Analytic-method rehearsal on synthetic entity-level data

If any of these matter for the decision, escalate to Phase 1.

### Step 7 — Decide Whether to Escalate

If the Phase-0 gap list contains items the human's decision rests on, run Phase 1. Phase 1 is **not** automatic — it is justified by gaps Phase 0 surfaced. Many studies legitimately stop at Phase 0.

---

## When a Closed-Form Pass Is Sufficient

- Power and minimum detectable effect (MDE) for a planned ITS / DiD
- Causal-chain feasibility checks (sensitivity × action_rate × efficacy)
- Bayes-bounded PPV / NNA at fixed prevalence
- Coarse parameter sweeps where decisions hinge on point estimates
- Equity arms expressed as deltas on the same point estimates

## When You Must Escalate to the SDK

You need entity-level Monte Carlo when **any** of the following are true:

- **Per-seed variance matters.** The decision rule is sensitive to realized variance, not just expected value.
- **Operator dynamics matter.** Alert fatigue half-life, capacity ceilings, queueing, score-dependent intervention efficacy.
- **Distributional tails matter.** The decision depends on the worst 5% of cells, not the mean.
- **The analytic method itself needs rehearsal.** Dry-run the planned ITS / DiD / RDD on synthetic entity-level data with known ground truth before committing to it on real data.
- **Score-fidelity-dependent decisions.** The decision depends on the *distribution* of model scores, not just their summary statistics. Specifically:
  - *Threshold selection* — choosing the operating point needs the score histogram.
  - *Ranking under capacity* — top-K prioritization depends on score *order*.
  - *Calibration-dependent use* — score consumed as a probability needs slope/intercept.
  - *AUC-to-PPV translation across operating points* — requires the joint score distribution.
  - *Subgroup calibration* — per-subgroup score distributions, not a single point estimate.
  - *Score-dependent efficacy* — intervention works better on higher-scoring patients.
  - *Analytic method rehearsal* — pre-registering an ITS / DiD / RDD that uses the score directly.

### Quick heuristic

If the decision question can be answered with point estimates of sensitivity, PPV, and prevalence, closed-form suffices. If it requires the score *distribution* — because a threshold is being chosen, a ranking is being used, a calibration is being trusted, or a subgroup is being stratified — the SDK's `ControlledMLModel` is required.

---

## Worked Example — PeriGen EFM Early-Warning Study

A concrete reference implementation of the Analytical-First Principle. Full source lives outside the SDK, in the Geisinger AI Programs repo under `simulation_experiments/2026-04-13-perigen-efm-early-warning/`.

### The Decision Question

"Does Geisinger have sufficient volume and evaluation design to determine whether PeriGen PeriWatch Vigilance reduces neonatal HIE (hypoxic-ischemic encephalopathy) at a pre-specified effect size, and can we do so without widening existing disparities?"

This is a deployment-evaluation question, not a deployment-optimization question. The vendor's model operating point is fixed. The evaluation will be an Interrupted Time Series (ITS) over 9,000 annual deliveries at a pre-period of 36 months and post-periods of 12 / 18 / 24 months.

### What Was Answerable Analytically

The 243-cell factorial sweep (A × B × C × D × E) ran as pure Python, no SDK imports:

- **A — baseline HIE rate** (0.030, 0.056, 0.112) — literature-derived
- **B — model sensitivity** (0.60, 0.75, 0.90) — vendor claim and stress tests
- **C — action rate** (0.40, 0.70, 0.95) — alert-acknowledge-act cascade
- **D — efficacy** (0.20, 0.40, 0.60) — clinical-team action → HIE reduction
- **E — post-period length** (12, 18, 24 months)

Causal chain:

```
factual_rate   = A × (1 − B × C × D)
counterfactual = A
```

Power: two-proportion z-test with ITS autocorrelation derate of 0.85 (Cruz 2019).

**Findings (Phase 0):**

- **H1 (sufficient power):** YES. At the H1-qualifying subset (A = 0.056, B ≥ 0.75, C ≥ 0.70, D ≥ 0.40, E = 18), fraction of cells achieving 80% power = 1.0.
- **H2 (HIE as Scorecard metric):** CONFIRMED NO. Fraction of cells where HIE is detectable at E = 18 and A = 0.0017 (the lower-bound literature rate) = 0.0. Recommendation: HIE must be a sentinel-event tracker, not a Scorecard pillar metric.
- **H4 (Bayes violation sanity):** PASS. No cell violated the PPV ≤ Bayes bound.
- **Key MDE curve:** at 8,500 deliveries / 12 post-months, MDE = 0.99 pp / 17.6% relative; at 18 months, 0.86 pp / 15.3%; at 24 months, 0.79 pp / 14.1%.

**Equity arm (Phase 0):** Group B (Medicaid + minority, 25% of volume) assigned baseline × 1.5, sensitivity − 0.10, action rate − 0.15. ΔD = RRR_A − RRR_B computed per cell. Verdict at H1-qualifying cells: determined analytically without a Monte Carlo run.

### What Validation Was Done

Every item on the seven-step checklist was completed and disclosed in `results.md`:

1. Bayes constraint check — PASS, H4 explicit
2. Monotonicity — verified (lower threshold → more flags in the sweep dimensions)
3. Boundary — tested at efficacy = 0 (rate equals counterfactual) and B × C × D = 0 (same)
4. CLT sanity — event rates at A = 0.056 lie within 4-sigma CLT bounds for the 36-month pre-period
5. Sensitivity sweep — 243 cells IS the sensitivity sweep
6. Gap list — written explicitly in the Phase-0 results doc (see below)
7. Counterfactual baseline — computed explicitly, not dropped from the algebra

### Validation Gaps Disclosed

The Phase-0 results doc explicitly disclosed what it could not answer:

- **Per-seed variance.** Analytical power is the expected power. Real-world evaluations on a single 18-month window will have realized variance around this number.
- **Time-to-action coupling.** The analytical chain treats action_rate and efficacy as independent constants. In reality, earlier alerts produce more lead time, which may produce higher efficacy — a score-dependent, time-dependent coupling the closed form cannot represent.
- **Alert fatigue half-life.** Action rate was treated as a single point estimate per cell. The Kumar half-life model in the `sepsis_early_alert` reference scenario shows that action_rate can decay over hours/days and that the decay constant itself is a decision-relevant parameter.
- **Score-distribution-dependent effects.** PeriGen's operating point is fixed by the vendor, so threshold selection is off the table for this study. But if a future question asks "should we move the threshold?", the closed-form pass cannot answer — the SDK's `ControlledMLModel` would be required.
- **Analytic method rehearsal.** The Phase-0 results project ITS power on a simplified two-proportion basis. A genuine ITS rehearsal — segmented regression on monthly event counts with autocorrelation handled via a real time-series model — would require entity-level synthetic data.

### Lessons for Future Experiments

1. **The analytical pass answered the primary question.** H1, H2, H4 and the MDE curve are what the AI Committee needs to approve the evaluation design. The SDK was not required.
2. **The gap list drives Phase 1, not enthusiasm.** If a future question about PeriGen depends on time-to-action coupling or alert fatigue, escalate to SDK with the `sepsis_early_alert` reference scenario as a template. Until that question arrives, building a scenario is premature.
3. **The equity arm was feasible analytically because the question was point-estimate deltas.** If the equity question becomes "how does subgroup calibration shift the operating point?", it would require the score distribution and the SDK.
4. **Document the gaps explicitly, in the same doc as the findings.** Do not let a clean analytical result lull the reader into thinking it answered more than it did.

---

## Reference Files

- `.claude/invariants.yaml` → `analytical_first_pass` block (closed_form_sufficient_for, sdk_required_for, validation_required_either_way)
- `.claude/agents/sim-guide.md` → Phase 2 (full protocol the agent follows in a live session)
- `CLAUDE.md` → Agent Guidance → Analytical-First Principle (one-paragraph summary for humans)
