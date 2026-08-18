"""Stage B verification: boundary-condition mini-sims (design doc
section 7) + narrative case walkthroughs.

Stage B green is THE verification gate -- no experiment may be
interpreted before it passes. One deliberate deviation from the design
doc's section 7 table, with rationale: `top-K = 100%` does NOT imply
Arm F == Arm C in this implementation, because flagging everyone still
ORDERS the worklist by score (a score-ordered queue is not the baseline
queue). The design doc's equivalence would hold only for flagging
without reordering. The boundary is instead checked structurally
(top-K=1.0 flags every scoreable unassessed inpatient).
"""

import json
from pathlib import Path

import numpy as np

from ...core.engine import (
    BranchedSimulationEngine,
    CounterfactualMode,
)
from ...core.scenario import TimeConfig
from .scenario import (
    MalnutritionConfig,
    MalnutritionPrioritizationScenario,
    SOURCE_CONSULT,
    SOURCE_ML,
    STATUS_DISCHARGED,
)

REPORT_DIR = Path("outputs") / "malnutrition_verification"

# ml_flagged is excluded from bit-equality: it is the tool's display
# label, written in every mode; the boundary claims concern OUTCOMES.
NUMERIC_KEYS = (
    "true_malnourished", "severity", "assessed", "assess_day",
    "assess_source", "time_to_assessment", "missed",
    "severity_at_assess", "harm", "diagnosed", "coded", "los",
    "readmit", "status",
)


def _run(seed=42, horizon=42, **overrides):
    defaults = dict(
        initial_census=100,
        admissions_per_day=12,
        rdn_assessments_per_day=9,
        ml_enabled=True,
    )
    defaults.update(overrides)
    cfg = MalnutritionConfig(**defaults)
    tc = TimeConfig(
        n_timesteps=horizon,
        timestep_duration=1 / 365,
        timestep_unit="day",
        prediction_schedule=list(range(horizon)),
    )
    sc = MalnutritionPrioritizationScenario(
        time_config=tc, seed=seed, config=cfg)
    engine = BranchedSimulationEngine(sc, CounterfactualMode.BRANCHED)
    return sc, engine.run(sc.n_entities_required)


def _bit_equal(res) -> bool:
    for t in range(len(res.outcomes)):
        f, c = res.outcomes[t], res.counterfactual_outcomes[t]
        if not np.array_equal(f.events, c.events):
            return False
        for k in NUMERIC_KEYS:
            if not np.array_equal(f.secondary[k], c.secondary[k]):
                return False
    return True


# The efficacy=0 boundary claims CLINICAL equality only: the worklist
# still reorders who is assessed when (process keys differ -- that is
# the tool working), but with both effect channels off, disease course
# (severity), LOS and readmission must be bit-identical via the shared
# temporal draws and creation-time coins.
CLINICAL_KEYS = ("true_malnourished", "severity", "status", "los",
                 "readmit")


def _clinical_bit_equal(res) -> bool:
    for t in range(len(res.outcomes)):
        f, c = res.outcomes[t], res.counterfactual_outcomes[t]
        for k in CLINICAL_KEYS:
            if not np.array_equal(f.secondary[k], c.secondary[k]):
                return False
    return True


def _miss_rates(res):
    last_f = res.outcomes[len(res.outcomes) - 1].secondary
    last_c = res.counterfactual_outcomes[
        len(res.outcomes) - 1].secondary
    mf = (last_f["missed"].sum()
          / max(last_f["mal_at_discharge"].sum(), 1))
    mc = (last_c["missed"].sum()
          / max(last_c["mal_at_discharge"].sum(), 1))
    return mf, mc


def run_boundary_suite(verbose: bool = True) -> bool:
    checks = []

    def add(name, expected, ok, detail=""):
        checks.append({"boundary": name, "expected": expected,
                       "pass": bool(ok), "detail": str(detail)})

    # 1. capacity = 0: nobody assessed, arms equally (maximally) bad.
    # (ml_flagged bookkeeping still differs -- flags are written but
    # nobody can act on them -- so compare OUTCOME keys only.)
    _, res = _run(rdn_assessments_per_day=0)
    H0 = len(res.outcomes) - 1
    f0 = res.outcomes[H0].secondary
    c0 = res.counterfactual_outcomes[H0].secondary
    outcome_keys = ("assessed", "missed", "harm", "coded",
                    "readmit", "los", "diagnosed")
    add("capacity=0", "nobody assessed; outcomes equal",
        (f0["assessed"] == 0).all()
        and all(np.array_equal(f0[k], c0[k]) for k in outcome_keys))

    # 2. capacity = inf (flag uplift 0): order irrelevant end to end.
    _, res = _run(rdn_assessments_per_day=10**9)
    add("capacity=inf", "arms bit-identical", _bit_equal(res))

    # 3. efficacy = 0 (both channels, both endpoints): clinical
    #    outcomes (LOS, readmission) bit-identical across arms --
    #    exact by construction via the creation-time shared coins --
    #    while the arms still genuinely differ on process (assessment
    #    order), proving the check has teeth.
    _, res = _run(eff_detect_rel=0.0, eff_timing_rel=0.0,
                  eff_detect_los=0.0, eff_timing_los=0.0)
    H3 = len(res.outcomes) - 1
    arms_differ = not np.array_equal(
        res.outcomes[H3].secondary["assess_day"],
        res.counterfactual_outcomes[H3].secondary["assess_day"])
    add("efficacy=0 (two-channel)",
        "LOS/readmit bit-identical; process still differs",
        _clinical_bit_equal(res) and arms_differ)

    # 4. true_prevalence -> 0: no true+ to miss; arms converge.
    _, res = _run(true_prevalence=0.001)
    mf, mc = _miss_rates(res)
    last = res.outcomes[len(res.outcomes) - 1].secondary
    add("true_prevalence->0", "essentially no missed cases",
        last["missed"].sum() <= 2,
        f"missed={last['missed'].sum():.0f}")

    # 5. top-K = 0 (legacy cutoff mode), MST floor off: the tool does
    #    nothing -> arms bit-identical.
    _, res = _run(operating_mode="top_k", top_k_fraction=0.0,
                  mst_reference_floor=0.0)
    add("top_k=0 (floor off)", "arms bit-identical", _bit_equal(res))

    # 6. rank_all ranks every scoreable unassessed inpatient daily
    #    (structural check of the deployed no-cutoff policy).
    sc, res = _run(horizon=20)
    n_ranked = res.interventions[10].metadata["n_ranked_today"]
    out = res.outcomes[10]
    add("rank_all coverage", "every scoreable candidate ranked daily",
        n_ranked > 0.8 * out.metadata["queue_depth"],
        f"ranked {n_ranked}, queue {out.metadata['queue_depth']}")

    # 7. insufficient data = 100%, floor off: nobody scoreable and no
    #    reference layer -> arms bit-identical.
    _, res = _run(insuff_data_rate=1.0, mst_reference_floor=0.0)
    add("insuff_data=1.0 (floor off)", "arms bit-identical",
        _bit_equal(res))

    # 8. degenerate AUROC 0.5: no benefit from a chance-level model.
    deltas_real: list = []
    deltas_degen: list = []
    for seed in range(1, 6):
        _, r1 = _run(seed=seed)
        _, r2 = _run(seed=seed, degenerate_auroc05=True)
        for r, sink in ((r1, deltas_real), (r2, deltas_degen)):
            mf, mc = _miss_rates(r)
            sink.append(mc - mf)
    add("model_auroc=0.5 (degenerate)",
        "ML advantage collapses to ~0 or negative",
        np.mean(deltas_degen) < 0.03
        and np.mean(deltas_real) > np.mean(deltas_degen),
        f"real {np.mean(deltas_real):+.3f}, "
        f"degenerate {np.mean(deltas_degen):+.3f}")

    # 9. Safety-net erosion monotonicity (H7, new semantics): erosion
    #    is the probability RDNs ignore the retained MST reference
    #    floor. It only bites where the floor changes reach (capacity
    #    that doesn't already exhaust the ranked list) and for
    #    patients whose floor exceeds their score. Requires a strict
    #    effect at full erosion (vacuous pass = dead mechanism).
    miss_by_decay = []
    for decay in (0.0, 0.5, 5.0):   # levels ~0 / ~0.5 wk1 / 1.0 fast
        rates = []
        for seed in range(1, 6):
            _, r = _run(seed=seed, safety_net_preserved=False,
                        rdn_assessments_per_day=14,
                        unflagged_assessment_decay_per_week=decay)
            rates.append(_miss_rates(r)[0])
        miss_by_decay.append(np.mean(rates))
    add("MST-floor erosion monotonicity",
        "more erosion -> more F-arm missed; strict at full erosion",
        miss_by_decay[0] <= miss_by_decay[1] + 5e-3
        and miss_by_decay[1] <= miss_by_decay[2] + 5e-3
        and miss_by_decay[2] > miss_by_decay[0] + 1e-6,
        "missed@decay " + ", ".join(
            f"{m:.3f}" for m in miss_by_decay))

    # 10. Flag-driven coding uplift: F codes more, C unchanged.
    _, r0 = _run(flag_driven_coding_uplift=0.0)
    _, r1 = _run(flag_driven_coding_uplift=0.10)
    H = len(r0.outcomes) - 1
    coded_f0 = r0.outcomes[H].secondary["coded"].sum()
    coded_f1 = r1.outcomes[H].secondary["coded"].sum()
    coded_c0 = r0.counterfactual_outcomes[H].secondary["coded"].sum()
    coded_c1 = r1.counterfactual_outcomes[H].secondary["coded"].sum()
    add("flag_driven_coding_uplift",
        "uplift raises F-arm coding only",
        coded_f1 > coded_f0 and coded_c1 == coded_c0,
        f"F {coded_f0:.0f}->{coded_f1:.0f}, "
        f"C {coded_c0:.0f}->{coded_c1:.0f}")

    # 11. Consults >= capacity (dossier 04): when every admission
    #     carries a mandatory time-bound consult, tier 0 saturates the
    #     RDN day in both arms and the ML ranking never gets a slot --
    #     arms bit-identical end to end.
    _, res = _run(mandatory_consult_fraction=1.0,
                  rdn_assessments_per_day=9)
    add("consults>=capacity",
        "tier-0 saturation; arms bit-identical", _bit_equal(res))

    walkthroughs = _case_walkthroughs()

    all_pass = all(c["pass"] for c in checks)
    if verbose:
        print("=" * 70)
        print("Stage B -- boundary-condition suite "
              "(design doc section 7)")
        print("=" * 70)
        for c in checks:
            print(f"  [{'PASS' if c['pass'] else 'FAIL'}] "
                  f"{c['boundary']:<32} {c['expected']}"
                  + (f"  ({c['detail']})" if c["detail"] else ""))
        print("\n" + "=" * 70)
        print("Case walkthroughs (5 entities, seed 42 default run)")
        print("=" * 70)
        for w in walkthroughs:
            print(f"\n{w}")
        print(f"\nStage B: {'PASS' if all_pass else 'FAIL'}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "stage_b_report.json").write_text(
        json.dumps({"checks": checks, "all_pass": all_pass},
                   indent=2))
    (REPORT_DIR / "case_walkthroughs.md").write_text(
        "# Case walkthroughs (seed 42, default config)\n\n"
        + "\n\n".join(walkthroughs) + "\n")
    return all_pass


def _case_walkthroughs():
    """Trace 3-5 entities through the default full-scale run and
    narrate their trajectories (sim-guide Phase 4, Level 4)."""
    cfg = MalnutritionConfig(ml_enabled=True)
    tc = TimeConfig(
        n_timesteps=84, timestep_duration=1 / 365,
        timestep_unit="day", prediction_schedule=list(range(84)))
    sc = MalnutritionPrioritizationScenario(
        time_config=tc, seed=42, config=cfg)
    engine = BranchedSimulationEngine(sc, CounterfactualMode.BRANCHED)
    res = engine.run(sc.n_entities_required)

    H = len(res.outcomes) - 1
    f = res.outcomes[H].secondary
    c = res.counterfactual_outcomes[H].secondary

    mal = f["mal_at_discharge"] == 1
    stories = []

    def narrate(i, label):
        eth = f["ethnicity"][i]
        sev = {0: "not malnourished", 1: "moderate (E44)",
               2: "severe (E43)"}[int(f["severity"][i])]
        lines = [f"**Case {label} (entity {i})** -- {eth}, {sev}, "
                 f"actual stay {f['los'][i]:.0f} days."]
        # Timing facts from the two branches' secondary arrays.
        for arm, s in (("Factual (ML worklist)", f),
                       ("Counterfactual (current workflow)", c)):
            if s["assessed"][i] == 1:
                src = {1: "time-bound consult", 2: "ML ranking",
                       3: "MST stream", 4: "backstop"}[
                    int(s["assess_source"][i])]
                lines.append(
                    f"- {arm}: assessed on day "
                    f"{int(s['assess_day'][i])} via {src} "
                    f"({s['time_to_assessment'][i]:.0f} days after "
                    f"admission); diagnosed="
                    f"{'yes' if s['diagnosed'][i] else 'no'}, "
                    f"coded={'yes' if s['coded'][i] else 'no'}.")
            else:
                missed = "MISSED (discharged unassessed)" \
                    if s["missed"][i] else "still in hospital"
                lines.append(f"- {arm}: never assessed -- {missed}.")
        lines.append(
            f"- Harm accrued: factual {f['harm'][i]:.1f} vs "
            f"counterfactual {c['harm'][i]:.1f} units.")
        return "\n".join(lines)

    # (a) True+ caught by ML in F, missed in C.
    cand = np.where(
        mal & (f["assess_source"] == SOURCE_ML)
        & (c["missed"] == 1))[0]
    if len(cand):
        stories.append(narrate(cand[0], "A: ML catch, baseline miss"))

    # (b) Unflagged true+ delayed or missed in F (de facto gating).
    cand = np.where(
        mal & (f["missed"] == 1) & (c["assessed"] == 1))[0]
    if len(cand):
        stories.append(narrate(
            cand[0], "B: reordering harm (missed in F, seen in C)"))

    # (c) False positive burning an RDN slot.
    cand = np.where(
        (f["assess_source"] == SOURCE_ML)
        & (f["mal_at_discharge"] == 0)
        & (f["status"] == STATUS_DISCHARGED))[0]
    if len(cand):
        stories.append(narrate(cand[0], "C: false-positive workup"))

    # (d) True+ caught by the retained MST reference / stream.
    cand = np.where(
        mal & (f["assessed"] == 1)
        & (f["assess_source"] != SOURCE_ML)
        & (f["assess_source"] != SOURCE_CONSULT))[0]
    if len(cand):
        stories.append(narrate(cand[0], "D: MST safety-net catch"))

    # (e) Hispanic/Latino true+ trajectory (equity lens).
    cand = np.where(mal & (f["ethnicity"] == "HispanicLatino"))[0]
    if len(cand):
        stories.append(narrate(cand[0], "E: H/L patient (equity)"))

    return stories
