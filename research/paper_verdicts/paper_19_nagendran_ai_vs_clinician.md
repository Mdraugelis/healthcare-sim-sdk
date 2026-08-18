# Paper 19: Nagendran et al. — AI vs. clinicians systematic review (BMJ 2020)

## Classification: NO_FIT
## Reproducibility: N/A

---

## Key Findings

Of 81 non-randomized studies comparing deep learning to clinicians: only 9 were prospective, only 6 tested in real-world settings, risk of bias was high in 58 of 81, and 95% provided no dataset access. Yet 61 of 81 claimed AI performance was "at least comparable to clinicians." The paper exposes a systematic overstatement pattern: the evidence base for AI-vs-clinician superiority is built almost entirely on retrospective, single-site studies with high bias risk, testing on held-out portions of the same dataset the model was trained on. The SDK cannot simulate a systematic review. However, Nagendran et al. provide the most rigorous quantitative characterization of the gap between what the clinical AI field claims and what it demonstrates — and this gap is precisely what the SDK is designed to close.

---

## Parameter Extraction

| Parameter | Paper Reports | SDK Needs | Status |
|-----------|--------------|-----------|--------|
| Study population | 81 AI-vs-clinician studies (aggregate) | Patient entities | ABSENT |
| Base outcome rate | Varies across 81 studies | Single population base rate | ABSENT |
| Model performance | Varies; 61/81 claimed AI ≥ clinician | Single AUC | ABSENT |
| Intervention type | Varies; none is a deployment scenario | State-change action | ABSENT |
| Effect size | None (clinical outcomes not measured in most) | Absolute risk reduction | ABSENT |
| Study design | 72/81 retrospective, 9 prospective, 6 real-world | RCT or quasi-experimental | ABSENT |
| Prospective design rate | 9/81 = 11.1% | Population representation | AVAILABLE (meta-statistic) |
| High bias risk rate | 58/81 = 71.6% | Quality indicator | AVAILABLE (meta-statistic) |
| Dataset availability | 5% provided dataset access | Reproducibility indicator | AVAILABLE (meta-statistic) |
| Median comparator clinicians | 4 experts | Comparator arm strength | AVAILABLE (meta-statistic) |
| TRIPOD adherence | <50% for 12/29 items | Reporting quality | AVAILABLE (meta-statistic) |

No simulatable parameters. All statistics describe the *distribution of other papers' methodological choices*, not a deployable healthcare intervention.

---

## Simulation Results

Not attempted. Classification: NO_FIT.

This paper is a quality assessment of the evidence base — a second-order analysis. Simulating it would require choosing one of the 81 constituent studies and simulating that; Nagendran et al. provide no primary data.

**The paper's core statistical finding is analytically reproducible, not simulatable:**
- P(AI ≥ clinician | study design is retrospective) > P(AI ≥ clinician | study design is prospective): This is a selection/reporting bias observation, not a causal mechanism that can be parameterized in the SDK.
- The bias is structural: case-control enrichment, same-source test sets, and expert comparator selection (4 median comparators, not representative clinicians) inflate AI performance estimates. No simulation of these biases is needed — the mechanism is described analytically.

---

## Verification Summary

No simulation run. SDK fitness criteria:
- [ ] Population-level intervention: Not applicable (systematic review)
- [ ] Predictive model drives intervention: Not applicable
- [ ] Intervention is a state change: Not applicable
- [ ] Measurable outcome at each timestep: Not applicable
- [ ] Counterfactual causal question: Not applicable
- [ ] Discrete-time dynamics: Not applicable

**0 of 6 criteria met → NO_FIT.**

---

## Discrepancies

None applicable. The paper's methodology (systematic search through December 2019, TRIPOD quality assessment, PROBAST bias evaluation) is rigorous and the findings are consistent with the broader literature. The 61/81 AI superiority claim rate is consistent with Plana et al.'s (Paper #12) finding that 70–81% of RCTs report positive primary endpoints — both reflect publication bias in the clinical AI literature.

**One notable nuance:** Nagendran et al. focus on deep learning studies specifically, not the broader ML literature. Classical ML methods (logistic regression, gradient boosting) may have different bias profiles — they are less frequently positioned as "AI vs. clinician" and more frequently evaluated for incremental improvement over existing tools. This distinction matters for SDK scenario design: the spectacular AUC claims in this literature (AUC 0.99 for diabetic retinopathy detection) are primarily deep learning claims; the more moderate AUC values in Papers #1–11 of this pipeline reflect more operationally realistic applications.

---

## Scientific Reporting Gaps

This paper is the meta-analysis of the field's reporting gaps. Key enumeration:

1. **95% of studies provide no dataset access.** Reproducibility requires data. Without data, a claim of AUC X vs. clinician Y cannot be independently verified. The SDK operationalizes reproducibility computationally — when the primary study data is unavailable, simulation with paper-reported parameters is the best available alternative.

2. **Median 4 expert comparators.** Using 4 carefully selected experts as "the clinicians" creates positive selection bias: expert performance may exceed average performance; 4 is insufficient to characterize variance; experts may be primed by context (they know it's an AI study). A robust comparison requires representative clinicians under typical conditions — which is exactly what prospective real-world deployment captures.

3. **6/81 real-world studies.** Only 7.4% of claims are based on real-world deployment evidence. The other 92.6% are comparisons on curated datasets, under conditions that do not replicate clinical workflow. This 7.4% figure is the clearest quantitative indictment of the field's evidence standards.

4. **TRIPOD adherence <50% for 12/29 items.** Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis (TRIPOD) standards are not consistently followed, making independent assessment of methods impossible for nearly half the evaluated items.

5. **Outcome reporting is missing from 81 studies.** None of the 81 AI-vs-clinician comparison studies reports patient outcomes. They all measure model performance on images, signals, or text — not what happens to patients whose diagnoses are made by AI vs. clinician. This is the fundamental methodological gap the SDK addresses: comparing model performance metrics ≠ comparing patient outcomes.

---

## Assumptions Made

None — no simulation attempted.

---

## SDK Design Contribution

**Nagendran et al. provide the empirical basis for the SDK's study design quality filter.** When processing a paper through the pipeline, the following flags should be raised automatically based on study characteristics:

### Study Quality Checklist (informed by Nagendran et al.)

| Factor | Risk Flag | Threshold |
|--------|-----------|-----------|
| Study design | HIGH if retrospective | Design ∉ {RCT, prospective cohort} |
| Real-world deployment | HIGH if not deployed | Deployment status = "benchmark only" |
| Comparator size | MEDIUM if small | n_comparators < 10 |
| Dataset availability | LOW (reproducibility) | Dataset not publicly available |
| TRIPOD adherence | MEDIUM | <50% adherence on key items |
| Outcome reporting | HIGH | No patient outcome measured |
| Bias risk (PROBAST) | HIGH | High risk in ≥1 of 4 domains |

For Papers processed in this pipeline:
- Paper #1 (Wong): Retrospective external validation → HIGH bias risk flag → but reports real-world operational parameters (AUC 0.63, alert rate 18%), partially offsetting
- Paper #13 (Henry/TREWScore): Retrospective, no deployment → HIGH bias risk, UNDERDETERMINED reproducibility
- Paper #14 (Edelson): Retrospective external validation, no deployment arm → HIGH bias risk
- Papers #2–8: Prospective designs with outcome measurement → lower bias risk; more reproducible

**This taxonomy explains why the pipeline's reproducibility classifications vary:** Papers from the high-bias-risk category (retrospective, no deployment) are consistently classified UNDERDETERMINED or N/A; papers from the prospective/deployed category (Papers #2–8) are classified FIT or PARTIAL_FIT.

**The SDK's quality filter should:**
1. Require prospective design evidence before classifying a paper as FIT for simulation
2. Flag any paper with no patient outcome measurement as PARTIAL_FIT at best
3. Require minimum reporting checklist (TRIPOD + CONSORT-AI elements) before proceeding past Phase 2

**Priority for SDK documentation:** HIGH — the Nagendran quality taxonomy should be embedded in Phase 2 (Fitness) assessment as a formal checklist.
