# Paper 14: Edelson/Churpek et al. — Head-to-head EWS comparison (JAMA Network Open 2024)

## Classification: PARTIAL_FIT
## Reproducibility: UNDERDETERMINED

---

## Key Findings

This head-to-head comparison of six early warning scores across 362,926 encounters at seven Yale New Haven Health hospitals found wide variation in discrimination: eCARTv5 AUROC 0.895, NEWS 0.831, Epic Deterioration Index 0.808. The study is a retrospective external validation — it measures model discrimination on historical data, not the impact of deploying these models on patient outcomes. The SDK can model the population and the prediction component (comparing three `ControlledMLModel` instances at different AUCs), but there is no intervention arm, no outcome change, and no counterfactual. Reproducibility is UNDERDETERMINED because the paper makes no causal claims about patient outcomes — there is nothing to reproduce in a simulation sense, only performance characteristics to instantiate.

**SDK value:** Provides a rigorous multi-model AUC benchmark for three widely-deployed EWS systems across a large, real-world population. Directly calibrates the `predict()` component for any deterioration warning scenario. The AUROC gap between models (0.895 vs. 0.808 = 0.087) quantifies the model selection decision as a simulatable design variable.

---

## Parameter Extraction

| Parameter | Paper Reports | SDK Needs | Status |
|-----------|--------------|-----------|--------|
| Setting | 7 hospitals, Yale New Haven Health | Inpatient (ward + ICU) | AVAILABLE |
| Population N | 362,926 encounters | Cohort size | AVAILABLE |
| Encounter types | All adult inpatient encounters | Population entity definition | AVAILABLE |
| Primary outcome | Composite: ICU transfer or in-hospital mortality | Binary outcome per encounter | AVAILABLE |
| Outcome base rate | ~4–6% (estimated from paper; not directly stated as overall rate) | Base rate | PARTIALLY AVAILABLE |
| eCARTv5 AUROC | 0.895 (95% CI reported) | Target AUC for predict() | AVAILABLE |
| NEWS AUROC | 0.831 | Target AUC for predict() | AVAILABLE |
| Epic DI AUROC | 0.808 | Target AUC for predict() | AVAILABLE |
| APACHE IVa AUROC | 0.894 | Target AUC for predict() | AVAILABLE |
| NEWS2 AUROC | 0.851 | Target AUC for predict() | AVAILABLE |
| mNEWS AUROC | 0.858 | Target AUC for predict() | AVAILABLE |
| Sensitivity at deployed threshold | Not specified (no deployed threshold described) | Sensitivity for alert rate | ABSENT |
| Specificity at deployed threshold | Not specified | False positive rate | ABSENT |
| Alert rate | Not reported | Operational alert volume | ABSENT |
| Calibration (Brier, E:O) | Not reported | Calibration for net benefit | ABSENT |
| Intervention triggered by score | Not described (validation only) | State change action | ABSENT |
| Clinical workflow integration | Not described | Alert delivery pathway | ABSENT |
| Response compliance rate | Not described | Clinician compliance for intervene() | ABSENT |
| Race/ethnicity breakdown | Not reported in this description | Equity analysis | ABSENT |
| Hospital-level heterogeneity | 7 hospitals included | Site variance for multi-site model | PARTIALLY AVAILABLE |
| FDA clearance | eCARTv5 cleared May 2024 (K233253) | Regulatory context | AVAILABLE (informational only) |

**Critical absent parameters:**
- No operational threshold specified for any model. AUROC is a rank statistic; at any given threshold, sensitivity/specificity/alert-rate differ dramatically.
- No calibration data. A model with AUROC 0.895 but E:O ratio 3.0 at deployed threshold would generate excess alerts without improving net benefit.
- No outcome comparison to "no EWS" baseline. The paper does not claim eCARTv5 improves outcomes vs. NEWS — only that it discriminates better.

---

## Simulation Results

**Assessment only — no code run (no intervention arm exists to simulate).**

What the SDK CAN model from this paper:
- Population: 362,926 encounter entities, inpatient ward/ICU, composite deterioration outcome
- Three `ControlledMLModel` instances: AUC 0.895, 0.831, 0.808
- Threshold sweep showing sensitivity-PPV-alert-rate trade-offs for each model

What the SDK CANNOT model from this paper:
- Any intervention (no treatment arm described)
- Any counterfactual outcome comparison
- Calibration characteristics (not reported)

**Simulatable design question (not reproducibility):** Given models at AUC 0.895, 0.831, and 0.808, what is the net benefit difference (Vickers DCA framework, Paper #17) across threshold probabilities? This is a forward-looking simulation question the paper doesn't answer, but the SDK can address using this paper's parameters.

---

## Verification Summary

No simulation run. SDK fitness criteria:
- [x] Population-level intervention: Yes (inpatient deterioration detection)
- [ ] Predictive model drives intervention: Model exists; intervention not described
- [ ] Intervention is a state change: Not described
- [x] Measurable outcome at each timestep: ICU transfer/mortality is measurable
- [ ] Counterfactual causal question: Not posed (retrospective validation only)
- [x] Discrete-time dynamics: Hourly score updates are discrete-time compatible

**3 of 6 criteria met → PARTIAL_FIT as benchmark source; NO_FIT for outcome reproducibility.**

The PARTIAL_FIT classification reflects that this paper could become FIT if combined with a deployment intervention assumption (e.g., "eCARTv5 triggers RRT call at threshold X, clinician responds with probability Y") — but those parameters are not in this paper.

---

## Discrepancies

No simulation to compare against. Contextual discrepancy worth flagging:

**AUROC vs. clinical utility:** eCARTv5 at AUROC 0.895 is substantially better than Epic DI at 0.808 — but this difference in discrimination does not directly translate to a proportional difference in clinical outcomes. At any given sensitivity target, the higher-AUC model will have lower FPR and thus lower alert burden, which affects alert fatigue dynamics. The magnitude of this clinical difference depends on threshold selection, compliance curves, and intervention effectiveness — none of which this paper reports.

**The FDA clearance finding is noteworthy but does not imply deployment superiority.** FDA clearance (K233253) reflects demonstrated safety and effectiveness claims, not necessarily superiority over uncredited alternatives. The SDK treats regulatory status as a metadata field, not a performance parameter.

---

## Scientific Reporting Gaps

1. **No calibration statistics.** Seven AUROCs are reported with confidence intervals, but no Brier scores, E:O ratios, or calibration curves. Per Van Calster et al. (Paper #18), this is the norm (only 36% of published models report calibration) and is explicitly harmful: it prevents net benefit calculation and clinical threshold selection.

2. **No operational threshold.** The paper compares discrimination, not operational performance. Without a threshold, sensitivity/PPV/alert-rate are undefined. This is the most actionable missing parameter for any hospital considering deployment.

3. **No equity analysis.** With 362,926 encounters across seven hospitals, subgroup analysis by race, age, sex, or insurance type was feasible and absent. Whether NEWS vs. eCARTv5 differences in AUROC are uniform or concentrated in specific demographic groups is unknown.

4. **No outcome comparison.** Higher AUROC does not guarantee better outcomes. A hospital currently using NEWS needs evidence that switching to eCARTv5 would change patient trajectories, not just improve rank statistics. This paper provides necessary but not sufficient evidence for that decision.

5. **Hospital-level heterogeneity not reported.** Seven hospitals are pooled. Whether eCARTv5's AUROC advantage holds consistently across all seven, or is driven by one or two sites, is unknown. This matters for generalizing to the target health system's context.

6. **Time-to-event not reported.** The paper doesn't report how far in advance of deterioration events the scores differ across models. Lead time (as in Paper #13) is clinically important: a model with AUROC 0.895 that fires 1 hour before deterioration may be less valuable than one with AUROC 0.831 that fires 12 hours before.

---

## Assumptions Made

| Assumption | Impact | Basis |
|------------|--------|-------|
| Base outcome rate ~4–6% | HIGH | Inferred from inpatient deterioration literature; not directly stated |
| AUROCs are reproducible at the target health system | MEDIUM | External validation at 7 hospitals provides evidence; the health system case mix may differ |
| Discrimination advantage translates proportionally to net benefit | MEDIUM | Requires calibration data to confirm; not validated in this paper |
| eCARTv5 vs. NEWS advantage uniform across demographics | HIGH | Not analyzed; assumption is that no differential bias exists — unverified |

---

## SDK Design Contribution

This paper establishes the **model selection benchmark** for EWS scenarios. The three-model AUC comparison (0.895 vs. 0.831 vs. 0.808) should be implemented as a parameterized comparison scenario in the SDK:

```
# Suggested scenario: ews_model_comparison
# Holds intervention parameters constant, varies predict() AUC across [0.895, 0.858, 0.851, 0.831, 0.808]
# Outputs: sensitivity/PPV/alert-rate/net-benefit curves for each
# Shows: how much of the AUROC difference translates to clinical utility at each threshold
```

**Priority for SDK benchmark library:** HIGH — provides the largest published EWS validation dataset and the widest model comparison. Directly relevant to an Epic EHR context (Epic DI is already deployed; eCARTv5 is a procurement-relevant alternative).

**Procurement relevance for the target health system:** Epic DI (AUROC 0.808) vs. eCARTv5 (AUROC 0.895) is a live procurement decision at most Epic deployments. The SDK can simulate what that 0.087 AUROC gap means in the target health system's operational context — across the local case mix, at local clinician response rates — before any contract is signed.
