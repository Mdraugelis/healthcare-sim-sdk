# Paper 21: Rajkomar et al. — Ensuring Fairness in Machine Learning to Advance Health Equity (2018)

## Classification: NO_FIT
## Reproducibility: N/A

## Key Findings

Rajkomar et al. define three competing fairness criteria for ML in healthcare — equal outcomes (demographic parity), equal performance (equalized odds/calibration), and equal allocation (equal treatment of equals) — and prove these are mathematically incompatible except in degenerate cases. The paper provides a practical local fairness checklist for health system deployment and argues that fairness definition selection is an ethical and institutional choice, not a technical one. The central contribution is not a deployable model or measurable intervention but a conceptual and normative framework that must be instantiated by system designers.

The mathematical incompatibility result (you cannot simultaneously achieve demographic parity, equalized odds, and individual fairness unless base rates are equal across groups) is the key finding with direct SDK design implications. It means any SDK equity audit must specify *which* fairness definition is being tested and why — and must flag tradeoffs explicitly.

## Parameter Extraction

| Parameter | Value | Source | Status |
|---|---|---|---|
| Fairness Definition 1 | Equal outcomes (demographic parity) | Paper | Extracted |
| Fairness Definition 2 | Equal performance (equalized odds, calibration) | Paper | Extracted |
| Fairness Definition 3 | Equal allocation (treat like cases alike) | Paper | Extracted |
| Mathematical relationship | Mutual incompatibility (except when base rates equal) | Paper | Extracted |
| Setting | General ML in healthcare (no specific deployment) | Paper | Extracted |
| Population N | Not applicable — framework paper | N/A | — |
| Model AUC | Not applicable | N/A | — |
| Intervention | Not applicable | N/A | — |
| Effect size | Not applicable | N/A | — |

**Missing parameters for SDK:** All operational parameters absent; this is a normative framework, not an empirical study.

## Simulation Results

No simulation conducted. Paper is a conceptual/normative framework with no empirical deployment.

**SDK Fitness Criteria Assessment:**

| Criterion | Met? | Notes |
|---|---|---|
| Population-level intervention | ✗ FAIL | No intervention defined |
| Predictive model drives intervention | ✗ FAIL | No model described, no deployment |
| Intervention is a state change | ✗ FAIL | No state change defined |
| Measurable outcome at each timestep | ✗ FAIL | No outcome defined |
| Counterfactual causal question | ✗ FAIL | No counterfactual; purely normative |
| Discrete-time dynamics | ✗ FAIL | No temporal structure |

**0 of 6 criteria met → NO_FIT**

## Verification Summary

Not applicable. No simulation was designed or executed.

## Discrepancies

Not applicable.

## Scientific Reporting Gaps

This is a framework paper — the relevant "gaps" are design choices the paper intentionally leaves to practitioners:
1. **No operationalization:** The paper defines three fairness criteria but does not specify how to measure them in a live system.
2. **No threshold guidance:** No guidance on which fairness definition to prioritize in which clinical contexts.
3. **No intersectionality treatment:** The framework addresses single demographic axes (race, sex) but does not address intersectional fairness (e.g., Black women vs. White women vs. Black men).
4. **No sensitivity analysis:** No discussion of how sensitive fairness metric tradeoffs are to base-rate estimation error.

## Assumptions Made

None — no simulation was run.

---

## SDK Design Rationale (Primary Contribution of This Paper)

This paper is **high-priority SDK design input** despite being NO_FIT for direct simulation.

**Concrete SDK requirements derived from this paper:**

1. **Triple fairness reporting:** Every SDK simulation that includes demographic subgroups must report all three fairness definitions simultaneously — demographic parity gap, equalized odds (TPR/FPR by group), and calibration by group. The incompatibility result means the SDK must display these as a tradeoff table, not a single pass/fail.

2. **Fairness definition selection prompt:** Before any equity audit, the SDK should prompt the user to declare which fairness definition is operationally primary and justify the choice. This prevents post-hoc selection of the metric that makes results look best.

3. **Base-rate stratification:** When subgroup base rates differ (e.g., sepsis prevalence by race), the SDK must report this explicitly, because Rajkomar's incompatibility theorem becomes binding when base rates diverge.

4. **Checklist integration:** The paper's local fairness checklist maps to the SDK's Phase 1 equity extraction step. Every scenario design document should answer: (a) Are subgroup base rates reported? (b) Are training demographics reported? (c) Are subgroup performance metrics reported? (d) Which fairness definition is the paper's primary claim?

**Connection to other papers:** Obermeyer et al. (Paper #15) demonstrates equal-performance–defined fairness masking massive outcome disparities (equal calibration ≠ equal treatment), which is a direct empirical instantiation of Rajkomar's mathematical incompatibility result. The SDK's equity module should cite both.
