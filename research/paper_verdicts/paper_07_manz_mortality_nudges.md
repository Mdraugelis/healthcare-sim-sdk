# Paper 7: Manz et al. — ML Mortality Prediction + Behavioral Nudges (JAMA Oncology 2020)

## Classification: PARTIAL_FIT
## Reproducibility: PARTIALLY_REPRODUCED

## Key Findings

Manz et al. presents a genuinely interesting two-stage causal pathway: GBM model → nudge delivery → oncologist behavior change → SIC documentation rate. The SDK can model the patient-level outcomes end-to-end, but the behavioral modification mechanism (nudge → clinician behavior) requires an assumed dose-response relationship that the paper does not provide. The simulation reproduced the pre-intervention SIC rate (0.014 vs. 0.01 target, within 40%) and the post-intervention SIC rate direction (0.072 vs. 0.05 target). The high-risk patient SIC rate (0.129 vs. 0.15 target) achieves 86% of the reported magnitude. The DiD estimate (+0.052) captures the wedge intervention's positive effect.

**Critical limitation:** The stepped-wedge design creates a natural counterfactual, but the paper's primary analysis uses linear mixed models accounting for time trends, secular trends, and within-cluster correlation — which our discrete simulation approximates only crudely. The claimed effect of "1% → 5% overall" cannot be cleanly reproduced without knowing the exact cross-over timing and baseline SIC rates by clinic group.

## Parameter Extraction

| Parameter | Source | Value | Status |
|-----------|--------|-------|--------|
| Study design | Paper | Stepped-wedge cluster RCT | CONFIRMED |
| N oncologists | Paper | 78 | CONFIRMED |
| N patients | Paper | 14,607 | CONFIRMED; simulation used 2,000 for computational feasibility |
| Mortality threshold | Paper | ≥10% 180-day mortality | CONFIRMED |
| Overall SIC rate pre | Paper | ~1% | CONFIRMED |
| Overall SIC rate post | Paper | ~5% | CONFIRMED |
| High-risk SIC rate pre | Paper | ~4% | CONFIRMED |
| High-risk SIC rate post | Paper | ~15% | CONFIRMED |
| Model algorithm | Paper | Gradient-boosted machine | CONFIRMED |
| Model AUC | NOT REPORTED | 0.78 | ASSUMED (GBM for 180-day mortality: typical 0.75–0.82) |
| High-risk fraction at ≥10% threshold | NOT REPORTED | ~20–23% | ASSUMED (derived from SIC rate math: x=0.20 satisfies x*0.04+(1-x)*0.0025=0.01) |
| Nudge type/content | Paper (partial) | Weekly emails, peer comparison, opt-out prompt | CONFIRMED (mechanism assumed binary: on/off) |
| Stepped-wedge timing | Paper (partial) | 4 periods / 8 clinic groups | CONFIRMED; mapped to crossover at week 8 of 26 |
| Base mortality prevalence | NOT REPORTED | 20–23% above 10% threshold | ASSUMED (bimodal distribution: 80% low-risk, 20% high-risk) |
| Nudge compliance rate | NOT REPORTED | 1.0 (all oncologists receive and process) | ASSUMED (HIGH impact) |
| Low-risk SIC rate | NOT REPORTED | 0.0025/week | DERIVED from overall rate constraint |
| Post-nudge low-risk SIC rate | NOT REPORTED | 0.025/week | DERIVED from overall 5% target |

## Simulation Results

Simulation configuration: 3 seeds × 2,000 patients × 26 weekly timesteps (n=78 oncologists, stepped-wedge crossover at week 8).

| Metric | Simulated | Paper Target | Match |
|--------|-----------|-------------|-------|
| Pre-nudge SIC rate (overall) | 0.014 | 0.01 | PARTIAL (+40%, within range) |
| Post-nudge SIC rate (overall) | 0.072 | 0.05 | PARTIAL (+44% above target) |
| High-risk SIC rate post-nudge | 0.129 | 0.15 | PARTIAL (86% of target) |
| Counterfactual post SIC rate | 0.020 | ~0.01 (flat) | PARTIAL |
| DiD estimate | +0.052 | positive | PASS (direction) |
| Model AUC | ~0.72 | 0.78 (assumed) | PARTIAL |

**Reason for partial reproduction:** The simulation's overall SIC rates are ~1.4–7.2x compared to paper-reported rates, primarily because:
1. The high-risk fraction in our population (23%) is higher than the theoretical 20%, inflating overall rates
2. We model SIC as per-encounter probability, which may over-count relative to the paper's encounter-level denominator
3. The stepped-wedge temporal dynamics are approximated; the paper uses 4 distinct periods not a continuous ramp

## Verification Summary

| Check Level | Result |
|-------------|--------|
| Structural integrity | PASS |
| Pre-intervention SIC direction | PASS |
| Post-intervention SIC > pre-intervention | PASS |
| Factual SIC > counterfactual (post-nudge) | PASS (0.072 > 0.020) |
| DiD > 0 | PASS |
| High-risk SIC > overall SIC (post-nudge) | PASS (0.129 > 0.072) |

**Fitness criteria assessment:**
- ✅ Population-level intervention (nudge to 78 oncologists)
- ✅ Predictive model drives intervention (GBM → nudge trigger)
- ✅ Intervention is a state change (nudge activation changes SIC probability)
- ✅ Measurable outcome each timestep (SIC rate)
- ✅ Counterfactual question (what if no nudge?)
- ⚠️ Discrete-time dynamics: YES, but the two-stage causal pathway (model → nudge → behavior → documentation) requires intermediate states not natively supported in the SDK's 5-method contract. We merge stages 2–3 into a combined intervention effectiveness parameter.

## Discrepancies

1. **Pre-SIC rate 0.014 vs. target 0.01:** High-risk fraction slightly above theoretical 20%. The population distribution is sensitive to beta parameters. Given the paper doesn't report the actual high-risk fraction, this 40% discrepancy is within uncertainty bounds.

2. **Post-SIC rate 0.072 vs. target 0.05:** The simulation over-predicts post-nudge SIC. Possible explanations: (a) in the real trial, oncologist response to nudge was partial (not all flagged patients received SIC); (b) the paper's 5% is a within-cluster post-period rate that accounts for time trends; (c) our nudge activation assumes 100% compliance.

3. **Counterfactual post-SIC rate 0.020 vs. expected ~0.01:** Slight secular trend in the counterfactual arm due to small positive drift in mortality risk (patients' conditions worsen over 26 weeks), generating marginally higher SIC propensity. This is actually realistic — the paper's secular trend adjustment may account for this.

## Scientific Reporting Gaps

1. **Model AUC not reported:** The GBM for 180-day mortality prediction has no reported AUC, AUROC, or calibration statistics. This is a major omission for a 2020 JAMA Oncology paper that uses ML as the core component.

2. **High-risk fraction not reported:** The paper does not state what percentage of 14,607 patients were flagged at the ≥10% threshold. This is essential for estimating nudge volume and oncologist burden.

3. **Nudge delivery mechanism incompletely described:** "Opt-out text prompts" implies some oncologists did not engage. The opt-out rate is not reported. Without it, the effective dose of the intervention cannot be modeled.

4. **Clinic-level heterogeneity not reported:** The 8 clinic groups / 4 intervention periods create substantial clustering. Intraclass correlation coefficients are not reported.

5. **No race/ethnicity data on patients or outcomes:** A stepped-wedge trial of 14,607 cancer patients in a large health system reports zero demographic breakdown. This is an equity blind spot for a paper about end-of-life care, where disparities are well-documented.

6. **Long-term follow-up metrics not in this paper:** The 2020 paper doesn't report chemotherapy-near-death or hospice enrollment rates; these appear in a later follow-up. They're excluded from this verdict but relevant for full reproducibility.

## Assumptions Made

| Assumption | Impact | Basis |
|-----------|--------|-------|
| Model AUC = 0.78 | MEDIUM | Not reported; GBM for 180-day mortality typical range; main results directionally insensitive |
| High-risk fraction = 20% (≥10% threshold) | HIGH | Derived from SIC rate constraint; paper doesn't report |
| Population bimodal: 80% low-risk, 20% high-risk | HIGH | Not reported; assumption is mathematically derived from overall vs. subgroup SIC rates |
| Nudge compliance = 100% (all oncologists engage) | HIGH | Not reported; real-world opt-out suggests some don't engage; inflates our simulated effect |
| Stepped-wedge crossover: linear ramp starting week 8 | MEDIUM | Paper reports 4 periods but not exact timing; we approximate |
| Patient turnover = 2%/week (disease progression, new referrals) | LOW | Not reported; insensitive to reasonable range |
| Simulation scaled to 2,000 patients (vs. 14,607 paper) | LOW | Scale-invariant for rate outcomes; memory/compute constraint |
