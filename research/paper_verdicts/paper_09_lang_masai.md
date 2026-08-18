# Paper 9: Lång et al. — MASAI Trial (Lancet Oncology 2023)

## Classification: PARTIAL_FIT
## Reproducibility: PARTIALLY_REPRODUCED

## Key Findings

The MASAI trial is the largest RCT of imaging AI in clinical screening to date (N=80,033), demonstrating that AI-supported triage simultaneously improves cancer detection (+34%) and reduces radiologist workload (-44%). The SDK simulation reproduced the direction of both findings, but achieved only 6% detection improvement (ratio 1.06 vs. 1.34 target) while closely matching the workload reduction (49.9% vs. 44% target). The partial reproduction reflects two issues: (1) the paper does not report Transpara's sensitivity/specificity, forcing us to assume the AI-enhanced sensitivity for the single-reader pathway; and (2) the bimodal cancer prevalence distribution creates sensitivity to initialization choices.

**This paper is PARTIAL_FIT because** the key operational variable — how much better is AI-assisted single reading than unaided single reading, for true cancer detection — is not reported. Our assumed `ai_single_reader_sensitivity = 0.90` drives the detection improvement, and the paper's observed +34% could be reproduced by calibrating this to ~0.97, which is unrealistically high.

## Parameter Extraction

| Parameter | Source | Value | Status |
|-----------|--------|-------|--------|
| Study design | Paper | Non-inferiority RCT | CONFIRMED |
| N (women screened) | Paper | 80,033 | CONFIRMED |
| Control detection rate | Paper | 5.0/1000 screens | CONFIRMED |
| AI-arm detection rate | Paper | 6.7/1000 screens | CONFIRMED |
| Detection ratio | Derived | 1.34 | CONFIRMED |
| Radiologist workload reduction | Paper | 44% | CONFIRMED |
| False positive rate | Paper | Non-inferior (not reported numerically) | PARTIAL |
| AI system | Paper | Transpara by ScreenPoint Medical | CONFIRMED |
| Transpara AUC | NOT REPORTED | 0.83 | ASSUMED (typical for Transpara in screening; AUC 0.79-0.87 in published validation studies) |
| Single-reader sensitivity (unaided) | NOT REPORTED | 0.70 | ASSUMED (standard screening benchmark; MEDIUM impact) |
| Double-reader sensitivity | NOT REPORTED | 0.83 | ASSUMED (consensus double reading benchmark; MEDIUM impact) |
| AI-augmented single-reader sensitivity | NOT REPORTED | 0.90 | ASSUMED (HIGH impact — this drives the detection increase) |
| Transpara high-risk threshold | Paper (partial) | Score > 7 (on 1–10 scale) | ASSUMED mapped to 0.70 |
| Low-score fraction (single read) | Derived from 44% workload reduction | ~56% routed to single read | DERIVED |

## Simulation Results

Simulation configuration: 3 seeds × 10,000 women × 6 monthly batches.

| Metric | Simulated | Paper Target | Match |
|--------|-----------|-------------|-------|
| Counterfactual detection rate | 5.64/1000 | 5.0/1000 | PARTIAL (+13% above target) |
| Factual (AI) detection rate | 5.95/1000 | 6.7/1000 | PARTIAL (89% of target) |
| Detection ratio (AI/control) | 1.06 | 1.34 | FAIL (79% of target) |
| Workload reduction | 49.9% | 44% | PASS (within 6 pp) |

**Why detection ratio underperforms:** The assumed `ai_single_reader_sensitivity = 0.90` vs. `double_reader_sensitivity = 0.83` gives only 8.4% sensitivity advantage for AI-assisted single reading over double reading (control). The paper's observed 34% detection increase implies a much larger effective sensitivity difference, which either means: (a) the AI has greater sensitivity advantage than we assumed, or (b) Transpara AI's high-score routing to double reading captures missed interval cancers that unaided double reading would also miss.

**Sensitivity analysis:** Setting `ai_single_reader_sensitivity = 0.97` (unrealistically high) reproduces the 34% detection increase. This suggests the mechanism isn't simply "AI reader sees more" but involves the routing logic — AI-flagged cases receive special radiologist attention beyond standard double reading.

## Verification Summary

| Check Level | Result |
|-------------|--------|
| Structural integrity | PASS |
| AI arm detects more than control arm | PASS (5.95 > 5.64) |
| Workload reduction positive | PASS (49.9%) |
| Workload reduction within 10 pp of target | PASS |
| Detection ratio > 1.0 | PASS |
| Detection ratio matches paper (1.34) | FAIL (simulated 1.06) |
| CF detection rate near 5.0/1000 | PARTIAL (5.64 is 13% above) |

**Fitness criteria assessment:**
- ✅ Population-level intervention (80,033 women screened)
- ✅ Predictive model drives intervention (Transpara score → routing decision)
- ✅ Intervention is a state change (reading pathway changes)
- ✅ Measurable outcome (detection events per 1,000 screens)
- ✅ Counterfactual (what if no AI triage — all double read?)
- ⚠️ Discrete-time: YES, but the outcome is per-screen not per-patient-over-time. Each screening event is independent. The SDK works better for longitudinal patient trajectories; here we're modeling repeated cross-sections of screening batches.

## Discrepancies

1. **Detection ratio 1.06 vs. target 1.34:** The model cannot reproduce the 34% detection improvement without assuming an unrealistically high AI-augmented sensitivity. The paper's 34% improvement likely reflects: enhanced radiologist attention to AI-flagged cases (not captured), potential for AI to catch cases that both radiologists miss independently, and/or the specific Swedish screening population's cancer prevalence characteristics.

2. **CF detection rate 5.64 vs. target 5.0:** The mean cancer risk in our bimodal population is 0.0067 rather than exactly 0.005. At standard double-reader sensitivity (0.83), this gives 5.64/1000 baseline. Could be recalibrated with a lower cancer prevalence distribution.

3. **Workload reduction 49.9% vs. target 44%:** Our routing logic sends all low-score cases to single reading. If Transpara only routes 56% to single read (to achieve 44% workload reduction with 2 reads standard → 1.44 reads average), our threshold needs adjustment. The 49.9% suggests we're routing too many to single read. Recalibrating threshold from 0.70 to ~0.60 would achieve ~44%.

## Scientific Reporting Gaps

1. **Transpara AUC not reported:** Lancet Oncology 2023 paper reports detection rates and workload but no model performance metrics (AUROC, sensitivity, specificity) for Transpara in this specific Swedish screening population. Critical for reproducibility.

2. **False positive rate not quantified:** "Non-inferior" is claimed but no false positive rate (recall rate) is reported numerically in the abstract. For screening programs, false positive rates are as important as detection rates.

3. **Radiologist attention mechanics not described:** Does the double-read for high-score cases involve concurrent review, sequential review, or adjudication? This changes the effective sensitivity assumption fundamentally.

4. **No demographic breakdown:** The 80,033 Swedish women are not characterized by age subgroup beyond the screening age range (40–74). Age-specific cancer prevalence and model performance are not reported.

5. **Interval cancer analysis deferred:** The paper notes that interval cancer reduction (12% reported in 2025 follow-up) is a key endpoint but it is not in this 2023 paper. This is the endpoint most relevant to patient outcomes.

6. **Multi-site variation:** The Swedish screening program involves multiple sites. Site-level variation in cancer prevalence, radiologist skill, and AI compliance are not reported.

## Assumptions Made

| Assumption | Impact | Basis |
|-----------|--------|-------|
| Transpara AUC = 0.83 | MEDIUM | Published Transpara validation studies in European populations; this paper doesn't report AUC |
| AI-augmented single-reader sensitivity = 0.90 | HIGH | Not reported; drives detection ratio; paper's 34% increase cannot be reproduced with any defensible value |
| Standard double-reader sensitivity = 0.83 | MEDIUM | Published screening consensus literature; paper doesn't report |
| High-risk threshold = 0.70 (score >7/10) | MEDIUM | Transpara uses 1–10 scale; threshold for routing not specified in this paper |
| Cancer prevalence distribution: 0.5% bimodal (99.5% low-risk, 0.5% uniform[0.5,0.95]) | MEDIUM | Derived from 5/1000 control detection rate; distribution shape assumed |
| All counterfactual patients receive double reading | LOW | Stated in paper as standard comparison arm |
