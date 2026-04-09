# Paper 8: Wijnberge et al. — HYPE RCT (JAMA 2020)

## Classification: PARTIAL_FIT
## Reproducibility: UNDERDETERMINED

## Key Findings

HYPE presents a compelling intraoperative hypotension prediction RCT — the Hypotension Prediction Index (HPI) prevents MAP < 65 mmHg before it occurs. The core challenge for the SDK: the outcome is a **continuous physiologic measurement** (time-weighted average MAP in mmHg) over a **continuous temporal process** (intraoperative hemodynamics, second-to-second). The SDK is designed for discrete-time population-level simulation with binary events. TWA hypotension (0.44 → 0.10 mmHg) is not a rate, not a probability, and not a population-level aggregate — it's a per-patient continuous integral over a surgical episode.

**No simulation was run.** A credible simulation of HYPE would require modeling MAP hemodynamics at 1-minute resolution with pharmacologic intervention dynamics (vasopressor dosing, fluid loading), which falls outside the SDK's discrete-time binary-events architecture. To force HYPE into the SDK, we would need to:
1. Discretize hypotension into binary "hypotension episodes" per minute
2. Sum across minutes to get episode burden per patient
3. Model HPI alert → anesthesiologist intervention → vasopressor effect → MAP recovery

This is technically feasible but would require 3+ major undocumented assumptions (MAP distribution model, vasopressor dose-response, hemodynamic autocorrelation structure). Under the AGENTS.md rule "if >3 major parameters assumed, classify UNDERDETERMINED," this paper falls there.

## Parameter Extraction

| Parameter | Source | Value | Status |
|-----------|--------|-------|--------|
| Study design | Paper | RCT | CONFIRMED |
| N | Paper | 68 patients | CONFIRMED |
| Setting | Paper | Elective noncardiac surgery, Amsterdam UMC | CONFIRMED |
| Primary outcome | Paper | TWA hypotension (mmHg) | CONFIRMED |
| Control arm TWA hypotension | Paper | 0.44 mmHg | CONFIRMED |
| Intervention arm TWA hypotension | Paper | 0.10 mmHg | CONFIRMED |
| Relative reduction | Paper | 77% | CONFIRMED |
| Protocol compliance with HPI alerts | Paper | 81% | CONFIRMED |
| Model | Paper | Hypotension Prediction Index (arterial waveform) | CONFIRMED |
| Model AUC | NOT REPORTED | Unknown | MISSING |
| Alert threshold | NOT REPORTED | Unknown | MISSING — paper doesn't report HPI threshold used |
| Time resolution of monitoring | NOT REPORTED | ~1 minute | ASSUMED (standard arterial line monitoring) |
| Vasopressor intervention type | NOT REPORTED | Unknown | MISSING — paper doesn't specify protocol |
| MAP distribution (intraoperative) | NOT REPORTED | Unknown | MISSING — required for simulation |
| Autocorrelation structure of MAP | NOT REPORTED | Unknown | MISSING |
| Surgery duration distribution | NOT REPORTED | Unknown | MISSING |

## Simulation Results

**No simulation executed.** PARTIAL_FIT classification — see fitness criteria below.

The SDK would require a custom hemodynamic module to simulate continuous MAP trajectories. The closest existing scenario (sepsis_early_alert) uses discrete timesteps of ~30 minutes, not the ~1-minute resolution needed for intraoperative monitoring. A future SDK extension for continuous physiologic monitoring scenarios would require:
- `step()` operating at 1-minute resolution with AR(1) or Ornstein-Uhlenbeck MAP dynamics
- `predict()` mapping the HPI (multi-feature arterial waveform index) to a binary alert
- `intervene()` modeling vasopressor dose-response (MAP recovery curve)
- `measure()` computing cumulative hypotension integral per timestep

## Verification Summary

Not applicable — simulation not run.

**Fitness criteria assessment:**
- ✅ Population-level intervention (68 patients, HPI alerts to anesthesiologists)
- ✅ Predictive model drives intervention (HPI → alert → vasopressor)
- ✅ Intervention is a state change (MAP increases after vasopressor)
- ⚠️ Measurable outcome: The outcome is continuous (TWA mmHg), not a discrete binary event rate. Convertible to "hypotension episodes per 10-minute block" but requires strong distributional assumptions.
- ✅ Counterfactual causal question (what if no HPI alert?)
- ❌ Discrete-time dynamics: Arterial blood pressure is fundamentally continuous. Discretizing to 1-minute bins introduces model error that cannot be validated against any published MAP distribution in this population.

**Why PARTIAL_FIT and not FIT:** The continuous dynamics of hemodynamics and the per-patient continuous outcome (TWA mmHg, not a rate) mean at least 2 fitness criteria fail cleanly. We could discretize (MAP < 65 mmHg in each 5-minute bin: yes/no), but this discards the severity information that makes TWA meaningful.

## Discrepancies

N/A — no simulation run.

## Scientific Reporting Gaps

1. **HPI threshold not reported:** The paper uses the Hypotension Prediction Index without specifying what threshold triggered intervention (the HPI is a proprietary score from 0–100). Without this, we cannot compute sensitivity/specificity.

2. **Model performance statistics absent:** No AUC, sensitivity, or specificity for the HPI is reported in this paper. The HPI is commercially available and its performance has been published elsewhere, but this paper provides no internal validation.

3. **N=68 is underpowered for subgroup analysis:** The paper reports no demographic breakdowns and the RCT is too small to detect differential effects by age, ASA class, or baseline hemodynamic status.

4. **Protocol compliance mechanism not described:** 81% compliance means 19% of alerts were not acted upon. The paper doesn't describe what happened in those cases (alert ignored? already treating? patient factors?).

5. **Clinical outcomes (mortality, organ dysfunction) not reported:** TWA hypotension is a physiologic surrogate. The paper does not report 30-day mortality, renal injury, or hospital LOS. These are what clinicians and payers actually care about.

6. **No follow-up on hemodynamic instability duration:** The 77% reduction in TWA hypotension is reported for the intraoperative period only. PACU and postoperative hemodynamics are not reported.

## Assumptions Made

No simulation-level assumptions made (no simulation run). 

If forced to simulate, the following would be required (all HIGH-impact assumptions):
- MAP autocorrelation structure (Ornstein-Uhlenbeck parameters for healthy vs. at-risk patients)
- Vasopressor dose-response curve (time-to-MAP-increase after phenylephrine/ephedrine)
- HPI alert threshold and sensitivity at that threshold
- Surgery duration distribution
- Baseline MAP distribution by patient risk category
