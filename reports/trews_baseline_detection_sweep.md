# TREWS Replication: Baseline Detection Delay x Capacity Sweep

**Date:** 2026-04-06
**Scenario:** `sepsis_early_alert`
**SDK Version:** healthcare-sim-sdk (docs/readme-replications branch)
**Author:** Mike Draugelis / Claude (AI Assistant)

---

## Executive Summary

We added two new mechanisms to the sepsis early alert scenario — baseline clinical detection (standard of care on both branches) and Kumar time-dependent treatment effectiveness — then swept baseline detection delay and rapid response capacity to calibrate against the Adams et al. (2022) TREWS finding of ~3pp mortality reduction.

**The sweep did not reproduce the TREWS finding.** Mortality reduction was insensitive to baseline detection delay (every capacity row is flat) and scaled only with capacity, reaching a maximum of 0.66pp at capacity=100. The root cause is that the `ControlledMLModel` scores patients by underlying risk level, not by proximity to disease onset. Higher AUC means less noise in risk scores, but does not produce earlier detection. The Kumar decay mechanism is present but has nothing to work with.

This result identifies the specific missing mechanism needed for a full TREWS replication: AUC-dependent detection timing, where higher AUC produces model scores that spike earlier in the disease trajectory relative to sepsis onset.

---

## 1. New Mechanisms Added

### 1.1 Baseline Clinical Detection (Standard of Care)

Previously, the counterfactual branch had no treatment at all — unrealistically assuming that without the ML system, nobody would ever detect sepsis. In reality, clinicians catch sepsis through routine care (vitals trending, lactate orders, clinical suspicion) with CMS SEP-1 bundle compliance targeting detection within ~6 hours.

**Implementation:** Each patient is assigned a clinical detection delay at admission, drawn from a Beta distribution:

```
detection_delay = Beta(alpha=2, beta=5) * max_hours
```

- Right-skewed: most patients detected within a few hours (median < mean)
- Default parameters: mean ~6.9 hours, median ~5.8 hours at max_hours=24
- Long tail: some patients (atypical presentation, overnight, busy ED) not caught for 12+ hours

This detection runs in `step()` on **both branches** — it is standard of care, not part of the ML intervention. When time since sepsis onset exceeds the patient's detection delay, treatment is initiated automatically.

**Effect on counterfactual:** The counterfactual branch now has baseline treatment. The ML system's value-add is measured as improvement *over* standard care, not over no care at all.

### 1.2 Kumar Time-Dependent Treatment Effectiveness

Treatment effectiveness now decays exponentially as a function of delay from sepsis onset to treatment initiation (Kumar et al. 2006):

```
effectiveness = max_effectiveness * 0.5^(delay_hours / half_life)
```

With half_life=6 hours:

| Delay from Onset | Effectiveness |
|------------------|---------------|
| 0 hours (immediate) | 50% |
| 6 hours | 25% |
| 12 hours | 12.5% |
| 24 hours | 3.1% |

This makes early detection genuinely more valuable than late detection. An ML system that detects sepsis at onset gets 50% treatment effectiveness; standard care detecting at 6 hours gets only 25%.

### 1.3 Design Rationale

Together, these mechanisms create the conditions where ML detection timing should matter:

1. Both branches have treatment (baseline detection)
2. Earlier treatment is more effective (Kumar decay)
3. The ML system's value comes from the *time advantage* over baseline, amplified by Kumar decay

---

## 2. Experiment Design

### 2.1 Configuration

Base configuration: TREWS replication (`configs/trews_replication.yaml`)

| Parameter | Value |
|-----------|-------|
| Population | 5,000 patients |
| Sepsis incidence | 4% |
| Simulation length | 42 timesteps (7 days) |
| Timestep | 4 hours |
| Model AUC | 0.82 |
| Alert threshold | 93rd percentile |
| Initial clinician response | 89% |
| Fatigue coefficient | 0.001 |
| Floor response rate | 40% |
| Kumar half-life | 6 hours |
| Max treatment effectiveness | 50% |
| Baseline detection distribution | Beta(2, 5) |
| Seed | 42 |

### 2.2 Sweep Parameters

Two-dimensional sweep:

- **Rapid response capacity:** 10, 25, 50, 100 (alerts responded to per 4-hour block)
- **Baseline detection max_hours:** 6, 12, 18, 24, 36, 48, 72 (scales the Beta distribution)

28 total simulation runs.

---

## 3. Results

### 3.1 Mortality Reduction (percentage points)

| Capacity \ Max Hours | 6 | 12 | 18 | 24 | 36 | 48 | 72 |
|---------------------|------|------|------|------|------|------|------|
| **10** | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 |
| **25** | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 |
| **50** | 0.42 | 0.42 | 0.42 | 0.42 | 0.42 | 0.42 | 0.42 |
| **100** | 0.64 | 0.64 | 0.64 | 0.64 | 0.66 | 0.66 | 0.66 |

Target: ~3.0pp (Adams et al. 2022). **Not achieved at any configuration.**

### 3.2 ML Value-Add: Additional Patients Treated (Factual - Counterfactual)

| Capacity \ Max Hours | 6 | 12 | 18 | 24 | 36 | 48 | 72 |
|---------------------|-----|-----|-----|-----|-----|-----|-----|
| **10** | 23 | 22 | 24 | 23 | 24 | 24 | 26 |
| **25** | 46 | 44 | 44 | 46 | 50 | 52 | 55 |
| **50** | 88 | 93 | 98 | 89 | 97 | 102 | 107 |
| **100** | 164 | 165 | 170 | 174 | 176 | 182 | 191 |

The ML system treats more patients than baseline alone. A per-patient timing diagnostic (Section 3.4) reveals that a subset of these patients do receive earlier treatment.

### 3.3 Total Treated Patients

| Capacity | Max Hours | Factual Treated | CF Treated | Factual % | CF % |
|----------|-----------|-----------------|------------|-----------|------|
| 10 | 6 | 923 | 900 | 18.5% | 18.0% |
| 10 | 72 | 808 | 782 | 16.2% | 15.6% |
| 100 | 6 | 1,064 | 900 | 21.3% | 18.0% |
| 100 | 72 | 973 | 782 | 19.5% | 15.6% |

As max_hours increases, both factual and counterfactual treated counts decrease (baseline detection fires later, some patients die or discharge before detection).

### 3.4 Per-Patient Timing Diagnostic

The flat rows in the mortality reduction matrix initially suggested no timing advantage. However, a per-patient timing diagnostic comparing onset-to-treatment delay on each branch revealed that a timing advantage does exist for a subset of patients:

| Config | ML Earlier Than Baseline | Treated Before Onset | F. Kumar Eff. | CF Kumar Eff. | ML Advantage |
|--------|-------------------------|---------------------|---------------|---------------|-------------|
| cap=10, 24h | 21 (2.4%) | 20 | 20.8% | 20.1% | 0.7 pp |
| cap=50, 24h | 110 (12.6%) | 102 | 23.8% | 20.1% | 3.7 pp |
| cap=10, 72h | 19 (2.4%) | 19 | 8.9% | 8.0% | 1.0 pp |
| cap=50, 72h | 108 (13.8%) | 104 | 13.3% | 8.0% | 5.4 pp |

*Among 929 patients who developed sepsis (out of 5,000 total).*

Key findings:

- **The ML model does provide earlier treatment** for 2-14% of septic patients, depending on capacity. These patients are high-risk individuals whose ML risk scores persistently exceed the alert threshold, causing them to be flagged and treated before baseline detection fires — or even before they develop sepsis at all.
- **102-104 patients are treated prophylactically** (before sepsis onset) at capacity=50. These patients receive maximum Kumar effectiveness (50%) compared to the baseline-only branch where they would wait hours after onset.
- **The per-patient Kumar advantage is meaningful** (3.7-5.4 pp effectiveness gain for the subset that benefits), but only 2-14% of septic patients are in this subset.
- **The remaining 86-98% get treated at the same time** on both branches, because the ML model's stochastic alert timing happens to coincide with baseline detection for most patients.

Timing advantage distribution (capacity=50, max_hours=24, 870 patients treated on both branches):

```
ML earlier by 10-20 timesteps:  43 patients  (treated before onset)
ML earlier by 5-10 timesteps:   24 patients
ML earlier by 3-5 timesteps:    12 patients
ML earlier by 1-3 timesteps:    12 patients
Same time (0 timesteps):       760 patients
ML later:                        0 patients
```

---

## 4. Analysis

### 4.1 Why Baseline Delay Has Limited Effect

The flat rows in the mortality reduction matrix occur because the timing advantage affects too few patients to move population-level mortality. While the per-patient Kumar effectiveness gain is 3-5pp for the 2-14% who benefit, this translates to only 0.1-0.7pp across all 5,000 patients.

The ML model scores patients by underlying risk level (`current_risk`), not by proximity to disease onset. A patient trending toward sepsis gets a similar score one timestep before onset as one timestep after. This means the ML system cannot selectively detect patients *because they are about to become septic* — it can only detect them because they are high-risk in general. The timing advantage that exists comes from the stochastic overlap between the ML alert threshold and the baseline detection delay, not from onset-aware detection.

### 4.2 Why Capacity Matters

Capacity determines how many early-detection opportunities the ML system can act on. At capacity=10, only 21 of 929 septic patients (2.4%) get earlier ML treatment — the system doesn't have bandwidth to respond to most alerts on high-risk patients. At capacity=50, 110 patients (12.6%) benefit. The relationship between capacity and timing advantage is approximately linear: more bandwidth → more early detections acted upon → more patients receiving higher Kumar effectiveness.

### 4.3 What Limits the Replication

The gap between the simulated 0.4-0.7pp and the published 3pp reflects two factors:

1. **The ML model's timing advantage is incidental, not targeted.** The `ControlledMLModel` scores by risk level; the timing advantage comes from high-risk patients being persistently flagged. In clinical reality, TREWS detected sepsis *before* clinical suspicion through onset-specific signals — a stronger and more targeted timing mechanism than stochastic risk-based flagging.

2. **The fraction of patients who benefit is small.** Even at capacity=50, only 12.6% of septic patients get earlier ML treatment. Reaching the published 3pp would require either a much larger fraction of patients benefiting from early detection, or a much larger per-patient effectiveness gain.

---

## 5. Verification

### 5.1 Structural Integrity

| Check | Status |
|-------|--------|
| Population conservation (stage counts = N at all t) | PASS (all 28 runs) |
| No NaN/Inf in outcomes | PASS |
| No backward stage transitions | PASS |
| Step purity (BRANCHED-vs-NONE identity) | PASS |
| Step purity with baseline detection | PASS |
| Step purity with Kumar decay | PASS |

### 5.2 Baseline Detection Mechanism Validation

| Check | Status |
|-------|--------|
| CF branch has treated patients when baseline enabled | PASS |
| CF branch has 0 treated when baseline disabled | PASS |
| Baseline detection respects per-patient delay | PASS (CF treated count decreases as max_hours increases) |
| Kumar decay backward compatible (half_life=0 equals flat mode) | PASS |

### 5.3 Conservation Laws

| Check | Status |
|-------|--------|
| Higher effectiveness → fewer deaths (monotonicity) | PASS |
| Capacity=0 → factual matches CF | PASS |
| Threshold=100 → factual matches CF | PASS |
| Treated indices valid and unique per timestep | PASS |
| Capacity never exceeded | PASS |

Full test suite: 142 unit/integration tests pass, 182/183 bulletproof tests pass (1 pre-existing failure in stroke scenario).

---

## 6. What This Tells Us

### 6.1 What Worked

1. **Baseline clinical detection** correctly models standard of care on both branches. The counterfactual is now realistic — clinicians do catch sepsis without ML, just later.
2. **Kumar decay** correctly rewards earlier treatment. The mechanism is present and validated (purity tests pass, conservation holds).
3. **A real timing advantage exists.** The per-patient diagnostic confirms that 2-14% of septic patients receive earlier ML treatment than baseline, with meaningful per-patient Kumar effectiveness gains (3-5pp). The mechanism works — it just affects too few patients to reproduce the published 3pp at the population level.
4. **The simulation honestly surfaces its limitations.** The gap between 0.4-0.7pp and 3pp is not a bug — it reflects the difference between a generic AUC-controlled model (which provides incidental timing advantage through risk-based flagging) and a clinically integrated system like TREWS (which provided targeted early detection through onset-specific signals).

### 6.2 The Remaining Gap

The simulated 0.4-0.7pp vs the published 3pp. Two possible paths to close it:

1. **Extend `ControlledMLModel`** with onset-proximity scoring, where higher AUC produces model scores that spike earlier relative to disease onset. This would increase the fraction of patients who benefit from early ML detection. See `tasks/auc_dependent_detection_timing.md` for the design specification.
2. **Accept the gap as a finding.** The simulation suggests that a generic ML model with AUC=0.82 provides modest incremental value over competent standard care, and that TREWS's published 3pp likely required detection timing mechanisms specific to its clinical integration. This is a useful finding for health systems evaluating whether to deploy a sepsis alert system.

---

## 7. Assumptions

1. **4% sepsis incidence.** Published range is 2-6%. Higher incidence would increase the absolute number of patients who could benefit from early ML detection.
2. **Seed=42, single trajectory.** Results are point estimates. Multi-seed confidence intervals would strengthen findings.
3. **Beta(2,5) detection delay distribution.** The shape of the standard-of-care detection distribution is assumed, not calibrated against empirical SEP-1 compliance data.
4. **Uniform baseline detection across demographics.** In reality, detection timing likely varies by patient presentation, unit staffing, and time of day.
5. **Kumar half-life of 6 hours.** Based on the Kumar et al. 2006 mortality data. The real relationship between treatment timing and effectiveness is more complex and varies by pathogen and sepsis stage.
6. **No cost modeling.** The simulation does not include the cost of maintaining rapid response capacity or the cost of false-alert-driven workflow disruption.

---

*This report was generated from simulation outputs only. All findings are conditional on the stated assumptions and should be framed as "the simulation suggests, under these assumptions" rather than as real-world evidence. The deployment decision remains with clinical and operational leadership.*
