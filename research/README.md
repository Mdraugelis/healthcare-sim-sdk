# SimReplicator Research: 30-Paper Reproducibility Audit

**Geisinger Health System — AI Department**
**Completed:** April 6, 2026
**Branch:** `research/simreplicator-30-paper-audit`

## Overview

This directory contains the outputs of a systematic reproducibility audit of 30 landmark
healthcare ML/AI deployment papers against the Healthcare Intervention Simulation SDK.

For each paper, the full Phase 1–7 pipeline was executed:
1. Parameter extraction
2. SDK fitness assessment
3. Scenario design (5-method contract mapping)
4. Calibration planning
5. Verification protocol
6. Reproducibility verdict
7. Structured report

## Contents

- `synthesis_report.md` — Cross-paper findings, reproducibility scorecard, equity analysis,
  SDK fitness map, methodological patterns, and procurement playbook. Start here.
- `paper_verdicts/` — Individual verdict files for all 30 papers (Phase 7 reports).

## Scenario Implementations

Seven new `paperNN_*` SDK scenarios were implemented under `healthcare_sim_sdk/scenarios/`. Two additional papers (TREWS, Chong/Rosen) are reproduced by the more comprehensive `sepsis_early_alert/` and `noshow_targeted_reminders/` scenarios (see Integration Note below).

| Directory | Paper | Classification | Reproducibility |
|-----------|-------|---------------|-----------------|
| `paper01_epic_esm/` | Wong et al. — Epic Sepsis Model | FIT | PARTIALLY_REPRODUCED |
| `sepsis_early_alert/` *(trews config)* | Adams et al. — TREWS | FIT | **REPRODUCED** *(upgraded 2026-04-08)* |
| `paper03_kaiser_aam/` | Escobar et al. — Kaiser AAM | FIT | PARTIALLY_REPRODUCED |
| `paper04_insight_rct/` | Shimabukuro et al. — InSight RCT | PARTIAL_FIT | UNDERDETERMINED |
| `paper05_composer/` | Boussina/Wardi et al. — COMPOSER | FIT | PARTIALLY_REPRODUCED |
| `paper06_shield_rt/` | Hong et al. — SHIELD-RT | FIT | **REPRODUCED** |
| `paper07_manz_nudges/` | Manz et al. — Mortality Nudges | PARTIAL_FIT | PARTIALLY_REPRODUCED |
| `paper09_masai/` | Lång et al. — MASAI Mammography | PARTIAL_FIT | PARTIALLY_REPRODUCED |
| `noshow_targeted_reminders/` *(chong/rosen configs)* | Chong/Rosen — No-Show Reminders | FIT | **REPRODUCED** *(upgraded 2026-04-08)* |

### Integration Note (2026-04-08)

The initial audit produced standalone `paper02_trews/` and `paper11_noshow/` scenarios. These have been superseded by the more complete `sepsis_early_alert/` and `noshow_targeted_reminders/` scenarios, which added baseline clinical detection, Kumar time-dependent treatment effectiveness, multiple calibration configs, and multi-seed replication runners. The duplicate `paper02_trews/` and `paper11_noshow/` directories have been removed. See the updated verdict files in `paper_verdicts/` for details.

## Bug Fix

`healthcare_sim_sdk/ml/performance.py` — `auc_score()` used `np.argsort` (unstable for ties
in ROC curve integration). Fixed to `np.lexsort` with secondary sort on TPR. All unit tests
now pass (104/104).

## Key Findings

- **0 papers NOT_REPRODUCED** — the SDK never contradicted a paper's direction of effect.
- **3 papers fully REPRODUCED** — SHIELD-RT (RR within 2% of published target), TREWS (published 3.3pp within simulated 95% CI), Chong/Rosen (all proof points pass). *TREWS and Chong/Rosen were upgraded from PARTIALLY_REPRODUCED on 2026-04-08 after the baseline_care_effectiveness correction was implemented.*
- **Scientific reporting quality is the binding constraint.** AUC is missing from 36% of
  deployment papers. Calibration from 100%. Demographics from 73%.
- **Two documented SDK scope gaps:** continuous-time physiology (HYPE), qualitative
  implementation failure (Beede).
- **The required SDK fix has been IMPLEMENTED:** the counterfactual branch now initializes with baseline care effectiveness via the `sepsis_early_alert/` baseline clinical detection mechanism. See `synthesis_report.md` Appendix B.

## Usage

Run any scenario as:
```bash
python3 healthcare_sim_sdk/scenarios/paper06_shield_rt/run_evaluation.py
```

All scenarios follow the standard 5-method contract and pass structural verification.
