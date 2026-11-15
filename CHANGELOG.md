# Changelog

Release notes for the Healthcare Intervention Simulation SDK. Newest first.

## 0.2.0 — 2026-11-15

The first release since `v0.1.0` (21 August 2026). It covers pull requests #5 to #26.

Results from 0.1.0 should not be compared with results from 0.2.0. The model simulator, the stroke scenario and the exports all changed.

### Breaking changes

| Change | What to do |
|---|---|
| Runner flags are gone. Every runner starts through Hydra and takes settings as `key=value`. | Replace `--n-days 90 --model-auc 0.83` with `n_days=90 model_auc=0.83`. A key that is not in the runner's YAML is refused. |
| `ControlledBinaryClassifier` is removed. It read the outcome label. | Use `ControlledMLModel`. See `examples/05_controlled_ml_model.ipynb`. |
| The stroke scenario gives different numbers. Its default model is now achievable (AUC 0.75), treatment is applied once, and a stroke ends follow-up. | Re-run any stroke result. Do not reuse 0.1.0 figures. |
| `ControlledMLModel` scores no longer use the outcome label. The fit report is computed on held-out data, and a target above the Bayes ceiling is refused in every mode. | Re-run any result that depends on model scores. A target that used to fit may now be refused. |
| Analysis exports changed: missing periods are explicit, rows are aligned by entity ID, and treatment has named options. | Read `docs/analysis_exports.md` before reusing analysis code written against 0.1.0. |
| A runner registers a run in the catalog only if it passed its checks. A run that fails is kept on disk, marked rejected, and the runner exits non-zero. | Read results with `ExperimentCatalog.list_verified()`. |
| Agent guidance moved. `AGENTS.md` is the source; `CLAUDE.md` is a symlink to it. The protocol is `docs/simulation_protocol.md`. The invariants are `config/invariants.yaml`. | Old paths under `.claude/` still resolve. Update links when convenient. |
| Work that used a published result to set its inputs is labeled a calibration exercise. This covers the `paper01`, `paper03`, `paper04` and `paper05` runners and the sepsis TREWS run. | Do not cite them as replications of those studies. See "Published-Study Calibration Exercises" in `README.md`. |

### Added

- **Registration gate.** `finalize_experiment()` saves, records provenance, validates, then registers (`docs/experiment_verification.md`).
- **Runner contract.** Every launchable script under `scenarios/` starts through `@hydra.main` and ends with a strict `finalize_experiment()` call. Seven named tools are exempt, each with a reason, in `config/invariants.yaml`.
- **Lint rules SDK001 to SDK007** and documentation checks DOC001 to DOC009 (`docs/lint_rules.md`).
- **Hermetic quality gate.** `scripts/quality_gate.sh` runs lint, the public tests, all notebooks and two identical wheel builds in a hash-locked Python 3.12 environment with the network off. CI runs the same script.
- **Review packets and case traces** for five kinds of reviewer (`docs/review_packets.md`).
- **Held-out model checks, analysis-design certification and a frozen-input benchmark runner.**
- **Guidance beside the code.** An `AGENTS.md` in each of the 14 scenario folders and in `core/`. `docs/AUTHORITY.md` says which document governs what.
- **Algorithm Description Document**, now version 0.5 (`docs/algorithm_description.md`).

### Fixed

- All 29 findings of the algorithm review, I-01 to I-29 (Appendix B of the Algorithm Description Document).
- Two scenarios' sweep YAML files existed but no code read them. They now drive their runners.
- The overbooking evaluation looked for its validation files in the wrong folder.

### Known limits

- The ITS design study in Section 11.5 of the Algorithm Description Document has not been re-run on the corrected stroke scenario.
- Difference-in-differences and regression-discontinuity have exports only. There is no estimator for either.
- Code outside `scenarios/`, such as a notebook, can still register a run without checks. Such a run is labeled unverified.
- The malnutrition scenario needs a private model file to run. Its public tests use an uncalibrated synthetic model.
- The quality gate runs on Python 3.12. The package declares 3.10 and later; other versions are not tested in CI.

### Install

```bash
pip install git+https://github.com/Mdraugelis/healthcare-sim-sdk.git@v0.2.0
```

## 0.1.0 — 2026-08-21

Initial public release.
