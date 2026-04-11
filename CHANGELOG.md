# Changelog

All notable changes to this project are documented in this file.

## [0.1.0] - 2026-04-11

Official stable release after `v0.1.0-beta`.

### Added
- New validated scenario families and replication assets, including no-show targeted reminders, sepsis early alert, and multiple paper-aligned scenario packages.
- Experiment lifecycle utilities (`finalize_experiment`) and sweep registration tooling.
- Generic validation framework enhancements and expanded bulletproof/integration test coverage.
- Invariant enforcement and safety hooks (`.claude/invariants.yaml`, pre-commit checks, and validation scripts).
- New design documentation in `docs/design_principles.md`.

### Changed
- Documentation and research artifacts were significantly expanded and restructured for reproducibility and release-readiness.
- Repository outputs handling was tightened to keep generated experiment artifacts out of source control.
- Organization-specific wording was removed from reports/configs to make scenarios portable across health systems.

### Improved
- `ControlledMLModel` gained Platt-scaling calibration support.
- Overbooking threshold optimization moved to Hydra-driven configuration sweeps.

### Notes
- Compared against previous release tag: `v0.1.0-beta`.
- Since `v0.1.0-beta`, mainline includes broad feature growth, validation hardening, and documentation maturity suitable for an official release.
