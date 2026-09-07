# Internal numerical audit of the teaching labs

Date: 2026-09-05. This is internal computational review, not external peer review or clinical validation.

`check_numerics.py` loads the exported functions from the actual `assets/app.js` in Node, without a browser DOM. The retained `numerical_results.json` records **137 passing checks across 66 JavaScript calls**, with source/checker hashes, runtime versions, expected values, actual values and limitations. No numerical defect was found within the checked slider domains and boundary cases. The Python checker also passes flake8 at 88 columns.

Reproduce from the repository root:

```bash
.venv/bin/python docs/textbook/qa/check_numerics.py
```

The script uses the bundled Node runtime when available, otherwise Node on PATH; `--node` and `--output` permit explicit paths. Its exit status is nonzero if a comparison fails. Rerun after editing the lab mathematics; a retained result applies only to its recorded source hash.

## Independent reference checks

- **Bayes:** hand-calculated confusion counts and PPV, empty populations, no positive flags, no cases and all cases. The ordinary case also agrees with the SDK helper. The lab correctly returns null/undefined for a zero positive-flag denominator; the SDK helper instead returns zero below its denominator guard. Chapter 06 now discloses this distinction.
- **Hazards:** 50-digit decimal survival powers, the hand-computable six-month risk of .20 from annual risk .36, and actual SDK conversions; zero, small and high risks are included. The JS stable `log1p`/`expm1` form avoids the tiny-value cancellation permitted in the SDK comparison tolerance.
- **Pairing and Monte Carlo precision:** correlations −1 through +1, unequal branch SDs, zero variance, standard-error scaling and SciPy's normal quantile. Negative covariance increases variance as documented.
- **Calibration:** independent SciPy transformations, rank-based AUC including ties and missing classes, and the expected-Brier decomposition into irreducible Bernoulli variance plus squared probability distortion. Positive logit slopes preserve AUC over the checked fixture.
- **Allocation:** hand-specified patient gains and both ordering rules under equal budgets, including exhausted/overreserved capacity. With one optional slot, risk ordering yields −.08 incremental expected events averted; the audit therefore includes a harmful allocation example.
- **ITS:** planted null/effect/confounded coefficients, different breaks, a singular system, deterministic filtered-noise reconstruction by convolution, NumPy least-squares coefficients/fitted values, and actual SDK segmented-regression coefficients. A coincident jump is recovered as part of the fitted level, not separated as a causal effect.
- **Monitoring:** a manually calculated null path and post-change CUSUM path, strict threshold crossing, alarm reset, ramp endpoints, and absence of a point alarm at a moderate shift. Fifty-two independent checks with per-check null probability .0027 yield a 13.1154432% chance of at least one alarm; high-precision decimal references cover this and rare-event limits.

## Explanatory corrections

Clarified adjacent prose in chapters 05–08, 10 and 11: the Monte Carlo mean stays fixed; the calibration lab does not fit a production model; the six-patient allocation fixture differs from the preceding three-patient example; the ITS fixture has its own baseline and deterministic filtered innovations; and neither an immediate rule nor CUSUM is universally faster. The monitoring lab varies standardized shift, allowance and threshold, not baseline variability.

The JavaScript was not edited during this audit. Browser interactions, accessible rendering, persistence and exports need separate UI checks. These deterministic calculations do not estimate clinical effects, estimator coverage, study power, or a CUSUM's real-world false-alarm probability. Arbitrary invalid inputs outside the teaching controls are not a supported public API claim.
