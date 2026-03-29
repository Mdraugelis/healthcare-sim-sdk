"""Bulletproof test suite: statistical, numerical, and structural validation.

These tests are designed to be run repeatedly to catch foundational issues
in the simulation SDK. Categories:

1. Statistical sanity checks -- distributions, rates, expected values
2. Numerical stability -- NaN, Inf, overflow, underflow, degenerate inputs
3. RNG reproducibility -- seed determinism, stream independence, fork sync
4. Boundary conditions -- zero/one/empty/huge inputs, edge cases
5. Conservation laws -- population counts, probability sums, monotonicity
6. Scenario-specific -- domain invariants for each implemented scenario
"""
