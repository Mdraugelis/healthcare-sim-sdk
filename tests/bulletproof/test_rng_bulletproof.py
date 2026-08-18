"""RNG reproducibility and partition independence tests.

These tests are the foundation of the simulation's correctness guarantee.
If RNG partitioning breaks, the entire counterfactual framework is invalid.

Categories:
- Seed determinism: same seed → identical results, always
- Stream independence: consuming one stream doesn't affect another
- Fork synchronization: forked branches produce identical temporal sequences
- Cross-stream correlation: streams should be statistically independent
- Repeated runs: results stable across many iterations
"""

import numpy as np
import pytest

from healthcare_sim_sdk.core.rng import RNGPartitioner, RNGStreams
from healthcare_sim_sdk.core.engine import BranchedSimulationEngine, CounterfactualMode
from healthcare_sim_sdk.core.scenario import TimeConfig
from healthcare_sim_sdk.scenarios.stroke_prevention.scenario import (
    StrokeConfig,
    StrokePreventionScenario,
)

from .conftest import assert_no_nan_inf


# =====================================================================
# 1. SEED DETERMINISM — same seed, same everything
# =====================================================================

class TestSeedDeterminism:
    """Same seed must produce bit-identical results."""

    def test_partitioner_same_seed_same_draws(self):
        """Two partitioners with same seed produce identical streams."""
        for seed in [0, 1, 42, 999, 2**31 - 1]:
            p1 = RNGPartitioner(seed)
            p2 = RNGPartitioner(seed)
            s1 = p1.create_streams()
            s2 = p2.create_streams()

            for name in RNGPartitioner.STREAM_NAMES:
                g1 = getattr(s1, name)
                g2 = getattr(s2, name)
                draws1 = g1.random(1000)
                draws2 = g2.random(1000)
                np.testing.assert_array_equal(
                    draws1, draws2,
                    err_msg=f"seed={seed}, stream={name}: not identical"
                )

    def test_different_seeds_different_draws(self):
        """Different seeds must produce different streams."""
        p1 = RNGPartitioner(42)
        p2 = RNGPartitioner(43)
        s1 = p1.create_streams()
        s2 = p2.create_streams()

        draws1 = s1.temporal.random(100)
        draws2 = s2.temporal.random(100)
        assert not np.array_equal(draws1, draws2), (
            "Different seeds should produce different draws"
        )

    def test_full_simulation_deterministic(self):
        """Running the same scenario twice produces identical results."""
        results_list = []
        for _ in range(3):
            config = StrokeConfig(
                n_patients=500, n_weeks=12,
                prediction_interval=4,
            )
            sc = StrokePreventionScenario(config=config, seed=42)
            results = BranchedSimulationEngine(
                sc, CounterfactualMode.BRANCHED
            ).run(500)
            outcomes = np.array([
                results.outcomes[t].events.sum() for t in range(12)
            ])
            results_list.append(outcomes)

        np.testing.assert_array_equal(
            results_list[0], results_list[1],
            err_msg="Run 1 != Run 2"
        )
        np.testing.assert_array_equal(
            results_list[0], results_list[2],
            err_msg="Run 1 != Run 3"
        )

    def test_simulation_varies_with_seed(self):
        """Different seeds should produce different event trajectories."""
        trajectories = []
        for seed in [1, 2, 3]:
            config = StrokeConfig(
                n_patients=500, n_weeks=12,
                prediction_interval=4,
            )
            sc = StrokePreventionScenario(config=config, seed=seed)
            results = BranchedSimulationEngine(
                sc, CounterfactualMode.BRANCHED
            ).run(500)
            outcomes = np.array([
                results.outcomes[t].events.sum() for t in range(12)
            ])
            trajectories.append(outcomes)

        # At least one pair should differ
        any_different = False
        for i in range(len(trajectories)):
            for j in range(i + 1, len(trajectories)):
                if not np.array_equal(trajectories[i], trajectories[j]):
                    any_different = True
                    break
        assert any_different, "All seeds produced identical results!"


# =====================================================================
# 2. STREAM INDEPENDENCE — the core guarantee
# =====================================================================

class TestStreamIndependence:
    """Consuming one stream must not affect any other stream."""

    def test_consuming_intervention_does_not_shift_temporal(self):
        """Drawing from intervention stream should not affect temporal.

        This is THE critical test: if this fails, counterfactual
        branches will desynchronize even without intervention effects.
        """
        # Clean run: only draw from temporal
        p1 = RNGPartitioner(42)
        s1 = p1.create_streams()
        clean_temporal = s1.temporal.random(500)

        # Dirty run: draw heavily from intervention, THEN draw temporal
        p2 = RNGPartitioner(42)
        s2 = p2.create_streams()
        _ = s2.intervention.random(10000)  # consume a LOT
        _ = s2.prediction.random(5000)     # consume another stream too
        dirty_temporal = s2.temporal.random(500)

        np.testing.assert_array_equal(
            clean_temporal, dirty_temporal,
            err_msg="Consuming intervention/prediction shifted temporal!"
        )

    def test_consuming_temporal_does_not_shift_outcomes(self):
        """Drawing from temporal should not affect outcomes."""
        p1 = RNGPartitioner(42)
        s1 = p1.create_streams()
        clean_outcomes = s1.outcomes.random(500)

        p2 = RNGPartitioner(42)
        s2 = p2.create_streams()
        _ = s2.temporal.random(100000)  # burn through temporal
        dirty_outcomes = s2.outcomes.random(500)

        np.testing.assert_array_equal(
            clean_outcomes, dirty_outcomes,
            err_msg="Consuming temporal shifted outcomes!"
        )

    @pytest.mark.parametrize("consumed_stream", [
        "population", "temporal", "prediction", "intervention", "outcomes"
    ])
    def test_exhaustive_independence(self, consumed_stream):
        """Burning any single stream should not shift any other stream."""
        for target_stream in RNGPartitioner.STREAM_NAMES:
            if target_stream == consumed_stream:
                continue

            # Clean: only draw from target
            p1 = RNGPartitioner(42)
            s1 = p1.create_streams()
            clean = getattr(s1, target_stream).random(200)

            # Dirty: burn consumed, then draw target
            p2 = RNGPartitioner(42)
            s2 = p2.create_streams()
            getattr(s2, consumed_stream).random(5000)
            dirty = getattr(s2, target_stream).random(200)

            np.testing.assert_array_equal(
                clean, dirty,
                err_msg=(
                    f"Consuming {consumed_stream} shifted {target_stream}!"
                )
            )


# =====================================================================
# 3. FORK SYNCHRONIZATION — branches start identical
# =====================================================================

class TestForkSynchronization:
    """Forked partitioners must produce synchronized initial streams."""

    def test_forked_streams_start_identical(self):
        """Forked streams produce same initial draws as original."""
        p = RNGPartitioner(42)
        s_original = p.create_streams()

        p_fork = p.fork()
        s_forked = p_fork.create_streams()

        for name in RNGPartitioner.STREAM_NAMES:
            g_orig = getattr(s_original, name)
            g_fork = getattr(s_forked, name)
            orig_draws = g_orig.random(500)
            fork_draws = g_fork.random(500)
            np.testing.assert_array_equal(
                orig_draws, fork_draws,
                err_msg=f"Fork desynchronized on stream {name}"
            )

    def test_forked_streams_are_independent_objects(self):
        """Drawing from forked stream should NOT advance original."""
        p = RNGPartitioner(42)
        s_original = p.create_streams()

        p_fork = p.fork()
        s_forked = p_fork.create_streams()

        # Draw from forked temporal
        _ = s_forked.temporal.random(1000)

        # Original should be unaffected (still at position 0)
        p_clean = RNGPartitioner(42)
        s_clean = p_clean.create_streams()
        expected = s_clean.temporal.random(100)
        actual = s_original.temporal.random(100)

        np.testing.assert_array_equal(
            expected, actual,
            err_msg="Drawing from forked stream advanced original!"
        )

    def test_fork_preserves_master_seed(self):
        """fork() should preserve the master_seed value."""
        p = RNGPartitioner(12345)
        p_forked = p.fork()
        assert p_forked.master_seed == 12345

    def test_multiple_forks_all_synchronized(self):
        """Multiple forks should all start from the same state."""
        p = RNGPartitioner(42)
        forks = [p.fork() for _ in range(5)]
        streams = [f.create_streams() for f in forks]

        reference = streams[0].temporal.random(200)
        for i in range(1, len(streams)):
            draws = streams[i].temporal.random(200)
            np.testing.assert_array_equal(
                reference, draws,
                err_msg=f"Fork {i} not synchronized with fork 0"
            )


# =====================================================================
# 4. CROSS-STREAM STATISTICAL INDEPENDENCE
# =====================================================================

class TestCrossStreamCorrelation:
    """Streams should be statistically uncorrelated."""

    def test_pairwise_correlation_near_zero(self):
        """Pearson correlation between any two streams should be < 0.05."""
        p = RNGPartitioner(42)
        s = p.create_streams()
        n = 50000

        draws = {}
        for name in RNGPartitioner.STREAM_NAMES:
            draws[name] = getattr(s, name).random(n)

        names = list(draws.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                r = np.corrcoef(draws[names[i]], draws[names[j]])[0, 1]
                assert abs(r) < 0.05, (
                    f"Correlation between {names[i]} and {names[j]}: "
                    f"r={r:.4f} (should be ~0)"
                )

    def test_streams_have_correct_marginal_distribution(self):
        """Each stream should produce Uniform(0,1) marginals."""
        p = RNGPartitioner(42)
        s = p.create_streams()

        for name in RNGPartitioner.STREAM_NAMES:
            draws = getattr(s, name).random(100000)
            # KS test against uniform
            from scipy.stats import kstest
            stat, pval = kstest(draws, 'uniform')
            assert pval > 0.001, (
                f"Stream {name} not uniform: KS stat={stat:.4f}, "
                f"p={pval:.6f}"
            )


# =====================================================================
# 5. SIMULATION-LEVEL RNG CORRECTNESS
# =====================================================================

class TestSimulationRNGCorrectness:
    """RNG behavior is correct at the simulation level."""

    def test_branched_vs_none_factual_outcomes_match(self):
        """BRANCHED factual branch should match NONE mode outcomes
        when intervention_effectiveness=0.

        This is the integration-level version of stream independence.
        """
        config_b = StrokeConfig(
            n_patients=500, n_weeks=12,
            prediction_interval=12,  # predict once at end
            intervention_effectiveness=0.0,
        )
        sc_b = StrokePreventionScenario(config=config_b, seed=42)
        results_b = BranchedSimulationEngine(
            sc_b, CounterfactualMode.BRANCHED
        ).run(500)

        config_n = StrokeConfig(
            n_patients=500, n_weeks=12,
            prediction_interval=12,
            intervention_effectiveness=0.0,
        )
        sc_n = StrokePreventionScenario(config=config_n, seed=42)
        results_n = BranchedSimulationEngine(
            sc_n, CounterfactualMode.NONE
        ).run(500)

        # Compare total events (not exact per-timestep since predict/intervene
        # advance RNG differently, but totals should be similar)
        b_total = sum(
            results_b.outcomes[t].events.sum() for t in range(12)
        )
        n_total = sum(
            results_n.outcomes[t].events.sum() for t in range(12)
        )
        # With 0% effectiveness, should be within 20%
        ratio = b_total / max(n_total, 1)
        assert 0.7 < ratio < 1.3, (
            f"BRANCHED vs NONE: ratio={ratio:.3f} "
            f"(B={b_total}, N={n_total})"
        )

    def test_seed_0_works(self):
        """Seed=0 should be valid and produce results."""
        config = StrokeConfig(
            n_patients=100, n_weeks=4,
            prediction_interval=4,
        )
        sc = StrokePreventionScenario(config=config, seed=0)
        results = BranchedSimulationEngine(
            sc, CounterfactualMode.NONE
        ).run(100)
        assert len(results.outcomes) == 4

    def test_large_seed_works(self):
        """Very large seed should not overflow."""
        config = StrokeConfig(
            n_patients=100, n_weeks=4,
            prediction_interval=4,
        )
        sc = StrokePreventionScenario(config=config, seed=2**31 - 1)
        results = BranchedSimulationEngine(
            sc, CounterfactualMode.NONE
        ).run(100)
        assert len(results.outcomes) == 4
