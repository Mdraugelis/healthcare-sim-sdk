"""Unit tests for RNG partitioning."""

import numpy as np

from sdk.core.rng import RNGPartitioner


class TestRNGStreams:
    def test_all_fields_are_generators(self):
        p = RNGPartitioner(42)
        streams = p.create_streams()
        for name in RNGPartitioner.STREAM_NAMES:
            gen = getattr(streams, name)
            assert isinstance(gen, np.random.Generator), (
                f"{name} is not a Generator"
            )

    def test_five_streams(self):
        assert len(RNGPartitioner.STREAM_NAMES) == 5


class TestReproducibility:
    def test_same_seed_same_draws(self):
        draws_a = RNGPartitioner(42).create_streams().temporal.random(10)
        draws_b = RNGPartitioner(42).create_streams().temporal.random(10)
        np.testing.assert_array_equal(draws_a, draws_b)

    def test_all_streams_reproducible(self):
        for name in RNGPartitioner.STREAM_NAMES:
            a = getattr(RNGPartitioner(99).create_streams(), name)
            b = getattr(RNGPartitioner(99).create_streams(), name)
            np.testing.assert_array_equal(
                a.random(5), b.random(5),
                err_msg=f"Stream '{name}' not reproducible",
            )

    def test_different_seeds_differ(self):
        a = RNGPartitioner(1).create_streams().temporal.random(10)
        b = RNGPartitioner(2).create_streams().temporal.random(10)
        assert not np.array_equal(a, b)


class TestStreamIndependence:
    def test_consuming_one_stream_does_not_affect_another(self):
        """The core independence proof.

        Draw from temporal, consume intervention heavily, draw from
        temporal again. Compare against a clean run with no intervention
        consumption. The temporal draws must be identical.
        """
        # Run A: consume intervention between temporal draws
        sa = RNGPartitioner(42).create_streams()
        val_a1 = sa.temporal.random()
        sa.intervention.random(1000)  # heavy consumption
        val_a2 = sa.temporal.random()

        # Run B: no intervention consumption
        sb = RNGPartitioner(42).create_streams()
        val_b1 = sb.temporal.random()
        val_b2 = sb.temporal.random()

        assert val_a1 == val_b1
        assert val_a2 == val_b2

    def test_streams_produce_different_sequences(self):
        """Different named streams should not produce the same values."""
        s = RNGPartitioner(42).create_streams()
        temporal = s.temporal.random(100)
        prediction = s.prediction.random(100)
        assert not np.array_equal(temporal, prediction)

    def test_cross_stream_zero_correlation(self):
        """Correlation between streams should be near zero."""
        s = RNGPartitioner(42).create_streams()
        n = 10_000
        draws = {
            name: getattr(s, name).random(n)
            for name in RNGPartitioner.STREAM_NAMES
        }
        for i, name_a in enumerate(RNGPartitioner.STREAM_NAMES):
            for name_b in RNGPartitioner.STREAM_NAMES[i + 1:]:
                corr = np.corrcoef(draws[name_a], draws[name_b])[0, 1]
                assert abs(corr) < 0.05, (
                    f"Correlation between {name_a} and {name_b}: {corr}"
                )


class TestFork:
    def test_fork_returns_partitioner(self):
        p = RNGPartitioner(42)
        forked = p.fork()
        assert isinstance(forked, RNGPartitioner)

    def test_fork_streams_are_independent_of_original(self):
        """Forked streams must not share mutable state with original."""
        p = RNGPartitioner(42)
        original = p.create_streams()
        forked_streams = p.fork().create_streams()

        # Consume original temporal heavily
        original.temporal.random(1000)

        # Forked temporal should be unaffected (independent Generator)
        p2 = RNGPartitioner(42)
        fresh_streams = p2.create_streams()
        np.testing.assert_array_equal(
            forked_streams.temporal.random(10),
            fresh_streams.temporal.random(10),
        )

    def test_fork_produces_identical_starting_streams(self):
        """Forked streams start from the same seed state as original."""
        p = RNGPartitioner(42)
        p.create_streams()  # consume the original's spawn slot
        forked = p.fork().create_streams()

        # Both should produce the same temporal draws (before any divergence)
        p2 = RNGPartitioner(42)
        fresh = p2.create_streams()
        np.testing.assert_array_equal(
            forked.temporal.random(10),
            fresh.temporal.random(10),
        )

    def test_fork_preserves_master_seed(self):
        p = RNGPartitioner(42)
        forked = p.fork()
        assert forked.master_seed == 42
