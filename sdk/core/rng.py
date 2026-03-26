"""RNG stream partitioning for reproducible branched simulation.

Uses numpy's SeedSequence to create statistically independent RNG streams
for different simulation processes. This ensures factual and counterfactual
branches diverge only where intervention changed state, not because of
RNG desynchronization.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class RNGStreams:
    """Named RNG streams for different simulation processes.

    Each stream is an independent np.random.Generator, deterministically
    derived from a master seed. This guarantees:
    1. Different processes (population gen, temporal evolution, etc.)
       don't interfere with each other across runs.
    2. The factual and counterfactual branches of a branched simulation
       produce identical stochastic evolution for processes they share
       (e.g., temporal dynamics) because they use the same stream state.
    3. Intervention randomness lives on its own stream, so the
       counterfactual branch (which skips intervention) doesn't
       advance the shared streams out of sync.
    """
    population: np.random.Generator
    temporal: np.random.Generator
    prediction: np.random.Generator
    intervention: np.random.Generator
    outcomes: np.random.Generator


class RNGPartitioner:
    """Creates deterministic, independent RNG streams from a master seed.

    Uses numpy's SeedSequence to spawn child streams that are
    statistically independent but fully reproducible.

    Usage:
        partitioner = RNGPartitioner(master_seed=42)
        streams = partitioner.create_streams()
        # streams.population, streams.temporal, etc.
    """

    STREAM_NAMES = [
        "population", "temporal", "prediction",
        "intervention", "outcomes",
    ]

    def __init__(self, master_seed: int = 42):
        self.master_seed = master_seed
        self._seed_seq = np.random.SeedSequence(master_seed)

    def create_streams(self) -> RNGStreams:
        """Create a fresh set of independent RNG streams."""
        child_seeds = self._seed_seq.spawn(len(self.STREAM_NAMES))
        generators = {
            name: np.random.default_rng(seed)
            for name, seed in zip(self.STREAM_NAMES, child_seeds)
        }
        return RNGStreams(**generators)

    def fork(self) -> "RNGPartitioner":
        """Create a new partitioner for a branched simulation.

        Returns a new partitioner that produces streams starting from
        the same seed state as the original. The forked streams are
        independent Generator objects (no shared mutable state) but
        produce identical values when consumed in the same order.

        This ensures the counterfactual branch's temporal and outcome
        streams start synchronized with the factual branch. The branches
        diverge only where intervention advances a stream on one branch
        but not the other.
        """
        return RNGPartitioner(self.master_seed)
