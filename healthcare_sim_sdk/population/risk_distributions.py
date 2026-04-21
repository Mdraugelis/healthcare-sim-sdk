"""Population risk distribution generation.

Creates heterogeneous patient risk profiles using beta distributions
that match real-world clinical patterns where most patients have
low risk but a small fraction drives the majority of events.
"""

import numpy as np


def beta_distributed_risks(
    n_patients: int,
    annual_incident_rate: float,
    concentration: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate patient-level annual risk scores from a beta distribution.

    The beta distribution is parameterized to produce a right-skewed
    distribution centered on the target incident rate. The concentration
    parameter controls heterogeneity: lower values produce more spread.

    Args:
        n_patients: Number of patients to generate.
        annual_incident_rate: Target population-level annual event rate.
        concentration: Beta distribution alpha parameter. Lower = more
            heterogeneous. Typical range: 0.3-1.0.
        rng: NumPy random Generator. Must be a partitioned stream from
            the scenario's RNGPartitioner (typically ``self.rng.population``)
            to preserve reproducibility under the RNG-partitioning invariant.

    Returns:
        Array of per-patient annual risk probabilities, shape (n_patients,).
    """
    alpha = concentration
    beta_param = alpha * (1.0 / annual_incident_rate - 1.0)

    raw_risks = rng.beta(alpha, beta_param, n_patients)

    # Scale to match target population rate
    scaling_factor = annual_incident_rate / np.mean(raw_risks)
    risks = np.clip(raw_risks * scaling_factor, 0, 0.99)

    return risks
