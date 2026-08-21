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
            heterogeneous.

            This choice sets a HARD CEILING on achievable model AUC.
            Outcomes are stochastic draws from risk, so even an oracle
            knowing every true risk cannot order them perfectly; its AUC
            depends entirely on how spread out risk is. Measured at
            annual_incident_rate=0.10 over a 4-week window:

                concentration   oracle AUC (Bayes ceiling)
                    5.00            0.626
                    1.00            0.745
                    0.50            0.821
                    0.30            0.862
                    0.15            0.919
                    0.06            0.958
                    0.03            0.967

            So a scenario configuring an AUC-0.95 model on a
            concentration=0.5 population is internally inconsistent: no
            such model can exist there. Published rare-event models with
            AUC > 0.90 imply very heterogeneous populations -- at
            concentration=0.15 the 10th percentile risk is ~0.0000
            against a 90th percentile of 0.367.

            Raising heterogeneity at a fixed mean also raises realized
            event counts in any finite window, which loosens PPV
            constraints; the two are not independent knobs.

            Typical range 0.3-1.0 for moderately heterogeneous
            populations; go lower when modelling a deployment whose
            reported model AUC exceeds ~0.86.
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
