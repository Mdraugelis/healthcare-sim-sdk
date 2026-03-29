"""Temporal risk dynamics using AR(1) processes.

Models time-varying patient risk through autoregressive processes
with optional seasonal effects. The AR(1) process maintains
realistic autocorrelation (>0.8) in risk trajectories.
"""

import numpy as np


def annual_risk_to_hazard(
    annual_risk: np.ndarray,
) -> np.ndarray:
    """Convert annual risk probability to hazard rate.

    S(1 year) = 1 - annual_risk = exp(-h)
    h = -ln(1 - annual_risk)
    """
    clipped = np.clip(annual_risk, 0, 0.999999)
    return -np.log(1 - clipped)


def hazard_to_timestep_probability(
    hazard: np.ndarray,
    timestep_duration: float,
) -> np.ndarray:
    """Convert hazard rate to event probability for a discrete timestep.

    P(event in timestep) = 1 - exp(-h * dt)
    """
    return 1 - np.exp(-hazard * timestep_duration)


class AR1Process:
    """Autoregressive process of order 1 for temporal risk modifiers.

    X_t = rho * X_{t-1} + (1 - rho) * mu + noise_t
    where noise_t ~ N(0, sigma^2)

    The process generates multiplicative risk modifiers that are
    applied to base risk scores. A modifier of 1.0 means no change;
    >1.0 means elevated risk; <1.0 means reduced risk.

    Args:
        n_entities: Number of entities (patients).
        rho: Persistence parameter (0.8-0.95 typical).
        sigma: Innovation standard deviation.
        mu: Long-run mean of the process.
        bounds: (lower, upper) bounds for risk modifiers.
    """

    def __init__(
        self,
        n_entities: int,
        rho: float = 0.9,
        sigma: float = 0.1,
        mu: float = 1.0,
        bounds: tuple[float, float] = (0.5, 2.0),
    ):
        self.n_entities = n_entities
        self.rho = rho
        self.sigma = sigma
        self.mu = mu
        self.bounds = bounds
        self.current = np.full(n_entities, mu)

    def step(self, rng: np.random.Generator) -> np.ndarray:
        """Advance one timestep. Returns current modifiers.

        Args:
            rng: Random generator (use scenario's temporal stream).

        Returns:
            Array of risk modifiers, shape (n_entities,).
        """
        noise = rng.normal(0, self.sigma, self.n_entities)
        self.current = (
            self.rho * self.current
            + (1 - self.rho) * self.mu
            + noise
        )
        self.current = np.clip(self.current, *self.bounds)
        return self.current.copy()

    def step_with_season(
        self,
        rng: np.random.Generator,
        t: int,
        seasonal_amplitude: float = 0.2,
        seasonal_period: int = 52,
    ) -> np.ndarray:
        """Advance one timestep with seasonal effect.

        Seasonal modifier: 1.0 + amplitude * sin(2*pi*t/period + pi/2)
        """
        seasonal = 1.0 + seasonal_amplitude * np.sin(
            2 * np.pi * t / seasonal_period + np.pi / 2
        )
        noise = rng.normal(0, self.sigma, self.n_entities)
        self.current = (
            self.rho * self.current
            + (1 - self.rho) * seasonal
            + noise
        )
        self.current = np.clip(self.current, *self.bounds)
        return self.current.copy()
