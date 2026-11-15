# lint-as: healthcare_sim_sdk/scenarios/example/scenario.py
import numpy as np
from numpy.random import Generator
from scipy import stats


def draw(rng: np.random.Generator, gen: Generator):
    x = rng.normal(size=3)
    y = stats.beta.rvs(1, 2, size=3, random_state=rng)
    return x, y, gen
