# lint-as: healthcare_sim_sdk/scenarios/example/scenario.py
import random
import numpy as xp
import numpy.random as npr
from numpy import random as nr
from numpy.random import default_rng
from scipy import stats

a = xp.random.normal(size=3)        # call through an alias
b = xp.random.default_rng           # reference without a call
c = npr.random(3)
d = nr.random(3)
e = random.random()
f = getattr(xp, "random")
g = stats.beta.rvs(1, 2, size=3)    # global NumPy state
h = default_rng(1)
