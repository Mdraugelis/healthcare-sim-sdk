# lint-as: healthcare_sim_sdk/scenarios/example/scenario.py
from healthcare_sim_sdk.core.scenario import BaseScenario


class Half(BaseScenario):
    def create_population(self, n):
        return n

    def step(self, state, t):
        pass

    def predict(self, state, t):
        ...

    def intervene(self, state, predictions, t):
        raise NotImplementedError

    def measure(self, state, t):
        """Docstring only."""
