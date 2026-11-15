# lint-as: healthcare_sim_sdk/scenarios/example/scenario.py
from healthcare_sim_sdk.core.scenario import BaseScenario


class Done(BaseScenario):
    def create_population(self, n):
        return n

    def step(self, state, t):
        return state

    def predict(self, state, t):
        return None

    def intervene(self, state, predictions, t):
        return state, None

    def measure(self, state, t):
        return None
