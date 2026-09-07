"""Inspect actual engine execution through a transparent scenario decorator.

The wrapper observes scenario hooks; it neither reproduces the engine loop nor
teaches the engine about scenario state. Projectors receive defensive copies and
must be deterministic, side-effect-free functions that draw no random numbers.
"""

import copy
import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

import numpy as np

from healthcare_sim_sdk.core.scenario import BaseScenario


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(
        f"Trace fact of type {type(value).__name__} is not JSON serializable")


@dataclass(frozen=True)
class TraceRecord:
    """Immutable JSON snapshot with a stable reference inside its run."""
    trace_id: str
    timestep: int
    branch: str
    stage: str
    facts_json: str

    def to_dict(self) -> dict:
        return {"trace_id": self.trace_id, "timestep": self.timestep,
                "branch": self.branch, "stage": self.stage,
                "facts": json.loads(self.facts_json)}


class TracingScenario(BaseScenario):
    """Wrap one fresh scenario for one ordinary ``BranchedSimulationEngine.run``.

    ``projector(state)`` returns selected, JSON-compatible state facts. Default
    output facts contain counts only, keeping large populations inexpensive.
    ``output_projector(kind, output)`` can instead select detailed facts from
    ``prediction``, ``intervention``, and ``outcome`` outputs. Trace IDs are local
    to this run: persist them alongside the run manifest for durable references.

    The cap fails explicitly, never silently truncates evidence. RNG assignment
    and all optional hooks delegate to the wrapped scenario without extra draws.
    Snapshot counterfactual measurement is labeled ``snapshot_unresolved``:
    that engine mode reuses the factual RNG so branch cannot safely be inferred.
    Use branched mode for fully labeled paired stories.
    """

    def __init__(
        self, scenario: BaseScenario, projector: Callable[[Any], Mapping], *,
        output_projector: Optional[Callable[[str, Any], Mapping]] = None,
        stages: Optional[tuple[str, ...]] = None, max_records: int = 10000,
    ):
        self.scenario = scenario
        self.projector = projector
        self.output_projector = output_projector
        self.stages = set(stages) if stages is not None else None
        if max_records < 1:
            raise ValueError("max_records must be positive")
        self.max_records = max_records
        self.time_config = scenario.time_config
        self.seed = scenario.seed
        self.unit_of_analysis = scenario.unit_of_analysis
        self._partitioner = scenario._partitioner
        self._factual_rng = scenario.rng
        self._records: list[TraceRecord] = []
        self._started = False
        self._measure_counts: dict[tuple[int, int], int] = {}

    @property
    def rng(self):
        return self.scenario.rng

    @rng.setter
    def rng(self, value):
        self.scenario.rng = value

    @property
    def records(self) -> tuple[TraceRecord, ...]:
        return tuple(self._records)

    def to_rows(self) -> list[dict]:
        return [record.to_dict() for record in self._records]

    def _branch(self):
        return "factual" if self.rng is self._factual_rng else "counterfactual"

    def _append(self, t, stage, facts, branch=None):
        if self.stages is not None and stage not in self.stages:
            return
        if len(self._records) >= self.max_records:
            raise ValueError(
                "Trace max_records exceeded; increase cap or select fewer stages")
        self._records.append(TraceRecord(
            trace_id=f"trace-{len(self._records):06d}", timestep=t,
            branch=branch or self._branch(), stage=stage,
            facts_json=json.dumps(facts, default=_json_default, sort_keys=True,
                                  allow_nan=False),
        ))

    def _state(self, state, t, stage, branch=None):
        if self.stages is None or stage in self.stages:
            self._append(t, stage, self.projector(copy.deepcopy(state)), branch)

    def _output(self, kind, output, t, branch=None):
        if self.stages is not None and kind not in self.stages:
            return
        if self.output_projector is not None:
            facts = self.output_projector(kind, copy.deepcopy(output))
        elif kind == "prediction":
            facts = {"n_scores": len(output.scores)}
        elif kind == "intervention":
            facts = {"n_assigned": len(output.treated_indices)}
        else:
            facts = {"n_entities": len(output.events),
                     "event_total": float(np.sum(output.events))}
        self._append(t, kind, facts, branch)

    def create_population(self, n_entities):
        if self._started:
            raise ValueError(
                "TracingScenario is single-use; construct a fresh scenario per run")
        self._started = True
        state = self.scenario.create_population(n_entities)
        self._state(state, -1, "population", "shared")
        return state

    def step(self, state, t):
        self._state(state, t, "step_before")
        state = self.scenario.step(state, t)
        self._state(state, t, "step_after")
        return state

    def predict(self, state, t):
        predictions = self.scenario.predict(state, t)
        self._output("prediction", predictions, t)
        return predictions

    def intervene(self, state, predictions, t):
        self._state(state, t, "intervene_before")
        state, interventions = self.scenario.intervene(state, predictions, t)
        self._state(state, t, "intervene_after")
        self._output("intervention", interventions, t)
        return state, interventions

    def measure(self, state, t):
        # Detect snapshot's two measurements using one RNG. Mark both records
        # as unresolved instead of claiming the first is factual.
        branch = self._branch()
        key = (t, id(self.rng))
        count = self._measure_counts.get(key, 0)
        self._measure_counts[key] = count + 1
        if count:
            branch = "snapshot_unresolved"
            self._records = [
                TraceRecord(r.trace_id, r.timestep, "snapshot_unresolved",
                            r.stage, r.facts_json)
                if r.timestep == t
                and r.stage in {"measure_before", "measure_after", "outcome"}
                and r.branch == "factual" else r for r in self._records
            ]
        self._state(state, t, "measure_before", branch)
        outcomes = self.scenario.measure(state, t)
        self._state(state, t, "measure_after", branch)
        self._output("outcome", outcomes, t, branch)
        return outcomes

    def clone_state(self, state):
        return self.scenario.clone_state(state)

    def validate_population(self, state):
        return self.scenario.validate_population(state)

    def validate_results(self, results):
        return self.scenario.validate_results(results)
