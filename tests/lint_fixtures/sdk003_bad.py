# lint-as: healthcare_sim_sdk/scenarios/example/runner.py
import logging

from healthcare_sim_sdk.experiments.lifecycle import finalize_experiment

logger = logging.getLogger(__name__)


def save(out, config, metrics):
    try:
        finalize_experiment(out, config, metrics)
    except Exception as exc:          # SDK003 block: lifecycle failure hidden
        logger.warning("could not save: %s", exc)


def other():
    try:
        return 1 / 0
    except Exception:                 # SDK003 warn: broad, not re-raised
        return None
