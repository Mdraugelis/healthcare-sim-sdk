# lint-as: healthcare_sim_sdk/scenarios/example/runner.py
from healthcare_sim_sdk.experiments.lifecycle import finalize_experiment


def save(out, config, metrics):
    finalize_experiment(out, config, metrics)


def narrow():
    try:
        return int("x")
    except ValueError:
        return None


def reraise():
    try:
        return 1 / 0
    except Exception as exc:
        raise RuntimeError("wrapped") from exc


def recorded(errors):
    try:
        return 1 / 0
    except Exception as exc:  # sdk-lint: allow SDK003 -- appended to errors and reported
        errors.append(exc)
