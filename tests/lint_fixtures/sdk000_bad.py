# lint-as: healthcare_sim_sdk/scenarios/example/runner.py
def a():
    try:
        return 1 / 0
    except Exception:  # sdk-lint: allow SDK003
        return None


def b():
    return 1  # sdk-lint: allow SDK003 -- nothing to suppress here


def c():
    return 2  # sdk-lint: please ignore
