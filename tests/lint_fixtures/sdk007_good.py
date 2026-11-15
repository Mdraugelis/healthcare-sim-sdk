# lint-as: healthcare_sim_sdk/scenarios/example/run_current.py
import hydra
from omegaconf import DictConfig

from healthcare_sim_sdk.experiments.lifecycle import finalize_experiment
from healthcare_sim_sdk.experiments.runner import (
    check, run_dir, runner_settings,
)


@hydra.main(version_base=None, config_path="configs", config_name="current")
def main(cfg: DictConfig) -> None:
    settings = runner_settings(cfg)
    finalize_experiment(
        run_dir(), settings, {"n": settings["n"]},
        validation_checks=[check("n_positive", "n is positive",
                                 settings["n"] > 0)],
        require_validated=True)


if __name__ == "__main__":
    main()
