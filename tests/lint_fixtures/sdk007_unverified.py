# lint-as: healthcare_sim_sdk/scenarios/example/run_unverified.py
import hydra
from omegaconf import DictConfig

from healthcare_sim_sdk.experiments.lifecycle import finalize_experiment
from healthcare_sim_sdk.experiments.runner import run_dir, runner_settings

STRICT = True


@hydra.main(version_base=None, config_path="configs", config_name="current")
def main(cfg: DictConfig) -> None:
    settings = runner_settings(cfg)
    finalize_experiment(run_dir(), settings, {"n": 1})
    finalize_experiment(run_dir(), settings, {"n": 1}, validation_checks=[],
                        require_validated=False)
    finalize_experiment(run_dir(), settings, {"n": 1}, validation_checks=[],
                        require_validated=STRICT)


if __name__ == "__main__":
    main()
