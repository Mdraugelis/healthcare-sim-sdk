# lint-as: healthcare_sim_sdk/scenarios/example/run_old_style.py
import argparse
import sys

from healthcare_sim_sdk.experiments.catalog import ExperimentCatalog
from healthcare_sim_sdk.experiments.lifecycle import save_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args(sys.argv[1:])
    save_experiment("outputs/x", {"n": args.n}, {})
    ExperimentCatalog().register("outputs/x", {"n": args.n}, {})


if __name__ == "__main__":
    main()
