"""Explicit modeler and custodian commands for a frozen benchmark."""

import argparse
import json
import sys

from .contracts import read_json
from . import protocol


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    seal = sub.add_parser("seal", help="Custodian: seal a reference outside the repo")
    seal.add_argument("--reference", required=True)
    seal.add_argument("--vault", required=True)
    freeze = sub.add_parser("freeze", help="Custodian: freeze inputs and identities")
    freeze.add_argument("--case", required=True)
    freeze.add_argument("--session", required=True)
    freeze.add_argument("--vault", required=True)
    run = sub.add_parser("run", help="Custodian supervises an isolated SDK child")
    run.add_argument("--session", required=True)
    run.add_argument("--vault", required=True)
    run.add_argument("--allow-unisolated-development", action="store_true")
    for command in ("commit", "reveal", "score", "verify"):
        item = sub.add_parser(command, help=f"Custodian: {command} evidence")
        item.add_argument("--session", required=True)
        item.add_argument("--vault", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "seal":
            result = protocol.seal_reference(read_json(args.reference), args.vault)
        elif args.command == "freeze":
            result = protocol.freeze_case(
                read_json(args.case), args.session, args.vault)
        elif args.command == "run":
            result = protocol.execute_registered(
                args.session, args.vault,
                require_isolation=not args.allow_unisolated_development,
            )
            result = {"execution": result["execution"],
                      "completed_runs": len(result["runs"]),
                      "failed_runs": sum(r["status"] == "FAIL" for r in result["runs"])}
        else:
            function = {"commit": protocol.commit_submission,
                        "reveal": protocol.reveal, "score": protocol.score,
                        "verify": protocol.verify_session}[args.command]
            result = function(args.session, args.vault)
        print(json.dumps(result, indent=2, allow_nan=False))
        # Numerical disagreement is an evaluation result, not a broken CLI.
        # Operational failure and protocol errors are nonzero exits.
        if args.command == "run" and result["failed_runs"]:
            return 1
        return 0
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f"Benchmark refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
