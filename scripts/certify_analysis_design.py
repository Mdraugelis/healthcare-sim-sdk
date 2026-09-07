#!/usr/bin/env python3
"""Run bounded synthetic ITS design evidence, including an adverse demo.

Usage: .venv/bin/python scripts/certify_analysis_design.py --repetitions 100
       .venv/bin/python scripts/certify_analysis_design.py --config design.json

Exit 0=PASS, 1=FAIL, 2=INCONCLUSIVE. The default demo intentionally includes
adverse designs and is expected to report FAIL. JSON config fields match
DesignCertificationConfig, with a list of ITSDesign objects under 'designs'.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from healthcare_sim_sdk.experiments.analysis.certification import (  # noqa: E402
    DesignCertificationConfig, ITSDesign, run_design_certification,
    write_design_evidence,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--repetitions', type=int, default=100)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'reports' / 'analysis_design_demo')
    args = parser.parse_args()
    if args.config:
        values = json.loads(args.config.read_text())
        designs = tuple(ITSDesign(**design) for design in values.pop('designs'))
        config = DesignCertificationConfig(designs=designs, **values)
    else:
        config = DesignCertificationConfig(
            repetitions=args.repetitions,
            designs=(
                ITSDesign('iid_long', n_pre=48, n_post=36),
                ITSDesign('short_pre_followup', n_pre=8, n_post=8),
                ITSDesign('autocorrelated_ols', autocorrelation=.8),
                ITSDesign('autocorrelated_hac', autocorrelation=.8, hac_maxlags=4),
                ITSDesign('varying_denominators', n_pre=48, denominator_variation=.75),
                ITSDesign('informative_censoring', informative_censoring=.5),
            ),
        )
    result = run_design_certification(config)
    paths = write_design_evidence(result, args.output)
    print(json.dumps({'status': result['status'], 'artifacts': paths}, indent=2))
    return {'PASS': 0, 'FAIL': 1, 'INCONCLUSIVE': 2}[result['status']]


if __name__ == '__main__':
    raise SystemExit(main())
