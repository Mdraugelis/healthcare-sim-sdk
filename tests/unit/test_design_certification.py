"""Planted truth, observation selection and honest MC certification gates."""
from dataclasses import replace
import json

import numpy as np
import pytest

from healthcare_sim_sdk.experiments.analysis.certification import (
    DesignCertificationConfig, ITSDesign, generate_design_data,
    run_design_certification, write_design_evidence,
)
from healthcare_sim_sdk.experiments.analysis.its import segmented_regression


def test_planted_truth_matches_hand_worked_segmented_series():
    # Independent golden design matrix: b0=3, b1=.2, b2=-2, b3=0.
    times = np.arange(16)
    observed = 3 + .2 * times - 2 * (times >= 8)
    fitted = segmented_regression(observed, 8)
    assert fitted.level_change == pytest.approx(-2)
    assert fitted.pre_slope == pytest.approx(.2)
    assert fitted.slope_change == pytest.approx(0, abs=1e-12)


def test_generator_plants_exact_mean_shift_even_when_observation_is_selected():
    design = ITSDesign('selected', n_pre=8, n_post=8,
                       informative_censoring=.5, denominator_variation=.5,
                       autocorrelation=.6)
    null = generate_design_data(design, 0, seed=10)
    effect = generate_design_data(design, -.7, seed=10)
    expected = np.r_[np.zeros(8), np.full(8, -.7)]
    np.testing.assert_allclose(effect.population_mean - null.population_mean, expected)
    np.testing.assert_allclose(effect.observations - null.observations, expected)
    assert effect.true_level_change == -.7
    assert np.ptp(effect.denominators) > 0
    np.testing.assert_array_equal(effect.observed_denominators[:8], effect.denominators[:8])
    assert (effect.observed_denominators[8:] < effect.denominators[8:]).all()


def test_informative_censoring_fails_despite_apparently_high_power():
    result = run_design_certification(DesignCertificationConfig(
        designs=(ITSDesign('adverse', n_pre=24, n_post=24,
                           informative_censoring=.7, temporal_sd=.01,
                           individual_sd=.5),), repetitions=50))
    cell = result['cells'][0]
    assert cell['status'] == result['status'] == 'FAIL'
    assert cell['arms']['null']['criteria']['type_i_error'] == 'FAIL'
    assert cell['arms']['effect']['criteria']['absolute_bias'] == 'FAIL'
    assert cell['arms']['effect']['power']['rate'] > .95
    assert cell['arms']['null']['truth'] == 0
    assert all(r['truth'] == -.5 for r in cell['arms']['effect']['repetitions'])


def test_small_repetition_count_cannot_be_certified_from_point_estimates(tmp_path):
    config = DesignCertificationConfig(designs=(ITSDesign('iid'),), repetitions=2)
    result = run_design_certification(config)
    assert result['status'] != 'PASS'
    null = result['cells'][0]['arms']['null']
    assert null['type_i_error']['ci95_high'] > config.max_type_i
    paths = write_design_evidence(result, tmp_path)
    encoded = json.loads(open(paths['json']).read())
    assert encoded['status'] == result['status']
    assert encoded['unsupported'][0]['status'] == 'NOT_RUN'
    assert 'INCONCLUSIVE' in open(paths['markdown']).read()
    assert run_design_certification(config) == result


def test_failed_fits_remain_in_manifest_not_selected_out(monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError('injected fitting failure')
    monkeypatch.setattr(
        'healthcare_sim_sdk.experiments.analysis.certification.segmented_regression',
        broken)
    result = run_design_certification(DesignCertificationConfig(
        designs=(ITSDesign('broken'),), repetitions=3))
    assert result['status'] == 'FAIL'
    for arm in result['cells'][0]['arms'].values():
        assert arm['completed_repetitions'] == 0
        assert len(arm['failures']) == arm['expected_repetitions'] == 3
        assert 'coverage' not in arm
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('change', [
    {'n_pre': 3}, {'autocorrelation': 1}, {'denominator_variation': 1},
    {'informative_censoring': 1}, {'temporal_sd': float('nan')},
])
def test_invalid_design_rejected(change):
    with pytest.raises(ValueError):
        run_design_certification(DesignCertificationConfig(
            designs=(replace(ITSDesign('invalid'), **change),), repetitions=2))


def test_well_behaved_design_can_meet_explicit_precision_aware_requirements():
    config = DesignCertificationConfig(
        designs=(ITSDesign('iid_long', n_pre=48, n_post=36),),
        repetitions=100, max_type_i=.15, min_coverage=.85, max_abs_bias=.15)
    result = run_design_certification(config)
    assert result['status'] == 'PASS'
    cell = result['cells'][0]
    assert abs(cell['arms']['effect']['bias']['mean']) < .05
    assert cell['arms']['null']['type_i_error']['ci95_high'] <= .15
    assert cell['arms']['effect']['coverage']['ci95_low'] >= .85
