"""Hand-worked monitoring event windows and censoring oracles."""

import pytest

from healthcare_sim_sdk.scenarios.nurse_retention.monitoring import (
    DetectionEvent, MonitoringClaim, MonitoringRun,
    evaluate_claim, evaluate_claims, summarize_claim_evaluations,
)


def _run(events=(), last_week=12, seed=1):
    return MonitoringRun("example", seed, last_week + 1,
                         weekly_history=[{"week": w} for w in range(last_week + 1)],
                         detection_events=list(events))


def _event(week, direction="down", metric="check_ins_done", tier=1):
    return DetectionEvent(tier, metric, week, "warning", 2, direction, "rule")


def _claim(**kwargs):
    return MonitoringClaim(**dict(claim_id="capacity_loss", tier=1,
                                  metric="check_ins_done", direction="down",
                                  change_week=8, horizon=12, **kwargs))


def test_pre_change_and_wrong_direction_do_not_count_as_detection():
    run = _run([_event(3), _event(8, "up"), _event(10), _event(9, metric="other")])
    actual = evaluate_claim(run, _claim())
    assert actual["pre_change_alert_count"] == 1
    assert actual["post_change_wrong_direction_alert_count"] == 1
    assert actual["first_post_change_detection_week"] == 10
    assert actual["detection_delay_weeks"] == 2
    assert actual["detected"] is True
    assert actual["right_censored"] is False


def test_nondetection_is_censored_and_delay_is_not_invented():
    actual = evaluate_claim(_run([_event(3)]), _claim())
    assert actual["detected"] is False
    assert actual["right_censored"] is True
    assert actual["censoring_week"] == 12
    assert actual["detection_delay_weeks"] is None
    assert actual["follow_up_weeks"] == 5


def test_ineligible_event_does_not_count_but_is_disclosed():
    actual = evaluate_claim(_run([_event(8)]), _claim(eligibility_start=10))
    assert actual["post_change_ineligible_alert_count"] == 1
    assert actual["detected"] is False


def test_incomplete_and_absent_followup_not_full_evaluations():
    result = evaluate_claim(_run(last_week=9), _claim())
    assert result["evaluation_status"] == "INCONCLUSIVE"
    assert evaluate_claim(_run(last_week=7), _claim())["evaluation_status"] == "NOT_RUN"
    missing = _run()
    missing.weekly_history.pop(9)
    assert evaluate_claim(missing, _claim())["missing_observation_weeks"] == [9]


def test_null_counts_full_horizon_and_both_directions():
    claim = MonitoringClaim("null", 1, "check_ins_done", "down", None, 12)
    actual = evaluate_claim(_run([_event(0), _event(12, "up"), _event(13)]), claim)
    assert actual["false_alert_count"] == 2
    assert actual["false_alert_any"] is True
    assert actual["detected"] is None


def test_summary_includes_exclusions_mc_uncertainty_and_censored_runs():
    claim = _claim()
    rows = [evaluate_claim(_run([_event(10)], seed=1), claim),
            evaluate_claim(_run(seed=2), claim),
            evaluate_claim(_run(last_week=2, seed=3), claim)]
    result = summarize_claim_evaluations(rows)
    assert result["estimated_probability"] == 0.5
    assert result["monte_carlo_standard_error"] == pytest.approx((0.25 / 2) ** 0.5)
    assert result["n_nondetections_censored"] == 1
    assert result["n_excluded_incomplete"] == 1
    assert result["mean_delay_weeks_conditional_on_detection"] == 2
    assert result["status"] == "INCONCLUSIVE"


def test_zero_false_alarms_still_has_nonzero_upper_bound():
    claim = MonitoringClaim("null", 1, "check_ins_done", "any", None, 12)
    result = summarize_claim_evaluations([
        evaluate_claim(_run(seed=seed), claim) for seed in range(30)
    ])
    assert result["estimated_probability"] == 0
    assert result["one_sided_95pct_exact_binomial_upper_bound"] == pytest.approx(
        1 - 0.05 ** (1 / 30)
    )


def test_duplicate_claims_and_incompatible_definitions_rejected():
    with pytest.raises(ValueError, match="unique"):
        evaluate_claims(_run(), [_claim(), _claim()])
    a = evaluate_claim(_run(), _claim())
    b = evaluate_claim(_run(), _claim(eligibility_start=10))
    with pytest.raises(ValueError, match="different"):
        summarize_claim_evaluations([a, b])


def test_repeated_seed_is_not_independent_validation():
    actual = evaluate_claim(_run(), _claim())
    with pytest.raises(ValueError, match="unique seeds"):
        summarize_claim_evaluations([actual, actual])


def test_null_before_monitor_is_active_cannot_certify_silence():
    claim = MonitoringClaim("null", 1, "check_ins_done", "any", None, 3,
                            eligibility_start=8)
    assert evaluate_claim(_run(last_week=3), claim)["evaluation_status"] == "NOT_RUN"
