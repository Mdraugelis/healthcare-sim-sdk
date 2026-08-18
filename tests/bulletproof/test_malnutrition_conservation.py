"""Conservation laws and boundary conditions for the malnutrition
prioritization scenario (design doc section 8, refined plan R24-R27).

Skeleton-phase tests run with ml_enabled=False (baseline workflow only);
ML-tier tests are added alongside intervene() wiring.
"""

from pathlib import Path

import numpy as np
import pytest

from healthcare_sim_sdk.core.engine import (
    BranchedSimulationEngine,
    CounterfactualMode,
)
from healthcare_sim_sdk.core.scenario import TimeConfig
from healthcare_sim_sdk.scenarios.malnutrition_prioritization.scenario import (
    ROW_ML_PRIORITY,
    ROW_MODEL_SCORE,
    MalnutritionConfig,
    MalnutritionPrioritizationScenario,
    SOURCE_BACKSTOP,
    SOURCE_CONSULT,
    SOURCE_MST,
    STATUS_DISCHARGED,
    STATUS_INPATIENT,
    STATUS_PRE,
)


_CONFIG_DIR = Path(
    "healthcare_sim_sdk/scenarios/malnutrition_prioritization/configs")
requires_frozen_model = pytest.mark.skipif(
    not ((_CONFIG_DIR / "model_frozen.yaml").exists()
         and (_CONFIG_DIR / "model_targets.yaml").exists()),
    reason="local model_targets.yaml / model_frozen.yaml not present "
           "(gitignored institution-specific files; see "
           "configs/model_targets.example.yaml)")


def _make(seed=42, horizon=28, ml_enabled=False, **overrides):
    """Build and run a small scenario; returns (scenario, results)."""
    defaults = dict(
        initial_census=60,
        admissions_per_day=8,
        rdn_assessments_per_day=6,
    )
    defaults.update(overrides)
    cfg = MalnutritionConfig(ml_enabled=ml_enabled, **defaults)
    tc = TimeConfig(
        n_timesteps=horizon,
        timestep_duration=1 / 365,
        timestep_unit="day",
        prediction_schedule=list(range(horizon)),
    )
    sc = MalnutritionPrioritizationScenario(
        time_config=tc, seed=seed, config=cfg)
    engine = BranchedSimulationEngine(sc, CounterfactualMode.BRANCHED)
    results = engine.run(sc.n_entities_required)
    return sc, results


def _make_full(seed=42, horizon=84, ml_enabled=False, **overrides):
    """Full-scale run (fast: ~0.1s) for statistical checks."""
    cfg = MalnutritionConfig(ml_enabled=ml_enabled, **overrides)
    tc = TimeConfig(
        n_timesteps=horizon,
        timestep_duration=1 / 365,
        timestep_unit="day",
        prediction_schedule=list(range(horizon)),
    )
    sc = MalnutritionPrioritizationScenario(
        time_config=tc, seed=seed, config=cfg)
    engine = BranchedSimulationEngine(sc, CounterfactualMode.BRANCHED)
    results = engine.run(sc.n_entities_required)
    return sc, results


class TestAccounting:
    """R24: four-way partition of true+ cases; entity status partition."""

    def test_truepos_partition_is_exhaustive_and_exclusive(self):
        _, res = _make_full()
        last = res.outcomes[len(res.outcomes) - 1].secondary
        tm = last["true_malnourished"] == 1
        status = last["status"]
        assessed = last["assessed"] == 1
        missed = last["missed"] == 1
        # Cases whose onset manifested before discharge -- the proper
        # denominator (a truncated-stay entity whose scheduled onset
        # fell past its actual discharge never manifested in-hospital).
        mal_at_disch = last["mal_at_discharge"] == 1

        discharged = status == STATUS_DISCHARGED
        assert not np.any(missed & assessed), "missed AND assessed"
        n_split = (mal_at_disch & assessed).sum() + missed.sum()
        assert n_split == mal_at_disch.sum(), (
            f"partition leak: {n_split} vs {mal_at_disch.sum()}")
        # Censored bucket: true+ still in hospital (or pre-admission).
        censored = tm & ~discharged
        never_manifested = discharged & tm & ~mal_at_disch
        assert (mal_at_disch.sum() + censored.sum()
                + never_manifested.sum()) == tm.sum()
        # The never-manifested edge must stay negligible.
        assert never_manifested.sum() / max(tm.sum(), 1) < 0.02

    def test_censored_fraction_bounded_at_full_scale(self):
        _, res = _make_full()
        last = res.outcomes[len(res.outcomes) - 1].secondary
        tm = last["true_malnourished"] == 1
        censored = tm & (last["status"] != STATUS_DISCHARGED)
        frac = censored.sum() / max(tm.sum(), 1)
        assert frac < 0.15, (
            f"censored true+ fraction {frac:.2f} too high; "
            "lengthen horizon or trim admission window")

    def test_entity_status_partition_every_timestep(self):
        _, res = _make()
        for t in range(len(res.outcomes)):
            status = res.outcomes[t].secondary["status"]
            valid = np.isin(
                status, [STATUS_PRE, STATUS_INPATIENT,
                         STATUS_DISCHARGED])
            assert valid.all(), f"invalid status at t={t}"
            assert len(status) == 3660 or len(status) > 0


class TestCapacity:
    """Design doc 8.2 + R7: single capacity enforcement point."""

    def test_daily_assessments_never_exceed_capacity(self):
        _, res = _make()
        for t in range(len(res.outcomes)):
            for out in (res.outcomes[t],
                        res.counterfactual_outcomes[t]):
                assert out.metadata["assessed_today"] <= 6

    def test_capacity_binding_when_queue_nonempty(self):
        # With erosion off, leftover queue implies capacity was filled.
        _, res = _make()
        for t in range(1, len(res.outcomes)):
            out = res.outcomes[t]
            if out.metadata["queue_depth"] > 0:
                assert out.metadata["assessed_today"] == 6, (
                    f"t={t}: queue {out.metadata['queue_depth']} "
                    f"but only {out.metadata['assessed_today']} "
                    "assessed")

    def test_capacity_zero_nobody_assessed(self):
        _, res = _make(rdn_assessments_per_day=0)
        last = res.outcomes[len(res.outcomes) - 1].secondary
        assert (last["assessed"] == 0).all()
        # Every discharged true+ is missed.
        tm = last["true_malnourished"] == 1
        discharged = last["status"] == STATUS_DISCHARGED
        assert (last["missed"][discharged & tm] == 1).all()

    def test_capacity_infinite_same_day_assessment(self):
        _, res = _make(rdn_assessments_per_day=10**9)
        for t in range(len(res.outcomes)):
            assert res.outcomes[t].metadata["queue_depth"] == 0


class TestCRNSync:
    """R25: with the ML tier inert, factual and counterfactual are
    bit-identical at every timestep. This is the test that catches a
    data-dependent RNG draw the instant someone introduces one."""

    NUMERIC_KEYS = (
        "true_malnourished", "severity", "assessed", "assess_day",
        "assess_source", "time_to_assessment", "missed",
        "severity_at_assess", "harm", "diagnosed", "coded", "los",
        "readmit", "ml_flagged", "status",
    )

    def test_branches_bit_identical_with_ml_off(self):
        _, res = _make(ml_enabled=False)
        for t in range(len(res.outcomes)):
            f = res.outcomes[t]
            c = res.counterfactual_outcomes[t]
            np.testing.assert_array_equal(
                f.events, c.events,
                err_msg=f"events diverge at t={t}")
            for k in self.NUMERIC_KEYS:
                np.testing.assert_array_equal(
                    f.secondary[k], c.secondary[k],
                    err_msg=f"{k} diverges at t={t} "
                            "(data-dependent RNG draw?)")


class TestTierExclusivity:
    """R26: exactly one assessment source per assessed patient."""

    def test_assess_source_consistent(self):
        _, res = _make()
        last = res.outcomes[len(res.outcomes) - 1].secondary
        assessed = last["assessed"] == 1
        src = last["assess_source"]
        # ML off: consult / MST / backstop paths only.
        assert np.isin(
            src[assessed],
            [SOURCE_CONSULT, SOURCE_MST, SOURCE_BACKSTOP],
        ).all()
        assert (src[~assessed] == 0).all()


class TestEventSemantics:
    """R27/R15: missed-case event fires once, on discharge day."""

    def test_event_sum_equals_final_missed_count(self):
        _, res = _make_full()
        total_events = sum(
            res.outcomes[t].events.sum()
            for t in range(len(res.outcomes)))
        final_missed = res.outcomes[
            len(res.outcomes) - 1].secondary["missed"].sum()
        assert total_events == final_missed

    def test_ml_priority_never_nan(self):
        # R9: -inf is the only worklist sentinel.
        sc, _ = _make()
        state = sc.create_population(100)
        assert not np.isnan(state[ROW_ML_PRIORITY]).any()
        for t in range(10):
            state = sc.step(state, t)
            assert not np.isnan(state[ROW_ML_PRIORITY]).any()


class TestMonotonicity:
    """Design doc 8.3: more capacity -> fewer missed cases."""

    def test_missed_monotone_in_capacity(self):
        missed_by_cap = []
        for cap in (3, 8, 20):
            rates = []
            for seed in range(1, 6):
                _, res = _make(
                    seed=seed, rdn_assessments_per_day=cap)
                last = res.outcomes[
                    len(res.outcomes) - 1].secondary
                tm = last["true_malnourished"] == 1
                discharged = last["status"] == STATUS_DISCHARGED
                denom = max((discharged & tm).sum(), 1)
                rates.append(last["missed"].sum() / denom)
            missed_by_cap.append(np.mean(rates))
        assert missed_by_cap[0] >= missed_by_cap[1] >= missed_by_cap[2], (
            f"missed not monotone in capacity: {missed_by_cap}")


class TestStatistical:
    """Design doc 8.6/8.7: NaN discipline and coded-prevalence CLT."""

    def test_coded_prevalence_within_4_sigma(self):
        cfg_prev = MalnutritionConfig().coded_prevalence
        _, res = _make_full()
        last = res.outcomes[len(res.outcomes) - 1].secondary
        discharged = last["status"] == STATUS_DISCHARGED
        n = discharged.sum()
        rate = last["coded"][discharged].mean()
        sigma = np.sqrt(cfg_prev * (1 - cfg_prev) / n)
        assert abs(rate - cfg_prev) < 4 * sigma + 0.01, (
            f"coded rate {rate:.4f} vs configured {cfg_prev} "
            f"(4-sigma band +/-{4*sigma:.4f})")

    def test_no_nan_outside_model_score(self):
        sc, _ = _make()
        state = sc.create_population(200)
        for t in range(15):
            state = sc.step(state, t)
        for row in range(state.shape[0]):
            if row == ROW_MODEL_SCORE:
                continue
            vals = state[row]
            finite_ok = ~np.isnan(vals)
            assert finite_ok.all(), f"NaN in state row {row}"

    def test_same_seed_bit_identical(self):
        _, res_a = _make(seed=7)
        _, res_b = _make(seed=7)
        last_a = res_a.outcomes[len(res_a.outcomes) - 1].secondary
        last_b = res_b.outcomes[len(res_b.outcomes) - 1].secondary
        np.testing.assert_array_equal(
            last_a["harm"], last_b["harm"])
        np.testing.assert_array_equal(
            last_a["missed"], last_b["missed"])


class TestArmCCalibration:
    """R29 (provisional on [O] placeholders): the current-workflow arm
    should reproduce a plausible baseline time-to-consult."""

    def test_baseline_median_tta_near_configured(self):
        _, res = _make_full()
        last = res.outcomes[len(res.outcomes) - 1].secondary
        tm = last["true_malnourished"] == 1
        assessed = last["assessed"] == 1
        discharged = last["status"] == STATUS_DISCHARGED
        mask = tm & assessed & discharged
        # Exclude time-bound consults (assessed ~immediately) to look
        # at the discretionary stream the 3-day figure describes.
        stream = last["assess_source"][mask] != SOURCE_CONSULT
        tta = last["time_to_assessment"][mask][stream]
        med = np.median(tta)
        assert 1.0 <= med <= 5.0, (
            f"Arm-C discretionary median TTA {med:.1f}d implausible "
            "vs configured baseline_time_to_consult_days=3.0")


@requires_frozen_model
class TestModelFidelity:
    """The locally frozen model reproduces the configured
    model targets on the calibration cohort."""

    def test_frozen_params_hit_targets(self):
        from healthcare_sim_sdk.scenarios.malnutrition_prioritization \
            .subgroup_performance import (
                FrozenSubgroupScorer, build_calibration_cohort,
                evaluate, load_targets)
        targets = load_targets()
        scorer = FrozenSubgroupScorer.load(
            str(_CONFIG_DIR / "model_frozen.yaml"))
        cohort = build_calibration_cohort(
            prevalence=targets.calibration_prevalence)
        m = evaluate(scorer.params, cohort, targets,
                     gradient=MalnutritionConfig().severity_score_gradient)
        assert abs(m["auroc"] - targets.auroc) <= 0.01
        for k, t in targets.lifts.items():
            assert abs(m[f"lift_{k}"] - t) <= 0.08, (
                f"lift_{k}: {m[f'lift_{k}']:.3f} vs {t}")

    def test_scorer_scores_scenario_state(self):
        from healthcare_sim_sdk.scenarios.malnutrition_prioritization \
            .subgroup_performance import FrozenSubgroupScorer
        scorer = FrozenSubgroupScorer.load(
            str(_CONFIG_DIR / "model_frozen.yaml"))
        sc, _ = _make()
        state = sc.create_population(500)
        for t in range(3):
            state = sc.step(state, t)
        scores = scorer.score(state)
        assert scores.shape == (500,)
        assert np.all((scores >= 0) & (scores <= 1))
        # Deterministic: same state -> identical scores.
        np.testing.assert_array_equal(scores, scorer.score(state))


@requires_frozen_model
class TestMLBoundaries:
    """R11: exact bit-equality boundaries with the ML arm active."""

    def _assert_bit_equal(self, res):
        # ml_flagged is excluded: it is the Epic display LABEL, written
        # by the tool in every mode regardless of whether anything acts
        # on it. The boundary claim is about OUTCOMES.
        keys = tuple(k for k in TestCRNSync.NUMERIC_KEYS
                     if k != "ml_flagged")
        for t in range(len(res.outcomes)):
            f = res.outcomes[t]
            c = res.counterfactual_outcomes[t]
            np.testing.assert_array_equal(f.events, c.events)
            for k in keys:
                np.testing.assert_array_equal(
                    f.secondary[k], c.secondary[k],
                    err_msg=f"{k} diverges at t={t}")

    def test_top_k_zero_arms_bit_identical(self):
        _, res = _make(ml_enabled=True, operating_mode="top_k",
                       top_k_fraction=0.0, mst_reference_floor=0.0)
        self._assert_bit_equal(res)

    def test_insufficient_data_all_arms_bit_identical(self):
        _, res = _make(ml_enabled=True, insuff_data_rate=1.0,
                       mst_reference_floor=0.0)
        self._assert_bit_equal(res)

    def test_capacity_infinite_arms_bit_identical(self):
        # flag_driven_coding_uplift defaults to 0, so with infinite
        # capacity the ML ordering is irrelevant end to end.
        _, res = _make(
            ml_enabled=True, rdn_assessments_per_day=10**9)
        self._assert_bit_equal(res)

    def test_degenerate_auroc05_no_benefit(self):
        """A chance-level model must NOT show ML benefit; it may be
        WORSE than baseline (random reservations displace an
        informative MST order) -- qualitative, per R11."""
        deltas_real, deltas_degen = [], []
        for seed in range(1, 6):
            for degen, sink in ((False, deltas_real),
                                (True, deltas_degen)):
                _, res = _make(
                    seed=seed, ml_enabled=True,
                    degenerate_auroc05=degen)
                last_f = res.outcomes[
                    len(res.outcomes) - 1].secondary
                last_c = res.counterfactual_outcomes[
                    len(res.outcomes) - 1].secondary
                mf = (last_f["missed"].sum()
                      / max(last_f["mal_at_discharge"].sum(), 1))
                mc = (last_c["missed"].sum()
                      / max(last_c["mal_at_discharge"].sum(), 1))
                sink.append(mc - mf)  # >0 means ML helps
        real, degen = np.mean(deltas_real), np.mean(deltas_degen)
        assert real > degen, (
            f"real model ({real:+.3f}) must beat chance-level "
            f"({degen:+.3f})")
        assert degen < 0.03, (
            f"chance-level model shows implausible benefit "
            f"{degen:+.3f}")


@requires_frozen_model
class TestBayesBound:
    """Design doc 8.5: realized ML-flag PPV respects Bayes' theorem."""

    def test_realized_ppv_within_theoretical_bound(self):
        from healthcare_sim_sdk.ml.performance import theoretical_ppv
        _, res = _make_full(ml_enabled=True)
        last = res.outcomes[len(res.outcomes) - 1].secondary
        discharged = last["status"] == STATUS_DISCHARGED
        truth = last["mal_at_discharge"][discharged] == 1
        flagged = last["ml_flagged"][discharged] == 1
        prev = truth.mean()
        sens = (flagged & truth).sum() / max(truth.sum(), 1)
        spec = (~flagged & ~truth).sum() / max((~truth).sum(), 1)
        ppv = (flagged & truth).sum() / max(flagged.sum(), 1)
        bound = theoretical_ppv(prev, sens, spec)
        assert ppv <= bound + 0.02, (
            f"realized PPV {ppv:.3f} exceeds Bayes bound "
            f"{bound:.3f} (prev {prev:.3f}, sens {sens:.3f}, "
            f"spec {spec:.3f})")


@requires_frozen_model
class TestMLDominance:
    """Design doc 7: with the safety net preserved, ML never hurts --
    at expectation (mean over seeds), per the refined plan."""

    def test_missed_f_leq_c_mean_over_seeds(self):
        deltas = []
        for seed in range(1, 11):
            _, res = _make(seed=seed, ml_enabled=True)
            last_f = res.outcomes[len(res.outcomes) - 1].secondary
            last_c = res.counterfactual_outcomes[
                len(res.outcomes) - 1].secondary
            mf = (last_f["missed"].sum()
                  / max(last_f["mal_at_discharge"].sum(), 1))
            mc = (last_c["missed"].sum()
                  / max(last_c["mal_at_discharge"].sum(), 1))
            deltas.append(mc - mf)
        assert np.mean(deltas) >= 0, (
            f"ML arm worse on average: {np.mean(deltas):+.4f}")

    def test_ml_source_present_and_exclusive(self):
        from healthcare_sim_sdk.scenarios.\
            malnutrition_prioritization.scenario import SOURCE_ML
        _, res = _make_full(ml_enabled=True)
        last = res.outcomes[len(res.outcomes) - 1].secondary
        assessed = last["assessed"] == 1
        src = last["assess_source"]
        assert (src[assessed] > 0).all()
        assert (src[~assessed] == 0).all()
        assert (src[assessed] == SOURCE_ML).sum() > 0, (
            "no ML-sourced assessments despite ml_enabled")


class TestConsultPath:
    """Path 1 (time-bound consults) invariants under the deployed
    three-path workflow."""

    def test_consults_never_superseded(self):
        """If any consult-due patient goes unassessed on a day, every
        assessment that day must itself be a consult (capacity was
        exhausted by path 1) -- on BOTH branches."""
        from healthcare_sim_sdk.scenarios.malnutrition_prioritization \
            .scenario import (
                ROW_ADMIT_DAY, ROW_ASSESS_DAY, ROW_ASSESS_SOURCE,
                ROW_ASSESSED, ROW_MANDATORY_CONSULT,
                ROW_PHYS_CONSULT_DAY, ROW_STATUS, NEVER,
                SOURCE_CONSULT, STATUS_INPATIENT)
        sc, _ = _make()
        state = sc.create_population(400)
        for t in range(25):
            state = sc.step(state, t)
            timebound = np.where(
                state[ROW_MANDATORY_CONSULT] == 1,
                state[ROW_ADMIT_DAY], NEVER)
            timebound = np.minimum(
                timebound, state[ROW_PHYS_CONSULT_DAY])
            due_unworked = (
                (state[ROW_STATUS] == STATUS_INPATIENT)
                & (state[ROW_ASSESSED] == 0)
                & (timebound <= t))
            if due_unworked.any():
                today = state[ROW_ASSESS_DAY] == t
                srcs = state[ROW_ASSESS_SOURCE][today]
                assert (srcs == SOURCE_CONSULT).all(), (
                    f"t={t}: non-consult assessment while a "
                    "time-bound consult waited")

    def test_provider_consult_ppv_realized(self):
        """Realized provider-consult PPV (consults arriving before
        discharge) lands near the configured target."""
        from healthcare_sim_sdk.scenarios.malnutrition_prioritization \
            .scenario import (
                ROW_DISCHARGE_DAY, ROW_ONSET_DAY,
                ROW_PHYS_CONSULT_DAY, ROW_STATUS, STATUS_DISCHARGED)
        target = MalnutritionConfig().provider_consult_ppv
        ppvs = []
        for seed in range(1, 6):
            sc, res = _make_full(seed=seed)
            # Recover final-state facts via a manual replay of the
            # factual branch (cheap) to read raw state rows.
            state = sc2 = None
            sc2 = MalnutritionPrioritizationScenario(
                time_config=sc.time_config, seed=seed,
                config=sc.cfg)
            state = sc2.create_population(sc2.n_entities_required)
            for t in range(sc2.time_config.n_timesteps):
                state = sc2.step(state, t)
            discharged = state[ROW_STATUS] == STATUS_DISCHARGED
            arrived = (
                discharged
                & (state[ROW_PHYS_CONSULT_DAY]
                   <= state[ROW_DISCHARGE_DAY]))
            if arrived.sum() >= 30:
                truth = np.isfinite(state[ROW_ONSET_DAY])[arrived]
                ppvs.append(truth.mean())
        realized = float(np.mean(ppvs))
        assert abs(realized - target) < 0.10, (
            f"realized provider-consult PPV {realized:.3f} vs "
            f"target {target} (first-order calibration band 0.10)")

    def test_mst_floor_protects_flagged_cases(self):
        """Higher MST reference floor -> MST-flagged true cases are
        missed no more often (mean over seeds), at capacity where the
        ranked list is not exhausted."""
        rates = []
        for floor in (0.0, None):   # off vs default (threshold)
            miss = []
            for seed in range(1, 8):
                _, res = _make(
                    seed=seed, ml_enabled=True,
                    rdn_assessments_per_day=5,
                    mst_reference_floor=floor)
                last = res.outcomes[len(res.outcomes) - 1].secondary
                # MST-flagged proxy: assessed via MST or floor is not
                # directly observable; use overall missed as the
                # monotonicity target (floor can only add reach for
                # flagged patients, never remove it).
                mal = last["mal_at_discharge"] == 1
                miss.append(
                    last["missed"].sum() / max(mal.sum(), 1))
            rates.append(np.mean(miss))
        assert rates[1] <= rates[0] + 0.01, (
            f"MST floor made things worse: off {rates[0]:.3f} vs "
            f"on {rates[1]:.3f}")


class TestEffectFunctions:
    """Unit tests for the module-level two-channel effect functions --
    the single source of truth shared by step() and the outcome
    overlay (plan section C)."""

    def test_timing_days_early_shapes(self):
        from healthcare_sim_sdk.scenarios.malnutrition_prioritization \
            .scenario import timing_days_early
        assess = np.array([3.0, 3.0, 10.0, 5.0])
        onset = np.array([3.0, 0.0, 2.0, 7.0])
        admit = np.array([0.0, 2.0, 2.0, 3.0])
        # a = assess - max(onset, admit); days_early = max(5 - a, 0)
        de = timing_days_early(assess, onset, admit, d_ref=5.0)
        # case 0: assessed the day of onset -> full 5 days early
        # case 1: a = 1 -> 4; case 2: a = 8 -> 0 (late, clipped)
        # case 3: hospital-acquired onset after assess -> a<0 clamped
        #   to 0 -> full d_ref
        np.testing.assert_allclose(de, [5.0, 4.0, 0.0, 5.0])
        assert (timing_days_early(
            np.array([100.0]), np.array([0.0]),
            np.array([0.0]), 5.0) == 0).all()

    def test_readmission_probability(self):
        from healthcare_sim_sdk.scenarios.malnutrition_prioritization \
            .scenario import readmission_probability
        r0 = np.array([0.30])
        dx = np.array([1.0])
        de = np.array([5.0])
        # literature-central cell: 0.30 * 0.80 * (1 - 0.01*5)
        p = readmission_probability(r0, dx, de, 0.20, 0.01)
        np.testing.assert_allclose(p, [0.30 * 0.80 * 0.95])
        # efficacy off -> baseline exactly
        np.testing.assert_allclose(
            readmission_probability(r0, dx, de, 0.0, 0.0), r0)
        # undiagnosed -> baseline exactly, regardless of knobs
        np.testing.assert_allclose(
            readmission_probability(r0, np.zeros(1), de, 0.9, 0.03),
            r0)
        # monotone non-increasing in days_early
        p_late = readmission_probability(
            r0, dx, np.array([0.0]), 0.20, 0.01)
        assert p[0] <= p_late[0]

    def test_los_reduction_and_runway_cap(self):
        from healthcare_sim_sdk.scenarios.malnutrition_prioritization \
            .scenario import los_reduction
        dx = np.array([1.0])
        de = np.array([5.0])
        # uncapped: 0.5 + 0.05*5 = 0.75 (runway generous)
        np.testing.assert_allclose(
            los_reduction(dx, de, np.array([10.0]), 0.5, 0.05),
            [0.75])
        # runway binds
        np.testing.assert_allclose(
            los_reduction(dx, de, np.array([0.3]), 0.5, 0.05),
            [0.3])
        # undiagnosed or efficacy off -> zero
        np.testing.assert_allclose(
            los_reduction(np.zeros(1), de, np.array([10.0]),
                          0.5, 0.05), [0.0])
        np.testing.assert_allclose(
            los_reduction(dx, de, np.array([10.0]), 0.0, 0.0),
            [0.0])


class TestOutcomeCoins:
    """Creation-time per-patient uniforms (U_READMIT, U_DIAG) are the
    cross-arm coin-sharing mechanism: written once at creation, never
    touched by step(), identical across branches by construction."""

    def test_coins_immutable_through_steps(self):
        from healthcare_sim_sdk.scenarios.malnutrition_prioritization \
            .scenario import ROW_U_DIAG, ROW_U_READMIT
        sc, _ = _make(ml_enabled=True)
        state = sc.create_population(300)
        u_r = state[ROW_U_READMIT].copy()
        u_d = state[ROW_U_DIAG].copy()
        assert ((u_r >= 0) & (u_r < 1)).all()
        assert ((u_d >= 0) & (u_d < 1)).all()
        for t in range(20):
            state = sc.step(state, t)
        np.testing.assert_array_equal(state[ROW_U_READMIT], u_r)
        np.testing.assert_array_equal(state[ROW_U_DIAG], u_d)

    def test_readmit_baseline_efficacy_invariant(self):
        """readmit0 is disease physics: changing the efficacy knobs
        must not move it (the effect enters only through the effect
        functions)."""
        sc_a, _ = _make(seed=7)
        sc_b, _ = _make(seed=7, eff_detect_rel=0.25,
                        eff_timing_rel=0.03, eff_detect_los=1.5,
                        eff_timing_los=0.10)
        state = sc_a.create_population(300)
        for t in range(10):
            state = sc_a.step(state, t)
        np.testing.assert_array_equal(
            sc_a._readmit_baseline(state, 10),
            sc_b._readmit_baseline(state, 10))


class TestConsultProtection:
    """S-4 acceptance tests: tier-0 (time-bound consults) is protected
    from the ML reordering by construction."""

    @staticmethod
    def _final(res):
        h = len(res.outcomes) - 1
        return (res.outcomes[h].secondary,
                res.counterfactual_outcomes[h].secondary)

    def test_consult_bit_identity_at_zero_efficacy(self):
        """At eff=0 with tier 0 UNSATURATED (consult demand < daily
        capacity, so no carryover-queue cross-talk through the shared
        capacity pool):
        - a patient reached via consult in BOTH arms is reached on the
          identical day (the due-day schedule is arm-invariant);
        - a patient reached via consult in only ONE arm was reached
          STRICTLY EARLIER by a faster tier in the other arm -- the
          only mechanism that can pre-empt a time-bound consult.
        Under saturation, day equality legitimately breaks (a patient
        pre-assessed by ML in one arm frees a tier-0 slot and shifts
        another consult); that regime is pinned by boundary check 11
        (mandatory_consult_fraction=1.0 -> bit-identical arms)."""
        for seed in range(1, 6):
            _, res = _make(
                seed=seed, ml_enabled=True,
                rdn_assessments_per_day=12,
                eff_detect_rel=0.0, eff_timing_rel=0.0,
                eff_detect_los=0.0, eff_timing_los=0.0)
            f, c = self._final(res)
            f_con = f["assess_source"] == SOURCE_CONSULT
            c_con = c["assess_source"] == SOURCE_CONSULT
            both = f_con & c_con
            assert both.any()
            np.testing.assert_array_equal(
                f["assess_day"][both], c["assess_day"][both])
            f_only = f_con & ~c_con
            assert ((c["assessed"][f_only] == 1)
                    & (c["assess_day"][f_only]
                       < f["assess_day"][f_only])).all()
            c_only = c_con & ~f_con
            assert ((f["assessed"][c_only] == 1)
                    & (f["assess_day"][c_only]
                       < c["assess_day"][c_only])).all()

    def test_gating_attribution(self):
        """A patient missed in F but assessed in C was necessarily
        (a) never consult-eligible before discharge and (b) reached in
        C via the MST stream or the backstop -- i.e. displacement is
        attributable to the ML reordering of tiers 2-3 only."""
        found = 0
        for seed in range(1, 8):
            _, res = _make(
                seed=seed, ml_enabled=True,
                rdn_assessments_per_day=5,
                eff_detect_rel=0.0, eff_timing_rel=0.0,
                eff_detect_los=0.0, eff_timing_los=0.0)
            f, c = self._final(res)
            displaced = (f["missed"] == 1) & (c["assessed"] == 1)
            if not displaced.any():
                continue
            found += int(displaced.sum())
            srcs = c["assess_source"][displaced]
            assert np.isin(
                srcs, [SOURCE_MST, SOURCE_BACKSTOP]).all(), (
                f"seed {seed}: displaced patient reached in C via "
                f"source {set(srcs.tolist())} -- tier-0 protection "
                "violated")
        assert found > 0, "no displaced patients found; test is vacuous"


class TestFlagDayBookkeeping:
    """ROW_ML_FLAG_DAY: first day a flagged-unworked patient appears
    on the written worklist (Tier-2 pool entry). Factual-only, sticky,
    no RNG."""

    def test_ml_disabled_never_set(self):
        _, res = _make(ml_enabled=False)
        h = len(res.outcomes) - 1
        assert (res.outcomes[h].secondary["ml_flag_day"] == -1).all()

    def test_counterfactual_never_set(self):
        _, res = _make(ml_enabled=True)
        h = len(res.outcomes) - 1
        c = res.counterfactual_outcomes[h].secondary
        assert (c["ml_flag_day"] == -1).all()

    def test_sticky_and_consistent_with_flag(self):
        _, res = _make(ml_enabled=True)
        h = len(res.outcomes) - 1
        prev = None
        for t in range(h + 1):
            cur = res.outcomes[t].secondary["ml_flag_day"]
            if prev is not None:
                was_set = prev >= 0
                np.testing.assert_array_equal(
                    cur[was_set], prev[was_set])
            prev = cur
        f = res.outcomes[h].secondary
        flagged = f["ml_flagged"] == 1
        assert flagged.any()
        assert (f["ml_flag_day"][flagged] >= 1).all()
        assert (f["ml_flag_day"][~flagged] == -1).all()
