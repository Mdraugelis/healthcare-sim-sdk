"""Malnutrition risk prioritization scenario.

An externally validated ML model (frozen at its published operating
characteristics; see configs/model_targets.example.yaml) scores every
inpatient daily; RDNs work ONE consolidated list in strict priority
order (the deployed three-path workflow): (1) TIME-BOUND CONSULTS --
mandatory service-line AND provider-initiated -- a clinical obligation
worked first, never superseded by the model; (2) the ML RISK RANKING
over everyone else (full census, no cutoff), worked down until daily
capacity runs out; (3) the RETAINED NURSING MST SCREEN as a visible
reference/safety-net layer, modeled as a priority floor for MST-flagged
patients. The counterfactual is the CURRENT pre-ML workflow: the same
time-bound consults first, then the MST-driven queue, under the SAME
capacity.

Design doc: local working document under tasks/ (not committed).

ARCHITECTURE (capacity accounting):
The engine calls step(F) -> step(C) -> predict(F) -> intervene(F) ->
measure(both) each day. intervene() never assesses anyone and never
spends capacity -- it only writes tomorrow's worklist (ML_PRIORITY /
ML_FLAGGED / EROSION_LEVEL) into state. ALL assessments and the single
capacity check live in step(), one code path, both branches. The
factual/counterfactual difference is mediated entirely by state rows
26-28, which stay at their inert values (-inf / 0 / 0) on the
counterfactual branch because intervene() never runs there.

COMMON-RANDOM-NUMBERS DISCIPLINE (load-bearing):
The counterfactual branch's RNG streams are forked copies that stay
synchronized ONLY if both branches consume the same number of draws in
the same order at every step. Therefore every rng.temporal draw in
step() is a fixed-shape length-n array drawn unconditionally, with
masks applied afterward. Never draw inside a data-dependent mask, loop,
or mask.sum()-sized call. TestCRNSync enforces this.

KNOWN SIMPLIFICATIONS (disclosed):
- Assessment is one-shot: a patient assessed before hospital-acquired
  onset is not re-assessed (rescreen cadence not yet modeled).
- MST screening happens once at admission against admission-time truth.
- Time-bound consults are ordered by arrival day (deadline modeling
  adds nothing at daily resolution).
- Provider-consult false positives use a first-order rate calibration
  to the target consult PPV (realization checked by test).
- Coding capture is a calibrated attrition multiplier so that Arm C
  coded prevalence reproduces the site's observed coded rate.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ...core.scenario import (
    BaseScenario,
    Interventions,
    Outcomes,
    Predictions,
    TimeConfig,
)

# -- State row layout -------------------------------------------------------

ROW_TRUE_MALNOURISHED = 0    # 1 once onset has occurred (derived in step)
ROW_TRUE_SEVERITY = 1        # 0 none / 1 moderate (E44) / 2 severe (E43)
ROW_ONSET_DAY = 2            # day malnutrition present; +inf if never
ROW_MODEL_SCORE = 3          # NaN until scored / insufficient-data
ROW_INSUFF_DATA = 4          # 1 = model cannot score (fail-safe path)
ROW_MST_FLAG = 5             # nurse-MST positive at admission
ROW_MANDATORY_CONSULT = 6    # mandatory service line (ICU/PCU etc.)
ROW_PHYS_CONSULT_DAY = 7     # day physician consult lands; +inf if never
ROW_CONSULT_ENTRY_DAY = 8    # day MST flag enters worklist; +inf if never
ROW_ASSESSED = 9             # 1 once an RDN has assessed
ROW_ASSESS_DAY = 10          # -1 until assessed
ROW_ASSESS_SOURCE = 11       # 0 none/1 consult/2 ML/3 MST/4 backstop
ROW_SEVERITY_AT_ASSESS = 12  # -1 until assessed
ROW_DIAGNOSED = 13           # RDN + physician confirmed
ROW_CODED = 14               # E43/E44 coded at discharge
ROW_ADMIT_DAY = 15           # <=0 for initial census
ROW_STATUS = 16              # 0 pre-admission / 1 inpatient / 2 discharged
ROW_LOS_REMAINING = 17       # days to scheduled discharge
ROW_READMIT = 18             # 30-day readmission (drawn at discharge)
ROW_DEMO_SEX = 19
ROW_DEMO_RACE = 20
ROW_DEMO_ETHNICITY = 21
ROW_DEMO_AGE_BAND = 22
ROW_DEMO_PAYER = 23
ROW_SITE = 24                # deployment site index
ROW_HARM_ACCRUED = 25        # cumulative harm units
ROW_ML_PRIORITY = 26         # worklist score written by intervene(); -inf
#                              when unflagged; all -inf on counterfactual
ROW_ML_FLAGGED = 27          # 1 if ever ML-flagged (sticky)
ROW_EROSION_LEVEL = 28       # safety-net erosion prob; 0 on counterfactual
ROW_FIFO_KEY = 29            # ADMIT_DAY + deterministic jitter (unique)
ROW_EPS_LOGIT = 30           # per-entity model-noise latent (creation-time)
ROW_DISCHARGE_DAY = 31       # -1 until discharged (event-day semantics)
ROW_EPS_MIX = 32             # scramble latent (subgroup coverage gaps)
ROW_EPS_HARD = 33            # hard-case latent (global difficulty mix)
ROW_U_READMIT = 34           # creation-time readmission coin (shared
#                              across arms -- exact CRN outcome sharing)
ROW_U_DIAG = 35              # creation-time diagnosis coin
ROW_ML_FLAG_DAY = 36         # first day flagged-unworked (Tier-2 pool
#                              entry); -1 never; bookkeeping, NO RNG

N_STATE_ROWS = 37

# Status / source enums
STATUS_PRE = 0
STATUS_INPATIENT = 1
STATUS_DISCHARGED = 2

SOURCE_NONE = 0
SOURCE_CONSULT = 1    # time-bound consult (mandatory or provider)
SOURCE_ML = 2         # reached via the ML risk ranking
SOURCE_MST = 3        # MST stream / reference layer
SOURCE_BACKSTOP = 4   # capacity reached the list tail

SEV_NONE = 0
SEV_MODERATE = 1
SEV_SEVERE = 2

NEVER = np.inf  # sentinel for "day never happens" rows


# -- Two-channel clinical-effect functions ----------------------------------
# SINGLE SOURCE OF TRUTH: imported verbatim by the outcome overlay.
# Detection (extensive margin): diagnosed-before-discharge carries the
# full relative effect regardless of timing. Timing (intensive margin):
# additional per-day-early effect, saturating at d_ref days.
# days_early(a) = max(d_ref - a, 0), a = assess - max(onset, admit);
# with d_ref = T_sat the dossier separate saturation clip never binds.

def timing_days_early(assess_day, onset_day, admit_day, d_ref):
    """Days-early credit for the timing channel (0 when late/never)."""
    a = assess_day - np.maximum(onset_day, admit_day)
    return np.maximum(d_ref - np.maximum(a, 0.0), 0.0)


def readmission_probability(readmit0, diagnosed, days_early,
                            eff_detect_rel, eff_timing_rel):
    """p = r0 * (1 - dx*eff_detect) * (1 - dx*eff_timing*days_early)."""
    return (readmit0
            * (1.0 - diagnosed * eff_detect_rel)
            * (1.0 - diagnosed * eff_timing_rel * days_early))


def los_reduction(diagnosed, days_early, runway,
                  eff_detect_los, eff_timing_los):
    """LOS days credited at diagnosis, capped by the remaining-stay
    runway (a last-day diagnosis cannot shorten a finished stay)."""
    return diagnosed * np.minimum(
        eff_detect_los + eff_timing_los * days_early, runway)


# Demographic label tables (index-aligned with encoded state ints)
SEX_LABELS = ["Male", "Female"]
RACE_LABELS = ["White", "Black", "Asian", "AIAN", "NHPI", "Other"]
ETHNICITY_LABELS = ["NotHispanic", "HispanicLatino", "Unknown"]
AGE_LABELS = ["18-44", "45-64", "65-74", "75+"]
PAYER_LABELS = ["Commercial", "Medicare", "Medicaid", "SelfPay"]
SITE_LABELS = ["site_0", "site_1"]


@dataclass
class MalnutritionConfig:
    """All scenario parameters.

    Provenance tags: [S] supplied by site artifacts, [O] must obtain
    locally (placeholder defaults), [L] literature-anchored assumption.
    Site-specific values (coded prevalence, severity split, benchmark
    band, secular trend, coding capture) belong in gitignored
    configs/campus/<site>.yaml files -- see campus/example.yaml.
    """

    # -- population / flow (design doc 4.1) --------------------------------
    initial_census: int = 300                # [O] placeholder
    admissions_per_day: int = 40             # [O] placeholder
    true_prevalence: float = 0.30            # [L] GLIM latent -- SWEEP
    coded_prevalence: float = 0.10           # [S] per-site coded rate
    severity_moderate_fraction: float = 0.60  # [S] per-site mod:sev split
    severity_progression_per_day: float = 0.08   # [L] untreated mod->sev
    poa_fraction: float = 0.70               # [O]/[L]; NOT the coded 97%
    los_mean_days: float = 5.5               # [O] placeholder
    los_mean_malnourished: float = 6.9       # [L]
    los_sd_days: float = 2.5                 # [L] non-malnourished SD
    los_sd_malnourished: float = 5.0         # [L] dossier 01 (SD ~5)
    hl_prevalence_multiplier: float = 0.55   # [L] confounded w/ detection

    # -- demographics (design doc 4.2) --------------------------------------
    sex_proportions: Tuple[float, ...] = (0.47, 0.53)              # [S]
    race_proportions: Tuple[float, ...] = (
        0.93, 0.043, 0.007, 0.003, 0.003, 0.014)                   # [S]
    ethnicity_proportions: Tuple[float, ...] = (0.93, 0.05, 0.02)  # [S]
    age_band_proportions: Tuple[float, ...] = (
        0.25, 0.35, 0.20, 0.20)              # [O] placeholder
    payer_proportions: Tuple[float, ...] = (
        0.35, 0.45, 0.15, 0.05)              # [O] placeholder
    site: int = 0                            # deployment site index

    # -- ML model (design doc 4.3) -- FROZEN facts; levers sweepable -------
    ml_enabled: bool = True
    model_params_path: Optional[str] = None  # configs/model_frozen.yaml
    operating_mode: str = "rank_all"         # {"rank_all","top_k","threshold"}
    #   rank_all = deployed policy: full-census score ranking, no cutoff
    top_k_fraction: float = 0.15             # top_k-mode worklist lever
    threshold: float = 0.60                  # [S] deployed model cutoff
    insuff_data_rate: float = 0.05           # [O] fail-safe path
    rescreen_cadence_days: int = 7           # [O] (not yet modeled)
    degenerate_auroc05: bool = False         # verification-only (7)

    # -- baseline workflow (design doc 4.4) -- the [O] sweep levers --------
    mst_sensitivity: float = 0.78            # [L] Arifin 2026
    mst_specificity: float = 0.82            # [L] Arifin 2026
    mandatory_consult_fraction: float = 0.25  # [O] placeholder
    phys_consult_rate_per_day: float = 0.10  # [O] placeholder (true+ arm)
    provider_consult_ppv: float = 0.60       # [L] fraction of provider
    #   consults that are true cases -- SWEEP 0.4-0.8; [O] validate locally
    mst_reference_floor: Optional[float] = None  # path-3 safety net:
    #   MST-flagged patients rank at least this high on the ML list.
    #   None -> use `threshold`; <= 0 disables the floor entirely.
    baseline_worklist_order: str = "mst_then_fifo"   # (vestigial: consults
    #   moved to path 1; retained for config compatibility)
    baseline_time_to_consult_days: float = 3.0  # [O] placeholder, unmeasured

    # -- RDN capacity (design doc 4.5) -- THE binding constraint -----------
    rdn_assessments_per_day: int = 30        # [O] Phillips-2019 anchored

    # -- clinical efficacy: TWO CHANNELS (dossier 06/07) -- sweep from 0 --
    assess_to_diagnosis_rate: float = 0.90   # [L]
    physician_confirm_rate: float = 0.95     # [L]
    eff_detect_rel: float = 0.20             # [L] detection channel,
    #   relative readmission reduction (ONS meta 0.15-0.24); SWEEP
    eff_timing_rel: float = 0.01             # [L] timing channel, per
    #   day-early relative readmission reduction; thin evidence; SWEEP
    eff_detect_los: float = 0.5              # [L] LOS days if diagnosed
    #   (runway-capped); EFFORT-null boundary at 0; SWEEP
    eff_timing_los: float = 0.05             # [L] LOS days per day-early
    timing_d_ref_days: float = 5.0           # [L] days-early reference =
    #   saturation window (single shape; see timing_days_early)
    min_stay_days: float = 1.0
    # severity-dependent readmission baselines (dossier 01; [L])
    readmit_base_nonmal: float = 0.13
    readmit_base_moderate: float = 0.20
    readmit_base_severe: float = 0.30
    severity_score_gradient: float = 0.3     # [L] within-positive score
    #   shift for severe cases (mean-preserving); creates realistic
    #   sicker-scored-higher confounding; Stage-A-gated at this default
    # vestigial (superseded by the two-channel mechanism; kept so old
    # YAML configs still load -- values are IGNORED)
    nutrition_efficacy_los: float = 0.0
    nutrition_efficacy_readmit: float = 0.0
    efficacy_time_slope_per_day: float = 0.0
    baseline_readmit_rate: float = 0.0

    # -- coding pipeline (design doc 4.8 + Appendix A) ----------------------
    cdi_coder_substantiation_rate: float = 0.95  # [L]
    coding_capture_rate: float = 0.585       # [L] calibrated so Arm-C
    #   coded ~= coded_prevalence (recalibrated after the lognormal
    #   LOS change shifted discharge/diagnosis composition)
    secular_coding_trend_per_quarter: float = 0.0   # [S] per-site drift
    flag_driven_coding_uplift: float = 0.0   # [L] SWEEP (Exp 6)
    cohort_benchmark_pct50: float = 0.10     # [S] per-site benchmark
    cohort_benchmark_pct90: float = 0.15     # [S] per-site benchmark

    # -- safety net (design doc 4.7) ----------------------------------------
    safety_net_preserved: bool = True
    unflagged_assessment_decay_per_week: float = 0.0  # [L] SWEEP (Exp 5)

    # -- harm weights (Exp 3 objective) --------------------------------------
    harm_per_day_moderate: float = 0.5
    harm_per_day_severe: float = 1.0
    harm_per_miss: float = 5.0
    harm_per_fp_assessment: float = 0.25


class MalnutritionPrioritizationScenario(BaseScenario[np.ndarray]):
    """Worklist prioritization under RDN capacity, per the 5-method
    contract. State is a (N_STATE_ROWS, n_entities) float array."""

    unit_of_analysis = "patient_admission"

    def __init__(
        self,
        time_config: TimeConfig,
        seed: Optional[int] = None,
        config: Optional[MalnutritionConfig] = None,
    ):
        super().__init__(time_config, seed)
        self.cfg = config if config is not None else MalnutritionConfig()
        self._scorer = None  # lazy-loaded frozen model (predict only)

    # -- helpers -------------------------------------------------------------

    @property
    def n_entities_required(self) -> int:
        """Entity count that fills the configured census + flow."""
        return (self.cfg.initial_census
                + self.cfg.admissions_per_day
                * self.time_config.n_timesteps)

    # -- 1. create_population -----------------------------------------------

    def create_population(self, n_entities: int) -> np.ndarray:
        cfg = self.cfg
        rng = self.rng.population
        n = n_entities
        horizon = self.time_config.n_timesteps

        state = np.zeros((N_STATE_ROWS, n))

        # Admission schedule: initial census admitted in the recent past
        # (mid-stay at t=0); the rest spread evenly across the horizon.
        census = min(cfg.initial_census, n)
        admit = np.empty(n)
        lookback = max(int(round(cfg.los_mean_days)), 1)
        admit[:census] = -rng.integers(0, lookback, size=census).astype(
            float
        )
        rest = n - census
        if rest > 0:
            admit[census:] = np.floor(
                np.arange(rest) * horizon / rest
            ).astype(float)
        state[ROW_ADMIT_DAY] = admit
        state[ROW_STATUS] = STATUS_PRE  # step() activates at ADMIT_DAY

        # Demographics (encoded ints; label tables at module level).
        state[ROW_DEMO_SEX] = rng.choice(
            len(SEX_LABELS), size=n, p=cfg.sex_proportions)
        state[ROW_DEMO_RACE] = rng.choice(
            len(RACE_LABELS), size=n, p=cfg.race_proportions)
        state[ROW_DEMO_ETHNICITY] = rng.choice(
            len(ETHNICITY_LABELS), size=n, p=cfg.ethnicity_proportions)
        state[ROW_DEMO_AGE_BAND] = rng.choice(
            len(AGE_LABELS), size=n, p=cfg.age_band_proportions)
        state[ROW_DEMO_PAYER] = rng.choice(
            len(PAYER_LABELS), size=n, p=cfg.payer_proportions)
        state[ROW_SITE] = float(cfg.site)

        # Latent truth: destined malnutrition with H/L prevalence
        # multiplier, rescaled so the population mean hits the target.
        mult = np.where(
            state[ROW_DEMO_ETHNICITY] == 1,
            cfg.hl_prevalence_multiplier, 1.0)
        raw_p = cfg.true_prevalence * mult
        raw_mean = raw_p.mean()
        scale = cfg.true_prevalence / raw_mean if raw_mean > 0 else 1.0
        p_destined = np.clip(raw_p * scale, 0.0, 0.95)
        destined = rng.random(n) < p_destined

        # Onset timing: POA cases are malnourished at admission;
        # hospital-acquired cases onset partway through the stay.
        poa = destined & (rng.random(n) < cfg.poa_fraction)
        hosp_acq = destined & ~poa

        # Scheduled LOS: per-status LOGNORMAL, moment-matched to the
        # configured (mean, SD) targets. Lognormal because a normal at
        # (6.9, 5) puts ~12% of malnourished at the 1-day floor and
        # biases the mean +0.3d; lognormal keeps both moments to <1%
        # with negligible clipping. Implemented as exp() of the same
        # single fixed-shape normal call, so the z-stream and all
        # downstream population draws stay bit-identical.
        los_mean = np.where(
            destined, cfg.los_mean_malnourished, cfg.los_mean_days)
        los_sd = np.where(
            destined, cfg.los_sd_malnourished, cfg.los_sd_days)
        sigma2 = np.log1p((los_sd / los_mean) ** 2)
        mu = np.log(los_mean) - sigma2 / 2.0
        los = np.clip(
            np.exp(rng.normal(mu, np.sqrt(sigma2))), 1.0, 30.0)
        # Initial census patients have already consumed part of their stay.
        consumed = np.maximum(-admit, 0.0)
        state[ROW_LOS_REMAINING] = np.maximum(los - consumed, 1.0)

        onset = np.full(n, NEVER)
        onset[poa] = admit[poa]
        # Hospital-acquired onset lands strictly within the scheduled
        # stay (offset in [1, los-1]) so it manifests before discharge.
        ha_offset = 1.0 + np.floor(
            rng.random(n) * np.maximum(los - 2.0, 0.0))
        onset[hosp_acq] = admit[hosp_acq] + ha_offset[hosp_acq]
        state[ROW_ONSET_DAY] = onset
        state[ROW_TRUE_MALNOURISHED] = 0.0  # derived in step()

        # Initial severity at onset (progression may worsen it later).
        sev = np.zeros(n)
        sev[destined] = np.where(
            rng.random(n)[destined] < cfg.severity_moderate_fraction,
            SEV_MODERATE, SEV_SEVERE)
        state[ROW_TRUE_SEVERITY] = sev

        # Nurse-MST screen at admission (vs admission-time truth).
        u_mst = rng.random(n)
        mst = np.where(
            poa,
            u_mst < cfg.mst_sensitivity,
            u_mst < (1.0 - cfg.mst_specificity),
        )
        state[ROW_MST_FLAG] = mst.astype(float)

        # Mandatory service-line membership; insufficient-data flag.
        state[ROW_MANDATORY_CONSULT] = (
            rng.random(n) < cfg.mandatory_consult_fraction
        ).astype(float)
        state[ROW_INSUFF_DATA] = (
            rng.random(n) < cfg.insuff_data_rate
        ).astype(float)

        # Baseline worklist entry days (creation-time latents; step()
        # compares t against them -- no per-step RNG needed).
        consult_entry = np.full(n, NEVER)
        mst_delay = np.maximum(
            1.0,
            np.round(rng.exponential(
                cfg.baseline_time_to_consult_days, size=n)),
        )
        consult_entry[mst] = admit[mst] + mst_delay[mst]
        state[ROW_CONSULT_ENTRY_DAY] = consult_entry

        # Provider-initiated (time-bound) consults: true cases draw an
        # arrival after onset; NON-cases draw at a rate calibrated so
        # realized consult PPV ~= provider_consult_ppv (first-order:
        # rate_fp = rate_tp * odds(prev) * (1-ppv)/ppv; the TP window
        # (post-onset stay) and FP window (full stay) are comparable at
        # these defaults -- realization is test-checked, not assumed).
        phys_day = np.full(n, NEVER)
        if cfg.phys_consult_rate_per_day > 0:
            phys_delay = rng.geometric(
                cfg.phys_consult_rate_per_day, size=n).astype(float)
            phys_day[destined] = onset[destined] + phys_delay[destined]
            ppv = cfg.provider_consult_ppv
            if 0 < ppv < 1 and cfg.true_prevalence < 1:
                odds = cfg.true_prevalence / (1 - cfg.true_prevalence)
                r_fp = min(
                    cfg.phys_consult_rate_per_day * odds
                    * (1 - ppv) / ppv, 0.95)
                if r_fp > 0:
                    fp_delay = rng.geometric(
                        r_fp, size=n).astype(float)
                    phys_day[~destined] = (
                        admit[~destined] + fp_delay[~destined])
        state[ROW_PHYS_CONSULT_DAY] = phys_day

        # Bookkeeping sentinels.
        state[ROW_MODEL_SCORE] = np.nan
        state[ROW_ASSESS_DAY] = -1.0
        state[ROW_ASSESS_SOURCE] = SOURCE_NONE
        state[ROW_SEVERITY_AT_ASSESS] = -1.0
        state[ROW_DISCHARGE_DAY] = -1.0
        state[ROW_ML_PRIORITY] = -np.inf
        state[ROW_EROSION_LEVEL] = 0.0

        # Stable FIFO key (admission order + unique jitter).
        state[ROW_FIFO_KEY] = admit + rng.random(n) * 0.9

        # Per-entity model latents (predict() is deterministic in them):
        # Gaussian score noise, scramble uniform (coverage gaps), and
        # hard-case uniform (difficulty mixture).
        state[ROW_EPS_LOGIT] = rng.standard_normal(n)
        state[ROW_EPS_MIX] = rng.random(n)
        state[ROW_EPS_HARD] = rng.random(n)

        # Outcome coins -- APPENDED LAST on the population stream so all
        # earlier population realizations stay bit-identical (pinned
        # order: ... eps_hard -> U_READMIT -> U_DIAG). Shared across
        # arms: exact CRN coin sharing for diagnosis and readmission.
        state[ROW_U_READMIT] = rng.random(n)
        state[ROW_U_DIAG] = rng.random(n)
        state[ROW_ML_FLAG_DAY] = -1.0  # bookkeeping, no RNG

        return state

    # -- 2. step -------------------------------------------------------------

    def step(self, state: np.ndarray, t: int) -> np.ndarray:
        cfg = self.cfg
        ns = state.copy()
        n = ns.shape[1]
        rng = self.rng.temporal

        # CRN DISCIPLINE: all temporal draws are fixed-shape, drawn
        # unconditionally and in a fixed order; masks applied afterward.
        # (Erosion lives on the intervention stream; diagnosis and
        # readmission use creation-time per-patient coins so outcomes
        # share randomness across arms exactly.)
        u_progress = np.asarray(rng.random(n))
        u_coding = np.asarray(rng.random(n))

        # 1. Admissions: activate entities whose day has come.
        newly_admitted = (
            (ns[ROW_STATUS] == STATUS_PRE) & (ns[ROW_ADMIT_DAY] <= t)
        )
        ns[ROW_STATUS, newly_admitted] = STATUS_INPATIENT

        inpatient = ns[ROW_STATUS] == STATUS_INPATIENT

        # 2. Truth update (deterministic from creation-time onset days).
        ns[ROW_TRUE_MALNOURISHED] = (
            ns[ROW_ONSET_DAY] <= t
        ).astype(float)
        malnourished = ns[ROW_TRUE_MALNOURISHED] == 1

        # 3. Severity progression: moderate worsens to severe along the
        #    UNDERLYING disease course, deliberately NOT gated on
        #    diagnosis. All treatment effect on outcomes flows through
        #    the four efficacy knobs (reduced-form, sweepable to zero);
        #    a diagnosis-halted progression would leak an un-sweepable
        #    readmission benefit through the severity-dependent
        #    baseline (~0.35 pp at eff=0, the size of the detection
        #    channel itself) and break the efficacy=0 boundary.
        #    Severity here is a case-mix risk marker: harm accrual and
        #    severity_at_assess only ever read it pre-assessment, so
        #    the process story is unchanged.
        progress = (
            inpatient & malnourished
            & (ns[ROW_TRUE_SEVERITY] == SEV_MODERATE)
            & (u_progress < cfg.severity_progression_per_day)
        )
        ns[ROW_TRUE_SEVERITY, progress] = SEV_SEVERE

        # 4. Tiered capacity-constrained assessment (both branches; the
        #    single capacity enforcement point in the whole scenario).
        assessed_today = self._run_worklist(ns, t)

        # 5. Diagnosis for newly assessed, currently-malnourished cases
        #    (creation-time coin -> deterministic per patient).
        p_dx = cfg.assess_to_diagnosis_rate * cfg.physician_confirm_rate
        diagnosed_today = (
            assessed_today & malnourished
            & (ns[ROW_U_DIAG] < p_dx)
        )
        ns[ROW_DIAGNOSED, diagnosed_today] = 1.0

        # Two-channel LOS effect at diagnosis: detection days plus
        # per-day-early timing days, capped by the remaining-stay
        # runway and a minimum realized stay.
        days_early = timing_days_early(
            float(t), ns[ROW_ONSET_DAY], ns[ROW_ADMIT_DAY],
            cfg.timing_d_ref_days)
        stay_so_far = float(t) - ns[ROW_ADMIT_DAY]
        min_stay_guard = np.maximum(
            cfg.min_stay_days - stay_so_far, 0.0)
        runway = np.maximum(
            ns[ROW_LOS_REMAINING] - min_stay_guard, 0.0)
        los_credit = np.where(
            diagnosed_today,
            los_reduction(1.0, days_early, runway,
                          cfg.eff_detect_los, cfg.eff_timing_los),
            0.0)
        # Apply the credit (and its zero-clip) ONLY to today's
        # diagnoses: an unmasked maximum(remaining, 0) would silently
        # zero the post-discharge fractional leftover in (-1, 0],
        # which the outcome overlay needs to recover continuous LOS.
        ns[ROW_LOS_REMAINING, diagnosed_today] = np.maximum(
            (ns[ROW_LOS_REMAINING] - los_credit)[diagnosed_today],
            0.0)

        # 6. Harm accrual: unassessed malnourished inpatients accrue
        #    per-day harm by severity; false-positive assessments burn
        #    RDN time (one-time burden).
        unassessed_harm = (
            inpatient & malnourished & (ns[ROW_ASSESSED] == 0))
        harm_rate = np.where(
            ns[ROW_TRUE_SEVERITY] == SEV_SEVERE,
            cfg.harm_per_day_severe, cfg.harm_per_day_moderate)
        ns[ROW_HARM_ACCRUED] += np.where(unassessed_harm, harm_rate, 0.0)
        fp_assessed = assessed_today & ~malnourished
        ns[ROW_HARM_ACCRUED, fp_assessed] += cfg.harm_per_fp_assessment

        # 7. LOS countdown and discharge.
        ns[ROW_LOS_REMAINING, inpatient] -= 1.0
        discharged_today = inpatient & (ns[ROW_LOS_REMAINING] <= 0)
        ns[ROW_STATUS, discharged_today] = STATUS_DISCHARGED
        ns[ROW_DISCHARGE_DAY, discharged_today] = float(t)
        missed = (
            discharged_today & malnourished & (ns[ROW_ASSESSED] == 0))
        ns[ROW_HARM_ACCRUED, missed] += cfg.harm_per_miss

        # 8. Coding at discharge (domain physics on BOTH branches):
        #    diagnosed cases survive the CDI/coder/documentation funnel;
        #    the campus secular CDI trend drifts the capture rate; the
        #    flag-driven uplift channel codes UNsubstantiated ML flags.
        trend = 1.0 + (cfg.secular_coding_trend_per_quarter
                       * (t / 91.25))
        p_code = np.clip(
            cfg.cdi_coder_substantiation_rate
            * cfg.coding_capture_rate * trend, 0.0, 1.0)
        coded_legit = (
            discharged_today & (ns[ROW_DIAGNOSED] == 1)
            & (u_coding < p_code))
        coded_uplift = (
            discharged_today & (ns[ROW_DIAGNOSED] == 0)
            & (ns[ROW_ML_FLAGGED] == 1)
            & (u_coding < cfg.flag_driven_coding_uplift))
        ns[ROW_CODED, coded_legit | coded_uplift] = 1.0

        # 9. Readmission at discharge: severity-dependent baseline,
        #    two-channel effect, creation-time coin (shared across
        #    arms -- the same patient flips the same coin).
        readmit0 = self._readmit_baseline(ns, t)
        de_at_assess = timing_days_early(
            ns[ROW_ASSESS_DAY], ns[ROW_ONSET_DAY], ns[ROW_ADMIT_DAY],
            cfg.timing_d_ref_days)
        p_readmit = readmission_probability(
            readmit0, ns[ROW_DIAGNOSED], de_at_assess,
            cfg.eff_detect_rel, cfg.eff_timing_rel)
        readmit = discharged_today & (ns[ROW_U_READMIT] < p_readmit)
        ns[ROW_READMIT, readmit] = 1.0

        return ns

    def _readmit_baseline(
        self, ns: np.ndarray, t: "float | np.ndarray"
    ) -> np.ndarray:
        """Baseline 30-day readmission probability by true status and
        severity at (or before) discharge (dossier 01 anchors).
        Accepts a scalar day (step) or per-patient days (measure's
        at-discharge oracle)."""
        cfg = self.cfg
        mal = ns[ROW_ONSET_DAY] <= t
        severe = ns[ROW_TRUE_SEVERITY] == SEV_SEVERE
        return np.where(
            mal,
            np.where(severe, cfg.readmit_base_severe,
                     cfg.readmit_base_moderate),
            cfg.readmit_base_nonmal)

    def _run_worklist(self, ns: np.ndarray, t: int) -> np.ndarray:
        """One day of RDN assessments. Mutates ns; returns the boolean
        mask of patients assessed today.

        The deployed three-path priority order, four exclusive tiers
        (lowest wins), one lexsort, first-C taken:
          0  time-bound consults (mandatory from admission + provider-
             initiated), arrival order -- a clinical obligation, worked
             first on BOTH branches (the model never supersedes them)
          1  the ML risk ranking (ML_PRIORITY desc; includes the MST
             reference floor written by intervene) -- structurally
             empty on the counterfactual branch
          2  the MST-driven stream (the current workflow's queue; on
             the factual branch this catches only patients outside the
             ranked list, e.g. unscored + floor-eroded)
          3  backstop tail, FIFO
        """
        cfg = self.cfg
        n = ns.shape[1]

        eligible = (
            (ns[ROW_STATUS] == STATUS_INPATIENT)
            & (ns[ROW_ASSESSED] == 0))

        # Time-bound consult day: mandatory lines are consult-bound
        # from admission; provider-initiated consults from arrival.
        timebound_day = np.where(
            ns[ROW_MANDATORY_CONSULT] == 1, ns[ROW_ADMIT_DAY], NEVER)
        timebound_day = np.minimum(
            timebound_day, ns[ROW_PHYS_CONSULT_DAY])

        tier = np.full(n, 9.0)
        tier[eligible] = 3.0
        tier[eligible & (ns[ROW_CONSULT_ENTRY_DAY] <= t)] = 2.0
        tier[eligible & (ns[ROW_ML_PRIORITY] > -np.inf)] = 1.0
        tier[eligible & (timebound_day <= t)] = 0.0

        # Within-tier ordering keys (tier is the lexsort primary).
        fifo = ns[ROW_FIFO_KEY] + 500.0    # positive for composites
        key = fifo.copy()
        t0 = tier == 0.0
        key[t0] = (timebound_day[t0] + 1000.0) * 1e4 + fifo[t0]
        t1 = tier == 1.0
        key[t1] = -ns[ROW_ML_PRIORITY, t1]

        queue = np.where(tier < 9.0)[0]
        assessed_today = np.zeros(n, dtype=bool)
        if len(queue) > 0:
            order = np.lexsort((key[queue], tier[queue]))
            capacity = int(min(cfg.rdn_assessments_per_day, len(queue)))
            chosen = queue[order[:capacity]] if capacity > 0 else queue[:0]
            assessed_today[chosen] = True
            ns[ROW_ASSESSED, chosen] = 1.0
            ns[ROW_ASSESS_DAY, chosen] = float(t)
            ns[ROW_ASSESS_SOURCE, chosen] = tier[chosen] + 1.0
            ns[ROW_SEVERITY_AT_ASSESS, chosen] = np.where(
                ns[ROW_TRUE_MALNOURISHED, chosen] == 1,
                ns[ROW_TRUE_SEVERITY, chosen], SEV_NONE)
        return assessed_today

    # -- 3. predict -----------------------------------------------------------

    def predict(self, state: np.ndarray, t: int) -> Predictions:
        """Daily batch scoring (factual branch only).

        Deterministic in (current truth, EPS_LOGIT, frozen params):
        the per-entity noise latent was drawn once at creation, so
        re-scoring produces no ranking churn. Read-only on state --
        intervene() persists scores into ROW_MODEL_SCORE.
        """
        cfg = self.cfg
        n = state.shape[1]
        scoreable = (
            (state[ROW_STATUS] == STATUS_INPATIENT)
            & (state[ROW_INSUFF_DATA] == 0))

        if not cfg.ml_enabled:
            return Predictions(
                scores=np.zeros(n),
                metadata={"ml_enabled": False,
                          "scoreable": np.zeros(n, dtype=bool)},
            )

        scorer = self._load_scorer()
        scores = scorer.score(
            state, degenerate=cfg.degenerate_auroc05,
            severity_gradient=cfg.severity_score_gradient,
            moderate_fraction=cfg.severity_moderate_fraction)
        scores = np.where(scoreable, scores, np.nan)
        true_labels = state[ROW_TRUE_MALNOURISHED] == 1
        return Predictions(
            scores=scores,
            metadata={
                "ml_enabled": True,
                "scoreable": scoreable,
                "true_labels": true_labels,
                "n_scoreable": int(scoreable.sum()),
            },
        )

    def _load_scorer(self):
        """Load frozen subgroup-aware scorer. Fails loudly if the
        frozen params have not been generated (run_verification.py
        Stage A) -- no silent defaults."""
        if self._scorer is None:
            from .subgroup_performance import FrozenSubgroupScorer
            path = self.cfg.model_params_path
            if path is None:
                path = str(
                    Path(__file__).parent / "configs"
                    / "model_frozen.yaml")
            self._scorer = FrozenSubgroupScorer.load(path)
        return self._scorer

    # -- 4. intervene ---------------------------------------------------

    def intervene(
        self, state: np.ndarray, predictions: Predictions, t: int
    ) -> tuple[np.ndarray, Interventions]:
        """Write tomorrow's worklist into state (factual only).

        Never assesses anyone; never spends capacity. Builds the
        path-2 ranking (rank_all: the full census; top_k/threshold:
        legacy cutoff policies kept for comparison), applies the
        path-3 MST reference floor (with erosion = the probability an
        RDN ignores the retained screen that day, drawn on the
        intervention stream -- factual-only by construction), marks
        the high-risk display label (ML_FLAGGED = score >= threshold,
        which drives only the coding-uplift channel), and updates the
        erosion level.
        """
        cfg = self.cfg
        ns = state.copy()
        n = ns.shape[1]

        if not cfg.ml_enabled:
            return ns, Interventions(
                treated_indices=np.array([], dtype=int),
                metadata={"ml_enabled": False},
            )

        scores = predictions.scores
        scoreable = predictions.metadata["scoreable"]

        # Persist today's scores (predict is read-only on state).
        ns[ROW_MODEL_SCORE, scoreable] = scores[scoreable]

        candidates = (
            (ns[ROW_STATUS] == STATUS_INPATIENT)
            & (ns[ROW_ASSESSED] == 0)
            & scoreable)
        cand_idx = np.where(candidates)[0]

        prio = np.full(n, -np.inf)
        if cfg.operating_mode == "rank_all":
            # Deployed policy: every scoreable unassessed inpatient is
            # on the ranked list -- no cutoff.
            prio[cand_idx] = scores[cand_idx]
        elif cfg.operating_mode == "top_k":
            n_inpatients = int(
                (ns[ROW_STATUS] == STATUS_INPATIENT).sum())
            k = int(round(cfg.top_k_fraction * n_inpatients))
            k = min(k, len(cand_idx))
            if k > 0:
                order = np.argsort(-scores[cand_idx], kind="stable")
                sel = cand_idx[order[:k]]
                prio[sel] = scores[sel]
        elif cfg.operating_mode == "threshold":
            sel = cand_idx[scores[cand_idx] >= cfg.threshold]
            prio[sel] = scores[sel]
        else:
            raise ValueError(
                f"unknown operating_mode: {cfg.operating_mode!r}")

        # High-risk display label (Epic list category). Under rank_all
        # it does NOT gate work order -- it feeds the coding-uplift
        # channel and reporting only. Sticky; candidates are unassessed
        # by construction, so today's flags are flagged-unworked, and
        # the first such day (the day tomorrow's worklist shows them)
        # defines the Tier-2 randomization pool entry.
        flagged = cand_idx[scores[cand_idx] >= cfg.threshold]
        ns[ROW_ML_FLAGGED, flagged] = 1.0
        first_flag = flagged[ns[ROW_ML_FLAG_DAY, flagged] < 0]
        ns[ROW_ML_FLAG_DAY, first_flag] = float(t + 1)

        # Safety-net erosion level (H7): probability an RDN ignores
        # the retained MST reference on a given day.
        if not cfg.safety_net_preserved:
            level = min(
                1.0,
                cfg.unflagged_assessment_decay_per_week * (t + 1) / 7.0)
            ns[ROW_EROSION_LEVEL] = level

        # Path 3: the retained MST screen as a priority floor. Drawn on
        # the intervention stream (fixed shape, factual-only branch).
        u_floor = self.rng.intervention.random(n)
        floor = (cfg.mst_reference_floor
                 if cfg.mst_reference_floor is not None
                 else cfg.threshold)
        if floor > 0:
            honored = (
                (ns[ROW_STATUS] == STATUS_INPATIENT)
                & (ns[ROW_ASSESSED] == 0)
                & (ns[ROW_MST_FLAG] == 1)
                & (u_floor >= ns[ROW_EROSION_LEVEL]))
            prio = np.where(
                honored & (prio < floor), floor, prio)

        ns[ROW_ML_PRIORITY] = prio

        return ns, Interventions(
            treated_indices=flagged,
            metadata={
                "n_ranked_today": int((prio > -np.inf).sum()),
                "n_flagged_today": int(len(flagged)),
                "erosion_level": float(ns[ROW_EROSION_LEVEL, 0]),
            },
        )

    # -- 5. measure -----------------------------------------------------

    def measure(self, state: np.ndarray, t: int) -> Outcomes:
        """Draw-free aggregation (both branches). Primary event =
        missed case, fired ONCE on the entity's discharge day."""
        n = state.shape[1]
        malnourished = state[ROW_TRUE_MALNOURISHED] == 1
        assessed = state[ROW_ASSESSED] == 1
        discharged_today = state[ROW_DISCHARGE_DAY] == t

        # A case counts as malnourished-at-discharge only if onset
        # occurred on or before the discharge day (a hospital-acquired
        # onset scheduled past discharge never manifested in-hospital).
        mal_at_discharge = (
            state[ROW_ONSET_DAY] <= state[ROW_DISCHARGE_DAY])

        events = (
            discharged_today & mal_at_discharge & ~assessed
        ).astype(float)

        missed_cum = (
            (state[ROW_STATUS] == STATUS_DISCHARGED)
            & mal_at_discharge & ~assessed)
        tta = np.where(
            assessed,
            state[ROW_ASSESS_DAY] - np.maximum(
                state[ROW_ADMIT_DAY], 0.0),
            -1.0)
        los_actual = np.where(
            state[ROW_STATUS] == STATUS_DISCHARGED,
            state[ROW_DISCHARGE_DAY] - state[ROW_ADMIT_DAY], -1.0)

        inpatient = state[ROW_STATUS] == STATUS_INPATIENT
        assessed_today_count = int((state[ROW_ASSESS_DAY] == t).sum())
        queue_depth = int((inpatient & ~assessed).sum())
        n_discharged = int(
            (state[ROW_STATUS] == STATUS_DISCHARGED).sum())
        n_coded = int((state[ROW_CODED] == 1).sum())

        def decode(labels, row):
            return np.array(labels)[state[row].astype(int)]

        return Outcomes(
            events=events,
            entity_ids=np.arange(n),
            secondary={
                "true_malnourished": malnourished.astype(float),
                # Oracle fields for the outcome overlay (per-patient).
                # readmit0 is evaluated at each patient's own discharge
                # day (not the reporting day t): an onset scheduled
                # past discharge never manifested, so the baseline the
                # step() readmission draw actually used is the one at
                # discharge time.
                "onset_day": state[ROW_ONSET_DAY].copy(),
                "admit_day": state[ROW_ADMIT_DAY].copy(),
                "u_readmit": state[ROW_U_READMIT].copy(),
                "readmit0": self._readmit_baseline(
                    state,
                    np.where(state[ROW_DISCHARGE_DAY] >= 0,
                             np.minimum(float(t),
                                        state[ROW_DISCHARGE_DAY]),
                             float(t))),
                "discharge_day": state[ROW_DISCHARGE_DAY].copy(),
                # Fractional stay leftover: continuous LOS is exactly
                # (discharge - admit) + los_remaining for discharged
                # patients (repeated -1.0 decrements are fp-exact).
                "los_remaining": state[ROW_LOS_REMAINING].copy(),
                "ml_flag_day": state[ROW_ML_FLAG_DAY].copy(),
                # Proper missed-rate denominator: onset manifested
                # in-hospital (on or before the discharge day).
                "mal_at_discharge": (
                    ((state[ROW_STATUS] == STATUS_DISCHARGED)
                     & mal_at_discharge).astype(float)),
                "severity": state[ROW_TRUE_SEVERITY].copy(),
                "assessed": assessed.astype(float),
                "assess_day": state[ROW_ASSESS_DAY].copy(),
                "assess_source": state[ROW_ASSESS_SOURCE].copy(),
                "time_to_assessment": tta,
                "missed": missed_cum.astype(float),
                "severity_at_assess": (
                    state[ROW_SEVERITY_AT_ASSESS].copy()),
                "harm": state[ROW_HARM_ACCRUED].copy(),
                "diagnosed": state[ROW_DIAGNOSED].copy(),
                "coded": state[ROW_CODED].copy(),
                "los": los_actual,
                "readmit": state[ROW_READMIT].copy(),
                "ml_flagged": state[ROW_ML_FLAGGED].copy(),
                "status": state[ROW_STATUS].copy(),
                "sex": decode(SEX_LABELS, ROW_DEMO_SEX),
                "race": decode(RACE_LABELS, ROW_DEMO_RACE),
                "ethnicity": decode(
                    ETHNICITY_LABELS, ROW_DEMO_ETHNICITY),
                "age_band": decode(AGE_LABELS, ROW_DEMO_AGE_BAND),
                "payer": decode(PAYER_LABELS, ROW_DEMO_PAYER),
                "site": decode(SITE_LABELS, ROW_SITE),
                "subgroup": decode(
                    ETHNICITY_LABELS, ROW_DEMO_ETHNICITY),
            },
            metadata={
                "census": int(inpatient.sum()),
                "queue_depth": queue_depth,
                "assessed_today": assessed_today_count,
                "n_discharged": n_discharged,
                "n_coded": n_coded,
                "coded_rate_of_discharged": (
                    n_coded / n_discharged if n_discharged else 0.0),
                # Tier-2 randomizable pool: flagged, unworked, inpatient.
                "n_flagged_unworked": int((
                    inpatient & ~assessed
                    & (state[ROW_ML_FLAGGED] == 1)).sum()),
            },
        )

    # -- optional hooks -------------------------------------------------

    def clone_state(self, state: np.ndarray) -> np.ndarray:
        return state.copy()

    def validate_population(self, state: np.ndarray) -> Dict[str, Any]:
        n = state.shape[1]
        destined = np.isfinite(state[ROW_ONSET_DAY])
        return {
            "n_entities": n,
            "destined_malnourished_frac": float(destined.mean()),
            "poa_frac_of_destined": float(
                (state[ROW_ONSET_DAY][destined]
                 == state[ROW_ADMIT_DAY][destined]).mean()
                if destined.any() else 0.0),
            "mandatory_frac": float(
                state[ROW_MANDATORY_CONSULT].mean()),
            "insuff_frac": float(state[ROW_INSUFF_DATA].mean()),
            "mst_flag_frac": float(state[ROW_MST_FLAG].mean()),
            "hl_frac": float(
                (state[ROW_DEMO_ETHNICITY] == 1).mean()),
            "mean_scheduled_los": float(
                state[ROW_LOS_REMAINING].mean()),
            # Effective LOS moments after lognormal + clip (the
            # honest-SD disclosure; clip mass should be <1%).
            "los_mal_mean_sd": (
                float(state[ROW_LOS_REMAINING][destined].mean()),
                float(state[ROW_LOS_REMAINING][destined].std()))
            if destined.any() else (0.0, 0.0),
            "los_clipped_frac": float(
                ((state[ROW_LOS_REMAINING] <= 1.0)
                 | (state[ROW_LOS_REMAINING] >= 30.0)).mean()),
            "population_created": True,
        }
