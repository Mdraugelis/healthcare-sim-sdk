"""Subgroup-differential model performance for a frozen, externally
validated ranking model.

No SDK-level mechanism exists for per-subgroup model performance (the
noshow scenarios model equity via prevalence only), so this scenario
carries its own. The frozen scorer is a three-component ranking model,
deterministic per entity in three creation-time latents:

    z_i = d_eff_i * y_i + sigma_i * EPS_LOGIT_i
    d_eff_i = d * (delta_hard if hard_i) * (scramble_depth if scr_i)
    sigma_i = softplus(BETA0 + (1-kappa) * theta_i)
    theta_i = th_f*female_i + th_b*black_i + th_hl*hl_i
    scr_i   = EPS_MIX_i  < kappa * scramble_scale * theta_i
    hard_i  = EPS_HARD_i < h

Interpretation: subgroup degradation is part score NOISE (sigma) and
part COVERAGE GAP (scramble -- the model effectively does not see a
fraction of subgroup patients, e.g. documentation/feature coverage);
a global HARD-CASE mass (fraction h at attenuated signal) shapes the
mid/deep ranking so a published capture@K curve and global AUROC are
jointly reproducible. kappa splits noise vs coverage.

MODEL TARGETS ARE INSTITUTION-SPECIFIC AND LIVE OUTSIDE VERSION
CONTROL: `configs/model_targets.yaml` (gitignored; template at
`configs/model_targets.example.yaml`). Calibration is a deterministic
Nelder-Mead polish (fixed cohort, fixed warm start from the targets
file, fixed iteration budget -> bit-reproducible) against the targets'
global AUROC, subgroup AUPRC lifts, and capture/precision@K curve.

Gates (run_verification.py Stage A, and TestModelFidelity):
  - calibration cohort: |AUROC - target| <= 0.005; lifts +/- 0.05.
    Lifts are prevalence-dependent and estimated on small margins, so
    they are gated where they were calibrated; independent-cohort lift
    realizations are REPORTED as informational -- their spread is a
    finding (published subgroup estimates carry sampling noise of the
    same magnitude).
  - independent cohort: capture@K +/- 0.03; P@K, sens and PPV at the
    deployed threshold +/- 0.04 (top-of-list precision on a real test
    set rests on few encounters; +/-0.04 is within the mutual
    measurement noise of the two estimates).

Calibrated params freeze into configs/model_frozen.yaml (gitignored --
they encode the institution's targets; regenerating requires re-running
the Stage A gate). The monotone probability map score = sigmoid(a*z+b)
anchors the flag rate at the deployed threshold; beyond that anchor it
is cosmetic (the score is consumed as a RANK, not a probability).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import yaml  # type: ignore[import-untyped]

from .scenario import (
    ROW_DEMO_ETHNICITY,
    ROW_DEMO_RACE,
    ROW_DEMO_SEX,
    ROW_EPS_HARD,
    ROW_EPS_LOGIT,
    ROW_EPS_MIX,
    ROW_TRUE_MALNOURISHED,
    MalnutritionConfig,
)

CALIBRATION_SEED = 20260703
CALIBRATION_N = 100000

TOL_AUROC = 0.005
TOL_LIFT = 0.05
TOL_CAPTURE = 0.03
TOL_PRECISION = 0.04

# softplus(BETA0) = 1.0 -> base noise scale exactly 1.
BETA0 = float(np.log(np.expm1(1.0)))

NM_MAX_ITER = 1500

TARGETS_PATH = Path(__file__).parent / "configs" / "model_targets.yaml"


class CalibrationError(RuntimeError):
    """Raised when calibration cannot hit the gated tolerances
    (never silently freeze off-target parameters)."""


@dataclass
class ModelTargets:
    """Institution-specific model performance targets (loaded from the
    gitignored configs/model_targets.yaml)."""

    auroc: float
    lifts: Dict[str, float]           # {female, black, hl}
    capture: Dict[float, float]       # {K fraction: capture@K}
    precision: Dict[float, float]     # {K fraction: P@K}
    sens: float                       # at the deployed threshold
    ppv: float
    flag_rate_anchor: float
    threshold_anchor: float
    calibration_prevalence: float
    warm_start: Tuple[float, ...]


def load_targets(path: Optional[str] = None) -> ModelTargets:
    p = Path(path) if path is not None else TARGETS_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"Model targets not found at {p}. Copy "
            "configs/model_targets.example.yaml to "
            "configs/model_targets.yaml and fill in your model's "
            "validated performance characteristics. The file is "
            "gitignored -- institution-specific model documentation "
            "must not be committed.")
    raw = yaml.safe_load(p.read_text())
    return ModelTargets(
        auroc=float(raw["target_auroc"]),
        lifts={k: float(v) for k, v in raw["target_lifts"].items()},
        capture={float(k): float(v)
                 for k, v in raw["target_capture"].items()},
        precision={float(k): float(v)
                   for k, v in raw["target_precision"].items()},
        sens=float(raw["target_sens"]),
        ppv=float(raw["target_ppv"]),
        flag_rate_anchor=float(raw["flag_rate_anchor"]),
        threshold_anchor=float(raw["threshold_anchor"]),
        calibration_prevalence=float(raw["calibration_prevalence"]),
        warm_start=tuple(float(v) for v in raw["warm_start"]),
    )


def _softplus(x):
    x = np.asarray(x, dtype=float)
    return np.where(x > 30.0, x, np.log1p(np.exp(np.minimum(x, 30.0))))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _rank_auc(y: np.ndarray, z: np.ndarray) -> float:
    """Mann-Whitney AUC via ranks (exact for continuous scores)."""
    from scipy.stats import rankdata  # type: ignore[import-untyped]
    r = rankdata(z)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _ap(y: np.ndarray, z: np.ndarray) -> float:
    """Average precision (rank-only; matches ml.performance.auprc)."""
    order = np.argsort(-z, kind="stable")
    hits = y[order] == 1
    n_pos = int(hits.sum())
    if n_pos == 0:
        return 0.0
    cum = np.cumsum(hits)
    ranks = np.arange(1, len(y) + 1)
    return float((cum[hits] / ranks[hits]).sum() / n_pos)


def build_calibration_cohort(
    prevalence: float,
    n: int = CALIBRATION_N,
    seed: int = CALIBRATION_SEED,
) -> Dict[str, np.ndarray]:
    """Fixed-seed synthetic cohort with the scenario's demographic
    structure (incl. the H/L prevalence multiplier) and the three
    per-entity model latents."""
    from ...core.rng import RNGPartitioner
    cfg = MalnutritionConfig()
    streams = RNGPartitioner(seed).create_streams()
    rng = streams.population   # demographic / label draws
    lat = streams.prediction   # per-entity model latents
    female = rng.random(n) < cfg.sex_proportions[1]
    race = rng.choice(
        len(cfg.race_proportions), size=n, p=cfg.race_proportions)
    black = race == 1
    hl = rng.random(n) < cfg.ethnicity_proportions[1]
    mult = np.where(hl, cfg.hl_prevalence_multiplier, 1.0)
    p = prevalence * mult
    p = p * (prevalence / p.mean())
    y = (rng.random(n) < p).astype(int)
    eps = lat.standard_normal(n)
    mix = lat.random(n)
    hard = lat.random(n)
    # Severity labels APPENDED AFTER all existing draws so earlier
    # cohort realizations stay bit-identical (realization stability).
    severe = (lat.random(n) < (1.0 - cfg.severity_moderate_fraction))
    return {
        "female": female, "black": black, "hl": hl, "y": y,
        "eps": eps, "mix": mix, "hard": hard, "severe": severe,
        "m_frac": cfg.severity_moderate_fraction,
    }


@dataclass
class ScorerParams:
    d: float               # base class separation
    kappa: float           # noise <-> coverage-gap split
    h: float               # global hard-case fraction
    theta_female: float    # subgroup degradation intensities
    theta_black: float
    theta_hl: float
    delta_hard: float      # hard-case signal attenuation
    scramble_scale: float  # coverage-gap prob per theta unit
    scramble_depth: float  # residual signal for scrambled patients
    map_a: float = 1.0     # monotone probability map
    map_b: float = 0.0


def _components(params, female, black, hl, eps_mix, eps_hard):
    theta = (params.theta_female * female.astype(float)
             + params.theta_black * black.astype(float)
             + params.theta_hl * hl.astype(float))
    sigma = _softplus(BETA0 + (1.0 - params.kappa) * theta)
    pi = np.clip(params.kappa * params.scramble_scale * theta,
                 0.0, 0.9)
    scrambled = eps_mix < pi
    hard = eps_hard < params.h
    d_eff = (params.d
             * np.where(hard, params.delta_hard, 1.0)
             * np.where(scrambled, params.scramble_depth, 1.0))
    return d_eff, sigma


def _gradient_term(y, severe, gradient, m_frac):
    """Mean-preserving within-positive severity gradient: severe
    positives shift up by g*m_frac, moderate down by g*(1-m_frac)
    (zero-mean at the configured onset mix). Reads CURRENT severity --
    realistic (worsening labs) and the estimand-3 confounding channel."""
    c = np.where(severe, m_frac, -(1.0 - m_frac))
    return gradient * c * y


def _raw_scores(params: ScorerParams, cohort,
                gradient: float = 0.0) -> np.ndarray:
    d_eff, sigma = _components(
        params, cohort["female"], cohort["black"], cohort["hl"],
        cohort["mix"], cohort["hard"])
    z = d_eff * cohort["y"] + sigma * cohort["eps"]
    if gradient:
        z = z + _gradient_term(
            cohort["y"], cohort["severe"], gradient,
            cohort.get("m_frac", 0.6))
    return z


def evaluate(params: ScorerParams, cohort,
             targets: ModelTargets,
             gradient: float = 0.0) -> Dict:
    """Achieved metrics: AUROC, AUPRC, marginal lifts, capture/P@K."""
    z = _raw_scores(params, cohort, gradient=gradient)
    y = cohort["y"]
    n = len(y)
    global_ap = _ap(y, z)
    ys = y[np.argsort(-z, kind="stable")]
    npos = int(y.sum())
    out = {
        "auroc": _rank_auc(y, z),
        "auprc": global_ap,
        "capture": {k: float(ys[:int(k * n)].sum() / npos)
                    for k in targets.capture},
        "precision": {k: float(ys[:int(k * n)].mean())
                      for k in targets.precision},
    }
    for name in ("female", "black", "hl"):
        mask = cohort[name]
        out[f"lift_{name}"] = (
            _ap(y[mask], z[mask]) / global_ap if global_ap > 0
            else 0.0)
    out["lift_male"] = (
        _ap(y[~cohort["female"]], z[~cohort["female"]]) / global_ap)
    return out


def _unpack(x: np.ndarray) -> ScorerParams:
    return ScorerParams(
        d=0.5 + 4.5 / (1 + np.exp(-x[0])),
        kappa=1.0 / (1 + np.exp(-x[1])),
        h=0.6 / (1 + np.exp(-x[2])),
        theta_female=float(np.exp(x[3])),
        theta_black=float(np.exp(x[4])),
        theta_hl=float(np.exp(x[5])),
        delta_hard=0.05 + 0.9 / (1 + np.exp(-x[6])),
        scramble_scale=0.5 / (1 + np.exp(-x[7])),
        scramble_depth=1.0 / (1 + np.exp(-x[8])),
    )


def _pack(p) -> np.ndarray:
    def inv(v, lo, hi):
        return float(np.log((v - lo) / (hi - v)))
    return np.array([
        inv(p[0], 0.5, 5.0), inv(p[1], 0.0, 1.0),
        inv(p[2], 0.0, 0.6),
        np.log(p[3]), np.log(p[4]), np.log(p[5]),
        inv(p[6], 0.05, 0.95), inv(p[7], 0.0, 0.5),
        inv(p[8], 0.0, 1.0)])


def calibrate(
    targets: Optional[ModelTargets] = None,
    n: int = CALIBRATION_N,
    seed: int = CALIBRATION_SEED,
) -> tuple:
    """Deterministic joint calibration (fixed cohort, fixed warm start
    from the targets file, fixed iteration budget).

    Returns (ScorerParams, achieved dict on the calibration cohort).
    Raises CalibrationError with a residual table if a gated
    tolerance is unmet.
    """
    from scipy.optimize import minimize  # type: ignore[import-untyped]

    if targets is None:
        targets = load_targets()
    cohort = build_calibration_cohort(
        prevalence=targets.calibration_prevalence, n=n, seed=seed)
    k_top = min(targets.precision)   # smallest K = top-of-list purity

    def objective(x):
        m = evaluate(_unpack(x), cohort, targets)
        r = 150 * (m["auroc"] - targets.auroc) ** 2 * 100
        for k, t in targets.lifts.items():
            r += 10 * (m[f"lift_{k}"] - t) ** 2 * 100
        r += 20 * (m["precision"][k_top]
                   - targets.precision[k_top]) ** 2 * 100
        for k, t in targets.capture.items():
            if k == k_top:
                continue
            w = 6 if abs(k - 0.10) < 1e-9 else 4
            r += w * (m["capture"][k] - t) ** 2 * 100
        return r

    res = minimize(
        objective, _pack(targets.warm_start), method="Nelder-Mead",
        options=dict(maxiter=NM_MAX_ITER, maxfev=NM_MAX_ITER,
                     xatol=1e-5, fatol=1e-9))
    params = _unpack(res.x)
    achieved = evaluate(params, cohort, targets)

    failures = []
    if abs(achieved["auroc"] - targets.auroc) > TOL_AUROC:
        failures.append(
            f"AUROC {achieved['auroc']:.4f} vs {targets.auroc} "
            f"(tol {TOL_AUROC})")
    for k, t in targets.lifts.items():
        if abs(achieved[f"lift_{k}"] - t) > TOL_LIFT:
            failures.append(
                f"lift_{k} {achieved[f'lift_{k}']:.3f} vs {t} "
                f"(tol {TOL_LIFT})")
    if failures:
        residuals = {
            "auroc": achieved["auroc"] - targets.auroc,
            **{f"lift_{k}": achieved[f"lift_{k}"] - t
               for k, t in targets.lifts.items()},
        }
        table = "\n".join(
            f"  {k}: residual {v:+.4f}" for k, v in residuals.items())
        raise CalibrationError(
            "Subgroup calibration failed the gate:\n"
            + "\n".join(f"  FAIL {f}" for f in failures)
            + f"\nresidual table:\n{table}")

    # Probability map: anchor the flag rate at the deployed threshold,
    # WITH the default severity gradient applied (the deployed scorer
    # runs with it; anchoring without it would shift the flag rate).
    from .scenario import MalnutritionConfig as _MC
    g_default = _MC().severity_score_gradient
    z = _raw_scores(params, cohort, gradient=g_default)
    params.map_a = 1.5 / float(z.std())
    z_q = float(np.quantile(z, 1.0 - targets.flag_rate_anchor))
    thr = targets.threshold_anchor
    logit_thr = float(np.log(thr / (1 - thr)))
    params.map_b = logit_thr - params.map_a * z_q

    return params, achieved


class FrozenSubgroupScorer:
    """Loads frozen params and scores scenario state deterministically
    in (current truth, the three per-entity latents, params)."""

    def __init__(self, params: ScorerParams, achieved: Dict):
        self.params = params
        self.achieved = achieved

    @classmethod
    def load(cls, path: str) -> "FrozenSubgroupScorer":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"Frozen model params not found at {p}. Run "
                "`python -m healthcare_sim_sdk.scenarios."
                "malnutrition_prioritization.run_verification "
                "--stage A` to calibrate and freeze them (requires "
                "configs/model_targets.yaml -- see "
                "model_targets.example.yaml). No silent defaults.")
        raw = yaml.safe_load(p.read_text())
        params = ScorerParams(**raw["params"])
        return cls(params, raw.get("achieved", {}))

    def score(self, state: np.ndarray,
              degenerate: bool = False,
              severity_gradient: float = 0.0,
              moderate_fraction: float = 0.6) -> np.ndarray:
        """Score scenario state. degenerate=True zeroes the signal
        component (AUROC ~0.5) for the verification-only chance-level
        run. severity_gradient adds the mean-preserving within-positive
        severity shift (reads CURRENT severity -- worsening cases score
        higher, which is what creates realistic early-vs-late
        confounding)."""
        p = self.params
        y = state[ROW_TRUE_MALNOURISHED]
        d_eff, sigma = _components(
            p,
            state[ROW_DEMO_SEX] == 1,
            state[ROW_DEMO_RACE] == 1,
            state[ROW_DEMO_ETHNICITY] == 1,
            state[ROW_EPS_MIX],
            state[ROW_EPS_HARD])
        if degenerate:
            d_eff = np.zeros_like(d_eff)
        z = d_eff * y + sigma * state[ROW_EPS_LOGIT]
        if severity_gradient:
            from .scenario import ROW_TRUE_SEVERITY, SEV_SEVERE
            z = z + _gradient_term(
                y, state[ROW_TRUE_SEVERITY] == SEV_SEVERE,
                severity_gradient, moderate_fraction)
        return _sigmoid(p.map_a * z + p.map_b)


def freeze(params: ScorerParams, achieved: Dict,
           targets: ModelTargets, path: str) -> None:
    """Write model_frozen.yaml (gitignored -- encodes the
    institution's model targets)."""
    header = (
        "# GENERATED by run_verification.py Stage A. GITIGNORED --\n"
        "# encodes institution-specific model targets; do not commit.\n"
        f"# Calibration: seed={CALIBRATION_SEED}, n={CALIBRATION_N}, "
        f"prevalence={targets.calibration_prevalence}\n")
    flat_achieved = {
        "auroc": achieved["auroc"],
        "auprc": achieved["auprc"],
        **{f"lift_{k}": achieved[f"lift_{k}"]
           for k in ("female", "black", "hl", "male")},
    }
    body = yaml.safe_dump(
        {
            "params": {k: float(v)
                       for k, v in vars(params).items()},
            "achieved": {k: float(v)
                         for k, v in flat_achieved.items()},
            "targets": {
                "auroc": targets.auroc,
                **{f"lift_{k}": v
                   for k, v in targets.lifts.items()},
            },
            "calibration": {
                "seed": CALIBRATION_SEED,
                "n": CALIBRATION_N,
                "prevalence": targets.calibration_prevalence,
            },
        },
        sort_keys=True)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(header + body)
