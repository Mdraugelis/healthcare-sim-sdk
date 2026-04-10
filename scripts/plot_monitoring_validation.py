"""Generate monitoring validation figures for the report.

Produces:
- Figure 1: Cross-regime detection heatmap (6x4 matrix)
- Figures 2-7: Per-regime 3-panel plots showing ground truth,
  Tier 2 CUSUM, and Tier 3 quarterly estimates across 30 seeds

Usage:
    python scripts/plot_monitoring_validation.py

Output: reports/monitoring_validation/figures/*.png
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

SWEEP_DIR = Path("outputs/monitoring_validation_sweep")
FIG_DIR = Path("reports/monitoring_validation/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

REGIMES = [
    ("calibrated_success", "A: Calibrated Success"),
    ("null_program", "B: Null Program"),
    ("gradual_decay", "C: Gradual Decay"),
    ("capacity_collapse", "D: Capacity Collapse"),
    ("model_drift", "E: Model Drift"),
    ("partial_adoption", "F: Partial Adoption"),
]

TIER_LABELS = ["Tier 1\nShewhart", "Tier 2\nCUSUM", "Tier 3\nCITS", "Tier 4\nModel"]


def load_all_runs():
    """Load metrics and monitoring runs for all cells."""
    data = defaultdict(list)
    for cell in sorted(SWEEP_DIR.glob("*_seed*")):
        mf = cell / "metrics.json"
        rf = cell / "monitoring_run.json"
        gf = cell / "ground_truth.npz"
        if not mf.exists():
            continue
        m = json.loads(mf.read_text())
        regime = m["regime"]

        entry = {"metrics": m, "cell_dir": cell}

        if rf.exists():
            entry["run"] = json.loads(rf.read_text())
        if gf.exists():
            gt = np.load(gf)
            entry["gt"] = {k: gt[k] for k in gt.files}

        data[regime].append(entry)
    return data


def fig1_detection_heatmap(data):
    """6x4 heatmap: regime rows, tier columns, colored by rate."""
    matrix = np.zeros((6, 4))
    annotations = [[""] * 4 for _ in range(6)]

    tier_keys = [
        "tier1_first_detection_week",
        "tier2_first_detection_week",
        "tier3_first_detection_week",
        "tier4_first_detection_week",
    ]

    for i, (regime_key, _) in enumerate(REGIMES):
        cells = data[regime_key]
        n = len(cells)
        for j, tk in enumerate(tier_keys):
            weeks = [
                c["metrics"][tk]
                for c in cells
                if c["metrics"][tk] is not None
            ]
            rate = len(weeks) / n if n > 0 else 0
            matrix[i, j] = rate
            if weeks:
                med = int(np.median(weeks))
                annotations[i][j] = f"{len(weeks)}/{n}\n@wk{med}"
            else:
                annotations[i][j] = "—"

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(
        matrix, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto",
    )

    ax.set_xticks(range(4))
    ax.set_xticklabels(TIER_LABELS, fontsize=10)
    ax.set_yticks(range(6))
    ax.set_yticklabels(
        [label for _, label in REGIMES], fontsize=10,
    )

    for i in range(6):
        for j in range(4):
            color = "white" if matrix[i, j] > 0.5 else "black"
            ax.text(
                j, i, annotations[i][j],
                ha="center", va="center",
                fontsize=8, color=color, fontweight="bold",
            )

    ax.set_title(
        "Detection Rate by Regime and Tier\n"
        "(N/30 seeds detected, median detection week)",
        fontsize=12, fontweight="bold",
    )
    plt.colorbar(im, ax=ax, label="Detection Rate", shrink=0.8)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig1_detection_heatmap.png", dpi=150)
    plt.close(fig)
    print("  Saved fig1_detection_heatmap.png")


def _recompute_cusum(turnover_rates, baseline_weeks=8):
    """Recompute upper CUSUM from weekly turnover rates."""
    rates = np.array(turnover_rates)
    if len(rates) < baseline_weeks:
        return np.zeros(len(rates))

    baseline = rates[:baseline_weeks]
    mu0 = np.mean(baseline)
    sigma = np.std(baseline, ddof=1)
    if sigma < 1e-9:
        sigma = 1e-6
    k = 0.5 * sigma

    cusum = np.zeros(len(rates))
    c_upper = 0.0
    for i in range(baseline_weeks, len(rates)):
        c_upper = max(0.0, c_upper + (rates[i] - mu0) - k)
        cusum[i] = c_upper
    return cusum


def fig_per_regime(data, regime_key, regime_label, fig_num):
    """3-panel figure for one regime across 30 seeds."""
    cells = data[regime_key]
    if not cells:
        return

    n_weeks = 104
    weeks = np.arange(n_weeks)

    # Collect trajectories across seeds
    f_departures = []
    cf_departures = []
    f_retention = []
    cf_retention = []
    turnover_rates = []
    cusum_trajectories = []
    t3_estimates = []

    for cell in cells:
        if "gt" in cell:
            fd = cell["gt"]["factual_departures"]
            cd = cell["gt"]["cf_departures"]
            fr = cell["gt"]["factual_retention"]
            cr = cell["gt"]["cf_retention"]
            f_departures.append(fd[:n_weeks])
            cf_departures.append(cd[:n_weeks])
            f_retention.append(fr[:n_weeks])
            cf_retention.append(cr[:n_weeks])

        if "run" in cell:
            wh = cell["run"]["weekly_history"]
            rates = [
                r["unit_turnover_rate"] for r in wh[:n_weeks]
            ]
            turnover_rates.append(rates)
            cusum_trajectories.append(
                _recompute_cusum(rates),
            )
            t3_estimates.append(
                cell["run"].get("tier3_estimates", []),
            )

    if not f_departures:
        return

    f_dep = np.array(f_departures)
    cf_dep = np.array(cf_departures)
    f_ret = np.array(f_retention) * 100
    cf_ret = np.array(cf_retention) * 100
    tr = np.array(turnover_rates)
    cusums = np.array(cusum_trajectories)

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    fig.suptitle(
        f"Regime {regime_label}\n(30 seeds, shaded = IQR)",
        fontsize=14, fontweight="bold", y=0.98,
    )

    # ── Panel A: Cumulative departures + retention ──
    ax = axes[0]
    # Factual band
    f_med = np.median(f_dep, axis=0)
    f_lo = np.percentile(f_dep, 25, axis=0)
    f_hi = np.percentile(f_dep, 75, axis=0)
    ax.fill_between(weeks, f_lo, f_hi, alpha=0.2, color="C0")
    ax.plot(
        weeks, f_med, color="C0", linewidth=2,
        label="Factual (AI+SOC)",
    )
    # CF band
    cf_med = np.median(cf_dep, axis=0)
    cf_lo = np.percentile(cf_dep, 25, axis=0)
    cf_hi = np.percentile(cf_dep, 75, axis=0)
    ax.fill_between(weeks, cf_lo, cf_hi, alpha=0.2, color="C1")
    ax.plot(
        weeks, cf_med, color="C1", linewidth=2,
        linestyle="--", label="Counterfactual (SOC only)",
    )

    # Final delta annotation
    delta = cf_med[-1] - f_med[-1]
    ax.annotate(
        f"Median saved: {delta:.0f}",
        xy=(n_weeks - 1, f_med[-1]),
        xytext=(n_weeks - 20, f_med[-1] - 30),
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="C0"),
        color="C0", fontweight="bold",
    )

    ax.set_ylabel("Cumulative Departures", fontsize=11)
    ax.legend(loc="upper left", fontsize=10)
    ax.set_title("Panel A: Ground Truth Trajectories", fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add retention on secondary axis
    ax2 = ax.twinx()
    ax2.plot(
        weeks, np.median(f_ret, axis=0),
        color="C0", linewidth=1, alpha=0.5, linestyle=":",
    )
    ax2.plot(
        weeks, np.median(cf_ret, axis=0),
        color="C1", linewidth=1, alpha=0.5, linestyle=":",
    )
    ax2.set_ylabel("Retention Rate (%)", fontsize=9, alpha=0.6)
    ax2.set_ylim(30, 105)

    # ── Panel B: Weekly turnover rate + CUSUM ──
    ax = axes[1]
    tr_med = np.median(tr, axis=0)
    tr_lo = np.percentile(tr, 25, axis=0)
    tr_hi = np.percentile(tr, 75, axis=0)
    ax.fill_between(weeks, tr_lo, tr_hi, alpha=0.2, color="C2")
    ax.plot(
        weeks, tr_med, color="C2", linewidth=1.5,
        label="Weekly turnover rate",
    )
    ax.set_ylabel("Weekly Turnover Rate", fontsize=11, color="C2")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    # CUSUM on secondary axis
    ax3 = ax.twinx()
    c_med = np.median(cusums, axis=0)
    c_lo = np.percentile(cusums, 25, axis=0)
    c_hi = np.percentile(cusums, 75, axis=0)
    ax3.fill_between(
        weeks, c_lo, c_hi, alpha=0.15, color="C3",
    )
    ax3.plot(
        weeks, c_med, color="C3", linewidth=1.5,
        label="CUSUM (upper)",
    )

    # Compute h threshold from baseline
    baseline_rates = tr[:, :8]
    sigma_baseline = np.mean(np.std(baseline_rates, axis=1, ddof=1))
    h = 4.0 * sigma_baseline
    if h > 0:
        ax3.axhline(
            y=h, color="C3", linestyle="--",
            alpha=0.6, label=f"h = 4σ = {h:.4f}",
        )
    ax3.set_ylabel("CUSUM Statistic", fontsize=9, color="C3")
    ax3.legend(loc="upper right", fontsize=9)
    ax.set_title(
        "Panel B: Tier 2 — Weekly Turnover Rate + CUSUM",
        fontsize=11,
    )

    # Mark detection events (Tier 1 and Tier 2) from a representative
    # seed
    for cell in cells[:1]:
        if "run" not in cell:
            continue
        for ev in cell["run"]["detection_events"]:
            if ev["tier"] == 1:
                ax.axvline(
                    x=ev["week"], color="red",
                    alpha=0.7, linewidth=1.5, linestyle="-.",
                )
                ax.text(
                    ev["week"], ax.get_ylim()[1] * 0.95,
                    "T1", fontsize=8, color="red",
                    ha="center", fontweight="bold",
                )
                break  # just mark the first T1

    # ── Panel C: Tier 3 quarterly estimates ──
    ax = axes[2]

    # Plot all 30 seeds' Tier 3 estimates
    for i, est_list in enumerate(t3_estimates):
        for est in est_list:
            color = "C0" if est["p_value"] < 0.05 else "C7"
            marker = "o" if est["p_value"] < 0.05 else "."
            size = 8 if est["p_value"] < 0.05 else 4
            ax.errorbar(
                est["week"] + i * 0.15 - 2.25,
                est["effect_estimate"],
                yerr=[
                    [est["effect_estimate"] - est["ci_lower"]],
                    [est["ci_upper"] - est["effect_estimate"]],
                ],
                fmt=marker, markersize=size, color=color,
                ecolor=color, alpha=0.4, elinewidth=0.5,
                capsize=0,
            )

    ax.axhline(y=0, color="black", linewidth=1, linestyle="-")
    ax.set_ylabel("Effect Estimate (turnover rate Δ)", fontsize=11)
    ax.set_xlabel("Simulation Week", fontsize=11)
    ax.set_title(
        "Panel C: Tier 3 — Quarterly CITS Estimates "
        "(blue=significant, gray=not)",
        fontsize=11,
    )
    ax.grid(True, alpha=0.3)

    # Regime-specific annotations
    if regime_key == "capacity_collapse":
        for a in axes:
            a.axvline(x=30, color="red", linestyle=":", alpha=0.7)
        axes[0].text(
            31, axes[0].get_ylim()[1] * 0.5,
            "Capacity\ncollapses", fontsize=9,
            color="red", fontweight="bold",
        )
    elif regime_key == "gradual_decay":
        for a in axes:
            a.axvspan(26, 52, alpha=0.08, color="orange")
        axes[0].text(
            39, axes[0].get_ylim()[1] * 0.15,
            "Decay ramp", fontsize=9, color="orange",
            ha="center", fontweight="bold",
        )
    elif regime_key == "model_drift":
        for a in axes:
            a.axvspan(20, 40, alpha=0.08, color="purple")
        axes[0].text(
            30, axes[0].get_ylim()[1] * 0.15,
            "AUC drift", fontsize=9, color="purple",
            ha="center", fontweight="bold",
        )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fname = f"fig{fig_num}_{regime_key}.png"
    fig.savefig(FIG_DIR / fname, dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


def fig8_departures_prevented_boxplot(data):
    """Box plot of departures prevented across regimes."""
    fig, ax = plt.subplots(figsize=(10, 5))

    positions = []
    labels = []
    box_data = []

    for i, (regime_key, regime_label) in enumerate(REGIMES):
        cells = data[regime_key]
        saved = [c["metrics"]["departures_prevented"] for c in cells]
        box_data.append(saved)
        positions.append(i)
        labels.append(regime_label.split(":")[1].strip())

    bp = ax.boxplot(
        box_data, positions=positions, widths=0.6,
        patch_artist=True, showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="red", markersize=6),
    )

    colors = ["#4CAF50", "#9E9E9E", "#FF9800", "#F44336", "#9C27B0", "#2196F3"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    ax.axhline(y=0, color="black", linewidth=1, linestyle="-")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=10, rotation=15, ha="right")
    ax.set_ylabel("Departures Prevented\n(Factual - Counterfactual)", fontsize=11)
    ax.set_title(
        "AI Marginal Value Over Standard-of-Care\n"
        "(30 seeds per regime, diamond = mean)",
        fontsize=12, fontweight="bold",
    )
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate means
    for i, bd in enumerate(box_data):
        mean_val = np.mean(bd)
        ax.text(
            i, max(bd) + 3, f"μ={mean_val:.1f}",
            ha="center", fontsize=9, fontweight="bold",
        )

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig8_departures_prevented.png", dpi=150)
    plt.close(fig)
    print("  Saved fig8_departures_prevented.png")


def fig9_tier1_capacity_collapse_detail(data):
    """Detail plot showing Tier 1 catching the capacity drop."""
    cells = data["capacity_collapse"]
    if not cells:
        return

    fig, ax = plt.subplots(figsize=(10, 4))

    for cell in cells:
        if "run" not in cell:
            continue
        wh = cell["run"]["weekly_history"]
        weeks = [r["week"] for r in wh]
        counts = [r["check_ins_done_this_week"] for r in wh]
        ax.plot(weeks, counts, color="C0", alpha=0.15, linewidth=0.8)

    # Median line
    all_counts = []
    for cell in cells:
        if "run" not in cell:
            continue
        wh = cell["run"]["weekly_history"]
        all_counts.append([r["check_ins_done_this_week"] for r in wh])
    counts_arr = np.array(all_counts)
    med = np.median(counts_arr, axis=0)
    ax.plot(range(len(med)), med, color="C0", linewidth=2.5, label="Median (30 seeds)")

    # Mark the collapse
    ax.axvline(x=30, color="red", linewidth=2, linestyle="--", label="Capacity collapse (wk 30)")

    # Approximate control limits from baseline
    baseline_med = np.median(counts_arr[:, :8], axis=0)
    xbar = np.mean(baseline_med)
    mr = np.abs(np.diff(baseline_med))
    sigma = np.mean(mr) / 1.128 if len(mr) > 0 else 1
    lcl = xbar - 3 * sigma

    ax.axhline(y=xbar, color="green", linewidth=1, linestyle="-", alpha=0.6, label=f"Baseline mean ({xbar:.0f})")
    ax.axhline(y=lcl, color="green", linewidth=1, linestyle="--", alpha=0.6, label=f"LCL ({lcl:.0f})")

    ax.set_xlabel("Simulation Week", fontsize=11)
    ax.set_ylabel("Weekly Check-Ins (raw count)", fontsize=11)
    ax.set_title(
        "Regime D: Tier 1 Detects Capacity Collapse\n"
        "Check-in count drops from ~60 to ~20 at week 30",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 104)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig9_tier1_capacity_collapse.png", dpi=150)
    plt.close(fig)
    print("  Saved fig9_tier1_capacity_collapse.png")


def main():
    print("Loading sweep data...")
    data = load_all_runs()
    print(f"  Loaded {sum(len(v) for v in data.values())} cells across {len(data)} regimes")
    print()

    print("Generating figures...")
    fig1_detection_heatmap(data)
    for i, (regime_key, regime_label) in enumerate(REGIMES):
        fig_per_regime(data, regime_key, regime_label, i + 2)
    fig8_departures_prevented_boxplot(data)
    fig9_tier1_capacity_collapse_detail(data)

    print()
    print(f"All figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
