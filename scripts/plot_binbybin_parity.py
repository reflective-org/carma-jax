"""Bin-by-bin parity plot: JAX vs Fortran sulfate ensemble.

Produces three panels:
1. Per-bin box plot of relative error across active scenarios
2. Per-scenario median rel-err CDF (the Phase 10 gate metric)
3. Timing + summary statistics table

Usage:
    python scripts/plot_binbybin_parity.py \
        --fortran data/sulfate_fortran_outputs_fresh.npz \
        --jax     data/sulfate_jax_outputs_fresh.npz \
        --timing-fortran 17.3 \
        --timing-jax     2262.5
"""
import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))


def bin_diameters_nm(nbin=38, rmin_cm=2e-8, rmrat=2.0, rho=1.78):
    vmin = (4.0 / 3.0) * math.pi * rmin_cm**3 * rho
    rmass = vmin * rmrat ** np.arange(nbin)
    r = (3.0 * rmass / (4.0 * math.pi * rho)) ** (1.0 / 3.0)
    return 2.0 * r * 1e7   # cm → nm


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fortran", type=Path,
                   default=ROOT / "data" / "sulfate_fortran_outputs_fresh.npz")
    p.add_argument("--jax", type=Path,
                   default=ROOT / "data" / "sulfate_jax_outputs_fresh.npz")
    p.add_argument("--timing-fortran", type=float, default=None)
    p.add_argument("--timing-jax",     type=float, default=None)
    args = p.parse_args()

    F = np.load(args.fortran)
    J = np.load(args.jax)
    pc_F = F["pc_final"]   # (N, NBIN) g/g per bin
    pc_J = J["pc_final"]
    n, nbin = pc_F.shape
    d_nm = bin_diameters_nm(nbin)

    denom = np.maximum(np.abs(pc_F), np.abs(pc_J))
    denom = np.where(denom > 1e-300, denom, 1.0)
    rel = np.abs(pc_F - pc_J) / denom   # (N, NBIN)

    # "Active" = bin where Fortran has meaningful signal
    active_floor = 1e-50   # g/g — below this Fortran considers bin empty
    active = pc_F > active_floor   # (N, NBIN)

    # Per-scenario median rel err across all 38 bins (Phase 10 gate metric)
    per_scen_med = np.median(rel, axis=1)   # (N,)
    pass1  = (per_scen_med < 0.01).mean() * 100
    pass5  = (per_scen_med < 0.05).mean() * 100
    pass10 = (per_scen_med < 0.10).mean() * 100

    print(f"N={n}, NBIN={nbin}")
    print(f"Per-scenario median rel err P50={np.percentile(per_scen_med,50):.3e}  "
          f"P90={np.percentile(per_scen_med,90):.3e}")
    print(f"Pass rates: <1%={pass1:.1f}%  <5%={pass5:.1f}%  <10%={pass10:.1f}%")

    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # --- Panel 1: per-bin box plot (active scenarios only per bin) ---
    ax = axes[0]
    rng = np.random.default_rng(42)
    data_for_box = []
    for b in range(nbin):
        mask = active[:, b]
        vals = rel[mask, b] if mask.any() else np.array([1e-16])
        vals = np.where(vals > 1e-16, vals, 1e-16)
        data_for_box.append(vals)
        # jitter scatter
        jitter = rng.normal(0, 0.06, size=vals.shape)
        xs = d_nm[b] * np.exp(jitter)
        ax.scatter(xs, vals, s=2, alpha=0.10, color="steelblue",
                   edgecolors="none", zorder=1)

    ax.boxplot(data_for_box, positions=d_nm, widths=d_nm * 0.14,
               showfliers=False, patch_artist=True,
               medianprops=dict(color="crimson", lw=1.5),
               boxprops=dict(facecolor="white", alpha=0.8, edgecolor="black"),
               whiskerprops=dict(color="black", lw=0.9), zorder=3)

    ax.axhline(0.01, color="green",  ls="--", lw=1.2, alpha=0.8, label="1% threshold")
    ax.axhline(0.05, color="orange", ls="--", lw=1.2, alpha=0.8, label="5% threshold")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.xaxis.set_minor_locator(LogLocator(numticks=15))
    ax.yaxis.set_minor_locator(LogLocator(numticks=15))
    ax.grid(True, alpha=0.25, which="both")
    ax.set_xlabel("Bin median diameter [nm]", fontsize=11)
    ax.set_ylabel("|JAX − Fortran| / max(|JAX|, |Fortran|)", fontsize=11)
    active_counts = active.sum(axis=0)
    ax.set_title(f"Per-bin relative error  (box = P25/median/P75)\n"
                 f"n per bin = {active_counts.min()}–{active_counts.max()} "
                 f"active scenarios of {n} total", fontsize=10)
    ax.legend(fontsize=9, loc="upper left")

    # --- Panel 2: per-scenario median CDF ---
    ax2 = axes[1]
    sorted_med = np.sort(per_scen_med)
    cdf = np.arange(1, n + 1) / n * 100
    ax2.semilogx(sorted_med, cdf, "b-", lw=2, label="CDF")
    ax2.axvline(0.01, color="green",  ls="--", lw=1.2,
                label=f"1%  → {pass1:.0f}% scen pass")
    ax2.axvline(0.05, color="orange", ls="--", lw=1.2,
                label=f"5%  → {pass5:.0f}% scen pass")
    ax2.axvline(0.10, color="red",    ls=":",  lw=1.2,
                label=f"10% → {pass10:.0f}% scen pass")
    ax2.set_xlabel("Per-scenario median rel err (all 38 bins)", fontsize=11)
    ax2.set_ylabel("Scenarios ≤ threshold [%]", fontsize=11)
    ax2.set_title("Per-scenario parity CDF\n"
                  "(Phase 10 gate metric: fraction with median < 1%)", fontsize=10)
    ax2.set_ylim(0, 105); ax2.grid(True, alpha=0.3, which="both")
    ax2.xaxis.set_minor_locator(LogLocator(numticks=15))
    ax2.legend(fontsize=9, loc="upper left")

    # Annotate Phase 10 baseline
    ax2.annotate("Phase 10 baseline\n(adaptive): 66.8%",
                 xy=(0.01, 66.8), xytext=(0.001, 50),
                 arrowprops=dict(arrowstyle="->", color="gray"),
                 fontsize=8, color="gray")

    # --- Panel 3: summary table ---
    ax3 = axes[2]
    ax3.axis("off")
    rows = [
        ["Metric", "Value"],
        ["N scenarios", f"{n:,}"],
        ["NBIN", f"{nbin}"],
        ["Outer steps", "100"],
        ["dtime", "1800 s"],
        ["", ""],
        ["Per-scenario median", ""],
        ["  P50", f"{np.percentile(per_scen_med,50):.2e}"],
        ["  P90", f"{np.percentile(per_scen_med,90):.2e}"],
        ["  P95", f"{np.percentile(per_scen_med,95):.2e}"],
        ["  P99", f"{np.percentile(per_scen_med,99):.2e}"],
        ["", ""],
        ["Pass rate", ""],
        ["  < 1% rel err",  f"{pass1:.1f}%"],
        ["  < 5% rel err",  f"{pass5:.1f}%"],
        ["  < 10% rel err", f"{pass10:.1f}%"],
    ]
    if args.timing_fortran is not None and args.timing_jax is not None:
        t_F, t_J = args.timing_fortran, args.timing_jax
        rows += [
            ["", ""],
            ["Timing (1000 scenarios)", ""],
            ["  Fortran", f"{t_F:.0f} s  ({t_F/n*1000:.0f} ms/scen)"],
            ["  JAX (CPU, float64)", f"{t_J:.0f} s  ({t_J/n*1000:.0f} ms/scen)"],
            ["  Ratio (JAX/F)", f"{t_J/t_F:.0f}×"],
        ]

    table = ax3.table(cellText=rows, loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.55)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        if r == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#eef2f7")
        else:
            cell.set_facecolor("white")
    ax3.set_title("Summary", fontsize=10)

    plt.suptitle(
        "CARMA-JAX sulfate ensemble parity: JAX vs Fortran\n"
        "(1800 s × 100 steps, NBIN=38, rmin=0.2 nm, rmrat=2.0 — 1000 scenarios)",
        fontsize=12, fontweight="bold", y=1.02,
    )

    out_dir = ROOT / "plots" / "diff" / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sulfate_parity_binbybin.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
