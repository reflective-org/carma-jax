"""Bin-by-bin relative error: JAX vs Fortran sulfate ensemble.

Two panels — mass and number — with a threshold to separate physically
meaningful bins from near-empty ones:

  Active (colored):   mass  > 1e-20 g/m²   OR   number > 1e-2 #/m²
  Empty  (gray):      below threshold — shown but de-emphasized

Column densities are computed as:
    mass_g_per_m2   = pc_mmr [g/g] × (p [hPa] × 100 / 9.81 × 1000) [g/m²]
    number_per_m2   = (pc_mmr / rmass_bin) × air_col_g_per_m2

Usage:
    python scripts/plot_binbybin_parity.py \
        --fortran data/sulfate_fortran_outputs_fresh.npz \
        --jax     data/sulfate_jax_outputs_fresh.npz \
        --scenarios data/sulfate_scenarios_1000.npz
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
sys.path.insert(0, str(ROOT / "scripts"))


def bin_diameters_nm(nbin=38, rmin_cm=2e-8, rmrat=2.0, rho=1.78):
    vmin = (4.0 / 3.0) * math.pi * rmin_cm**3 * rho
    rmass = vmin * rmrat ** np.arange(nbin)
    r = (3.0 * rmass / (4.0 * math.pi * rho)) ** (1.0 / 3.0)
    return 2.0 * r * 1e7   # cm → nm


def plot_panel(ax, d_nm, rel_F, rel_J, active_F,
               ylabel, title, threshold_label):
    """Draw one relative-error panel (mass or number)."""
    rng = np.random.default_rng(0)
    n, nbin = rel_F.shape

    # active = F is above threshold for a given (scenario, bin)
    active = active_F   # use Fortran as reference for "populated"
    inactive = ~active

    # Scatter: inactive (gray) first, active (blue) on top
    for b in range(nbin):
        # Relative error for each scenario at this bin
        denom = np.maximum(np.abs(rel_F[:, b]), np.abs(rel_J[:, b]))
        denom = np.where(denom > 1e-300, denom, 1.0)
        err = np.abs(rel_F[:, b] - rel_J[:, b]) / denom

        jitter = rng.normal(0, 0.07, size=n)
        xs = d_nm[b] * np.exp(jitter)

        # Inactive scenarios
        if inactive[:, b].any():
            ax.scatter(xs[inactive[:, b]], np.clip(err[inactive[:, b]], 1e-16, 1),
                       s=2, alpha=0.15, color="0.75", edgecolors="none", zorder=1)

        # Active scenarios
        if active[:, b].any():
            ax.scatter(xs[active[:, b]], np.clip(err[active[:, b]], 1e-16, 1),
                       s=3, alpha=0.25, color="steelblue", edgecolors="none", zorder=2)

    # Box plot — active pairs only
    box_data = []
    for b in range(nbin):
        denom = np.maximum(np.abs(rel_F[:, b]), np.abs(rel_J[:, b]))
        denom = np.where(denom > 1e-300, denom, 1.0)
        err = np.abs(rel_F[:, b] - rel_J[:, b]) / denom
        vals = err[active[:, b]]
        box_data.append(np.clip(vals, 1e-16, 1) if vals.size else np.array([1e-16]))

    bp = ax.boxplot(box_data, positions=d_nm, widths=d_nm * 0.13,
                    showfliers=False, patch_artist=True,
                    medianprops=dict(color="crimson", lw=1.5),
                    boxprops=dict(facecolor="white", alpha=0.85, edgecolor="black"),
                    whiskerprops=dict(color="black", lw=0.8),
                    capprops=dict(color="black", lw=0.8),
                    zorder=3)

    ax.axhline(0.01, color="green",  ls="--", lw=1.2, alpha=0.8, label="1% threshold")
    ax.axhline(0.05, color="orange", ls="--", lw=1.2, alpha=0.8, label="5% threshold")

    # Dummy handles for legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [
        Line2D([0], [0], color="green",  ls="--", lw=1.2, label="1%"),
        Line2D([0], [0], color="orange", ls="--", lw=1.2, label="5%"),
        ax.scatter([], [], s=8, color="steelblue", alpha=0.6,
                   label=f"active (≥ {threshold_label})"),
        ax.scatter([], [], s=8, color="0.75", alpha=0.6,
                   label=f"empty  (< {threshold_label})"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="lower right")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(5e-17, 2.0)
    ax.xaxis.set_minor_locator(LogLocator(numticks=15))
    ax.yaxis.set_minor_locator(LogLocator(numticks=15))
    ax.grid(True, alpha=0.2, which="both")
    ax.set_xlabel("Bin median diameter [nm]", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=10)

    # Summary stats on active pairs
    all_err_active = []
    for b in range(nbin):
        denom = np.maximum(np.abs(rel_F[:, b]), np.abs(rel_J[:, b]))
        denom = np.where(denom > 1e-300, denom, 1.0)
        err = np.abs(rel_F[:, b] - rel_J[:, b]) / denom
        all_err_active.extend(err[active[:, b]].tolist())
    all_err_active = np.array(all_err_active)
    n_active = active.sum()
    pct_pass = (all_err_active < 0.01).mean() * 100

    ax.text(0.02, 0.97,
            f"Active pairs: {n_active:,} / {n*nbin:,}\n"
            f"Median err: {np.median(all_err_active):.1e}\n"
            f"< 1% : {pct_pass:.0f}%",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=8, family="monospace",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="0.8"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fortran",   type=Path, default=ROOT/"data"/"sulfate_fortran_outputs_fresh.npz")
    p.add_argument("--jax",       type=Path, default=ROOT/"data"/"sulfate_jax_outputs_fresh.npz")
    p.add_argument("--scenarios", type=Path, default=ROOT/"data"/"sulfate_scenarios_1000.npz")
    args = p.parse_args()

    F    = np.load(args.fortran)
    J    = np.load(args.jax)
    scen = np.load(args.scenarios)

    pc_F = F["pc_final"]   # (N, NBIN) g/g per bin
    pc_J = J["pc_final"]
    n, nbin = pc_F.shape

    # Load bin geometry from config
    from jax_ensemble import _minimal_config
    cfg = _minimal_config()
    rmass = np.asarray(cfg.groups[0].rmass)   # g per particle
    d_nm = bin_diameters_nm(nbin)

    # Air column mass per scenario [g/m²]
    g_accel = 9.81   # m/s²
    air_col = scen["p"] * 100.0 / g_accel * 1000.0   # (N,) g/m²

    # Column mass [g/m²] and number [#/m²] per (scenario, bin)
    mass_F = pc_F * air_col[:, None]                   # (N, NBIN)
    mass_J = pc_J * air_col[:, None]
    num_F  = pc_F / rmass[None, :] * air_col[:, None]  # (N, NBIN)
    num_J  = pc_J / rmass[None, :] * air_col[:, None]

    MASS_FLOOR = 1e-20   # g/m²
    NUM_FLOOR  = 1e-2    # #/m²

    active_mass = mass_F > MASS_FLOOR
    active_num  = num_F  > NUM_FLOOR

    print(f"N={n}, NBIN={nbin}")
    print(f"Active pairs — mass > {MASS_FLOOR:.0e} g/m²:   {active_mass.sum():,} / {n*nbin}")
    print(f"Active pairs — num  > {NUM_FLOOR:.0e} #/m²:  {active_num.sum():,} / {n*nbin}")

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    plot_panel(
        axes[0], d_nm,
        mass_F, mass_J, active_mass,
        ylabel="|JAX − Fortran| / max(|JAX|, |Fortran|)",
        title=f"Mass per bin — relative error\n"
              f"(active = column mass > {MASS_FLOOR:.0e} g/m²)",
        threshold_label=f"{MASS_FLOOR:.0e} g/m²",
    )

    plot_panel(
        axes[1], d_nm,
        num_F, num_J, active_num,
        ylabel="|JAX − Fortran| / max(|JAX|, |Fortran|)",
        title=f"Number per bin — relative error\n"
              f"(active = column number > {NUM_FLOOR:.0e} #/m²)",
        threshold_label=f"{NUM_FLOOR:.0e} #/m²",
    )

    plt.suptitle(
        "CARMA-JAX sulfate ensemble: JAX vs Fortran — bin-by-bin relative error\n"
        "(1800 s × 100 steps, NBIN=38, rmin=0.2 nm, rmrat=2.0, N=1000 scenarios)\n"
        "Box = P25/median/P75 over active scenarios  |  steelblue = active  |  gray = empty",
        fontsize=11, fontweight="bold", y=1.03,
    )

    out_dir = ROOT / "plots" / "diff" / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sulfate_parity_binbybin.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
