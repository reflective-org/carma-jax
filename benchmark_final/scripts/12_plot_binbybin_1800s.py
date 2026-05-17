"""Bin-by-bin parity at 1800 s outer timestep (adaptive JAX vs Fortran).

Note: at 1800 s, some scenarios hit Fortran's convergence ceiling
(maxretries=16 × maxsubsteps=32 = 65,536 substeps per outer step) and
produce unreliable outputs. We flag those scenarios and show both
"all" and "converged-only" statistics.

Output: benchmark_final/plots/12_binbybin_1800s.png
"""
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "scripts"))
from jax_ensemble import _minimal_config


def bin_diameters_nm(nbin=38, rmin_cm=2e-8, rmrat=2.0, rho=1.923):
    vmin = (4.0 / 3.0) * math.pi * rmin_cm**3 * rho
    rmass = vmin * rmrat ** np.arange(nbin)
    r = (3.0 * rmass / (4.0 * math.pi * rho)) ** (1.0 / 3.0)
    return 2.0 * r * 1e7


def per_volume(pc_mmr, scens, rmass):
    from carma.constants import R_AIR
    rho_air = scens["p"] * 100.0 * 10.0 / (float(R_AIR) * scens["T"])
    return (pc_mmr * rho_air[:, None] / rmass[None, :],
            pc_mmr * rho_air[:, None] * 1e12)


def panel(ax, F_arr, J_arr, active, ceiling_scens, d_nm, threshold_label,
          title, ylim_low=1e-13):
    rng = np.random.default_rng(0)
    nbin_ = F_arr.shape[1]
    data_for_box = []

    for b in range(nbin_):
        denom = np.maximum(np.abs(F_arr[:, b]), np.abs(J_arr[:, b]))
        denom = np.where(denom > 1e-300, denom, 1.0)
        err = np.abs(F_arr[:, b] - J_arr[:, b]) / denom

        jitter = rng.normal(0, 0.07, size=err.shape)
        xs = d_nm[b] * np.exp(jitter)

        # Inactive scenarios (gray)
        inactive_mask = ~active[:, b]
        if inactive_mask.any():
            ax.scatter(xs[inactive_mask], np.clip(err[inactive_mask], ylim_low, 1),
                       s=4, alpha=0.2, color="0.75", edgecolors="none", zorder=1)

        # Active scenarios — split by convergence status
        active_converged = active[:, b] & ~ceiling_scens
        active_ceiling = active[:, b] & ceiling_scens
        if active_converged.any():
            ax.scatter(xs[active_converged],
                       np.clip(err[active_converged], ylim_low, 1),
                       s=6, alpha=0.55, color="steelblue", edgecolors="none", zorder=2)
        if active_ceiling.any():
            ax.scatter(xs[active_ceiling],
                       np.clip(err[active_ceiling], ylim_low, 1),
                       s=12, alpha=0.85, color="crimson",
                       edgecolors="black", lw=0.4, marker="x",
                       zorder=4)

        # Box plot — converged active only
        vals = err[active_converged]
        data_for_box.append(np.clip(vals, ylim_low, 1) if vals.size
                             else np.array([ylim_low]))

    ax.boxplot(data_for_box, positions=d_nm, widths=d_nm * 0.13,
                showfliers=False, patch_artist=True,
                medianprops=dict(color="darkblue", lw=1.5),
                boxprops=dict(facecolor="white", alpha=0.85, edgecolor="black"),
                whiskerprops=dict(color="black", lw=0.8),
                capprops=dict(color="black", lw=0.8), zorder=3)

    ax.axhline(0.01, color="green",  ls="--", lw=1.2, alpha=0.8)
    ax.axhline(0.05, color="orange", ls="--", lw=1.2, alpha=0.8)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="green",  ls="--", lw=1.2, label="1% threshold"),
        Line2D([0], [0], color="orange", ls="--", lw=1.2, label="5% threshold"),
        ax.scatter([], [], s=10, color="steelblue", alpha=0.7,
                   label=f"converged active (≥ {threshold_label})"),
        ax.scatter([], [], s=16, color="crimson", marker="x",
                   label="Fortran hit retry ceiling"),
        ax.scatter([], [], s=10, color="0.75", alpha=0.5,
                   label=f"empty (< {threshold_label})"),
    ]
    ax.legend(handles=handles, fontsize=8.5, loc="lower right")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(ylim_low, 2.0)
    ax.xaxis.set_minor_locator(LogLocator(numticks=15))
    ax.yaxis.set_minor_locator(LogLocator(numticks=15))
    ax.grid(True, alpha=0.2, which="both")
    ax.set_xlabel("Bin median diameter [nm]", fontsize=11)
    ax.set_ylabel("|JAX − Fortran| / max(|JAX|, |Fortran|)", fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")

    # Stats inset — converged-only stats (excludes ceiling-hit scenarios)
    all_err_converged = []
    all_err_ceiling = []
    for b in range(nbin_):
        denom = np.maximum(np.abs(F_arr[:, b]), np.abs(J_arr[:, b]))
        denom = np.where(denom > 1e-300, denom, 1.0)
        err = np.abs(F_arr[:, b] - J_arr[:, b]) / denom
        active_conv = active[:, b] & ~ceiling_scens
        active_ceil = active[:, b] & ceiling_scens
        all_err_converged.extend(err[active_conv].tolist())
        all_err_ceiling.extend(err[active_ceil].tolist())
    all_err_converged = np.array(all_err_converged)
    all_err_ceiling = np.array(all_err_ceiling)

    info_lines = []
    if all_err_converged.size:
        info_lines.append(
            f"Converged ({all_err_converged.size:,} pairs):\n"
            f"  Median {np.median(all_err_converged):.2e}\n"
            f"  Max    {all_err_converged.max():.2e}")
    if all_err_ceiling.size:
        info_lines.append(
            f"Ceiling-hit ({all_err_ceiling.size:,} pairs):\n"
            f"  Median {np.median(all_err_ceiling):.2e}\n"
            f"  Max    {all_err_ceiling.max():.2e}")
    ax.text(0.02, 0.97, "\n\n".join(info_lines),
            transform=ax.transAxes, va="top", ha="left",
            fontsize=8.5, family="monospace",
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="0.8"))


def main():
    cfg = _minimal_config()
    rmass = np.asarray(cfg.groups[0].rmass)
    d_nm = bin_diameters_nm(cfg.nbin)

    scens = np.load(ROOT / "scenarios" / "realistic_scenarios_100.npz")
    F = np.load(ROOT / "outputs" / "dt1800" / "adaptive" / "fortran_outputs.npz")
    J = np.load(ROOT / "outputs" / "dt1800" / "adaptive" / "jax_outputs.npz")
    # Detect scenarios with mathematically impossible state.
    # gc < 0 is unphysical — the explicit-Euler solver couldn't keep
    # H₂SO₄ vapor non-negative within the retry ceiling, so Fortran's
    # adaptive retry gave up. The output is unreliable for those.
    ceiling_scens = F["gc_h2so4_final"] < 0
    n_ceiling = int(ceiling_scens.sum())
    print(f"Scenarios hitting Fortran retry ceiling at any outer step: "
          f"{n_ceiling}/{len(ceiling_scens)}")

    num_F, mass_F = per_volume(F["pc_final"], scens, rmass)
    num_J, mass_J = per_volume(J["pc_final"], scens, rmass)

    NUM_FLOOR = 1e-3
    MASS_FLOOR = 1e-6
    active_num = num_F > NUM_FLOOR
    active_mass = mass_F > MASS_FLOOR

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    panel(axes[0], num_F, num_J, active_num, ceiling_scens, d_nm,
          f"{NUM_FLOOR:.0e} #/cm³",
          "Number per bin — relative error (1800 s × 48 = 24 h, adaptive)")
    panel(axes[1], mass_F, mass_J, active_mass, ceiling_scens, d_nm,
          f"{MASS_FLOOR:.0e} µg/m³",
          "Mass per bin — relative error (1800 s × 48 = 24 h, adaptive)")

    plt.suptitle(
        f"Bin-by-bin JAX vs Fortran at 1800 s outer timestep — adaptive retry on both sides\n"
        f"{n_ceiling}/100 scenarios hit Fortran's retry ceiling (×, unreliable outputs)\n"
        f"Box statistics use only converged scenarios. "
        f"H₂SO₄ injection per step ≈ 30× the 60s case, so 1800 s is at the stability boundary",
        fontsize=11, fontweight="bold", y=1.02,
    )

    out = ROOT / "plots" / "12_binbybin_1800s.png"
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
