"""Bin-by-bin parity for the prescribed-substep run (with scatter dots).

Same style as the adaptive baseline plot (04/05) but for the prescribed
run where JAX uses Fortran's exact substep schedule per outer step.

Two panels: number / mass. Per-bin scatter dots colored active (blue,
above threshold) or empty (gray, below). Boxes show cross-scenario
P25/median/P75 over active points.

Output: benchmark_final/plots/11_binbybin_prescribed_dots.png
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


def panel(ax, F_arr, J_arr, active, d_nm, threshold_label, title, ylim_low=1e-13):
    rng = np.random.default_rng(0)
    nbin_ = F_arr.shape[1]
    data_for_box = []

    for b in range(nbin_):
        denom = np.maximum(np.abs(F_arr[:, b]), np.abs(J_arr[:, b]))
        denom = np.where(denom > 1e-300, denom, 1.0)
        err = np.abs(F_arr[:, b] - J_arr[:, b]) / denom

        jitter = rng.normal(0, 0.07, size=err.shape)
        xs = d_nm[b] * np.exp(jitter)

        # Inactive (gray)
        if (~active[:, b]).any():
            ax.scatter(xs[~active[:, b]], np.clip(err[~active[:, b]], ylim_low, 1),
                       s=4, alpha=0.25, color="0.75", edgecolors="none", zorder=1)
        # Active (blue)
        if active[:, b].any():
            ax.scatter(xs[active[:, b]], np.clip(err[active[:, b]], ylim_low, 1),
                       s=6, alpha=0.6, color="steelblue", edgecolors="none", zorder=2)

        vals = err[active[:, b]]
        data_for_box.append(np.clip(vals, ylim_low, 1) if vals.size else np.array([ylim_low]))

    ax.boxplot(data_for_box, positions=d_nm, widths=d_nm * 0.13,
                showfliers=False, patch_artist=True,
                medianprops=dict(color="crimson", lw=1.5),
                boxprops=dict(facecolor="white", alpha=0.85, edgecolor="black"),
                whiskerprops=dict(color="black", lw=0.8),
                capprops=dict(color="black", lw=0.8), zorder=3)

    ax.axhline(0.01, color="green",  ls="--", lw=1.2, alpha=0.8, label="1% threshold")
    ax.axhline(0.05, color="orange", ls="--", lw=1.2, alpha=0.8, label="5% threshold")
    ax.axhline(1e-9, color="purple", ls=":", lw=1.5, alpha=0.8, label="float64 ULP scale")

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="green",  ls="--", lw=1.2, label="1%"),
        Line2D([0], [0], color="orange", ls="--", lw=1.2, label="5%"),
        Line2D([0], [0], color="purple", ls=":",  lw=1.5, label="float64 ULP"),
        ax.scatter([], [], s=10, color="steelblue", alpha=0.7,
                   label=f"active (≥ {threshold_label})"),
        ax.scatter([], [], s=10, color="0.75", alpha=0.5,
                   label=f"empty  (< {threshold_label})"),
    ]
    ax.legend(handles=handles, fontsize=8.5, loc="upper left")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(ylim_low, 2.0)
    ax.xaxis.set_minor_locator(LogLocator(numticks=15))
    ax.yaxis.set_minor_locator(LogLocator(numticks=15))
    ax.grid(True, alpha=0.2, which="both")
    ax.set_xlabel("Bin median diameter [nm]", fontsize=11)
    ax.set_ylabel("|JAX − Fortran| / max(|JAX|, |Fortran|)", fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")

    # Stats inset
    all_err = []
    for b in range(nbin_):
        denom = np.maximum(np.abs(F_arr[:, b]), np.abs(J_arr[:, b]))
        denom = np.where(denom > 1e-300, denom, 1.0)
        err = np.abs(F_arr[:, b] - J_arr[:, b]) / denom
        all_err.extend(err[active[:, b]].tolist())
    all_err = np.array(all_err)
    if all_err.size:
        ax.text(0.98, 0.97,
                f"Active pairs: {active.sum():,} / {F_arr.size:,}\n"
                f"Median: {np.median(all_err):.2e}\n"
                f"P95:    {np.percentile(all_err, 95):.2e}\n"
                f"Max:    {all_err.max():.2e}",
                transform=ax.transAxes, va="top", ha="right",
                fontsize=9, family="monospace",
                bbox=dict(facecolor="white", alpha=0.9, edgecolor="0.8"))


def main():
    cfg = _minimal_config()
    rmass = np.asarray(cfg.groups[0].rmass)
    d_nm = bin_diameters_nm(cfg.nbin)

    scens = np.load(ROOT / "scenarios" / "realistic_scenarios_100.npz")
    F = np.load(ROOT / "outputs" / "prescribed" / "fortran_outputs.npz")
    J = np.load(ROOT / "outputs" / "prescribed" / "jax_outputs.npz")

    num_F, mass_F = per_volume(F["pc_final"], scens, rmass)
    num_J, mass_J = per_volume(J["pc_final"], scens, rmass)

    NUM_FLOOR = 1e-3
    MASS_FLOOR = 1e-6
    active_num = num_F > NUM_FLOOR
    active_mass = mass_F > MASS_FLOOR

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    panel(axes[0], num_F, num_J, active_num, d_nm,
          f"{NUM_FLOOR:.0e} #/cm³",
          "Number per bin — relative error  (prescribed substeps)")
    panel(axes[1], mass_F, mass_J, active_mass, d_nm,
          f"{MASS_FLOOR:.0e} µg/m³",
          "Mass per bin — relative error  (prescribed substeps)")

    plt.suptitle(
        "Bin-by-bin JAX vs Fortran with Fortran's exact substep schedule\n"
        "All active pairs cluster around 10⁻⁹ — float64 FMA-fusion ULP scale\n"
        "100 scenarios, 60 s × 1440 steps, sulfate-only, 24 h simulation",
        fontsize=12, fontweight="bold", y=1.02,
    )

    out = ROOT / "plots" / "11_binbybin_prescribed_dots.png"
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
