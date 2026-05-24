"""Adaptive vs prescribed-substep parity — 2×2 panel.

Top:    total Number / Mass (1 dot per scenario)
Bottom: bin-by-bin Number / Mass relative error (boxes)
Left:   adaptive baseline
Right:  prescribed (Fortran's exact substep schedule)
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


def scatter_panel(ax, F, J, label, unit, color):
    mask = (F > 0) & (J > 0)
    if mask.sum() == 0:
        return
    ax.scatter(F[mask], J[mask], s=18, alpha=0.7, color=color,
                edgecolor="black", lw=0.4)
    lo, hi = F[mask].min(), F[mask].max()
    ref = np.logspace(np.log10(lo), np.log10(hi), 5)
    ax.plot(ref, ref, "r--", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(f"Fortran  {label}  [{unit}]", fontsize=10)
    ax.set_ylabel(f"JAX  {label}  [{unit}]", fontsize=10)
    ax.grid(True, alpha=0.3, which="both")
    denom = np.maximum(np.abs(F[mask]), np.abs(J[mask]))
    rel = np.abs(F[mask] - J[mask]) / denom
    ax.text(0.03, 0.97,
            f"Median: {np.median(rel):.2e}\nMax: {rel.max():.2e}",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=9, family="monospace",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.8"))


def bin_panel(ax, F_arr, J_arr, active, d_nm, title):
    nbin_ = F_arr.shape[1]
    data_for_box = []
    rng = np.random.default_rng(0)
    for b in range(nbin_):
        denom = np.maximum(np.abs(F_arr[:, b]), np.abs(J_arr[:, b]))
        denom = np.where(denom > 1e-300, denom, 1.0)
        err = np.abs(F_arr[:, b] - J_arr[:, b]) / denom
        jitter = rng.normal(0, 0.07, size=err.shape)
        xs = d_nm[b] * np.exp(jitter)
        if active[:, b].any():
            ax.scatter(xs[active[:, b]], np.clip(err[active[:, b]], 1e-16, 1),
                       s=2, alpha=0.3, color="steelblue", edgecolors="none", zorder=1)
        vals = err[active[:, b]]
        data_for_box.append(np.clip(vals, 1e-16, 1) if vals.size else np.array([1e-16]))
    ax.boxplot(data_for_box, positions=d_nm, widths=d_nm * 0.13,
                showfliers=False, patch_artist=True,
                medianprops=dict(color="crimson", lw=1.5),
                boxprops=dict(facecolor="white", alpha=0.85, edgecolor="black"),
                whiskerprops=dict(color="black", lw=0.8),
                capprops=dict(color="black", lw=0.8), zorder=2)
    ax.axhline(0.01, color="green",  ls="--", lw=1.2, alpha=0.8)
    ax.axhline(0.05, color="orange", ls="--", lw=1.2, alpha=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(5e-17, 2.0)
    ax.xaxis.set_minor_locator(LogLocator(numticks=15))
    ax.yaxis.set_minor_locator(LogLocator(numticks=15))
    ax.grid(True, alpha=0.2, which="both")
    ax.set_xlabel("Bin median diameter [nm]", fontsize=10)
    ax.set_ylabel("|JAX − Fortran| / max(|J|,|F|)", fontsize=10)
    ax.set_title(title, fontsize=10, fontweight="bold")


def main():
    cfg = _minimal_config()
    rmass = np.asarray(cfg.groups[0].rmass)
    d_nm = bin_diameters_nm(cfg.nbin)

    scens = np.load(ROOT / "scenarios" / "realistic_scenarios_100.npz")
    F_base = np.load(ROOT / "outputs" / "fortran_outputs.npz")
    J_base = np.load(ROOT / "outputs" / "jax_outputs.npz")
    F_pre  = np.load(ROOT / "outputs" / "prescribed" / "fortran_outputs.npz")
    J_pre  = np.load(ROOT / "outputs" / "prescribed" / "jax_outputs.npz")

    # Per-volume
    num_F_base, mass_F_base = per_volume(F_base["pc_final"], scens, rmass)
    num_J_base, mass_J_base = per_volume(J_base["pc_final"], scens, rmass)
    num_F_pre,  mass_F_pre  = per_volume(F_pre["pc_final"],  scens, rmass)
    num_J_pre,  mass_J_pre  = per_volume(J_pre["pc_final"],  scens, rmass)

    fig, axes = plt.subplots(3, 2, figsize=(14, 16))

    # Row 1: total Number scatter
    scatter_panel(axes[0, 0], num_F_base.sum(axis=1), num_J_base.sum(axis=1),
                  "N_total", "#/cm³", "steelblue")
    axes[0, 0].set_title("Total NUMBER — adaptive JAX retry\n"
                          "(JAX picks its own substep count)",
                          fontsize=11, fontweight="bold")
    scatter_panel(axes[0, 1], num_F_pre.sum(axis=1),  num_J_pre.sum(axis=1),
                  "N_total", "#/cm³", "darkgreen")
    axes[0, 1].set_title("Total NUMBER — prescribed substeps\n"
                          "(JAX forced to Fortran's exact schedule)",
                          fontsize=11, fontweight="bold")

    # Row 2: total Mass scatter
    scatter_panel(axes[1, 0], mass_F_base.sum(axis=1), mass_J_base.sum(axis=1),
                  "M_total", "µg/m³", "steelblue")
    axes[1, 0].set_title("Total MASS — adaptive JAX retry",
                          fontsize=11, fontweight="bold")
    scatter_panel(axes[1, 1], mass_F_pre.sum(axis=1),  mass_J_pre.sum(axis=1),
                  "M_total", "µg/m³", "darkgreen")
    axes[1, 1].set_title("Total MASS — prescribed substeps",
                          fontsize=11, fontweight="bold")

    # Row 3: bin-by-bin Number
    NUM_FLOOR = 1e-3
    active_base = num_F_base > NUM_FLOOR
    active_pre  = num_F_pre  > NUM_FLOOR
    bin_panel(axes[2, 0], num_F_base, num_J_base, active_base, d_nm,
              "Bin-by-bin NUMBER rel err — adaptive")
    bin_panel(axes[2, 1], num_F_pre,  num_J_pre,  active_pre, d_nm,
              "Bin-by-bin NUMBER rel err — prescribed")

    plt.suptitle(
        "Adaptive vs prescribed-substep: JAX vs Fortran parity\n"
        "Prescribed = JAX runs exactly the substep counts Fortran chose adaptively\n"
        "Result: all errors collapse to ~10⁻⁹ (FMA fusion ULP level)",
        fontsize=12, fontweight="bold", y=0.998,
    )
    out = ROOT / "plots" / "09_prescribed_vs_adaptive.png"
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
