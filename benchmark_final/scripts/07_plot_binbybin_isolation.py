"""Bin-by-bin parity for the three isolation configurations.

Produces benchmark_final/plots/08_binbybin_isolation.png — a 2×3 panel grid:
  rows: number, mass
  cols: baseline | no_coag (growth only) | coag_only (no growth)
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
    num = pc_mmr * rho_air[:, None] / rmass[None, :]
    mass = pc_mmr * rho_air[:, None] * 1e12
    return num, mass


def panel(ax, F_arr, J_arr, active, d_nm, threshold_label, title, ylim_low=5e-17):
    rng = np.random.default_rng(0)
    nbin_ = F_arr.shape[1]
    data_for_box = []
    for b in range(nbin_):
        denom = np.maximum(np.abs(F_arr[:, b]), np.abs(J_arr[:, b]))
        denom = np.where(denom > 1e-300, denom, 1.0)
        err = np.abs(F_arr[:, b] - J_arr[:, b]) / denom
        jitter = rng.normal(0, 0.07, size=err.shape)
        xs = d_nm[b] * np.exp(jitter)
        if (~active[:, b]).any():
            ax.scatter(xs[~active[:, b]], np.clip(err[~active[:, b]], ylim_low, 1),
                       s=2, alpha=0.15, color="0.75", edgecolors="none", zorder=1)
        if active[:, b].any():
            ax.scatter(xs[active[:, b]], np.clip(err[active[:, b]], ylim_low, 1),
                       s=3, alpha=0.4, color="steelblue", edgecolors="none", zorder=2)
        vals = err[active[:, b]]
        data_for_box.append(np.clip(vals, ylim_low, 1) if vals.size else np.array([ylim_low]))

    ax.boxplot(data_for_box, positions=d_nm, widths=d_nm * 0.13,
                showfliers=False, patch_artist=True,
                medianprops=dict(color="crimson", lw=1.5),
                boxprops=dict(facecolor="white", alpha=0.85, edgecolor="black"),
                whiskerprops=dict(color="black", lw=0.8),
                capprops=dict(color="black", lw=0.8), zorder=3)

    ax.axhline(0.01, color="green", ls="--", lw=1.2, alpha=0.8)
    ax.axhline(0.05, color="orange", ls="--", lw=1.2, alpha=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(ylim_low, 2.0)
    ax.xaxis.set_minor_locator(LogLocator(numticks=15))
    ax.yaxis.set_minor_locator(LogLocator(numticks=15))
    ax.grid(True, alpha=0.2, which="both")
    ax.set_xlabel("Bin median diameter [nm]", fontsize=9)
    ax.set_ylabel("|JAX − Fortran| / max(|J|,|F|)", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")

    # Inline stats box
    all_err = []
    for b in range(nbin_):
        denom = np.maximum(np.abs(F_arr[:, b]), np.abs(J_arr[:, b]))
        denom = np.where(denom > 1e-300, denom, 1.0)
        err = np.abs(F_arr[:, b] - J_arr[:, b]) / denom
        all_err.extend(err[active[:, b]].tolist())
    all_err = np.array(all_err)
    if all_err.size:
        pct_pass1 = (all_err < 0.01).mean() * 100
        pct_pass5 = (all_err < 0.05).mean() * 100
        ax.text(0.02, 0.97,
                f"Active: {active.sum():,}\n"
                f"Median: {np.median(all_err):.1e}\n"
                f"Max:    {all_err.max():.1e}\n"
                f"<1%: {pct_pass1:.0f}%   <5%: {pct_pass5:.0f}%",
                transform=ax.transAxes, va="top", ha="left",
                fontsize=8, family="monospace",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.8"))


def main():
    cfg = _minimal_config()
    rmass = np.asarray(cfg.groups[0].rmass)
    d_nm = bin_diameters_nm(cfg.nbin)

    base_scens = ROOT / "scenarios" / "realistic_scenarios_100.npz"
    coag_scens = ROOT / "scenarios" / "realistic_scenarios_100_coag_only.npz"

    runs = [
        ("baseline",                base_scens,
         ROOT / "outputs" / "fortran_outputs.npz",
         ROOT / "outputs" / "jax_outputs.npz"),
        ("no_coag (growth only)",   base_scens,
         ROOT / "outputs" / "no_coag" / "fortran_outputs.npz",
         ROOT / "outputs" / "no_coag" / "jax_outputs.npz"),
        ("coag_only (no growth)",   coag_scens,
         ROOT / "outputs" / "coag_only" / "fortran_outputs.npz",
         ROOT / "outputs" / "coag_only" / "jax_outputs.npz"),
    ]

    NUM_FLOOR = 1e-3   # #/cm³
    MASS_FLOOR = 1e-6  # µg/m³

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))

    for i, (label, scen_path, fF, fJ) in enumerate(runs):
        scens = np.load(scen_path)
        F = np.load(fF); J = np.load(fJ)
        num_F, mass_F = per_volume(F["pc_final"], scens, rmass)
        num_J, mass_J = per_volume(J["pc_final"], scens, rmass)
        active_num = num_F > NUM_FLOOR
        active_mass = mass_F > MASS_FLOOR

        panel(axes[0, i], num_F, num_J, active_num, d_nm,
              f"{NUM_FLOOR:.0e} #/cm³",
              f"NUMBER — {label}")
        panel(axes[1, i], mass_F, mass_J, active_mass, d_nm,
              f"{MASS_FLOOR:.0e} µg/m³",
              f"MASS — {label}")

    plt.suptitle(
        "Process isolation: bin-by-bin JAX vs Fortran relative error\n"
        "100 scenarios, 60 s × 1440 steps, sulfate-only — green box → empty, blue → active",
        fontsize=12, fontweight="bold", y=1.005,
    )
    out = ROOT / "plots" / "08_binbybin_isolation.png"
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
