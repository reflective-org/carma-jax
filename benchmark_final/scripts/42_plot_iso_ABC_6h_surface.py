"""Surface-area + N-S-M side-by-side plots for 6h ABC strat39 run.

The user's observation: the only F-vs-Diffrax disparity is at the
small bins. Surface area S is biased toward smaller particles, mass
M toward larger; this script makes the bin-by-bin location of each
quantity (and its discrepancy) visible.

Per-bin surface area: S_i = N_i × 4π r_i²   [cm²/cm³]
Per-bin volume:       V_i = N_i × (4/3)π r_i³ = N_i × rmass_i / ρ
                                                   (sulfate density)

Plots:
  10_totals_S.png      — total surface area vs time
  11_dS_dlogD_logy.png — final surface-area distribution, log y
  12_dS_dlogD_linear.png — same, linear y
  13_NSM_sidebyside.png — three rows: dN/dlogD, dS/dlogD, dM/dlogD,
                          all log-y, side-by-side for visual "where
                          does each quantity live + where's the diff"
  14_banana_S_diff.png — diffrax-fortran surface area heatmap
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "scripts"))

from carma.constants import R_AIR
from jax_ensemble import _minimal_config

NBIN = 38
OUT_DIR = ROOT / "plots" / "iso_summary" / "ABC_6h_full"
DATA = ROOT / "outputs" / "iso_test_ABC_6h" / "strat39_dt1" / "all_solvers.npz"


def bin_diam_nm(nbin=NBIN, rmin=2e-8, rmrat=2.0, rho=1.923):
    vmin = (4 / 3) * math.pi * rmin ** 3 * rho
    rm = vmin * rmrat ** np.arange(nbin)
    r = (3 * rm / (4 * math.pi * rho)) ** (1 / 3)
    return 2 * r * 1e7, r        # diameter in nm, radius in cm


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = _minimal_config()
    rmass = np.asarray(cfg.groups[0].rmass)
    r_cm = np.asarray(cfg.groups[0].r)
    d_nm = 2 * r_cm * 1e7
    dlog10 = math.log10(d_nm[1] / d_nm[0])
    surf_per_particle = 4 * math.pi * r_cm ** 2    # cm² per particle

    data = np.load(DATA)
    scen = json.loads(str(data["scenario_info"]))
    rho_air = scen["p"] * 100.0 * 10.0 / (float(R_AIR) * scen["T"])
    dt = scen["dt"]; nstep = scen["nstep"]
    time_h = (np.arange(nstep) + 1) * dt / 3600

    pc_F = data["fortran_pc"]
    pc_D = data["diffrax_pc"]
    N_F = pc_F * rho_air / rmass[None, :]
    N_D = pc_D
    S_F = N_F * surf_per_particle[None, :]
    S_D = N_D * surf_per_particle[None, :]
    M_F = N_F * rmass[None, :]
    M_D = N_D * rmass[None, :]
    dN_F = N_F / dlog10; dN_D = N_D / dlog10
    dS_F = S_F / dlog10; dS_D = S_D / dlog10
    dM_F = M_F / dlog10; dM_D = M_D / dlog10

    totS_F = S_F.sum(axis=1); totS_D = S_D.sum(axis=1)

    base_title = (
        f"Test ABC strat39 — dt=1 s × {nstep} = {dt*nstep/3600:.1f} h"
    )

    # ---- 10. Total S vs time ----
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(time_h, totS_F, "-", color="tab:red", lw=2, label="Fortran")
    ax.plot(time_h, totS_D, "-", color="tab:blue", lw=2, label="Diffrax")
    ax.set_xlabel("time [h]"); ax.set_ylabel("Total surface area [cm²/cm³]")
    ax.grid(True, alpha=0.3); ax.legend()
    ax.set_title(f"Total surface area vs time\n{base_title}", fontsize=10)
    plt.tight_layout(); plt.savefig(OUT_DIR / "10_totals_S.png", dpi=110); plt.close()
    print("Saved 10_totals_S.png")

    # ---- 11 & 12. Final dS/dlogD log + linear ----
    for yscale, fname in [("log", "11_dS_dlogD_logy"),
                            ("linear", "12_dS_dlogD_linear")]:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(d_nm, dS_F[-1], "-o", color="tab:red", ms=5, lw=2, label="Fortran")
        ax.plot(d_nm, dS_D[-1], "-d", color="tab:blue", ms=5, lw=2, label="Diffrax")
        ax.set_xlabel("diameter [nm]")
        ax.set_ylabel("dS/dlog₁₀(D) [cm²/cm³]")
        ax.set_yscale(yscale)
        peak = max(dS_F[-1].max(), dS_D[-1].max())
        if yscale == "log":
            ax.set_ylim(peak * 1e-6, peak * 3)
        active = (dS_F[-1] > peak * 1e-4) | (dS_D[-1] > peak * 1e-4)
        if active.any():
            i_lo = int(np.argmax(active))
            i_hi = int(len(active) - 1 - np.argmax(active[::-1]))
            ax.set_xlim(d_nm[max(0, i_lo - 1)], d_nm[min(NBIN - 1, i_hi + 1)])
        ax.grid(True, alpha=0.3, which="both"); ax.legend()
        ax.set_title(f"Final dS/dlog₁₀(D) at 6 h ({yscale}-y)\n{base_title}", fontsize=10)
        plt.tight_layout(); plt.savefig(OUT_DIR / f"{fname}.png", dpi=110); plt.close()
        print(f"Saved {fname}.png")

    # ---- 13. NSM side-by-side ----
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    for ax, (Fval, Dval, label, ylabel) in zip(axes, [
        (dN_F[-1], dN_D[-1], "Number", "dN/dlog₁₀(D) [#/cm³]"),
        (dS_F[-1], dS_D[-1], "Surface area", "dS/dlog₁₀(D) [cm²/cm³]"),
        (dM_F[-1], dM_D[-1], "Mass", "dM/dlog₁₀(D) [g/cm³]"),
    ]):
        ax.plot(d_nm, Fval, "-o", color="tab:red", ms=5, lw=2, label="Fortran")
        ax.plot(d_nm, Dval, "-d", color="tab:blue", ms=5, lw=2, label="Diffrax")
        ax.set_yscale("log"); ax.set_ylabel(ylabel)
        peak = max(Fval.max(), Dval.max())
        ax.set_ylim(peak * 1e-6, peak * 3)
        # Mark median-of-distribution by mass median diameter
        # Just add a vertical line at the peak bin for each.
        ax.axvline(d_nm[np.argmax(Fval)], color="tab:red", lw=0.7,
                    ls=":", alpha=0.5)
        ax.axvline(d_nm[np.argmax(Dval)], color="tab:blue", lw=0.7,
                    ls=":", alpha=0.5)
        ax.text(0.02, 0.97, label, transform=ax.transAxes, fontsize=12,
                 fontweight="bold", va="top",
                 bbox=dict(boxstyle="round", fc="white", alpha=0.8))
        ax.grid(True, alpha=0.3, which="both"); ax.legend(loc="lower center")
    axes[-1].set_xlabel("diameter [nm]")
    axes[-1].set_xscale("log")   # keep full range visible
    fig.suptitle(f"Where does each quantity live? F vs D at 6 h\n{base_title}", fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(OUT_DIR / "13_NSM_sidebyside.png", dpi=110); plt.close()
    print("Saved 13_NSM_sidebyside.png")

    # ---- 14. Banana S diff ----
    stride = max(1, nstep // 600)
    t_ax = time_h[::stride]
    diff_S = (dS_D[::stride] - dS_F[::stride]).T   # (nbin, ntime)
    diff_S_vmax = float(np.abs(diff_S).max())
    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.pcolormesh(t_ax, d_nm, diff_S, cmap="RdBu_r",
                        vmin=-diff_S_vmax, vmax=diff_S_vmax, shading="auto")
    ax.set_yscale("log"); ax.set_xlabel("time [h]")
    ax.set_ylabel("diameter [nm]")
    ax.set_title(f"Diffrax − Fortran — dS/dlog₁₀(D) difference (signed)\n{base_title}",
                  fontsize=10)
    fig.colorbar(im, ax=ax, label="Δ dS/dlog₁₀(D) [cm²/cm³]")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "14_banana_S_diff.png", dpi=110); plt.close()
    print("Saved 14_banana_S_diff.png")

    # Also numerical summary
    relS = (totS_D[-1] - totS_F[-1]) / totS_F[-1] * 100
    print(f"\nFinal Surface area: F={totS_F[-1]:.3e}, D={totS_D[-1]:.3e} ({relS:+.2f}%)")


if __name__ == "__main__":
    main()
