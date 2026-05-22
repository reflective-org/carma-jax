"""Full plot set for the 6h dt=1s Test ABC strat39 reference run.

Produces six figures into plots/iso_summary/ABC_6h_full/:

  1. totals_N.png            — total N vs time, log y
  2. totals_M.png            — total mass vs time, linear y
  3. dN_dlogD_logy.png       — final size distribution, log y
  4. dN_dlogD_linear.png     — final size distribution, linear y
  5. dM_dlogD_logy.png       — final mass distribution, log y
  6. dM_dlogD_linear.png     — final mass distribution, linear y
  7. banana_Fortran.png      — Fortran dN/dlogD heatmap (time × diameter)
  8. banana_Diffrax.png      — Diffrax dN/dlogD heatmap
  9. banana_diff.png         — |Diffrax − Fortran| heatmap
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
    return 2 * r * 1e7


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = _minimal_config()
    rmass = np.asarray(cfg.groups[0].rmass)
    d_nm = bin_diam_nm()
    dlog10 = math.log10(d_nm[1] / d_nm[0])

    data = np.load(DATA)
    scen = json.loads(str(data["scenario_info"]))
    rho_air = scen["p"] * 100.0 * 10.0 / (float(R_AIR) * scen["T"])
    dt = scen["dt"]; nstep = scen["nstep"]
    time_h = (np.arange(nstep) + 1) * dt / 3600

    pc_F = data["fortran_pc"]                       # (nstep, nbin) mmr
    pc_D = data["diffrax_pc"]                       # (nstep, nbin) #/cm^3
    # Convert F to #/cm^3 per bin for consistency.
    N_F = pc_F * rho_air / rmass[None, :]
    N_D = pc_D

    # dN/dlogD = N / dlog10(D); dM/dlogD = N * rmass / dlog10(D)
    dN_F = N_F / dlog10
    dN_D = N_D / dlog10
    dM_F = (N_F * rmass[None, :]) / dlog10
    dM_D = (N_D * rmass[None, :]) / dlog10

    totN_F = N_F.sum(axis=1)
    totN_D = N_D.sum(axis=1)
    totM_F = (N_F * rmass[None, :]).sum(axis=1)
    totM_D = (N_D * rmass[None, :]).sum(axis=1)

    base_title = (
        f"Test ABC strat39 — dt=1 s × {nstep} = {dt*nstep/3600:.1f} h  "
        f"(T={scen['T']:.1f} K, p={scen['p']:.1f} hPa, M₀={scen['M_total_ug_m3']:.0f} µg/m³, "
        f"[H₂SO₄]={scen['fixed_h2so4_molec_cm3']:.0e} #/cm³)"
    )

    # ---- 1. Total N vs time (log y) ----
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(time_h, totN_F, "-", color="tab:red", lw=2, label="Fortran")
    ax.plot(time_h, totN_D, "-", color="tab:blue", lw=2, label="Diffrax")
    ax.set_xlabel("time [h]"); ax.set_ylabel("Total N [#/cm³]")
    ax.set_yscale("log"); ax.grid(True, alpha=0.3, which="both")
    ax.legend(); ax.set_title(f"Total N vs time\n{base_title}", fontsize=10)
    plt.tight_layout(); plt.savefig(OUT_DIR / "1_totals_N.png", dpi=110); plt.close()
    print(f"Saved 1_totals_N.png")

    # ---- 2. Total mass vs time (linear y) ----
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(time_h, totM_F, "-", color="tab:red", lw=2, label="Fortran")
    ax.plot(time_h, totM_D, "-", color="tab:blue", lw=2, label="Diffrax")
    ax.set_xlabel("time [h]"); ax.set_ylabel("Total mass [g/cm³]")
    ax.grid(True, alpha=0.3); ax.legend()
    ax.set_title(f"Total mass vs time\n{base_title}", fontsize=10)
    plt.tight_layout(); plt.savefig(OUT_DIR / "2_totals_M.png", dpi=110); plt.close()
    print(f"Saved 2_totals_M.png")

    # ---- 3 & 4. Final dN/dlogD log-y and linear-y ----
    for yscale, fname, idx in [("log", "3_dN_dlogD_logy", "3"),
                                 ("linear", "4_dN_dlogD_linear", "4")]:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(d_nm, dN_F[-1], "-o", color="tab:red", ms=5, lw=2, label="Fortran")
        ax.plot(d_nm, dN_D[-1], "-d", color="tab:blue", ms=5, lw=2, label="Diffrax")
        ax.set_xlabel("diameter [nm]"); ax.set_ylabel("dN/dlog₁₀(D) [#/cm³]")
        ax.set_yscale(yscale)
        # zoom to active range
        peak = max(dN_F[-1].max(), dN_D[-1].max())
        if yscale == "log":
            ax.set_ylim(peak * 1e-6, peak * 3)
        active = (dN_F[-1] > peak * 1e-4) | (dN_D[-1] > peak * 1e-4)
        if active.any():
            i_lo = int(np.argmax(active))
            i_hi = int(len(active) - 1 - np.argmax(active[::-1]))
            ax.set_xlim(d_nm[max(0, i_lo - 1)], d_nm[min(NBIN - 1, i_hi + 1)])
        ax.grid(True, alpha=0.3, which="both")
        ax.legend()
        ax.set_title(f"Final dN/dlog₁₀(D) at {dt*nstep/3600:.0f} h ({yscale}-y)\n{base_title}", fontsize=10)
        plt.tight_layout(); plt.savefig(OUT_DIR / f"{fname}.png", dpi=110); plt.close()
        print(f"Saved {fname}.png")

    # ---- 5 & 6. Final dM/dlogD log-y and linear-y ----
    for yscale, fname in [("log", "5_dM_dlogD_logy"),
                            ("linear", "6_dM_dlogD_linear")]:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(d_nm, dM_F[-1], "-o", color="tab:red", ms=5, lw=2, label="Fortran")
        ax.plot(d_nm, dM_D[-1], "-d", color="tab:blue", ms=5, lw=2, label="Diffrax")
        ax.set_xlabel("diameter [nm]"); ax.set_ylabel("dM/dlog₁₀(D) [g/cm³]")
        ax.set_yscale(yscale)
        peak = max(dM_F[-1].max(), dM_D[-1].max())
        if yscale == "log":
            ax.set_ylim(peak * 1e-6, peak * 3)
        active = (dM_F[-1] > peak * 1e-4) | (dM_D[-1] > peak * 1e-4)
        if active.any():
            i_lo = int(np.argmax(active))
            i_hi = int(len(active) - 1 - np.argmax(active[::-1]))
            ax.set_xlim(d_nm[max(0, i_lo - 1)], d_nm[min(NBIN - 1, i_hi + 1)])
        ax.grid(True, alpha=0.3, which="both")
        ax.legend()
        ax.set_title(f"Final dM/dlog₁₀(D) at {dt*nstep/3600:.0f} h ({yscale}-y)\n{base_title}", fontsize=10)
        plt.tight_layout(); plt.savefig(OUT_DIR / f"{fname}.png", dpi=110); plt.close()
        print(f"Saved {fname}.png")

    # ---- 7, 8, 9. Banana plots ----
    # Subsample time axis to ~600 cells so the heatmap stays renderable.
    stride = max(1, nstep // 600)
    t_ax = time_h[::stride]
    dN_F_b = dN_F[::stride].T   # (nbin, ntime)
    dN_D_b = dN_D[::stride].T
    diff_b = dN_D_b - dN_F_b

    # Set common color range so F and D heatmaps are comparable.
    vmax = max(dN_F_b.max(), dN_D_b.max())
    vmin = max(vmax * 1e-6, 1e-3)

    def banana(arr, title, fname, vmin_=vmin, vmax_=vmax, cmap="viridis",
                norm_log=True, label="dN/dlog₁₀(D) [#/cm³]"):
        fig, ax = plt.subplots(figsize=(11, 5))
        # pcolormesh with time on x, diameter on y
        if norm_log:
            arr_pl = np.where(arr > 0, arr, vmin_ * 0.1)
            im = ax.pcolormesh(t_ax, d_nm, arr_pl, cmap=cmap,
                                norm=LogNorm(vmin=vmin_, vmax=vmax_),
                                shading="auto")
        else:
            im = ax.pcolormesh(t_ax, d_nm, arr, cmap=cmap,
                                vmin=-vmax_, vmax=vmax_, shading="auto")
        ax.set_yscale("log")
        ax.set_xlabel("time [h]"); ax.set_ylabel("diameter [nm]")
        ax.set_title(f"{title}\n{base_title}", fontsize=10)
        fig.colorbar(im, ax=ax, label=label)
        plt.tight_layout(); plt.savefig(OUT_DIR / fname, dpi=110); plt.close()
        print(f"Saved {fname}")

    banana(dN_F_b, "(7) Fortran — dN/dlog₁₀(D) banana plot",
           "7_banana_Fortran.png")
    banana(dN_D_b, "(8) Diffrax — dN/dlog₁₀(D) banana plot",
           "8_banana_Diffrax.png")
    # Diff plot uses diverging colormap, linear scale, symmetric range
    diff_vmax = float(np.abs(diff_b).max())
    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.pcolormesh(t_ax, d_nm, diff_b, cmap="RdBu_r",
                        vmin=-diff_vmax, vmax=diff_vmax, shading="auto")
    ax.set_yscale("log")
    ax.set_xlabel("time [h]"); ax.set_ylabel("diameter [nm]")
    ax.set_title(f"(9) Diffrax − Fortran — dN/dlog₁₀(D) difference\n{base_title}", fontsize=10)
    fig.colorbar(im, ax=ax, label="Δ dN/dlog₁₀(D) [#/cm³]  (D − F)")
    plt.tight_layout(); plt.savefig(OUT_DIR / "9_banana_diff.png", dpi=110); plt.close()
    print(f"Saved 9_banana_diff.png")

    print(f"\nAll plots in {OUT_DIR}")


if __name__ == "__main__":
    main()
