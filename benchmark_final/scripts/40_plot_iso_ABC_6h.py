"""Plot the 6h dt=1s Test ABC strat39 run — Fortran vs Diffrax.

Three-panel figure:
  (a) Final dN/dlogD overlay (linear-x diameter, log-y N).
  (b) Per-bin rel err bar chart (linear x = bin index).
  (c) Time evolution: total N and total mass vs time.

Reads benchmark_final/outputs/iso_test_ABC_6h/strat39_dt1/all_solvers.npz.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "scripts"))

from carma.constants import R_AIR
from jax_ensemble import _minimal_config

NBIN = 38


def bin_diam_nm(nbin=NBIN, rmin=2e-8, rmrat=2.0, rho=1.923):
    vmin = (4 / 3) * math.pi * rmin ** 3 * rho
    rm = vmin * rmrat ** np.arange(nbin)
    r = (3 * rm / (4 * math.pi * rho)) ** (1 / 3)
    return 2 * r * 1e7


def main():
    cfg = _minimal_config()
    rmass = np.asarray(cfg.groups[0].rmass)
    r = np.asarray(cfg.groups[0].r)
    d_nm = bin_diam_nm()
    dlog10 = math.log10(d_nm[1] / d_nm[0])

    data = np.load(
        ROOT / "outputs" / "iso_test_ABC_6h" / "strat39_dt1" / "all_solvers.npz"
    )
    scen = json.loads(str(data["scenario_info"]))
    rho_air = scen["p"] * 100.0 * 10.0 / (float(R_AIR) * scen["T"])
    dt = scen["dt"]; nstep = scen["nstep"]

    # Final dN/dlogD per solver
    N_f = (data["fortran_pc"][-1] * rho_air / rmass) / dlog10
    N_d = data["diffrax_pc"][-1] / dlog10

    # Initial seed
    log_mu = math.log(scen["aerosol_mu_nm"] * 1e-7)
    log_s = math.log(scen["aerosol_sigma_g"])
    pdf = (np.exp(-0.5 * ((np.log(r) - log_mu) / log_s) ** 2)
            / (r * log_s * math.sqrt(2 * math.pi)) * rmass)
    M_target = (scen["M_total_ug_m3"] * 1e-12) / rho_air
    N_init = pdf * (M_target / pdf.sum()) * rho_air / rmass / dlog10

    # Time series of totals
    time_s = (np.arange(nstep) + 1) * dt
    time_h = time_s / 3600
    pc_F_hist = data["fortran_pc"]
    pc_D_hist = data["diffrax_pc"]
    totN_F = (pc_F_hist * rho_air / rmass[None, :]).sum(axis=1)
    totN_D = pc_D_hist.sum(axis=1)
    totM_F = pc_F_hist.sum(axis=1) * rho_air
    totM_D = (pc_D_hist * rmass[None, :]).sum(axis=1)

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])

    # (a) Final dN/dlogD
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(d_nm, N_init, "k--", lw=1, alpha=0.6, label="initial seed (2 µg/m³)")
    ax1.plot(d_nm, N_f, "-o", color="tab:red", ms=5, lw=1.8, label="Fortran")
    ax1.plot(d_nm, N_d, "-d", color="tab:blue", ms=5, lw=1.8, label="Diffrax")
    ax1.set_yscale("log")
    peak = max(N_f.max(), N_d.max(), N_init.max())
    ax1.set_ylim(peak * 1e-7, peak * 3)
    # Zoom to active bins
    floor = peak * 1e-5
    any_vis = np.maximum.reduce([N_f, N_d, N_init]) > floor
    if any_vis.any():
        i_lo, i_hi = int(np.argmax(any_vis)), int(len(any_vis) - 1 - np.argmax(any_vis[::-1]))
        ax1.set_xlim(d_nm[max(0, i_lo - 1)], d_nm[min(len(d_nm) - 1, i_hi + 1)])
    ax1.set_xlabel("diameter [nm]")
    ax1.set_ylabel("dN/dlog₁₀(D) [#/cm³]")
    ax1.set_title(f"(a) Final size distribution after {dt*nstep/3600:.1f} h")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3, which="both")

    # (b) Per-bin rel err bar chart
    ax2 = fig.add_subplot(gs[0, 1])
    bin_idx = np.arange(NBIN)
    denom = np.maximum(N_f, 1e-30)
    rel = (N_d - N_f) / denom * 100
    active = N_f > N_f.max() * 1e-4
    colors = ["tab:green" if active[i] and abs(rel[i]) <= 1
               else "tab:orange" if active[i] and abs(rel[i]) <= 5
               else "tab:red" if active[i]
               else "lightgray"
               for i in range(NBIN)]
    ax2.bar(bin_idx, np.where(active, rel, 0.0), color=colors, alpha=0.85)
    ax2.axhline(0, color="black", lw=0.5)
    if active.any():
        ax2.set_xlim(int(np.argmax(active)) - 1.5,
                       int(NBIN - 1 - np.argmax(active[::-1])) + 1.5)
    rel_active = rel[active] if active.any() else np.zeros(1)
    ymax = max(5.0, float(np.abs(rel_active).max()) * 1.2)
    ax2.set_ylim(-ymax, ymax)
    ax2.set_xlabel("bin index")
    ax2.set_ylabel("Diffrax − Fortran  [%]")
    ax2.set_title(f"(b) Per-bin rel err (active bins, "
                   f"med={np.median(np.abs(rel_active)):.2f}%, "
                   f"max={np.abs(rel_active).max():.2f}%)")
    ax2.grid(True, alpha=0.3, axis="y")

    # (c) Time series of total N
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(time_h, totN_F, "-", color="tab:red", lw=1.5, label="Fortran")
    ax3.plot(time_h, totN_D, "-", color="tab:blue", lw=1.5, label="Diffrax")
    ax3.set_xlabel("time [h]")
    ax3.set_ylabel("Total N [#/cm³]")
    ax3.set_yscale("log")
    ax3.set_title("(c) Total N vs time")
    ax3.grid(True, alpha=0.3, which="both")
    ax3.legend()

    # (d) Time series of total mass
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(time_h, totM_F, "-", color="tab:red", lw=1.5, label="Fortran")
    ax4.plot(time_h, totM_D, "-", color="tab:blue", lw=1.5, label="Diffrax")
    ax4.set_xlabel("time [h]")
    ax4.set_ylabel("Total particle mass [g/cm³]")
    ax4.set_title("(d) Total particle mass vs time")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    fig.suptitle(
        f"Test ABC strat39 — dt=1 s × {nstep} = 6 h physical  "
        f"(T={scen['T']:.1f} K, p={scen['p']:.1f} hPa, "
        f"M₀={scen['M_total_ug_m3']:.0f} µg/m³, "
        f"GMD={scen['aerosol_mu_nm']:.0f} nm, σ_g={scen['aerosol_sigma_g']:.1f}, "
        f"[H₂SO₄]_fix={scen['fixed_h2so4_molec_cm3']:.0e} #/cm³)\n"
        f"Wall: Fortran 2.1 s vs Diffrax 386 s (184× slower at this 21600-outer-step setup)",
        fontsize=11, y=0.98,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    out = ROOT / "plots" / "iso_summary" / "ABC_strat39_6h_dt1.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=110)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
