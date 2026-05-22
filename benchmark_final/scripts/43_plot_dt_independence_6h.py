"""Compare 6h ABC strat39 final states across outer dt values.

Reads:
  iso_test_ABC_6h/strat39_dt1/all_solvers.npz       (reference, dt=1s × 21600)
  iso_test_ABC_6h_dt10/strat39_dt10/all_solvers.npz (dt=10s × 2160)
  iso_test_ABC_6h_dt60/strat39_dt60/all_solvers.npz (dt=60s × 360)
  iso_test_ABC_6h_dt300/strat39_dt300/all_solvers.npz (dt=300s × 72)
  iso_test_ABC_6h_dt1800/strat39_dt1800/all_solvers.npz (dt=1800s × 12)

All at 6 h physical, same scenario (strat39, M=2 µg/m³ seed, fixed
[H2SO4]=1e7). The dt-sensitivity is real because fixed-[H2SO4] is
reset between outer steps — at dt=1800 the gas drifts down between
resets, at dt=10 it stays effectively constant. So this tests both:
  (a) diffrax's adaptive-step accuracy *within* one outer step
  (b) the cumulative effect of the reset cadence on the trajectory

Plots:
  1. Final dN/dlogD overlay across all dt (Fortran and Diffrax)
  2. Final dM/dlogD overlay across all dt
  3. Total N, M, S at 6h vs outer dt
  4. Wall time vs outer dt (Fortran vs Diffrax)
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

OUT = ROOT / "plots" / "iso_summary" / "dt_independence_6h"
SOURCES = {
    1:    ROOT / "outputs" / "iso_test_ABC_6h"      / "strat39_dt1"    / "all_solvers.npz",
    10:   ROOT / "outputs" / "iso_test_ABC_6h_dt10" / "strat39_dt10"   / "all_solvers.npz",
    60:   ROOT / "outputs" / "iso_test_ABC_6h_dt60" / "strat39_dt60"   / "all_solvers.npz",
    300:  ROOT / "outputs" / "iso_test_ABC_6h_dt300"/ "strat39_dt300"  / "all_solvers.npz",
    1800: ROOT / "outputs" / "iso_test_ABC_6h_dt1800"/ "strat39_dt1800"/ "all_solvers.npz",
}
NBIN = 38


def bin_diam_nm(nbin=NBIN, rmin=2e-8, rmrat=2.0, rho=1.923):
    vmin = (4 / 3) * math.pi * rmin ** 3 * rho
    rm = vmin * rmrat ** np.arange(nbin)
    r = (3 * rm / (4 * math.pi * rho)) ** (1 / 3)
    return 2 * r * 1e7, r


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = _minimal_config()
    rmass = np.asarray(cfg.groups[0].rmass)
    d_nm, r_cm = bin_diam_nm()
    dlog10 = math.log10(d_nm[1] / d_nm[0])
    surf_pp = 4 * math.pi * r_cm ** 2

    # Collect final-state data per dt.
    data = {}
    for dt, path in SOURCES.items():
        if not path.exists():
            print(f"skipping dt={dt}: file not found")
            continue
        d = np.load(path)
        scen = json.loads(str(d["scenario_info"]))
        rho_air = scen["p"] * 100.0 * 10.0 / (float(R_AIR) * scen["T"])
        N_F = d["fortran_pc"][-1] * rho_air / rmass
        N_D = d["diffrax_pc"][-1]
        data[dt] = dict(
            N_F=N_F, N_D=N_D,
            totN_F=N_F.sum(), totN_D=N_D.sum(),
            totM_F=(N_F * rmass).sum(), totM_D=(N_D * rmass).sum(),
            totS_F=(N_F * surf_pp).sum(), totS_D=(N_D * surf_pp).sum(),
            wall_F=float(d["fortran_wall"]),
            wall_D=float(d["diffrax_wall"]),
        )

    dts = sorted(data.keys())
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(dts)))

    # ---- 1. Final dN/dlogD overlay (linear y, zoomed) ----
    fig, (axF, axD) = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for icol, (ax, key, label) in enumerate([(axF, "N_F", "Fortran"),
                                                 (axD, "N_D", "Diffrax")]):
        for dt, color in zip(dts, cmap):
            ax.plot(d_nm, data[dt][key] / dlog10, "-o", color=color,
                     lw=1.5, ms=4, label=f"dt={dt}s")
        ax.set_yscale("log")
        peak = max(max(data[dt][key]) / dlog10 for dt in dts)
        ax.set_ylim(peak * 1e-6, peak * 3)
        # Zoom to active range across all dts
        all_active = np.zeros(NBIN, dtype=bool)
        for dt in dts:
            all_active |= (data[dt][key] / dlog10 > peak * 1e-4)
        if all_active.any():
            i_lo = int(np.argmax(all_active))
            i_hi = int(len(all_active) - 1 - np.argmax(all_active[::-1]))
            ax.set_xlim(d_nm[max(0, i_lo - 1)], d_nm[min(NBIN - 1, i_hi + 1)])
        ax.set_xlabel("diameter [nm]")
        if icol == 0: ax.set_ylabel("dN/dlog₁₀(D) [#/cm³]")
        ax.set_title(label)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("ABC strat39 at 6 h — dN/dlogD final state across outer dt", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT / "1_dN_overlay.png", dpi=110); plt.close()
    print("Saved 1_dN_overlay.png")

    # ---- 2. Final dM/dlogD overlay ----
    fig, (axF, axD) = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for icol, (ax, key, label) in enumerate([(axF, "N_F", "Fortran"),
                                                 (axD, "N_D", "Diffrax")]):
        for dt, color in zip(dts, cmap):
            dM = (data[dt][key] * rmass) / dlog10
            ax.plot(d_nm, dM, "-o", color=color, lw=1.5, ms=4,
                     label=f"dt={dt}s")
        peak = max(max((data[dt][key] * rmass)) / dlog10 for dt in dts)
        all_active = np.zeros(NBIN, dtype=bool)
        for dt in dts:
            all_active |= ((data[dt][key] * rmass) / dlog10 > peak * 1e-3)
        if all_active.any():
            i_lo = int(np.argmax(all_active))
            i_hi = int(len(all_active) - 1 - np.argmax(all_active[::-1]))
            ax.set_xlim(d_nm[max(0, i_lo - 1)], d_nm[min(NBIN - 1, i_hi + 1)])
        ax.set_xlabel("diameter [nm]")
        if icol == 0: ax.set_ylabel("dM/dlog₁₀(D) [g/cm³]")
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("ABC strat39 at 6 h — dM/dlogD final state across outer dt", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT / "2_dM_overlay.png", dpi=110); plt.close()
    print("Saved 2_dM_overlay.png")

    # ---- 3. Totals at 6h vs dt ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (quant, label) in zip(axes, [
        ("totN", "Total N [#/cm³]"),
        ("totM", "Total mass [g/cm³]"),
        ("totS", "Total surface [cm²/cm³]"),
    ]):
        f_vals = [data[dt][f"{quant}_F"] for dt in dts]
        d_vals = [data[dt][f"{quant}_D"] for dt in dts]
        ax.plot(dts, f_vals, "-o", color="tab:red",  lw=2, ms=7, label="Fortran")
        ax.plot(dts, d_vals, "-d", color="tab:blue", lw=2, ms=7, label="Diffrax")
        ax.set_xscale("log")
        ax.set_xlabel("outer-step dt [s]")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend()
    fig.suptitle("Final-state totals at 6 h vs outer dt", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT / "3_totals_vs_dt.png", dpi=110); plt.close()
    print("Saved 3_totals_vs_dt.png")

    # ---- 4. Wall times ----
    fig, ax = plt.subplots(figsize=(9, 5))
    f_wall = [data[dt]["wall_F"] for dt in dts]
    d_wall = [data[dt]["wall_D"] for dt in dts]
    ax.plot(dts, f_wall, "-o", color="tab:red", lw=2, ms=7, label="Fortran")
    ax.plot(dts, d_wall, "-d", color="tab:blue", lw=2, ms=7, label="Diffrax")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("outer-step dt [s]"); ax.set_ylabel("wall time [s]")
    ax.set_title("Wall time vs outer dt (6 h physical, ABC strat39)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / "4_wall_vs_dt.png", dpi=110); plt.close()
    print("Saved 4_wall_vs_dt.png")

    # Summary table
    print(f"\n{'dt':>6} {'F totN':>11} {'D totN':>11} {'rel%':>7} "
          f"{'F totM':>11} {'D totM':>11} {'rel%':>7} "
          f"{'F wall':>8} {'D wall':>8}")
    for dt in dts:
        d_ = data[dt]
        relN = (d_["totN_D"] - d_["totN_F"]) / d_["totN_F"] * 100
        relM = (d_["totM_D"] - d_["totM_F"]) / d_["totM_F"] * 100
        print(f"{dt:>5}s {d_['totN_F']:>11.3e} {d_['totN_D']:>11.3e} {relN:>+6.2f}% "
              f"{d_['totM_F']:>11.3e} {d_['totM_D']:>11.3e} {relM:>+6.2f}% "
              f"{d_['wall_F']:>7.1f}s {d_['wall_D']:>7.1f}s")


if __name__ == "__main__":
    main()
