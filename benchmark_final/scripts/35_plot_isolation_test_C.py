"""Plot Test C (nucleation+growth) results across timesteps × atmospheres.

Identical structure to 33_plot_isolation_test_A.py / 31_plot_isolation_test_B.py
but reads from ``iso_test_C/`` and plots into ``plots/iso_test_C/``.

Test C expected physics:
  - Sulfnuc adds particles into the smallest few bins
  - Growth (PPM mass-space advection) shifts them to larger bins
  - Mass: gas → particles (fixed H2SO4 reset each step, so apparent
          mass gain over the run equals nuc-mass × nstep)
  - Total N: grows over time as nucleation keeps adding particles

Known gap: diffrax has no bin-0 evap-out sink. With sulfnuc piping
particles into the smallest bins, this manifests as bin-0
accumulation. Faithful was fixed for this in Phase 4 ("Diagnose 60s
parity gap (coag + bin-0 sink)"); diffrax was not.
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


ATMOSPHERES = ["strat39", "trop96"]
DT_VALUES = [1, 10, 60, 300, 1800]
DATA_DIR = ROOT / "outputs" / "iso_test_C"
OUT_DIR = ROOT / "plots" / "iso_test_C"
NBIN = 38


def bin_diameters_nm(nbin=NBIN, rmin_cm=2e-8, rmrat=2.0, rho=1.923):
    vmin = (4.0 / 3.0) * math.pi * rmin_cm ** 3 * rho
    rmass = vmin * rmrat ** np.arange(nbin)
    r = (3.0 * rmass / (4.0 * math.pi * rho)) ** (1.0 / 3.0)
    return 2.0 * r * 1e7


def _load(atm, dt):
    path = DATA_DIR / f"{atm}_dt{dt}" / "all_solvers.npz"
    if not path.exists():
        return None
    return np.load(path, allow_pickle=False), json.loads(str(
        np.load(path, allow_pickle=False)["scenario_info"]
    ))


def _solver_curves(data, scen):
    cfg = _minimal_config()
    rmass = np.asarray(cfg.groups[0].rmass)
    d_nm = bin_diameters_nm()
    dlog10_d = np.log10(d_nm[1] / d_nm[0])
    rho_air = scen["p"] * 100.0 * 10.0 / (float(R_AIR) * scen["T"])
    out = {}
    for key, unit in [("fortran", "mmr"), ("faithful", "per_cm3"),
                       ("diffrax", "per_cm3")]:
        pc = data[f"{key}_pc"][-1]
        N = pc * rho_air / rmass if unit == "mmr" else pc
        out[key] = N / dlog10_d
    return d_nm, out, rmass, rho_air


def plot_grid(atm):
    fig, axes = plt.subplots(len(DT_VALUES), 2, figsize=(13, 16),
                              sharex=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for irow, dt in enumerate(DT_VALUES):
        loaded = _load(atm, dt)
        if loaded is None:
            continue
        data, scen = loaded
        d_nm, curves, rmass, rho_air = _solver_curves(data, scen)
        ax = axes[irow, 0]
        ax.plot(d_nm, curves["fortran"], "-o", color="tab:red", ms=3,
                 lw=1.5, label="Fortran")
        ax.plot(d_nm, curves["faithful"], "-s", color="tab:orange", ms=3,
                 lw=1.5, label="Faithful")
        ax.plot(d_nm, curves["diffrax"], "-d", color="tab:blue", ms=3,
                 lw=1.5, label="Diffrax")
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1e-2)
        ax.set_ylabel("dN/dlog₁₀(D)")
        ax.set_title(f"dt = {dt} s × 48  ({dt*48:.0f} s phys)")
        ax.grid(True, alpha=0.3)
        if irow == 0:
            ax.legend(loc="upper right", fontsize=8)

        ax = axes[irow, 1]
        f = curves["fortran"]
        denom = np.where(np.abs(f) > 1e-30, np.abs(f), 1e-30)
        ax.plot(d_nm, (curves["faithful"] - f) / denom * 100, "-s",
                 color="tab:orange", ms=3, lw=1.2, label="Faithful − F")
        ax.plot(d_nm, (curves["diffrax"] - f) / denom * 100, "-d",
                 color="tab:blue", ms=3, lw=1.2, label="Diffrax − F")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xscale("log")
        ax.set_ylabel("rel diff vs Fortran [%]")
        ax.set_ylim(-100, 500)   # show big positive divergence
        ax.grid(True, alpha=0.3)
        if irow == 0:
            ax.legend(loc="upper right", fontsize=8)

    axes[-1, 0].set_xlabel("diameter [nm]")
    axes[-1, 1].set_xlabel("diameter [nm]")
    first_scen = _load(atm, DT_VALUES[-1])[1]
    fig.suptitle(
        f"Test C — nuc + growth, {atm}  "
        f"(T={first_scen['T']:.1f} K, p={first_scen['p']:.1f} hPa, "
        f"[H₂SO₄]={first_scen['fixed_h2so4_molec_cm3']:.0e} #/cm³)",
        fontsize=13,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.985))
    out = OUT_DIR / f"{atm}_grid.png"
    plt.savefig(out, dpi=110)
    plt.close(fig)
    print(f"Saved {out}")


def plot_dt_convergence(atm):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = _minimal_config()
    rmass = np.asarray(cfg.groups[0].rmass)
    totals_n = {k: [] for k in ["fortran", "faithful", "diffrax"]}
    totals_m = {k: [] for k in ["fortran", "faithful", "diffrax"]}
    bin0 = {k: [] for k in ["fortran", "faithful", "diffrax"]}
    dts = []
    for dt in DT_VALUES:
        loaded = _load(atm, dt)
        if loaded is None:
            continue
        data, scen = loaded
        rho_air = scen["p"] * 100.0 * 10.0 / (float(R_AIR) * scen["T"])
        dts.append(dt)
        for key, unit in [("fortran", "mmr"), ("faithful", "per_cm3"),
                           ("diffrax", "per_cm3")]:
            pc = data[f"{key}_pc"][-1]
            if unit == "mmr":
                N = pc * rho_air / rmass
                M = pc.sum() * rho_air
            else:
                N = pc
                M = (pc * rmass).sum()
            totals_n[key].append(N.sum())
            totals_m[key].append(M)
            bin0[key].append(N[0])
    colors = dict(fortran="tab:red", faithful="tab:orange",
                   diffrax="tab:blue")
    for ax, totals, ylabel, yscale in [
        (axes[0], totals_n, "Total N [#/cm³]", "log"),
        (axes[1], totals_m, "Total mass [g/cm³]", "linear"),
    ]:
        for key in ["fortran", "faithful", "diffrax"]:
            ax.plot(dts, totals[key], "-o", color=colors[key],
                     label=key, lw=2, ms=6)
        ax.set_xlabel("outer-step dt [s]")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log")
        ax.set_yscale(yscale)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend()
    fig.suptitle(f"Test C convergence — {atm}  "
                  "(diffrax bin-0 trap visible in Total N panel)", fontsize=13)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    out = OUT_DIR / f"dt_convergence_{atm}.png"
    plt.savefig(out, dpi=110)
    plt.close(fig)
    print(f"Saved {out}")


def main():
    for atm in ATMOSPHERES:
        plot_grid(atm)
        plot_dt_convergence(atm)


if __name__ == "__main__":
    main()
