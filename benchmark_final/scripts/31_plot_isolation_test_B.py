"""Plot Test B (condensation-only) results across timesteps × atmospheres.

Reads ``benchmark_final/outputs/iso_test_B/<atm>_dt<dt>/all_solvers.npz``
files written by ``30_isolation_test_B_condensation.py``, builds a
plot grid showing how each of the three solvers (Fortran, faithful
JAX, diffrax-PPM) evolves the final size distribution as a function
of timestep.

The plot is organised as two figures, one per atmosphere
(strat39, trop96). Each figure has a row per timestep (5 rows) and
two columns:

  (left)  final dN/dlogD overlay — Fortran red, Faithful orange,
          Diffrax blue, initial black-dashed.
  (right) relative-error vs Fortran in dN/dlogD for the two JAX
          variants — shows where (which bins) the JAX paths diverge.

Output:
  plots/iso_test_B/<atm>_grid.png
  plots/iso_test_B/dt_convergence_<atm>.png  (totals vs dt)
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
DATA_DIR = ROOT / "outputs" / "iso_test_B"
OUT_DIR = ROOT / "plots" / "iso_test_B"

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
    """Return per-solver final dN/dlogD on the same diameter grid."""
    cfg = _minimal_config()
    rmass = np.asarray(cfg.groups[0].rmass)
    d_nm = bin_diameters_nm()
    dlog10_d = np.log10(d_nm[1] / d_nm[0])
    rho_air = scen["p"] * 100.0 * 10.0 / (float(R_AIR) * scen["T"])

    out = {}
    for key, unit in [("fortran", "mmr"), ("faithful", "per_cm3"),
                       ("diffrax", "per_cm3")]:
        pc = data[f"{key}_pc"][-1]
        if unit == "mmr":
            N_per_bin = pc * rho_air / rmass
        else:
            N_per_bin = pc
        out[key] = N_per_bin / dlog10_d
    return d_nm, out, rmass, rho_air


def _initial_curve(scen):
    """Build the initial dN/dlogD from scen metadata (matches script 30)."""
    cfg = _minimal_config()
    rmass = np.asarray(cfg.groups[0].rmass)
    r = np.asarray(cfg.groups[0].r)
    d_nm = bin_diameters_nm()
    dlog10_d = np.log10(d_nm[1] / d_nm[0])
    rho_air = scen["p"] * 100.0 * 10.0 / (float(R_AIR) * scen["T"])
    log_mu_cm = math.log(scen["aerosol_mu_nm"] * 1e-7)
    log_sigma = math.log(scen["aerosol_sigma_g"])
    pdf = (np.exp(-0.5 * ((np.log(r) - log_mu_cm) / log_sigma) ** 2)
            / (r * log_sigma * math.sqrt(2.0 * math.pi)) * rmass)
    M_target_mmr = (scen["M_total_ug_m3"] * 1e-12) / rho_air
    mmr_per_bin = pdf * (M_target_mmr / pdf.sum())
    N_init = mmr_per_bin * rho_air / rmass
    return d_nm, N_init / dlog10_d


def plot_grid_for_atm(atm):
    """One figure per atmosphere: 5 rows × 2 cols (final dist, rel err)."""
    fig, axes = plt.subplots(len(DT_VALUES), 2, figsize=(13, 16),
                              sharex=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for irow, dt in enumerate(DT_VALUES):
        loaded = _load(atm, dt)
        if loaded is None:
            for c in range(2):
                axes[irow, c].text(0.5, 0.5, f"(missing data dt={dt})",
                                     ha="center", va="center",
                                     transform=axes[irow, c].transAxes)
            continue
        data, scen = loaded
        d_nm, curves, rmass, rho_air = _solver_curves(data, scen)
        d_init, n_init = _initial_curve(scen)

        ax = axes[irow, 0]
        ax.plot(d_init, n_init, "k--", lw=1, label="initial", alpha=0.6)
        ax.plot(d_nm, curves["fortran"], "-o", color="tab:red", ms=3,
                 lw=1.5, label="Fortran")
        ax.plot(d_nm, curves["faithful"], "-s", color="tab:orange", ms=3,
                 lw=1.5, label="Faithful")
        ax.plot(d_nm, curves["diffrax"], "-d", color="tab:blue", ms=3,
                 lw=1.5, label="Diffrax-PPM")
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1e-10)
        ax.set_ylabel("dN/dlog₁₀(D)")
        ax.set_title(f"dt = {dt} s × 48  ({dt*48:.0f} s phys)")
        ax.grid(True, alpha=0.3)
        if irow == 0:
            ax.legend(loc="upper right", fontsize=8)

        ax = axes[irow, 1]
        f = curves["fortran"]
        denom = np.where(np.abs(f) > 1e-30, np.abs(f), 1e-30)
        rel_j = (curves["faithful"] - f) / denom
        rel_d = (curves["diffrax"] - f) / denom
        ax.plot(d_nm, rel_j * 100, "-s", color="tab:orange", ms=3,
                 lw=1.2, label="Faithful − F")
        ax.plot(d_nm, rel_d * 100, "-d", color="tab:blue", ms=3,
                 lw=1.2, label="Diffrax − F")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xscale("log")
        ax.set_ylabel("rel diff vs Fortran [%]")
        ax.set_ylim(-100, 100)
        ax.grid(True, alpha=0.3)
        if irow == 0:
            ax.legend(loc="upper right", fontsize=8)

    axes[-1, 0].set_xlabel("diameter [nm]")
    axes[-1, 1].set_xlabel("diameter [nm]")
    first_scen = _load(atm, DT_VALUES[-1])[1]
    fig.suptitle(
        f"Test B — condensation only, {atm}  "
        f"(T={first_scen['T']:.1f} K, p={first_scen['p']:.1f} hPa, "
        f"[H₂SO₄] fixed = {first_scen['fixed_h2so4_molec_cm3']:.0e} #/cm³)",
        fontsize=13,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.985))
    out = OUT_DIR / f"{atm}_grid.png"
    plt.savefig(out, dpi=110)
    plt.close(fig)
    print(f"Saved {out}")


def plot_dt_convergence(atm):
    """Single figure per atmosphere: totals vs dt for all three solvers.

    Shows that as dt → small, all three solvers should converge to the
    same total mass / total N / gas residual (well, gas is fixed).
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = _minimal_config()
    rmass = np.asarray(cfg.groups[0].rmass)

    totals_n = {k: [] for k in ["fortran", "faithful", "diffrax"]}
    totals_m = {k: [] for k in ["fortran", "faithful", "diffrax"]}
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

    colors = dict(fortran="tab:red", faithful="tab:orange",
                   diffrax="tab:blue")
    for ax, totals, ylabel in [
        (axes[0], totals_n, "Total N [#/cm³]"),
        (axes[1], totals_m, "Total mass [g/cm³]"),
    ]:
        for key in ["fortran", "faithful", "diffrax"]:
            ax.plot(dts, totals[key], "-o", color=colors[key],
                     label=key, lw=2, ms=6)
        ax.set_xlabel("outer-step dt [s]")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(f"Test B convergence — {atm}", fontsize=13)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    out = OUT_DIR / f"dt_convergence_{atm}.png"
    plt.savefig(out, dpi=110)
    plt.close(fig)
    print(f"Saved {out}")


def main():
    for atm in ATMOSPHERES:
        plot_grid_for_atm(atm)
        plot_dt_convergence(atm)


if __name__ == "__main__":
    main()
