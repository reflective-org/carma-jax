"""Diffrax-focused per-bin comparison vs Fortran across isolation tests.

For each (test, atm, dt) combo the user picks (default: all combos),
produces a 2-row × N-dt-cols figure:

  Row 1: dN/dlogD size distribution (linear x-axis diameter, log y).
          Fortran (red), Diffrax (blue), initial (black dashed).
  Row 2: per-bin rel err vs Fortran, as a bar chart.

The bin-by-bin bar chart makes it unambiguous which bins disagree
and by how much.

Reads from benchmark_final/outputs/iso_test_<TEST>/<atm>_dt<dt>/
                all_solvers.npz.
Writes to plots/iso_diffrax_perbin/<test>_<atm>.png.
"""
from __future__ import annotations

import argparse
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

ALL_TESTS = ["A", "B", "C", "AB", "BC", "ABC"]
ATMS = ["strat39", "trop96"]
DTS = [1, 10, 60, 300, 1800]
NBIN = 38


def bin_diam_nm(nbin=NBIN, rmin=2e-8, rmrat=2.0, rho=1.923):
    vmin = (4 / 3) * math.pi * rmin ** 3 * rho
    rm = vmin * rmrat ** np.arange(nbin)
    r = (3 * rm / (4 * math.pi * rho)) ** (1 / 3)
    return 2 * r * 1e7


def _load(test, atm, dt):
    path = ROOT / "outputs" / f"iso_test_{test}" / f"{atm}_dt{dt}" / "all_solvers.npz"
    if not path.exists():
        return None
    return np.load(path, allow_pickle=False)


def _initial_curve(scen, d_nm, dlog10):
    cfg = _minimal_config()
    rmass = np.asarray(cfg.groups[0].rmass)
    r = np.asarray(cfg.groups[0].r)
    rho_air = scen["p"] * 100.0 * 10.0 / (float(R_AIR) * scen["T"])
    log_mu_cm = math.log(scen["aerosol_mu_nm"] * 1e-7)
    log_sigma = math.log(scen["aerosol_sigma_g"])
    pdf = (np.exp(-0.5 * ((np.log(r) - log_mu_cm) / log_sigma) ** 2)
            / (r * log_sigma * math.sqrt(2.0 * math.pi)) * rmass)
    M_target_mmr = (scen["M_total_ug_m3"] * 1e-12) / rho_air
    if pdf.sum() > 0:
        mmr_per_bin = pdf * (M_target_mmr / pdf.sum())
        N_init = mmr_per_bin * rho_air / rmass
    else:
        N_init = np.zeros_like(pdf)
    return N_init / dlog10


def plot_one(test, atm, out_dir):
    cfg = _minimal_config()
    rmass = np.asarray(cfg.groups[0].rmass)
    d_nm = bin_diam_nm()
    dlog10 = math.log10(d_nm[1] / d_nm[0])
    bin_idx = np.arange(NBIN)

    fig, axes = plt.subplots(2, len(DTS), figsize=(20, 7), sharex=False)
    has_data = False
    for icol, dt in enumerate(DTS):
        d = _load(test, atm, dt)
        if d is None:
            for r in range(2):
                axes[r, icol].text(0.5, 0.5, "(missing)",
                                     ha="center", va="center",
                                     transform=axes[r, icol].transAxes)
            continue
        has_data = True
        scen = json.loads(str(d["scenario_info"]))
        rho_air = scen["p"] * 100.0 * 10.0 / (float(R_AIR) * scen["T"])
        N_f = (d["fortran_pc"][-1] * rho_air / rmass) / dlog10
        N_d = d["diffrax_pc"][-1] / dlog10
        N_init = _initial_curve(scen, d_nm, dlog10)

        # Row 1: dN/dlogD with linear-x diameter, log-y.
        ax = axes[0, icol]
        ax.plot(d_nm, N_init, "k--", lw=1, alpha=0.5, label="initial")
        ax.plot(d_nm, N_f, "-o", color="tab:red", ms=4, lw=1.5,
                 label="Fortran")
        ax.plot(d_nm, N_d, "-d", color="tab:blue", ms=4, lw=1.5,
                 label="Diffrax")
        ax.set_yscale("log")
        peak = max(N_f.max(), N_d.max(), N_init.max())
        if peak > 0:
            ax.set_ylim(peak * 1e-6, peak * 3)
        # Zoom to where there's something to see: bins above floor of
        # peak/1e4 — typical span 20-500 nm for our seed-based tests.
        floor_visible = peak * 1e-4
        any_visible = (np.maximum.reduce([N_f, N_d, N_init]) > floor_visible)
        if any_visible.any():
            i_lo = int(np.argmax(any_visible))
            i_hi = int(len(any_visible) - 1 - np.argmax(any_visible[::-1]))
            d_lo = d_nm[max(0, i_lo - 1)]
            d_hi = d_nm[min(len(d_nm) - 1, i_hi + 1)]
            ax.set_xlim(d_lo, d_hi)
        ax.set_xlabel("diameter [nm]")
        ax.set_title(f"dt = {dt} s × 48  ({dt*48:.0f} s)")
        ax.grid(True, alpha=0.3, which="both")
        if icol == 0:
            ax.set_ylabel("dN/dlog₁₀(D) [#/cm³]")
            ax.legend(loc="upper right", fontsize=8)

        # Row 2: per-bin rel err (bar chart).
        ax = axes[1, icol]
        denom = np.maximum(np.abs(N_f), 1e-30)
        rel = (N_d - N_f) / denom * 100
        floor = max(N_f.max() * 1e-4, 1e-30)
        active = N_f > floor
        rel_plot = np.where(active, rel, np.nan)
        colors = ["tab:green" if active[i] and abs(rel[i]) <= 1
                   else "tab:orange" if active[i] and abs(rel[i]) <= 10
                   else "tab:red" if active[i]
                   else "lightgray"
                   for i in range(NBIN)]
        ax.bar(bin_idx, np.where(active, rel, 0.0),
                color=colors, alpha=0.85)
        ax.axhline(0, color="black", lw=0.5)
        # Set ylim wide enough to show the bars but not so wide they're
        # invisible. Use max(|active rel|) + a margin.
        rel_active = rel[active] if active.any() else np.zeros(1)
        ymax = max(5.0, float(np.abs(rel_active).max()) * 1.2) if active.any() else 5.0
        ymax = min(ymax, 25.0)  # cap at 25%
        ax.set_ylim(-ymax, ymax)
        # Limit x-axis to the active bin range for readability.
        if active.any():
            ax.set_xlim(max(-0.5, int(np.argmax(active)) - 1.5),
                          min(NBIN - 0.5,
                              int(NBIN - 1 - np.argmax(active[::-1])) + 1.5))
        ax.grid(True, alpha=0.3, axis="y")
        if icol == 0:
            ax.set_ylabel("Diffrax − Fortran  [%]")
        ax.set_xlabel("bin index")

    if not has_data:
        plt.close(fig)
        return
    first = next((_load(test, atm, dt) for dt in DTS if _load(test, atm, dt) is not None), None)
    scen = json.loads(str(first["scenario_info"]))
    fig.suptitle(
        f"Test {test} — {atm} — diffrax vs Fortran  "
        f"(T={scen['T']:.1f} K, p={scen['p']:.1f} hPa, "
        f"M₀={scen['M_total_ug_m3']:.1f} µg/m³, "
        f"GMD={scen['aerosol_mu_nm']:.0f} nm, σ_g={scen['aerosol_sigma_g']:.1f})",
        fontsize=12,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    out = out_dir / f"{test}_{atm}.png"
    plt.savefig(out, dpi=110)
    plt.close(fig)
    print(f"Saved {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tests", nargs="*", default=ALL_TESTS,
                    choices=ALL_TESTS)
    args = p.parse_args()
    out_dir = ROOT / "plots" / "iso_diffrax_perbin"
    out_dir.mkdir(parents=True, exist_ok=True)
    for test in args.tests:
        for atm in ATMS:
            plot_one(test, atm, out_dir)


if __name__ == "__main__":
    main()
