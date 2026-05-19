"""Validation figures for binary_nuc_zhao1995 (Phase 7.4a).

Produces four figures:

1. ``fig1_zhao_rate_vs_h2so4.png`` — nucleation rate as function of
   H2SO4 concentration at four temperatures, log-log scale.
2. ``fig2_zhao_vs_vehkamaki.png`` — side-by-side comparison of
   Zhao-Turco 1995 vs Vehkamaki 2002 across the strat-to-tropo
   envelope.
3. ``fig3_rstar_and_ftry.png`` — critical-cluster radius and Gibbs
   exponent (ftry) as functions of T and H2SO4.
4. ``fig4_nucleation_map.png`` — heatmap of log10(rate) over
   (T, log10(H2SO4)) for RH=0.5.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from carma.constants import AVG, BK, WTMOL_H2O
from carma.nucleation.sulfnucrate import binary_nuc_zhao1995, binary_nuc_vehk2002
from carma.sulfate_utils import wtpct_tabaz
from carma.vapor_pressure import vaporp_h2o_murphy2005


OUTDIR = Path(__file__).parent.parent / "plots" / "phase7_zhao1995"


def _zhao_rate(T, rh, h2so4_num, beta1=2e4):
    """Wrapper building consistent inputs for Zhao-Turco from (T, RH, H2SO4)."""
    pvapl = float(vaporp_h2o_murphy2005(jnp.array(T))[0])
    h2o_num = rh * pvapl / (float(BK) * T)
    h2so4_cgs = h2so4_num * 98.0 / float(AVG)
    h2o_cgs = h2o_num * float(WTMOL_H2O) / float(AVG)
    h2o_mass_wtp = rh * pvapl * float(WTMOL_H2O) / (float(BK) * T * float(AVG))
    wtp = float(wtpct_tabaz(T, h2o_mass_wtp, pvapl))
    return binary_nuc_zhao1995(
        T, wtp, rh, h2so4_num, h2so4_cgs, h2o_num, h2o_cgs, beta1, 98.0, 18.0)


def fig_zhao_rate_vs_h2so4():
    """Rate vs H2SO4 at four temperatures."""
    h2so4_grid = np.logspace(5, 11, 60)
    rh = 0.5

    fig, ax = plt.subplots(figsize=(7, 5))
    for T, color in [(200.0, "C0"), (220.0, "C1"), (240.0, "C2"), (260.0, "C3")]:
        rates = []
        for Na in h2so4_grid:
            nuc, *_ = _zhao_rate(T, rh, Na)
            rates.append(float(nuc))
        rates = np.array(rates)
        rates_clip = np.maximum(rates, 1e-40)
        ax.plot(h2so4_grid, rates_clip, color=color, lw=2, label=f"T = {int(T)} K")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"H$_2$SO$_4$ number density [cm$^{-3}$]")
    ax.set_ylabel(r"nucleation rate [cm$^{-3}$ s$^{-1}$]")
    ax.set_title("Zhao-Turco 1995 — binary H$_2$SO$_4$/H$_2$O nucleation rate vs H$_2$SO$_4$ (RH = 50%)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    ax.set_ylim(1e-20, 1e20)
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig1_zhao_rate_vs_h2so4.png", dpi=140)
    plt.close(fig)


def fig_zhao_vs_vehkamaki():
    """Side-by-side Zhao vs Vehkamaki across T at fixed H2SO4."""
    T_grid = np.linspace(200.0, 290.0, 40)
    h2so4_num = 1e9
    rh = 0.5

    zhao, vehk = [], []
    for T in T_grid:
        nuc_z, *_ = _zhao_rate(T, rh, h2so4_num)
        nuc_v, *_ = binary_nuc_vehk2002(T, rh, h2so4_num, 98.0)
        zhao.append(max(float(nuc_z), 1e-40))
        vehk.append(max(float(nuc_v), 1e-40))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(T_grid, zhao, "C0-", lw=2, label="Zhao-Turco 1995 (classical)")
    ax.plot(T_grid, vehk, "C3--", lw=2, label="Vehkamaki 2002 (parameterised)")
    ax.set_xlabel("temperature [K]")
    ax.set_ylabel(r"nucleation rate [cm$^{-3}$ s$^{-1}$]")
    ax.set_title(
        r"Classical vs parameterised — "
        r"H$_2$SO$_4$ = 10$^9$ cm$^{-3}$, RH = 50%"
    )
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig2_zhao_vs_vehkamaki.png", dpi=140)
    plt.close(fig)


def fig_rstar_and_ftry():
    """r_star and ftry vs T and H2SO4."""
    T_grid = np.linspace(200.0, 270.0, 30)
    rh = 0.5

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for Na, color in [(1e7, "C0"), (1e8, "C1"), (1e9, "C2"), (1e10, "C3")]:
        rstars, ftrys = [], []
        for T in T_grid:
            _, _, rstar, ftry = _zhao_rate(T, rh, Na)
            rstars.append(float(rstar) * 1e7)   # nm
            ftrys.append(float(ftry))
        ax1.plot(T_grid, rstars, color=color, lw=2,
                 label=rf"H$_2$SO$_4$ = $10^{{{int(np.log10(Na))}}}$ cm$^{{-3}}$")
        ax2.plot(T_grid, ftrys, color=color, lw=2,
                 label=rf"H$_2$SO$_4$ = $10^{{{int(np.log10(Na))}}}$ cm$^{{-3}}$")
    ax1.set_xlabel("temperature [K]")
    ax1.set_ylabel(r"critical cluster radius r$_*$ [nm]")
    ax1.set_title("Critical cluster size")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)

    ax2.set_xlabel("temperature [K]")
    ax2.set_ylabel(r"ftry = $-G^*/k_B T$")
    ax2.set_title("Classical nucleation Gibbs exponent")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(OUTDIR / "fig3_rstar_and_ftry.png", dpi=140)
    plt.close(fig)


def fig_nucleation_map():
    """Heatmap log10(rate) over (T, log10(H2SO4))."""
    T_grid = np.linspace(195.0, 290.0, 60)
    log_h2so4 = np.linspace(5, 11, 60)
    rh = 0.5

    rates = np.zeros((len(T_grid), len(log_h2so4)))
    for i, T in enumerate(T_grid):
        for j, lN in enumerate(log_h2so4):
            Na = 10 ** lN
            nuc, *_ = _zhao_rate(T, rh, Na)
            rates[i, j] = max(float(nuc), 1e-40)

    fig, ax = plt.subplots(figsize=(8, 5))
    log_rates = np.log10(rates)
    cs = ax.contourf(log_h2so4, T_grid, log_rates,
                     levels=np.linspace(-20, 20, 41), cmap="viridis")
    cbar = fig.colorbar(cs, ax=ax)
    cbar.set_label(r"log$_{10}$(rate [cm$^{-3}$ s$^{-1}$])")

    cl = ax.contour(log_h2so4, T_grid, log_rates,
                    levels=[-10, -5, 0, 5, 10], colors="white", linewidths=0.8)
    ax.clabel(cl, fmt="%d", inline=True, fontsize=8)
    ax.set_xlabel(r"log$_{10}$(H$_2$SO$_4$ [cm$^{-3}$])")
    ax.set_ylabel("temperature [K]")
    ax.set_title("Zhao-Turco 1995 nucleation rate — RH = 50%")
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig4_nucleation_map.png", dpi=140)
    plt.close(fig)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig_zhao_rate_vs_h2so4()
    fig_zhao_vs_vehkamaki()
    fig_rstar_and_ftry()
    fig_nucleation_map()
    print(f"Figures saved in {OUTDIR}")


if __name__ == "__main__":
    main()
