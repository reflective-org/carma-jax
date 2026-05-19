"""Validation figures for the sulfnuc driver (Phase 7.5).

Generates:

1. ``fig1_rhompe_vs_T.png`` — homogeneous production rate (summed over
   bins) vs temperature at three H2SO4 levels. Overlays Zhao-Turco vs
   Vehkamaki so the Phase 7.4a method dispatch is visible.
2. ``fig2_rhompe_bin_placement.png`` — where the homogeneous critical
   cluster lands in the bin grid, as T varies.
3. ``fig3_rnuclg_vs_seed_bin.png`` — per-bin heterogeneous nucleation
   rate, assuming one seed per bin, at three temperatures.
4. ``fig4_rhompe_vs_rnuclg_map.png`` — heatmap of log10(ratio of
   summed heterogeneous to homogeneous production) over (T, H2SO4),
   for a fixed aerosol seed distribution.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from carma.constants import AVG, BK, WTMOL_H2O
from carma.nucleation.sulfnuc import sulfnuc
from carma.sulfate_utils import wtpct_tabaz
from carma.vapor_pressure import vaporp_h2o_murphy2005


OUTDIR = Path(__file__).parent.parent / "plots" / "phase7_sulfnuc"


def _args(T, rh, h2so4_num):
    pvapl = float(vaporp_h2o_murphy2005(jnp.array(T))[0])
    h2o_num = rh * pvapl / (float(BK) * T)
    h2so4_cgs = h2so4_num * 98.0 / float(AVG)
    h2o_cgs = h2o_num * float(WTMOL_H2O) / float(AVG)
    h2o_mass_wtp = rh * pvapl * float(WTMOL_H2O) / (float(BK) * T * float(AVG))
    wtp = float(wtpct_tabaz(T, h2o_mass_wtp, pvapl))
    return dict(
        temp=T, weight_percent=wtp, rh=rh,
        h2so4=h2so4_num, h2so4_cgs=h2so4_cgs,
        h2o=h2o_num, h2o_cgs=h2o_cgs,
    )


def _bins(nbin=20, rmin_cm=1e-7, rmrat=2.0, rho=1.8):
    vmin = (4.0 / 3.0) * np.pi * rmin_cm**3 * rho
    rmass = vmin * rmrat ** np.arange(nbin)
    rmassup = rmass * rmrat**0.5
    r_bins = (3.0 * rmass / (4.0 * np.pi * rho)) ** (1.0 / 3.0)
    return jnp.asarray(rmassup), jnp.asarray(r_bins), rmrat


def fig_rhompe_vs_T():
    """Homogeneous rate vs T for three H2SO4 levels, both methods."""
    T_grid = np.linspace(195.0, 295.0, 40)
    rmassup, r_bins, rmrat = _bins()

    fig, ax = plt.subplots(figsize=(7, 5))
    for h2so4_num, color in [(1e7, "C0"), (1e8, "C1"), (1e9, "C2")]:
        rates_zt = []
        rates_vk = []
        for T in T_grid:
            args = _args(T, rh=0.5, h2so4_num=h2so4_num)
            rh_zt, _ = sulfnuc(
                **args, r_bins=r_bins, rmassup=rmassup, rmrat_val=rmrat,
                zmet=1.0, method="ZhaoTurco", do_heterogeneous=False,
            )
            rh_vk, _ = sulfnuc(
                **args, r_bins=r_bins, rmassup=rmassup, rmrat_val=rmrat,
                zmet=1.0, method="Vehkamaki", do_heterogeneous=False,
            )
            rates_zt.append(max(float(rh_zt.sum()), 1e-40))
            rates_vk.append(max(float(rh_vk.sum()), 1e-40))
        ax.plot(T_grid, rates_zt, color=color, lw=2, ls="-",
                label=f"Zhao-Turco  H$_2$SO$_4$=10$^{{{int(np.log10(h2so4_num))}}}$")
        ax.plot(T_grid, rates_vk, color=color, lw=1.5, ls="--")
    ax.set_xlabel("temperature [K]")
    ax.set_ylabel("homogeneous rate [# cm$^{-3}$ s$^{-1}$]")
    ax.set_yscale("log")
    ax.set_title("rhompe (summed over bins) vs T — Zhao-Turco (solid) vs Vehkamaki (dashed)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig1_rhompe_vs_T.png", dpi=140)
    plt.close(fig)


def fig_rhompe_bin_placement():
    """Where the critical cluster lands in the bin grid as T varies."""
    T_grid = np.linspace(195.0, 295.0, 40)
    rmassup, r_bins, rmrat = _bins()
    nbin = rmassup.shape[0]

    fig, ax = plt.subplots(figsize=(7, 5))
    for h2so4_num, color in [(1e7, "C0"), (1e8, "C1"), (1e9, "C2")]:
        bin_idx = []
        for T in T_grid:
            args = _args(T, rh=0.5, h2so4_num=h2so4_num)
            rh_zt, _ = sulfnuc(
                **args, r_bins=r_bins, rmassup=rmassup, rmrat_val=rmrat,
                zmet=1.0, method="ZhaoTurco", do_heterogeneous=False,
            )
            # Locate the non-zero bin (argmax — rhompe is 0 everywhere except nucbin)
            rh_np = np.asarray(rh_zt)
            if rh_np.max() > 0:
                bin_idx.append(int(np.argmax(rh_np)))
            else:
                bin_idx.append(np.nan)
        ax.plot(T_grid, bin_idx, "o-", color=color, lw=1.5, ms=4,
                label=f"H$_2$SO$_4$=10$^{{{int(np.log10(h2so4_num))}}}$")
    ax.set_xlabel("temperature [K]")
    ax.set_ylabel("critical-cluster bin index (0-based)")
    ax.set_ylim(-0.5, nbin - 0.5)
    ax.set_title("Nucleation target bin vs T (Zhao-Turco)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig2_rhompe_bin_placement.png", dpi=140)
    plt.close(fig)


def fig_rnuclg_vs_seed_bin():
    """Per-bin heterogeneous rate, three temperatures."""
    rmassup, r_bins, rmrat = _bins()
    r_nm = np.asarray(r_bins) * 1e7  # cm → nm

    fig, ax = plt.subplots(figsize=(7, 5))
    for T, color in [(200.0, "C0"), (220.0, "C1"), (240.0, "C2"), (260.0, "C3")]:
        args = _args(T, rh=0.5, h2so4_num=1e9)
        _, rn = sulfnuc(
            **args, r_bins=r_bins, rmassup=rmassup, rmrat_val=rmrat, zmet=1.0)
        ax.plot(r_nm, np.maximum(np.asarray(rn), 1e-40),
                "o-", color=color, lw=1.5, ms=4, label=f"T = {int(T)} K")
    ax.set_xlabel("seed-bin radius [nm]")
    ax.set_ylabel("heterogeneous rate [embryos/s/seed]")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(r"rnuclg per seed bin (H$_2$SO$_4$=10$^9$ cm$^{-3}$, RH=50%)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig3_rnuclg_vs_seed_bin.png", dpi=140)
    plt.close(fig)


def fig_rhompe_vs_rnuclg_map():
    """Heatmap log10(sum(rnuclg) / sum(rhompe)) over (T, H2SO4),
    assuming one seed per bin."""
    T_grid = np.linspace(195.0, 290.0, 30)
    log_h2so4 = np.linspace(6.0, 10.5, 30)
    rmassup, r_bins, rmrat = _bins()

    ratio = np.zeros((len(T_grid), len(log_h2so4)))
    for i, T in enumerate(T_grid):
        for j, lh in enumerate(log_h2so4):
            args = _args(T, rh=0.5, h2so4_num=10**lh)
            rh_, rn = sulfnuc(
                **args, r_bins=r_bins, rmassup=rmassup,
                rmrat_val=rmrat, zmet=1.0)
            hom = float(rh_.sum())
            het = float(rn.sum())
            if hom > 1e-40 and het > 1e-40:
                ratio[i, j] = np.log10(het / hom)
            else:
                ratio[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(8, 5))
    cs = ax.contourf(log_h2so4, T_grid, ratio,
                      levels=np.linspace(-10, 10, 21), cmap="RdBu_r")
    cbar = fig.colorbar(cs, ax=ax)
    cbar.set_label(r"log$_{10}$($\Sigma$ rnuclg / $\Sigma$ rhompe)")
    cl = ax.contour(log_h2so4, T_grid, ratio, levels=[0], colors="black",
                    linewidths=1.2)
    ax.clabel(cl, fmt="het = hom", inline=True, fontsize=9)
    ax.set_xlabel(r"log$_{10}$(H$_2$SO$_4$ [cm$^{-3}$])")
    ax.set_ylabel("temperature [K]")
    ax.set_title(
        r"Heterogeneous vs homogeneous (1 seed per bin, RH=50%)")
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig4_rhompe_vs_rnuclg_map.png", dpi=140)
    plt.close(fig)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig_rhompe_vs_T()
    fig_rhompe_bin_placement()
    fig_rnuclg_vs_seed_bin()
    fig_rhompe_vs_rnuclg_map()
    print(f"Figures saved in {OUTDIR}")


if __name__ == "__main__":
    main()
