"""Validation figures for sulfhetnucrate (Phase 7.4b).

Generates:

1. ``fig1_fletcher_factor.png`` — Fletcher factor fv1 as a function of
   xm = r_pre / r*. Shows both piecewise branches meeting at xm=1.
2. ``fig2_het_rate_vs_seed_size.png`` — heterogeneous rate per
   pre-existing particle vs seed radius, at three temperatures.
3. ``fig3_het_vs_hom_comparison.png`` — side-by-side heterogeneous
   (per seed) and homogeneous (per cm³) rates vs T.
4. ``fig4_het_rate_map.png`` — heatmap log10(rate/seed) over
   (T, log10(seed radius)) at fixed H2SO4 and RH.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from carma.constants import AVG, BK, WTMOL_H2O
from carma.nucleation.sulfhetnucrate import sulfhetnucrate, _fletcher_factor
from carma.nucleation.sulfnucrate import binary_nuc_zhao1995
from carma.sulfate_utils import wtpct_tabaz
from carma.vapor_pressure import vaporp_h2o_murphy2005


OUTDIR = Path(__file__).parent.parent / "plots" / "phase7_sulfhetnucrate"


def _args(T, rh, h2so4_num, beta1=2e4, beta2=1.0):
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
        beta1=beta1, beta2=beta2,
    )


def fig_fletcher_factor():
    """Fletcher factor fv1(xm) across both branches."""
    xm_lo = np.linspace(0.001, 0.999, 200)
    xm_hi = np.linspace(1.001, 50.0, 200)
    fv1_lo = np.asarray(jax.vmap(_fletcher_factor)(jnp.asarray(xm_lo)))
    fv1_hi = np.asarray(jax.vmap(_fletcher_factor)(jnp.asarray(xm_hi)))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(xm_lo, fv1_lo, "C0-", lw=2, label="xm < 1 branch")
    ax.plot(xm_hi, fv1_hi, "C3-", lw=2, label="xm ≥ 1 branch")
    ax.axvline(1.0, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel(r"$x_m = r_{pre} / r_*$")
    ax.set_ylabel(r"Fletcher factor $f_{v1}$")
    ax.set_title("Fletcher 1958 geometric factor — contact angle 50°")
    ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig1_fletcher_factor.png", dpi=140)
    plt.close(fig)


def fig_het_rate_vs_seed_size():
    """Per-seed nucleation rate vs seed radius at three T."""
    r_nm = np.logspace(-1, 3, 50)  # 0.1 nm to 1 μm
    r_cm = r_nm * 1e-7

    fig, ax = plt.subplots(figsize=(7, 5))
    for T, color in [(200.0, "C0"), (220.0, "C1"), (240.0, "C2"), (260.0, "C3")]:
        args = _args(T, rh=0.5, h2so4_num=1e9)
        rates = []
        for r in r_cm:
            nuc = sulfhetnucrate(**args, r_preexist=r)
            rates.append(max(float(nuc), 1e-40))
        ax.plot(r_nm, rates, color=color, lw=2, label=f"T = {int(T)} K")
    ax.set_xlabel("pre-existing seed radius [nm]")
    ax.set_ylabel("heterogeneous rate [embryos/s per seed]")
    ax.set_title(
        r"H$_2$SO$_4$ heterogeneous nucleation rate vs seed size "
        r"(H$_2$SO$_4$=10$^9$ cm$^{-3}$, RH=50%)"
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig2_het_rate_vs_seed_size.png", dpi=140)
    plt.close(fig)


def fig_het_vs_hom_comparison():
    """Heterogeneous rate (per seed, assuming 5-nm pre-existing seeds) vs
    homogeneous rate (per cm³) as T varies."""
    T_grid = np.linspace(200.0, 290.0, 40)
    het, hom = [], []
    for T in T_grid:
        args = _args(T, rh=0.5, h2so4_num=1e9)
        nuc_het = sulfhetnucrate(**args, r_preexist=5e-6)  # 50 nm
        nuc_hom, *_ = binary_nuc_zhao1995(
            args["temp"], args["weight_percent"], args["rh"],
            args["h2so4"], args["h2so4_cgs"],
            args["h2o"], args["h2o_cgs"],
            args["beta1"], 98.0, 18.0,
        )
        het.append(max(float(nuc_het), 1e-40))
        hom.append(max(float(nuc_hom), 1e-40))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(T_grid, het, "C0-", lw=2, label="heterogeneous (per 50-nm seed)")
    ax.plot(T_grid, hom, "C3--", lw=2, label=r"homogeneous (per cm$^3$)")
    ax.set_xlabel("temperature [K]")
    ax.set_ylabel(r"nucleation rate")
    ax.set_yscale("log")
    ax.set_title("Heterogeneous (embryos/s/seed) vs Homogeneous (/cm³/s)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig3_het_vs_hom_comparison.png", dpi=140)
    plt.close(fig)


def fig_het_rate_map():
    """Heatmap log10(rate/seed) over (T, log10(seed size))."""
    T_grid = np.linspace(200.0, 280.0, 40)
    log_r_nm = np.linspace(0, 3, 40)   # 1 nm to 1 μm

    rates = np.zeros((len(T_grid), len(log_r_nm)))
    for i, T in enumerate(T_grid):
        args = _args(T, rh=0.5, h2so4_num=1e9)
        for j, lr in enumerate(log_r_nm):
            r_cm = 10 ** lr * 1e-7
            nuc = sulfhetnucrate(**args, r_preexist=r_cm)
            rates[i, j] = max(float(nuc), 1e-40)

    fig, ax = plt.subplots(figsize=(8, 5))
    log_rates = np.log10(rates)
    cs = ax.contourf(log_r_nm, T_grid, log_rates,
                     levels=np.linspace(-20, 10, 31), cmap="viridis")
    cbar = fig.colorbar(cs, ax=ax)
    cbar.set_label(r"log$_{10}$(rate [embryos/s/seed])")
    cl = ax.contour(log_r_nm, T_grid, log_rates,
                     levels=[-10, -5, 0, 5], colors="white", linewidths=0.8)
    ax.clabel(cl, fmt="%d", inline=True, fontsize=8)
    ax.set_xlabel(r"log$_{10}$(seed radius [nm])")
    ax.set_ylabel("temperature [K]")
    ax.set_title(
        r"Heterogeneous nucleation rate per seed "
        r"(H$_2$SO$_4$=10$^9$ cm$^{-3}$, RH=50%)"
    )
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig4_het_rate_map.png", dpi=140)
    plt.close(fig)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig_fletcher_factor()
    fig_het_rate_vs_seed_size()
    fig_het_vs_hom_comparison()
    fig_het_rate_map()
    print(f"Figures saved in {OUTDIR}")


if __name__ == "__main__":
    main()
