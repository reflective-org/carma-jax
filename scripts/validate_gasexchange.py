"""Validation figures for the gasexchange driver (Phase 7.6a).

Wires ``sulfnuc`` output into ``gasexchange`` and plots H2SO4
consumption rates as T and H2SO4 vary for a single-group, single-gas
sulfate scenario.

1. ``fig1_h2so4_loss_vs_T.png`` — gas loss rate broken down by
   homogeneous vs heterogeneous, vs T at three H2SO4 levels.
2. ``fig2_h2so4_loss_vs_h2so4.png`` — loss rate vs H2SO4 at three T.
3. ``fig3_h2so4_loss_map.png`` — heatmap of log10(|gas loss|) over
   (T, log10(H2SO4)) with homogeneous vs heterogeneous dominance
   contour.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from carma.constants import AVG, BK, WTMOL_H2O
from carma.gasexchange import gasexchange
from carma.nucleation.sulfnuc import sulfnuc
from carma.sulfate_utils import wtpct_tabaz
from carma.vapor_pressure import vaporp_h2o_murphy2005


OUTDIR = Path(__file__).parent.parent / "plots" / "phase7_gasexchange"


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
    return jnp.asarray(rmassup), jnp.asarray(r_bins), jnp.asarray(rmass), rmrat


def _pc_lognormal(r_bins_cm, mu_cm, sigma_g, total_num):
    """Lognormal particle distribution centred at mu with sigma_g."""
    r = np.asarray(r_bins_cm)
    x = np.log(r / mu_cm) / np.log(sigma_g)
    dN_dlnr = total_num / (np.sqrt(2 * np.pi) * np.log(sigma_g)) * np.exp(-0.5 * x**2)
    # Bin width in ln(r) ≈ const for geometric grid
    dlnr = np.log(r[1] / r[0])
    return jnp.asarray(dN_dlnr * dlnr)


def _sulfate_config(nbin):
    """Minimal single-group sulfate indexing tables."""
    i2_map = np.arange(nbin) + 1
    i2_map[-1] = -1            # top bin has no target
    return dict(
        if_nuc=np.asarray([[True]]),
        ienconc=np.asarray([0]),
        igelem=np.asarray([0]),
        inucgas=np.asarray([0]),
        nnuc2elem=np.asarray([1]),
        igrowgas=np.asarray([-1]),    # disable growth branch for these figs
        inuc2bin=jnp.asarray(i2_map.reshape(nbin, 1, 1)),
    )


def _diffmass(rmass_array):
    """Build diffmass(tgt_bin, tgt_grp, src_bin, src_grp) from rmass[nbin, 1]."""
    rmass = rmass_array[:, None]
    tgt = rmass[:, :, None, None]
    src = rmass[None, None, :, :]
    return jnp.asarray(tgt - src)


def _gas_breakdown(T, rh, h2so4_num, pc_seed):
    """Compute (hom_loss, het_loss) at given T, RH, H2SO4."""
    args = _args(T, rh, h2so4_num)
    rmassup, r_bins, rmass, rmrat = _bins()
    nbin = rmass.shape[0]
    cfg = _sulfate_config(nbin)
    rmass_2d = rmass[:, None]
    diffmass = _diffmass(rmass)

    rhompe_1d, rnuclg_1d = sulfnuc(
        **args, r_bins=r_bins, rmassup=rmassup, rmrat_val=rmrat, zmet=1.0,
    )

    # Shape into (nbin, nelem) for homogeneous, (nbin, ngroup, ngroup) for het
    rhompe = rhompe_1d[:, None]
    rnuclg = rnuclg_1d[:, None, None]

    # Full call
    pc_iz = pc_seed[:, None]
    zeros_rng = jnp.zeros((nbin, 1))
    cmf = jnp.zeros((nbin, 1))
    totevap = jnp.zeros((nbin, 1), dtype=bool)

    out_full = gasexchange(
        pc_iz=pc_iz, rhompe=rhompe, rnuclg=rnuclg,
        growlg=zeros_rng, evaplg=zeros_rng,
        rmass=rmass_2d, diffmass=diffmass,
        cmf=cmf, totevap=totevap, inuc2bin=cfg["inuc2bin"],
        if_nuc=cfg["if_nuc"], ienconc=cfg["ienconc"], igelem=cfg["igelem"],
        inucgas=cfg["inucgas"], nnuc2elem=cfg["nnuc2elem"],
        igrowgas=cfg["igrowgas"],
        ngas=1, ngroup=1, nelem=1, nbin=nbin,
    )
    # Homogeneous-only call: zero rnuclg
    out_hom = gasexchange(
        pc_iz=pc_iz, rhompe=rhompe, rnuclg=jnp.zeros_like(rnuclg),
        growlg=zeros_rng, evaplg=zeros_rng,
        rmass=rmass_2d, diffmass=diffmass,
        cmf=cmf, totevap=totevap, inuc2bin=cfg["inuc2bin"],
        if_nuc=cfg["if_nuc"], ienconc=cfg["ienconc"], igelem=cfg["igelem"],
        inucgas=cfg["inucgas"], nnuc2elem=cfg["nnuc2elem"],
        igrowgas=cfg["igrowgas"],
        ngas=1, ngroup=1, nelem=1, nbin=nbin,
    )

    hom_loss = -float(out_hom[0])       # positive = consumed
    het_loss = -(float(out_full[0]) - float(out_hom[0]))
    return hom_loss, het_loss


def fig_h2so4_loss_vs_T():
    T_grid = np.linspace(195.0, 290.0, 30)
    _, r_bins, _, _ = _bins()
    pc_seed = _pc_lognormal(np.asarray(r_bins), mu_cm=1e-6, sigma_g=1.6,
                             total_num=10.0)

    fig, ax = plt.subplots(figsize=(7, 5))
    for h2so4_num, color in [(1e7, "C0"), (1e8, "C1"), (1e9, "C2")]:
        hom = []
        het = []
        for T in T_grid:
            h, het_ = _gas_breakdown(T, 0.5, h2so4_num, pc_seed)
            hom.append(max(h, 1e-40))
            het.append(max(het_, 1e-40))
        ax.plot(T_grid, hom, color=color, lw=2, ls="-",
                label=f"hom H$_2$SO$_4$=10$^{{{int(np.log10(h2so4_num))}}}$")
        ax.plot(T_grid, het, color=color, lw=1.5, ls="--")
    ax.set_xlabel("temperature [K]")
    ax.set_ylabel("H$_2$SO$_4$ consumption [g/cm$^3$/s]")
    ax.set_yscale("log")
    ax.set_title("gasexchange: homogeneous (solid) vs heterogeneous (dashed) H2SO4 loss")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig1_h2so4_loss_vs_T.png", dpi=140)
    plt.close(fig)


def fig_h2so4_loss_vs_h2so4():
    log_h2so4 = np.linspace(6.0, 10.5, 30)
    _, r_bins, _, _ = _bins()
    pc_seed = _pc_lognormal(np.asarray(r_bins), mu_cm=1e-6, sigma_g=1.6,
                             total_num=10.0)

    fig, ax = plt.subplots(figsize=(7, 5))
    for T, color in [(210.0, "C0"), (230.0, "C1"), (250.0, "C2"), (270.0, "C3")]:
        hom = []
        het = []
        for lh in log_h2so4:
            h, het_ = _gas_breakdown(T, 0.5, 10**lh, pc_seed)
            hom.append(max(h, 1e-40))
            het.append(max(het_, 1e-40))
        ax.plot(log_h2so4, hom, color=color, lw=2, ls="-", label=f"hom T={int(T)} K")
        ax.plot(log_h2so4, het, color=color, lw=1.5, ls="--")
    ax.set_xlabel(r"log$_{10}$(H$_2$SO$_4$ [cm$^{-3}$])")
    ax.set_ylabel("H$_2$SO$_4$ consumption [g/cm$^3$/s]")
    ax.set_yscale("log")
    ax.set_title("gasexchange: H2SO4 loss vs ambient H2SO4 (hom solid, het dashed)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig2_h2so4_loss_vs_h2so4.png", dpi=140)
    plt.close(fig)


def fig_h2so4_loss_map():
    T_grid = np.linspace(200.0, 285.0, 24)
    log_h2so4 = np.linspace(6.0, 10.5, 24)
    _, r_bins, _, _ = _bins()
    pc_seed = _pc_lognormal(np.asarray(r_bins), mu_cm=1e-6, sigma_g=1.6,
                             total_num=10.0)

    total_loss = np.zeros((len(T_grid), len(log_h2so4)))
    dominance = np.zeros_like(total_loss)
    for i, T in enumerate(T_grid):
        for j, lh in enumerate(log_h2so4):
            h, het_ = _gas_breakdown(T, 0.5, 10**lh, pc_seed)
            total_loss[i, j] = max(h + het_, 1e-40)
            dominance[i, j] = np.log10(max(het_, 1e-40) / max(h, 1e-40))

    fig, ax = plt.subplots(figsize=(8, 5))
    cs = ax.contourf(log_h2so4, T_grid, np.log10(total_loss),
                      levels=20, cmap="viridis")
    cbar = fig.colorbar(cs, ax=ax)
    cbar.set_label(r"log$_{10}$(total H$_2$SO$_4$ loss [g/cm$^3$/s])")
    cl = ax.contour(log_h2so4, T_grid, dominance,
                     levels=[0], colors="red", linewidths=1.5)
    if len(cl.allsegs[0]) > 0:
        ax.clabel(cl, fmt="het = hom", inline=True, fontsize=9)
    ax.set_xlabel(r"log$_{10}$(H$_2$SO$_4$ [cm$^{-3}$])")
    ax.set_ylabel("temperature [K]")
    ax.set_title(
        r"gasexchange total H2SO4 loss (red line: heterogeneous = homogeneous)")
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig3_h2so4_loss_map.png", dpi=140)
    plt.close(fig)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig_h2so4_loss_vs_T()
    fig_h2so4_loss_vs_h2so4()
    fig_h2so4_loss_map()
    print(f"Figures saved in {OUTDIR}")


if __name__ == "__main__":
    main()
