"""Bin-by-bin relative error at t=1s: JAX vs Fortran for all elements.

Shows exactly which bins have errors and of what magnitude.
Uses single-step validation (t=0→1s) where physics is known to work.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from carma.precision import DTYPE
from carma.constants import *
from carma.bins import setup_bins
from carma.setup_atm import setup_atm
from carma.setup_grow import setup_grow
from carma.setup_gkern import setup_gkern
from carma.setup_nuc import setup_nuc
from carma.growth.growevapl import growevapl
from carma.growth.growp import growp
from carma.growth.evapp import evapp, downgevapply
from carma.growth.upgxfer import upgxfer
from carma.solvers.psolve import psolve
from carma.solvers.totalcondensate import totalcondensate
from carma.vapor_pressure import vaporp_h2o_murphy2005
from carma.supersaturation import supersat
from carma.nucleation.freezaerl_koop2000 import freezaerl_koop2000
from carma.nucleation.freezglaerl_murray2010 import freezglaerl_murray2010
from carma.enums import GridType, ElementType, NucProcess
from scripts.run_nuctest_jax import compute_ppm_coefficients, parse_nuctest_bench


def run_one_step():
    """Run one timestep and return JAX MMR per element per bin."""
    NBIN, NGROUP, NELEM, NGAS, NZ = 16, 2, 3, 1, 1
    DTIME = 1.0

    r1, rmass1, vol1, dr1, dm1, rup1, rlow1, rmassup1 = setup_bins(1e-7, 4.0, NBIN, 1.78)
    r2, rmass2, vol2, dr2, dm2, rup2, rlow2, rmassup2 = setup_bins(5e-5, 4.0, NBIN, float(RHO_I))

    igroup_arr = np.array([0, 1, 1], dtype=np.int32)
    itype_arr = np.array([int(ElementType.I_INVOLATILE), int(ElementType.I_VOLATILE), int(ElementType.I_COREMASS)])
    igrowgas_arr = np.array([-1, 0, -1], dtype=np.int32)
    ienconc_arr = np.array([0, 1], dtype=np.int32)
    is_ice_arr = np.array([False, True])
    gwtmol_arr = np.array([float(WTMOL_H2O)])

    nf = int(NucProcess.I_AERFREEZE) + int(NucProcess.I_AF_KOOP_2000) + int(NucProcess.I_AF_MURRAY_2010)
    nuc_tables = setup_nuc(NBIN, NGROUP, NELEM,
        groups_rmass=[np.array(rmass1), np.array(rmass2)],
        nuc_from_elem=[0, 0], nuc_to_elem=[1, 2], nuc_proc=[nf, nf])

    pratt, prat, pden1, palr = compute_ppm_coefficients(
        [np.array(dm1), np.array(dm2)], [np.array(rmass1), np.array(rmass2)],
        [np.array(rmassup1), np.array(rmassup2)], 4.0, NBIN, NGROUP)

    rmass_2d = jnp.stack([rmass1, rmass2], axis=1)
    dm_2d = jnp.stack([dm1, dm2], axis=1)
    rup_wet = jnp.stack([rup1[None, :], rup2[None, :]], axis=2)
    rlow_wet = jnp.stack([rlow1[None, :], rlow2[None, :]], axis=2)
    r_wet = jnp.stack([r1[None, :], r2[None, :]], axis=2)
    re = jnp.zeros((1, NBIN, NGROUP), dtype=DTYPE)

    t = jnp.array([DTYPE(205.0)])
    p_cgs = jnp.array([DTYPE(9e4)])
    zc = jnp.array([DTYPE(17e5)])
    zl = jnp.array([DTYPE(169e4), DTYPE(171e4)])
    rho_a = float(9e4) / float(R_AIR * 205)
    pl = jnp.array([(9000 + 100 * rho_a * float(GRAV) / 100) * float(RPA2CGS),
                     (9000 - 100 * rho_a * float(GRAV) / 100) * float(RPA2CGS)])
    rhoa, dz, zmet, zmetl, rmu, thcond, _ = setup_atm(t, p_cgs, pl, zc, zl, GridType.I_CART)

    pc = jnp.full((1, NBIN, NELEM), SMALL_PC, dtype=DTYPE)
    ri = DTYPE(1e5) / R_AIR / DTYPE(200.0)
    for ib in range(NBIN):
        rv = float(r1[ib]); dv = float(dr1[ib]); ls = np.log(1.5)
        dn = 100. / (rv * np.sqrt(2 * np.pi) * ls) * np.exp(-np.log(rv / 2.5e-6)**2 / (2 * ls**2))
        ne = dn * dv * float(rhoa[0] / zmet[0]) / float(ri)
        pc = pc.at[0, ib, 0].set(DTYPE(ne) * zmet[0])
    gc = jnp.array([[DTYPE(4e-5) * rhoa[0]]])
    iz = 0

    # Store t=0 MMR
    mmr0 = np.zeros((NELEM, NBIN))
    for ib in range(NBIN):
        mmr0[0, ib] = float(pc[0, ib, 0]) * float(rmass1[ib]) / float(rhoa[0])
        mmr0[1, ib] = float(pc[0, ib, 1]) * float(rmass2[ib]) / float(rhoa[0])
        mmr0[2, ib] = float(pc[0, ib, 2]) / float(rhoa[0])

    # --- One step ---
    diffus, rlhe, rlhm = setup_grow(t, p_cgs, rhoa, zmet, 0, -1, NGAS, False)
    _, akelvin, akelvini, gro, gro1, _, _, _ = setup_gkern(
        t, p_cgs, rhoa, zmet, rmu, thcond, diffus, rlhe, rlhm,
        re, r_wet, rlow_wet, jnp.ones((NBIN, NGROUP), dtype=DTYPE),
        jnp.array([1.0, 3.0]), is_ice_arr, gwtmol_arr, igrowgas_arr,
        1.0, 1.0, 1.0, NBIN, NGROUP, NGAS)

    pvapl, pvapi = vaporp_h2o_murphy2005(t)
    pvapl_2d = pvapl[:, None]; pvapi_2d = pvapi[:, None]
    ssl, ssi = supersat(t, gc[:, 0], pvapl_2d[:, 0], pvapi_2d[:, 0], float(WTMOL_H2O), zmet)
    supsatl_2d = ssl[:, None]; supsati_2d = ssi[:, None]

    pconmax = jnp.zeros((1, NGROUP), dtype=DTYPE)
    for ig in range(NGROUP):
        ie = int(ienconc_arr[ig])
        pconmax = pconmax.at[:, ig].set(jnp.max(pc[:, :, ie], axis=1) / zmet)

    rk = freezaerl_koop2000(t[0], p_cgs[0], ssi[0], ssl[0], akelvin[0, 0],
                             r1, vol1, DTYPE(1.38), pconmax[0, 0], NBIN)  # H2SO4 solute density
    rm = freezglaerl_murray2010(t[0], ssi[0], DTYPE(0.0), pconmax[0, 0], DTIME, NBIN)
    rnuclg = jnp.zeros((NBIN, NGROUP, NGROUP), dtype=DTYPE).at[:, 0, 1].set(rk + rm)

    prev_ice, prev_liq = totalcondensate(pc, rmass_2d, igroup_arr, is_ice_arr, igrowgas_arr, NBIN, NGROUP, NGAS, iz)

    growlg = jnp.zeros((NBIN, NGROUP), dtype=DTYPE)
    evaplg = jnp.zeros((NBIN, NGROUP), dtype=DTYPE)
    growlg, evaplg = growevapl(pc, growlg, evaplg, supsatl_2d, supsati_2d, pvapl_2d, pvapi_2d,
        akelvin, akelvini, gro, gro1, rup_wet, rmass_2d, dm_2d, pconmax,
        pratt, prat, pden1, palr, is_ice_arr, igrowgas_arr, ienconc_arr,
        DTIME, iz, NBIN, NGROUP)

    growpe_arr = jnp.zeros((NBIN, NELEM), dtype=DTYPE)
    evappe_arr = jnp.zeros((NBIN, NELEM), dtype=DTYPE)
    rnucpe_arr = jnp.zeros((NBIN, NELEM), dtype=DTYPE)
    rhompe_arr = jnp.zeros((NBIN, NELEM), dtype=DTYPE)
    pc_nucl = jnp.zeros_like(pc)

    for ielem in range(NELEM):
        ig = int(igroup_arr[ielem])
        iepart = int(ienconc_arr[ig])
        igrow = int(igrowgas_arr[iepart])
        for ibin in range(NBIN):
            growpe_arr = growp(pc, growpe_arr, growlg, pconmax, iz, ibin, ielem, ig, igrow)
            rnucpe_arr = upgxfer(rnucpe_arr, rnuclg, pc, rmass_2d, pconmax, ielem, ibin, iz,
                nuc_tables['nnucelem'], nuc_tables['inucelem'],
                nuc_tables['nnucbin'], nuc_tables['inucbin'],
                igroup_arr, itype_arr, NBIN, NGROUP)
            pc, pc_nucl = psolve(pc, pc_nucl, growpe_arr, evappe_arr, rnucpe_arr, rhompe_arr,
                growlg, evaplg, rnuclg, DTIME, iz, ibin, ielem, ig, NGROUP)

    evappe_arr = jnp.zeros((NBIN, NELEM), dtype=DTYPE)
    evappe_arr = evapp(pc, evappe_arr, evaplg, pconmax, ienconc_arr, itype_arr,
                       igroup_arr, iz, NBIN, NGROUP, NELEM)
    pc = downgevapply(pc, evappe_arr, jnp.zeros((NBIN, NELEM), dtype=DTYPE), DTIME, iz, NBIN, NELEM)

    curr_ice, curr_liq = totalcondensate(pc, rmass_2d, igroup_arr, is_ice_arr, igrowgas_arr, NBIN, NGROUP, NGAS, iz)
    gasprod = (prev_ice[0] - curr_ice[0] + prev_liq[0] - curr_liq[0]) / DTIME
    gc = gc.at[iz, 0].add(DTIME * gasprod)

    # Store t=1 MMR
    mmr1 = np.zeros((NELEM, NBIN))
    for ib in range(NBIN):
        mmr1[0, ib] = float(pc[0, ib, 0]) * float(rmass1[ib]) / float(rhoa[0])
        mmr1[1, ib] = float(pc[0, ib, 1]) * float(rmass2[ib]) / float(rhoa[0])
        mmr1[2, ib] = float(pc[0, ib, 2]) / float(rhoa[0])

    # Also return nucleation rates for diagnostics
    nuc_rates = np.array([float(rk[i] + rm[i]) for i in range(NBIN)])

    return mmr0, mmr1, float(gc[0, 0] / rhoa[0]), nuc_rates, r1, rmass1, r2, rmass2, NBIN


def main():
    bench_path = (Path(__file__).parent.parent.parent
                  / "original-carma" / "CARMA" / "tests" / "bench" / "carma_nuctest.txt")
    fortran = parse_nuctest_bench(bench_path)

    mmr0_jax, mmr1_jax, gas_jax, nuc_rates, r1, rmass1, r2, rmass2, NBIN = run_one_step()
    mmr0_fort = fortran["mmr"][0]
    mmr1_fort = fortran["mmr"][1]

    r_um1 = np.array(r1) * 1e4
    r_um2 = np.array(r2) * 1e4

    outdir = Path("plots/nuctest_validation")
    outdir.mkdir(parents=True, exist_ok=True)

    elem_names = ["Element 1: Sulfate", "Element 2: Ice Volatile", "Element 3: Ice Core"]
    r_ums = [r_um1, r_um2, r_um2]

    # ================================================================
    # FIGURE: Bin-by-bin relative error + values at t=0 and t=1
    # ================================================================
    fig, axes = plt.subplots(3, 3, figsize=(20, 14))

    for ie in range(3):
        r_um = r_ums[ie]

        # Column 1: t=0 comparison
        ax = axes[ie, 0]
        j0 = mmr0_jax[ie]
        f0 = mmr0_fort[ie]
        ax.semilogy(r_um, np.maximum(j0, 1e-50), "b-o", ms=4, lw=1.5, label="JAX")
        ax.semilogy(r_um, np.maximum(f0, 1e-50), "r--s", ms=3, lw=1, label="Fortran")
        ax.set_xlabel("Radius [um]")
        ax.set_ylabel("MMR [g/g]")
        ax.set_title(f"{elem_names[ie]}\nt=0 (initial)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Column 2: t=1 comparison
        ax = axes[ie, 1]
        j1 = mmr1_jax[ie]
        f1 = mmr1_fort[ie]
        ax.semilogy(r_um, np.maximum(j1, 1e-50), "b-o", ms=4, lw=1.5, label="JAX")
        ax.semilogy(r_um, np.maximum(f1, 1e-50), "r--s", ms=3, lw=1, label="Fortran")
        ax.set_xlabel("Radius [um]")
        ax.set_ylabel("MMR [g/g]")
        ax.set_title(f"{elem_names[ie]}\nt=1s (after 1 step)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Column 3: Relative error at t=1
        ax = axes[ie, 2]
        # Only compute error for bins where Fortran has signal
        rel_err = np.full(NBIN, np.nan)
        for ib in range(NBIN):
            if f1[ib] > 1e-30:
                rel_err[ib] = (j1[ib] - f1[ib]) / f1[ib] * 100  # signed %

        colors = ["green" if abs(e) < 5 else "orange" if abs(e) < 20 else "red"
                  for e in rel_err]
        bars = ax.bar(np.arange(1, NBIN + 1), rel_err, color=colors, edgecolor="black", lw=0.5)
        ax.axhline(0, color="black", lw=0.5)
        ax.axhline(5, color="green", ls="--", alpha=0.5, lw=0.8)
        ax.axhline(-5, color="green", ls="--", alpha=0.5, lw=0.8)
        ax.axhline(20, color="red", ls="--", alpha=0.5, lw=0.8)
        ax.axhline(-20, color="red", ls="--", alpha=0.5, lw=0.8)
        ax.set_xlabel("Bin number")
        ax.set_ylabel("Relative error [%]")
        ax.set_title(f"{elem_names[ie]}\nRelative Error at t=1s")
        ax.set_ylim(-30, 10)
        ax.grid(True, alpha=0.3)

        # Annotate bins with large errors
        for ib in range(NBIN):
            if not np.isnan(rel_err[ib]) and abs(rel_err[ib]) > 5:
                ax.annotate(f"{rel_err[ib]:.1f}%", (ib + 1, rel_err[ib]),
                           textcoords="offset points", xytext=(0, 8 if rel_err[ib] > 0 else -12),
                           fontsize=7, ha="center")

    fig.suptitle("Bin-by-Bin Comparison: JAX vs Fortran at t=0s and t=1s\n"
                 "Green=<5%, Orange=5-20%, Red=>20%", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outdir / "binbybin_errors_t1.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {outdir / 'binbybin_errors_t1.png'}")

    # ================================================================
    # FIGURE 2: Nucleation rate comparison + sulfate error decomposition
    # ================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Koop nucleation rate per bin
    ax = axes[0]
    ax.semilogy(np.arange(1, NBIN + 1), np.maximum(nuc_rates, 1e-30), "b-o", ms=5, lw=2)
    ax.set_xlabel("Bin number")
    ax.set_ylabel("Nucleation rate [1/s]")
    ax.set_title("Koop+Murray Nucleation Rate per Bin")
    ax.grid(True, alpha=0.3)
    ax.axhline(1.0, color="red", ls="--", alpha=0.5, label="rate=1/s")
    ax.legend()

    # Sulfate depletion: fraction remaining
    ax = axes[1]
    fort_frac = mmr1_fort[0] / np.maximum(mmr0_fort[0], 1e-50)
    jax_frac = mmr1_jax[0] / np.maximum(mmr0_jax[0], 1e-50)
    ax.semilogy(np.arange(1, NBIN + 1), np.maximum(fort_frac, 1e-10), "r--s", ms=4, label="Fortran")
    ax.semilogy(np.arange(1, NBIN + 1), np.maximum(jax_frac, 1e-10), "b-o", ms=4, label="JAX")
    ax.set_xlabel("Bin number")
    ax.set_ylabel("Fraction remaining after 1s")
    ax.set_title("Sulfate Depletion by Nucleation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Gas error
    ax = axes[2]
    gas_fort = fortran["gas"][1]
    gas_err = (gas_jax - gas_fort) / gas_fort * 100
    ax.bar(["Gas MMR"], [gas_err], color="green" if abs(gas_err) < 1 else "orange")
    ax.set_ylabel("Relative error [%]")
    ax.set_title(f"Gas MMR at t=1s\nJAX={gas_jax:.6e}, Fort={gas_fort:.6e}")
    ax.grid(True, alpha=0.3)

    # Print detailed table
    print("\n" + "="*80)
    print("BIN-BY-BIN RELATIVE ERROR AT t=1s")
    print("="*80)
    print(f"{'Bin':>4s} {'Sulfate%':>10s} {'IceVol%':>10s} {'IceCore%':>10s} "
          f"{'NucRate':>12s} {'SulfFrac':>10s}")
    for ib in range(NBIN):
        s_err = (mmr1_jax[0, ib] - mmr1_fort[0, ib]) / mmr1_fort[0, ib] * 100 if mmr1_fort[0, ib] > 1e-30 else np.nan
        v_err = (mmr1_jax[1, ib] - mmr1_fort[1, ib]) / mmr1_fort[1, ib] * 100 if mmr1_fort[1, ib] > 1e-30 else np.nan
        c_err = (mmr1_jax[2, ib] - mmr1_fort[2, ib]) / mmr1_fort[2, ib] * 100 if mmr1_fort[2, ib] > 1e-30 else np.nan
        frac = mmr1_jax[0, ib] / mmr0_jax[0, ib] if mmr0_jax[0, ib] > 1e-50 else 0
        print(f"{ib+1:4d} {s_err:10.2f} {v_err:10.2f} {c_err:10.2f} "
              f"{nuc_rates[ib]:12.2e} {frac:10.4f}")

    print(f"\nGas: err={gas_err:.3f}%")

    fig.suptitle("Nucleation Diagnostics", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outdir / "binbybin_nucleation_diag.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {outdir / 'binbybin_nucleation_diag.png'}")


if __name__ == "__main__":
    main()
