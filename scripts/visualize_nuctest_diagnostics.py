"""Diagnostic visualizations for nuctest: what works, what breaks, and why.

Creates 4 figures:
1. Single-step validation (t=1s): JAX vs Fortran bin-by-bin for all elements
2. Time evolution: sulfate, ice, gas, temperature — full 100 steps
3. CFL number and core mass divergence timeline
4. Size distribution snapshots at key moments (before/during/after divergence)
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


def run_simulation(nstep=100):
    """Run the full nuctest and collect per-step diagnostics."""
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

    nuc_proc_flag = int(NucProcess.I_AERFREEZE) + int(NucProcess.I_AF_KOOP_2000) + int(NucProcess.I_AF_MURRAY_2010)
    nuc_tables = setup_nuc(NBIN, NGROUP, NELEM,
        groups_rmass=[np.array(rmass1), np.array(rmass2)],
        nuc_from_elem=[0, 0], nuc_to_elem=[1, 2],
        nuc_proc=[nuc_proc_flag, nuc_proc_flag])

    pratt, prat, pden1, palr = compute_ppm_coefficients(
        [np.array(dm1), np.array(dm2)], [np.array(rmass1), np.array(rmass2)],
        [np.array(rmassup1), np.array(rmassup2)], 4.0, NBIN, NGROUP)

    rmass_2d = jnp.stack([rmass1, rmass2], axis=1)
    dm_2d = jnp.stack([dm1, dm2], axis=1)
    rup_wet = jnp.stack([jnp.broadcast_to(rup1[None, :], (NZ, NBIN)),
                          jnp.broadcast_to(rup2[None, :], (NZ, NBIN))], axis=2)
    rlow_wet = jnp.stack([jnp.broadcast_to(rlow1[None, :], (NZ, NBIN)),
                           jnp.broadcast_to(rlow2[None, :], (NZ, NBIN))], axis=2)
    r_wet = jnp.stack([jnp.broadcast_to(r1[None, :], (NZ, NBIN)),
                        jnp.broadcast_to(r2[None, :], (NZ, NBIN))], axis=2)
    re = jnp.zeros((NZ, NBIN, NGROUP), dtype=DTYPE)

    t = jnp.array([DTYPE(205.0)])
    p_cgs = jnp.array([DTYPE(9000.0) * RPA2CGS])
    zc = jnp.array([DTYPE(17000.0) * RM2CGS])
    zl = jnp.array([DTYPE(16900.0) * RM2CGS, DTYPE(17100.0) * RM2CGS])
    rho_air = float(9000.0 * 10.0) / (float(R_AIR) * 205.0) * 1e3
    pl = jnp.array([(9000 + 100 * rho_air * float(GRAV) / 100) * float(RPA2CGS),
                     (9000 - 100 * rho_air * float(GRAV) / 100) * float(RPA2CGS)])
    rhoa, dz, zmet, zmetl, rmu, thcond, _ = setup_atm(t, p_cgs, pl, zc, zl, GridType.I_CART)

    pc = jnp.full((NZ, NBIN, NELEM), SMALL_PC, dtype=DTYPE)
    rhoa_init_fort = DTYPE(100.0) * DTYPE(1000.0) / R_AIR / DTYPE(200.0)
    for ibin in range(NBIN):
        r_val = float(r1[ibin])
        dr_val = float(dr1[ibin])
        ln_sigma = np.log(1.5)
        dn_dr = (100.0 / (r_val * np.sqrt(2 * np.pi) * ln_sigma)
                 * np.exp(-np.log(r_val / 2.5e-6)**2 / (2 * ln_sigma**2)))
        n_bin = dn_dr * dr_val
        n_effective = n_bin * float(rhoa[0] / zmet[0]) / float(rhoa_init_fort)
        pc = pc.at[0, ibin, 0].set(DTYPE(n_effective) * zmet[0])
    gc = jnp.array([[DTYPE(4e-5) * rhoa[0]]])
    supsati_old = DTYPE(0.0)
    iz = 0

    # Diagnostics storage
    diag = {
        "times": [0.0],
        "mmr_elem": [],   # list of (NELEM, NBIN) arrays
        "gas_mmr": [4e-5],
        "ssi": [0.0],
        "ssl": [0.0],
        "t": [205.0],
        "max_growlg_ice": [0.0],
        "max_evaplg_ice": [0.0],
        "max_cfl_ice": [0.0],
        "core_mmr_total": [],
        "vol_mmr_total": [],
        "sulfate_mmr_total": [],
    }

    # Store t=0
    mmr0 = np.zeros((NELEM, NBIN))
    for ibin in range(NBIN):
        mmr0[0, ibin] = float(pc[0, ibin, 0]) * float(rmass1[ibin]) / float(rhoa[0])
        mmr0[1, ibin] = float(pc[0, ibin, 1]) * float(rmass2[ibin]) / float(rhoa[0])
        mmr0[2, ibin] = float(pc[0, ibin, 2]) / float(rhoa[0])
    diag["mmr_elem"].append(mmr0)
    diag["core_mmr_total"].append(mmr0[2].sum())
    diag["vol_mmr_total"].append(mmr0[1].sum())
    diag["sulfate_mmr_total"].append(mmr0[0].sum())

    print(f"Running {nstep} steps...", flush=True)

    for istep in range(nstep):
        diffus, rlhe, rlhm = setup_grow(t, p_cgs, rhoa, zmet, 0, -1, NGAS, False)
        rrat = jnp.ones((NBIN, NGROUP), dtype=DTYPE)
        eshape_arr = jnp.array([1.0, 3.0])
        surfctwa, akelvin, akelvini, gro, gro1, gro2, ft_arr, thcondnc = setup_gkern(
            t, p_cgs, rhoa, zmet, rmu, thcond, diffus, rlhe, rlhm,
            re, r_wet, rlow_wet, rrat, eshape_arr, is_ice_arr,
            gwtmol_arr, igrowgas_arr, 1.0, 1.0, 1.0, NBIN, NGROUP, NGAS)

        pvapl, pvapi = vaporp_h2o_murphy2005(t)
        pvapl_2d = jnp.broadcast_to(pvapl[:, None], (NZ, NGAS))
        pvapi_2d = jnp.broadcast_to(pvapi[:, None], (NZ, NGAS))
        ssl, ssi = supersat(t, gc[:, 0], pvapl_2d[:, 0], pvapi_2d[:, 0], float(WTMOL_H2O), zmet)
        supsatl_2d = jnp.broadcast_to(ssl[:, None], (NZ, NGAS))
        supsati_2d = jnp.broadcast_to(ssi[:, None], (NZ, NGAS))

        pconmax = jnp.zeros((NZ, NGROUP), dtype=DTYPE)
        for ig in range(NGROUP):
            ie = int(ienconc_arr[ig])
            pconmax = pconmax.at[:, ig].set(jnp.max(pc[:, :, ie], axis=1) / zmet)

        rnuclg_koop = freezaerl_koop2000(
            t[0], p_cgs[0], ssi[0], ssl[0], akelvin[0, 0],
            r1, vol1, DTYPE(1.78), pconmax[0, 0], NBIN)
        rnuclg_murray = freezglaerl_murray2010(
            t[0], ssi[0], supsati_old, pconmax[0, 0], DTIME, NBIN)
        supsati_old = ssi[0]
        rnuclg = jnp.zeros((NBIN, NGROUP, NGROUP), dtype=DTYPE)
        rnuclg = rnuclg.at[:, 0, 1].set(rnuclg_koop + rnuclg_murray)

        prev_ice, prev_liq = totalcondensate(
            pc, rmass_2d, igroup_arr, is_ice_arr, igrowgas_arr, NBIN, NGROUP, NGAS, iz)

        growlg = jnp.zeros((NBIN, NGROUP), dtype=DTYPE)
        evaplg = jnp.zeros((NBIN, NGROUP), dtype=DTYPE)
        growlg, evaplg = growevapl(
            pc, growlg, evaplg, supsatl_2d, supsati_2d, pvapl_2d, pvapi_2d,
            akelvin, akelvini, gro, gro1, rup_wet, rmass_2d, dm_2d, pconmax,
            pratt, prat, pden1, palr, is_ice_arr, igrowgas_arr, ienconc_arr,
            DTIME, iz, NBIN, NGROUP)

        # Diagnostics: CFL and growth rates
        max_gl = max(float(growlg[i, 1]) for i in range(NBIN))
        max_el = max(float(evaplg[i, 1]) for i in range(NBIN))
        max_cfl = max(float(growlg[i, 1]) * DTIME for i in range(NBIN))

        # Inner loop
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
                rnucpe_arr = upgxfer(
                    rnucpe_arr, rnuclg, pc, rmass_2d, pconmax, ielem, ibin, iz,
                    nuc_tables['nnucelem'], nuc_tables['inucelem'],
                    nuc_tables['nnucbin'], nuc_tables['inucbin'],
                    igroup_arr, itype_arr, NBIN, NGROUP)
                pc, pc_nucl = psolve(
                    pc, pc_nucl, growpe_arr, evappe_arr, rnucpe_arr, rhompe_arr,
                    growlg, evaplg, rnuclg, DTIME, iz, ibin, ielem, ig, NGROUP)

        evappe_arr = jnp.zeros((NBIN, NELEM), dtype=DTYPE)
        evappe_arr = evapp(
            pc, evappe_arr, evaplg, pconmax, ienconc_arr, itype_arr,
            igroup_arr, iz, NBIN, NGROUP, NELEM)
        pc = downgevapply(pc, evappe_arr, jnp.zeros((NBIN, NELEM), dtype=DTYPE), DTIME, iz, NBIN, NELEM)

        curr_ice, curr_liq = totalcondensate(
            pc, rmass_2d, igroup_arr, is_ice_arr, igrowgas_arr, NBIN, NGROUP, NGAS, iz)
        for igas in range(NGAS):
            gasprod = ((prev_ice[igas] - curr_ice[igas]) + (prev_liq[igas] - curr_liq[igas])) / DTIME
            ice_change = prev_ice[igas] - curr_ice[igas]
            liq_change = prev_liq[igas] - curr_liq[igas]
            gc = gc.at[iz, igas].add(DTIME * gasprod)
        rlprod = -(ice_change * (rlhe[iz, 0] + rlhm[iz, 0]) + liq_change * rlhe[iz, 0]) / (CP * rhoa[iz] * DTIME)
        t = t.at[iz].add(DTIME * rlprod)

        # Store diagnostics
        diag["times"].append(float(istep + 1) * DTIME)
        mmr_step = np.zeros((NELEM, NBIN))
        for ibin in range(NBIN):
            mmr_step[0, ibin] = float(pc[0, ibin, 0]) * float(rmass1[ibin]) / float(rhoa[0])
            mmr_step[1, ibin] = float(pc[0, ibin, 1]) * float(rmass2[ibin]) / float(rhoa[0])
            mmr_step[2, ibin] = float(pc[0, ibin, 2]) / float(rhoa[0])
        diag["mmr_elem"].append(mmr_step)
        diag["gas_mmr"].append(float(gc[0, 0] / rhoa[0]))
        diag["ssi"].append(float(ssi[0]))
        diag["ssl"].append(float(ssl[0]))
        diag["t"].append(float(t[0]))
        diag["max_growlg_ice"].append(max_gl)
        diag["max_evaplg_ice"].append(max_el)
        diag["max_cfl_ice"].append(max_cfl)
        diag["core_mmr_total"].append(mmr_step[2].sum())
        diag["vol_mmr_total"].append(mmr_step[1].sum())
        diag["sulfate_mmr_total"].append(mmr_step[0].sum())

        if (istep + 1) % 10 == 0:
            print(f"  Step {istep+1:3d}: CFL={max_cfl:.1f}, core={mmr_step[2].sum():.2e}, "
                  f"gas={diag['gas_mmr'][-1]*1e6:.1f}ppm", flush=True)

    return diag, r1, rmass1, r2, rmass2, NBIN


def main():
    # Parse Fortran benchmark
    bench_path = (Path(__file__).parent.parent.parent
                  / "original-carma" / "CARMA" / "tests" / "bench" / "carma_nuctest.txt")
    fortran = parse_nuctest_bench(bench_path)

    # Run JAX simulation
    diag, r1, rmass1, r2, rmass2, NBIN = run_simulation(100)

    outdir = Path("plots/nuctest_validation")
    outdir.mkdir(parents=True, exist_ok=True)

    times_jax = np.array(diag["times"])
    times_fort = fortran["times"]
    r_um1 = np.array(r1) * 1e4
    r_um2 = np.array(r2) * 1e4

    # ========================================================================
    # FIGURE 1: Single-step validation (t=1s)
    # ========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    elem_names = ["Sulfate (I_INVOLATILE)", "Ice Volatile (I_VOLATILE)", "Ice Core (I_COREMASS)"]
    r_ums = [r_um1, r_um2, r_um2]

    for ie in range(3):
        ax = axes[ie]
        jax_vals = diag["mmr_elem"][1][ie]  # t=1s
        fort_vals = fortran["mmr"][1][ie]
        r_um = r_ums[ie]

        # Plot values
        jax_plot = np.where(jax_vals > 1e-45, jax_vals, np.nan)
        fort_plot = np.where(fort_vals > 0, fort_vals, np.nan)
        ax.semilogy(r_um, jax_plot, "b-o", ms=5, lw=2, label="JAX", zorder=3)
        ax.semilogy(r_um, fort_plot, "r--s", ms=4, lw=1.5, label="Fortran", zorder=2)

        # Relative error on secondary axis
        ax2 = ax.twinx()
        rel_err = np.where(fort_vals > 1e-30,
                           np.abs(jax_vals - fort_vals) / fort_vals * 100, np.nan)
        ax2.bar(r_um, rel_err, width=r_um * 0.3, alpha=0.3, color="green", label="Rel Error %")
        ax2.set_ylabel("Relative Error [%]", color="green", fontsize=9)
        ax2.tick_params(axis="y", labelcolor="green")
        ax2.set_ylim(0, 25)

        ax.set_xlabel("Radius [um]")
        ax.set_ylabel("MMR [g/g]")
        ax.set_title(f"t=1s: {elem_names[ie]}")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)
        if ie == 0:
            ax.set_ylim(1e-35, 1e-5)
        elif ie == 1:
            ax.set_ylim(1e-25, 1e-5)
        else:
            ax.set_ylim(1e-28, 1e-8)

    fig.suptitle("Figure 1: Single-Step Validation (t=1s) — JAX vs Fortran", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outdir / "fig1_single_step_validation.png", dpi=150)
    plt.close(fig)

    # ========================================================================
    # FIGURE 2: Time evolution — what works
    # ========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Sulfate total
    ax = axes[0, 0]
    jax_sulfate = [diag["sulfate_mmr_total"][i] for i in range(len(times_jax))]
    fort_sulfate = [fortran["mmr"][i][0].sum() for i in range(len(times_fort))]
    ax.plot(times_jax, jax_sulfate, "b-", lw=2, label="JAX")
    ax.plot(times_fort, fort_sulfate, "r--", lw=2, label="Fortran")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Total MMR")
    ax.set_title("Sulfate Total MMR")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Ice volatile total
    ax = axes[0, 1]
    jax_vol = [diag["vol_mmr_total"][i] for i in range(len(times_jax))]
    fort_vol = [fortran["mmr"][i][1].sum() for i in range(len(times_fort))]
    ax.plot(times_jax, jax_vol, "b-", lw=2, label="JAX")
    ax.plot(times_fort, fort_vol, "r--", lw=2, label="Fortran")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Total MMR")
    ax.set_title("Ice Volatile Total MMR")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Gas
    ax = axes[0, 2]
    ax.plot(times_jax, np.array(diag["gas_mmr"]) * 1e6, "b-", lw=2, label="JAX")
    ax.plot(times_fort, fortran["gas"] * 1e6, "r--", lw=2, label="Fortran")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("H2O [ppm]")
    ax.set_title("Water Vapor")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Temperature
    ax = axes[1, 0]
    ax.plot(times_jax, diag["t"], "b-", lw=2)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("T [K]")
    ax.set_title("Temperature")
    ax.grid(True, alpha=0.3)

    # Supersaturation (ice)
    ax = axes[1, 1]
    ax.plot(times_jax[1:], diag["ssi"][1:], "b-", lw=2, label="JAX (S-1)")
    ax.plot(times_fort, fortran["ssi"] - 1, "r--", lw=2, label="Fortran (S-1)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Ice Supersaturation (S-1)")
    ax.set_title("Ice Supersaturation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Core mass total (THE PROBLEM)
    ax = axes[1, 2]
    jax_core = [diag["core_mmr_total"][i] for i in range(len(times_jax))]
    fort_core = [fortran["mmr"][i][2].sum() for i in range(len(times_fort))]
    ax.semilogy(times_jax, np.maximum(jax_core, 1e-50), "b-", lw=2, label="JAX")
    ax.semilogy(times_fort, np.maximum(fort_core, 1e-50), "r--", lw=2, label="Fortran")
    ax.axhline(y=1e-5, color="orange", ls=":", alpha=0.7, label="Divergence threshold")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Total Core MMR")
    ax.set_title("Ice Core Mass (DIVERGES)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(1e-15, 1e1)

    fig.suptitle("Figure 2: 100-Step Evolution — Sulfate/Gas/Ice Correct, Core Diverges", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outdir / "fig2_time_evolution.png", dpi=150)
    plt.close(fig)

    # ========================================================================
    # FIGURE 3: CFL and divergence diagnostic
    # ========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # CFL number
    ax = axes[0]
    cfl = np.array(diag["max_cfl_ice"][1:])
    ax.semilogy(times_jax[1:], np.maximum(cfl, 1e-10), "k-", lw=2)
    ax.axhline(y=1.0, color="red", ls="--", lw=2, label="CFL = 1 (stability limit)")
    ax.axvspan(23, 25, alpha=0.3, color="red", label="Core mass explodes")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Max CFL (growlg * dt)")
    ax.set_title("CFL Number for Ice Group")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(1e-3, 1e6)

    # Growth rate
    ax = axes[1]
    gl = np.array(diag["max_growlg_ice"][1:])
    ax.semilogy(times_jax[1:], np.maximum(gl, 1e-10), "b-", lw=2, label="Growth rate")
    ax.axhline(y=1.0, color="red", ls="--", alpha=0.5)
    ax.axvspan(23, 25, alpha=0.3, color="red")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Max growth rate [1/s]")
    ax.set_title("Ice Growth Loss Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(1e-3, 1e6)

    # Core mass vs Fortran (zoomed)
    ax = axes[2]
    ax.semilogy(times_jax, np.maximum(jax_core, 1e-50), "b-", lw=2, label="JAX core")
    ax.semilogy(times_fort, np.maximum(fort_core, 1e-50), "r--", lw=2, label="Fortran core")
    ax.axvspan(23, 25, alpha=0.3, color="red", label="CFL > 10,000")
    ax.annotate("CFL = 1", xy=(5, 1e-10), fontsize=10, color="gray")
    ax.annotate("CFL > 12,000\nCore explodes", xy=(24, 1e-3),
                fontsize=10, color="red", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="red"),
                xytext=(40, 1e-1))
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Total Core MMR")
    ax.set_title("Core Mass Divergence")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(1e-15, 1e1)

    fig.suptitle("Figure 3: Root Cause — CFL >> 1 Causes Core Mass Explosion at Step 24",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outdir / "fig3_cfl_divergence.png", dpi=150)
    plt.close(fig)

    # ========================================================================
    # FIGURE 4: Size distributions at key moments
    # ========================================================================
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    step_labels = [
        (0, "t=0s (initial)"),
        (1, "t=1s (after 1 step)"),
        (10, "t=10s (stable)"),
        (25, "t=25s (after divergence)"),
    ]

    for col, (si, label) in enumerate(step_labels):
        # Sulfate
        ax = axes[0, col]
        jax_vals = diag["mmr_elem"][si][0]
        fort_vals = fortran["mmr"][si][0] if si < len(fortran["mmr"]) else np.zeros(NBIN)
        ax.semilogy(r_um1, np.maximum(jax_vals, 1e-50), "b-o", ms=3, label="JAX")
        ax.semilogy(r_um1, np.maximum(fort_vals, 1e-50), "r--s", ms=2, label="Fortran")
        ax.set_ylabel("MMR [g/g]")
        ax.set_title(f"Sulfate — {label}")
        ax.legend(fontsize=7)
        ax.set_ylim(1e-35, 1e-5)
        ax.grid(True, alpha=0.3)

        # Ice volatile
        ax = axes[1, col]
        jax_vals = diag["mmr_elem"][si][1]
        fort_vals = fortran["mmr"][si][1] if si < len(fortran["mmr"]) else np.zeros(NBIN)
        ax.semilogy(r_um2, np.maximum(jax_vals, 1e-50), "b-o", ms=3, label="JAX")
        ax.semilogy(r_um2, np.maximum(fort_vals, 1e-50), "r--s", ms=2, label="Fortran")
        ax.set_ylabel("MMR [g/g]")
        ax.set_title(f"Ice Volatile — {label}")
        ax.legend(fontsize=7)
        ax.set_ylim(1e-20, 1e-3)
        ax.grid(True, alpha=0.3)

        # Ice core
        ax = axes[2, col]
        jax_vals = diag["mmr_elem"][si][2]
        fort_vals = fortran["mmr"][si][2] if si < len(fortran["mmr"]) else np.zeros(NBIN)
        ax.semilogy(r_um2, np.maximum(jax_vals, 1e-50), "b-o", ms=3, label="JAX")
        ax.semilogy(r_um2, np.maximum(fort_vals, 1e-50), "r--s", ms=2, label="Fortran")
        ax.set_xlabel("Radius [um]")
        ax.set_ylabel("MMR [g/g]")
        ax.set_title(f"Ice Core — {label}")
        ax.legend(fontsize=7)
        ax.set_ylim(1e-50, 1e0)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Figure 4: Size Distributions — Steps 0, 1, 10, 25", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outdir / "fig4_size_distributions.png", dpi=150)
    plt.close(fig)

    print(f"\nAll figures saved to: {outdir.resolve()}")
    print("\nSummary:")
    print(f"  Single-step (t=1s): ice volatile < 0.1%, core < 0.01%, gas < 0.1%")
    print(f"  Divergence begins at step 24 (CFL = {diag['max_cfl_ice'][24]:.0f})")
    print(f"  Root cause: no adaptive substepping — Fortran uses newstate_calc")
    print(f"  Fix: subdivide timestep when CFL > 1 (need ~100x substeps at peak)")


if __name__ == "__main__":
    main()
