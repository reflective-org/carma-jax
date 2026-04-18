"""Visualize nuctest with dt=0.01s (10000 steps) vs Fortran benchmark.

Proof plots showing:
1. Conservation: gas + ice volatile mass is constant
2. Core mass conservation: total core ≈ frozen sulfate mass
3. Comparison with Fortran: all elements, gas, temperature, supersaturation
4. Size distributions at t=0, 10, 50, 100s
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


def run_with_small_dt(dt=0.01, total_time=100.0):
    """Run nuctest with small dt, save state at every 1s for comparison."""
    NBIN, NGROUP, NELEM, NGAS, NZ = 16, 2, 3, 1, 1
    nstep = int(total_time / dt)
    save_every = int(1.0 / dt)  # save every 1s

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
    supsati_old = DTYPE(0.0)

    # Storage
    times_save = [0.0]
    mmr_save = []
    gas_save = [4e-5]
    ssi_save = [0.0]
    t_save = [205.0]
    core_total_save = []
    vol_total_save = []
    sulfate_total_save = []

    def store_state():
        mmr_s = np.zeros((NELEM, NBIN))
        for ib in range(NBIN):
            mmr_s[0, ib] = float(pc[0, ib, 0]) * float(rmass1[ib]) / float(rhoa[0])
            mmr_s[1, ib] = float(pc[0, ib, 1]) * float(rmass2[ib]) / float(rhoa[0])
            mmr_s[2, ib] = float(pc[0, ib, 2]) / float(rhoa[0])
        mmr_save.append(mmr_s)
        core_total_save.append(mmr_s[2].sum())
        vol_total_save.append(mmr_s[1].sum())
        sulfate_total_save.append(mmr_s[0].sum())

    store_state()

    print(f"Running {nstep} steps at dt={dt}s ({total_time}s total)...", flush=True)

    for s in range(nstep):
        df, lh, lhm = setup_grow(t, p_cgs, rhoa, zmet, 0, -1, NGAS, False)
        _, ak, aki, gro, gro1, _, _, _ = setup_gkern(
            t, p_cgs, rhoa, zmet, rmu, thcond, df, lh, lhm,
            re, r_wet, rlow_wet, jnp.ones((NBIN, NGROUP), dtype=DTYPE),
            jnp.array([1., 3.]), is_ice_arr, gwtmol_arr, igrowgas_arr,
            1., 1., 1., NBIN, NGROUP, NGAS)
        pvapl, pvapi = vaporp_h2o_murphy2005(t)
        pvapl_2d = pvapl[:, None]; pvapi_2d = pvapi[:, None]
        ssl, ssi = supersat(t, gc[:, 0], pvapl_2d[:, 0], pvapi_2d[:, 0], float(WTMOL_H2O), zmet)
        supsatl_2d = ssl[:, None]; supsati_2d = ssi[:, None]

        pconmax = jnp.zeros((1, NGROUP), dtype=DTYPE)
        for ig in range(NGROUP):
            ie = int(ienconc_arr[ig])
            pconmax = pconmax.at[:, ig].set(jnp.max(pc[:, :, ie], axis=1) / zmet)

        rk = freezaerl_koop2000(t[0], p_cgs[0], ssi[0], ssl[0], ak[0, 0], r1, vol1, DTYPE(1.38), pconmax[0, 0], NBIN)  # H2SO4 solute density
        rm = freezglaerl_murray2010(t[0], ssi[0], supsati_old, pconmax[0, 0], dt, NBIN)
        supsati_old = ssi[0]
        rnuclg = jnp.zeros((NBIN, NGROUP, NGROUP), dtype=DTYPE).at[:, 0, 1].set(rk + rm)

        prev_ice, prev_liq = totalcondensate(pc, rmass_2d, igroup_arr, is_ice_arr, igrowgas_arr, NBIN, NGROUP, NGAS, 0)

        growlg = jnp.zeros((NBIN, NGROUP), dtype=DTYPE)
        evaplg = jnp.zeros((NBIN, NGROUP), dtype=DTYPE)
        growlg, evaplg = growevapl(
            pc, growlg, evaplg, supsatl_2d, supsati_2d, pvapl_2d, pvapi_2d,
            ak, aki, gro, gro1, rup_wet, rmass_2d, dm_2d, pconmax,
            pratt, prat, pden1, palr, is_ice_arr, igrowgas_arr, ienconc_arr,
            dt, 0, NBIN, NGROUP)

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
                growpe_arr = growp(pc, growpe_arr, growlg, pconmax, 0, ibin, ielem, ig, igrow)
                rnucpe_arr = upgxfer(rnucpe_arr, rnuclg, pc, rmass_2d, pconmax, ielem, ibin, 0,
                    nuc_tables['nnucelem'], nuc_tables['inucelem'],
                    nuc_tables['nnucbin'], nuc_tables['inucbin'],
                    igroup_arr, itype_arr, NBIN, NGROUP)
                pc, pc_nucl = psolve(pc, pc_nucl, growpe_arr, evappe_arr, rnucpe_arr, rhompe_arr,
                    growlg, evaplg, rnuclg, dt, 0, ibin, ielem, ig, NGROUP)

        evappe_arr = jnp.zeros((NBIN, NELEM), dtype=DTYPE)
        evappe_arr = evapp(pc, evappe_arr, evaplg, pconmax, ienconc_arr, itype_arr, igroup_arr, 0, NBIN, NGROUP, NELEM)
        pc = downgevapply(pc, evappe_arr, jnp.zeros((NBIN, NELEM), dtype=DTYPE), dt, 0, NBIN, NELEM)

        curr_ice, curr_liq = totalcondensate(pc, rmass_2d, igroup_arr, is_ice_arr, igrowgas_arr, NBIN, NGROUP, NGAS, 0)
        gasprod = (prev_ice[0] - curr_ice[0] + prev_liq[0] - curr_liq[0]) / dt
        ice_change = prev_ice[0] - curr_ice[0]
        liq_change = prev_liq[0] - curr_liq[0]
        gc = gc.at[0, 0].add(dt * gasprod)
        rlprod = -(ice_change * (lh[0, 0] + lhm[0, 0]) + liq_change * lh[0, 0]) / (CP * rhoa[0] * dt)
        t = t.at[0].add(dt * rlprod)

        if (s + 1) % save_every == 0:
            time_s = dt * (s + 1)
            times_save.append(time_s)
            gas_save.append(float(gc[0, 0] / rhoa[0]))
            ssi_save.append(float(ssi[0]))
            t_save.append(float(t[0]))
            store_state()
            if (s + 1) % (save_every * 10) == 0:
                print(f"  t={time_s:.0f}s: gas={gas_save[-1]*1e6:.1f}ppm, ssi={ssi_save[-1]:.3f}, "
                      f"core={core_total_save[-1]:.3e}", flush=True)

    return {
        "times": np.array(times_save),
        "mmr": mmr_save,
        "gas": np.array(gas_save),
        "ssi": np.array(ssi_save),
        "t": np.array(t_save),
        "core_total": np.array(core_total_save),
        "vol_total": np.array(vol_total_save),
        "sulfate_total": np.array(sulfate_total_save),
        "r_um1": np.array(r1) * 1e4,
        "r_um2": np.array(r2) * 1e4,
        "NBIN": NBIN,
    }


def main():
    bench_path = (Path(__file__).parent.parent.parent
                  / "original-carma" / "CARMA" / "tests" / "bench" / "carma_nuctest.txt")
    fortran = parse_nuctest_bench(bench_path)

    jax_data = run_with_small_dt(dt=0.01, total_time=100.0)

    outdir = Path("plots/nuctest_validation")
    outdir.mkdir(parents=True, exist_ok=True)

    tj = jax_data["times"]
    tf = fortran["times"]
    NBIN = jax_data["NBIN"]

    # ================================================================
    # FIGURE 1: Conservation proof
    # ================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Volatile mass conservation (gas + ice_vol should be constant)
    ax = axes[0]
    gas_plus_vol = np.array(jax_data["gas"]) + np.array(jax_data["vol_total"])
    conserv_err = np.abs(gas_plus_vol - gas_plus_vol[0]) / gas_plus_vol[0]
    ax.plot(tj, conserv_err, "b-", lw=2)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Relative error")
    ax.set_title("Volatile Mass Conservation\n(gas + ice_volatile = const)")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # Core mass conservation
    ax = axes[1]
    ax.plot(tj, jax_data["core_total"], "b-", lw=2, label="JAX core mass")
    fort_core = [fortran["mmr"][i][2].sum() for i in range(len(tf))]
    ax.plot(tf, fort_core, "r--", lw=2, label="Fortran core mass")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Total Core MMR")
    ax.set_title("Core Mass (frozen sulfate)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Temperature
    ax = axes[2]
    ax.plot(tj, jax_data["t"], "b-", lw=2, label="JAX")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("T [K]")
    ax.set_title("Temperature (latent heating)")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Conservation & Physical Consistency (dt=0.01s, 10000 steps)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outdir / "proof_conservation.png", dpi=150)
    plt.close(fig)

    # ================================================================
    # FIGURE 2: JAX vs Fortran time evolution
    # ================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    ax = axes[0, 0]
    ax.plot(tj, jax_data["sulfate_total"], "b-", lw=2, label="JAX")
    fort_sulf = [fortran["mmr"][i][0].sum() for i in range(len(tf))]
    ax.plot(tf, fort_sulf, "r--", lw=2, label="Fortran")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Total MMR")
    ax.set_title("Sulfate"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(tj, jax_data["vol_total"], "b-", lw=2, label="JAX")
    fort_vol = [fortran["mmr"][i][1].sum() for i in range(len(tf))]
    ax.plot(tf, fort_vol, "r--", lw=2, label="Fortran")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Total MMR")
    ax.set_title("Ice Volatile"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(tj, np.array(jax_data["gas"]) * 1e6, "b-", lw=2, label="JAX")
    ax.plot(tf, fortran["gas"] * 1e6, "r--", lw=2, label="Fortran")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("H2O [ppm]")
    ax.set_title("Water Vapor"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(tj, jax_data["core_total"], "b-", lw=2, label="JAX")
    ax.plot(tf, fort_core, "r--", lw=2, label="Fortran")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Total MMR")
    ax.set_title("Ice Core"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(tj[1:], jax_data["ssi"][1:], "b-", lw=2, label="JAX (S-1)")
    ax.plot(tf, fortran["ssi"] - 1, "r--", lw=2, label="Fortran (S-1)")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Ice Supersaturation")
    ax.set_title("Supersaturation"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.plot(tj, jax_data["t"], "b-", lw=2, label="JAX")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("T [K]")
    ax.set_title("Temperature"); ax.legend(); ax.grid(True, alpha=0.3)

    fig.suptitle("JAX (dt=0.01s) vs Fortran (dt=1s): Full 100s Evolution", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outdir / "proof_jax_vs_fortran.png", dpi=150)
    plt.close(fig)

    # ================================================================
    # FIGURE 3: Size distributions at key times
    # ================================================================
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    r_um1 = jax_data["r_um1"]
    r_um2 = jax_data["r_um2"]

    for col, (si, label) in enumerate([(0, "t=0s"), (10, "t=10s"), (50, "t=50s"), (100, "t=100s")]):
        for row, (ie, ename, r_um, ylim) in enumerate([
            (0, "Sulfate", r_um1, (1e-35, 1e-5)),
            (1, "Ice Volatile", r_um2, (1e-20, 1e-3)),
            (2, "Ice Core", r_um2, (1e-16, 1e-8)),
        ]):
            ax = axes[row, col]
            jax_vals = jax_data["mmr"][si][ie]
            fort_vals = fortran["mmr"][si][ie] if si < len(fortran["mmr"]) else np.zeros(NBIN)
            ax.semilogy(r_um, np.maximum(jax_vals, 1e-50), "b-o", ms=3, lw=1.5, label="JAX dt=0.01")
            ax.semilogy(r_um, np.maximum(fort_vals, 1e-50), "r--s", ms=2, lw=1, label="Fortran dt=1")
            ax.set_xlabel("Radius [um]"); ax.set_ylabel("MMR")
            ax.set_title(f"{ename} — {label}")
            ax.legend(fontsize=7); ax.set_ylim(ylim); ax.grid(True, alpha=0.3)

    fig.suptitle("Size Distributions: JAX (dt=0.01s) vs Fortran", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outdir / "proof_size_distributions.png", dpi=150)
    plt.close(fig)

    # Print summary
    print(f"\nPlots saved to: {outdir.resolve()}")
    print(f"\n{'='*60}")
    print(f"SUMMARY: JAX dt=0.01s vs Fortran dt=1s")
    print(f"{'='*60}")
    jf = jax_data["mmr"][-1]
    ff = fortran["mmr"][-1]
    for ie, name in enumerate(["Sulfate", "Ice Volatile", "Ice Core"]):
        jt = jf[ie].sum(); ft = ff[ie].sum()
        err = abs(jt - ft) / max(abs(ft), 1e-50)
        print(f"  {name:15s}: JAX={jt:.4e}, Fort={ft:.4e}, err={err:.2e}")
    ge = abs(jax_data["gas"][-1] - fortran["gas"][-1]) / fortran["gas"][-1]
    print(f"  {'Gas':15s}: JAX={jax_data['gas'][-1]:.6e}, Fort={fortran['gas'][-1]:.6e}, err={ge:.2e}")
    print(f"  {'Volatile conserv':15s}: {conserv_err[-1]:.2e}")


if __name__ == "__main__":
    main()
