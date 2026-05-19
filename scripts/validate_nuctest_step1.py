"""Validate a SINGLE timestep of JAX nuctest against Fortran t=1s.

Isolates per-step physics accuracy before testing accumulation over 100 steps.
Compares all 3 elements bin-by-bin, plus gas and supersaturation.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
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


def parse_nuctest_bench(filepath):
    """Parse carma_nuctest.txt benchmark (same as in run_nuctest_jax.py)."""
    with open(filepath) as f:
        lines = f.readlines()
    idx = 0
    parts = lines[idx].split()
    ngroup, nelem, nbin, ngas = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
    idx += 1
    radii, rmass_f = {}, {}
    for ig in range(ngroup):
        radii[ig] = np.zeros(nbin)
        rmass_f[ig] = np.zeros(nbin)
        for i in range(nbin):
            parts = lines[idx].split()
            radii[ig][i] = float(parts[2]) * 1e-4
            rmass_f[ig][i] = float(parts[3])
            idx += 1
    times, mmr_data, gas_data, ssl_data, ssi_data = [], [], [], [], []
    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue
        try:
            time_val = float(line)
        except ValueError:
            idx += 1
            continue
        times.append(time_val)
        idx += 1
        mmr = np.zeros((nelem, nbin))
        for ie in range(nelem):
            for ib in range(nbin):
                parts = lines[idx].split()
                mmr[ie, ib] = float(parts[2])
                idx += 1
        mmr_data.append(mmr)
        parts = lines[idx].split()
        gas_data.append(float(parts[1]))
        ssl_data.append(float(parts[2]))
        ssi_data.append(float(parts[3]))
        idx += 1
    return {
        "ngroup": ngroup, "nelem": nelem, "nbin": nbin, "ngas": ngas,
        "radii": radii, "rmass": rmass_f,
        "times": np.array(times), "mmr": mmr_data,
        "gas": np.array(gas_data), "ssl": np.array(ssl_data), "ssi": np.array(ssi_data),
    }


def main():
    NBIN, NGROUP, NELEM, NGAS, NZ = 16, 2, 3, 1, 1
    DTIME = 1.0

    # Parse Fortran benchmark
    bench_path = (Path(__file__).parent.parent.parent
                  / "original-carma" / "CARMA" / "tests" / "bench" / "carma_nuctest.txt")
    fortran = parse_nuctest_bench(bench_path)
    print(f"Fortran benchmark: {len(fortran['times'])} timesteps")

    # --- Setup (identical to run_nuctest_jax.py) ---
    rmin1, rmrat1, rho1 = 1e-7, 4.0, 1.78
    r1, rmass1, vol1, dr1, dm1, rup1, rlow1, rmassup1 = setup_bins(rmin1, rmrat1, NBIN, rho1)
    rmin2, rmrat2, rho2 = 5e-5, 4.0, float(RHO_I)
    r2, rmass2, vol2, dr2, dm2, rup2, rlow2, rmassup2 = setup_bins(rmin2, rmrat2, NBIN, rho2)

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

    # PPM coefficients
    from scripts.run_nuctest_jax import compute_ppm_coefficients
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

    # Atmosphere
    T_init = 205.0
    p_pa = 9000.0
    t = jnp.array([DTYPE(T_init)])
    p_cgs = jnp.array([DTYPE(p_pa) * RPA2CGS])
    zc = jnp.array([DTYPE(17000.0) * RM2CGS])
    zl = jnp.array([DTYPE(16900.0) * RM2CGS, DTYPE(17100.0) * RM2CGS])
    rho_air = float(p_pa * 10.0) / (float(R_AIR) * T_init) * 1e3
    pl = jnp.array([(p_pa + 100 * rho_air * float(GRAV) / 100) * float(RPA2CGS),
                     (p_pa - 100 * rho_air * float(GRAV) / 100) * float(RPA2CGS)])
    rhoa, dz, zmet, zmetl, rmu, thcond, rhoa_wet = setup_atm(t, p_cgs, pl, zc, zl, GridType.I_CART)

    # Initial conditions — match Fortran's approximate rhoa for init conversion
    pc = jnp.full((NZ, NBIN, NELEM), SMALL_PC, dtype=DTYPE)
    rhoa_init_fort = DTYPE(100.0) * DTYPE(1000.0) / R_AIR / DTYPE(200.0)
    for ibin in range(NBIN):
        r_val = float(r1[ibin])
        dr_val = float(dr1[ibin])
        ln_sigma = np.log(1.5)
        dn_dr = (100.0 / (r_val * np.sqrt(2 * np.pi) * ln_sigma)
                 * np.exp(-np.log(r_val / 2.5e-6)**2 / (2 * ln_sigma**2)))
        n_bin = dn_dr * dr_val
        rhoa_wet_val = float(rhoa[0] / zmet[0])
        n_effective = n_bin * rhoa_wet_val / float(rhoa_init_fort)
        pc = pc.at[0, ibin, 0].set(DTYPE(n_effective) * zmet[0])
    gc = jnp.array([[DTYPE(4e-5) * rhoa[0]]])
    iz = 0

    # --- Verify initial condition against Fortran t=0 ---
    print("\n=== Initial condition (t=0) comparison ===")
    print(f"{'Bin':>4s} {'JAX sulfate':>12s} {'Fort sulfate':>12s} {'RelErr':>10s}")
    for ibin in range(NBIN):
        jax_mmr = float(pc[0, ibin, 0]) * float(rmass1[ibin]) / float(rhoa[0])
        fort_mmr = fortran["mmr"][0][0, ibin]
        err = abs(jax_mmr - fort_mmr) / max(abs(fort_mmr), 1e-50) if fort_mmr != 0 else 0
        print(f"{ibin+1:4d} {jax_mmr:12.4e} {fort_mmr:12.4e} {err:10.2e}")

    # --- RUN ONE STEP ---
    print("\n=== Running ONE timestep (dt=1s) ===")

    # Growth setup
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

    # Nucleation rates
    rnuclg_koop = freezaerl_koop2000(
        t[0], p_cgs[0], ssi[0], ssl[0], akelvin[0, 0],
        r1, vol1, DTYPE(1.38), pconmax[0, 0], NBIN)  # H2SO4 solute density
    rnuclg_murray = freezglaerl_murray2010(
        t[0], ssi[0], DTYPE(0.0), pconmax[0, 0], DTIME, NBIN)
    rnuclg = jnp.zeros((NBIN, NGROUP, NGROUP), dtype=DTYPE)
    rnuclg = rnuclg.at[:, 0, 1].set(rnuclg_koop + rnuclg_murray)

    # Save condensate
    prev_ice, prev_liq = totalcondensate(
        pc, rmass_2d, igroup_arr, is_ice_arr, igrowgas_arr, NBIN, NGROUP, NGAS, iz)

    # growevapl
    growlg = jnp.zeros((NBIN, NGROUP), dtype=DTYPE)
    evaplg = jnp.zeros((NBIN, NGROUP), dtype=DTYPE)
    growlg, evaplg = growevapl(
        pc, growlg, evaplg, supsatl_2d, supsati_2d, pvapl_2d, pvapi_2d,
        akelvin, akelvini, gro, gro1, rup_wet, rmass_2d, dm_2d, pconmax,
        pratt, prat, pden1, palr, is_ice_arr, igrowgas_arr, ienconc_arr,
        DTIME, iz, NBIN, NGROUP)

    # growp + upgxfer + psolve
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

    # evapp + downgevapply
    evappe_arr = jnp.zeros((NBIN, NELEM), dtype=DTYPE)
    evappe_arr = evapp(
        pc, evappe_arr, evaplg, pconmax, ienconc_arr, itype_arr,
        igroup_arr, iz, NBIN, NGROUP, NELEM)
    pc = downgevapply(pc, evappe_arr, jnp.zeros((NBIN, NELEM), dtype=DTYPE), DTIME, iz, NBIN, NELEM)

    # gsolve
    curr_ice, curr_liq = totalcondensate(
        pc, rmass_2d, igroup_arr, is_ice_arr, igrowgas_arr, NBIN, NGROUP, NGAS, iz)
    for igas in range(NGAS):
        gasprod = ((prev_ice[igas] - curr_ice[igas]) + (prev_liq[igas] - curr_liq[igas])) / DTIME
        gc = gc.at[iz, igas].add(DTIME * gasprod)

    # --- COMPARE t=1s bin-by-bin ---
    fort_t1 = fortran["mmr"][1]  # t=1s
    fort_gas_t1 = fortran["gas"][1]
    fort_ssl_t1 = fortran["ssl"][1]
    fort_ssi_t1 = fortran["ssi"][1]

    elem_names = ["Sulfate (I_INVOLATILE)", "Ice Volatile (I_VOLATILE)", "Ice Core (I_COREMASS)"]
    rmass_list = [rmass1, rmass2, None]

    for ie in range(NELEM):
        print(f"\n--- Element {ie+1}: {elem_names[ie]} ---")
        print(f"{'Bin':>4s} {'JAX MMR':>12s} {'Fort MMR':>12s} {'RelErr':>10s}")
        for ibin in range(NBIN):
            if ie == 2:  # I_COREMASS
                jax_mmr = float(pc[0, ibin, ie]) / float(rhoa[0])
            else:
                rmass_g = rmass1 if ie == 0 else rmass2
                jax_mmr = float(pc[0, ibin, ie]) * float(rmass_g[ibin]) / float(rhoa[0])
            fort_mmr = fort_t1[ie, ibin]
            err = abs(jax_mmr - fort_mmr) / max(abs(fort_mmr), 1e-50) if fort_mmr != 0 else 0
            marker = " ***" if err > 0.1 and fort_mmr > 1e-30 else ""
            print(f"{ibin+1:4d} {jax_mmr:12.4e} {fort_mmr:12.4e} {err:10.2e}{marker}")

    # Gas
    jax_gas = float(gc[0, 0] / rhoa[0])
    print(f"\nGas MMR:   JAX={jax_gas:.6e}, Fortran={fort_gas_t1:.6e}, "
          f"RelErr={abs(jax_gas - fort_gas_t1)/fort_gas_t1:.2e}")
    print(f"SSL:       JAX={float(ssl[0]):.4e}, Fortran={fort_ssl_t1:.4e}")
    print(f"SSI:       JAX={float(ssi[0]):.4e}, Fortran={fort_ssi_t1:.4e}")

    # Total mass conservation check
    total_jax = jax_gas
    for ie in range(NELEM):
        for ibin in range(NBIN):
            if ie == 2:
                total_jax += float(pc[0, ibin, ie]) / float(rhoa[0])
            else:
                rmass_g = rmass1 if ie == 0 else rmass2
                total_jax += float(pc[0, ibin, ie]) * float(rmass_g[ibin]) / float(rhoa[0])
    total_fort = fort_gas_t1 + sum(fort_t1[ie].sum() for ie in range(NELEM))
    print(f"\nTotal mass: JAX={total_jax:.6e}, Fortran={total_fort:.6e}")


if __name__ == "__main__":
    main()
