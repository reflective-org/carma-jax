"""Run JAX nuctest matching Fortran configuration.

2 groups (sulfate + ice), 3 elements, 16 bins, T=205K, p=90hPa.
Koop 2000 + Murray 2010 freezing + ice crystal growth.
dt=1s, 100 steps.

Full physics: nucleation → growevapl → growp+upgxfer+psolve → evapp → gsolve → tsolve.
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
from carma.setup_vf import setup_vf
from carma.vapor_pressure import vaporp_h2o_murphy2005
from carma.supersaturation import supersat
from carma.growth.growevapl import growevapl
from carma.growth.growp import growp
from carma.growth.evapp import evapp, downgevapply
from carma.growth.upgxfer import upgxfer
from carma.solvers.psolve import psolve
from carma.solvers.totalcondensate import totalcondensate
from carma.nucleation.freezaerl_koop2000 import freezaerl_koop2000
from carma.nucleation.freezglaerl_murray2010 import freezglaerl_murray2010
from carma.setup_nuc import setup_nuc
from carma.enums import GridType, ElementType, NucProcess


def compute_ppm_coefficients(dm_arr, rmass_arr, rmassup_arr, rmrat_val, nbin, ngroup):
    """Compute PPM coefficients for all groups.

    Exact Fortran formulas from carma_mod.F90 lines 532-569.

    Args:
        dm_arr: Bin mass widths per group, list of (NBIN,) arrays.
        rmass_arr: Bin masses per group, list of (NBIN,) arrays.
        rmassup_arr: Upper boundary masses per group, list of (NBIN,) arrays.
        rmrat_val: Mass ratio between bins.
        nbin, ngroup: Dimensions.

    Returns:
        Tuple of (pratt, prat, pden1, palr) as jnp arrays.
    """
    pratt = np.zeros((3, nbin, ngroup), dtype=np.float64)
    prat = np.zeros((4, nbin, ngroup), dtype=np.float64)
    pden1 = np.zeros((nbin, ngroup), dtype=np.float64)
    palr = np.zeros((4, ngroup), dtype=np.float64)

    for ig in range(ngroup):
        dm_np = np.array(dm_arr[ig])
        rmass_np = np.array(rmass_arr[ig])
        rmassup_np = np.array(rmassup_arr[ig])

        # pratt: gradient coefficients
        for ibin in range(1, nbin - 1):
            dm_im1 = dm_np[ibin - 1]
            dm_i = dm_np[ibin]
            dm_ip1 = dm_np[ibin + 1]
            pratt[0, ibin, ig] = dm_i / (dm_im1 + dm_i + dm_ip1)
            pratt[1, ibin, ig] = (2.0 * dm_im1 + dm_i) / (dm_ip1 + dm_i)
            pratt[2, ibin, ig] = (2.0 * dm_ip1 + dm_i) / (dm_im1 + dm_i)

        # prat, pden1: polynomial coefficients
        for ibin in range(1, nbin - 2):
            dm_im1 = dm_np[ibin - 1]
            dm_i = dm_np[ibin]
            dm_ip1 = dm_np[ibin + 1]
            dm_ip2 = dm_np[min(ibin + 2, nbin - 1)]
            prat[0, ibin, ig] = dm_i / (dm_i + dm_ip1)
            prat[1, ibin, ig] = 2.0 * dm_ip1 * dm_i / (dm_i + dm_ip1)
            prat[2, ibin, ig] = (dm_im1 + dm_i) / (2.0 * dm_i + dm_ip1)
            prat[3, ibin, ig] = (dm_ip2 + dm_ip1) / (2.0 * dm_ip1 + dm_i)
            pden1[ibin, ig] = dm_im1 + dm_i + dm_ip1 + dm_ip2

        # palr: edge coefficients
        denom_low = rmass_np[1] - rmass_np[0]
        denom_high = rmass_np[nbin - 1] - rmass_np[nbin - 2]
        palr[0, ig] = (rmassup_np[0] - rmass_np[0]) / denom_low
        palr[1, ig] = (rmassup_np[0] / rmrat_val - rmass_np[0]) / denom_low
        palr[2, ig] = (rmassup_np[nbin - 2] - rmass_np[nbin - 2]) / denom_high
        palr[3, ig] = (rmassup_np[nbin - 1] - rmass_np[nbin - 2]) / denom_high

    return jnp.array(pratt), jnp.array(prat), jnp.array(pden1), jnp.array(palr)


def parse_nuctest_bench(filepath):
    """Parse carma_nuctest.txt benchmark."""
    with open(filepath) as f:
        lines = f.readlines()
    idx = 0
    parts = lines[idx].split()
    ngroup, nelem, nbin, ngas = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
    idx += 1

    # Bin structure per group
    radii = {}
    rmass_f = {}
    for ig in range(ngroup):
        radii[ig] = np.zeros(nbin)
        rmass_f[ig] = np.zeros(nbin)
        for i in range(nbin):
            parts = lines[idx].split()
            radii[ig][i] = float(parts[2]) * 1e-4  # um to cm
            rmass_f[ig][i] = float(parts[3])
            idx += 1

    # Parse timestep blocks
    times = []
    mmr_data = []
    gas_data = []
    ssl_data = []
    ssi_data = []

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
        "times": np.array(times),
        "mmr": mmr_data,
        "gas": np.array(gas_data),
        "ssl": np.array(ssl_data),
        "ssi": np.array(ssi_data),
    }


def main():
    # ======================== CONFIGURATION ========================
    NBIN = 16
    NGROUP = 2
    NELEM = 3
    NGAS = 1
    NZ = 1
    DTIME = 1.0
    NSTEP = 100

    # --- Group 1: Sulfate aerosol ---
    rmin1, rmrat1, rho1 = 1e-7, 4.0, 1.78  # cm, mass ratio, g/cm^3
    r1, rmass1, vol1, dr1, dm1, rup1, rlow1, rmassup1 = setup_bins(rmin1, rmrat1, NBIN, rho1)

    # --- Group 2: Ice crystal ---
    rmin2, rmrat2, rho2 = 5e-5, 4.0, float(RHO_I)
    r2, rmass2, vol2, dr2, dm2, rup2, rlow2, rmassup2 = setup_bins(rmin2, rmrat2, NBIN, rho2)

    print(f"Group 1 (sulfate): {float(r1[0])*1e4:.4f} to {float(r1[-1])*1e4:.2f} um")
    print(f"Group 2 (ice):     {float(r2[0])*1e4:.2f} to {float(r2[-1])*1e4:.0f} um")

    # --- Element config ---
    # Element 0: sulfate (I_INVOLATILE, group 0)
    # Element 1: ice volatile (I_VOLATILE, group 1) — grows by H2O
    # Element 2: ice core (I_COREMASS, group 1)
    igroup_arr = np.array([0, 1, 1], dtype=np.int32)
    itype_arr = np.array([int(ElementType.I_INVOLATILE),
                          int(ElementType.I_VOLATILE),
                          int(ElementType.I_COREMASS)], dtype=np.int32)
    igrowgas_arr = np.array([-1, 0, -1], dtype=np.int32)  # only element 1 grows by gas 0
    ienconc_arr = np.array([0, 1], dtype=np.int32)  # number conc element per group
    is_ice_arr = np.array([False, True])
    gwtmol_arr = np.array([float(WTMOL_H2O)])

    # --- Nucleation mapping ---
    # TWO pairs: element 0→1 (number transfer) + element 0→2 (core mass transfer)
    nuc_proc_flag = int(NucProcess.I_AERFREEZE) + int(NucProcess.I_AF_KOOP_2000) + int(NucProcess.I_AF_MURRAY_2010)
    nuc_tables = setup_nuc(
        NBIN, NGROUP, NELEM,
        groups_rmass=[np.array(rmass1), np.array(rmass2)],
        nuc_from_elem=[0, 0],      # source: element 0 (sulfate) for both
        nuc_to_elem=[1, 2],        # target: element 1 (ice number) + element 2 (ice core)
        nuc_proc=[nuc_proc_flag, nuc_proc_flag],
    )

    print(f"\nNucleation bin mapping:")
    for i in range(NBIN):
        tgt = int(nuc_tables['inuc2bin'][i, 0, 1])
        if tgt >= 0:
            print(f"  Sulfate bin {i+1} (r={float(r1[i])*1e4:.4f}um, m={float(rmass1[i]):.2e}g) "
                  f"→ Ice bin {tgt+1} (r={float(r2[tgt])*1e4:.2f}um, m={float(rmass2[tgt]):.2e}g)")

    # --- PPM coefficients ---
    pratt, prat, pden1, palr = compute_ppm_coefficients(
        dm_arr=[np.array(dm1), np.array(dm2)],
        rmass_arr=[np.array(rmass1), np.array(rmass2)],
        rmassup_arr=[np.array(rmassup1), np.array(rmassup2)],
        rmrat_val=rmrat1,  # same for both groups
        nbin=NBIN, ngroup=NGROUP,
    )

    # --- Multi-group arrays ---
    rmass_2d = jnp.stack([rmass1, rmass2], axis=1)  # (NBIN, NGROUP)
    dm_2d = jnp.stack([dm1, dm2], axis=1)            # (NBIN, NGROUP)
    rup_wet = jnp.stack([
        jnp.broadcast_to(rup1[None, :], (NZ, NBIN)),
        jnp.broadcast_to(rup2[None, :], (NZ, NBIN)),
    ], axis=2)  # (NZ, NBIN, NGROUP)
    rlow_wet = jnp.stack([
        jnp.broadcast_to(rlow1[None, :], (NZ, NBIN)),
        jnp.broadcast_to(rlow2[None, :], (NZ, NBIN)),
    ], axis=2)  # (NZ, NBIN, NGROUP)
    r_wet = jnp.stack([
        jnp.broadcast_to(r1[None, :], (NZ, NBIN)),
        jnp.broadcast_to(r2[None, :], (NZ, NBIN)),
    ], axis=2)  # (NZ, NBIN, NGROUP)

    # Reynolds number (small particles, Re ≈ 0)
    re = jnp.zeros((NZ, NBIN, NGROUP), dtype=DTYPE)

    # ======================== ATMOSPHERE ========================
    T_init = 205.0
    p_pa = 9000.0  # 90 hPa

    t = jnp.array([DTYPE(T_init)])
    p_cgs = jnp.array([DTYPE(p_pa) * RPA2CGS])
    zc = jnp.array([DTYPE(17000.0) * RM2CGS])
    zl = jnp.array([DTYPE(16900.0) * RM2CGS, DTYPE(17100.0) * RM2CGS])

    rho_air = float(p_pa * 10.0) / (float(R_AIR) * T_init) * 1e3
    pl = jnp.array([(p_pa + 100 * rho_air * float(GRAV) / 100) * float(RPA2CGS),
                     (p_pa - 100 * rho_air * float(GRAV) / 100) * float(RPA2CGS)])

    rhoa, dz, zmet, zmetl, rmu, thcond, rhoa_wet = setup_atm(
        t, p_cgs, pl, zc, zl, GridType.I_CART
    )

    print(f"\nAtmosphere: T={T_init}K, p={p_pa}Pa, rhoa_cgs={float(rhoa[0]/zmet[0]):.4e} g/cm³")

    # ======================== INITIAL CONDITIONS ========================
    pc = jnp.full((NZ, NBIN, NELEM), SMALL_PC, dtype=DTYPE)

    # Lognormal sulfate distribution: n=100 cm^-3, r0=2.5e-6 cm, sigma=1.5
    # Match Fortran initial condition exactly: the Fortran test converts
    # dN*rmass to MMR using an APPROXIMATE rhoa (100mbar, 200K), then
    # SetBin converts MMR * rhoa_wet back to internal pc. We reproduce
    # this to match the Fortran's effective particle count.
    n_total = 100.0
    r0 = 2.5e-6
    sigma = 1.5
    # Fortran's approximate rhoa for initial condition
    rhoa_init_fort = DTYPE(100.0) * DTYPE(1000.0) / R_AIR / DTYPE(200.0)
    for ibin in range(NBIN):
        r_val = float(r1[ibin])
        dr_val = float(dr1[ibin])
        ln_sigma = np.log(sigma)
        dn_dr = (n_total / (r_val * np.sqrt(2 * np.pi) * ln_sigma)
                 * np.exp(-np.log(r_val / r0)**2 / (2 * ln_sigma**2)))
        n_bin = dn_dr * dr_val
        # Match Fortran: mmr = dN*rmass/rhoa_init, then pc = mmr*rhoa_wet
        # Our pc = N * zmet, so N_effective = dN * rhoa_wet / rhoa_init
        rhoa_wet_val = float(rhoa[0] / zmet[0])
        n_effective = n_bin * rhoa_wet_val / float(rhoa_init_fort)
        pc = pc.at[0, ibin, 0].set(DTYPE(n_effective) * zmet[0])

    # Gas: 40 ppm H2O
    gas_mmr = 4e-5
    gc = jnp.array([[DTYPE(gas_mmr) * rhoa[0]]])

    print(f"\nInitial sulfate: N_total = {float(pc[0, :, 0].sum() / zmet[0]):.1f} cm^-3")
    print(f"Initial gas: {gas_mmr*1e6:.0f} ppm H2O")

    # ======================== TIME INTEGRATION ========================
    print(f"\nRunning {NSTEP} steps at dt={DTIME}s...")
    print(f"Full physics: nucleation + growth + gas conservation + temperature\n")

    iz = 0
    supsati_old = DTYPE(0.0)

    # Storage for comparison
    history = {
        "times": [0.0],
        "mmr_elem": [],  # list of (NELEM, NBIN) arrays
        "gas_mmr": [gas_mmr],
        "ssl": [0.0],
        "ssi": [0.0],
        "t": [T_init],
    }

    # Store initial MMR
    mmr0 = np.zeros((NELEM, NBIN))
    for ibin in range(NBIN):
        mmr0[0, ibin] = float(pc[0, ibin, 0]) * float(rmass1[ibin]) / float(rhoa[0])
        mmr0[1, ibin] = float(pc[0, ibin, 1]) * float(rmass2[ibin]) / float(rhoa[0])
        mmr0[2, ibin] = float(pc[0, ibin, 2]) / float(rhoa[0])  # core mass element
    history["mmr_elem"].append(mmr0)

    # --- Substepping parameters ---
    MAX_SUBSTEPS = 256
    MAX_RETRIES = 8

    for istep in range(NSTEP):
        # Save state for retry
        pc_saved = pc
        gc_saved = gc
        t_saved = t
        ssi_saved = supsati_old

        # --- Estimate substeps from physical CFL (Fortran nsubsteps approach) ---
        # Compute dmdt at each ice bin boundary, set nsub = ceil(max_cfl)
        ntsubsteps = 1

        # Recompute growth kernels for estimation
        diffus_e, rlhe_e, rlhm_e = setup_grow(t, p_cgs, rhoa, zmet, 0, -1, NGAS, False)
        _, akelvin_e, akelvini_e, gro_e, gro1_e, _, _, _ = setup_gkern(
            t, p_cgs, rhoa, zmet, rmu, thcond, diffus_e, rlhe_e, rlhm_e,
            re, r_wet, rlow_wet, jnp.ones((NBIN, NGROUP), dtype=DTYPE),
            jnp.array([1.0, 3.0]), is_ice_arr, gwtmol_arr, igrowgas_arr,
            1.0, 1.0, 1.0, NBIN, NGROUP, NGAS)
        pvapl_e, pvapi_e = vaporp_h2o_murphy2005(t)
        _, ssi_e = supersat(t, gc[:, 0], pvapl_e, pvapi_e, float(WTMOL_H2O), zmet)
        pconmax_e = jnp.zeros((NZ, NGROUP), dtype=DTYPE)
        for ig in range(NGROUP):
            ie = int(ienconc_arr[ig])
            pconmax_e = pconmax_e.at[:, ig].set(jnp.max(pc[:, :, ie], axis=1) / zmet)

        if float(pconmax_e[0, 1]) > float(FEW_PC):
            max_cfl = 0.0
            ig_ice = 1
            for ibin in range(NBIN - 1):
                g0 = float(gro_e[0, ibin + 1, ig_ice])
                g1 = float(gro1_e[0, ibin + 1, ig_ice])
                if g0 > 0:
                    pvap_i = float(pvapi_e[0])
                    ss_i = float(ssi_e[0])
                    r_bnd = float(rup_wet[0, ibin, ig_ice])
                    akelv = float(akelvini_e[0, 0])
                    expon = min(akelv / max(r_bnd, 1e-30), 700.0)
                    akas = np.exp(expon)
                    dmdt_val = abs(pvap_i * (ss_i + 1.0 - akas) * g0 / (1 + g0 * g1 * pvap_i))
                    dm_val = float(dm_2d[ibin, ig_ice])
                    if dm_val > 0:
                        max_cfl = max(max_cfl, dmdt_val * DTIME / dm_val)
            if max_cfl > 1.0:
                ntsubsteps = min(MAX_SUBSTEPS, int(np.ceil(max_cfl)) * 2)

        # Also check Fortran Condition 3: freezing at cold T with high SSi
        if float(ssi_e[0]) > 0.4 and float(t[0]) < 233.0:
            ntsubsteps = max(ntsubsteps, 128)

        # --- Run substeps ---
        nretries = 0
        converged = False

        while not converged:
            # Reset to saved state
            pc = pc_saved
            gc = gc_saved
            t = t_saved
            supsati_old_sub = ssi_saved

            dt_sub = DTIME / ntsubsteps

            # Record initial supersaturation for convergence check
            pvapl_0, pvapi_0 = vaporp_h2o_murphy2005(t)
            ssl_0, ssi_0 = supersat(t, gc[:, 0], pvapl_0, pvapi_0, float(WTMOL_H2O), zmet)
            ssi_start = float(ssi_0[0])

            for isub in range(ntsubsteps):
                # --- Recompute T-dependent quantities each substep ---
                diffus, rlhe, rlhm = setup_grow(
                    t, p_cgs, rhoa, zmet, igash2o=0, igash2so4=-1,
                    ngas=NGAS, do_cnst_rlh=False,
                )

                rrat = jnp.ones((NBIN, NGROUP), dtype=DTYPE)
                eshape_arr = jnp.array([1.0, 3.0])
                surfctwa, akelvin, akelvini, gro, gro1, gro2, ft_arr, thcondnc = setup_gkern(
                    t, p_cgs, rhoa, zmet, rmu, thcond, diffus, rlhe, rlhm,
                    re, r_wet, rlow_wet, rrat, eshape_arr, is_ice_arr,
                    gwtmol_arr, igrowgas_arr, 1.0, 1.0, 1.0,
                    NBIN, NGROUP, NGAS,
                )

                pvapl, pvapi = vaporp_h2o_murphy2005(t)
                pvapl_2d = jnp.broadcast_to(pvapl[:, None], (NZ, NGAS))
                pvapi_2d = jnp.broadcast_to(pvapi[:, None], (NZ, NGAS))
                ssl, ssi = supersat(t, gc[:, 0], pvapl_2d[:, 0], pvapi_2d[:, 0],
                                    float(WTMOL_H2O), zmet)
                supsatl_2d = jnp.broadcast_to(ssl[:, None], (NZ, NGAS))
                supsati_2d = jnp.broadcast_to(ssi[:, None], (NZ, NGAS))

                pconmax = jnp.zeros((NZ, NGROUP), dtype=DTYPE)
                for ig in range(NGROUP):
                    ie = int(ienconc_arr[ig])
                    pconmax = pconmax.at[:, ig].set(
                        jnp.max(pc[:, :, ie], axis=1) / zmet)

                # Nucleation rates (scale Murray by substep fraction)
                # rhosol = H2SO4 SOLUTE density (1.38), NOT sulfate particle
                # density (1.78). The Fortran uses rhosol(isol) for volrat in
                # freezaerl_koop2000.F90.
                rnuclg_koop = freezaerl_koop2000(
                    t[0], p_cgs[0], ssi[0], ssl[0], akelvin[0, 0],
                    r1, vol1, DTYPE(1.38), pconmax[0, 0], NBIN)
                rnuclg_murray = freezglaerl_murray2010(
                    t[0], ssi[0], supsati_old_sub, pconmax[0, 0], dt_sub, NBIN)
                supsati_old_sub = ssi[0]
                rnuclg = jnp.zeros((NBIN, NGROUP, NGROUP), dtype=DTYPE)
                rnuclg = rnuclg.at[:, 0, 1].set(rnuclg_koop + rnuclg_murray)

                # Save condensate
                prev_ice, prev_liq = totalcondensate(
                    pc, rmass_2d, igroup_arr, is_ice_arr, igrowgas_arr,
                    NBIN, NGROUP, NGAS, iz)

                # growevapl with substep dt
                growlg = jnp.zeros((NBIN, NGROUP), dtype=DTYPE)
                evaplg = jnp.zeros((NBIN, NGROUP), dtype=DTYPE)
                growlg, evaplg = growevapl(
                    pc, growlg, evaplg,
                    supsatl_2d, supsati_2d, pvapl_2d, pvapi_2d,
                    akelvin, akelvini, gro, gro1,
                    rup_wet, rmass_2d, dm_2d, pconmax,
                    pratt, prat, pden1, palr,
                    is_ice_arr, igrowgas_arr, ienconc_arr,
                    dt_sub, iz, NBIN, NGROUP,
                )

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
                        growpe_arr = growp(pc, growpe_arr, growlg, pconmax,
                                          iz, ibin, ielem, ig, igrow)
                        rnucpe_arr = upgxfer(
                            rnucpe_arr, rnuclg, pc, rmass_2d, pconmax,
                            ielem, ibin, iz,
                            nuc_tables['nnucelem'], nuc_tables['inucelem'],
                            nuc_tables['nnucbin'], nuc_tables['inucbin'],
                            igroup_arr, itype_arr, NBIN, NGROUP)
                        pc, pc_nucl = psolve(
                            pc, pc_nucl, growpe_arr, evappe_arr, rnucpe_arr, rhompe_arr,
                            growlg, evaplg, rnuclg, dt_sub, iz, ibin, ielem, ig, NGROUP)

                # evapp + downgevapply
                evappe_arr = jnp.zeros((NBIN, NELEM), dtype=DTYPE)
                evappe_arr = evapp(
                    pc, evappe_arr, evaplg, pconmax, ienconc_arr, itype_arr,
                    igroup_arr, iz, NBIN, NGROUP, NELEM)
                pc = downgevapply(pc, evappe_arr, jnp.zeros((NBIN, NELEM), dtype=DTYPE),
                                  dt_sub, iz, NBIN, NELEM)

                # gsolve
                curr_ice, curr_liq = totalcondensate(
                    pc, rmass_2d, igroup_arr, is_ice_arr, igrowgas_arr,
                    NBIN, NGROUP, NGAS, iz)
                for igas in range(NGAS):
                    gasprod = ((prev_ice[igas] - curr_ice[igas])
                               + (prev_liq[igas] - curr_liq[igas])) / dt_sub
                    ice_change = prev_ice[igas] - curr_ice[igas]
                    liq_change = prev_liq[igas] - curr_liq[igas]
                    gc = gc.at[iz, igas].add(dt_sub * gasprod)

                # tsolve
                rlprod = -(ice_change * (rlhe[iz, 0] + rlhm[iz, 0])
                           + liq_change * rlhe[iz, 0]) / (CP * rhoa[iz] * dt_sub)
                t = t.at[iz].add(dt_sub * rlprod)
            # --- Convergence check: supersaturation sign change ---
            pvapl_f, pvapi_f = vaporp_h2o_murphy2005(t)
            ssl_f, ssi_f = supersat(t, gc[:, 0], pvapl_f, pvapi_f, float(WTMOL_H2O), zmet)
            ssi_end = float(ssi_f[0])

            sign_changed = (ssi_start * ssi_end) < 0
            large_change = abs(ssi_end) > 0.05

            if sign_changed and large_change and ntsubsteps < MAX_SUBSTEPS and nretries < MAX_RETRIES:
                ntsubsteps = min(ntsubsteps * 2, MAX_SUBSTEPS)
                nretries += 1
            else:
                converged = True

        # Update supsati_old for Murray (use end-of-step value)
        supsati_old = ssi_f[0]

        # --- Store history ---
        history["times"].append(float(istep + 1) * DTIME)

        mmr_step = np.zeros((NELEM, NBIN))
        for ibin in range(NBIN):
            mmr_step[0, ibin] = float(pc[0, ibin, 0]) * float(rmass1[ibin]) / float(rhoa[0])
            mmr_step[1, ibin] = float(pc[0, ibin, 1]) * float(rmass2[ibin]) / float(rhoa[0])
            mmr_step[2, ibin] = float(pc[0, ibin, 2]) / float(rhoa[0])
        history["mmr_elem"].append(mmr_step)
        history["gas_mmr"].append(float(gc[0, 0] / rhoa[0]))
        history["ssl"].append(float(ssl[0]))
        history["ssi"].append(float(ssi[0]))
        history["t"].append(float(t[0]))

        if (istep + 1) % 10 == 0:
            sulfate_N = float(pc[0, :, 0].sum() / zmet[0])
            ice_N = float(pc[0, :, 1].sum() / zmet[0])
            gas_ppm = float(gc[0, 0] / rhoa[0]) * 1e6
            print(f"  Step {istep+1:3d}: T={float(t[0]):.2f}K, "
                  f"sulfate_N={sulfate_N:.1f}, ice_N={ice_N:.2e}, "
                  f"gas={gas_ppm:.1f}ppm, ssi={float(ssi[0]):.3f}, "
                  f"nsub={ntsubsteps}",
                  flush=True)

    # ======================== COMPARE WITH FORTRAN ========================
    bench_path = (Path(__file__).parent.parent.parent
                  / "original-carma" / "CARMA" / "tests" / "bench" / "carma_nuctest.txt")
    if not bench_path.exists():
        bench_path = (Path(__file__).parent.parent.parent
                      / "original-carma" / "CARMA" / "build" / "carma_nuctest.txt")

    fortran = None
    if bench_path.exists():
        print(f"\nParsing Fortran benchmark from {bench_path}...")
        fortran = parse_nuctest_bench(bench_path)
        print(f"  {len(fortran['times'])} timesteps")

        # Compare final state
        print(f"\n{'='*60}")
        print(f"Final state comparison (t={NSTEP}s):")
        print(f"{'='*60}")
        jax_final = history["mmr_elem"][-1]
        fort_final = fortran["mmr"][-1]
        for ie in range(NELEM):
            names = ["Sulfate", "Ice volatile", "Ice core"]
            jax_total = np.sum(jax_final[ie])
            fort_total = np.sum(fort_final[ie])
            rel_err = abs(jax_total - fort_total) / max(abs(fort_total), 1e-50) if fort_total != 0 else 0
            print(f"  {names[ie]:15s}: JAX={jax_total:.4e}, Fortran={fort_total:.4e}, rel_err={rel_err:.2e}")

        print(f"  {'Gas MMR':15s}: JAX={history['gas_mmr'][-1]:.6e}, "
              f"Fortran={fortran['gas'][-1]:.6e}, "
              f"rel_err={abs(history['gas_mmr'][-1] - fortran['gas'][-1])/fortran['gas'][-1]:.2e}")
        print(f"  {'Temperature':15s}: JAX={history['t'][-1]:.4f}K")

    # ======================== PLOTS ========================
    outdir = Path("plots/nuctest_validation")
    outdir.mkdir(parents=True, exist_ok=True)

    times_jax = np.array(history["times"])

    # --- Plot 1: Time evolution ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Sulfate number
    ax = axes[0, 0]
    sulfate_N_hist = [np.sum(h[0]) for h in history["mmr_elem"]]
    ax.plot(times_jax, sulfate_N_hist, "b-", lw=2, label="JAX")
    if fortran:
        fort_sulfate = [np.sum(fortran["mmr"][i][0]) for i in range(len(fortran["times"]))]
        ax.plot(fortran["times"], fort_sulfate, "r--", lw=2, label="Fortran")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Total MMR")
    ax.set_title("Element 1: Sulfate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Ice volatile total
    ax = axes[0, 1]
    ice_vol_hist = [np.sum(h[1]) for h in history["mmr_elem"]]
    ax.plot(times_jax, ice_vol_hist, "b-", lw=2, label="JAX")
    if fortran:
        fort_ice = [np.sum(fortran["mmr"][i][1]) for i in range(len(fortran["times"]))]
        ax.plot(fortran["times"], fort_ice, "r--", lw=2, label="Fortran")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Total MMR")
    ax.set_title("Element 2: Ice Volatile")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Ice core total
    ax = axes[0, 2]
    ice_core_hist = [np.sum(h[2]) for h in history["mmr_elem"]]
    ax.plot(times_jax, ice_core_hist, "b-", lw=2, label="JAX")
    if fortran:
        fort_core = [np.sum(fortran["mmr"][i][2]) for i in range(len(fortran["times"]))]
        ax.plot(fortran["times"], fort_core, "r--", lw=2, label="Fortran")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Total MMR")
    ax.set_title("Element 3: Ice Core")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Gas
    ax = axes[1, 0]
    ax.plot(times_jax, np.array(history["gas_mmr"]) * 1e6, "b-", lw=2, label="JAX")
    if fortran:
        ax.plot(fortran["times"], fortran["gas"] * 1e6, "r--", lw=2, label="Fortran")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("H2O [ppm]")
    ax.set_title("Water Vapor")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Temperature
    ax = axes[1, 1]
    ax.plot(times_jax, history["t"], "b-", lw=2)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("T [K]")
    ax.set_title("Temperature")
    ax.grid(True, alpha=0.3)

    # Supersaturation
    ax = axes[1, 2]
    ax.plot(times_jax[1:], history["ssi"][1:], "b-", lw=2, label="JAX SSi")
    if fortran:
        ax.plot(fortran["times"], fortran["ssi"], "r--", lw=2, label="Fortran SSi")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Ice supersaturation")
    ax.set_title("Supersaturation (ice)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("JAX vs Fortran Nuctest: Sulfate Freezing + Ice Growth", fontsize=14)
    fig.tight_layout()
    fig.savefig(outdir / "nuctest_jax_vs_fortran_timeseries.png", dpi=150)
    plt.close(fig)

    # --- Plot 2: Size distributions at key times ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    r_um1 = np.array(r1) * 1e4
    r_um2 = np.array(r2) * 1e4

    for col, (si, label) in enumerate([(0, "t=0s"), (10, "t=10s"), (-1, f"t={NSTEP}s")]):
        # Sulfate
        ax = axes[0, col]
        jax_vals = history["mmr_elem"][si][0]
        jax_vals_plot = np.where(jax_vals > 0, jax_vals, 1e-50)
        ax.semilogy(r_um1, jax_vals_plot, "b-o", ms=4, label="JAX")
        if fortran and si < len(fortran["mmr"]):
            fort_vals = fortran["mmr"][si][0]
            fort_vals_plot = np.where(fort_vals > 0, fort_vals, 1e-50)
            ax.semilogy(r_um1, fort_vals_plot, "r--s", ms=3, label="Fortran")
        ax.set_xlabel("Radius [um]")
        ax.set_ylabel("MMR [g/g]")
        ax.set_title(f"Sulfate ({label})")
        ax.legend(fontsize=8)
        ax.set_ylim(1e-35, 1e-5)
        ax.grid(True, alpha=0.3)

        # Ice volatile
        ax = axes[1, col]
        jax_vals = history["mmr_elem"][si][1]
        jax_vals_plot = np.where(jax_vals > 0, jax_vals, 1e-50)
        ax.semilogy(r_um2, jax_vals_plot, "b-o", ms=4, label="JAX ice vol")
        if fortran and si < len(fortran["mmr"]):
            fort_vals = fortran["mmr"][si][1]
            fort_vals_plot = np.where(fort_vals > 0, fort_vals, 1e-50)
            ax.semilogy(r_um2, fort_vals_plot, "r--s", ms=3, label="Fortran ice vol")
        # Also plot ice core
        jax_core = history["mmr_elem"][si][2]
        jax_core_plot = np.where(jax_core > 0, jax_core, 1e-50)
        ax.semilogy(r_um2, jax_core_plot, "g-^", ms=4, label="JAX ice core")
        if fortran and si < len(fortran["mmr"]):
            fort_core = fortran["mmr"][si][2]
            fort_core_plot = np.where(fort_core > 0, fort_core, 1e-50)
            ax.semilogy(r_um2, fort_core_plot, "m--v", ms=3, label="Fortran ice core")
        ax.set_xlabel("Radius [um]")
        ax.set_ylabel("MMR [g/g]")
        ax.set_title(f"Ice ({label})")
        ax.legend(fontsize=7)
        ax.set_ylim(1e-45, 1e-3)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Size Distributions: JAX vs Fortran", fontsize=14)
    fig.tight_layout()
    fig.savefig(outdir / "nuctest_jax_vs_fortran_distributions.png", dpi=150)
    plt.close(fig)

    print(f"\nPlots saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
