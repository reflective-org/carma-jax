"""Realistic sulfate scenarios with all processes: nucleation + condensation + coagulation.

Uses realistic lognormal initial size distributions based on observations:

Stratospheric Junge layer (~20 km, Deshler et al. 2003):
  - Lognormal: r_mode ~= 0.07 um, sigma = 1.6, N ~= 10 /cm^3
  - Background H2SO4 from COS/OCS oxidation: ~1-5 pptv
  - T = 215 K, p = 50 hPa

Tropospheric boundary layer (polluted continental):
  - Bi-modal: Aitken (r=0.03um, sigma=1.6, N=5000/cm3) + accumulation (r=0.1um, sigma=1.8, N=1000/cm3)
  - H2SO4 from photochemistry: realistic source ~5e5-1e6 /cm3/s during day
  - T = 288 K, p = 900 hPa (near surface)

Remote marine boundary layer (clean):
  - Single mode: r=0.08um, sigma=1.5, N=150/cm3
  - Low H2SO4 (no anthropogenic): source ~1e5 /cm3/s
  - T = 290 K, p = 1000 hPa

All simulations include: Vehkamaki nucleation + H2SO4 condensation + Brownian
coagulation (no sedimentation/transport in these box-model tests).
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
from carma.setup_ckern import setup_ckern
from carma.setup_vf import setup_vf
from carma.growth.growevapl import growevapl
from carma.growth.growp import growp
from carma.growth.evapp import evapp, downgevapply
from carma.solvers.psolve import psolve
from carma.solvers.totalcondensate import totalcondensate
from carma.vapor_pressure import vaporp_h2o_murphy2005, vaporp_h2so4_ayers1980
from carma.supersaturation import supersat
from carma.nucleation.sulfnucrate import binary_nuc_vehk2002
from carma.coagulation.setup_coag import setup_coag
from carma.microslow import make_microslow
from carma.config import CarmaConfig, ElementConfig, GroupConfig
from carma.enums import GridType, ElementType, Shape
from scripts.run_nuctest_jax import compute_ppm_coefficients


WTMOL_H2SO4 = 98.0
WTMOL_H2O_L = 18.016


def make_lognormal_modes(r_cm, dr_cm, rmass, modes):
    """Sum of lognormal modes for the initial size distribution.

    Args:
        r_cm, dr_cm: Bin center radii and widths [cm].
        rmass: Bin masses [g].
        modes: List of dicts with keys 'r_mode' (cm), 'sigma', 'N' (#/cm^3).

    Returns:
        dN/dr * dr array — number per bin [#/cm^3].
    """
    n_total = np.zeros_like(r_cm)
    for mode in modes:
        r_mode = mode["r_mode"]
        sigma = mode["sigma"]
        N = mode["N"]
        ln_sigma = np.log(sigma)
        dn_dr = (N / (r_cm * np.sqrt(2 * np.pi) * ln_sigma)
                 * np.exp(-np.log(r_cm / r_mode)**2 / (2 * ln_sigma**2)))
        n_total += dn_dr * dr_cm
    return n_total


def setup_scenario(name):
    """Return realistic atmospheric/aerosol configuration."""
    base = dict(
        nbin=38,
        rmrat=2.0,
        rho_aer=1.78,
        rho_sol=1.38,
        do_coag=True,
    )

    if name == "stratosphere":
        return dict(base,
            title="Stratosphere Junge Layer (~20 km)",
            T=215.0,
            p_hpa=50.0,
            z_km=20.0,
            h2o_ppmv=5.0,
            h2so4_pptv=3.0,
            h2so4_source_per_cm3_s=5e3,
            rmin_cm=1e-7,  # 1 nm smallest
            modes=[
                {"r_mode": 0.07e-4, "sigma": 1.6, "N": 10.0},  # Junge mode
            ],
            dtime=60.0,
            nstep=360,   # 6 hours
            snapshot_min=[0, 60, 180, 360],
        )
    elif name == "troposphere_polluted":
        return dict(base,
            title="Polluted Continental BL (near surface)",
            T=288.0,
            p_hpa=900.0,
            z_km=1.0,
            h2o_ppmv=10000.0,  # ~60% RH at 288K
            h2so4_pptv=5.0,     # background
            h2so4_source_per_cm3_s=5e5,  # daytime photochem source
            rmin_cm=1e-7,
            modes=[
                {"r_mode": 0.03e-4, "sigma": 1.6, "N": 5000.0},   # Aitken
                {"r_mode": 0.10e-4, "sigma": 1.8, "N": 1000.0},   # accumulation
            ],
            dtime=60.0,
            nstep=360,    # 6 hours
            snapshot_min=[0, 30, 120, 360],
        )
    elif name == "troposphere_marine":
        return dict(base,
            title="Remote Marine BL (clean)",
            T=290.0,
            p_hpa=1000.0,
            z_km=0.1,
            h2o_ppmv=15000.0,   # ~80% RH
            h2so4_pptv=0.5,     # very clean
            h2so4_source_per_cm3_s=1e5,
            rmin_cm=1e-7,
            modes=[
                {"r_mode": 0.08e-4, "sigma": 1.5, "N": 150.0},
            ],
            dtime=60.0,
            nstep=360,
            snapshot_min=[0, 60, 180, 360],
        )
    else:
        raise ValueError(f"Unknown scenario: {name}")


def run_scenario(config):
    """Run a single scenario with nucleation + condensation + coagulation."""
    NBIN = config["nbin"]
    NGROUP = 1
    NELEM = 1
    NGAS = 1
    NZ = 1

    rmin = config["rmin_cm"]
    rmrat = config["rmrat"]
    rho_aer = config["rho_aer"]
    rho_sol = config["rho_sol"]

    r1, rmass1, vol1, dr1, dm1, rup1, rlow1, rmassup1 = setup_bins(rmin, rmrat, NBIN, rho_aer)
    print(f"  Bin range: {float(r1[0])*1e7:.2f} nm to {float(r1[-1])*1e4:.2f} um")

    # --- Element/Group config ---
    igroup_arr = np.array([0], dtype=np.int32)
    itype_arr = np.array([int(ElementType.I_VOLATILE)], dtype=np.int32)
    igrowgas_arr = np.array([0], dtype=np.int32)  # grows by gas 0 (H2SO4)
    ienconc_arr = np.array([0], dtype=np.int32)
    is_ice_arr = np.array([False])
    gwtmol_arr = np.array([WTMOL_H2SO4])

    # --- PPM coefficients ---
    pratt, prat, pden1, palr = compute_ppm_coefficients(
        [np.array(dm1)], [np.array(rmass1)], [np.array(rmassup1)],
        rmrat, NBIN, NGROUP)

    rmass_2d = rmass1[:, None]
    dm_2d = dm1[:, None]
    rup_wet = jnp.broadcast_to(rup1[None, :, None], (NZ, NBIN, NGROUP))
    rlow_wet = jnp.broadcast_to(rlow1[None, :, None], (NZ, NBIN, NGROUP))
    r_wet = jnp.broadcast_to(r1[None, :, None], (NZ, NBIN, NGROUP))

    # --- Coagulation setup ---
    groups = (GroupConfig(
        name='sulfate', ishape=int(Shape.I_SPHERE), ienconc=0, is_ice=False,
        is_cloud=False, is_sulfate=True,
        do_vtran=False, do_drydep=False, ifallrtn=1, irhswell=0,
        rmrat=rmrat, eshape=1.0, rmin=rmin,
        r=r1, rmass=rmass1, vol=vol1, dr=dr1, dm=dm1,
        rmassup=rmassup1, rup=rup1, rlow=rlow1,
        rrat=jnp.ones(NBIN, dtype=DTYPE),
        rprat=jnp.ones(NBIN, dtype=DTYPE),
        arat=jnp.ones(NBIN, dtype=DTYPE),
    ),)
    elements = (ElementConfig(
        name='sulfate', rho=jnp.full(NBIN, rho_aer, dtype=DTYPE),
        igroup=0, itype=int(ElementType.I_VOLATILE),
        icomposition=0, isolute=-1, kappa=0.0,
    ),)
    icoag_tbl = np.array([[0]], dtype=np.int32)    # self-coag into group 0
    icoagelem_tbl = np.array([[0]], dtype=np.int32)
    coag = setup_coag(NBIN, NGROUP, NELEM, groups, elements, icoag_tbl, icoagelem_tbl)

    # Pre-compile microslow
    microslow_jit = make_microslow(
        nbin=NBIN, nelem=NELEM, ngroup=NGROUP,
        elem_igroup=jnp.array([0]),
        icoag=coag.icoag, volx=coag.volx, icoagelem=coag.icoagelem,
        npairu=coag.npairu, npairl=coag.npairl,
        iup=coag.iup, jup=coag.jup, igup=coag.igup, jgup=coag.jgup,
        ilow=coag.ilow, jlow=coag.jlow, iglow=coag.iglow, jglow=coag.jglow,
        pkernel=coag.pkernel,
        ienconc_arr=jnp.array([0]),
        elem_itypes=jnp.array([int(ElementType.I_VOLATILE)]),
    )

    # --- Atmosphere ---
    T_init = config["T"]
    p_pa = config["p_hpa"] * 100
    t = jnp.array([DTYPE(T_init)])
    p_cgs = jnp.array([DTYPE(p_pa) * RPA2CGS])
    zc = jnp.array([DTYPE(config["z_km"] * 1e3) * RM2CGS])
    zl = jnp.array([DTYPE((config["z_km"] * 1e3 - 50.0)) * RM2CGS,
                     DTYPE((config["z_km"] * 1e3 + 50.0)) * RM2CGS])
    rho_air = float(p_pa * 10.0) / (float(R_AIR) * T_init)
    pl = jnp.array([(p_pa + 50 * rho_air * float(GRAV) / 100) * float(RPA2CGS),
                     (p_pa - 50 * rho_air * float(GRAV) / 100) * float(RPA2CGS)])
    rhoa, dz, zmet, zmetl, rmu, thcond, rhoa_wet = setup_atm(
        t, p_cgs, pl, zc, zl, GridType.I_CART)

    print(f"  Atm: T={T_init}K, p={config['p_hpa']} hPa, "
          f"rhoa={float(rhoa[0]/zmet[0]):.2e} g/cm³")

    # --- Initial conditions: lognormal ---
    r_np = np.array(r1)
    dr_np = np.array(dr1)
    rmass_np = np.array(rmass1)
    n_per_bin = make_lognormal_modes(r_np, dr_np, rmass_np, config["modes"])

    N_total_init = n_per_bin.sum()
    total_mass_init = (n_per_bin * rmass_np).sum()
    print(f"  Initial lognormal: N_total = {N_total_init:.2f} /cm³, "
          f"mass = {total_mass_init:.2e} g/cm³")
    print(f"  Modes:")
    for mode in config["modes"]:
        print(f"    r_mode={mode['r_mode']*1e4:.3f}um, sigma={mode['sigma']}, N={mode['N']}/cm³")

    pc = jnp.full((NZ, NBIN, NELEM), SMALL_PC, dtype=DTYPE)
    for ib in range(NBIN):
        pc = pc.at[0, ib, 0].set(DTYPE(n_per_bin[ib]) * zmet[0])

    # --- Initial H2SO4 and H2O ---
    h2o_ppmv = config["h2o_ppmv"]
    h2o_mmr = h2o_ppmv * 1e-6 * (WTMOL_H2O_L / WTMOL_AIR)
    h2o_conc = h2o_mmr * float(rhoa[0] / zmet[0])  # g/cm³

    h2so4_pptv = config["h2so4_pptv"]
    h2so4_mmr = h2so4_pptv * 1e-12 * (WTMOL_H2SO4 / WTMOL_AIR)
    h2so4_conc = h2so4_mmr * float(rhoa[0] / zmet[0])
    h2so4_numconc = h2so4_conc / WTMOL_H2SO4 * float(AVG)

    print(f"  H2O:   {h2o_ppmv:.1f} ppmv → {h2o_conc:.2e} g/cm³")
    print(f"  H2SO4: {h2so4_pptv:.2f} pptv → {h2so4_numconc:.2e} molec/cm³")
    print(f"  H2SO4 source: {config['h2so4_source_per_cm3_s']:.1e} /cm³/s")

    gc = jnp.array([[DTYPE(h2so4_conc) * zmet[0]]])

    # --- Particle properties for fall velocity / coagulation kernel ---
    rhop_wet = jnp.full((NZ, NBIN, NGROUP), rho_aer, dtype=DTYPE)
    rrat_arr = jnp.ones((NBIN, NGROUP), dtype=DTYPE)
    rprat_arr = jnp.ones((NBIN, NGROUP), dtype=DTYPE)

    cfg_for_vf = CarmaConfig(
        nbin=NBIN, nelem=NELEM, ngroup=NGROUP, ngas=NGAS, nsolute=0,
        elements=elements, groups=groups, gases=(), solutes=(),
        coag=coag,
        do_coag=True, do_grow=True, do_vtran=False, do_vdiff=False,
        do_thermo=False, do_substep=False, do_explised=False,
        do_incloud=False, do_clearsky=False, do_detrain=False,
        do_pheat=False, do_pheatatm=False, do_cnst_rlh=False,
        itbnd_pc=1, ibbnd_pc=1,
        maxsubsteps=1, minsubsteps=1, maxretries=5, conmax=0.0,
        cstick=1.0, gsticki=1.0, gstickl=1.0, tstick=1.0, dt_threshold=0.0,
        igash2o=-1, igash2so4=0, igasso2=-1,
    )

    vf, re, bpm = setup_vf(
        cfg_for_vf, t, rhoa, zmet, rmu, r_wet, rhop_wet, rrat_arr, rprat_arr)

    DTIME = config["dtime"]
    NSTEP = config["nstep"]

    # --- Storage ---
    history = {
        "times": [0.0],
        "h2so4_conc": [h2so4_numconc],
        "nuc_rate": [0.0],
        "total_N": [float(pc[0, :, 0].sum() / zmet[0])],
        "total_mass": [float((pc[0, :, 0] * rmass1).sum() / zmet[0])],
        "dist_snapshots": [np.array(pc[0, :, 0] / zmet[0])],
        "snapshot_times": [0.0],
        "r_cm": r_np.copy(),
        "dm_cgs": np.array(dm1),
        "rmass_cgs": np.array(rmass1),
    }
    snapshot_mins = config.get("snapshot_min", [0, NSTEP * DTIME / 60])
    snap_times_s = [m * 60.0 for m in snapshot_mins]

    print(f"  Running {NSTEP} steps at dt={DTIME}s ({NSTEP*DTIME/60:.0f} min total)...")

    iz = 0
    for istep in range(NSTEP):
        # -- 1. Vapor pressures, supersaturations --
        pvapl_h2o, pvapi_h2o = vaporp_h2o_murphy2005(t)
        pvapl_h2so4, pvapi_h2so4 = vaporp_h2so4_ayers1980(
            t, jnp.array([DTYPE(h2o_conc)]), pvapl_h2o, zmet)

        ssl, ssi = supersat(t, gc[:, 0], pvapl_h2so4, pvapi_h2so4,
                             DTYPE(WTMOL_H2SO4), zmet)

        h2o_rvap = float(RGAS) / WTMOL_H2O_L
        rh = float(jnp.clip(
            h2o_conc * h2o_rvap * T_init / float(pvapl_h2o[0]),
            0.001, 1.0))

        # -- 2. Nucleation (Vehkamaki) --
        h2so4_now_num = (float(gc[0, 0] / zmet[0]) / WTMOL_H2SO4 * float(AVG))
        nucrate_cgs, mass_cluster, r_cluster = binary_nuc_vehk2002(
            t[0], DTYPE(rh), DTYPE(h2so4_now_num), DTYPE(WTMOL_H2SO4))
        nucrate = float(nucrate_cgs)

        # Target bin for nucleated cluster
        if float(mass_cluster) < float(rmassup1[0]):
            nucbin = 0
        else:
            nucbin = int(np.clip(
                1 + np.floor(np.log(float(mass_cluster) / float(rmassup1[0]))
                             / np.log(rmrat)),
                0, NBIN - 1))

        # -- 3. Growth kernels (recompute each step for T/p changes) --
        diffus_arr, rlhe_arr, rlhm_arr = setup_grow(
            t, p_cgs, rhoa, zmet, -1, 0, NGAS, False)

        rrat_gkern = jnp.ones((NBIN, NGROUP), dtype=DTYPE)
        eshape_arr_gkern = jnp.array([1.0])
        _, akelvin, akelvini, gro, gro1, _, _, _ = setup_gkern(
            t, p_cgs, rhoa, zmet, rmu, thcond, diffus_arr, rlhe_arr, rlhm_arr,
            re, r_wet, rlow_wet, rrat_gkern, eshape_arr_gkern, is_ice_arr,
            gwtmol_arr, igrowgas_arr, 1.0, 1.0, 1.0, NBIN, NGROUP, NGAS)

        # -- 4. Coagulation kernel --
        ckernel = setup_ckern(
            cfg_for_vf, t, rhoa, zmet, rmu, r_wet, rrat_arr, rprat_arr,
            bpm, rmass_2d, re, vf)

        # Broadcast for supersat arrays
        pvapl_2d = pvapl_h2so4[:, None]
        pvapi_2d = pvapi_h2so4[:, None]
        supsatl_2d = ssl[:, None]
        supsati_2d = ssi[:, None]

        pconmax = jnp.zeros((NZ, NGROUP), dtype=DTYPE)
        pconmax = pconmax.at[:, 0].set(jnp.max(pc[:, :, 0], axis=1) / zmet)

        # -- 5. Save condensate --
        prev_ice, prev_liq = totalcondensate(
            pc, rmass_2d, igroup_arr, is_ice_arr, igrowgas_arr,
            NBIN, NGROUP, NGAS, iz)

        # -- 6. Growth loss rates (PPM) --
        growlg = jnp.zeros((NBIN, NGROUP), dtype=DTYPE)
        evaplg = jnp.zeros((NBIN, NGROUP), dtype=DTYPE)
        growlg, evaplg = growevapl(
            pc, growlg, evaplg,
            supsatl_2d, supsati_2d, pvapl_2d, pvapi_2d,
            akelvin, akelvini, gro, gro1,
            rup_wet, rmass_2d, dm_2d, pconmax,
            pratt, prat, pden1, palr,
            is_ice_arr, igrowgas_arr, ienconc_arr,
            DTIME, iz, NBIN, NGROUP)

        # -- 7. Nucleation production --
        rhompe_arr = jnp.zeros((NBIN, NELEM), dtype=DTYPE)
        if nucrate > 0:
            rhompe_arr = rhompe_arr.at[nucbin, 0].set(DTYPE(nucrate) * zmet[0])
        gas_loss_nuc = float(nucrate) * float(mass_cluster)  # g/cm³/s

        # -- 8. Growth + psolve --
        growpe_arr = jnp.zeros((NBIN, NELEM), dtype=DTYPE)
        evappe_arr = jnp.zeros((NBIN, NELEM), dtype=DTYPE)
        rnucpe_arr = jnp.zeros((NBIN, NELEM), dtype=DTYPE)
        pc_nucl = jnp.zeros_like(pc)
        rnuclg_zero = jnp.zeros((NBIN, NGROUP, NGROUP), dtype=DTYPE)

        for ibin in range(NBIN):
            growpe_arr = growp(pc, growpe_arr, growlg, pconmax,
                               iz, ibin, 0, 0, 0)
            pc, pc_nucl = psolve(
                pc, pc_nucl, growpe_arr, evappe_arr, rnucpe_arr, rhompe_arr,
                growlg, evaplg, rnuclg_zero, DTIME, iz, ibin, 0, 0, NGROUP)

        # -- 9. Evaporation --
        evappe_arr = jnp.zeros((NBIN, NELEM), dtype=DTYPE)
        evappe_arr = evapp(
            pc, evappe_arr, evaplg, pconmax, ienconc_arr, itype_arr,
            igroup_arr, iz, NBIN, NGROUP, NELEM)
        pc = downgevapply(pc, evappe_arr,
                          jnp.zeros((NBIN, NELEM), dtype=DTYPE),
                          DTIME, iz, NBIN, NELEM)

        # -- 10. Coagulation (microslow) --
        if config["do_coag"]:
            pcl = pc  # save before coag
            pc = microslow_jit(pc, pcl, ckernel, pconmax, zmet, DTIME)

        # -- 11. Gas solver --
        curr_ice, curr_liq = totalcondensate(
            pc, rmass_2d, igroup_arr, is_ice_arr, igrowgas_arr,
            NBIN, NGROUP, NGAS, iz)
        condensation_loss = (curr_ice[0] - prev_ice[0]) + (curr_liq[0] - prev_liq[0])
        gasprod = -condensation_loss / DTIME - gas_loss_nuc * zmet[0]

        # H2SO4 source (photochemistry)
        h2so4_src = config["h2so4_source_per_cm3_s"]
        src_g = h2so4_src * WTMOL_H2SO4 / float(AVG)  # g/cm³/s
        gasprod += src_g * zmet[0]

        gc = gc.at[iz, 0].add(DTIME * gasprod)
        gc = jnp.maximum(gc, DTYPE(0.0))

        # -- Store diagnostics --
        total_N = float(pc[0, :, 0].sum() / zmet[0])
        total_mass = float((pc[0, :, 0] * rmass1).sum() / zmet[0])
        h2so4_now = (float(gc[0, 0] / zmet[0]) / WTMOL_H2SO4 * float(AVG))

        time_s = DTIME * (istep + 1)
        history["times"].append(time_s)
        history["h2so4_conc"].append(h2so4_now)
        history["nuc_rate"].append(nucrate)
        history["total_N"].append(total_N)
        history["total_mass"].append(total_mass)

        # Snapshot if time matches
        for t_target in snap_times_s:
            if abs(time_s - t_target) < DTIME * 0.5:
                history["dist_snapshots"].append(np.array(pc[0, :, 0] / zmet[0]))
                history["snapshot_times"].append(time_s)
                break

        if (istep + 1) % 30 == 0:
            print(f"    t={time_s/60:5.0f}min: "
                  f"N={total_N:.2e} /cm³, "
                  f"mass={total_mass:.2e} g/cm³, "
                  f"H2SO4={h2so4_now:.2e} /cm³, "
                  f"nuc={nucrate:.2e} /cm³/s", flush=True)

    # Ensure final snapshot
    if history["snapshot_times"][-1] < DTIME * NSTEP - DTIME:
        history["dist_snapshots"].append(np.array(pc[0, :, 0] / zmet[0]))
        history["snapshot_times"].append(DTIME * NSTEP)

    return history


def plot_scenario(history, config, outpath):
    """Create diagnostic plots."""
    times_min = np.array(history["times"]) / 60.0
    r_um = history["r_cm"] * 1e4
    dm = history["dm_cgs"]

    fig, axes = plt.subplots(2, 3, figsize=(19, 10))

    # Top row: time series
    ax = axes[0, 0]
    ax.semilogy(times_min, history["total_N"], "b-", lw=2)
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Total N [#/cm³]")
    ax.set_title("Particle Number")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.semilogy(times_min, np.maximum(history["total_mass"], 1e-30), "b-", lw=2)
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Mass [g/cm³]")
    ax.set_title("Particle Mass")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.semilogy(times_min, np.maximum(history["h2so4_conc"], 1e4), "g-", lw=2)
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("H2SO4 [molec/cm³]")
    ax.set_title("H2SO4 Gas")
    ax.grid(True, alpha=0.3)

    # Bottom row: distributions + nuc rate
    ax = axes[1, 0]
    ax.semilogy(times_min[1:], np.maximum(history["nuc_rate"][1:], 1e-10), "r-", lw=2)
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("J [#/cm³/s]")
    ax.set_title("Vehkamaki Nucleation Rate")
    ax.grid(True, alpha=0.3)

    # Number size distribution (dN/dlogD)
    ax = axes[1, 1]
    for t_s, dist in zip(history["snapshot_times"], history["dist_snapshots"]):
        # dN/d(log10 r) — more physical representation
        logr = np.log10(r_um)
        dlogr = np.gradient(logr)
        dN_dlogr = dist / dlogr
        ax.loglog(r_um, np.maximum(dN_dlogr, 1e-5), "-",
                  label=f"t={t_s/60:.0f}min", alpha=0.8)
    ax.set_xlabel("Radius [um]")
    ax.set_ylabel("dN/dlog(r) [#/cm³]")
    ax.set_title("Number Size Distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Mass distribution
    ax = axes[1, 2]
    for t_s, dist in zip(history["snapshot_times"], history["dist_snapshots"]):
        logr = np.log10(r_um)
        dlogr = np.gradient(logr)
        dM_dlogr = (dist * history["rmass_cgs"]) / dlogr
        ax.loglog(r_um, np.maximum(dM_dlogr, 1e-25), "-",
                  label=f"t={t_s/60:.0f}min", alpha=0.8)
    ax.set_xlabel("Radius [um]")
    ax.set_ylabel("dM/dlog(r) [g/cm³]")
    ax.set_title("Mass Size Distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"{config['title']}\n"
                 f"T={config['T']}K, p={config['p_hpa']} hPa, "
                 f"H2O={config['h2o_ppmv']} ppmv, "
                 f"H2SO4_init={config['h2so4_pptv']} pptv, "
                 f"source={config['h2so4_source_per_cm3_s']:.0e} /cm³/s, "
                 f"coag={'ON' if config['do_coag'] else 'OFF'}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {outpath}")


def main():
    outdir = Path("plots/sulfate_realistic")
    outdir.mkdir(parents=True, exist_ok=True)

    for name in ["stratosphere", "troposphere_polluted", "troposphere_marine"]:
        print(f"\n{'='*60}")
        print(f"Scenario: {name}")
        print(f"{'='*60}")
        config = setup_scenario(name)
        history = run_scenario(config)
        plot_scenario(history, config, outdir / f"sulfate_{name}.png")

    print(f"\nAll plots saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
