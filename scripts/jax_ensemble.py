"""JAX ensemble runner (Phase 10).

Drives the **faithful** Phase-9 step (``make_step_full_faithful``)
over the 1000-scenario ensemble. Each scenario runs the same
``nstep × dtime`` sulfate simulation as the Fortran ensemble
(``carma_sulfatetest_ensemble.F90``) and records ``(T_final,
gc_h2so4_final, pc_final)``.

The faithful chain (Phase 9.4) replaces the legacy operator-split
``make_step_full`` + ``make_step_coag`` pair: one outer step now
runs ``microslow`` once, then the adaptive-substep retry loop over
``microfast_full`` (sulfnuc + growevapl + growp/psolve + evapp +
gsolve + tsolve). Bench-validated 1000/1000 against Fortran
microfast at near-machine ε (Phase 9.1) and across multi-substep
schedules (Phase 9.2).

Output is a compressed NPZ with the same keys as the Fortran
aggregator (``T_final``, ``gc_h2so4_final``, ``pc_final``,
``nstep_ran``, ``status``).
"""

import argparse
import sys
import time
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "tests" / "unit"))

from generate_sulfate_scenarios import load_scenarios
from carma.constants import AVG, BK, WTMOL_H2O
from carma.precision import DTYPE
from carma.prestep import prestep
from carma.step_full_faithful import make_step_full_faithful
from carma.utils.smallconc import maxconc
from carma.vapor_pressure import vaporp_h2o_murphy2005

# Reuse the env fixture from the step_full tests. _minimal_config
# is rebuilt with Fortran-matching bin grid (see _fortran_config).
from test_step_full import _growth_env, _minimal_config as _test_minimal_config


_GWTMOL_H2SO4 = 98.078479    # matches Fortran CARMAGAS_Create call

# Fortran sulfatetest bin grid (must match
# carma_sulfatetest_ensemble.F90 for apples-to-apples parity):
_FORTRAN_NBIN = 38
_FORTRAN_RMIN_CM = 2.0e-8       # 20 nm
_FORTRAN_RMRAT = 2.0
_FORTRAN_RHO_SULF = 1.923       # g/cm³


def _fortran_matching_config():
    """Build the CarmaConfig with the same bin grid as the Fortran
    ensemble binary. Uses the same structural layout as the test
    fixture, but with (nbin, rmin, rmrat, rho) taken from the
    Fortran constants."""
    from carma.config import (
        CarmaConfig, ElementConfig, GroupConfig, GasConfig, SoluteConfig,
    )
    import jax.numpy as jnp

    # Use carma.bins.setup_bins for the bin geometry — it already
    # implements Fortran's setupbins.F90:140-164 formulas correctly.
    # The previous open-coded geometry here used rmassup = rmass·√rmrat
    # and rlow = rup-derived, but Fortran uses
    # rmassup = 2·rmrat/(rmrat+1)·rmass and dr = vrfact·(rmass/rho)^(1/3),
    # which give rup/r ≈ 1.10 (not 1.122) for rmrat=2. The mismatch
    # biased setup_gkern's Knudsen → gro by ~4% and rup_wet by ~2%.
    from carma.bins import setup_bins
    nbin = _FORTRAN_NBIN
    rho = _FORTRAN_RHO_SULF
    rmin = _FORTRAN_RMIN_CM
    rmrat = _FORTRAN_RMRAT
    r, rmass, vol, dr, dm, rup, rlow, rmassup = setup_bins(rmin, rmrat, nbin, rho)
    r_np = np.asarray(r)

    group = GroupConfig(
        name="sulfate", ishape=1, ienconc=0,
        is_ice=False, is_cloud=False, is_sulfate=True,
        do_vtran=False, do_drydep=False,
        ifallrtn=1, irhswell=3, rmrat=rmrat, eshape=1.0, rmin=rmin,
        r=r, rmass=rmass, vol=vol,
        dr=dr, dm=dm,
        rmassup=rmassup,
        rup=rup, rlow=rlow,
        rrat=jnp.ones(nbin), rprat=jnp.ones(nbin), arat=jnp.ones(nbin),
    )
    element = ElementConfig(
        name="sulfate_num", rho=jnp.full((nbin,), rho), igroup=0,
        itype=2, icomposition=0, isolute=0, kappa=0.65,
    )
    # Thresholds match Fortran sulfatetest_ensemble's CARMAGAS_Create calls
    # (0.1 / 0.1 for both H2O and H2SO4).
    gas_h2o = GasConfig(name="H2O", wtmol=18.016, ivaprtn=2, icomposition=1,
                         dgc_threshold=0.1, ds_threshold=0.1)
    gas_h2so4 = GasConfig(name="H2SO4", wtmol=_GWTMOL_H2SO4, ivaprtn=4,
                           icomposition=2,
                           dgc_threshold=0.1, ds_threshold=0.1)
    solute = SoluteConfig(name="sulfate", ions=3, wtmol=_GWTMOL_H2SO4, rho=rho)

    # Build coagulation tables (sulfate ⊕ sulfate → sulfate, source=number).
    from carma.coagulation.setup_coag import setup_coag
    coag_cfg = setup_coag(
        nbin=nbin, ngroup=1, nelem=1,
        groups=(group,), elements=(element,),
        icoag_input=np.asarray([[0]], dtype=np.int32),         # sulfate→sulfate
        icoagelem_input=np.asarray([[0]], dtype=np.int32),     # number contributes
    )

    return CarmaConfig(
        nbin=nbin, nelem=1, ngroup=1, ngas=2, nsolute=1,
        elements=(element,), groups=(group,),
        gases=(gas_h2o, gas_h2so4), solutes=(solute,),
        coag=coag_cfg,
        do_coag=True, do_grow=True, do_vtran=False, do_vdiff=False,
        do_thermo=True, do_substep=True, do_explised=False,
        do_incloud=False, do_clearsky=True, do_detrain=False,
        do_pheat=False, do_pheatatm=False, do_cnst_rlh=False,
        itbnd_pc=1, ibbnd_pc=1,
        maxsubsteps=32, minsubsteps=1, maxretries=16, conmax=1e-4,
        cstick=1.0, gsticki=1.0, gstickl=1.0, tstick=1.0, dt_threshold=1.0,
        igash2o=0, igash2so4=1, igasso2=-1,
    )


_minimal_config = _fortran_matching_config


def _init_lognormal_pc(r_bins_cm, mu_nm, sigma_g, total_mass_g_per_cm3=1e-18):
    """Build a lognormal number-density vector per bin, normalised so
    total mass equals ``total_mass_g_per_cm3`` (matches Fortran
    ensemble's ``1e-18 g/g`` tiny seed)."""
    r = np.asarray(r_bins_cm)
    mu_cm = mu_nm * 1e-7
    log_sigma = np.log(sigma_g)
    log_r = np.log(r)
    pc = np.exp(-0.5 * ((log_r - np.log(mu_cm)) / log_sigma) ** 2) \
         / (r * log_sigma * np.sqrt(2 * np.pi))
    rho_sulf = 1.923
    rmass = (4.0 / 3.0) * np.pi * r**3 * rho_sulf
    total = np.sum(pc * rmass)
    if total > 0:
        pc = pc * (total_mass_g_per_cm3 / total)
    return jnp.asarray(pc)


def _compute_ppm_coefs(cfg):
    """PPM coefficient arrays used by growevapl (Colella-Woodward 1984).
    Depend only on the bin widths, so computed once per config."""
    import jax.numpy as jnp
    nbin = cfg.nbin
    ngroup = cfg.ngroup
    dm_arr = [np.asarray(g.dm) for g in cfg.groups]
    rmass_arr = [np.asarray(g.rmass) for g in cfg.groups]
    rmassup_arr = [np.asarray(g.rmassup) for g in cfg.groups]
    rmrat_val = float(cfg.groups[0].rmrat)

    pratt = np.zeros((3, nbin, ngroup))
    prat = np.zeros((4, nbin, ngroup))
    pden1 = np.zeros((nbin, ngroup))
    palr = np.zeros((4, ngroup))
    for ig in range(ngroup):
        dm_np = dm_arr[ig]
        rmass_np = rmass_arr[ig]
        rmassup_np = rmassup_arr[ig]
        for ibin in range(1, nbin - 1):
            dm_im1, dm_i, dm_ip1 = dm_np[ibin - 1], dm_np[ibin], dm_np[ibin + 1]
            pratt[0, ibin, ig] = dm_i / (dm_im1 + dm_i + dm_ip1)
            pratt[1, ibin, ig] = (2.0 * dm_im1 + dm_i) / (dm_ip1 + dm_i)
            pratt[2, ibin, ig] = (2.0 * dm_ip1 + dm_i) / (dm_im1 + dm_i)
        for ibin in range(1, nbin - 2):
            dm_im1, dm_i = dm_np[ibin - 1], dm_np[ibin]
            dm_ip1 = dm_np[ibin + 1]
            dm_ip2 = dm_np[min(ibin + 2, nbin - 1)]
            prat[0, ibin, ig] = dm_i / (dm_i + dm_ip1)
            prat[1, ibin, ig] = 2.0 * dm_ip1 * dm_i / (dm_i + dm_ip1)
            prat[2, ibin, ig] = (dm_im1 + dm_i) / (2.0 * dm_i + dm_ip1)
            prat[3, ibin, ig] = (dm_ip2 + dm_ip1) / (2.0 * dm_ip1 + dm_i)
            pden1[ibin, ig] = dm_im1 + dm_i + dm_ip1 + dm_ip2
        denom_low = rmass_np[1] - rmass_np[0]
        denom_high = rmass_np[nbin - 1] - rmass_np[nbin - 2]
        palr[0, ig] = (rmassup_np[0] - rmass_np[0]) / denom_low
        palr[1, ig] = (rmassup_np[0] / rmrat_val - rmass_np[0]) / denom_low
        palr[2, ig] = (rmassup_np[nbin - 2] - rmass_np[nbin - 2]) / denom_high
        palr[3, ig] = (rmassup_np[nbin - 1] - rmass_np[nbin - 2]) / denom_high
    return (jnp.asarray(pratt), jnp.asarray(prat),
            jnp.asarray(pden1), jnp.asarray(palr))


def _env_for_scenario(T_K, p_hPa, rh, h2so4_pptv, mu_nm, sigma_g, cfg,
                        ppm_coefs):
    """Produce a populated env dict for one scenario, matching the
    make_step_full call signature. Growth kernels come from
    setup_atm + setup_grow + setup_gkern per scenario (Option A-lite)."""
    import jax.numpy as jnp
    from carma.setup_atm import setup_atm
    from carma.setup_grow import setup_grow
    from carma.setup_gkern import setup_gkern
    from carma.constants import GRAV, R_AIR, RPA2CGS

    nz = 1
    nbin = cfg.nbin
    ngroup = cfg.ngroup
    ngas = cfg.ngas

    # Build a single-layer atmosphere column at the scenario's T, p
    T = jnp.asarray([T_K], dtype=DTYPE)
    p_cgs = jnp.asarray([p_hPa * 100.0 * float(RPA2CGS)], dtype=DTYPE)   # hPa → Pa → dyne/cm²
    rho_approx = float(p_cgs[0]) / (float(R_AIR) * T_K)
    # Build thin (10 m) column so setup_atm is well-defined; results are
    # per-layer anyway.
    deltaz = 1.0e3    # cm
    zc = jnp.asarray([1.0e5])     # 1 km above the virtual ground
    zl = jnp.asarray([zc[0] - deltaz / 2, zc[0] + deltaz / 2])
    pl = jnp.asarray([p_cgs[0] + 0.5 * deltaz * rho_approx * float(GRAV),
                       p_cgs[0] - 0.5 * deltaz * rho_approx * float(GRAV)])

    from carma.enums import GridType
    rhoa, dz, zmet, zmetl, rmu, thcond, rhoa_wet = setup_atm(
        T, p_cgs, pl, zc, zl, GridType.I_CART,
    )

    # setup_grow: gas diffusivity + latent heats. igash2o=0, igash2so4=1 from cfg.
    diffus, rlhe, rlhm = setup_grow(
        T, p_cgs, rhoa, zmet,
        igash2o=cfg.igash2o, igash2so4=cfg.igash2so4,
        ngas=ngas, do_cnst_rlh=False,
    )

    # H2SO4 gas mmr (g/g) and gas mass density (g/cm³) — used both for
    # the gc tensor and for the wet-radius hygroscopic-growth call.
    h2so4_mmr = h2so4_pptv * 1.0e-12 * (98.0 / 29.0)        # g/g
    # Initial H2O mmr seed: match Fortran's `carma_sulfatetest_ensemble.F90`
    # line 173-175 which uses the simplified Murphy-Koop analytic. (The
    # exact Murphy-Koop formula gets re-applied inside microfast.)
    pvapl_Pa = jnp.exp(54.842763 - 6763.22 / T_K
                        - 4.210 * jnp.log(T_K) + 0.000367 * T_K)
    h2o_mmr = rh * float(pvapl_Pa) * 18.0 / (29.0 * p_hPa * 100.0)
    rhoa_pure = float(rhoa[0]) / float(zmet[0])      # g/cm³ pure-air density
    h2o_mass_cgs = h2o_mmr * rhoa_pure                # g/cm³
    # h2o_vp for wet-radius / wtpct: must use the EXACT Murphy-Koop
    # (vaporp_h2o_murphy2005), not the seed analytic. Fortran's wetr and
    # setupgkern use cstate%pvapl which is set by vaporp_h2o_murphy2005
    # inside microfast — using the analytic here biases pvapl 1.7% high
    # at stratospheric temps and ripples into Kelvin / wtpct / rwet.
    pvapl_h2o_arr, _ = vaporp_h2o_murphy2005(T)
    h2o_vp_cgs = float(pvapl_h2o_arr[0])              # already dyne/cm²

    # Wet radii via Fortran's I_WTPCT_H2SO4 path (sulfate group's irhswell).
    # Recomputed once per scenario at initial T/RH/H2O — Fortran would refresh
    # each Step call, but for this single-cell test T is constant and H2O
    # drifts negligibly, so a static approximation tracks closely.
    from carma.wetr import get_wetr
    from carma.enums import SwellMethod
    grp = cfg.groups[0]
    r_dry = jnp.asarray(grp.r, dtype=DTYPE)
    rup_dry = jnp.asarray(grp.rup, dtype=DTYPE)
    rlow_dry = jnp.asarray(grp.rlow, dtype=DTYPE)
    rho_dry_per_bin = jnp.full(r_dry.shape, _FORTRAN_RHO_SULF, dtype=DTYPE)
    swell = int(SwellMethod.I_WTPCT_H2SO4)
    r_wet_1d, rho_wet = get_wetr(r_dry, rho_dry_per_bin, rh, T_K, swell,
        h2o_mass=h2o_mass_cgs, h2o_vp=h2o_vp_cgs,
        gwtmol_h2so4=DTYPE(_GWTMOL_H2SO4))
    rup_wet_1d, _ = get_wetr(rup_dry, rho_dry_per_bin, rh, T_K, swell,
        h2o_mass=h2o_mass_cgs, h2o_vp=h2o_vp_cgs,
        gwtmol_h2so4=DTYPE(_GWTMOL_H2SO4))
    rlow_wet_1d, _ = get_wetr(rlow_dry, rho_dry_per_bin, rh, T_K, swell,
        h2o_mass=h2o_mass_cgs, h2o_vp=h2o_vp_cgs,
        gwtmol_h2so4=DTYPE(_GWTMOL_H2SO4))
    r_wet = r_wet_1d[None, :, None]                 # (nz, nbin, ngroup)
    rup_wet = rup_wet_1d[None, :, None]
    rlow_wet = rlow_wet_1d[None, :, None]
    re = jnp.zeros((nz, nbin, ngroup), dtype=DTYPE)
    rrat = jnp.ones((nbin, ngroup), dtype=DTYPE)
    eshape_arr = jnp.asarray([float(g.eshape) for g in cfg.groups], dtype=DTYPE)
    is_ice_arr = np.asarray([bool(g.is_ice) for g in cfg.groups])
    gwtmol_arr = jnp.asarray([float(g.wtmol) for g in cfg.gases], dtype=DTYPE)
    igrowgas_arr = np.asarray(
        [cfg.igash2so4 for _ in cfg.elements], dtype=np.int32)

    # Fall velocity, Reynolds number, slip correction (needed for coag kernel
    # AND for setup_gkern — re is the real Reynolds, not zeros). Uses wet
    # radii and per-bin wet density from get_wetr above.
    from carma.setup_vf import setup_vf_jit
    rmass_2d = jnp.asarray(cfg.groups[0].rmass)[:, None]              # (nbin, ngroup)
    rho_p_2d = rho_wet[:, None]                                        # (nbin, ngroup)
    rrat_2d = jnp.ones((nbin, ngroup), dtype=DTYPE)
    rprat_2d = jnp.ones((nbin, ngroup), dtype=DTYPE)
    vf, re_real, bpm = setup_vf_jit(
        T, rhoa, zmet, rmu, r_wet, rho_p_2d, rrat_2d, rprat_2d,
    )

    # wtpct (sulfate weight percent) for setup_gkern's H2SO4 Kelvin
    # override. Same Tabazadeh1997 formula sulfnuc uses internally.
    from carma.sulfate_utils import wtpct_tabaz
    wtpct_arr = wtpct_tabaz(T, jnp.asarray([h2o_mass_cgs]), jnp.asarray([h2o_vp_cgs]))

    _, akelvin, akelvini, gro, gro1, gro2, _, _ = setup_gkern(
        T, p_cgs, rhoa, zmet, rmu, thcond, diffus, rlhe, rlhm,
        re_real, r_wet, rlow_wet, rrat, eshape_arr, is_ice_arr,
        gwtmol_arr, igrowgas_arr,
        cfg.gstickl, cfg.gsticki, cfg.tstick,
        nbin, ngroup, ngas,
        igash2o=cfg.igash2o, igash2so4=cfg.igash2so4, wtpct=wtpct_arr,
    )

    # Coagulation kernel (Brownian + Van der Waals + Fuchs + grav).
    # use_vw mask is (NGROUP, NGROUP) bool: True where BOTH colliding
    # groups are is_sulfate (Chan & Mozurkewich 2001 VdW enhancement).
    # Sulfate test has a single sulfate group → use_vw = [[True]].
    from carma.setup_ckern import setup_ckern_jit
    is_sulfate = np.asarray([bool(g.is_sulfate) for g in cfg.groups])
    use_vw = jnp.asarray(is_sulfate[:, None] & is_sulfate[None, :])
    ckernel = setup_ckern_jit(
        T, rhoa, zmet, rmu, r_wet, rrat_2d, rprat_2d, bpm, rmass_2d,
        re_real, vf, cfg.cstick,
        use_vw=use_vw,
    )

    # gc in CGS: mmr × rhoa (rhoa already includes zmet factor)
    gc = jnp.asarray([[h2o_mmr * float(rhoa[0]),
                        h2so4_mmr * float(rhoa[0])]], dtype=DTYPE)

    # Initial lognormal aerosol seed. Fortran's
    # `carma_sulfatetest_ensemble.F90` normalises so Σ mmr = 1e-18 g/g.
    # In JAX-internal CGS, pc is #/cm³/z and the equivalent mass-density
    # constraint is Σ pc·rmass = 1e-18 · rhoa, so we multiply by rhoa
    # here for apples-to-apples parity with Fortran.
    r_bins_np = np.asarray(r_dry)
    pc_sulf = _init_lognormal_pc(r_bins_np, mu_nm, sigma_g,
                                   total_mass_g_per_cm3=1.0e-18 * float(rhoa[0]))
    pc = jnp.zeros((nz, nbin, cfg.nelem), dtype=DTYPE)
    pc = pc.at[0, :, 0].set(pc_sulf)

    pvapl_arr = pvapl_Pa * float(RPA2CGS)   # Pa → dyne/cm²; used only as scalar hint

    pratt, prat, pden1, palr = ppm_coefs

    ds_threshold_arr = jnp.zeros((ngas,), dtype=DTYPE)

    return dict(
        pc=pc, gc=gc, t=T,
        rhoa=rhoa, zmet=zmet, rlhe=rlhe, rlhm=rlhm, diffus=diffus,
        akelvin=akelvin, akelvini=akelvini,
        gro=gro, gro1=gro1, gro2=gro2,
        rup_wet=rup_wet, rlow_wet=rlow_wet,
        pratt=pratt, prat=prat, pden1=pden1, palr=palr,
        pvapl=float(pvapl_arr),
        ds_threshold_arr=ds_threshold_arr,
        # Coag-side inputs (Phase 10.4b):
        ckernel=ckernel,
    )


def run_jax_ensemble(scenarios, dtime_s=1800.0, nstep=100):
    """Run every scenario through ``make_step_full_faithful`` end-to-end.

    The faithful chain bundles microslow (coag) + microfast_full inside
    one step closure and runs the adaptive substep retry per Fortran's
    ``newstate_calc.F90``. ``prestep`` is invoked between steps to
    refresh ``(pcl, gcl, told, d_gc, d_t, pconmax)`` exactly as Fortran
    does.

    Loops in Python over scenarios (not vmap); per-scenario warm time
    is small enough that 1000 × 100 = 100k inner steps complete in a
    few minutes."""
    cfg = _minimal_config()
    print(f"  bin grid: nbin={cfg.nbin}, rmin={float(cfg.groups[0].r[0])*1e7:.1f} nm, "
          f"rmrat={cfg.groups[0].rmrat}, rmax={float(cfg.groups[0].r[-1])*1e4:.2f} μm",
          flush=True)
    ppm_coefs = _compute_ppm_coefs(cfg)
    step = make_step_full_faithful(cfg, ppm_coefs=ppm_coefs)

    # Static element/group descriptors for prestep.
    itype_arr = jnp.asarray([e.itype for e in cfg.elements])
    ienconc_arr = jnp.asarray([g.ienconc for g in cfg.groups])
    igelem_arr = jnp.asarray([e.igroup for e in cfg.elements])
    rmass_2d = jnp.stack(
        [jnp.asarray(g.rmass, dtype=DTYPE) for g in cfg.groups], axis=1,
    )

    n = int(scenarios["_n"])
    T_final = np.zeros(n, dtype=np.float64)
    gc_final = np.zeros(n, dtype=np.float64)
    pc_final = np.zeros((n, cfg.nbin), dtype=np.float64)
    nstep_ran = np.full(n, nstep, dtype=np.int64)
    status = ["ok"] * n

    for i in range(n):
        env = _env_for_scenario(
            scenarios["T"][i], scenarios["p"][i], scenarios["rh"][i],
            scenarios["h2so4_pptv"][i], scenarios["aerosol_mu_nm"][i],
            scenarios["aerosol_sigma_g"][i], cfg, ppm_coefs,
        )
        pc = env["pc"]
        gc = env["gc"]
        t = env["t"]

        for _step_idx in range(nstep):
            # No external advection / transport in this single-cell test, so
            # the "saved" state at the start of each outer step is exactly
            # the current state. Pass it as both (pc, gc, t) and
            # (pcl, gcl, told); prestep then computes d_gc=d_t=0 and
            # produces pconmax for microfast.
            pc, gc, t, pcl, gcl, d_gc, d_t, pconmax = prestep(
                pc, gc, t, pc, gc, t, env["zmet"],
                itype_arr, ienconc_arr, igelem_arr, rmass_2d,
                do_substep=True, do_coag=cfg.do_coag,
            )

            pc, gc, t, _diag = step(
                pc=pc, gc=gc, t=t, dtime=dtime_s,
                rhoa=env["rhoa"], zmet=env["zmet"],
                akelvin=env["akelvin"], akelvini=env["akelvini"],
                gro=env["gro"], gro1=env["gro1"], rup_wet=env["rup_wet"],
                rlhe=env["rlhe"], rlhm=env["rlhm"],
                ckernel=env["ckernel"], pconmax=pconmax,
                ds_threshold_arr=env["ds_threshold_arr"],
                pcl=pcl, gcl=gcl, told=t, d_gc=d_gc, d_t=d_t,
                dt_threshold=DTYPE(cfg.dt_threshold),
            )

        # Convert JAX internal-CGS outputs to Fortran's MMR convention:
        #   gc [g/cm³/z] / rhoa [g/cm³/z]  →  mmr [g/g]
        #   pc [#/cm³/z] × rmass [g] / rhoa → mmr_per_bin [g/g]
        rhoa = float(env["rhoa"][0])              # g/cm³/z
        rmass = np.asarray(cfg.groups[0].rmass)     # g per particle per bin
        T_final[i] = float(t[0])
        gc_final[i] = float(gc[0, 1]) / rhoa                    # g/g
        pc_final[i] = np.asarray(pc[0, :, 0]) * rmass / rhoa   # g/g per bin

        if i % 50 == 0:
            print(f"  scenario {i:4d}/{n}: T={T_final[i]:.1f}K, "
                  f"gc_H2SO4={gc_final[i]:.3e}", flush=True)

    return dict(
        T_final=T_final, gc_h2so4_final=gc_final, pc_final=pc_final,
        nstep_ran=nstep_ran, status=status,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=Path,
                        default=Path("data/sulfate_scenarios_1000.npz"))
    parser.add_argument("--out", type=Path,
                        default=Path("data/sulfate_jax_outputs.npz"))
    parser.add_argument("--dtime", type=float, default=1800.0)
    parser.add_argument("--nstep", type=int, default=100)
    parser.add_argument("--n", type=int, default=None,
                        help="Override scenario count (for smoke tests)")
    args = parser.parse_args()

    scenarios = load_scenarios(args.scenarios)
    if args.n is not None:
        # Slice to N scenarios for quick runs
        scenarios = {k: (v[:args.n] if hasattr(v, "shape") and v.ndim > 0
                         else v)
                     for k, v in scenarios.items()}
        scenarios["_n"] = args.n

    n = int(scenarios["_n"])
    print(f"Running JAX ensemble over {n} scenarios "
          f"(dtime={args.dtime}s × {args.nstep} steps)...")

    t0 = time.perf_counter()
    results = run_jax_ensemble(scenarios,
                                dtime_s=args.dtime, nstep=args.nstep)
    wall = time.perf_counter() - t0

    n_ok = sum(1 for s in results["status"] if s == "ok")
    print(f"\n  wall time: {wall:.1f} s ({wall/n*1000:.2f} ms / scenario)")
    print(f"  status: {n_ok}/{n} ok, {n - n_ok} errors")

    np.savez_compressed(args.out, **{
        k: v for k, v in results.items() if k != "status"
    })
    # Write status list separately as a plain text sidecar
    (args.out.with_suffix(".status.txt")).write_text(
        "\n".join(f"{i:5d}\t{s}" for i, s in enumerate(results["status"]))
    )
    print(f"  saved to {args.out}")


if __name__ == "__main__":
    main()
