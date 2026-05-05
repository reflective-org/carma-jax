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


def _refresh_env(T, p_cgs, gc, cfg, ppm_coefs):
    """Rebuild every state-dependent env array from current ``(T, p_cgs, gc)``.

    Fortran rebuilds the full env each ``CARMASTATE_Step`` call (see
    ``carma_sulfatetest_ensemble.F90:204-234``: ``CARMASTATE_Create`` +
    ``SetGas`` + ``Step`` per outer iteration). Doing the same in JAX
    closes the bias that creeps in once H₂SO₄ has been consumed and
    H₂O has redistributed — wet radii / Kelvin / growth & coag kernels
    all depend on current gc[H₂O], not initial RH.

    Static inputs (bin geometry, group/element flags, PPM coefs) come
    from ``cfg`` and ``ppm_coefs`` and don't need rebuilding.

    Returns:
        env dict matching the ``make_step_full_faithful`` step closure
        signature (plus ``rmu`` etc. for callers that want them).
    """
    from carma.setup_atm import setup_atm
    from carma.setup_grow import setup_grow
    from carma.setup_gkern import setup_gkern
    from carma.setup_vf import setup_vf_jit
    from carma.setup_ckern import setup_ckern_jit
    from carma.wetr import get_wetr
    from carma.sulfate_utils import wtpct_tabaz
    from carma.enums import GridType, SwellMethod
    from carma.constants import GRAV, R_AIR

    nbin = cfg.nbin
    ngroup = cfg.ngroup
    ngas = cfg.ngas
    nz = 1

    # Single thin-layer atmosphere column at current (T, p_cgs).
    rho_approx = float(p_cgs[0]) / (float(R_AIR) * float(T[0]))
    deltaz = 1.0e3                          # cm
    zc = jnp.asarray([1.0e5])               # 1 km above the virtual ground
    zl = jnp.asarray([zc[0] - deltaz / 2, zc[0] + deltaz / 2])
    pl = jnp.asarray([p_cgs[0] + 0.5 * deltaz * rho_approx * float(GRAV),
                       p_cgs[0] - 0.5 * deltaz * rho_approx * float(GRAV)])
    rhoa, dz, zmet, zmetl, rmu, thcond, rhoa_wet = setup_atm(
        T, p_cgs, pl, zc, zl, GridType.I_CART,
    )

    diffus, rlhe, rlhm = setup_grow(
        T, p_cgs, rhoa, zmet,
        igash2o=cfg.igash2o, igash2so4=cfg.igash2so4,
        ngas=ngas, do_cnst_rlh=False,
    )

    # H2O state from current gc — gc is in JAX CGS = mmr × rhoa, so
    # h2o_mass [g/cm³] = gc[H2O] / zmet (since rhoa already has the
    # zmet factor folded in for non-cartesian grids; here zmet=1).
    h2o_mass_cgs = gc[0, cfg.igash2o] / zmet[0]
    pvapl_h2o_arr, _ = vaporp_h2o_murphy2005(T)
    h2o_vp_cgs = pvapl_h2o_arr[0]                  # dyne/cm² (CGS already)

    # Wet radii via I_WTPCT_H2SO4 (sulfate group's irhswell). Use
    # Fortran's actual relhum = h2o_vp / pvapl ratio, not the scenario
    # rh, since gc evolves over the run.
    relhum = h2o_mass_cgs / (h2o_vp_cgs * 18.016 / (8.314e7 * T[0]))
    grp = cfg.groups[0]
    r_dry = jnp.asarray(grp.r, dtype=DTYPE)
    rup_dry = jnp.asarray(grp.rup, dtype=DTYPE)
    rlow_dry = jnp.asarray(grp.rlow, dtype=DTYPE)
    rho_dry_per_bin = jnp.full(r_dry.shape, _FORTRAN_RHO_SULF, dtype=DTYPE)
    swell = int(SwellMethod.I_WTPCT_H2SO4)
    r_wet_1d, rho_wet = get_wetr(r_dry, rho_dry_per_bin, relhum, T[0], swell,
        h2o_mass=h2o_mass_cgs, h2o_vp=h2o_vp_cgs,
        gwtmol_h2so4=DTYPE(_GWTMOL_H2SO4))
    rup_wet_1d, _ = get_wetr(rup_dry, rho_dry_per_bin, relhum, T[0], swell,
        h2o_mass=h2o_mass_cgs, h2o_vp=h2o_vp_cgs,
        gwtmol_h2so4=DTYPE(_GWTMOL_H2SO4))
    rlow_wet_1d, _ = get_wetr(rlow_dry, rho_dry_per_bin, relhum, T[0], swell,
        h2o_mass=h2o_mass_cgs, h2o_vp=h2o_vp_cgs,
        gwtmol_h2so4=DTYPE(_GWTMOL_H2SO4))
    r_wet = r_wet_1d[None, :, None]
    rup_wet = rup_wet_1d[None, :, None]
    rlow_wet = rlow_wet_1d[None, :, None]

    rrat = jnp.ones((nbin, ngroup), dtype=DTYPE)
    eshape_arr = jnp.asarray([float(g.eshape) for g in cfg.groups], dtype=DTYPE)
    is_ice_arr = np.asarray([bool(g.is_ice) for g in cfg.groups])
    gwtmol_arr = jnp.asarray([float(g.wtmol) for g in cfg.gases], dtype=DTYPE)
    igrowgas_arr = np.asarray(
        [cfg.igash2so4 for _ in cfg.elements], dtype=np.int32)

    rmass_2d = jnp.asarray(cfg.groups[0].rmass)[:, None]
    rho_p_2d = rho_wet[:, None]
    rrat_2d = jnp.ones((nbin, ngroup), dtype=DTYPE)
    rprat_2d = jnp.ones((nbin, ngroup), dtype=DTYPE)
    vf, re_real, bpm = setup_vf_jit(
        T, rhoa, zmet, rmu, r_wet, rho_p_2d, rrat_2d, rprat_2d,
    )

    wtpct_arr = wtpct_tabaz(T, jnp.atleast_1d(h2o_mass_cgs),
                             jnp.atleast_1d(h2o_vp_cgs))

    _, akelvin, akelvini, gro, gro1, gro2, _, _ = setup_gkern(
        T, p_cgs, rhoa, zmet, rmu, thcond, diffus, rlhe, rlhm,
        re_real, r_wet, rlow_wet, rrat, eshape_arr, is_ice_arr,
        gwtmol_arr, igrowgas_arr,
        cfg.gstickl, cfg.gsticki, cfg.tstick,
        nbin, ngroup, ngas,
        igash2o=cfg.igash2o, igash2so4=cfg.igash2so4, wtpct=wtpct_arr,
    )

    is_sulfate = np.asarray([bool(g.is_sulfate) for g in cfg.groups])
    use_vw = jnp.asarray(is_sulfate[:, None] & is_sulfate[None, :])
    ckernel = setup_ckern_jit(
        T, rhoa, zmet, rmu, r_wet, rrat_2d, rprat_2d, bpm, rmass_2d,
        re_real, vf, cfg.cstick,
        use_vw=use_vw,
    )

    pratt, prat, pden1, palr = ppm_coefs
    ds_threshold_arr = jnp.zeros((ngas,), dtype=DTYPE)

    return dict(
        rhoa=rhoa, zmet=zmet, rlhe=rlhe, rlhm=rlhm, diffus=diffus,
        akelvin=akelvin, akelvini=akelvini,
        gro=gro, gro1=gro1, gro2=gro2,
        rup_wet=rup_wet, rlow_wet=rlow_wet,
        pratt=pratt, prat=prat, pden1=pden1, palr=palr,
        ds_threshold_arr=ds_threshold_arr,
        ckernel=ckernel,
        rmu=rmu, thcond=thcond,
    )


def _env_for_scenario(T_K, p_hPa, rh, h2so4_pptv, mu_nm, sigma_g, cfg,
                        ppm_coefs):
    """Build initial state + env for one scenario.

    Initial pc/gc/T/p_cgs are scenario-only; the env arrays come from
    ``_refresh_env`` so that subsequent outer steps can re-call it with
    the evolved (T, gc).
    """
    from carma.constants import RPA2CGS

    nbin = cfg.nbin
    nz = 1

    T = jnp.asarray([T_K], dtype=DTYPE)
    p_cgs = jnp.asarray([p_hPa * 100.0 * float(RPA2CGS)], dtype=DTYPE)

    # Initial gas mmrs (Fortran ensemble line 173-176)
    h2so4_mmr = h2so4_pptv * 1.0e-12 * (98.0 / 29.0)
    pvapl_Pa = jnp.exp(54.842763 - 6763.22 / T_K
                        - 4.210 * jnp.log(T_K) + 0.000367 * T_K)
    h2o_mmr = rh * float(pvapl_Pa) * 18.0 / (29.0 * p_hPa * 100.0)

    # First env build to get rhoa for the gas-density and seed-mass
    # conversions; gc is rebuilt below from the same rhoa.
    gc_seed = jnp.asarray([[0.0, 0.0]], dtype=DTYPE)
    env0 = _refresh_env(T, p_cgs, gc_seed, cfg, ppm_coefs)
    rhoa_init = env0["rhoa"]

    # gc in CGS = mmr × rhoa
    gc = jnp.asarray([[h2o_mmr * float(rhoa_init[0]),
                        h2so4_mmr * float(rhoa_init[0])]], dtype=DTYPE)

    # Initial lognormal aerosol seed. Σ mmr = 1e-18 g/g (Fortran
    # convention) → Σ pc·rmass = 1e-18 × rhoa for JAX CGS.
    r_dry = jnp.asarray(cfg.groups[0].r, dtype=DTYPE)
    pc_sulf = _init_lognormal_pc(np.asarray(r_dry), mu_nm, sigma_g,
                                   total_mass_g_per_cm3=1.0e-18 * float(rhoa_init[0]))
    pc = jnp.zeros((nz, nbin, cfg.nelem), dtype=DTYPE)
    pc = pc.at[0, :, 0].set(pc_sulf)

    # Now rebuild env with the actual gc so wetr/akelvin/etc. reflect
    # initial gas state (not zeros). Caller will _refresh_env again at
    # each outer step.
    env = _refresh_env(T, p_cgs, gc, cfg, ppm_coefs)
    env["pc"] = pc
    env["gc"] = gc
    env["t"] = T
    env["p_cgs"] = p_cgs
    return env


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
        p_cgs = env["p_cgs"]   # constant over the run (no compression)

        for _step_idx in range(nstep):
            # Phase 10.5: rebuild the env from current (T, gc) at the
            # start of each outer step. Mirrors Fortran's
            # CARMASTATE_Create + SetGas + Step pattern, where wet
            # radii / Kelvin / growth-coag kernels are recomputed each
            # outer iteration. p stays constant in this single-cell
            # test (no compression / advection), so we keep the
            # initial p_cgs.
            env = _refresh_env(t, p_cgs, gc, cfg, ppm_coefs)

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
