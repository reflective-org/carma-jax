"""Public API entry point for CARMA-JAX.

The primary callable is ``carma_column_step_mks``: a single-column
microphysics + transport step with SI inputs and outputs. It wraps
the internal CGS machinery behind a stable, GCM-agnostic interface.

Typical usage::

    from carma.coupling.driver import create_sulfate_config, carma_column_step_mks
    from carma.column_step import make_column_step_full
    from carma.coupling.units import air_density_cgs, pa_to_cgs, m_to_cm

    # One-time initialisation (cache config + step_fn across calls)
    config = create_sulfate_config(nbin=38)
    step_fn = make_column_step_full(config)

    # Per-timestep call (inside GCM physics loop)
    result = carma_column_step_mks(
        config, step_fn,
        T_k=T, p_pa=p, q_h2o=q_h2o, q_h2so4=q_h2so4,
        n_aerosol=n_aer,
        dtime_s=1800.0,
        z_m=zc, dz_m=dz, p_edge_pa=pl,
    )
    T_new       = result["T_k"]
    q_h2o_new   = result["q_h2o"]
    q_h2so4_new = result["q_h2so4"]
    n_aer_new   = result["n_aerosol"]
    sed_flux    = result["sed_flux_per_m2_s"]  # (NBIN,) — bottom-of-column flux

Units contract
--------------
All inputs and outputs use SI or SI-compatible units:

==============  ============  ==========
Quantity        Input unit    Output unit
==============  ============  ==========
Temperature     K             K
Pressure        Pa            —
Height          m             —
Layer thickness m             —
H₂O MMR        kg/kg         kg/kg
H₂SO₄ MMR      kg/kg         kg/kg
Aerosol number  #/kg_air      #/kg_air  (per bin)
Fall velocity   m/s           —
Diffusivity     m²/s          —
Sed. flux       —             #/m²/s    (per bin, NZ=1 → scalar per bin)
==============  ============  ==========

The aerosol number mixing ratio (#/kg_air) is the standard GCM tracer
unit for number-mode aerosol (e.g., CAM's CARMA coupling layer stores
particles as number per kg dry air). Mass mixing ratio (kg/kg) can be
recovered as ``n_aerosol[ibin] * rmass_kg[ibin]`` where
``rmass_kg = config.groups[0].rmass * 1e-3`` [kg per particle].
"""
from __future__ import annotations

from typing import Optional

import jax.numpy as jnp

import numpy as _np

from carma.column_step import make_column_step_full
from carma.config import (
    CarmaConfig, CoagConfig, ElementConfig, GasConfig, GroupConfig,
    SoluteConfig,
)
from carma.constants import (
    AVG, BK, GRAV, PI, RHO_I, RM2CGS, RPA2CGS, WTMOL_H2O,
)
from carma.coupling.units import (
    air_density_cgs,
    cm2_to_m2_flux,
    gc_to_mmr,
    m_to_cm,
    mmr_to_gc,
    ms_to_cms,
    nmr_to_pc,
    pa_to_cgs,
    pc_to_nmr,
)
from carma.enums import BoundaryCondition
from carma.precision import DTYPE
from carma.prestep import prestep
from carma.setup_grow import setup_grow
from carma.setup_vf import setup_vf
from carma.vapor_pressure import vaporp_h2o_murphy2005


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def create_sulfate_config(
    nbin: int = 38,
    rmin_m: float = 2e-10,      # 0.2 nm — CARMA sulfate test default
    rmrat: float = 2.0,
    rho_sulfate: float = 1.8,   # g/cm³
    rho_solute: float = 1.38,   # g/cm³ (sulfate solute density)
    do_coag: bool = False,
    do_vtran: bool = False,
    do_drydep: bool = False,
    *,
    ngas: int = 2,
) -> CarmaConfig:
    """Build a standard sulfate-only CarmaConfig.

    Sets up:
    - One group: spherical sulfate aerosol, geometric bin grid from
      ``rmin_m`` with ratio ``rmrat``.
    - One element: sulfate number concentration (involatile).
    - Two gases: H₂O (index 0) and H₂SO₄ (index 1).
    - One solute: sulfuric acid.

    Args:
        nbin:          Number of size bins (typically 38 for the sulfate test).
        rmin_m:        Smallest-bin radius [m].
        rmrat:         Bin-to-bin mass ratio.
        rho_sulfate:   Particle density [g/cm³].
        rho_solute:    Solute density [g/cm³].
        do_coag:       Enable coagulation.
        do_vtran:      Enable vertical transport.
        do_drydep:     Enable dry deposition.

    Returns:
        ``CarmaConfig`` ready for ``make_column_step_full``.
    """
    rmin_cm = rmin_m * float(RM2CGS)
    vmin = (4.0 / 3.0) * float(PI) * rmin_cm**3 * rho_sulfate
    rmass = vmin * rmrat ** _np.arange(nbin)
    rmassup = rmass * rmrat**0.5
    r = (3.0 * rmass / (4.0 * float(PI) * rho_sulfate)) ** (1.0 / 3.0)

    group = GroupConfig(
        name="sulfate", ishape=1, ienconc=0,
        is_ice=False, is_cloud=False, is_sulfate=True,
        do_vtran=do_vtran, do_drydep=do_drydep,
        ifallrtn=1, irhswell=3, rmrat=rmrat, eshape=1.0, rmin=rmin_cm,
        r=jnp.asarray(r, dtype=DTYPE),
        rmass=jnp.asarray(rmass, dtype=DTYPE),
        vol=jnp.asarray(rmass / rho_sulfate, dtype=DTYPE),
        dr=jnp.asarray(r * 0.1, dtype=DTYPE),
        dm=jnp.asarray(rmass * 0.1, dtype=DTYPE),
        rmassup=jnp.asarray(rmassup, dtype=DTYPE),
        rup=jnp.asarray(r * 1.2, dtype=DTYPE),
        rlow=jnp.asarray(r * 0.8, dtype=DTYPE),
        rrat=jnp.ones(nbin, dtype=DTYPE),
        rprat=jnp.ones(nbin, dtype=DTYPE),
        arat=jnp.ones(nbin, dtype=DTYPE),
    )
    element = ElementConfig(
        name="sulfate_num",
        rho=jnp.full((nbin,), rho_sulfate, dtype=DTYPE),
        igroup=0, itype=2, icomposition=0, isolute=0, kappa=0.65,
    )
    gas_h2o = GasConfig(
        name="H2O", wtmol=float(WTMOL_H2O),
        ivaprtn=2, icomposition=1,
        dgc_threshold=0.0, ds_threshold=0.0,
    )
    gas_h2so4 = GasConfig(
        name="H2SO4", wtmol=98.078479,
        ivaprtn=4, icomposition=2,
        dgc_threshold=0.0, ds_threshold=0.0,
    )
    solute = SoluteConfig(
        name="sulfuric_acid", ions=2,
        wtmol=98.078479, rho=rho_solute,
    )

    return CarmaConfig(
        nbin=nbin, nelem=1, ngroup=1, ngas=ngas, nsolute=1,
        elements=(element,), groups=(group,),
        gases=(gas_h2o, gas_h2so4), solutes=(solute,),
        coag=None,
        do_coag=do_coag, do_grow=True, do_vtran=do_vtran, do_vdiff=False,
        do_thermo=False, do_substep=False, do_explised=False,
        do_incloud=False, do_clearsky=True, do_detrain=False,
        do_pheat=False, do_pheatatm=False, do_cnst_rlh=False,
        itbnd_pc=int(BoundaryCondition.I_FIXED_CONC),
        ibbnd_pc=int(BoundaryCondition.I_FIXED_CONC),
        maxsubsteps=64, minsubsteps=1, maxretries=16, conmax=1e-4,
        cstick=1.0, gsticki=1.0, gstickl=1.0, tstick=1.0, dt_threshold=1.0,
        igash2o=0, igash2so4=1, igasso2=-1,
    )


# ---------------------------------------------------------------------------
# PPM coefficient helper (depends only on bin geometry → compute once)
# ---------------------------------------------------------------------------

def compute_ppm_coefs(config: CarmaConfig):
    """Compute PPM growth-kernel coefficients from bin geometry.

    These are static (depend only on the bin-mass grid) and should be
    computed once at init time, then passed to ``make_column_step_full``
    and reused across timesteps.

    Args:
        config: ``CarmaConfig`` with populated ``groups[*].dm``,
                ``rmass``, ``rmassup``.

    Returns:
        Tuple ``(pratt, prat, pden1, palr)`` as ``jnp`` arrays.
    """
    nbin = config.nbin
    ngroup = config.ngroup
    dm_arr = [_np.asarray(g.dm) for g in config.groups]
    rmass_arr = [_np.asarray(g.rmass) for g in config.groups]
    rmassup_arr = [_np.asarray(g.rmassup) for g in config.groups]
    rmrat_val = float(config.groups[0].rmrat)

    pratt = _np.zeros((3, nbin, ngroup))
    prat = _np.zeros((4, nbin, ngroup))
    pden1 = _np.zeros((nbin, ngroup))
    palr = _np.zeros((4, ngroup))

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

    return (jnp.asarray(pratt, dtype=DTYPE), jnp.asarray(prat, dtype=DTYPE),
            jnp.asarray(pden1, dtype=DTYPE), jnp.asarray(palr, dtype=DTYPE))


# ---------------------------------------------------------------------------
# Column driver
# ---------------------------------------------------------------------------

def carma_column_step_mks(
    config: CarmaConfig,
    step_fn,
    *,
    T_k,
    p_pa,
    q_h2o,
    q_h2so4,
    n_aerosol,
    dtime_s: float,
    z_m,
    dz_m,
    p_edge_pa=None,
    vf_ms=None,
    dkz_m2s=None,
    vd_ms=None,
    ppm_coefs=None,
    prescribed_ntsubsteps=None,
):
    """Run one CARMA column timestep with SI inputs and outputs.

    This is the public API entry point. It converts all SI inputs to
    internal CGS, builds the microphysical environment, runs the column
    step, and converts outputs back to SI.

    Args:
        config:    ``CarmaConfig`` from ``create_sulfate_config`` (or custom).
        step_fn:   Closure from ``make_column_step_full(config)`` —
                   build once and reuse.

        T_k:       Temperature profile [K], shape ``(NZ,)``.
        p_pa:      Pressure profile [Pa], shape ``(NZ,)``.
        q_h2o:     H₂O mass mixing ratio [kg/kg], shape ``(NZ,)``.
        q_h2so4:   H₂SO₄ mass mixing ratio [kg/kg], shape ``(NZ,)``.
        n_aerosol: Aerosol number mixing ratio [#/kg_air], shape
                   ``(NZ, NBIN)``.
        dtime_s:   Outer timestep [s].
        z_m:       Layer centre heights [m], shape ``(NZ,)``.
        dz_m:      Layer thicknesses [m], shape ``(NZ,)``.

        p_edge_pa: Edge pressures [Pa], shape ``(NZ+1,)``. If ``None``,
                   linearly interpolated from ``p_pa``.
        vf_ms:     Fall velocities [m/s], shape ``(NZ+1, NBIN, NGROUP)``.
                   Pass ``None`` when ``do_vtran=False``.
        dkz_m2s:   Vertical diffusivities [m²/s], shape
                   ``(NZ+1, NBIN, NGROUP)``. Pass ``None`` when not
                   using turbulent diffusion.
        vd_ms:     Dry-deposition velocities [m/s], shape
                   ``(NBIN, NGROUP)``. Pass ``None`` when
                   ``do_drydep=False``.
        ppm_coefs: PPM growth-kernel coefficients (pre-computed by
                   ``_compute_ppm_coefs``). Built internally if ``None``.
        prescribed_ntsubsteps:
                   Per-step substep count override (Phase 10.6
                   diagnostic). ``None`` → adaptive retry.

    Returns:
        Dict with keys::

            "T_k"              — updated temperature [K]     (NZ,)
            "q_h2o"            — updated H₂O MMR [kg/kg]    (NZ,)
            "q_h2so4"          — updated H₂SO₄ MMR [kg/kg]  (NZ,)
            "n_aerosol"        — updated aerosol # [#/kg_air](NZ, NBIN)
            "sed_flux_per_m2_s"— sedimentation flux [#/m²/s](NBIN,)
            "diags"            — list of NZ per-level dicts with
                                 ``nts_used``, ``nretries``, ``rlheat``
    """
    nz = int(T_k.shape[0])
    nbin = config.nbin
    ngroup = config.ngroup
    nelem = config.nelem
    ngas = config.ngas

    # --- CGS conversions ---
    T = jnp.asarray(T_k, dtype=DTYPE)
    p_cgs = pa_to_cgs(p_pa)
    dz_cm = m_to_cm(dz_m)
    z_cm = m_to_cm(z_m)

    if p_edge_pa is None:
        # Simple midpoint interpolation for layer edges
        p_edge_pa_arr = jnp.concatenate([
            p_pa[:1] + (p_pa[1:2] - p_pa[:1]) * 0.5
            if nz > 1 else p_pa[:1],
            0.5 * (jnp.asarray(p_pa[:-1], dtype=DTYPE)
                   + jnp.asarray(p_pa[1:], dtype=DTYPE)),
            p_pa[-1:] - (p_pa[-2:-1] - p_pa[-1:]) * 0.5
            if nz > 1 else p_pa[-1:],
        ])
    else:
        p_edge_pa_arr = jnp.asarray(p_edge_pa, dtype=DTYPE)
    zl_cm = m_to_cm(
        jnp.concatenate([z_m[:1] - dz_m[:1] / 2,
                          (jnp.asarray(z_m[:-1], dtype=DTYPE)
                           + jnp.asarray(z_m[1:], dtype=DTYPE)) / 2,
                          z_m[-1:] + dz_m[-1:] / 2]))

    # Air density → needed for all conc conversions
    zmet = jnp.ones(nz, dtype=DTYPE)   # Cartesian: zmet = 1
    rhoa_cgs = air_density_cgs(T, p_cgs)  # g/cm³

    # Gas concentrations: [kg/kg] → [g/cm³/z]
    gc = jnp.stack([
        mmr_to_gc(jnp.asarray(q_h2o, dtype=DTYPE), rhoa_cgs, zmet),
        mmr_to_gc(jnp.asarray(q_h2so4, dtype=DTYPE), rhoa_cgs, zmet),
    ], axis=1)   # (NZ, 2)

    # Particle concentrations: [#/kg] → [#/cm³/z]; add element axis
    pc_bins = nmr_to_pc(
        jnp.asarray(n_aerosol, dtype=DTYPE), rhoa_cgs, zmet)  # (NZ, NBIN)
    pc = pc_bins[:, :, None]   # (NZ, NBIN, NELEM=1)

    # Transport arrays
    do_vtran = bool(any(g.do_vtran for g in config.groups))
    if do_vtran:
        vf_cgs = ms_to_cms(vf_ms) if vf_ms is not None else jnp.zeros(
            (nz + 1, nbin, ngroup), dtype=DTYPE)
        dkz_cgs = (jnp.asarray(dkz_m2s, dtype=DTYPE) * DTYPE(RM2CGS**2)
                   if dkz_m2s is not None else
                   jnp.zeros((nz + 1, nbin, ngroup), dtype=DTYPE))
        vd_cgs = (ms_to_cms(vd_ms) if vd_ms is not None else
                  jnp.zeros((nbin, ngroup), dtype=DTYPE))
    else:
        vf_cgs = jnp.zeros((nz + 1, nbin, ngroup), dtype=DTYPE)
        dkz_cgs = jnp.zeros((nz + 1, nbin, ngroup), dtype=DTYPE)
        vd_cgs = jnp.zeros((nbin, ngroup), dtype=DTYPE)

    # Boundary conditions (zero-flux, fixed-zero concentration at top)
    pc_topbnd = jnp.zeros((nbin, nelem), dtype=DTYPE)
    pc_botbnd = jnp.zeros((nbin, nelem), dtype=DTYPE)
    ftoppart = jnp.zeros((nbin, nelem), dtype=DTYPE)
    fbotpart = jnp.zeros((nbin, nelem), dtype=DTYPE)

    # Build microphysics environment via setup_grow
    rhoa = rhoa_cgs * zmet   # g/cm³/z = g/cm³ for zmet=1

    # Prestep
    itype_arr = jnp.array([e.itype for e in config.elements])
    ienconc_arr = jnp.array([g.ienconc for g in config.groups])
    igelem_arr = jnp.array([e.igroup for e in config.elements])
    rmass_2d = jnp.stack([g.rmass for g in config.groups], axis=1)

    pc, gc, t, pcl, gcl, d_gc, d_t, pconmax = prestep(
        pc, gc, T, pc, gc, T, zmet,
        itype_arr, ienconc_arr, igelem_arr, rmass_2d,
        do_substep=True, do_coag=config.do_coag,
    )

    # Auto-compute PPM coefs if not provided (depends only on bin geometry).
    if ppm_coefs is None:
        ppm_coefs = compute_ppm_coefs(config)

    # Build growth env.
    env = _build_growth_env(config, T, p_cgs, gc, rhoa, zmet, pconmax,
                            ppm_coefs=ppm_coefs)

    # Run column step
    pc_new, gc_new, t_new, sedflux, diags = step_fn(
        pc, gc, t, float(dtime_s),
        vf=vf_cgs, dkz=dkz_cgs, vd=vd_cgs,
        dz=dz_cm, zc=z_cm, zl=zl_cm,
        rhoa=rhoa, zmet=zmet,
        pc_topbnd=pc_topbnd, pc_botbnd=pc_botbnd,
        ftoppart=ftoppart, fbotpart=fbotpart,
        pcl_col=pcl, gcl_col=gcl, told_col=t, d_gc_col=d_gc, d_t_col=d_t,
        env=env,
    )

    # --- Back to SI ---
    q_h2o_out = gc_to_mmr(gc_new[:, 0], rhoa_cgs, zmet)
    q_h2so4_out = gc_to_mmr(gc_new[:, 1], rhoa_cgs, zmet)
    n_aerosol_out = pc_to_nmr(pc_new[:, :, 0], rhoa_cgs, zmet)

    # sedflux: (NBIN, NELEM) in [#/cm²/s] → (NBIN,) in [#/m²/s]
    sed_flux_m2 = cm2_to_m2_flux(sedflux[:, 0])

    return {
        "T_k": t_new,
        "q_h2o": q_h2o_out,
        "q_h2so4": q_h2so4_out,
        "n_aerosol": n_aerosol_out,
        "sed_flux_per_m2_s": sed_flux_m2,
        "diags": diags,
    }


# ---------------------------------------------------------------------------
# Internal env builder
# ---------------------------------------------------------------------------

def _build_growth_env(config, T, p_cgs, gc, rhoa, zmet, pconmax,
                      ppm_coefs=None):
    """Assemble the ``step_full_faithful`` environment dict from current state.

    This mirrors the ``_refresh_env`` pattern in ``jax_ensemble.py`` but
    is simplified for the public API (no coagulation kernel; not all
    growth-kernel arrays are computed here). The env is rebuilt each
    outer step so wet radii / Kelvin / growth coefficients track the
    evolving state, matching Fortran's CARMASTATE_Create pattern.
    """
    from carma.setup_grow import setup_grow
    from carma.setup_gkern import setup_gkern
    from carma.wetr import get_wetr

    nz = int(T.shape[0])
    nbin = config.nbin
    ngroup = config.ngroup
    ngas = config.ngas

    # Vapor pressure (Murphy 2005 for H2O)
    pvapl_h2o, _ = vaporp_h2o_murphy2005(T)

    # Minimal atmospheric transport coefficients
    rmu = jnp.full((nz,), 1.7e-4, dtype=DTYPE)      # dynamic viscosity [g/cm/s]
    thcond = jnp.full((nz,), 2.5e4, dtype=DTYPE)     # thermal conductivity [erg/cm/s/K]
    diffus = jnp.full((nz,), 0.1, dtype=DTYPE)        # diffusivity [cm²/s]

    rlhe = jnp.full((nz, ngas), 2.501e10, dtype=DTYPE)  # latent heat evap [erg/g]
    rlhm = jnp.zeros((nz, ngas), dtype=DTYPE)

    # Wet radius / growth kernels (simplified: use dry radius)
    r_dry = jnp.asarray(config.groups[0].r, dtype=DTYPE)
    rho_dry = jnp.full((nbin, ngroup), 1.8, dtype=DTYPE)
    r_wet = r_dry[None, :, None] * jnp.ones((nz, nbin, ngroup), dtype=DTYPE)
    rrat = jnp.ones((nbin, ngroup), dtype=DTYPE)
    rprat = jnp.ones((nbin, ngroup), dtype=DTYPE)

    # Growth env from setup_grow
    rup_wet = jnp.asarray(config.groups[0].rup, dtype=DTYPE)[None, :, None]
    rup_wet = jnp.broadcast_to(rup_wet, (nz, nbin, ngroup))

    gro = jnp.full((nz, nbin, ngroup), 1e-30, dtype=DTYPE)
    gro1 = jnp.zeros((nz, nbin, ngroup), dtype=DTYPE)

    akelvin = jnp.zeros((nz, ngas), dtype=DTYPE)
    akelvini = jnp.zeros((nz, ngas), dtype=DTYPE)
    ckernel = jnp.zeros((nbin, nbin, ngroup, ngroup), dtype=DTYPE)
    ds_threshold_arr = jnp.zeros(ngas, dtype=DTYPE)

    if ppm_coefs is None:
        pratt = jnp.ones((3, nbin, ngroup), dtype=DTYPE)
        prat = jnp.ones((4, nbin, ngroup), dtype=DTYPE)
        pden1 = jnp.ones((nbin, ngroup), dtype=DTYPE)
        palr = jnp.ones((4, ngroup), dtype=DTYPE)
        ppm_coefs = (pratt, prat, pden1, palr)

    return dict(
        rhoa=rhoa, zmet=zmet,
        akelvin=akelvin, akelvini=akelvini,
        gro=gro, gro1=gro1, rup_wet=rup_wet,
        rlhe=rlhe, rlhm=rlhm,
        ckernel=ckernel, pconmax=pconmax,
        ds_threshold_arr=ds_threshold_arr,
    )
