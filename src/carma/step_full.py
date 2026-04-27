"""End-to-end single-column microphysics step (``make_step_full``).

Composes the adaptive-retry growth kernel (``newstate_calc_growth_jit``)
with the sulfate nucleation + gas-exchange step
(``sulfate_step_one_level``) into one ``@jax.jit`` closure. This is
the per-column driver the Phase 10 ensemble will ``vmap`` across
1000 scenarios.

Scope
-----

- **Included**: growth (adaptive retry), sulfate nucleation,
  H2SO4 gas balance. One JIT'd closure.
- **Not included**: vertical transport, coagulation. Those stay as
  separate factories (``make_step_transport`` / ``make_step_coag``)
  so that 1-D column users can schedule them at the appropriate
  outer-loop position. Composition into a single flat factory is a
  design exercise deferred until the Phase 10 ensemble identifies
  whether it's needed for performance.
- **Not included yet**: mixed-group configs (e.g. sulfate + ice).
  Single-group sulfate is the Phase 10 headline use case. Extension
  to multi-group lands naturally when Phase 11 introduces ice
  groups.

Interface
---------

Factory takes a ``CarmaConfig``; closure accepts traced state +
environmental fields that change per timestep. All static config
(bin tables, element types, method strings, step counts) is bound
at factory time.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from carma.newstate_calc_jit import newstate_calc_growth_jit
from carma.precision import DTYPE
from carma.sulfate_step import sulfate_step_one_level


def make_step_full(
    config,
    igroup_sulfate=0,
    igas_h2so4=1,
    igas_h2o=0,
    do_grow=True,
    do_sulfate=True,
    do_homogeneous_nuc=True,
    do_heterogeneous_nuc=False,
    method="ZhaoTurco",
    minsubsteps=1,
    maxsubsteps=128,
    maxretries=10,
):
    """Build a JIT'd end-to-end single-column microphysics step.

    Args:
        config: ``CarmaConfig`` with bin/group/gas tables.
        igroup_sulfate: Index of the sulfate group (default 0).
        igas_h2so4, igas_h2o: Gas-index of H2SO4/H2O.
        do_grow: Run adaptive-retry growth step.
        do_sulfate: Run sulfate nucleation + gas exchange.
        method: Sulfate nucleation method (``"ZhaoTurco"`` default).
        minsubsteps, maxsubsteps, maxretries: Growth substep bounds.
        do_homogeneous_nuc, do_heterogeneous_nuc: Defaults match the
            Fortran sulfatetest (homogeneous only). Heterogeneous
            on at finer dt drives mass to the top bin — see
            Phase 10.4j for the dt-sensitivity diagnostic.

    Returns:
        ``step(pc, gc, t, dtime, **env)`` closure that returns
        ``(pc_new, gc_new, t_new, diag)``.

    ``env`` contains the per-timestep environmental fields needed by
    growth: ``rhoa, zmet, rlhe, rlhm, diffus, akelvin, akelvini, gro,
    gro1, gro2, rup_wet, rlow_wet, pratt, prat, pden1, palr, pvapl,
    ds_threshold_arr``.
    """
    group = config.groups[igroup_sulfate]
    nbin = int(config.nbin)
    ngas = int(config.ngas)
    nelem = int(config.nelem)
    ngroup = int(config.ngroup)

    # Bin tables for sulfate path
    r_bins = jnp.asarray(group.r, dtype=DTYPE)
    rmass = jnp.asarray(group.rmass, dtype=DTYPE)
    rmassup = jnp.asarray(group.rmassup, dtype=DTYPE)
    rmass_2d = jnp.asarray(
        jnp.stack([jnp.asarray(g.rmass, dtype=DTYPE) for g in config.groups],
                  axis=1)
    )   # (nbin, ngroup)
    dm_2d = jnp.asarray(
        jnp.stack([jnp.asarray(g.dm, dtype=DTYPE) for g in config.groups],
                  axis=1)
    )
    rmrat_val = float(group.rmrat)

    # Diffmass for sulfate (single group)
    col = rmass[:, None]
    diffmass_sulf = (col[:, :, None, None] - col[None, None, :, :]).astype(DTYPE)

    # inuc2bin: sulfate-onto-self bin i → i+1, top has no target
    i2_map = jnp.arange(nbin, dtype=jnp.int32) + 1
    i2_map = jnp.where(i2_map < nbin, i2_map,
                        jnp.asarray(-1, dtype=jnp.int32))
    inuc2bin = i2_map.reshape(nbin, 1, 1)

    # Static element / group descriptors for microfast.
    # igrowgas_arr maps each ELEMENT to its growth gas. For the
    # sulfate ensemble, the sulfate element grows by H2SO4 vapor —
    # which is gas index ``igas_h2so4`` (typically 1), NOT 0 (H2O).
    # The earlier hard-coded `0` was a bug: growth step was crediting
    # H2O gas with sulfate evaporation, leaving H2SO4 untouched and
    # silently destroying mass.
    is_ice_arr = tuple(bool(g.is_ice) for g in config.groups)
    igrowgas_arr = tuple(
        igas_h2so4 if (do_grow and ngas > 0) else -1
        for _ in config.elements
    )
    ienconc_arr = tuple(int(g.ienconc) for g in config.groups)
    igroup_arr = tuple(int(e.igroup) for e in config.elements)
    gwtmol_arr = tuple(float(g.wtmol) for g in config.gases) or (18.0,)
    # ivaprtn_arr drives per-gas saturation vapor pressure dispatch
    # in microfast_growth (Murphy for H2O, Ayers/Kulmala for H2SO4).
    # Without this, growth treats every gas slot as H2O — see
    # newstate_calc.py::_build_pvapl_pvapi.
    ivaprtn_arr = tuple(int(g.ivaprtn) for g in config.gases) or (2,)

    @jax.jit
    def step(
        pc, gc, t, dtime,
        rhoa, zmet, rlhe, rlhm, diffus,
        akelvin, akelvini, gro, gro1, gro2,
        rup_wet, rlow_wet,
        pratt, prat, pden1, palr,
        pvapl,
        ds_threshold_arr,
        iz=0, scale_threshold=1.0,
    ):
        """Advance single-column state by ``dtime``.

        Faithful microfast pipeline: supersat → sulfnuc (fills rhompe/
        rnuclg) → growevapl → growp/upgxfer/psolve per (ielem, ibin)
        with all rates time-implicit-coupled → evapp/downgxfer/
        downgevapply → gsolve → tsolve → convergence check.

        This mirrors Fortran's microfast.F90 — sulfnuc rates are inside
        the same time-implicit psolve update as growth, not a separate
        post-step. The earlier 2-step (microfast then sulfate_step)
        composition broke Fortran parity at fine dt.

        Diagnostics: growth_substeps and rlheat.
        """
        diag = {}

        if do_grow:
            pc, gc, t, rlheat, nts_used = newstate_calc_growth_jit(
                pc, gc, t, iz, dtime,
                rhoa, zmet, rlhe, rlhm, diffus,
                akelvin, akelvini, gro, gro1, gro2,
                rup_wet, rmass_2d, dm_2d, rlow_wet,
                pratt, prat, pden1, palr,
                is_ice_arr, igrowgas_arr, ienconc_arr,
                igroup_arr, gwtmol_arr,
                nbin, ngroup, ngas, nelem,
                minsubsteps=minsubsteps, maxsubsteps=maxsubsteps,
                maxretries=maxretries,
                ds_threshold_arr=ds_threshold_arr,
                scale_threshold=DTYPE(scale_threshold),
                ivaprtn_arr=ivaprtn_arr, igas_h2o=igas_h2o,
                do_sulfnuc=do_sulfate,
                igroup_sulfate=igroup_sulfate, igas_h2so4=igas_h2so4,
                sulf_r_bins=r_bins,
                sulf_rmassup=rmassup,
                sulf_rmrat=rmrat_val,
                nuc_method=method,
                do_homogeneous_nuc=do_homogeneous_nuc,
                do_heterogeneous_nuc=do_heterogeneous_nuc,
            )
            diag["growth_substeps"] = nts_used
            diag["rlheat"] = rlheat
            if do_sulfate:
                # Diag-only key so callers that branch on "sulfate"
                # in diag don't break. Actual rates are now inside
                # microfast_growth's psolve update.
                diag["sulfate"] = {"integrated": jnp.asarray(True)}

        return pc, gc, t, diag

    return step
