"""Faithful single-column microphysics step (sulfate scope).

Mirrors Fortran's ``CARMASTATE_Step → newstate → newstate_calc → microfast``
chain exactly, using the validated Phase 7-9 kernels:

    prestep → microslow → adaptive substep retry over microfast_full

Replaces the legacy operator-split ``step_full`` (which separately
ran growth and sulfate_step in series, producing physically wrong
sulfate output — see Phase 9.1 gap analysis).

Bench result against Fortran microfast (1000 scenarios):
- Single substep (Phase 9.1 ``test_microfast_full_jit_matches_fortran``):
  pc/gc 1000/1000 at rtol=1e-10, near machine ε.
- Multi-substep (Phase 9.2 ``test_newstate_calc_full_substep_evolution_matches_fortran``):
  474/474 (subset zsubsteps≤1024) at rtol=1e-10, near machine ε.

Scope
-----

- Sulfate test single-column path. NGROUP=1, NELEM=1, NGAS=2 (H2O, H2SO4).
- Adaptive substep retry around ``microfast_full`` (the heart of the
  faithful port).
- Coag (microslow) is run once per outer step before the substep loop,
  matching Fortran's flow.
- No vertical transport (handled by the outer caller in 1-D / 3-D
  frameworks).

Out-of-scope (deferred to later phases or future refactors)
-----------------------------------------------------------

- Full ``@jax.jit`` of the outer step. The retry loop's Python ``while``
  on a traced rc value is fundamentally incompatible with ``@jax.jit``;
  a future refactor can use ``lax.while_loop`` with a worst-case substep
  ceiling to fully JIT the outer step. For now the inner
  ``microfast_full_jit`` is JIT'd and dispatched once per substep.
- Multi-element groups (Phase 11 — ice / mixed groups).
"""

from carma.microfast_full import make_microfast_full_jit
from carma.microslow import make_microslow
from carma.newstate_calc_full import (
    make_substep_loop_jit,
    newstate_calc_full,
)
from carma.precision import DTYPE
import jax.numpy as jnp


def make_step_full_faithful(
    config,
    igroup_sulfate=0,
    igas_h2o=0,
    igas_h2so4=1,
    method="ZhaoTurco",
    minsubsteps=1,
    maxsubsteps=32,
    maxretries=16,
    do_homogeneous=True,
    do_heterogeneous=False,
    do_pheatatm=False,
    ppm_coefs=None,
):
    """Build a single-column faithful microphysics step closure.

    Args:
        config: ``CarmaConfig`` with bin/group/gas tables.
        igroup_sulfate: index of the sulfate group (default 0).
        igas_h2o, igas_h2so4: gas indices.
        method: sulfnuc method (``"ZhaoTurco"`` or ``"Vehkamaki"``).
        minsubsteps, maxsubsteps, maxretries: substep bounds.
        do_homogeneous, do_heterogeneous: sulfnuc gates.
        do_pheatatm: enable particle heating in tsolve.
        ppm_coefs: optional ``(pratt, prat, pden1, palr)`` tuple. If
            provided, these PPM coefficients are baked into the closure.
            If ``None``, falls back to ``getattr(config, ...)``.

    Returns:
        ``step(pc, gc, t, dtime, **env, **state) -> (pc_new, gc_new, t_new,
        diag)`` closure where:

        - ``env``: rhoa, zmet, akelvin, akelvini, gro, gro1, rup_wet,
          rlhe, rlhm, ckernel, pconmax, ds_threshold_arr.
        - ``state``: pcl, gcl, told, d_gc, d_t — the prestep snapshots.
        - ``diag`` reports the converged ``ntsubsteps`` and ``nretries``
          per call.
    """
    nbin = int(config.nbin)
    ngas = int(config.ngas)
    nelem = int(config.nelem)
    ngroup = int(config.ngroup)

    group = config.groups[igroup_sulfate]
    rmass_2d = jnp.stack(
        [jnp.asarray(g.rmass, dtype=DTYPE) for g in config.groups], axis=1,
    )
    dm_2d = jnp.stack(
        [jnp.asarray(g.dm, dtype=DTYPE) for g in config.groups], axis=1,
    )
    rmassup = jnp.asarray(group.rmassup, dtype=DTYPE)
    r_bins = jnp.asarray(group.r, dtype=DTYPE)
    rmrat_val = float(group.rmrat)

    if ppm_coefs is not None:
        pratt, prat, pden1, palr = (jnp.asarray(x, dtype=DTYPE) for x in ppm_coefs)
    else:
        pratt = jnp.asarray(getattr(config, "pratt"), dtype=DTYPE)
        prat = jnp.asarray(getattr(config, "prat"), dtype=DTYPE)
        pden1 = jnp.asarray(getattr(config, "pden1"), dtype=DTYPE)
        palr = jnp.asarray(getattr(config, "palr"), dtype=DTYPE)

    # JIT'd microfast (faithful single-substep)
    mf_jit = make_microfast_full_jit(
        nbin=nbin, ngroup=ngroup, nelem=nelem, ngas=ngas,
        igas_h2o=igas_h2o, igas_h2so4=igas_h2so4,
        igroup_sulf=igroup_sulfate,
        ielem_sulf=int(group.ienconc),
        gwtmol_h2o=float(config.gases[igas_h2o].wtmol) if igas_h2o < len(config.gases) else 18.016,
        gwtmol_h2so4=float(config.gases[igas_h2so4].wtmol) if igas_h2so4 < len(config.gases) else 98.078479,
        method=method,
        do_homogeneous=do_homogeneous,
        do_heterogeneous=do_heterogeneous,
        do_pheatatm=do_pheatatm,
    )

    # Build the JIT'd substep loop once per config — it bakes in mf_jit
    # via closure so that thousands of substeps inside a single outer
    # step run as a single lax.while_loop call rather than per-substep
    # Python dispatches.
    substep_loop_jit = make_substep_loop_jit(mf_jit)

    # JIT'd microslow (coag)
    if config.do_coag:
        from carma.enums import ElementType
        microslow_jit = make_microslow(
            nbin=nbin, nelem=nelem, ngroup=ngroup,
            elem_igroup=jnp.array([e.igroup for e in config.elements]),
            icoag=config.coag.icoag, volx=config.coag.volx,
            icoagelem=config.coag.icoagelem,
            npairu=config.coag.npairu, npairl=config.coag.npairl,
            iup=config.coag.iup, jup=config.coag.jup,
            igup=config.coag.igup, jgup=config.coag.jgup,
            ilow=config.coag.ilow, jlow=config.coag.jlow,
            iglow=config.coag.iglow, jglow=config.coag.jglow,
            pkernel=config.coag.pkernel,
            ienconc_arr=jnp.array([g.ienconc for g in config.groups]),
            elem_itypes=jnp.array([e.itype for e in config.elements]),
        )
    else:
        microslow_jit = None

    def step(
        pc, gc, t, dtime,
        # Environmental fields
        rhoa, zmet, akelvin, akelvini, gro, gro1, rup_wet,
        rlhe, rlhm, ckernel, pconmax,
        ds_threshold_arr,
        # Prestep snapshots (saved state)
        pcl, gcl, told, d_gc, d_t,
        iz=0,
        dt_threshold=DTYPE(1.0),
        scale_threshold=DTYPE(1.0),
        prescribed_ntsubsteps=None,
        initial_ntsubsteps=1,
    ):
        """Advance single-column state faithfully.

        Pipeline:
          1. (microslow if do_coag) — applied to ``pc`` before substep loop.
          2. Adaptive substep retry over microfast_full, starting from
             ``(pcl, gcl, told)`` saved state, drifting gc/t linearly via
             ``d_gc/d_t``.

        Returns ``(pc, gc, t, diag)`` where ``diag = {"nts_used", "nretries"}``.
        """
        diag = {}

        # Microslow (coag) — runs once per outer step. Mirrors Fortran
        # newstate_calc.F90:65,93 which saves `pcl = pc` AFTER microslow,
        # so the substep retry restarts from the post-coag state, not
        # the prestep snapshot. Without this re-snapshot, microslow's
        # update is silently overwritten when newstate_calc_full uses
        # `pcl` as the retry restart — coag becomes a no-op.
        if microslow_jit is not None:
            pc = microslow_jit(pc, pcl, ckernel, pconmax, zmet, dtime)
            pcl = pc

        # Adaptive substep retry — or, if prescribed_ntsubsteps is given,
        # bypass adaptive retry and run exactly that many substeps
        # (Phase 10.6 prescribed-substep diagnostic).
        pc, gc, t, rlheat, nts_used, nret_used = newstate_calc_full(
            pcl, gcl, told,    # saved state for retry — start of outer step
            d_gc, d_t,
            dtime,
            rhoa, zmet, akelvin, akelvini, gro, gro1, rup_wet,
            rmass_2d, dm_2d, rmassup, r_bins, rmrat_val,
            pratt, prat, pden1, palr,
            rlhe, rlhm,
            ds_threshold_arr,
            microfast_full_jit=mf_jit,
            substep_loop_jit=substep_loop_jit,
            initial_ntsubsteps=int(initial_ntsubsteps),
            minsubsteps=minsubsteps,
            maxsubsteps=maxsubsteps,
            maxretries=maxretries,
            iz=iz,
            dt_threshold=dt_threshold,
            scale_threshold=scale_threshold,
            prescribed_ntsubsteps=prescribed_ntsubsteps,
        )
        diag["nts_used"] = nts_used
        diag["nretries"] = nret_used
        diag["rlheat"] = rlheat

        return pc, gc, t, diag

    return step
