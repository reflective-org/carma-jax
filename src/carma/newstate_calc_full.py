"""Adaptive-substep / retry driver for ``microfast_full``.

Mirrors the retry block in Fortran ``newstate_calc.F90`` lines 95-225.

Per-step loop:

1. Initial substep count from ``nsubsteps`` heuristic (capped at
   ``minsubsteps`` floor, ``maxsubsteps`` ceiling).
2. Linear drift of gas / temperature over substeps via
   ``gc += d_gc/ntsubsteps``, ``t += d_t/ntsubsteps``.
3. Each substep calls ``microfast_full_jit``. If it returns
   ``RC_WARNING_RETRY`` (gc went negative or |dt| > dt_threshold),
   the entire substep loop restarts from the saved (pcl, gcl, told)
   state with ``ntsubsteps *= 2``.
4. Stops when either microfast succeeds for all substeps, or
   ``nretries > maxretries`` (failure mode — final state may be
   inconsistent).

This driver runs the inner substep loop in pure Python with the
JIT'd ``microfast_full_jit`` inside, so each substep dispatches
once. The retry loop is also Python — it depends on the traced rc
which can't be passed through ``lax.while_loop`` without
materialising. For Phase 10 vmap'd ensembles a future refactor
will need ``lax.while_loop`` with a worst-case substep ceiling,
but Python loops are fine for the differential bench.
"""

import jax.numpy as jnp

from carma.enums import RC_OK, RC_WARNING_RETRY
from carma.precision import DTYPE


def newstate_calc_full(
    pc_init, gc_init, t_init,
    d_gc, d_t,
    dtime_orig,
    rhoa, zmet,
    akelvin, akelvini, gro, gro1, rup_wet,
    rmass_2d, dm_2d, rmassup, r_bins, rmrat_val,
    pratt, prat, pden1, palr,
    rlhe, rlhm,
    ds_threshold_arr,
    microfast_full_jit,
    initial_ntsubsteps=1,
    minsubsteps=1, maxsubsteps=32,
    maxretries=16,
    iz=0,
    dt_threshold=DTYPE(1.0),
    scale_threshold=DTYPE(1.0),
    prescribed_ntsubsteps=None,
):
    """Adaptive-substep retry around ``microfast_full_jit``.

    Args:
        pc_init, gc_init, t_init: state at start of outer step
            (== prestep's pcl/gcl/told).
        d_gc, d_t: per-step deltas for gas / temperature drift
            (computed by prestep).
        dtime_orig: outer-step size [s].
        rhoa..palr: environmental fields, see ``microfast_full_jit``.
        rlhe, rlhm: latent-heat coefs (zeros OK if not bench-validating
            tsolve's rlheat output).
        ds_threshold_arr: gsolve convergence thresholds per gas.
        microfast_full_jit: the JIT'd closure from
            ``make_microfast_full_jit``.
        initial_ntsubsteps: starting substep count (typically from
            ``nsubsteps`` kernel; pass 1 to start at the minimum).
        minsubsteps, maxsubsteps, maxretries: substep bounds.
        iz: vertical level index.
        dt_threshold, scale_threshold: tsolve / gsolve thresholds.

    Returns:
        Tuple ``(pc, gc, t, rlheat_total, ntsubsteps_used, nretries_used)``.
    """
    # Phase 10.6 prescribed-substep mode: bypass adaptive retry and run
    # exactly `prescribed_ntsubsteps` substeps (rc ignored). Used to
    # quantify how much of the JAX↔Fortran residual is retry-boundary
    # drift vs. something else.
    if prescribed_ntsubsteps is not None:
        ntsubsteps = int(prescribed_ntsubsteps)
        pc = pc_init
        gc = gc_init
        t = t_init
        rlheat_total = jnp.float64(0.0)
        dtime_sub = dtime_orig / ntsubsteps
        fraction = jnp.float64(1.0) / ntsubsteps
        for _isubstep in range(ntsubsteps):
            gc = gc + d_gc * fraction
            t = t + d_t * fraction
            pc, gc, t, rlheat_val, _rc = microfast_full_jit(
                pc, gc, t, dtime_sub,
                rhoa, zmet,
                akelvin, akelvini, gro, gro1, rup_wet,
                rmass_2d, dm_2d, rmassup, r_bins, rmrat_val,
                pratt, prat, pden1, palr,
                rlhe, rlhm,
                ds_threshold_arr,
                dt_threshold, scale_threshold, iz,
            )
            rlheat_total = rlheat_total + rlheat_val
        return pc, gc, t, rlheat_total, ntsubsteps, 0

    ntsubsteps = max(int(initial_ntsubsteps), int(minsubsteps))
    nretries = 0

    while True:
        # Reset to saved state.
        pc = pc_init
        gc = gc_init
        t = t_init
        rlheat_total = jnp.float64(0.0)

        dtime_sub = dtime_orig / ntsubsteps
        fraction = jnp.float64(1.0) / ntsubsteps
        retried = False

        for _isubstep in range(int(ntsubsteps)):
            # Linear gas / temperature drift over substep.
            gc = gc + d_gc * fraction
            t = t + d_t * fraction

            pc, gc, t, rlheat_val, rc = microfast_full_jit(
                pc, gc, t, dtime_sub,
                rhoa, zmet,
                akelvin, akelvini, gro, gro1, rup_wet,
                rmass_2d, dm_2d, rmassup, r_bins, rmrat_val,
                pratt, prat, pden1, palr,
                rlhe, rlhm,
                ds_threshold_arr,
                dt_threshold, scale_threshold, iz,
            )
            rlheat_total = rlheat_total + rlheat_val

            # rc is traced but we can pull a Python-int via int()/jnp.asarray
            if int(rc) == RC_WARNING_RETRY:
                retried = True
                break

        if not retried:
            break  # success — converged at this ntsubsteps

        nretries += 1
        if nretries > maxretries:
            break  # give up — Fortran also gives up at this point

        ntsubsteps *= 2

    return pc, gc, t, rlheat_total, ntsubsteps, nretries
