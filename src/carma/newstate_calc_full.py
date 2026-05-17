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

Two implementations of the substep loop coexist:

- ``make_substep_loop_jit(microfast_full_jit)`` returns a fully
  JIT'd ``lax.while_loop`` version. For hard scenarios (thousands
  of substeps per outer step) this collapses the substep loop into
  a single JAX call, eliminating per-substep Python dispatch.
  *Preferred path*: callers should build this once per config
  (cheap — happens at ``make_step_full_faithful`` time) and pass
  it via ``substep_loop_jit=...``.

- The legacy Python loop with chunked rc-sync (every 32 substeps).
  Still used when no ``substep_loop_jit`` is supplied. Slower for
  hard scenarios but easier to debug; kept for backwards-compat
  with older callers.
"""

import jax
import jax.lax as lax
import jax.numpy as jnp

from carma.enums import RC_OK, RC_WARNING_RETRY
from carma.precision import DTYPE


def make_substep_loop_jit(microfast_full_jit):
    """Build a ``@jax.jit``-wrapped substep loop bound to ``microfast_full_jit``.

    The returned callable runs up to ``ntsubsteps`` iterations of
    ``microfast_full_jit`` inside a single ``lax.while_loop`` so JAX can
    trace and compile the entire substep loop once. Early-breaks on the
    first substep that returns ``RC_WARNING_RETRY``.

    Build this **once per CarmaConfig** (typically inside
    ``make_step_full_faithful``) — the JIT cache is then reused across
    all outer steps and scenarios that share the same array shapes.
    """
    @jax.jit
    def _substep_loop_jit(
        pc_init, gc_init, t_init,
        ntsubsteps,                   # dynamic int32 scalar
        dtime_orig, d_gc, d_t,
        rhoa, zmet,
        akelvin, akelvini, gro, gro1, rup_wet,
        rmass_2d, dm_2d, rmassup, r_bins, rmrat_val,
        pratt, prat, pden1, palr,
        rlhe, rlhm, ds_threshold_arr,
        dt_threshold, scale_threshold, iz,
    ):
        nts_f = ntsubsteps.astype(DTYPE)
        dtime_sub = dtime_orig / nts_f
        fraction = DTYPE(1.0) / nts_f

        def cond_fn(carry):
            i, _pc, _gc, _t, _rlheat, failed = carry
            return jnp.logical_and(i < ntsubsteps, jnp.logical_not(failed))

        def body_fn(carry):
            i, pc, gc, t, rlheat, _failed = carry
            gc_new = gc + d_gc * fraction
            t_new = t + d_t * fraction
            pc_out, gc_out, t_out, rlheat_val, rc = microfast_full_jit(
                pc, gc_new, t_new, dtime_sub,
                rhoa, zmet,
                akelvin, akelvini, gro, gro1, rup_wet,
                rmass_2d, dm_2d, rmassup, r_bins, rmrat_val,
                pratt, prat, pden1, palr,
                rlhe, rlhm,
                ds_threshold_arr,
                dt_threshold, scale_threshold, iz,
            )
            return (i + jnp.int32(1),
                    pc_out, gc_out, t_out,
                    rlheat + rlheat_val,
                    rc == RC_WARNING_RETRY)

        init = (jnp.int32(0),
                pc_init, gc_init, t_init,
                jnp.float64(0.0),
                jnp.bool_(False))
        final = lax.while_loop(cond_fn, body_fn, init)
        i_final, pc, gc, t, rlheat, failed = final
        return pc, gc, t, rlheat, i_final, failed

    return _substep_loop_jit


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
    substep_loop_jit=None,
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

        # Fast path: use the JIT'd substep loop if available.
        if substep_loop_jit is not None:
            pc, gc, t, rlheat_total, _i_final, _failed = substep_loop_jit(
                pc_init, gc_init, t_init,
                jnp.int32(ntsubsteps),
                dtime_orig, d_gc, d_t,
                rhoa, zmet,
                akelvin, akelvini, gro, gro1, rup_wet,
                rmass_2d, dm_2d, rmassup, r_bins, rmrat_val,
                pratt, prat, pden1, palr,
                rlhe, rlhm,
                ds_threshold_arr,
                dt_threshold, scale_threshold, iz,
            )
            return pc, gc, t, rlheat_total, ntsubsteps, 0

        # Fallback: Python loop (slow, retained for backwards compat).
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

    # ----- Fast path: pre-built JIT'd substep loop -----
    if substep_loop_jit is not None:
        while True:
            pc, gc, t, rlheat_total, _i_final, failed = substep_loop_jit(
                pc_init, gc_init, t_init,
                jnp.int32(ntsubsteps),
                dtime_orig, d_gc, d_t,
                rhoa, zmet,
                akelvin, akelvini, gro, gro1, rup_wet,
                rmass_2d, dm_2d, rmassup, r_bins, rmrat_val,
                pratt, prat, pden1, palr,
                rlhe, rlhm,
                ds_threshold_arr,
                dt_threshold, scale_threshold, iz,
            )
            # One Python sync per retry attempt (cheap — max ~16 retries).
            if not bool(failed):
                break
            nretries += 1
            if nretries > maxretries:
                break
            ntsubsteps *= 2
        return pc, gc, t, rlheat_total, ntsubsteps, nretries

    # ----- Legacy path: Python loop with chunked rc-sync every 32 substeps -----
    CHECK_EVERY = 32

    while True:
        # Reset to saved state.
        pc = pc_init
        gc = gc_init
        t = t_init
        rlheat_total = jnp.float64(0.0)

        dtime_sub = dtime_orig / ntsubsteps
        fraction = jnp.float64(1.0) / ntsubsteps
        retried = False
        any_failed = jnp.bool_(False)

        for _isubstep in range(int(ntsubsteps)):
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
            any_failed = any_failed | (rc == RC_WARNING_RETRY)
            # Sync periodically — JAX pipelines CHECK_EVERY ops in between.
            if (_isubstep + 1) % CHECK_EVERY == 0:
                if bool(any_failed):
                    retried = True
                    break

        # Final sync at loop end for the last partial chunk.
        if not retried and bool(any_failed):
            retried = True

        if not retried:
            break  # success — converged at this ntsubsteps

        nretries += 1
        if nretries > maxretries:
            break  # give up — Fortran also gives up at this point

        ntsubsteps *= 2

    return pc, gc, t, rlheat_total, ntsubsteps, nretries
