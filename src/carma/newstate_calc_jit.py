"""JIT-clean adaptive retry for growth microphysics.

Ports the ``newstate_calc.F90`` retry logic into nested
``jax.lax.while_loop`` + ``jax.lax.while_loop``:

- Outer loop: one iteration per retry attempt. On failure, doubles
  ``ntsubsteps`` (clipped to ``maxsubsteps``). Terminates when
  either (a) the inner loop returns ``rc = RC_OK``, or (b) the
  retry counter exceeds ``maxretries``.
- Inner loop: runs up to ``ntsubsteps`` calls of
  ``microfast_growth``, accumulating ``rlheat``. Exits early when
  ``microfast_growth`` returns ``RC_WARNING_RETRY``.

Both loops use traced bounds, so the whole wrapper compiles under
``jax.jit`` — the pre-existing ``newstate_calc.newstate_calc_growth``
retained a Python ``while`` that broke JIT. Phase 9.4's
``make_step_full`` factory uses this variant; the Python-while
version remains available for debugging.

Ported from: retry block of ``newstate_calc.F90``.
"""

import jax
import jax.numpy as jnp

from carma.enums import RC_OK, RC_WARNING_RETRY
from carma.newstate_calc import microfast_growth
from carma.precision import DTYPE


def newstate_calc_growth_jit(
    pc, gc, t, iz, dtime_orig,
    rhoa, zmet, rlhe, rlhm, diffus,
    akelvin, akelvini, gro, gro1, gro2,
    rup_wet, rmass_2d, dm_2d, rlow_wet,
    pratt, prat, pden1, palr,
    is_ice_arr, igrowgas_arr, ienconc_arr,
    igroup_arr, gwtmol_arr,
    nbin, ngroup, ngas, nelem,
    minsubsteps=1, maxsubsteps=128, maxretries=10,
    dt_threshold=0.0, ds_threshold_arr=None, scale_threshold=1.0,
    itype_arr=None,
):
    """JIT-clean adaptive-retry substepped growth.

    Starts with ``minsubsteps`` substeps. If ``microfast_growth``
    returns ``RC_WARNING_RETRY`` inside any substep, doubles the
    count and retries from saved state. Gives up after
    ``maxretries`` failures.

    Returns:
        ``(pc, gc, t, rlheat_total, ntsubsteps_final)`` where
        ``ntsubsteps_final`` is the count that actually succeeded
        (or the last attempted count if ``maxretries`` exhausted).
    """
    if ds_threshold_arr is None:
        ds_threshold_arr = jnp.zeros(max(ngas, 1), dtype=DTYPE)

    pc_saved = pc
    gc_saved = gc
    t_saved = t

    dtime_orig = DTYPE(dtime_orig)
    scale_threshold = DTYPE(scale_threshold)
    dt_threshold = DTYPE(dt_threshold)
    maxsubsteps_i = jnp.asarray(maxsubsteps, dtype=jnp.int32)
    max_retries_i = jnp.asarray(maxretries, dtype=jnp.int32)

    def _do_substep(pc_c, gc_c, t_c, rlh, dtime):
        """One microfast call. Returns new state + rc."""
        pc_n, gc_n, t_n, rlh_val, rc_n = microfast_growth(
            pc_c, gc_c, t_c, iz, dtime,
            rhoa, zmet, rlhe, rlhm, diffus,
            akelvin, akelvini, gro, gro1, gro2,
            rup_wet, rmass_2d, dm_2d, rlow_wet,
            pratt, prat, pden1, palr,
            is_ice_arr, igrowgas_arr, ienconc_arr,
            igroup_arr, gwtmol_arr,
            nbin, ngroup, ngas, nelem,
            dt_threshold, ds_threshold_arr, scale_threshold,
            itype_arr=itype_arr,
        )
        return pc_n, gc_n, t_n, rlh + rlh_val, rc_n

    def _inner_loop(nts):
        """Run up to ``nts`` substeps of microfast. Early exit on
        ``RC_WARNING_RETRY``. Returns final (pc, gc, t, rlh, rc)."""
        dtime = dtime_orig / nts.astype(DTYPE)

        def cond(state):
            _, _, _, _, i, rc_i = state
            return (i < nts) & (rc_i != RC_WARNING_RETRY)

        def body(state):
            pc_c, gc_c, t_c, rlh, i, _rc = state
            pc_n, gc_n, t_n, rlh_n, rc_n = _do_substep(
                pc_c, gc_c, t_c, rlh, dtime
            )
            return (pc_n, gc_n, t_n, rlh_n, i + 1, rc_n)

        init = (pc_saved, gc_saved, t_saved,
                DTYPE(0.0),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(RC_OK, dtype=jnp.int64))
        return jax.lax.while_loop(cond, body, init)

    # --- Outer retry loop ---
    # State: (pc, gc, t, rlh_total, nts_used, rc, nretries)
    # nts_used is the count attempted in the current iteration.
    def retry_cond(state):
        _, _, _, _, _, rc, nretries = state
        return (rc == RC_WARNING_RETRY) & (nretries < max_retries_i + 1)

    def retry_body(state):
        _, _, _, _, nts, _, nretries = state
        pc_f, gc_f, t_f, rlh_f, _, rc_f = _inner_loop(nts)

        # Decide next-iteration ntsubsteps.
        # If failed, double (capped); if succeeded, loop exits via cond.
        nts_next = jnp.where(
            rc_f == RC_WARNING_RETRY,
            jnp.minimum(nts * 2, maxsubsteps_i),
            nts,
        )
        # Count only failed attempts as retries.
        nretries_next = jnp.where(
            rc_f == RC_WARNING_RETRY,
            nretries + 1,
            nretries,
        )
        return (pc_f, gc_f, t_f, rlh_f, nts_next, rc_f, nretries_next)

    # Initial state: sentinel RC_WARNING_RETRY forces first attempt.
    # rc type must match microfast_growth's return (int64, from jnp.where).
    init_state = (
        pc_saved, gc_saved, t_saved,
        DTYPE(0.0),
        jnp.asarray(minsubsteps, dtype=jnp.int32),
        jnp.asarray(RC_WARNING_RETRY, dtype=jnp.int64),
        jnp.asarray(0, dtype=jnp.int32),
    )
    final = jax.lax.while_loop(retry_cond, retry_body, init_state)
    pc_f, gc_f, t_f, rlh_f, nts_f, _, _ = final
    return pc_f, gc_f, t_f, rlh_f, nts_f
