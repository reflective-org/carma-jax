"""State-update dispatcher for clear-sky / in-cloud microphysics paths.

Ports ``newstate.F90``. This is the orchestration layer that sits
above ``newstate_calc`` (the substepped microfast driver). Its job
is to decide **how** the microphysics is called given the cloud
configuration:

- ``do_incloud=False`` (the gridbox-average path): one call through
  ``newstate_calc_growth`` with ``scale_threshold = 1.0``. This is the
  path used by every single-column test and every Phase 7 sulfate
  scenario. **Fully ported here.**
- ``do_incloud=True`` (cloudy + optional clear-sky path): state is
  split into cloudy / clear-sky halves, each run through
  ``newstate_calc_growth`` separately, then blended back with
  ``cldfrc`` weights. Uses cloud-fraction-scaled particle
  concentrations for "cloud" groups. **Stubbed** here; full wiring
  lands in Phase 11 alongside cloud / ice microphysics. Calling
  with ``do_incloud=True`` raises ``NotImplementedError``.

``newstate`` is the "outer" driver; ``newstate_calc_growth`` (from
``newstate_calc.py``) remains the inner one that handles the retry
loop. Phase 9.3 replaces its Python ``while`` with a
``lax.while_loop`` so the whole thing becomes JIT-compilable
end-to-end.

Ported from: ``newstate.F90``.
"""

import jax.numpy as jnp

from carma.newstate_calc import newstate_calc_growth
from carma.precision import DTYPE


def newstate(
    pc, gc, t, iz, dtime,
    rhoa, zmet, rlhe, rlhm, diffus,
    akelvin, akelvini, gro, gro1, gro2,
    rup_wet, rmass_2d, dm_2d, rlow_wet,
    pratt, prat, pden1, palr,
    is_ice_arr, igrowgas_arr, ienconc_arr,
    igroup_arr, gwtmol_arr,
    nbin, ngroup, ngas, nelem,
    minsubsteps=1, maxsubsteps=128, maxretries=10,
    dt_threshold=0.0, ds_threshold_arr=None, scale_threshold=1.0,
    do_incloud=False, do_clearsky=True,
    cldfrc=None, is_grp_cloud=None,
):
    """Dispatch microphysics update for one column level.

    Args:
        All the state + config args threaded through to
        ``newstate_calc_growth``. Plus:

        do_incloud: If True, use the in-cloud/clear-sky blended path
            (stubbed — raises NotImplementedError).
        do_clearsky: Only meaningful when ``do_incloud=True``.
        cldfrc: Cloud fraction (NZ,) — only used in the in-cloud
            path. Optional for clear-sky.
        is_grp_cloud: (ngroup,) bool — marks groups whose mass is
            entirely in-cloud. Only used in the in-cloud path.

    Returns:
        ``(pc, gc, t, rlheat_total, ntsubsteps_used)`` — same shape
        as ``newstate_calc_growth``.
    """
    if do_incloud:
        raise NotImplementedError(
            "newstate in-cloud / clear-sky blending is deferred to "
            "Phase 11 (cloud/ice microphysics). The Phase 7 sulfate "
            "and Phase 10 ensemble tests all use do_incloud=False. "
            "Set do_incloud=False to use the gridbox-average path."
        )

    # Gridbox-average path. Fortran sets scale_threshold=1.0 here;
    # ds_threshold_arr defaults to zero if not supplied.
    if ds_threshold_arr is None:
        ds_threshold_arr = jnp.zeros(max(ngas, 1), dtype=DTYPE)

    # newstate_calc_growth uses positional `dtime_orig`; we name the
    # dispatcher's argument `dtime` to match the Fortran naming and
    # the other step factories.
    return newstate_calc_growth(
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
        dt_threshold=dt_threshold, ds_threshold_arr=ds_threshold_arr,
        scale_threshold=DTYPE(scale_threshold),
    )
