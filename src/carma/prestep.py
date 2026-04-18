"""Pre-timestep processing for CARMA-JAX.

Saves previous-step state, computes gas/temperature increments,
applies particle-concentration floors, and computes per-level max
concentrations.

Ported from: prestep.F90
"""

from functools import partial

import jax
import jax.numpy as jnp

from carma.precision import DTYPE
from carma.utils.smallconc import maxconc, smallconc


@partial(jax.jit, static_argnames=("do_substep", "do_coag"))
def prestep(pc, gc, t, pcl, gcl, told, zmet,
            itype_arr, ienconc_arr, igelem_arr, rmass_2d,
            do_substep, do_coag):
    """Pre-timestep setup.

    Matches Fortran prestep.F90:
      1. If substepping is on and NGAS>0: compute d_gc = gc - gcl.
         Where d_gc < 0, don't substep this gas: d_gc=0, gcl=gc (latest).
         Where d_gc >= 0, rewind gc to gcl (start from old state, step forward).
      2. If substepping is on: compute d_t = t - told, rewind t = told.
      3. Apply smallconc to all bins/elements.
      4. If substepping or coag: save pcl = pc.
      5. Compute pconmax per (iz, group).

    Args:
        pc, gc, t: Current state arrays.
        pcl, gcl, told: Saved state from previous step (for substepping).
        zmet: Vertical metric (NZ,).
        itype_arr, ienconc_arr, igelem_arr, rmass_2d: Element metadata.
        do_substep, do_coag: Static config flags.

    Returns:
        Dict-like tuple of updated state arrays + substep increments:
        (pc, gc, t, pcl, gcl, d_gc, d_t, pconmax).
    """
    # Gas substep increment
    if do_substep and gc.size > 0:
        d_gc_raw = gc - gcl                  # (NZ, NGAS)
        neg = d_gc_raw < DTYPE(0.0)
        d_gc = jnp.where(neg, DTYPE(0.0), d_gc_raw)
        gcl = jnp.where(neg, gc, gcl)        # update baseline for "latest" flows
        gc = jnp.where(neg, gc, gcl)         # rewind to old state where positive
    else:
        d_gc = jnp.zeros_like(gc)

    # Temperature substep increment
    if do_substep:
        d_t = t - told
        t = told
    else:
        d_t = jnp.zeros_like(t)

    # Floor particle concentrations
    pc = smallconc(pc, itype_arr, ienconc_arr, igelem_arr, rmass_2d)

    # Save pcl for coagulation / substepping retry
    if do_substep or do_coag:
        pcl = pc

    # Per-level max concentrations
    pconmax = maxconc(pc, ienconc_arr, zmet)

    return pc, gc, t, pcl, gcl, d_gc, d_t, pconmax
