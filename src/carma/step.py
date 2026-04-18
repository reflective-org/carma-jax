"""Top-level timestep driver for CARMA-JAX.

Composes the per-step physics: prestep → vertical transport →
[microslow (coag)] → [microfast (growth/nucleation)].

This is the MVP driver; microslow and microfast orchestration are
handled through their existing make_*/driver functions and must be
plumbed in by callers via the step_microphysics argument. The minimal
path (transport only) is fully implemented here.

Ported from: step.F90, newstate.F90
"""

from functools import partial
from typing import Callable, Optional

import jax
import jax.numpy as jnp

from carma.enums import BoundaryCondition, GridType
from carma.precision import DTYPE
from carma.prestep import prestep
from carma.transport.vertical import vertical


def step_transport(
    pc, gc, t, pcl, gcl, told, zmet,
    itype_arr, ienconc_arr, igelem_arr, rmass_2d,
    vf, dkz, vd, dz, zc, zl, rhoa, dtime,
    pc_topbnd, pc_botbnd, ftoppart, fbotpart,
    igroup_arr_tr, grp_do_vtran, grp_do_drydep,
    itbnd_pc, ibbnd_pc, igridv, nbin, nelem, ngroup,
    do_substep, do_coag,
):
    """Minimal step: prestep → vertical. No microphysics.

    Used for transport-only validation (falltest, vdiftest, drydeptest).

    All atmospheric and particle fields must be in CGS. Boundary
    condition codes are ints matching BoundaryCondition.

    Returns:
        (pc, gc, t, pcl, gcl, pconmax, sedflux).
    """
    pc, gc, t, pcl, gcl, d_gc, d_t, pconmax = prestep(
        pc, gc, t, pcl, gcl, told, zmet,
        itype_arr, ienconc_arr, igelem_arr, rmass_2d,
        do_substep=do_substep, do_coag=do_coag,
    )

    pc, sedflux = vertical(
        pc, vf, dkz, vd, dz, zc, zl, rhoa, zmet, DTYPE(dtime),
        itbnd_pc=itbnd_pc, ibbnd_pc=ibbnd_pc,
        pc_topbnd=pc_topbnd, pc_botbnd=pc_botbnd,
        ftoppart=ftoppart, fbotpart=fbotpart,
        igroup_arr=igroup_arr_tr,
        grp_do_vtran=grp_do_vtran, grp_do_drydep=grp_do_drydep,
        igridv=igridv, nbin=nbin, nelem=nelem, ngroup=ngroup,
    )

    # If substepping applies the gas/temp increment, add it here so the
    # downstream step sees the full updated gc/t. For transport-only, no
    # microfast consumes d_gc, so just add back.
    if do_substep:
        gc = gc + d_gc
        t = t + d_t

    return pc, gc, t, pcl, gcl, pconmax, sedflux
