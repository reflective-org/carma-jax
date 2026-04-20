"""Top-level timestep driver for CARMA-JAX.

Composes the per-step physics: prestep → vertical transport →
[microslow (coag)] → [microfast (growth/nucleation)].

Currently implemented:
  step_transport — prestep + vertical (no microphysics).
  step_coag      — prestep + microslow.
microfast orchestration is still handled by Phase 3 driver scripts;
its integration into step() is deferred until the microfast driver
is refactored onto the unified CarmaState interface.

Ported from: step.F90, newstate.F90
"""

from functools import partial
from typing import Callable, Optional

import jax
import jax.numpy as jnp

from carma.enums import BoundaryCondition, GridType
from carma.microslow import make_microslow
from carma.newstate_calc import microfast_growth
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


def make_step_transport(config):
    """Build a JIT'd transport-only step from a CarmaConfig.

    Pre-binds element metadata + static shape / boundary-condition args.
    Returned closure accepts only the per-timestep state arrays.

    Usage:
        step_transport = make_step_transport(config)
        for istep in range(nstep):
            pc, gc, t, pcl, gcl, pconmax, sedflux = step_transport(
                pc, gc, t, pcl, gcl, told, zmet,
                vf, dkz, vd, dz, zc, zl, rhoa, dtime,
                pc_topbnd, pc_botbnd, ftoppart, fbotpart)
    """
    itype_arr = jnp.array([e.itype for e in config.elements])
    ienconc_arr = jnp.array([g.ienconc for g in config.groups])
    igelem_arr = jnp.array([e.igroup for e in config.elements])
    rmass_2d = jnp.stack([g.rmass for g in config.groups], axis=1)
    igroup_arr_tr = igelem_arr  # transport uses element→group mapping
    grp_do_vtran = jnp.array([g.do_vtran for g in config.groups])
    grp_do_drydep = jnp.array([g.do_drydep for g in config.groups])

    do_substep = bool(config.do_substep)
    do_coag = bool(config.do_coag)
    nbin, nelem, ngroup = config.nbin, config.nelem, config.ngroup
    itbnd_pc, ibbnd_pc = int(config.itbnd_pc), int(config.ibbnd_pc)
    # igridv isn't on CarmaConfig yet; default to Cartesian.
    igridv = GridType.I_CART

    @jax.jit
    def step_fn(pc, gc, t, pcl, gcl, told, zmet,
                vf, dkz, vd, dz, zc, zl, rhoa, dtime,
                pc_topbnd, pc_botbnd, ftoppart, fbotpart):
        return step_transport(
            pc, gc, t, pcl, gcl, told, zmet,
            itype_arr, ienconc_arr, igelem_arr, rmass_2d,
            vf, dkz, vd, dz, zc, zl, rhoa, dtime,
            pc_topbnd, pc_botbnd, ftoppart, fbotpart,
            igroup_arr_tr, grp_do_vtran, grp_do_drydep,
            itbnd_pc=itbnd_pc, ibbnd_pc=ibbnd_pc, igridv=igridv,
            nbin=nbin, nelem=nelem, ngroup=ngroup,
            do_substep=do_substep, do_coag=do_coag,
        )

    return step_fn


def make_step_coag(config):
    """Build a JIT'd coagulation-only step from a CarmaConfig.

    Pre-binds the expensive pair arrays and element metadata once so that
    repeated calls inside a simulation loop avoid re-tracing.

    Usage:
        step_coag = make_step_coag(config)
        for istep in range(nstep):
            pc, gc, t, pcl, gcl, pconmax = step_coag(
                pc, gc, t, pcl, gcl, told, zmet, ckernel, dtime)
    """
    itype_arr = jnp.array([e.itype for e in config.elements])
    ienconc_arr = jnp.array([g.ienconc for g in config.groups])
    igelem_arr = jnp.array([e.igroup for e in config.elements])
    rmass_2d = jnp.stack([g.rmass for g in config.groups], axis=1)  # (NBIN, NGROUP)

    microslow_jit = make_microslow(
        nbin=config.nbin, nelem=config.nelem, ngroup=config.ngroup,
        elem_igroup=jnp.array([e.igroup for e in config.elements]),
        icoag=config.coag.icoag,
        volx=config.coag.volx,
        icoagelem=config.coag.icoagelem,
        npairu=config.coag.npairu, npairl=config.coag.npairl,
        iup=config.coag.iup, jup=config.coag.jup,
        igup=config.coag.igup, jgup=config.coag.jgup,
        ilow=config.coag.ilow, jlow=config.coag.jlow,
        iglow=config.coag.iglow, jglow=config.coag.jglow,
        pkernel=config.coag.pkernel,
        ienconc_arr=ienconc_arr,
        elem_itypes=itype_arr,
    )

    do_substep = bool(config.do_substep)
    do_coag = bool(config.do_coag)

    @jax.jit
    def step_coag(pc, gc, t, pcl, gcl, told, zmet, ckernel, dtime):
        """Prestep → microslow for one timestep.

        Gas / temperature are carried through unchanged (coagulation does
        not touch them).
        """
        pc, gc, t, pcl, gcl, d_gc, d_t, pconmax = prestep(
            pc, gc, t, pcl, gcl, told, zmet,
            itype_arr, ienconc_arr, igelem_arr, rmass_2d,
            do_substep=do_substep, do_coag=do_coag,
        )
        pc = microslow_jit(pc, pcl, ckernel, pconmax, zmet, DTYPE(dtime))
        if do_substep:
            gc = gc + d_gc
            t = t + d_t
        return pc, gc, t, pcl, gcl, pconmax

    return step_coag


def make_step_microfast(config, ntsubsteps=1, igrowgas_arr=None):
    """Build a JIT'd single-level microfast step from a CarmaConfig.

    Fixed-ntsubsteps growth/evaporation. Does NOT yet include the
    adaptive retry loop from newstate_calc_growth — see that function
    for the Python-level retry wrapper.

    Args:
        config: CarmaConfig.
        ntsubsteps: Static number of substeps per call.
        igrowgas_arr: Optional tuple of length NELEM giving the growth gas
            index for each element (-1 if none). If omitted, defaults to
            element 0 → gas 0 for every element in a growing group; set
            explicitly for mixed element/gas configurations.

    Usage:
        step_microfast = make_step_microfast(config, ntsubsteps=4)
        for istep in range(nstep):
            pc, gc, t, rlheat_total = step_microfast(
                pc, gc, t, dtime,
                rhoa, zmet, rlhe, rlhm, diffus,
                akelvin, akelvini, gro, gro1, gro2,
                rup_wet, rmass_2d, dm_2d, rlow_wet,
                pratt, prat, pden1, palr,
                ds_threshold_arr=jnp.zeros(ngas))
    """
    # Pre-extract metadata as Python tuples (JIT-friendly).
    is_ice_arr = tuple(bool(g.is_ice) for g in config.groups)
    ienconc_arr = tuple(int(g.ienconc) for g in config.groups)
    igroup_arr = tuple(int(e.igroup) for e in config.elements)
    itype_arr = tuple(int(e.itype) for e in config.elements)
    gwtmol_arr = tuple(float(g.wtmol) for g in config.gases) or (18.0,)

    if igrowgas_arr is None:
        # Default: every element in a growing group maps to gas 0.
        igrowgas_arr = tuple(
            0 if (config.do_grow and config.ngas > 0) else -1
            for _ in config.elements
        )
    else:
        igrowgas_arr = tuple(int(x) for x in igrowgas_arr)

    nbin = config.nbin
    nelem = config.nelem
    ngroup = config.ngroup
    ngas = config.ngas
    dt_threshold = float(config.dt_threshold)
    _ntsubsteps = int(ntsubsteps)

    @jax.jit
    def step_microfast(pc, gc, t, dtime,
                       rhoa, zmet, rlhe, rlhm, diffus,
                       akelvin, akelvini, gro, gro1, gro2,
                       rup_wet, rmass_2d, dm_2d, rlow_wet,
                       pratt, prat, pden1, palr,
                       ds_threshold_arr, iz=0, scale_threshold=1.0):
        """Run ntsubsteps of microfast_growth at one level."""
        dt_sub = DTYPE(dtime) / DTYPE(_ntsubsteps)

        def body(_, carry):
            pc_c, gc_c, t_c, rlheat_acc = carry
            pc_c, gc_c, t_c, rl_val, _rc = microfast_growth(
                pc_c, gc_c, t_c, iz, dt_sub,
                rhoa, zmet, rlhe, rlhm, diffus,
                akelvin, akelvini, gro, gro1, gro2,
                rup_wet, rmass_2d, dm_2d, rlow_wet,
                pratt, prat, pden1, palr,
                is_ice_arr, igrowgas_arr, ienconc_arr,
                igroup_arr, gwtmol_arr,
                nbin, ngroup, ngas, nelem,
                DTYPE(dt_threshold), ds_threshold_arr, DTYPE(scale_threshold),
                itype_arr=itype_arr,
            )
            return (pc_c, gc_c, t_c, rlheat_acc + rl_val)

        init = (pc, gc, t, DTYPE(0.0))
        pc, gc, t, rlheat_total = jax.lax.fori_loop(0, _ntsubsteps, body, init)
        return pc, gc, t, rlheat_total

    return step_microfast
