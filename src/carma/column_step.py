"""Combined column step: vertical transport → per-level microphysics.

Mirrors the Fortran ``CARMASTATE_Step → newstate`` ordering exactly:

    1. ``vertical(pc_col, ...)``     — sedimentation + diffusion (if do_vtran)
    2. ``pcl[:] = pc[:]``            — snapshot after transport (newstate.F90:59)
    3. Per-level: ``step_full_faithful(pc[iz], gc, t, ..., iz=iz)``

Ported from: ``newstate.F90:52-109``

The per-level chemistry loop uses a Python ``for`` over the static NZ
dimension (which unrolls at JAX trace time). This is safe and correct
because:

- ``step_full_faithful`` takes ``pc[iz]`` (per-level, shape ``(NBIN, NELEM)``)
  but ``gc`` / ``t`` as full-column arrays ``(NZ, NGAS)`` / ``(NZ,)`` with an
  explicit ``iz`` kwarg that selects the level inside ``microfast_full_jit``.
- Levels are independent in CARMA's chemistry (no horizontal/vertical
  coupling within the chemistry step), so the sequential loop is equivalent
  to an embarrassingly-parallel vmap.
- The Python loop bounds are over static NZ, so unrolling is fine.

A future refactor can replace the Python loop with ``lax.fori_loop`` once
the adaptive-retry while loop inside ``step_full_faithful`` moves to
``lax.while_loop``.

For NZ = 1, the column step reduces to a single call of ``step_full_faithful``
with zero overhead (transport is a no-op for single-level columns).
"""
import jax.numpy as jnp

from carma.enums import BoundaryCondition, GridType
from carma.precision import DTYPE
from carma.step_full_faithful import make_step_full_faithful
from carma.transport.vertical import vertical


def make_column_step_full(
    config,
    *,
    igroup_sulfate: int = 0,
    igas_h2o: int = 0,
    igas_h2so4: int = 1,
    method: str = "ZhaoTurco",
    minsubsteps: int = 1,
    maxsubsteps: int = 32,
    maxretries: int = 16,
    do_homogeneous: bool = True,
    do_heterogeneous: bool = False,
    do_pheatatm: bool = False,
    ppm_coefs=None,
    itbnd_pc: int = int(BoundaryCondition.I_FIXED_CONC),
    ibbnd_pc: int = int(BoundaryCondition.I_FIXED_CONC),
    igridv: int = int(GridType.I_CART),
):
    """Build a column step: vertical transport + per-level faithful chemistry.

    The returned closure has the signature::

        pc_col, gc_col, t_col, sedflux, diags = column_step(
            pc_col, gc_col, t_col, dtime,
            vf, dkz, vd, dz, zc, zl, rhoa, zmet,
            pc_topbnd, pc_botbnd, ftoppart, fbotpart,
            pcl_col, gcl_col, told_col, d_gc_col, d_t_col,
            env,
        )

    where:

    - ``pc_col``:  ``(NZ, NBIN, NELEM)``
    - ``gc_col``, ``gcl_col``, ``told_col``:  ``(NZ, NGAS)`` / ``(NZ,)``
    - ``t_col``, ``told_col``:  ``(NZ,)``
    - ``d_gc_col``:  ``(NZ, NGAS)`` — per-step gas drift from ``prestep``
    - ``d_t_col``:   ``(NZ,)`` — per-step T drift from ``prestep``
    - ``vf``:   ``(NZ+1, NBIN, NGROUP)`` fall velocities [cm/s]
    - ``dkz``:  ``(NZ+1, NBIN, NGROUP)`` diffusivities [cm²/s]
    - ``vd``:   ``(NBIN, NGROUP)`` dry-dep velocities [cm/s]
    - ``dz``:   ``(NZ,)`` layer thicknesses [cm]
    - ``zc``, ``zl``: layer centres and edges [cm]
    - ``rhoa``, ``zmet``:  ``(NZ,)``
    - ``env``:  dict of per-column chemistry env arrays (``rhoa``,
                ``zmet``, ``akelvin``, ``akelvini``, ``gro``, ``gro1``,
                ``rup_wet``, ``rlhe``, ``rlhm``, ``ckernel``, ``pconmax``,
                ``ds_threshold_arr``) matching the ``step_full_faithful``
                kwarg signature.

    Returns ``(pc_col, gc_col, t_col, sedflux, diags)`` where
    ``sedflux`` is ``(NBIN, NELEM)`` and ``diags`` is a list of NZ
    per-level diagnostic dicts.
    """
    nbin = int(config.nbin)
    nelem = int(config.nelem)
    ngroup = int(config.ngroup)
    igroup_arr = jnp.array([e.igroup for e in config.elements])
    grp_do_vtran = jnp.array([g.do_vtran for g in config.groups])
    grp_do_drydep = jnp.array([g.do_drydep for g in config.groups])
    do_vtran = bool(any(g.do_vtran for g in config.groups))

    _cell_step = make_step_full_faithful(
        config,
        igroup_sulfate=igroup_sulfate,
        igas_h2o=igas_h2o,
        igas_h2so4=igas_h2so4,
        method=method,
        minsubsteps=minsubsteps,
        maxsubsteps=maxsubsteps,
        maxretries=maxretries,
        do_homogeneous=do_homogeneous,
        do_heterogeneous=do_heterogeneous,
        do_pheatatm=do_pheatatm,
        ppm_coefs=ppm_coefs,
    )

    def column_step(
        pc_col, gc_col, t_col, dtime,
        vf, dkz, vd, dz, zc, zl, rhoa, zmet,
        pc_topbnd, pc_botbnd, ftoppart, fbotpart,
        pcl_col, gcl_col, told_col, d_gc_col, d_t_col,
        env,
    ):
        """Run one timestep over a full column.

        Args:
            pc_col:       ``(NZ, NBIN, NELEM)``
            gc_col:       ``(NZ, NGAS)``
            t_col:        ``(NZ,)``
            dtime:        timestep [s]
            vf:           ``(NZ+1, NBIN, NGROUP)`` fall velocities
            dkz:          ``(NZ+1, NBIN, NGROUP)`` diffusivities
            vd:           ``(NBIN, NGROUP)`` dry-dep velocities
            dz:           ``(NZ,)`` layer thicknesses [cm]
            zc, zl:       layer centres / edges [cm]
            rhoa:         ``(NZ,)``
            zmet:         ``(NZ,)``
            pc_topbnd, pc_botbnd: ``(NBIN, NELEM)``
            ftoppart, fbotpart:   ``(NBIN, NELEM)``
            pcl_col:      ``(NZ, NBIN, NELEM)`` prestep snapshot (overridden
                          after transport, but used as a reasonable initial
                          pconmax estimate before the transport overwrites)
            gcl_col:      ``(NZ, NGAS)`` prestep gas snapshot
            told_col:     ``(NZ,)`` prestep T snapshot
            d_gc_col:     ``(NZ, NGAS)`` per-step gas drift
            d_t_col:      ``(NZ,)`` per-step T drift
            env:          dict of chemistry environment arrays — keys
                          ``rhoa, zmet, akelvin, akelvini, gro, gro1,
                          rup_wet, rlhe, rlhm, ckernel, pconmax,
                          ds_threshold_arr``

        Returns:
            ``(pc_col, gc_col, t_col, sedflux, diags)``
        """
        nz = pc_col.shape[0]

        # 1. Vertical transport (Fortran newstate.F90:52-56)
        if do_vtran:
            pc_col, sedflux = vertical(
                pc_col, vf, dkz, vd, dz, zc, zl, rhoa, zmet, DTYPE(dtime),
                itbnd_pc=itbnd_pc, ibbnd_pc=ibbnd_pc,
                pc_topbnd=pc_topbnd, pc_botbnd=pc_botbnd,
                ftoppart=ftoppart, fbotpart=fbotpart,
                igroup_arr=igroup_arr,
                grp_do_vtran=grp_do_vtran, grp_do_drydep=grp_do_drydep,
                igridv=igridv, nbin=nbin, nelem=nelem, ngroup=ngroup,
            )
        else:
            sedflux = jnp.zeros((nbin, nelem), dtype=DTYPE)

        # 2. Post-transport pcl snapshot (Fortran newstate.F90:59).
        #    Overrides the prestep pcl for the chemistry retry restart so
        #    that sedimentation-induced concentration changes are included
        #    in the starting point for the adaptive retry.
        pcl_after_transport = pc_col

        # 3. Per-level microphysics (Fortran newstate_calc per iz).
        #
        # ``step_full_faithful`` takes the FULL-COLUMN pc/gc/t arrays
        # (shapes (NZ,NBIN,NELEM), (NZ,NGAS), (NZ,)) and uses ``iz`` to
        # select the level inside ``microfast_full_jit``. It returns an
        # updated full-column tuple where only level ``iz`` is modified.
        #
        # Levels are chemistry-independent in CARMA (no vertical coupling
        # within the chemistry step), so the sequential loop is physically
        # equivalent to an embarrassingly-parallel vmap over levels.
        diags = []
        for iz in range(nz):
            pc_col, gc_col, t_col, diag = _cell_step(
                pc_col, gc_col, t_col, dtime,
                pcl=pcl_after_transport,   # full column, level selected by iz
                gcl=gcl_col,
                told=told_col,
                d_gc=d_gc_col,
                d_t=d_t_col,
                iz=iz,
                **env,
            )
            diags.append(diag)

        return pc_col, gc_col, t_col, sedflux, diags

    return column_step
