"""Explicit-substepping vertical sedimentation solver.

Ports ``versub.F90`` — an alternative to the implicit Thomas solver in
``versol.py``. Simpler and faster for large CFL and irregular grids,
at the cost of being more diffusive. Fortran selects between the two
via ``GroupConfig.ifallrtn`` (or equivalent); Phase 9 will wire the
dispatch in ``vertical.py``.

Algorithm (per bin, per column):

1. Build net edge velocities ``up = vertadvu + vertdifu`` and
   ``dn = vertadvd + vertdifd`` (both shape ``(NZ+1,)``).
2. Estimate ``cfl_max`` over levels where the particle concentration
   is above ``SMALL_PC``.
3. Number of substeps ``nstep_sed = floor(1 + cfl_max)``; doubled if
   any edge has opposite-direction velocities (prevents over-transport
   from either side in a single step).
4. Set boundary fluxes ``fvert_1`` (bottom in Cartesian, top in
   Sigma/Hybrid) and ``fvert_nz`` (the other boundary) from either
   specified flux or ``cvert_bnd · velocity``.
5. For ``istep in 0..nstep_sed-1``, compute the net flux at each
   level, update ``cvert += fvert · dtime / nstep_sed / dz``.

The JAX port is a pure function returning the updated ``cvert`` vector.
The substep loop is a ``jax.lax.fori_loop`` with a traced upper bound.

Ported from: ``versub.F90``.
"""

import jax
import jax.numpy as jnp

from carma.constants import SMALL_PC
from carma.enums import BoundaryCondition, GridType
from carma.precision import DTYPE


def versub(
    cvert,
    pcmax,
    dz,
    dtime,
    itbnd,
    ibbnd,
    ftop,
    fbot,
    cvert_tbnd,
    cvert_bbnd,
    vertadvu,
    vertadvd,
    vertdifu,
    vertdifd,
    igridv,
):
    """Solve explicit substepped sedimentation for one bin column.

    Args:
        cvert: Quantity at layer centers, shape ``(NZ,)``.
        pcmax: Max particle concentration across bins (same column),
            shape ``(NZ,)``. Used to skip CFL contributions from
            empty layers.
        dz: Layer thickness [cm], shape ``(NZ,)``.
        dtime: Timestep [s].
        itbnd, ibbnd: Boundary-condition enum values (ints).
        ftop, fbot: Boundary fluxes (scalars).
        cvert_tbnd, cvert_bbnd: Boundary values (scalars).
        vertadvu, vertadvd, vertdifu, vertdifd: Edge-defined
            advection/diffusion rates, shape ``(NZ+1,)``.
        igridv: Grid type enum value (Python int).

    Returns:
        ``cvert`` of shape ``(NZ,)`` after explicit substepped
        sedimentation.
    """
    cvert = jnp.asarray(cvert, dtype=DTYPE)
    pcmax = jnp.asarray(pcmax, dtype=DTYPE)
    dz = jnp.asarray(dz, dtype=DTYPE)
    dtime = DTYPE(dtime)

    up = jnp.asarray(vertadvu, dtype=DTYPE) + jnp.asarray(vertdifu, dtype=DTYPE)
    dn = jnp.asarray(vertadvd, dtype=DTYPE) + jnp.asarray(vertdifd, dtype=DTYPE)

    nz = cvert.shape[0]

    # --- CFL max across levels with significant particle count ---
    # max over |up[iz]|, |up[iz+1]|, |dn[iz]|, |dn[iz+1]| at each level
    abs_up = jnp.abs(up)
    abs_dn = jnp.abs(dn)
    per_level = jnp.maximum(
        jnp.maximum(abs_up[:-1], abs_up[1:]),
        jnp.maximum(abs_dn[:-1], abs_dn[1:]),
    )                                                      # (NZ,)
    cfl_per_level = per_level * dtime / dz                 # (NZ,)
    mask = pcmax > SMALL_PC
    cfl_max = jnp.max(jnp.where(mask, cfl_per_level, DTYPE(0.0)))

    # --- Substep count ---
    # Fortran: nstep_sed = int(1 + cfl_max)  (truncation)
    nstep_sed = jnp.maximum(
        jnp.asarray(1, dtype=jnp.int32),
        jnp.floor(DTYPE(1.0) + cfl_max).astype(jnp.int32),
    )
    # Double if any edge has both up and dn flowing: prevents
    # over-transport from opposing directions within one substep.
    opposing = jnp.max(up * dn) > DTYPE(0.0)
    nstep_sed = jnp.where(opposing, nstep_sed * 2, nstep_sed)

    # --- Boundary fluxes ---
    # In Fortran-Sigma/Hybrid the top ↔ bottom roles are swapped; the
    # branches below match `versub.F90` line-for-line.
    is_flux_t = (int(itbnd) == int(BoundaryCondition.I_FLUX_SPEC))
    is_flux_b = (int(ibbnd) == int(BoundaryCondition.I_FLUX_SPEC))
    sigma_or_hybrid = (
        int(igridv) == int(GridType.I_SIG)
        or int(igridv) == int(GridType.I_HYBRID)
    )

    if sigma_or_hybrid:
        fvert_nz = (DTYPE(-1.0) * fbot if is_flux_t else cvert_bbnd * dn[nz])
        fvert_1 = (DTYPE(-1.0) * ftop if is_flux_b else cvert_tbnd * up[0])
    else:
        fvert_nz = (ftop if is_flux_t else cvert_tbnd * dn[nz])
        fvert_1 = (fbot if is_flux_b else cvert_bbnd * up[0])

    fvert_1 = jnp.asarray(fvert_1, dtype=DTYPE)
    fvert_nz = jnp.asarray(fvert_nz, dtype=DTYPE)

    dt_sub = dtime / nstep_sed.astype(DTYPE)

    # --- One substep body ---
    def step_body(i, cv):
        # Interior: fvert[i] = -cv[i]·dn[i] + cv[i-1]·up[i] + cv[i+1]·dn[i+1] - cv[i]·up[i+1]
        cv_prev = jnp.concatenate([jnp.zeros(1, dtype=DTYPE), cv[:-1]])
        cv_next = jnp.concatenate([cv[1:], jnp.zeros(1, dtype=DTYPE)])

        flux_in_below = cv_prev * up[:-1]            # cv[i-1]*up[i]
        flux_in_above = cv_next * dn[1:]             # cv[i+1]*dn[i+1]
        flux_out_down = cv * dn[:-1]                  # cv[i]*dn[i]
        flux_out_up = cv * up[1:]                     # cv[i]*up[i+1]

        fvert = (
            -flux_out_down + flux_in_below + flux_in_above - flux_out_up
        )
        # Boundary overrides
        # level 0: replace `cv[-1]·up[0]` with fvert_1
        fvert = fvert.at[0].add(-flux_in_below[0] + fvert_1)
        # level nz-1: replace `cv[nz]·dn[nz]` with fvert_nz
        fvert = fvert.at[nz - 1].add(-flux_in_above[nz - 1] + fvert_nz)

        return cv + fvert * dt_sub / dz

    # --- Loop with traced upper bound ---
    cvert_out = jax.lax.fori_loop(0, nstep_sed, step_body, cvert)

    return cvert_out
