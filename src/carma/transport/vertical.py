"""Vertical transport driver for CARMA-JAX.

JIT-compiled, vmap'd over (NBIN, NELEM). For each (ibin, ielem) column:
  1. Build vtrans = -vf (downward), overridden at surface for drydep.
  2. vertadv → advection rates
  3. vertdif → diffusion rates
  4. versol → updated concentrations

Ported from: vertical.F90
"""

from functools import partial

import jax
import jax.numpy as jnp

from carma.enums import GridType
from carma.precision import DTYPE
from carma.transport.vertadv import vertadv
from carma.transport.vertdif import vertdif
from carma.transport.versol import versol


@partial(jax.jit, static_argnames=(
    "itbnd_pc", "ibbnd_pc", "igridv", "nbin", "nelem", "ngroup"))
def vertical(pc, vf, dkz, vd, dz, zc, zl, rhoa, zmet, dtime,
             itbnd_pc, ibbnd_pc,
             pc_topbnd, pc_botbnd, ftoppart, fbotpart,
             igroup_arr, grp_do_vtran, grp_do_drydep,
             igridv, nbin, nelem, ngroup):
    """Apply vertical transport to all bins and elements.

    Args:
        pc: Particle concentrations (NZ, NBIN, NELEM).
        vf: Fall velocity (NZ+1, NBIN, NGROUP) [cm/s] (positive = downward magnitude).
        dkz: Diffusion coefficient at edges (NZ+1, NBIN, NGROUP) [cm²/s].
        vd: Dry deposition velocity (NBIN, NGROUP) [cm/s], or zeros if no drydep.
        dz: Layer thickness (NZ,) [cm].
        zc: Layer centers (NZ,) [cm].
        zl: Layer edges (NZ+1,) [cm].
        rhoa: Air density × zmet (NZ,) [g/cm²/z].
        zmet: Vertical metric (NZ,).
        dtime: Timestep [s].
        itbnd_pc, ibbnd_pc: Boundary conditions for particles.
        pc_topbnd, pc_botbnd: Boundary concentration values (NBIN, NELEM).
        ftoppart, fbotpart: Boundary fluxes (NBIN, NELEM).
        igroup_arr: (NELEM,) group index per element.
        grp_do_vtran: (NGROUP,) bool.
        grp_do_drydep: (NGROUP,) bool.
        igridv: Grid type (static).
        nbin, nelem, ngroup: Dimensions (static).

    Returns:
        (pc, sedflux) where pc is updated (NZ, NBIN, NELEM) and
        sedflux is sedimentation flux (NBIN, NELEM) [g/cm²/s].
    """
    # Broadcast (bin, group) quantities to (bin, elem) via igroup_arr.
    igroup_arr_j = jnp.asarray(igroup_arr)
    grp_do_vtran_j = jnp.asarray(grp_do_vtran)
    grp_do_drydep_j = jnp.asarray(grp_do_drydep)

    vf_e = vf[:, :, igroup_arr_j]        # (NZ+1, NBIN, NELEM)
    dkz_e = dkz[:, :, igroup_arr_j]      # (NZ+1, NBIN, NELEM)
    vd_e = vd[:, igroup_arr_j]           # (NBIN, NELEM)
    do_vtran_e = grp_do_vtran_j[igroup_arr_j]    # (NELEM,)
    do_drydep_e = grp_do_drydep_j[igroup_arr_j]  # (NELEM,)

    # Build vtrans: downward velocity at edges, with drydep override at surface.
    vtrans = -vf_e  # (NZ+1, NBIN, NELEM)
    if igridv == GridType.I_CART:
        # Surface = index 0
        surface = jnp.where(do_drydep_e[None, :], -vd_e, vtrans[0])  # (NBIN, NELEM)
        vtrans = vtrans.at[0].set(surface)
    else:
        # Sigma/hybrid: surface = last edge
        surface = jnp.where(do_drydep_e[None, :], -vd_e, vtrans[-1])
        vtrans = vtrans.at[-1].set(surface)

    def one_column(vtrans_col, dkz_col, cvert_col, ftop_s, fbot_s, ctop_s, cbot_s):
        vertadvu, vertadvd = vertadv(
            vtrans_col, cvert_col, dz, zc, zl, dtime,
            itbnd=itbnd_pc, ibbnd=ibbnd_pc,
            cvert_tbnd=ctop_s, cvert_bbnd=cbot_s)
        vertdifu, vertdifd = vertdif(
            dkz_col, dz, rhoa, zmet, igridv,
            itbnd=itbnd_pc, ibbnd=ibbnd_pc)
        new_col = versol(
            cvert_col, dz, dtime,
            itbnd=itbnd_pc, ibbnd=ibbnd_pc,
            ftop=ftop_s, fbot=fbot_s,
            cvert_tbnd=ctop_s, cvert_bbnd=cbot_s,
            vertadvu=vertadvu, vertadvd=vertadvd,
            vertdifu=vertdifu, vertdifd=vertdifd, igridv=igridv)
        return new_col

    # vmap over bin axis (axis 1 of column args, axis 0 of scalar args)
    col_vmap_bin = jax.vmap(
        one_column,
        in_axes=(1, 1, 1, 0, 0, 0, 0),
        out_axes=1,
    )
    # Then vmap over elem axis
    col_vmap_bin_elem = jax.vmap(
        col_vmap_bin,
        in_axes=(2, 2, 2, 1, 1, 1, 1),
        out_axes=2,
    )

    new_pc = col_vmap_bin_elem(
        vtrans, dkz_e, pc, ftoppart, fbotpart, pc_topbnd, pc_botbnd
    )  # (NZ, NBIN, NELEM)

    # Keep old pc for elements that don't transport.
    pc_out = jnp.where(do_vtran_e[None, None, :], new_pc, pc)

    # Sedimentation flux: clip to nonnegative, divide by dtime.
    dz_col = dz[:, None, None]
    old_tot = jnp.sum(pc * dz_col, axis=0)
    new_tot = jnp.sum(pc_out * dz_col, axis=0)
    sedflux = jnp.maximum(old_tot - new_tot, DTYPE(0.0)) / dtime

    return pc_out, sedflux
