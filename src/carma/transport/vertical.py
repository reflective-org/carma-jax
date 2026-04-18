"""Vertical transport driver for CARMA-JAX.

Orchestrates fall velocity + Brownian diffusion transport for all bins/elements.
Ported from: vertical.F90
"""

import jax.numpy as jnp

from carma.precision import DTYPE
from carma.transport.vertadv import vertadv
from carma.transport.vertdif import vertdif
from carma.transport.versol import versol


def vertical(pc, vf, dkz, vd, dz, zc, zl, rhoa, zmet, dtime,
             itbnd_pc, ibbnd_pc,
             pc_topbnd, pc_botbnd, ftoppart, fbotpart,
             igroup_arr, grp_do_vtran, grp_do_drydep,
             igridv, nbin, nelem, ngroup):
    """Apply vertical transport to all bins and elements.

    For each (ielem, ibin) pair:
      1. Build vtrans = -vf (downward)
      2. If dry dep active, set surface vtrans = -vd
      3. vertadv → advection rates
      4. vertdif → diffusion rates
      5. versol → updated concentrations

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
        grp_do_vtran: (NGROUP,) bool — does this group participate in sedimentation.
        grp_do_drydep: (NGROUP,) bool — dry deposition active.
        igridv: Grid type.
        nbin, nelem, ngroup: Dimensions.

    Returns:
        (pc, sedflux) where pc is updated (NZ, NBIN, NELEM) and
        sedflux is the sedimentation flux (NBIN, NELEM) [g/cm²/s].
    """
    old_pc = pc
    sedflux = jnp.zeros((nbin, nelem), dtype=DTYPE)

    for ielem in range(nelem):
        ig = int(igroup_arr[ielem])
        if not bool(grp_do_vtran[ig]):
            continue

        for ibin in range(nbin):
            # Downward velocity at each edge
            vtrans = -vf[:, ibin, ig]

            # Dry deposition: override surface velocity
            if bool(grp_do_drydep[ig]):
                if int(igridv) == 1:  # I_CART: surface = index 0
                    vtrans = vtrans.at[0].set(-vd[ibin, ig])
                else:  # Sigma/hybrid: surface = top (last index)
                    vtrans = vtrans.at[-1].set(-vd[ibin, ig])

            # Advection rates
            vertadvu, vertadvd = vertadv(
                vtrans, pc[:, ibin, ielem], dz, zc, zl, dtime,
                itbnd=itbnd_pc, ibbnd=ibbnd_pc,
                cvert_tbnd=pc_topbnd[ibin, ielem],
                cvert_bbnd=pc_botbnd[ibin, ielem])

            # Diffusion rates
            vertdifu, vertdifd = vertdif(
                dkz[:, ibin, ig], dz, rhoa, zmet, igridv,
                itbnd=itbnd_pc, ibbnd=ibbnd_pc)

            # Tridiagonal solver
            new_col = versol(
                pc[:, ibin, ielem], dz, dtime,
                itbnd=itbnd_pc, ibbnd=ibbnd_pc,
                ftop=ftoppart[ibin, ielem], fbot=fbotpart[ibin, ielem],
                cvert_tbnd=pc_topbnd[ibin, ielem],
                cvert_bbnd=pc_botbnd[ibin, ielem],
                vertadvu=vertadvu, vertadvd=vertadvd,
                vertdifu=vertdifu, vertdifd=vertdifd, igridv=igridv)
            pc = pc.at[:, ibin, ielem].set(new_col)

            # Sedimentation flux: how much mass left the column
            old_total = jnp.sum(old_pc[:, ibin, ielem] * dz)
            new_total = jnp.sum(pc[:, ibin, ielem] * dz)
            sed = jnp.maximum(old_total - new_total, DTYPE(0.0)) / dtime
            sedflux = sedflux.at[ibin, ielem].set(sed)

    return pc, sedflux
