"""Coagulation loss rate calculation for CARMA-JAX.

Computes loss rates for each bin/group due to collisions with all other
bins/groups.
Ported from: coagl.F90
"""

import jax.numpy as jnp

from carma.constants import FEW_PC


def coagl(config, ckernel, pcl, pconmax):
    """Compute coagulation loss rates.

    Args:
        config: CarmaConfig.
        ckernel: Coagulation kernel (NZ, NBIN, NBIN, NGROUP, NGROUP).
        pcl: Particle concentrations at start of step (NZ, NBIN, NELEM).
        pconmax: Maximum concentration per group (NZ, NGROUP).

    Returns:
        coaglg: Coagulation loss rates (NZ, NBIN, NGROUP).
    """
    nz = pcl.shape[0]
    nbin = config.nbin
    ngroup = config.ngroup

    coaglg = jnp.zeros((nz, nbin, ngroup), dtype=pcl.dtype)

    for ig in range(ngroup):
        for jg in range(ngroup):
            igrp = config.coag.icoag[ig, jg]
            if igrp < 0:
                # No coagulation for this pair (icoag==0 stored as -1)
                continue

            # Number concentration element of collision partner
            je = config.groups[jg].ienconc

            # Mask: both groups must have significant particles
            active = (pconmax[:, jg] > FEW_PC) & (pconmax[:, ig] > FEW_PC)
            active_f = active.astype(pcl.dtype)  # (NZ,)

            if igrp == ig:
                # Partial loss: product stays in same group
                # Only bins 0..NBIN-2 (Fortran: 1..NBIN-1)
                # coaglg[iz,i,ig] += ckernel[iz,i,j,ig,jg] * pcl[iz,j,je] * volx[igrp,ig,jg,i,j]
                volx_slice = config.coag.volx[igrp, ig, jg, :nbin - 1, :]  # (NBIN-1, NBIN)
                kern_slice = ckernel[:, :nbin - 1, :, ig, jg]  # (NZ, NBIN-1, NBIN)
                pcl_je = pcl[:, :, je]  # (NZ, NBIN)

                # loss[iz,i] = sum_j(kern[iz,i,j] * pcl[iz,j] * volx[i,j])
                loss = jnp.einsum(
                    "zij,zj,ij->zi", kern_slice, pcl_je, volx_slice
                )  # (NZ, NBIN-1)
                loss = loss * active_f[:, None]
                coaglg = coaglg.at[:, :nbin - 1, ig].add(loss)

            else:
                # Complete loss: product goes to different group
                # All bins 0..NBIN-1
                kern_slice = ckernel[:, :, :, ig, jg]  # (NZ, NBIN, NBIN)
                pcl_je = pcl[:, :, je]  # (NZ, NBIN)

                # loss[iz,i] = sum_j(kern[iz,i,j] * pcl[iz,j])
                loss = jnp.einsum(
                    "zij,zj->zi", kern_slice, pcl_je
                )  # (NZ, NBIN)
                loss = loss * active_f[:, None]
                coaglg = coaglg.at[:, :, ig].add(loss)

    return coaglg
