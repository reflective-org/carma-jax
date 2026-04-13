"""Slow microphysics driver (coagulation) for CARMA-JAX.

Orchestrates the coagulation calculation sequence: loss rates, production
rates, and concentration updates for one timestep.
Ported from: microslow.F90
"""

import jax.numpy as jnp

from carma.coagulation.coagl import coagl
from carma.coagulation.coagp import coagp
from carma.coagulation.csolve import csolve


def microslow(config, pc, pcl, ckernel, pconmax, zmet, dtime):
    """Execute coagulation for one timestep.

    Args:
        config: CarmaConfig.
        pc: Particle concentrations (NZ, NBIN, NELEM).
        pcl: Particle concentrations at start of step (NZ, NBIN, NELEM).
        ckernel: Coagulation kernel (NZ, NBIN, NBIN, NGROUP, NGROUP).
        pconmax: Maximum concentration per group (NZ, NGROUP).
        zmet: Vertical metric (NZ,).
        dtime: Timestep [s].

    Returns:
        Updated pc array (NZ, NBIN, NELEM).
    """
    nz, nbin, nelem = pc.shape

    # Initialize production and loss arrays
    coagpe = jnp.zeros_like(pc)
    coaglg = jnp.zeros((nz, nbin, config.ngroup), dtype=pc.dtype)

    # Step 1: Compute loss rates
    coaglg = coagl(config, ckernel, pcl, pconmax)

    # Step 2: Loop over elements and bins — compute production and solve
    for ielem in range(nelem):
        igroup = config.elements[ielem].igroup
        for ibin in range(nbin):
            coagpe = coagp(
                config, ckernel, pc, pcl, pconmax, coagpe, ibin, ielem
            )
            pc = csolve(pc, coagpe, coaglg, zmet, dtime, ibin, ielem, igroup)

    return pc
