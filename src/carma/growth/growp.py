"""Growth production for CARMA-JAX.

Computes production terms from condensational growth: particles
growing from bin i-1 into bin i at rate growlg(i-1).
Ported from: growp.F90
"""

import jax.numpy as jnp

from carma.constants import FEW_PC
from carma.precision import DTYPE


def growp(pc, growpe, growlg, pconmax, iz, ibin, ielem, igroup, igrowgas):
    """Compute growth production for one bin/element.

    Growth production = concentration in bin i-1 × loss rate from bin i-1.
    Particles grow from smaller to larger bins.

    Args:
        pc: Particle concentrations (NZ, NBIN, NELEM).
        growpe: Growth production array (NBIN, NELEM) to update.
        growlg: Growth loss rates (NBIN, NGROUP).
        pconmax: Max concentration per group (NZ, NGROUP).
        iz: Vertical level index.
        ibin: Target bin index (0-based).
        ielem: Element index.
        igroup: Group index.
        igrowgas: Growth gas index for this element (-1 if none).

    Returns:
        Updated growpe array (NBIN, NELEM).
    """
    # Only if this element has a growth gas and not the first bin
    if igrowgas < 0 or ibin == 0:
        return growpe

    # Only if significant concentration in this group
    has_particles = pconmax[iz, igroup] > FEW_PC

    # Growth production: particles from bin i-1 growing into bin i
    prod = pc[iz, ibin - 1, ielem] * growlg[ibin - 1, igroup]
    prod = jnp.where(has_particles, prod, DTYPE(0.0))

    growpe = growpe.at[ibin, ielem].set(prod)

    return growpe
