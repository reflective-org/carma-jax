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

    All elements in a growing group participate in growth, not just the
    volatile element. When particles grow from bin i-1 to bin i, ALL
    element masses move together (volatile, core, etc.). The igrowgas
    parameter should be the group-level growth gas (from the number
    concentration element), NOT the per-element growth gas.

    Args:
        pc: Particle concentrations (NZ, NBIN, NELEM).
        growpe: Growth production array (NBIN, NELEM) to update.
        growlg: Growth loss rates (NBIN, NGROUP).
        pconmax: Max concentration per group (NZ, NGROUP).
        iz: Vertical level index.
        ibin: Target bin index (0-based).
        ielem: Element index.
        igroup: Group index.
        igrowgas: Growth gas index for the GROUP's number concentration
            element (-1 if the group doesn't grow).

    Returns:
        Updated growpe array (NBIN, NELEM).
    """
    # Gate as a traced expression so growp works under lax.scan (where
    # ibin / igrowgas may be traced) as well as under unrolled Python
    # loops (where they are concrete ints).
    igrowgas_j = jnp.asarray(igrowgas)
    ibin_j = jnp.asarray(ibin)
    has_particles = pconmax[iz, igroup] > FEW_PC
    gate = (igrowgas_j >= 0) & (ibin_j > 0) & has_particles

    # Growth production from bin i-1 into bin i. Reading at ibin-1 with
    # ibin=0 would yield pc[-1] = pc[NBIN-1], but the gate masks that to
    # zero so it doesn't pollute downstream.
    src_bin = jnp.maximum(ibin_j - 1, 0)
    prod_raw = pc[iz, src_bin, ielem] * growlg[src_bin, igroup]
    prod = jnp.where(gate, prod_raw, DTYPE(0.0))

    growpe = growpe.at[ibin, ielem].set(prod)

    return growpe
