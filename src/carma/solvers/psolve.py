"""Particle concentration solver for CARMA-JAX.

Implicit Euler solver for particle number concentration from growth,
evaporation, and nucleation production/loss terms.
Ported from: psolve.F90
"""

import jax.numpy as jnp

from carma.constants import SMALL_PC
from carma.precision import DTYPE


def psolve(pc, pc_nucl, growpe, evappe, rnucpe, rhompe,
           growlg, evaplg, rnuclg, dtime, iz, ibin, ielem, igroup, ngroup):
    """Update particle concentration via implicit Euler.

    Solves: (pc_new - pc_old) / dt = ppd - pls * pc_new
    Where:
        ppd = rnucpe + rhompe + growpe + evappe (total production)
        pls = sum(rnuclg) + growlg + evaplg (total loss)

    Args:
        pc: Particle concentrations (NZ, NBIN, NELEM).
        pc_nucl: Nucleation production accumulator (NZ, NBIN, NELEM).
        growpe: Growth production (NBIN, NELEM).
        evappe: Evaporation production (NBIN, NELEM).
        rnucpe: Nucleation production (NBIN, NELEM).
        rhompe: Homogeneous nucleation production (NBIN, NELEM).
        growlg: Growth loss rate (NBIN, NGROUP).
        evaplg: Evaporation loss rate (NBIN, NGROUP).
        rnuclg: Nucleation loss rate (NBIN, NGROUP, NGROUP).
        dtime: Timestep [s].
        iz: Vertical level index.
        ibin: Bin index.
        ielem: Element index.
        igroup: Group index.
        ngroup: Number of groups.

    Returns:
        Tuple of (pc, pc_nucl) updated arrays.
    """
    # Total production
    ppd = (rnucpe[ibin, ielem] + rhompe[ibin, ielem]
           + growpe[ibin, ielem] + evappe[ibin, ielem])

    # Total nucleation loss (sum over target groups)
    rnuclgtot = jnp.sum(rnuclg[ibin, igroup, :])

    # Total loss
    pls = rnuclgtot + growlg[ibin, igroup] + evaplg[ibin, igroup]

    # Particle concentration without nucleation (for diagnostic)
    ppd_nonuc = growpe[ibin, ielem] + evappe[ibin, ielem]
    pls_nonuc = growlg[ibin, igroup] + evaplg[ibin, igroup]
    pc_nonuc = (pc[iz, ibin, ielem] + dtime * ppd_nonuc) / (DTYPE(1.0) + pls_nonuc * dtime)

    # Full solution with nucleation
    pc_new = (pc[iz, ibin, ielem] + dtime * ppd) / (DTYPE(1.0) + pls * dtime)

    # Floor at SMALL_PC
    pc_new = jnp.maximum(pc_new, SMALL_PC)

    # Track nucleation contribution
    pc_nucl = pc_nucl.at[iz, ibin, ielem].add(pc_new - pc_nonuc)

    pc = pc.at[iz, ibin, ielem].set(pc_new)

    return pc, pc_nucl
