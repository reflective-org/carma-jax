"""Coagulation solver for CARMA-JAX.

Implicit Euler update of particle concentrations from coagulation
production and loss rates.
Ported from: csolve.F90
"""

import jax.numpy as jnp


def csolve(pc, coagpe, coaglg, zmet, dtime, ibin, ielem, igroup):
    """Update particle concentrations via implicit Euler for coagulation.

    Solves: (pc_new - pc_old) / dt = ppd - pls * pc_new
    Result: pc_new = (pc_old + dt * ppd) / (1 + pls * dt)

    Args:
        pc: Particle concentrations (NZ, NBIN, NELEM).
        coagpe: Coagulation production rate (NZ, NBIN, NELEM).
        coaglg: Coagulation loss rate (NZ, NBIN, NGROUP).
        zmet: Vertical metric (NZ,).
        dtime: Timestep [s].
        ibin: Bin index.
        ielem: Element index.
        igroup: Group index for this element.

    Returns:
        Updated pc array (NZ, NBIN, NELEM).
    """
    ppd = coagpe[:, ibin, ielem] / zmet
    pls = coaglg[:, ibin, igroup] / zmet

    pc_old = pc[:, ibin, ielem]
    pc_new = (pc_old + dtime * ppd) / (1.0 + pls * dtime)

    return pc.at[:, ibin, ielem].set(pc_new)
