"""Coagulation production rate calculation for CARMA-JAX.

Computes production terms for each bin/element from collisions of
particle pairs that produce mass in the target bin.
Ported from: coagp.F90
"""

import jax.numpy as jnp

from carma.constants import FEW_PC
from carma.enums import ElementType


def coagp(config, ckernel, pc, pcl, pconmax, coagpe, ibin, ielem):
    """Compute coagulation production for one bin/element.

    Production comes from two sources:
    - Upper pairs: collision products land in bin ibin+1 (contribute via pkernel indices 1,3,5)
    - Lower pairs: collision products land in bin ibin (contribute via pkernel indices 2,4,6)

    Args:
        config: CarmaConfig.
        ckernel: Coagulation kernel (NZ, NBIN, NBIN, NGROUP, NGROUP).
        pc: Current particle concentrations (NZ, NBIN, NELEM).
        pcl: Particle concentrations at start of step (NZ, NBIN, NELEM).
        pconmax: Maximum concentration per group (NZ, NGROUP).
        coagpe: Coagulation production array to update (NZ, NBIN, NELEM).
        ibin: Target bin index (0-based).
        ielem: Target element index (0-based).

    Returns:
        Updated coagpe array (NZ, NBIN, NELEM).
    """
    nz = pc.shape[0]
    igroup = config.elements[ielem].igroup
    itype = config.elements[ielem].itype

    # Determine pkernel index based on element type
    # Upper: 0,2,4 (Fortran 1,3,5), Lower: 1,3,5 (Fortran 2,4,6)
    if itype == ElementType.I_COREMASS or itype == ElementType.I_VOLCORE:
        i_pkern_upper = 2  # Fortran index 3
        i_pkern_lower = 3  # Fortran index 4
    elif itype == ElementType.I_CORE2MOM:
        i_pkern_upper = 4  # Fortran index 5
        i_pkern_lower = 5  # Fortran index 6
    else:
        i_pkern_upper = 0  # Fortran index 1
        i_pkern_lower = 1  # Fortran index 2

    # Process upper bin pairs
    for igrp_idx in range(config.ngroup):
        n_upper = int(config.coag.npairu[igrp_idx, ibin])
        for iquad in range(n_upper):
            ig = int(config.coag.igup[igrp_idx, ibin, iquad])
            jg = int(config.coag.jgup[igrp_idx, ibin, iquad])
            i = int(config.coag.iup[igrp_idx, ibin, iquad])
            j = int(config.coag.jup[igrp_idx, ibin, iquad])

            iefrom = int(config.coag.icoagelem[ielem, ig])
            if iefrom < 0:
                continue

            je = config.groups[jg].ienconc
            pk = config.coag.pkernel[i, j, ig, jg, igrp_idx, i_pkern_upper]

            active = (pconmax[:, ig] > FEW_PC) & (pconmax[:, jg] > FEW_PC)
            active_f = active.astype(pc.dtype)

            prod = (
                pc[:, i, iefrom]
                * pcl[:, j, je]
                * ckernel[:, i, j, ig, jg]
                * pk
                * active_f
            )
            coagpe = coagpe.at[:, ibin, ielem].add(prod)

    # Process lower bin pairs
    for igrp_idx in range(config.ngroup):
        n_lower = int(config.coag.npairl[igrp_idx, ibin])
        for iquad in range(n_lower):
            ig = int(config.coag.iglow[igrp_idx, ibin, iquad])
            jg = int(config.coag.jglow[igrp_idx, ibin, iquad])
            i = int(config.coag.ilow[igrp_idx, ibin, iquad])
            j = int(config.coag.jlow[igrp_idx, ibin, iquad])

            iefrom = int(config.coag.icoagelem[ielem, ig])
            if iefrom < 0:
                continue

            je = config.groups[jg].ienconc
            pk = config.coag.pkernel[i, j, ig, jg, igrp_idx, i_pkern_lower]

            active = (pconmax[:, ig] > FEW_PC) & (pconmax[:, jg] > FEW_PC)
            active_f = active.astype(pc.dtype)

            prod = (
                pc[:, i, iefrom]
                * pcl[:, j, je]
                * ckernel[:, i, j, ig, jg]
                * pk
                * active_f
            )
            coagpe = coagpe.at[:, ibin, ielem].add(prod)

    return coagpe
