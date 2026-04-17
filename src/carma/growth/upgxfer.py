"""Nucleation production transfer (upgxfer) for CARMA-JAX.

Computes production terms for particles nucleated from one group
into another. Transfers mass/number from source bins to target bins
based on nucleation loss rates (rnuclg).
Ported from: upgxfer.F90
"""

import jax.numpy as jnp

from carma.constants import FEW_PC
from carma.enums import ElementType
from carma.precision import DTYPE


def upgxfer(rnucpe, rnuclg, pc, rmass, pconmax,
            ielem, ibin, iz,
            nnucelem, inucelem, nnucbin, inucbin,
            igelem_arr, itype_arr, nbin, ngroup):
    """Compute nucleation production for one target bin/element.

    When particles in source group freeze/nucleate into target group,
    this routine computes the production rate in the target bin.

    rnucpe[ibin, ielem] += rnuclg[ifrom, igfrom, igroup] * pc[iz, ifrom, iefrom] * elemass^ipow

    Args:
        rnucpe: Nucleation production array (NBIN, NELEM) to update.
        rnuclg: Nucleation loss rates (NBIN, NGROUP, NGROUP).
        pc: Particle concentrations (NZ, NBIN, NELEM).
        rmass: Particle mass per group (NBIN, NGROUP) [g].
        pconmax: Max concentration per group (NZ, NGROUP).
        ielem: Target element index.
        ibin: Target bin index.
        iz: Vertical level index.
        nnucelem: Number of source elements for each target element (NELEM,).
        inucelem: Source element indices (max_nuc_elem, NELEM).
        nnucbin: Number of source bins for each target bin (NGROUP, NBIN, NGROUP).
        inucbin: Source bin indices (max_nuc_bin, NGROUP, NBIN, NGROUP).
        igelem_arr: Group index per element (NELEM,).
        itype_arr: Element type per element (NELEM,).
        nbin, ngroup: Dimensions.

    Returns:
        Updated rnucpe array (NBIN, NELEM).
    """
    igroup = int(igelem_arr[ielem])  # target group

    # Determine mass power based on element types
    itype_to = int(itype_arr[ielem])
    if itype_to == ElementType.I_COREMASS or itype_to == ElementType.I_VOLCORE:
        ipow_to = 1
    elif itype_to == ElementType.I_CORE2MOM:
        ipow_to = 2
    else:
        ipow_to = 0

    # Loop over source elements that nucleate into this element
    n_nuc = int(nnucelem[ielem])
    for jefrom in range(n_nuc):
        iefrom = int(inucelem[jefrom, ielem])

        # Only process if target > source (upgxfer direction)
        if ielem <= iefrom:
            continue

        igfrom = int(igelem_arr[iefrom])

        # Source element mass power
        itype_from = int(itype_arr[iefrom])
        if itype_from == ElementType.I_COREMASS or itype_from == ElementType.I_VOLCORE:
            ipow_from = 1
        elif itype_from == ElementType.I_CORE2MOM:
            ipow_from = 2
        else:
            ipow_from = 0

        ipow = ipow_to - ipow_from

        # Loop over source bins that nucleate into target bin
        n_nucbin = int(nnucbin[igfrom, ibin, igroup])
        for jfrom in range(n_nucbin):
            ifrom = int(inucbin[jfrom, igfrom, ibin, igroup])

            has_particles = pconmax[iz, igfrom] > FEW_PC
            has_nucleation = rnuclg[ifrom, igfrom, igroup] > DTYPE(0.0)

            if not (has_particles and has_nucleation):
                continue

            # Element mass for source particle
            # Simple case: no cores in source group
            elemass = rmass[ifrom, igfrom]

            # Production rate
            rnucprod = (rnuclg[ifrom, igfrom, igroup]
                        * pc[iz, ifrom, iefrom]
                        * elemass ** ipow)

            rnucpe = rnucpe.at[ibin, ielem].add(rnucprod)

    return rnucpe
