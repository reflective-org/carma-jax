"""Evaporation production for CARMA-JAX.

Computes evaporation source terms: particles shrinking from bin i to bin i-1.
Simple case: single element, no cores (I_VOLATILE only).
Ported from: evapp.F90, evap_ingrp.F90
"""

import jax.numpy as jnp

from carma.constants import FEW_PC
from carma.precision import DTYPE


def evapp(pc, evappe, evaplg, pconmax, ienconc_arr, itype_arr,
          igroup_arr, iz, nbin, ngroup, nelem):
    """Compute evaporation production terms.

    For the simple case (no cores, I_VOLATILE), particles evaporating
    from bin i contribute to bin i-1:
        evappe[i-1, ie] += pc[iz, i, ie] * evaplg[i, ig]

    Only elements belonging to the current group are processed (matching
    Fortran evap_ingrp which uses nelemg(ig) to iterate group elements).

    Args:
        pc: Particle concentrations (NZ, NBIN, NELEM).
        evappe: Evaporation production array (NBIN, NELEM) to update.
        evaplg: Evaporation loss rates (NBIN, NGROUP).
        pconmax: Max concentration per group (NZ, NGROUP).
        ienconc_arr: Number concentration element per group (NGROUP,).
        itype_arr: Element type per element (NELEM,).
        igroup_arr: Group index per element (NELEM,).
        iz: Vertical level index.
        nbin, ngroup, nelem: Dimensions.

    Returns:
        Updated evappe array (NBIN, NELEM).
    """
    from carma.enums import ElementType

    for ig in range(ngroup):
        ip = int(ienconc_arr[ig])
        itype = int(itype_arr[ip])

        # Only volatile elements
        if itype != ElementType.I_VOLATILE:
            continue

        has_particles = pconmax[iz, ig] > FEW_PC

        for ibin in range(1, nbin):  # bin 0 can't evaporate to bin -1
            evdrop = pc[iz, ibin, ip] * evaplg[ibin, ig]

            # Only if evaporation is happening (evdrop > 0)
            should_evap = has_particles & (evdrop > DTYPE(0.0))

            # Within-group evaporation: only elements in this group
            for ie in range(nelem):
                if int(igroup_arr[ie]) != ig:
                    continue
                prod = pc[iz, ibin, ie] * evaplg[ibin, ig]
                prod = jnp.where(should_evap, prod, DTYPE(0.0))
                evappe = evappe.at[ibin - 1, ie].add(prod)

    return evappe


def downgevapply(pc, evappe, rnucpe, dtime, iz, nbin, nelem):
    """Apply evaporation and nucleation production to particle state.

    Simple explicit update: pc += dtime * (evappe + rnucpe)

    Args:
        pc: Particle concentrations (NZ, NBIN, NELEM).
        evappe: Evaporation production (NBIN, NELEM).
        rnucpe: Nucleation production (NBIN, NELEM) — zero for growth-only.
        dtime: Timestep [s].
        iz: Vertical level index.
        nbin, nelem: Dimensions.

    Returns:
        Updated pc array.
    """
    for ielem in range(nelem):
        for ibin in range(nbin):
            pc = pc.at[iz, ibin, ielem].add(
                dtime * (evappe[ibin, ielem] + rnucpe[ibin, ielem])
            )
            # Floor at SMALL_PC
            from carma.constants import SMALL_PC
            pc = pc.at[iz, ibin, ielem].set(
                jnp.maximum(pc[iz, ibin, ielem], SMALL_PC)
            )

    return pc
