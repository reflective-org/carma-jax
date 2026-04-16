"""Total condensate calculation for CARMA-JAX.

Computes total ice and liquid mass from particle concentrations,
used by gsolve for mass conservation.
Ported from: totalcondensate.F90
"""

import jax.numpy as jnp

from carma.precision import DTYPE


def totalcondensate(pc, rmass, igroup_arr, is_ice_arr, igrowgas_arr,
                    nbin, ngroup, ngas, iz):
    """Compute total ice and liquid condensate mass at one level.

    For each gas species, sums the volatile mass (total particle mass
    minus core mass) across all bins and groups.

    Args:
        pc: Particle concentrations (NZ, NBIN, NELEM).
        rmass: Particle mass per bin (NBIN, NGROUP) [g].
        igroup_arr: Group index per element (NELEM,).
        is_ice_arr: Ice flag per group (NGROUP,).
        igrowgas_arr: Growth gas index per element (NELEM,), -1 if none.
        nbin, ngroup, ngas: Dimensions.
        iz: Vertical level index.

    Returns:
        Tuple of (total_ice, total_liquid) each shape (NGAS,).
    """
    total_ice = jnp.zeros(max(ngas, 1), dtype=DTYPE)
    total_liquid = jnp.zeros(max(ngas, 1), dtype=DTYPE)

    nelem = len(igroup_arr)

    for ielem in range(nelem):
        ig = int(igroup_arr[ielem])
        igas = int(igrowgas_arr[ielem])
        if igas < 0:
            continue

        is_ice = bool(is_ice_arr[ig])

        for ibin in range(nbin):
            # Volatile mass = total mass - core mass
            # For single-element groups (no cores), all mass is volatile
            volatilemass = pc[iz, ibin, ielem] * rmass[ibin, ig]

            if is_ice:
                total_ice = total_ice.at[igas].add(
                    jnp.maximum(volatilemass, DTYPE(0.0))
                )
            else:
                total_liquid = total_liquid.at[igas].add(
                    jnp.maximum(volatilemass, DTYPE(0.0))
                )

    return total_ice, total_liquid
