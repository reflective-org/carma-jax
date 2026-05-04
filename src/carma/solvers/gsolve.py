"""Gas concentration solver for CARMA-JAX.

Forward Euler solver for gas concentrations. Computes gas production
from change in condensate mass (mass conservation).
Ported from: gsolve.F90
"""

import jax.numpy as jnp

from carma.constants import CP
from carma.enums import RC_OK, RC_WARNING_RETRY
from carma.precision import DTYPE


def gsolve(gc, rlprod, previous_ice, previous_liquid, total_ice, total_liquid,
           rlhe, rlhm, rhoa, dtime, iz, ngas, dgc_threshold_arr, scale_threshold):
    """Update gas concentrations from condensate change.

    Gas production = -(change in condensate) / dt.
    Also updates latent heat production rate.

    Args:
        gc: Gas concentrations (NZ, NGAS) [g/cm^3/z].
        rlprod: Latent heat production rate [K/s], scalar for level iz.
        previous_ice: Ice condensate at start of substep (NGAS,).
        previous_liquid: Liquid condensate at start of substep (NGAS,).
        total_ice: Current ice condensate (NGAS,).
        total_liquid: Current liquid condensate (NGAS,).
        rlhe: Latent heat of evaporation (NZ, NGAS) [cm^2/s^2].
        rlhm: Latent heat of melting (NZ, NGAS) [cm^2/s^2].
        rhoa: Air density * zmet (NZ,) [g/cm^2/z].
        dtime: Timestep [s].
        iz: Vertical level index.
        ngas: Number of gas species.
        dgc_threshold_arr: Convergence thresholds per gas (NGAS,).
        scale_threshold: Scaling factor for thresholds.

    Returns:
        Tuple of (gc, rlprod, rc) where rc is return code.
    """
    rc = RC_OK

    for igas in range(ngas):
        # Gas production from condensate change
        gasprod = (
            (previous_ice[igas] - total_ice[igas])
            + (previous_liquid[igas] - total_liquid[igas])
        ) / dtime

        # Update latent heat production rate
        ice_change = previous_ice[igas] - total_ice[igas]
        liq_change = previous_liquid[igas] - total_liquid[igas]
        rlprod = rlprod - (
            ice_change * (rlhe[iz, igas] + rlhm[iz, igas])
            + liq_change * rlhe[iz, igas]
        ) / (CP * rhoa[iz] * dtime)

        # Update gas concentration
        gc = gc.at[iz, igas].add(dtime * gasprod)
        gc_val = gc[iz, igas]

        # If gc went negative, request retry — mass conservation is broken
        # (mirrors gsolve.F90:65-78). This is the dominant retry trigger
        # for sulfate scenarios where nucleation depletes H2SO4 in one
        # substep.
        rc = jnp.where(gc_val < DTYPE(0.0), RC_WARNING_RETRY, rc)

        # Check convergence threshold (only when dgc_threshold > 0;
        # sulfate test has dgc_threshold=0 so this branch is gated off).
        threshold = dgc_threshold_arr[igas] / scale_threshold
        relative_change = jnp.where(
            jnp.abs(gc_val) > DTYPE(1e-50),
            jnp.abs(dtime * gasprod / gc_val),
            DTYPE(0.0),
        )
        rc = jnp.where(
            (threshold > DTYPE(0.0)) & (relative_change > threshold),
            RC_WARNING_RETRY,
            rc,
        )

    return gc, rlprod, rc
