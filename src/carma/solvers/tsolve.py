"""Temperature solver for CARMA-JAX.

Forward Euler solver for temperature from latent heat release
and particle radiative heating.
Ported from: tsolve.F90
"""

import jax.numpy as jnp

from carma.enums import RC_OK, RC_WARNING_RETRY
from carma.precision import DTYPE


def tsolve(t, rlheat, partheat, rlprod, phprod, dtime, iz,
           dt_threshold, scale_threshold):
    """Update temperature from latent heat and particle heating.

    Args:
        t: Temperature (NZ,) [K].
        rlheat: Latent heating accumulator (NZ,) [K].
        partheat: Particle heating accumulator (NZ,) [K].
        rlprod: Latent heat production rate [K/s].
        phprod: Particle heat production rate [K/s].
        dtime: Timestep [s].
        iz: Vertical level index.
        dt_threshold: Temperature convergence threshold [K].
        scale_threshold: Scaling factor for threshold.

    Returns:
        Tuple of (t, rlheat, partheat, rc).
    """
    rc = RC_OK

    # Temperature change from latent heat
    dt_val = dtime * rlprod

    # Accumulate latent heating
    rlheat = rlheat.at[iz].add(rlprod * dtime)

    # Add particle heating
    dt_val = dt_val + dtime * phprod
    partheat = partheat.at[iz].add(phprod * dtime)

    # Update temperature
    t = t.at[iz].add(dt_val)

    # Check convergence threshold
    threshold = dt_threshold / scale_threshold
    rc = jnp.where(
        (threshold > DTYPE(0.0)) & (jnp.abs(dt_val) > threshold),
        RC_WARNING_RETRY,
        rc,
    )

    return t, rlheat, partheat, rc
