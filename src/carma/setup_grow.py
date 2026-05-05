"""Growth parameter setup for CARMA-JAX.

Computes gas diffusivity and latent heats as functions of temperature
and pressure. These are vertically-dependent but time-independent
within a timestep.
Ported from: setupgrow.F90
"""

import jax.numpy as jnp

from carma.constants import AVG, WTMOL_AIR
from carma.precision import DTYPE


def setup_grow(t, p, rhoa, zmet, igash2o, igash2so4, ngas, do_cnst_rlh,
               rlhe_cnst=None, rlhm_cnst=None):
    """Compute gas diffusivity and latent heats.

    Args:
        t: Temperature [K], shape (NZ,).
        p: Pressure [dyne/cm^2], shape (NZ,).
        rhoa: Air density * zmet [g/cm^2/z], shape (NZ,).
        zmet: Vertical metric, shape (NZ,).
        igash2o: Index of H2O gas (-1 if not present).
        igash2so4: Index of H2SO4 gas (-1 if not present).
        ngas: Number of gas species.
        do_cnst_rlh: Use constant latent heats.
        rlhe_cnst: Constant latent heat of evaporation [cm^2/s^2] (if do_cnst_rlh).
        rlhm_cnst: Constant latent heat of melting [cm^2/s^2] (if do_cnst_rlh).

    Returns:
        Tuple of (diffus, rlhe, rlhm) where:
            diffus: Gas diffusivity [cm^2/s], shape (NZ, NGAS).
            rlhe: Latent heat of evaporation [cm^2/s^2], shape (NZ, NGAS).
            rlhm: Latent heat of melting [cm^2/s^2], shape (NZ, NGAS).
    """
    nz = t.shape[0]
    diffus = jnp.zeros((nz, ngas), dtype=DTYPE)
    rlhe = jnp.zeros((nz, ngas), dtype=DTYPE)
    rlhm = jnp.zeros((nz, ngas), dtype=DTYPE)

    if igash2o >= 0:
        # H2O diffusivity: D = 0.211 * (P0/P) * (T/T0)^1.94
        diffus_h2o = (
            DTYPE(0.211)
            * (DTYPE(1.01325e6) / p)
            * (t / DTYPE(273.15)) ** DTYPE(1.94)
        )
        diffus = diffus.at[:, igash2o].set(diffus_h2o)

        if do_cnst_rlh:
            rlhe = rlhe.at[:, igash2o].set(DTYPE(rlhe_cnst))
            rlhm = rlhm.at[:, igash2o].set(DTYPE(rlhm_cnst))
        else:
            # Temperature-dependent latent heat of evaporation
            rlhe_h2o = (
                DTYPE(2.5) - DTYPE(0.00239) * (t - DTYPE(273.16))
            ) * DTYPE(1e10)
            rlhe = rlhe.at[:, igash2o].set(rlhe_h2o)

            # Temperature-dependent latent heat of melting
            rlhm_h2o = (
                DTYPE(79.7)
                + DTYPE(0.485) * (t - DTYPE(273.16))
                - DTYPE(2.5e-3) * (t - DTYPE(273.16)) ** 2
            ) * DTYPE(4.186e7)
            rlhm = rlhm.at[:, igash2o].set(rlhm_h2o)

    if igash2so4 >= 0:
        # H2SO4 diffusivity
        rhoa_cgs = rhoa / zmet
        aden = rhoa_cgs * AVG / WTMOL_AIR
        diffus_h2so4 = DTYPE(1.76575e17) * jnp.sqrt(t) / aden
        diffus = diffus.at[:, igash2so4].set(diffus_h2so4)

        # H2SO4 latent heats. Fortran's setupgrow.F90:91-92 carries a
        # commented "HACK": both rlhe AND rlhm of H2SO4 are set to
        # rlhe(igash2o) — the second line uses rlhe, not rlhm. Mirror
        # this exactly; the previous JAX port used rlhm[igash2o] for the
        # second line which under-predicts rlhm[H2SO4] by ~12× at
        # stratospheric temps and breaks tsolve parity downstream.
        if igash2o >= 0:
            rlhe = rlhe.at[:, igash2so4].set(rlhe[:, igash2o])
            rlhm = rlhm.at[:, igash2so4].set(rlhe[:, igash2o])

    return diffus, rlhe, rlhm
