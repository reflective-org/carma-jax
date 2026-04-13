"""US Standard Atmosphere 1976 lookup for testing.

Provides reference atmospheric profiles matching the Fortran test harness.
Ported from: atmosphere_mod.F90 in CARMA tests.
"""

import jax.numpy as jnp

from carma.precision import DTYPE

# Reference values
P0 = DTYPE(1.013250e5)  # Standard sea-level pressure [Pa]
T0_STD = DTYPE(288.15)  # Standard sea-level temperature [K]

# Earth radius for geopotential conversion [km]
_REARTH_KM = DTYPE(6369.0)

# Hydrostatic constant g*M/(R*) [K/km]
_GMR = DTYPE(34.163195)

# 1976 US Standard Atmosphere tables (8 layers)
_HTAB = jnp.array(
    [0.0, 11.0, 20.0, 32.0, 47.0, 51.0, 71.0, 84.852], dtype=DTYPE
)  # Geopotential altitude [km]

_TTAB = jnp.array(
    [288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.946],
    dtype=DTYPE,
)  # Temperature [K]

_PTAB = jnp.array(
    [
        1.0,
        2.233611e-1,
        5.403295e-2,
        8.5666784e-3,
        1.0945601e-3,
        6.6063531e-4,
        3.9046834e-5,
        3.68501e-6,
    ],
    dtype=DTYPE,
)  # Pressure ratio (normalized to sea level)

_GTAB = jnp.array(
    [-6.5, 0.0, 1.0, 2.8, 0.0, -2.8, -2.0, 0.0], dtype=DTYPE
)  # Temperature gradient [K/km]


def _atmosphere_scalar(alt_km):
    """Compute standard atmosphere for a single altitude.

    Args:
        alt_km: Geometric altitude [km].

    Returns:
        Tuple of (pressure_ratio, temperature_ratio).
    """
    # Convert geometric to geopotential altitude
    h = alt_km * _REARTH_KM / (alt_km + _REARTH_KM)

    # Find layer index via linear search (JAX-friendly)
    # Layer i where _HTAB[i] <= h < _HTAB[i+1]
    layer = jnp.searchsorted(_HTAB, h, side="right") - 1
    layer = jnp.clip(layer, 0, 6)  # Clamp to valid range [0, 6]

    tgrad = _GTAB[layer]
    tbase = _TTAB[layer]
    deltah = h - _HTAB[layer]
    tlocal = tbase + tgrad * deltah

    # Temperature ratio
    theta = tlocal / _TTAB[0]

    # Pressure ratio (isothermal vs gradient layer)
    delta_isothermal = _PTAB[layer] * jnp.exp(-_GMR * deltah / tbase)
    delta_gradient = _PTAB[layer] * (tbase / tlocal) ** (_GMR / tgrad)

    # Select based on whether gradient is zero
    delta = jnp.where(jnp.abs(tgrad) < DTYPE(1e-10), delta_isothermal, delta_gradient)

    return delta, theta


def get_standard_atmosphere(z_m):
    """Get standard atmosphere pressure and temperature at given altitudes.

    Args:
        z_m: Altitudes in meters. Can be scalar or 1D array.

    Returns:
        Tuple of (p, t) where:
            p: Pressure [Pa], same shape as z_m.
            t: Temperature [K], same shape as z_m.
    """
    z_km = z_m / DTYPE(1000.0)

    # Vectorize over altitude array
    delta, theta = jnp.vectorize(_atmosphere_scalar)(z_km)

    p = P0 * delta
    t = T0_STD * theta

    return p, t
