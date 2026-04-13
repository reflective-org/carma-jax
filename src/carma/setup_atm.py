"""Atmospheric property setup for CARMA-JAX.

Computes air density, viscosity, thermal conductivity, and vertical metrics
from temperature, pressure, and altitude profiles.
Ported from: setupatm.F90
"""

import jax.numpy as jnp

from carma.constants import GRAV, R_AIR, T0
from carma.enums import GridType
from carma.precision import DTYPE

# Sutherland's equation constants for air viscosity
_RMU_0 = DTYPE(1.8325e-4)  # Reference viscosity [g/cm/s]
_RMU_T0 = DTYPE(296.16)  # Reference temperature [K]
_RMU_C = DTYPE(120.0)  # Sutherland constant [K]
_RMU_CONST = _RMU_0 * (_RMU_T0 + _RMU_C)


def setup_atm(t, p, pl, zc, zl, igridv):
    """Compute atmospheric properties from thermodynamic state.

    All inputs must be in CGS units (pressure in dyne/cm^2, altitude in cm).

    Args:
        t: Temperature at layer centers [K], shape (NZ,).
        p: Pressure at layer centers [dyne/cm^2], shape (NZ,).
        pl: Pressure at layer edges [dyne/cm^2], shape (NZ+1,).
        zc: Altitude at layer centers [cm], shape (NZ,).
        zl: Altitude at layer edges [cm], shape (NZ+1,).
        igridv: Grid type (GridType enum value).

    Returns:
        Tuple of (rhoa, dz, zmet, zmetl, rmu, thcond, rhoa_wet) where:
            rhoa: Air density scaled by zmet [g/cm^2/z], shape (NZ,).
            dz: Layer thickness [cm], shape (NZ,).
            zmet: Vertical metric ds/dz at centers, shape (NZ,).
            zmetl: Vertical metric at edges, shape (NZ+1,).
            rmu: Dynamic viscosity [g/cm/s], shape (NZ,).
            thcond: Thermal conductivity [erg/cm/s/K], shape (NZ,).
            rhoa_wet: Wet air density [g/cm^3], shape (NZ,).
    """
    nz = t.shape[0]

    # Dry air density [g/cm^3]
    rhoa = p / (R_AIR * t)

    # Layer thickness [cm]
    dz = jnp.abs(zl[1:] - zl[:-1])

    # Vertical metrics
    if igridv == GridType.I_CART:
        zmet = jnp.ones(nz, dtype=DTYPE)
    else:
        # Sigma or Hybrid coordinates
        zmet = jnp.abs((pl[:-1] - pl[1:]) / (zl[:-1] - zl[1:])) / (GRAV * rhoa)

    # Interpolate zmet to layer edges
    if nz == 1:
        zmetl = jnp.full(nz + 1, zmet[0], dtype=DTYPE)
    else:
        # Extrapolate bottom edge
        zmetl_0 = zmet[0] + (zmet[1] - zmet[0]) / (zc[1] - zc[0]) * (zl[0] - zc[0])
        # Extrapolate top edge
        zmetl_top = (
            zmet[-1]
            + (zmet[-1] - zmet[-2]) / (zc[-1] - zc[-2]) * (zl[-1] - zc[-1])
        )
        # Interpolate interior edges
        zmetl_interior = (
            zmet[:-1]
            + (zmet[1:] - zmet[:-1]) / (zc[1:] - zc[:-1]) * (zl[1:-1] - zc[:-1])
        )
        zmetl = jnp.concatenate([zmetl_0[None], zmetl_interior, zmetl_top[None]])

    # Scale density by vertical metric
    rhoa = rhoa * zmet

    # Wet air density from hydrostatic balance [g/cm^3]
    rhoa_wet = jnp.abs(pl[1:] - pl[:-1]) / GRAV / dz

    # Air viscosity via Sutherland's equation [g/cm/s]
    rmu = _RMU_CONST / (t + _RMU_C) * (t / _RMU_T0) ** DTYPE(1.5)

    # Thermal conductivity [erg/cm/s/K]
    thcond = (DTYPE(5.69) + DTYPE(0.017) * (t - T0)) * DTYPE(4.186e2)

    return rhoa, dz, zmet, zmetl, rmu, thcond, rhoa_wet
