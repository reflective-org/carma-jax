"""Bin structure computation for CARMA-JAX.

Computes particle size bin arrays from minimum radius and mass ratio.
Ported from: setupbins logic in CARMA Fortran source.
"""

import jax.numpy as jnp

from carma.constants import PI
from carma.precision import DTYPE


def setup_bins(rmin, rmrat, nbin, rho):
    """Compute bin structure arrays from minimum radius and mass ratio.

    Args:
        rmin: Minimum particle radius [cm].
        rmrat: Mass ratio between adjacent bins (>1).
        nbin: Number of size bins.
        rho: Particle mass density [g/cm^3] (scalar or array of length nbin).

    Returns:
        Tuple of (r, rmass, vol, dr, dm, rup, rlow, rmassup) where:
            r: Bin center radii [cm], shape (nbin,)
            rmass: Bin center masses [g], shape (nbin,)
            vol: Bin center volumes [cm^3], shape (nbin,)
            dr: Bin widths in radius [cm], shape (nbin,)
            dm: Bin widths in mass [g], shape (nbin,)
            rup: Upper bin boundary radii [cm], shape (nbin,)
            rlow: Lower bin boundary radii [cm], shape (nbin,)
            rmassup: Upper bin boundary masses [g], shape (nbin,)
    """
    rmin = DTYPE(rmin)
    rmrat = DTYPE(rmrat)
    rho = DTYPE(rho)

    # Volume ratio between bins
    vrfact = DTYPE(4.0 / 3.0) * PI

    # Mass of smallest bin
    rmass_min = vrfact * rho * rmin**3

    # Geometric progression of masses
    ibin = jnp.arange(nbin, dtype=DTYPE)
    rmass = rmass_min * rmrat**ibin

    # Radii from masses
    vol = rmass / rho
    r = (vol / vrfact) ** (DTYPE(1.0) / DTYPE(3.0))

    # Bin boundaries (geometric mean of adjacent bin masses)
    # Upper boundary of bin i = geometric mean of rmass[i] and rmass[i+1]
    # For the last bin, extrapolate
    rmassup_interior = jnp.sqrt(rmass[:-1] * rmass[1:])
    rmassup_last = rmass[-1] * jnp.sqrt(rmrat)
    rmassup = jnp.concatenate([rmassup_interior, rmassup_last[None]])

    # Lower boundary of bin i = upper boundary of bin i-1
    # For the first bin, extrapolate
    rlow_first = rmass[0] / jnp.sqrt(rmrat)
    rlow_first = (rlow_first / (rho * vrfact)) ** (DTYPE(1.0) / DTYPE(3.0))
    rlow_interior = (rmassup[:-1] / (rho * vrfact)) ** (DTYPE(1.0) / DTYPE(3.0))
    rlow = jnp.concatenate([rlow_first[None], rlow_interior])

    rup = (rmassup / (rho * vrfact)) ** (DTYPE(1.0) / DTYPE(3.0))

    # Bin widths
    dr = rup - rlow
    dm = rmassup - jnp.concatenate([
        (rmass[0] / jnp.sqrt(rmrat))[None],
        rmassup[:-1],
    ])

    return r, rmass, vol, dr, dm, rup, rlow, rmassup
