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

    # Constant cpi = 4/3 * PI (mass = cpi * rho * r^3)
    cpi = DTYPE(4.0 / 3.0) * PI

    # Mass of smallest bin
    rmass_min = cpi * rho * rmin**3

    # Geometric progression of masses
    ibin = jnp.arange(nbin, dtype=DTYPE)
    rmass = rmass_min * rmrat**ibin

    # Radii from masses
    vol = rmass / rho
    r = (rmass / rho / cpi) ** (DTYPE(1.0) / DTYPE(3.0))

    # --- Bin boundaries (EXACT Fortran formulas from setupbins.F90) ---
    # rmassup = 2*rmrat/(rmrat+1) * rmass
    rmassup = DTYPE(2.0) * rmrat / (rmrat + DTYPE(1.0)) * rmass

    # dm = 2*(rmrat-1)/(rmrat+1) * rmass
    dm = DTYPE(2.0) * (rmrat - DTYPE(1.0)) / (rmrat + DTYPE(1.0)) * rmass

    # rup from rmassup
    rup = (rmassup / rho / cpi) ** (DTYPE(1.0) / DTYPE(3.0))

    # dr = vrfact * (rmass/rho)^(1/3) where vrfact is from Fortran
    vrfact = ((DTYPE(3.0) / (DTYPE(2.0) * PI * (rmrat + DTYPE(1.0))))
              ** (DTYPE(1.0) / DTYPE(3.0))
              * (rmrat ** (DTYPE(1.0) / DTYPE(3.0)) - DTYPE(1.0)))
    dr = vrfact * (rmass / rho) ** (DTYPE(1.0) / DTYPE(3.0))

    # rlow = rup - dr
    rlow = rup - dr

    return r, rmass, vol, dr, dm, rup, rlow, rmassup
