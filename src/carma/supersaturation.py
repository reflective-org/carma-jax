"""Supersaturation calculation for CARMA-JAX.

Computes supersaturation over liquid and ice from gas concentrations
and saturation vapor pressures.
Ported from: supersat.F90
"""

import jax.numpy as jnp

from carma.constants import RGAS
from carma.precision import DTYPE


def supersat(t, gc, pvapl, pvapi, gwtmol, zmet):
    """Compute supersaturation over liquid and ice.

    Args:
        t: Temperature [K], shape (NZ,).
        gc: Gas concentration [g/cm^3/z], shape (NZ,).
        pvapl: Saturation vapor pressure over liquid [dyne/cm^2], shape (NZ,).
        pvapi: Saturation vapor pressure over ice [dyne/cm^2], shape (NZ,).
        gwtmol: Gas molecular weight [g/mol], scalar.
        zmet: Vertical metric, shape (NZ,).

    Returns:
        Tuple of (supsatl, supsati) both shape (NZ,).
    """
    rvap = RGAS / DTYPE(gwtmol)
    gc_cgs = gc / zmet

    supsatl = (gc_cgs * rvap * t - pvapl) / pvapl
    supsati = (gc_cgs * rvap * t - pvapi) / pvapi

    return supsatl, supsati


def supersat_with_cloud(t, gc, pvapl, pvapi, gwtmol, zmet, cldfrc, rhcrit):
    """Compute supersaturation with in-cloud scaling.

    When cloud fraction is present, supersaturation is computed relative
    to the effective saturation including clear-sky fraction.

    Args:
        t: Temperature [K], shape (NZ,).
        gc: Gas concentration [g/cm^3/z], shape (NZ,).
        pvapl: Saturation vapor pressure over liquid [dyne/cm^2], shape (NZ,).
        pvapi: Saturation vapor pressure over ice [dyne/cm^2], shape (NZ,).
        gwtmol: Gas molecular weight [g/mol], scalar.
        zmet: Vertical metric, shape (NZ,).
        cldfrc: Cloud fraction [0-1], shape (NZ,).
        rhcrit: Critical relative humidity, shape (NZ,).

    Returns:
        Tuple of (supsatl, supsati) both shape (NZ,).
    """
    rvap = RGAS / DTYPE(gwtmol)
    gc_cgs = gc / zmet

    alpha = rhcrit * (DTYPE(1.0) - cldfrc) + cldfrc

    supsatl = (gc_cgs * rvap * t - alpha * pvapl) / pvapl
    supsati = (gc_cgs * rvap * t - alpha * pvapi) / pvapi

    # Cap supersaturation for in-cloud conditions
    supsatl = jnp.minimum(supsatl, DTYPE(0.0))
    supsati = jnp.minimum(supsati, (pvapl - alpha * pvapi) / pvapi)

    return supsatl, supsati
