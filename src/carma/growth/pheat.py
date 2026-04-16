"""Particle mass growth rate (dmdt) for CARMA-JAX.

Computes dm/dt including Kelvin curvature effect. Radiative heating
is deferred to a future update.
Ported from: pheat.F90
"""

import jax.numpy as jnp

from carma.precision import DTYPE, POWMAX


def pheat(pc, supsatl, supsati, pvapl, pvapi,
          akelvin, akelvini, gro, gro1,
          rup_wet, rmass, is_ice, iz, igroup, ibin, igas):
    """Compute mass growth rate dm/dt for one bin.

    dm/dt = pvap * (S + 1 - akas) * g0 / (1 + g0*g1*pvap)

    Where:
        pvap = saturation vapor pressure (ice or liquid)
        S = supersaturation
        akas = exp(akelvin/r) = Kelvin curvature factor
        g0 = gro(iz, ibin+1, igroup) = growth kernel
        g1 = gro1(iz, ibin+1, igroup) = conduction term

    Note: ibin is the LEFT bin; growth occurs across the boundary
    between bin ibin and bin ibin+1. Growth kernel is evaluated at
    the boundary (index ibin+1 in Fortran's gro array, which uses
    bin boundaries stored at index i+1).

    Args:
        pc: Particle concentrations (NZ, NBIN, NELEM).
        supsatl: Supersaturation over liquid (NZ, NGAS).
        supsati: Supersaturation over ice (NZ, NGAS).
        pvapl: Saturation vapor pressure liquid (NZ, NGAS) [dyne/cm^2].
        pvapi: Saturation vapor pressure ice (NZ, NGAS) [dyne/cm^2].
        akelvin: Kelvin factor liquid (NZ, NGAS) [cm].
        akelvini: Kelvin factor ice (NZ, NGAS) [cm].
        gro: Growth kernel (NZ, NBIN, NGROUP).
        gro1: Growth conduction (NZ, NBIN, NGROUP).
        rup_wet: Wet radius at upper bin boundary (NZ, NBIN, NGROUP) [cm].
        rmass: Particle mass (NBIN, NGROUP) [g].
        is_ice: Whether this group is ice (bool).
        iz: Vertical level index.
        igroup: Group index.
        ibin: Bin index (0-based; growth across boundary ibin→ibin+1).
        igas: Gas index.

    Returns:
        dmdt: Mass growth rate [g/s], scalar.
    """
    if is_ice:
        pvap = pvapi[iz, igas]
        ss = supsati[iz, igas]
        expon = akelvini[iz, igas] / jnp.maximum(rup_wet[iz, ibin, igroup], DTYPE(1e-30))
    else:
        pvap = pvapl[iz, igas]
        ss = supsatl[iz, igas]
        expon = akelvin[iz, igas] / jnp.maximum(rup_wet[iz, ibin, igroup], DTYPE(1e-30))

    # Clamp exponent to prevent overflow
    expon = jnp.maximum(-POWMAX, expon)
    akas = jnp.exp(expon)

    # Growth kernel at bin boundary (ibin+1 in 0-based indexing)
    # gro array stores values at bin boundaries, indexed by the upper bin
    g0 = gro[iz, ibin, igroup]
    g1 = gro1[iz, ibin, igroup]

    # Mass growth rate [g/s]
    dmdt = pvap * (ss + DTYPE(1.0) - akas) * g0 / (DTYPE(1.0) + g0 * g1 * pvap)

    return dmdt
