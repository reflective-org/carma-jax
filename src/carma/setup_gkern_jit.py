"""JIT-compiled growth kernel setup for CARMA-JAX.

Vectorized over bins — no Python loops. Computes gro, gro1, gro2,
akelvin, akelvini, surface tensions, ventilation factors.
Ported from: setupgkern.F90
"""

import jax
import jax.numpy as jnp

from carma.constants import AVG, BK, CP, PI, RGAS, RHO_W, WTMOL_AIR, T0
from carma.precision import DTYPE


@jax.jit
def setup_gkern_jit(t, p, rhoa, zmet, rmu, thcond, diffus, rlhe, rlhm,
                     re, r_wet, rlow_wet, rrat, gwtmol, gstick, tstick,
                     is_ice):
    """Compute growth kernels — vectorized over bins, JIT-compiled.

    Single group, single gas, single level (NZ=1).

    Args:
        t: Temperature (NZ,) [K].
        p: Pressure (NZ,) [dyne/cm^2].
        rhoa: Air density * zmet (NZ,) [g/cm^2/z].
        zmet: Vertical metric (NZ,).
        rmu: Dynamic viscosity (NZ,) [g/cm/s].
        thcond: Thermal conductivity (NZ,) [erg/cm/s/K].
        diffus: Gas diffusivity (NZ, NGAS) [cm^2/s].
        rlhe: Latent heat evaporation (NZ, NGAS) [cm^2/s^2].
        rlhm: Latent heat melting (NZ, NGAS) [cm^2/s^2].
        re: Reynolds number (NZ, NBIN, NGROUP).
        r_wet: Wet radius (NZ, NBIN, NGROUP) [cm].
        rlow_wet: Wet lower boundary radius (NZ, NBIN, NGROUP) [cm].
        rrat: Radius ratio (NBIN, NGROUP).
        gwtmol: Gas molecular weight [g/mol], scalar.
        gstick: Growth sticking coefficient, scalar.
        tstick: Thermal accommodation coefficient, scalar.
        is_ice: Whether group is ice, bool.

    Returns:
        Tuple of (akelvin, akelvini, gro, gro1, gro2) where:
            akelvin: Kelvin factor liquid (NZ, 1) [cm].
            akelvini: Kelvin factor ice (NZ, 1) [cm].
            gro: Growth kernel (NZ, NBIN, 1).
            gro1: Growth conduction (NZ, NBIN, 1).
            gro2: Growth radiation (NZ, 1).
    """
    nz = t.shape[0]
    nbin = r_wet.shape[1]

    rhoa_cgs = rhoa / zmet  # (NZ,)
    D = diffus[:, 0]        # (NZ,) gas diffusivity

    # --- Surface tensions ---
    surfctwa = DTYPE(76.10) - DTYPE(0.155) * (t - T0)    # water-air [dyne/cm]
    surfctia = DTYPE(141.0) - DTYPE(0.15) * t             # ice-air [dyne/cm]

    # --- Kelvin curvature factors ---
    akelvin = (DTYPE(2.0) * gwtmol * surfctwa / (t * RHO_W * RGAS))[:, None]   # (NZ, 1)
    akelvini = (DTYPE(2.0) * gwtmol * surfctia / (t * RHO_W * RGAS))[:, None]  # (NZ, 1)

    # --- Molecular free paths ---
    freep = DTYPE(3.0) * D * jnp.sqrt(PI * gwtmol / (DTYPE(8.0) * RGAS * t))  # (NZ,)
    freept = freep * thcond / (D * rhoa_cgs * (CP - RGAS / (DTYPE(2.0) * WTMOL_AIR)))  # (NZ,)

    # --- Latent heat ---
    rlh = jnp.where(is_ice, rlhe[:, 0] + rlhm[:, 0], rlhe[:, 0])  # (NZ,)

    # --- Vectorized over bins ---
    # Bin boundary radius: rlow_wet (lower boundary of each bin)
    br = rlow_wet[:, :, 0]  # (NZ, NBIN)

    # Knudsen numbers
    rknudn = freep[:, None] / jnp.maximum(br, DTYPE(1e-30))    # (NZ, NBIN)
    rknudnt = freept[:, None] / jnp.maximum(br, DTYPE(1e-30))  # (NZ, NBIN)

    # Lambda corrections
    rlam = ((DTYPE(1.33) * rknudn + DTYPE(0.71)) / (rknudn + DTYPE(1.0))
            + DTYPE(4.0) * (DTYPE(1.0) - gstick) / (DTYPE(3.0) * gstick))
    rlamt = ((DTYPE(1.33) * rknudnt + DTYPE(0.71)) / (rknudnt + DTYPE(1.0))
             + DTYPE(4.0) * (DTYPE(1.0) - tstick) / (DTYPE(3.0) * tstick))

    # Corrected diffusivity and thermal conductivity (sphere: cor=phish=1)
    diffus1 = D[:, None] / (DTYPE(1.0) + rlam * rknudn)           # (NZ, NBIN)
    thcond1 = thcond[:, None] / (DTYPE(1.0) + rlamt * rknudnt)    # (NZ, NBIN)

    # Schmidt and Prandtl numbers
    schn = rmu[:, None] / (rhoa_cgs[:, None] * diffus1)           # (NZ, NBIN)
    prnum = rmu[:, None] * CP / thcond1                           # (NZ, NBIN)

    # Reynolds number (sphere)
    reyn = re[:, :, 0]  # (NZ, NBIN)

    # Ventilation factors
    x1 = schn ** (DTYPE(1.0) / DTYPE(3.0)) * jnp.sqrt(reyn)
    x2 = prnum ** (DTYPE(1.0) / DTYPE(3.0)) * jnp.sqrt(reyn)

    # Ice ventilation
    fv_ice = jnp.where(x1 <= DTYPE(1.0),
                        DTYPE(1.0) + DTYPE(0.14) * x1**2,
                        DTYPE(0.86) + DTYPE(0.28) * x1)
    ft_ice = jnp.where(x2 <= DTYPE(1.0),
                        DTYPE(1.0) + DTYPE(0.14) * x2**2,
                        DTYPE(0.86) + DTYPE(0.28) * x2)

    # Liquid ventilation
    fv_liq = jnp.where(x1 <= DTYPE(1.4),
                        DTYPE(1.0) + DTYPE(0.108) * x1**2,
                        DTYPE(0.78) + DTYPE(0.308) * x1)
    ft_liq = jnp.where(x2 <= DTYPE(1.4),
                        DTYPE(1.0) + DTYPE(0.108) * x2**2,
                        DTYPE(0.78) + DTYPE(0.308) * x2)

    fv = jnp.where(is_ice, fv_ice, fv_liq)  # (NZ, NBIN)
    ft = jnp.where(is_ice, ft_ice, ft_liq)  # (NZ, NBIN)

    # --- Growth kernel coefficients ---
    gro = (DTYPE(4.0) * PI * br * diffus1 * fv * gwtmol
           / (BK * t[:, None] * AVG))                              # (NZ, NBIN)

    gro1 = (gwtmol * rlh[:, None]**2
            / (RGAS * t[:, None]**2 * ft * thcond1)
            / (DTYPE(4.0) * PI * br))                              # (NZ, NBIN)

    gro2 = DTYPE(1.0) / rlh  # (NZ,)

    # Reshape to (NZ, NBIN, 1) for group dimension
    return (akelvin, akelvini,
            gro[:, :, None], gro1[:, :, None], gro2[:, None])
