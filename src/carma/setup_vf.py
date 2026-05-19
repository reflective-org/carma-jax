"""Fall velocity computation for CARMA-JAX.

Fully vectorized over (NZ, NBIN, NGROUP) — no Python loops.
JIT-compilable.
Ported from: setupvf.F90, setupvf_std.F90
"""

import jax
import jax.numpy as jnp

from carma.constants import GRAV, PI, R_AIR
from carma.precision import DTYPE, POWMAX


@jax.jit
def setup_vf_jit(t, rhoa, zmet, rmu, r_wet, rhop_wet, rrat, rprat):
    """Compute fall velocities — fully vectorized, JIT-compiled.

    All computations broadcast over (NZ, NBIN, NGROUP) simultaneously.

    Args:
        t: Temperature (NZ,) [K].
        rhoa: Air density * zmet (NZ,) [g/cm^2/z].
        zmet: Vertical metric (NZ,).
        rmu: Dynamic viscosity (NZ,) [g/cm/s].
        r_wet: Wet particle radius (NZ, NBIN, NGROUP) [cm].
        rhop_wet: Wet particle density (NZ, NBIN, NGROUP) [g/cm^3].
        rrat: Radius ratio for shape (NBIN, NGROUP).
        rprat: Drag ratio for shape (NBIN, NGROUP).

    Returns:
        Tuple of (vf, re, bpm):
            vf: Fall velocity (NZ+1, NBIN, NGROUP) [cm/s].
            re: Reynolds number (NZ, NBIN, NGROUP).
            bpm: Cunningham slip correction (NZ, NBIN, NGROUP).
    """
    nz = t.shape[0]
    nbin = r_wet.shape[1]
    ngroup = r_wet.shape[2]

    rhoa_cgs = rhoa / zmet  # (NZ,)

    # Effective radius: (NZ, NBIN, NGROUP)
    r_eff = r_wet * rrat[None, :, :]

    # Mean free path of air: (NZ,) -> broadcast to (NZ, 1, 1)
    vg = jnp.sqrt(DTYPE(8.0) / PI * R_AIR * t)  # (NZ,)
    rmfp = DTYPE(2.0) * rmu / (rhoa_cgs * vg)    # (NZ,)

    # Knudsen number: (NZ, NBIN, NGROUP)
    rkn = rmfp[:, None, None] / r_eff

    # Cunningham slip correction: (NZ, NBIN, NGROUP)
    expon = jnp.clip(-DTYPE(0.87) / rkn, -POWMAX, POWMAX)
    bpm = DTYPE(1.0) + DTYPE(1.246) * rkn + DTYPE(0.42) * rkn * jnp.exp(expon)

    # Stokes fall velocity: (NZ, NBIN, NGROUP)
    vf_stokes = (
        DTYPE(2.0) / DTYPE(9.0)
        * rhop_wet * r_wet**2 * GRAV * bpm
        / rmu[:, None, None]
        / rprat[None, :, :]
    )

    # Stokes Reynolds number: (NZ, NBIN, NGROUP)
    re_stokes = (
        DTYPE(2.0) * rhoa_cgs[:, None, None]
        * r_wet * rprat[None, :, :] * vf_stokes
        / rmu[:, None, None]
    )

    # Transitional regime (1 <= Re < 1000): P&K y-function
    x = jnp.log(jnp.maximum(re_stokes / bpm, DTYPE(1e-30)))
    y = x * (DTYPE(0.83) - DTYPE(0.013) * x)
    re_trans = jnp.exp(y) * bpm
    vf_trans = re_trans * rmu[:, None, None] / (
        DTYPE(2.0) * r_wet * rprat[None, :, :] * rhoa_cgs[:, None, None]
    )

    # High Re regime (Re >= 1000): Cd = 0.45
    vf_high = bpm * jnp.sqrt(
        DTYPE(8.0) * rhop_wet * r_wet * GRAV
        / (DTYPE(3.0) * DTYPE(0.45) * rhoa_cgs[:, None, None] * rprat[None, :, :]**2)
    )

    # Select regime: (NZ, NBIN, NGROUP)
    vf_center = jnp.where(
        re_stokes < DTYPE(1.0), vf_stokes,
        jnp.where(re_stokes < DTYPE(1000.0), vf_trans, vf_high),
    )
    re_final = jnp.where(
        re_stokes < DTYPE(1.0), re_stokes,
        jnp.where(re_stokes < DTYPE(1000.0), re_trans, re_stokes),
    )

    # Interpolate to layer boundaries (geometric mean)
    # vf has shape (NZ+1, NBIN, NGROUP)
    # Top boundary = copy of top center
    vf_top = vf_center[-1:, :, :]  # (1, NBIN, NGROUP)

    # Interior boundaries: geometric mean of adjacent centers
    # vf_boundary[k] = sqrt(vf_center[k-1] * vf_center[k]) for k=1..NZ-1
    vf_interior = jnp.sqrt(vf_center[:-1, :, :] * vf_center[1:, :, :])  # (NZ-1, NBIN, NGROUP)

    # Bottom boundary: same as first center (no level below)
    vf_bottom = vf_center[:1, :, :]  # (1, NBIN, NGROUP)

    # Assemble: [bottom, interior boundaries, top]
    vf = jnp.concatenate([vf_bottom, vf_interior, vf_top], axis=0)  # (NZ+1, NBIN, NGROUP)

    return vf, re_final, bpm


def setup_vf(config, t, rhoa, zmet, rmu, r_wet, rhop_wet, rrat, rprat):
    """Convenience wrapper matching the original signature."""
    return setup_vf_jit(t, rhoa, zmet, rmu, r_wet, rhop_wet, rrat, rprat)
