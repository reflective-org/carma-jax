"""Fall velocity computation for CARMA-JAX.

Computes particle fall velocities, Reynolds numbers, and Cunningham slip
correction factors. Supports standard spherical, non-spherical shape,
and Heymsfield 2010 ice crystal routines.
Ported from: setupvf.F90, setupvf_std.F90
"""

import jax.numpy as jnp

from carma.constants import GRAV, PI, R_AIR
from carma.precision import DTYPE, POWMAX


def setup_vf(config, t, rhoa, zmet, rmu, r_wet, rhop_wet, rrat, rprat):
    """Compute fall velocities and related quantities.

    Args:
        config: CarmaConfig.
        t: Temperature (NZ,) [K].
        rhoa: Air density * zmet (NZ,) [g/cm^2/z].
        zmet: Vertical metric (NZ,).
        rmu: Dynamic viscosity (NZ,) [g/cm/s].
        r_wet: Wet particle radius (NZ, NBIN, NGROUP) [cm].
        rhop_wet: Wet particle density (NZ, NBIN, NGROUP) [g/cm^3].
        rrat: Radius ratio for shape (NBIN, NGROUP).
        rprat: Drag ratio for shape (NBIN, NGROUP).

    Returns:
        Tuple of (vf, re, bpm) where:
            vf: Fall velocity (NZ+1, NBIN, NGROUP) [cm/s].
            re: Reynolds number (NZ, NBIN, NGROUP).
            bpm: Cunningham slip correction (NZ, NBIN, NGROUP).
    """
    nz = t.shape[0]
    nbin = config.nbin
    ngroup = config.ngroup

    vf = jnp.zeros((nz + 1, nbin, ngroup), dtype=DTYPE)
    re = jnp.zeros((nz, nbin, ngroup), dtype=DTYPE)
    bpm = jnp.zeros((nz, nbin, ngroup), dtype=DTYPE)

    rhoa_cgs = rhoa / zmet  # (NZ,)

    for ig in range(ngroup):
        for i in range(nbin):
            # Mean free path of air molecules
            vg = jnp.sqrt(DTYPE(8.0) / PI * R_AIR * t)  # thermal velocity (NZ,)
            rmfp = DTYPE(2.0) * rmu / (rhoa_cgs * vg)  # mean free path (NZ,)

            # Knudsen number
            r_eff = r_wet[:, i, ig] * rrat[i, ig]
            rkn = rmfp / r_eff  # (NZ,)

            # Cunningham slip correction
            expon = jnp.clip(-DTYPE(0.87) / rkn, -POWMAX, POWMAX)
            bpm_val = DTYPE(1.0) + (
                DTYPE(1.246) * rkn + DTYPE(0.42) * rkn * jnp.exp(expon)
            )
            bpm = bpm.at[:, i, ig].set(bpm_val)

            # Stokes regime fall velocity
            vf_stokes = (
                DTYPE(2.0)
                / DTYPE(9.0)
                * rhop_wet[:, i, ig]
                * r_wet[:, i, ig] ** 2
                * GRAV
                * bpm_val
                / rmu
                / rprat[i, ig]
            )

            # Reynolds number
            re_val = (
                DTYPE(2.0)
                * rhoa_cgs
                * r_wet[:, i, ig]
                * rprat[i, ig]
                * vf_stokes
                / rmu
            )

            # Use Stokes for Re < 1; for larger Re, apply corrections
            # Transitional regime (1 <= Re < 1000)
            x = jnp.log(jnp.maximum(re_val / bpm_val, DTYPE(1e-30)))
            y = x * (DTYPE(0.83) - DTYPE(0.013) * x)
            re_trans = jnp.exp(y) * bpm_val
            vf_trans = re_trans * rmu / (
                DTYPE(2.0) * r_wet[:, i, ig] * rprat[i, ig] * rhoa_cgs
            )

            # High Re regime (Re >= 1000)
            cdrag = DTYPE(0.45)
            vf_high = bpm_val * jnp.sqrt(
                DTYPE(8.0)
                * rhop_wet[:, i, ig]
                * r_wet[:, i, ig]
                * GRAV
                / (DTYPE(3.0) * cdrag * rhoa_cgs * rprat[i, ig] ** 2)
            )

            # Select regime
            vf_val = jnp.where(
                re_val < DTYPE(1.0),
                vf_stokes,
                jnp.where(re_val < DTYPE(1000.0), vf_trans, vf_high),
            )
            re_final = jnp.where(
                re_val < DTYPE(1.0),
                re_val,
                jnp.where(re_val < DTYPE(1000.0), re_trans, re_val),
            )

            vf = vf.at[:nz, i, ig].set(vf_val)
            re = re.at[:, i, ig].set(re_final)

    # Interpolate to layer boundaries (geometric mean)
    vf = vf.at[nz, :, :].set(vf[nz - 1, :, :])
    if nz > 1:
        vf = vf.at[nz - 1, :, :].set(
            jnp.sqrt(vf[nz - 2, :, :] * vf[nz - 1, :, :])
        )
        for iz in range(nz - 2, 0, -1):
            vf = vf.at[iz, :, :].set(
                jnp.sqrt(vf[iz - 1, :, :] * vf[iz, :, :])
            )

    return vf, re, bpm
