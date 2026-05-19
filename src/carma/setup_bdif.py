"""Brownian diffusion coefficient setup for CARMA-JAX.

Computes dkz(k, i, j) [cm^2/s] using Seinfeld & Pandis (1998) Eq. 8.73
with the Cunningham slip correction factor (bpm) from setup_vf.

Ported from: setupbdif.F90
"""

import jax.numpy as jnp

from carma.constants import BK, PI
from carma.enums import GridType
from carma.precision import DTYPE


def setup_bdif(t, rmu, r_wet, bpm, rprat, zmetl, igridv):
    """Compute Brownian diffusion coefficient at layer boundaries.

    Einstein-Stokes:
        dkz_center = k_B * T * bpm / (6*pi * rmu * r_wet * rprat)

    Then interpolate centers -> boundaries via geometric mean:
        dkz_boundary[k] = sqrt(dkz_center[k-1] * dkz_center[k])

    For non-Cartesian grids, scale by 1 / zmetl^2.

    Args:
        t: Temperature (NZ,) [K].
        rmu: Dynamic viscosity (NZ,) [g/cm/s].
        r_wet: Wet radius (NZ, NBIN, NGROUP) [cm].
        bpm: Cunningham slip correction (NZ, NBIN, NGROUP).
        rprat: Drag ratio for shape (NBIN, NGROUP).
        zmetl: Vertical metric at layer edges (NZ+1,).
        igridv: Grid type (GridType enum).

    Returns:
        dkz: Brownian diffusion coefficient at layer edges (NZ+1, NBIN, NGROUP) [cm^2/s].
    """
    nz = t.shape[0]
    nbin = r_wet.shape[1]
    ngroup = r_wet.shape[2]

    # Layer-center diffusivities (shape NZ, NBIN, NGROUP)
    # broadcast: t(nz,1,1), rmu(nz,1,1), bpm/r_wet(nz,nbin,ngroup), rprat(1,nbin,ngroup)
    t_b = t[:, None, None]
    rmu_b = rmu[:, None, None]
    rprat_b = rprat[None, :, :]
    dkz_center = (BK * t_b * bpm) / (
        DTYPE(6.0) * PI * rmu_b * r_wet * rprat_b
    )  # (NZ, NBIN, NGROUP)

    # Interpolate centers -> boundaries via geometric mean.
    # dkz_boundary[k] = sqrt(dkz_center[k-1] * dkz_center[k])  for k = 1..NZ-1
    # dkz_boundary[0] = dkz_center[0]  (extrapolate bottom)
    # dkz_boundary[NZ] = dkz_center[NZ-1]  (extrapolate top)
    if nz > 1:
        interior = jnp.sqrt(dkz_center[:-1] * dkz_center[1:])  # (NZ-1, NBIN, NGROUP)
        bottom = dkz_center[0:1]  # extrapolate
        top = dkz_center[-1:]
        dkz = jnp.concatenate([bottom, interior, top], axis=0)  # (NZ+1, NBIN, NGROUP)
    else:
        dkz = jnp.concatenate([dkz_center, dkz_center], axis=0)

    # Non-Cartesian: scale by 1/zmetl^2
    if igridv != GridType.I_CART:
        zmetl_b = zmetl[:, None, None]  # (NZ+1, 1, 1)
        dkz = dkz / (zmetl_b ** 2)

    return dkz
