"""Vertical Brownian diffusion transport rates for CARMA-JAX.

Computes upward/downward diffusion rates at layer edges from dkz.
Ported from: vertdif.F90
"""

import jax.numpy as jnp

from carma.enums import BoundaryCondition, GridType
from carma.precision import ALMOST_ONE, ALMOST_ZERO, DTYPE, POWMAX


def vertdif(dkz_column, dz, rhoa, zmet, igridv, itbnd, ibbnd):
    """Compute vertical diffusion transport rates at layer edges.

    For Cartesian grid with density variation:
        rhofact = log(rho[k]/rho[k-1] * zmet[k-1]/zmet[k])
        xex = rho[k-1]/rho[k] * zmet[k]/zmet[k-1]
        vertdifu[k] = rhofact * dkz[k] / dz[k] / (1 - xex)
        vertdifd[k] = vertdifu[k] * xex

    For sigma/hybrid (no density correction):
        vertdifu = vertdifd = dkz / dz

    Args:
        dkz_column: Diffusion coefficient at layer edges (NZ+1,) [cm^2/s].
            This is a single (bin, group) column.
        dz: Layer thickness (NZ,) [cm].
        rhoa: Air density * zmet (NZ,) [g/cm^2/z].
        zmet: Vertical metric (NZ,).
        igridv: Grid type.
        itbnd, ibbnd: Boundary conditions.

    Returns:
        (vertdifu, vertdifd) both (NZ+1,) [cm/s] — transport rates into
        level k from level k-1.
    """
    nz = dz.shape[0]
    nzm1 = max(1, nz - 1)
    itwo = min(1, nz - 1)  # 0-based Fortran 2

    vertdifu = jnp.zeros(nz + 1, dtype=DTYPE)
    vertdifd = jnp.zeros(nz + 1, dtype=DTYPE)

    # Interior k=1..NZ-1 (0-based) — Fortran k=2..NZ
    if igridv == GridType.I_CART:
        # Vectorized: k from 1 to nz-1 (0-based)
        rhofact = jnp.log(rhoa[1:] / rhoa[:-1] * zmet[:-1] / zmet[1:])
        xex = (rhoa[:-1] / rhoa[1:]) * (zmet[1:] / zmet[:-1])
        # Guard against xex ≈ 1 (would divide by zero)
        denom = jnp.where(jnp.abs(DTYPE(1.0) - xex) < ALMOST_ZERO,
                          DTYPE(1.0) - ALMOST_ONE,
                          DTYPE(1.0) - xex)
        u = (rhofact * dkz_column[1:nz] / dz[1:]) / denom
        d = u * xex
        vertdifu = vertdifu.at[1:nz].set(u)
        vertdifd = vertdifd.at[1:nz].set(d)
    else:  # I_SIG or I_HYBRID
        u = dkz_column[1:nz] / dz[1:]
        vertdifu = vertdifu.at[1:nz].set(u)
        vertdifd = vertdifd.at[1:nz].set(u)

    # Bottom boundary
    is_fixed_b = (ibbnd == int(BoundaryCondition.I_FIXED_CONC))
    if nz >= 2:
        rhofact_b = jnp.log(rhoa[itwo] / rhoa[0])
        ttheta_b = jnp.clip(rhofact_b, -POWMAX, POWMAX)
        xex_b = jnp.exp(-ttheta_b)
        xex_b = jnp.where(jnp.abs(DTYPE(1.0) - xex_b) < ALMOST_ZERO,
                          ALMOST_ONE, xex_b)
        u_b = (rhofact_b * dkz_column[0] / dz[0]) / (DTYPE(1.0) - xex_b)
        d_b = u_b * xex_b
    else:
        u_b = DTYPE(0.0)
        d_b = DTYPE(0.0)
    vertdifu = vertdifu.at[0].set(jnp.where(is_fixed_b, u_b, DTYPE(0.0)))
    vertdifd = vertdifd.at[0].set(jnp.where(is_fixed_b, d_b, DTYPE(0.0)))

    # Top boundary
    is_fixed_t = (itbnd == int(BoundaryCondition.I_FIXED_CONC))
    if nz >= 2:
        rhofact_t = jnp.log(rhoa[nz - 1] / rhoa[nzm1])
        ttheta_t = jnp.clip(rhofact_t, -POWMAX, POWMAX)
        xex_t = jnp.exp(-ttheta_t)
        xex_t = jnp.where(jnp.abs(DTYPE(1.0) - xex_t) < ALMOST_ZERO,
                          ALMOST_ONE, xex_t)
        u_t = (rhofact_t * dkz_column[nz] / dz[nz - 1]) / (DTYPE(1.0) - xex_t)
        d_t = u_t * xex_t
    else:
        u_t = DTYPE(0.0)
        d_t = DTYPE(0.0)
    vertdifu = vertdifu.at[nz].set(jnp.where(is_fixed_t, u_t, DTYPE(0.0)))
    vertdifd = vertdifd.at[nz].set(jnp.where(is_fixed_t, d_t, DTYPE(0.0)))

    return vertdifu, vertdifd
