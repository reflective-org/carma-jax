"""Shared test fixtures for CARMA-JAX."""

import jax
import jax.numpy as jnp
import pytest

# Enable float64 for all tests
jax.config.update("jax_enable_x64", True)

from carma.precision import DTYPE


@pytest.fixture
def standard_atmosphere_80():
    """80-level standard atmosphere from 0 to 80 km (matching coagtest)."""
    from carma.atmosphere_std import get_standard_atmosphere
    from carma.constants import RM2CGS, RPA2CGS

    nz = 80
    dz_m = 1000.0  # 1 km spacing
    zc_m = jnp.arange(0.5, nz, 1.0) * dz_m  # centers at 500m, 1500m, ...
    zl_m = jnp.arange(0.0, nz + 1, 1.0) * dz_m  # edges at 0, 1000m, ...

    p_pa, t = get_standard_atmosphere(zc_m)
    pl_pa, _ = get_standard_atmosphere(zl_m)

    return {
        "nz": nz,
        "zc": zc_m * RM2CGS,
        "zl": zl_m * RM2CGS,
        "p": p_pa * RPA2CGS,
        "pl": pl_pa * RPA2CGS,
        "t": t,
    }
