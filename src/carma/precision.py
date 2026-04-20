"""Precision control for CARMA-JAX.

DTYPE is selected at first import via the ``CARMA_DTYPE`` environment
variable. Valid values: ``fp64`` (default) and ``fp32``. The matching
``jax_enable_x64`` flag is set automatically so downstream code sees
a consistent precision throughout.

Ported from: carma_precision_mod.F90
"""

import os

import jax
import jax.numpy as jnp


_MODE = os.environ.get("CARMA_DTYPE", "fp64").lower()
if _MODE not in ("fp32", "fp64"):
    raise ValueError(
        f"CARMA_DTYPE must be 'fp32' or 'fp64' (got {_MODE!r})"
    )

_USE_FP64 = (_MODE == "fp64")
jax.config.update("jax_enable_x64", _USE_FP64)

DTYPE = jnp.float64 if _USE_FP64 else jnp.float32

ONE = DTYPE(1.0)
ALMOST_ZERO = jnp.finfo(DTYPE).eps
ALMOST_ONE = ONE - ALMOST_ZERO

# Maximum safe exponent for exp() to avoid overflow.
# Fortran selected_real_kind(15,307) → ~706 (f64).  Fortran float32 limit ≈ 85.
POWMAX = DTYPE(706.0 if _USE_FP64 else 85.0)
