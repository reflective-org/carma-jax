"""Precision control for CARMA-JAX.

Ported from: carma_precision_mod.F90
"""

import jax.numpy as jnp

# Default: double precision (matching Fortran selected_real_kind(15,307))
DTYPE = jnp.float64

# Machine epsilon for the selected precision
ONE = DTYPE(1.0)
ALMOST_ZERO = jnp.finfo(DTYPE).eps
ALMOST_ONE = ONE - ALMOST_ZERO

# Maximum safe exponent for exp() to avoid overflow
# Double precision: 706.0 (from Fortran)
# Single precision would be: 85.0
POWMAX = DTYPE(706.0)
