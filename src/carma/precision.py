"""Precision control for CARMA-JAX.

Locks the project to float64 throughout. fp32 was evaluated in Phase
6b (PR #15) and dropped: SMALL_PC = 1e-50 underflows below float32's
minimum subnormal (~1.4e-45), silently zeroing denominator floors in
falltest / drydeptest / growtest. Fortran-parity precision is the
reference and the parity bench requires fp64.

Importing this module sets ``jax_enable_x64`` once for the whole
process — without that, JAX silently downcasts everything to float32.

Ported from: carma_precision_mod.F90
"""
import jax
import jax.numpy as jnp


jax.config.update("jax_enable_x64", True)

DTYPE = jnp.float64

ONE = DTYPE(1.0)
ALMOST_ZERO = jnp.finfo(DTYPE).eps
ALMOST_ONE = ONE - ALMOST_ZERO

# Maximum safe exponent for exp() to avoid overflow.
# Fortran selected_real_kind(15,307) → ~706 for float64.
POWMAX = DTYPE(706.0)
