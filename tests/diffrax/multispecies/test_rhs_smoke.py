"""Smoke tests for the multispecies RHS."""
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

from carma_diffrax.multispecies import make_rhs_ms, pack_ms

from tests.diffrax.multispecies._env_builder import make_env_and_state


@pytest.mark.diffrax
def test_rhs_shape_and_finite():
    ms, shape, env, pc0, gc0, T0 = make_env_and_state()
    rhs = make_rhs_ms(env, ms, shape)
    y0 = pack_ms(pc0, gc0, T0)
    dydt = rhs(0.0, y0, None)
    assert dydt.shape == y0.shape
    assert jnp.all(jnp.isfinite(dydt)), "RHS produced non-finite values"


@pytest.mark.diffrax
def test_rhs_quiescent_with_only_sulfate():
    """With small h2so4 and 1 µg/m³ sulfate, growth/nuc terms should be tiny."""
    ms, shape, env, pc0, gc0, T0 = make_env_and_state(
        h2so4_g_per_cm3=1e-15, M_sulf_ug_m3=1.0,
    )
    rhs = make_rhs_ms(env, ms, shape)
    y0 = pack_ms(pc0, gc0, T0)
    dydt = rhs(0.0, y0, None)
    # All finite; no NaN/inf
    assert jnp.all(jnp.isfinite(dydt))
    # T should change at most slowly with tiny chemistry
    n_pc = shape.n_pc
    dT = dydt[-1]
    assert abs(float(dT)) < 1e-2, f"dT/dt too large: {dT}"
