"""Smoke tests: RHS evaluates, finite, right shapes; one diffeqsolve runs."""
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

from carma_diffrax import DiffraxConfig, diffrax_step
from carma_diffrax.rhs import make_rhs
from carma_diffrax.state import pack

from tests.diffrax._env_builder import make_env_and_state


@pytest.mark.diffrax
def test_rhs_shape_and_finite():
    env, shape, pc0, gc0, T0 = make_env_and_state()
    rhs = make_rhs(env, shape)
    y0 = pack(pc0, gc0, T0)

    dydt = rhs(0.0, y0, None)
    assert dydt.shape == y0.shape
    assert jnp.all(jnp.isfinite(dydt)), "RHS produced non-finite values"


@pytest.mark.diffrax
def test_rhs_zero_when_no_chemistry():
    """With zero pc and zero H2SO4 vapor, all rates should be ~zero."""
    env, shape, pc0, gc0, T0 = make_env_and_state(h2so4_g_per_cm3=0.0,
                                                    M_ug_m3=0.0)
    rhs = make_rhs(env, shape)
    y0 = pack(pc0, gc0, T0)
    dydt = rhs(0.0, y0, None)

    n_pc = shape.n_pc
    dpc = dydt[:n_pc]
    dgc = dydt[n_pc:n_pc + shape.ngas]
    dT = dydt[-1]

    assert float(jnp.max(jnp.abs(dpc))) < 1e-20, "dpc should be near zero"
    assert float(jnp.abs(dgc[1])) < 1e-25, "dgc[H2SO4] should be near zero"
    assert float(jnp.abs(dT)) < 1e-15, f"dT should be near zero, got {dT}"


@pytest.mark.diffrax
def test_one_outer_step_60s():
    """Single 60s integration completes without error and returns finite state."""
    env, shape, pc0, gc0, T0 = make_env_and_state(
        h2so4_g_per_cm3=1e-14, M_ug_m3=1.0,
    )
    cfg = DiffraxConfig(rtol=1e-5, atol=1e-5, max_steps=5_000)
    pc, gc, T, stats = diffrax_step(pc0, gc0, T0, 60.0, env, shape, cfg)

    assert pc.shape == (shape.nbin, shape.nelem)
    assert gc.shape == (shape.ngas,)
    assert jnp.all(jnp.isfinite(pc))
    assert jnp.all(jnp.isfinite(gc))
    assert jnp.isfinite(T)
    assert float(gc[1]) >= 0.0, f"gc[H2SO4] went negative: {gc[1]}"
    assert float(jnp.min(pc)) >= -1e-30, f"pc went negative: {jnp.min(pc)}"
    print(f"\n60 s step stats: {stats}")
