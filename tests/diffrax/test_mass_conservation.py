"""Total-sulfur mass conservation across one diffrax outer step.

The diffrax RHS is constructed so that dgc/dt = -d(particle_mass)/dt exactly.
This test checks that the diffeqsolve integration honors that invariant
within its own tolerance budget.
"""
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from carma_diffrax import DiffraxConfig, diffrax_step

from tests.diffrax._env_builder import make_env_and_state


def _total_sulfur_mass(pc, gc, env):
    """Total sulfur mass per cm^3 = condensed mass + H2SO4 vapor mass."""
    rmass_g = env.rmass[:, 0]                       # (nbin,)
    condensed = jnp.sum(rmass_g * pc[:, 0])         # g/cm^3
    vapor = gc[1] / env.zmet[0]                     # gc[H2SO4] / zmet → g/cm^3
    return condensed + vapor


@pytest.mark.diffrax
@pytest.mark.parametrize("dtime", [60.0, 600.0, 1800.0])
def test_total_sulfur_conserved_within_outer_step(dtime):
    env, shape, pc0, gc0, T0 = make_env_and_state(
        h2so4_g_per_cm3=1e-14, M_ug_m3=1.0,
    )
    cfg = DiffraxConfig(rtol=1e-7, atol=1e-7, max_steps=20_000)
    pc, gc, T, stats = diffrax_step(pc0, gc0, T0, dtime, env, shape, cfg)

    m0 = float(_total_sulfur_mass(pc0, gc0, env))
    m1 = float(_total_sulfur_mass(pc, gc, env))
    drift = abs(m1 - m0) / max(m0, 1e-30)

    print(f"\ndt={dtime}: m0={m0:.6e} m1={m1:.6e} "
          f"drift={drift:.2e} stats={stats}")
    assert drift < 1e-6, (
        f"Mass drift {drift:.2e} exceeds tolerance at dt={dtime}s. "
        f"m0={m0:.6e}, m1={m1:.6e}"
    )


@pytest.mark.diffrax
def test_positivity_preserved_over_1800s():
    """gc >= 0 and pc >= 0 over a full 1800 s outer step.

    This is the headline behavior we're trying to add — the explicit-Euler
    semi-implicit scheme in the faithful port fails this for 7 of 100
    realistic-ensemble scenarios at this dtime.
    """
    env, shape, pc0, gc0, T0 = make_env_and_state(
        T_K=240.0, p_hPa=200.0, rh=0.3,
        h2so4_g_per_cm3=5e-13, M_ug_m3=1.0,
    )
    cfg = DiffraxConfig(rtol=1e-5, atol=1e-5, max_steps=20_000)
    pc, gc, T, stats = diffrax_step(pc0, gc0, T0, 1800.0, env, shape, cfg)

    print(f"\n1800s step stats: {stats}")
    assert float(gc[1]) >= 0.0, f"gc[H2SO4] went negative: {gc[1]:.3e}"
    assert float(jnp.min(pc)) >= -1e-30, (
        f"pc went negative: {float(jnp.min(pc)):.3e}"
    )
