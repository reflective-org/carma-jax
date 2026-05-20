"""Mass-conservation tests for each gas independently.

H2SO4: total = gc[H2SO4] + Σ rmass_sulfate · pc_sulfate must be conserved.
H2O:   total = gc[H2O] + Σ rmass_water · pc_water + Σ rmass_ice · pc_ice
       must be conserved.
"""
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

from carma_diffrax import DiffraxConfig
from carma_diffrax.multispecies import diffrax_step_ms

from tests.diffrax.multispecies._env_builder import make_env_and_state


def _total_h2so4(pc, gc, env, ms):
    igroup_sulf = 0
    rmass_sulf = env.rmass_2d[:, igroup_sulf]
    condensed = jnp.sum(rmass_sulf * pc[:, int(ms.ienconc[igroup_sulf])])
    vapor = gc[ms.cfg.igash2so4] / env.zmet[0]
    return condensed + vapor


def _total_h2o(pc, gc, env, ms):
    igroup_cw, igroup_ice = 1, 2
    rmass_cw = env.rmass_2d[:, igroup_cw]
    rmass_ice = env.rmass_2d[:, igroup_ice]
    cond_cw = jnp.sum(rmass_cw * pc[:, int(ms.ienconc[igroup_cw])])
    cond_ice = jnp.sum(rmass_ice * pc[:, int(ms.ienconc[igroup_ice])])
    vapor = gc[ms.cfg.igash2o] / env.zmet[0]
    return cond_cw + cond_ice + vapor


@pytest.mark.diffrax
def test_h2so4_mass_conservation_60s():
    """Sulfate-only seed, 60s step: H2SO4 budget must be conserved."""
    ms, shape, env, pc0, gc0, T0 = make_env_and_state(
        h2so4_g_per_cm3=1e-14, M_sulf_ug_m3=1.0,
    )
    cfg_d = DiffraxConfig(rtol=1e-7, atol=1e-7, max_steps=20_000)
    pc, gc, T, stats = diffrax_step_ms(pc0, gc0, T0, 60.0, env, ms, shape, cfg_d)
    m0 = float(_total_h2so4(pc0, gc0, env, ms))
    m1 = float(_total_h2so4(pc, gc, env, ms))
    drift = abs(m1 - m0) / max(m0, 1e-30)
    print(f"\nH2SO4: m0={m0:.6e} m1={m1:.6e} drift={drift:.2e} stats={stats}")
    assert drift < 1e-6, (
        f"H2SO4 mass drift {drift:.2e} exceeds tol. m0={m0:.6e}, m1={m1:.6e}"
    )


@pytest.mark.diffrax
def test_h2o_mass_conservation_60s_with_water_seed():
    """Cloud-water + ice seed, 60s step: H2O budget must be conserved."""
    ms, shape, env, pc0, gc0, T0 = make_env_and_state(
        T_K=250.0, p_hPa=500.0, rh=0.5,
        h2so4_g_per_cm3=1e-15, M_sulf_ug_m3=0.1,
        M_cw_ug_m3=10.0, M_ice_ug_m3=5.0,
    )
    cfg_d = DiffraxConfig(rtol=1e-7, atol=1e-7, max_steps=20_000)
    pc, gc, T, stats = diffrax_step_ms(pc0, gc0, T0, 60.0, env, ms, shape, cfg_d)
    m0 = float(_total_h2o(pc0, gc0, env, ms))
    m1 = float(_total_h2o(pc, gc, env, ms))
    drift = abs(m1 - m0) / max(m0, 1e-30)
    print(f"\nH2O: m0={m0:.6e} m1={m1:.6e} drift={drift:.2e} stats={stats}")
    assert drift < 1e-6, (
        f"H2O mass drift {drift:.2e} exceeds tol. m0={m0:.6e}, m1={m1:.6e}"
    )
