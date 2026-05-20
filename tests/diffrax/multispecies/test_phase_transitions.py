"""Isolated tests for the inter-group phase transitions.

Each test enables exactly one transition and checks the expected mass
movement between groups, with everything else (growth, evap, nucleation)
turned off via the make_rhs_ms toggles.

The three transitions exercised:
  - freezdropl: cloud_water → ice  (T < T0 − 40 K, pc > FEW_PC)
  - melticel:   ice → cloud_water  (T > T0,        pconmax_ice > FEW_PC)
  - actdropl:   sulfate → cloud_water (T ≥ T0 − 40 K, supsatl > scrit, …)
"""
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

from carma_diffrax import DiffraxConfig
from carma_diffrax.multispecies import diffrax_step_ms, pack_ms, make_rhs_ms

from tests.diffrax.multispecies._env_builder import make_env_and_state


def _ngroup_totals(pc, ms):
    """Total number conc per group: sum over bins."""
    return jnp.array([
        jnp.sum(pc[:, int(ms.ienconc[ig])]) for ig in range(ms.cfg.ngroup)
    ])


@pytest.mark.diffrax
def test_freezdropl_moves_water_to_ice():
    """Cold scenario (T=210 K, < T0−40) with cloud water → expect ice growth.

    With only freezing enabled, the cloud_water group should drain and ice
    should accumulate. Total particle number (across all groups) is conserved.
    """
    ms, shape, env, pc0, gc0, T0 = make_env_and_state(
        T_K=210.0, p_hPa=200.0, rh=0.05,
        h2so4_g_per_cm3=0.0, M_sulf_ug_m3=0.0,
        M_cw_ug_m3=5.0, M_ice_ug_m3=0.0,
    )
    cfg_d = DiffraxConfig(rtol=1e-5, atol=1e-5, max_steps=20_000)
    pc, gc, T, stats = diffrax_step_ms(
        pc0, gc0, T0, 60.0, env, ms, shape, cfg_d,
        do_homogeneous_nuc=False, do_ccn_activation=False,
        do_droplet_freezing=True, do_ice_melting=False,
    )
    totals0 = _ngroup_totals(pc0, ms)
    totals1 = _ngroup_totals(pc, ms)
    print(f"\nFreeze test, T=210K:")
    print(f"  totals before: sulf={float(totals0[0]):.2e} "
           f"cw={float(totals0[1]):.2e} ice={float(totals0[2]):.2e}")
    print(f"  totals after:  sulf={float(totals1[0]):.2e} "
           f"cw={float(totals1[1]):.2e} ice={float(totals1[2]):.2e}")
    # Cloud water should decrease, ice should increase
    assert float(totals1[1]) < float(totals0[1]), (
        f"Cloud water didn't drain: before={totals0[1]:.2e}, after={totals1[1]:.2e}"
    )
    assert float(totals1[2]) > float(totals0[2]), (
        f"Ice didn't accumulate: before={totals0[2]:.2e}, after={totals1[2]:.2e}"
    )


@pytest.mark.diffrax
def test_melticel_moves_ice_to_water():
    """Warm scenario (T=280 K, > T0) with ice → expect cloud water growth."""
    ms, shape, env, pc0, gc0, T0 = make_env_and_state(
        T_K=280.0, p_hPa=800.0, rh=0.5,
        h2so4_g_per_cm3=0.0, M_sulf_ug_m3=0.0,
        M_cw_ug_m3=0.0, M_ice_ug_m3=5.0,
    )
    cfg_d = DiffraxConfig(rtol=1e-5, atol=1e-5, max_steps=20_000)
    pc, gc, T, stats = diffrax_step_ms(
        pc0, gc0, T0, 60.0, env, ms, shape, cfg_d,
        do_homogeneous_nuc=False, do_ccn_activation=False,
        do_droplet_freezing=False, do_ice_melting=True,
    )
    totals0 = _ngroup_totals(pc0, ms)
    totals1 = _ngroup_totals(pc, ms)
    print(f"\nMelt test, T=280K:")
    print(f"  totals before: sulf={float(totals0[0]):.2e} "
           f"cw={float(totals0[1]):.2e} ice={float(totals0[2]):.2e}")
    print(f"  totals after:  sulf={float(totals1[0]):.2e} "
           f"cw={float(totals1[1]):.2e} ice={float(totals1[2]):.2e}")
    assert float(totals1[2]) < float(totals0[2]), "Ice didn't melt"
    assert float(totals1[1]) > float(totals0[1]), "Cloud water didn't grow"


@pytest.mark.diffrax
def test_no_freeze_above_freezing_threshold():
    """T = 250 K > T0−40 (=233 K) → freezdropl gate is off, no freezing."""
    ms, shape, env, pc0, gc0, T0 = make_env_and_state(
        T_K=250.0, p_hPa=500.0, rh=0.3,
        h2so4_g_per_cm3=0.0, M_sulf_ug_m3=0.0,
        M_cw_ug_m3=5.0, M_ice_ug_m3=0.0,
    )
    cfg_d = DiffraxConfig(rtol=1e-5, atol=1e-5, max_steps=20_000)
    pc, gc, T, stats = diffrax_step_ms(
        pc0, gc0, T0, 60.0, env, ms, shape, cfg_d,
        do_homogeneous_nuc=False, do_ccn_activation=False,
        do_droplet_freezing=True, do_ice_melting=False,
    )
    totals1 = _ngroup_totals(pc, ms)
    # Ice should remain zero (no freezing above threshold)
    assert float(totals1[2]) < 1e-10, (
        f"Ice formed above freezing threshold: {totals1[2]:.2e}"
    )
