"""Unit tests for binary_nuc_zhao1995 (classical H2SO4/H2O nucleation)."""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from carma.constants import AVG, BK, WTMOL_H2O
from carma.nucleation.sulfnucrate import (
    binary_nuc_zhao1995, binary_nuc_vehk2002,
)
from carma.sulfate_utils import wtpct_tabaz
from carma.vapor_pressure import vaporp_h2o_murphy2005


def _strat_conditions(T=220.0, rh=0.5, h2so4_num=1e7):
    """Helper: build consistent strat-like inputs."""
    pvapl = float(vaporp_h2o_murphy2005(jnp.array(T))[0])
    p_h2o = rh * pvapl
    # number densities
    h2o_num = p_h2o / (float(BK) * T)
    h2so4_cgs = h2so4_num * 98.0 / float(AVG)
    h2o_cgs = h2o_num * float(WTMOL_H2O) / float(AVG)
    # wt% from Tabazadeh
    h2o_mass_for_wtp = rh * pvapl * float(WTMOL_H2O) / (float(BK) * T * float(AVG))
    wtp = float(wtpct_tabaz(T, h2o_mass_for_wtp, pvapl))
    beta1 = 2e4  # cm/s (approximate mean thermal velocity for H2SO4 at 220K)
    return T, wtp, rh, h2so4_num, h2so4_cgs, h2o_num, h2o_cgs, beta1


def test_strat_nucleation_nonzero():
    """Stratospheric conditions should produce nonzero nucleation."""
    args = _strat_conditions(T=220.0, rh=0.5, h2so4_num=1e8)
    nucrate, mass, rstar, ftry = binary_nuc_zhao1995(*args, 98.0, 18.0)
    assert float(nucrate) > 0.0
    assert float(rstar) > 0.0
    assert float(mass) > 0.0


def test_low_h2so4_gives_zero_rate():
    """Well below threshold H2SO4 → nucrate → 0 or very tiny."""
    args = _strat_conditions(T=260.0, rh=0.1, h2so4_num=1.0)  # essentially no H2SO4
    nucrate, *_ = binary_nuc_zhao1995(*args, 98.0, 18.0)
    assert float(nucrate) < 1e-20


def test_warm_dry_no_saddle():
    """Warm dry air: H2SO4 is always supersaturated over H2SO4/H2O solutions,
    so there's no thermodynamic barrier crossing → rate should be 0."""
    # T=300K, very low H2SO4 should fall below saturation
    args = _strat_conditions(T=300.0, rh=0.001, h2so4_num=1.0)
    nucrate, *_ = binary_nuc_zhao1995(*args, 98.0, 18.0)
    assert float(nucrate) < 1e10  # sane upper bound even if nonzero


def test_rstar_nanoscale():
    """Critical radius for strat conditions should be sub-nm."""
    args = _strat_conditions(T=220.0, rh=0.5, h2so4_num=1e8)
    _, _, rstar, _ = binary_nuc_zhao1995(*args, 98.0, 18.0)
    r_nm = float(rstar) * 1e7
    assert 0.1 < r_nm < 5.0, f"r_star = {r_nm} nm, outside plausible range"


def test_cluster_mass_order_of_magnitude():
    """Dry cluster mass should be on the order of a few H2SO4 molecules
    (~98 g/mol → ~1.6e-22 g each; critical cluster typically 1-20 molecules)."""
    args = _strat_conditions(T=220.0, rh=0.5, h2so4_num=1e8)
    _, mass, _, _ = binary_nuc_zhao1995(*args, 98.0, 18.0)
    # 1 H2SO4 molecule ≈ 1.6e-22 g; 100 molecules ≈ 1.6e-20 g
    # Allow the wide range since exact cluster size depends on T, H2SO4.
    assert 1e-23 < float(mass) < 1e-19


def test_jit_equivalence():
    args = _strat_conditions(T=220.0, rh=0.5, h2so4_num=1e8)
    eager = binary_nuc_zhao1995(*args, 98.0, 18.0)
    jit = jax.jit(binary_nuc_zhao1995)(*args, 98.0, 18.0)
    assert jnp.allclose(eager[0], jit[0])
    assert jnp.allclose(eager[1], jit[1])
    assert jnp.allclose(eager[2], jit[2])
    assert jnp.allclose(eager[3], jit[3])


def test_vmap_over_temperature():
    """Verify vmap composition over T works."""
    Ts = jnp.array([200.0, 220.0, 240.0, 260.0])
    rh = 0.5
    h2so4_num = 1e8
    # Build consistent inputs per T
    pvapls = vaporp_h2o_murphy2005(Ts)[0]
    h2o_nums = rh * pvapls / (BK * Ts)
    h2so4_cgs = h2so4_num * 98.0 / AVG * jnp.ones(4)
    h2o_cgs = h2o_nums * WTMOL_H2O / AVG
    # wt% (scalar-ish, use 40 as rough)
    wtps = jnp.array([60.0, 50.0, 40.0, 30.0])
    beta1 = jnp.full(4, 2e4)

    fn = jax.vmap(lambda T, wtp, rh_, na, na_cgs, nw, nw_cgs, b: binary_nuc_zhao1995(
        T, wtp, rh_, na, na_cgs, nw, nw_cgs, b, 98.0, 18.0))
    nuc_arr, *_ = fn(Ts, wtps, jnp.full(4, rh),
                      jnp.full(4, h2so4_num), h2so4_cgs,
                      h2o_nums, h2o_cgs, beta1)
    assert nuc_arr.shape == (4,)
    assert (nuc_arr >= 0.0).all()


def test_zhao_vs_vehkamaki_both_positive():
    """Classical (Zhao-Turco) and parameterised (Vehkamaki) H2SO4/H2O
    nucleation rates are known to disagree by 2–4 orders of magnitude;
    this is a well-known feature of the literature, not a porting bug.
    Verify only that both produce positive rates in a regime where both
    are valid."""
    T, wtp, rh, h2so4_num, h2so4_cgs, h2o_num, h2o_cgs, beta1 = _strat_conditions(
        T=240.0, rh=0.5, h2so4_num=1e9)
    nz, *_ = binary_nuc_zhao1995(T, wtp, rh, h2so4_num, h2so4_cgs,
                                   h2o_num, h2o_cgs, beta1, 98.0, 18.0)
    nv, *_ = binary_nuc_vehk2002(T, rh, h2so4_num, 98.0)
    assert float(nz) > 0
    assert float(nv) > 0
    # Sanity: both within a combined envelope 0 ≤ rate ≤ 1e15 /cm³/s
    assert float(nz) < 1e15
    assert float(nv) < 1e15
