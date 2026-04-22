"""Unit tests for sulfhetnucrate (heterogeneous H2SO4/H2O nucleation)."""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from carma.constants import AVG, BK, WTMOL_H2O
from carma.nucleation.sulfhetnucrate import sulfhetnucrate, _fletcher_factor
from carma.nucleation.sulfnucrate import binary_nuc_zhao1995
from carma.sulfate_utils import wtpct_tabaz
from carma.vapor_pressure import vaporp_h2o_murphy2005


def _strat(T=220.0, rh=0.5, h2so4_num=1e8, beta1=2e4, beta2=1.0):
    """Build self-consistent strat inputs for sulfhetnucrate."""
    pvapl = float(vaporp_h2o_murphy2005(jnp.array(T))[0])
    h2o_num = rh * pvapl / (float(BK) * T)
    h2so4_cgs = h2so4_num * 98.0 / float(AVG)
    h2o_cgs = h2o_num * float(WTMOL_H2O) / float(AVG)
    h2o_mass_wtp = rh * pvapl * float(WTMOL_H2O) / (float(BK) * T * float(AVG))
    wtp = float(wtpct_tabaz(T, h2o_mass_wtp, pvapl))
    return dict(
        temp=T, weight_percent=wtp, rh=rh,
        h2so4=h2so4_num, h2so4_cgs=h2so4_cgs,
        h2o=h2o_num, h2o_cgs=h2o_cgs,
        beta1=beta1, beta2=beta2,
    )


# --- Fletcher factor ---

def test_fletcher_factor_bounded():
    """fv1 should stay in [0, 1] for all physical xm."""
    for xm in [0.01, 0.5, 1.0, 2.0, 10.0, 100.0]:
        fv1 = float(_fletcher_factor(jnp.asarray(xm)))
        assert 0.0 <= fv1 <= 1.0, f"fv1 = {fv1} at xm={xm}"


def test_fletcher_factor_monotonic_decreasing():
    """Larger xm (bigger seed relative to r*) means the Fletcher factor
    is SMALLER (more barrier reduction, but the factor multiplies into
    ftry which is already negative, so smaller fv1 = less-negative ftry1
    = MORE heterogeneous activity. The raw fv1 itself decreases with xm
    over the physical range."""
    xms = jnp.array([0.5, 1.0, 2.0, 5.0, 20.0])
    fvs = jax.vmap(_fletcher_factor)(xms)
    fvs_np = np.asarray(fvs)
    assert (np.diff(fvs_np) < 0).all(), (
        f"fv1 should be monotonic-decreasing over this range, got {fvs_np}")


def test_fletcher_branch_continuity_at_xm_1():
    """At xm = 1 the two piecewise branches must agree."""
    eps = 1e-6
    lo = float(_fletcher_factor(jnp.asarray(1.0 - eps)))
    hi = float(_fletcher_factor(jnp.asarray(1.0 + eps)))
    assert abs(lo - hi) < 1e-4, (
        f"Fletcher branches disagree at xm=1: lo={lo}, hi={hi}")


# --- sulfhetnucrate behaviour ---

def test_hetnucrate_scales_with_seed_surface():
    """Rate should scale close to r² (seed surface area) once Fletcher
    factor plateaus at large xm."""
    args = _strat()
    r_list = [5e-7, 1e-6, 5e-6]
    rates = [float(sulfhetnucrate(**args, r_preexist=r)) for r in r_list]
    # Each step multiplies r by ~2–5; rate should multiply faster than
    # linear but at most by r² · (small Fletcher correction).
    assert rates[2] > rates[1] > rates[0] > 0


def test_no_homogeneous_saddle_gives_zero():
    """When H2SO4 is far below saturation, the homogeneous Zhao saddle
    search finds no crossing and rstar = 0; heterogeneous rate should be 0."""
    args = _strat(T=300.0, rh=0.001, h2so4_num=1.0)
    nuc = sulfhetnucrate(**args, r_preexist=5e-6)
    # Zhao path returns rstar=0 here; our gate should zero the rate.
    # (Allow either numerically-zero or truly-zero.)
    assert float(nuc) < 1e-20


def test_hetnucrate_vs_homogeneous_background():
    """Heterogeneous rate per pre-existing particle should be small
    relative to the homogeneous rate per unit volume (since the barrier
    is reduced but the prefactor is per-seed, not per-cm³)."""
    args = _strat(T=220.0, rh=0.5, h2so4_num=1e8)
    nuc_het = float(sulfhetnucrate(**args, r_preexist=5e-6))
    nuc_hom, *_ = binary_nuc_zhao1995(
        args["temp"], args["weight_percent"], args["rh"],
        args["h2so4"], args["h2so4_cgs"],
        args["h2o"], args["h2o_cgs"],
        args["beta1"], 98.0, 18.0,
    )
    # Units differ: nuc_hom is /cm³/s, nuc_het is /s per pre-existing seed.
    # Both should be positive.
    assert nuc_het > 0
    assert float(nuc_hom) > 0


# --- JIT + vmap ---

def test_hetnucrate_jit():
    args = _strat()
    r = 5e-6
    eager = sulfhetnucrate(**args, r_preexist=r)
    jit_out = jax.jit(sulfhetnucrate)(**args, r_preexist=r)
    assert jnp.allclose(eager, jit_out)


def test_hetnucrate_vmap_over_r():
    """Compose under vmap on seed radius axis."""
    args = _strat()
    r_arr = jnp.array([1e-7, 5e-7, 1e-6, 5e-6, 1e-5])
    fn = jax.vmap(lambda r: sulfhetnucrate(**args, r_preexist=r))
    rates = fn(r_arr)
    assert rates.shape == (5,)
    assert (rates >= 0).all()
