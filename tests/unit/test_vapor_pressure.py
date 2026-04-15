"""Tests for vapor pressure routines."""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy.testing as npt

from carma.vapor_pressure import (
    vaporp_h2o_buck1981,
    vaporp_h2o_murphy2005,
    vaporp_h2o_goff1946,
)
from carma.precision import DTYPE


def test_buck_at_triple_point():
    """At T=273.16K, vapor pressure ~611 (Fortran units, Pa-equivalent)."""
    pvapl, pvapi = vaporp_h2o_buck1981(DTYPE(273.16))
    # Buck constants produce values in Pa (matching Fortran exactly)
    npt.assert_allclose(float(pvapl), 611.21, rtol=0.01)
    npt.assert_allclose(float(pvapi), 611.15, rtol=0.01)


def test_buck_liquid_gt_ice_below_freezing():
    """Below 273.16K, liquid vapor pressure exceeds ice (Bergeron process)."""
    pvapl, pvapi = vaporp_h2o_buck1981(DTYPE(260.0))
    # Supercooled liquid has higher vapor pressure than ice
    assert float(pvapl) > float(pvapi)


def test_buck_increases_with_temperature():
    t = jnp.array([250.0, 270.0, 290.0, 310.0], dtype=DTYPE)
    pvapl, pvapi = vaporp_h2o_buck1981(t)
    assert jnp.all(jnp.diff(pvapl) > 0)
    assert jnp.all(jnp.diff(pvapi) > 0)


def test_murphy_at_triple_point():
    pvapl, pvapi = vaporp_h2o_murphy2005(DTYPE(273.16))
    npt.assert_allclose(float(pvapl), 6110.0, rtol=0.02)
    npt.assert_allclose(float(pvapi), 6110.0, rtol=0.02)


def test_murphy_increases_with_temperature():
    t = jnp.array([200.0, 230.0, 260.0, 290.0], dtype=DTYPE)
    pvapl, pvapi = vaporp_h2o_murphy2005(t)
    assert jnp.all(jnp.diff(pvapl) > 0)
    assert jnp.all(jnp.diff(pvapi) > 0)


def test_goff_at_triple_point():
    pvapl, pvapi = vaporp_h2o_goff1946(DTYPE(273.16))
    npt.assert_allclose(float(pvapl), 6110.0, rtol=0.02)
    npt.assert_allclose(float(pvapi), 6110.0, rtol=0.02)


def test_goff_increases_with_temperature():
    t = jnp.array([200.0, 240.0, 280.0, 320.0], dtype=DTYPE)
    pvapl, pvapi = vaporp_h2o_goff1946(t)
    assert jnp.all(jnp.diff(pvapl) > 0)
    assert jnp.all(jnp.diff(pvapi) > 0)


def test_murphy_goff_agree_at_300K():
    """Murphy and Goff should agree within ~2% at 300K (both in dyne/cm^2)."""
    t = DTYPE(300.0)
    p_murphy, _ = vaporp_h2o_murphy2005(t)
    p_goff, _ = vaporp_h2o_goff1946(t)
    npt.assert_allclose(float(p_murphy), float(p_goff), rtol=0.02)


def test_buck_is_10x_smaller():
    """Buck outputs in Pa, Murphy/Goff in dyne/cm^2 (10x larger)."""
    t = DTYPE(300.0)
    p_buck, _ = vaporp_h2o_buck1981(t)
    p_murphy, _ = vaporp_h2o_murphy2005(t)
    # Buck ~ Pa, Murphy ~ dyne/cm^2, ratio should be ~10
    ratio = float(p_murphy / p_buck)
    npt.assert_allclose(ratio, 10.0, rtol=0.02)


def test_vectorized():
    """All routines should handle arrays."""
    t = jnp.linspace(200.0, 350.0, 20, dtype=DTYPE)
    for fn in [vaporp_h2o_buck1981, vaporp_h2o_murphy2005, vaporp_h2o_goff1946]:
        pvapl, pvapi = fn(t)
        assert pvapl.shape == (20,)
        assert pvapi.shape == (20,)
        assert jnp.all(jnp.isfinite(pvapl))
        assert jnp.all(jnp.isfinite(pvapi))
