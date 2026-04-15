"""Tests for US Standard Atmosphere 1976 lookup."""

import jax.numpy as jnp
import numpy.testing as npt

from carma.atmosphere_std import P0, T0_STD, get_standard_atmosphere


def test_sea_level():
    """Sea level should match reference values exactly."""
    p, t = get_standard_atmosphere(jnp.array([0.0]))
    npt.assert_allclose(float(p[0]), 101325.0, rtol=1e-6)
    npt.assert_allclose(float(t[0]), 288.15, rtol=1e-10)


def test_tropopause():
    """At 11 km geometric, temperature should be near 216.65 K.

    Not exact because geometric != geopotential altitude (differs by ~19m at 11km).
    """
    p, t = get_standard_atmosphere(jnp.array([11000.0]))
    npt.assert_allclose(float(t[0]), 216.65, rtol=1e-3)


def test_stratosphere_isothermal():
    """Between 11-20 km, temperature is constant at 216.65 K."""
    z = jnp.array([12000.0, 15000.0, 18000.0])
    _, t = get_standard_atmosphere(z)
    npt.assert_allclose(t, 216.65, rtol=1e-4)


def test_pressure_decreases_with_altitude():
    z = jnp.linspace(0.0, 70000.0, 20)
    p, _ = get_standard_atmosphere(z)
    assert jnp.all(jnp.diff(p) < 0)


def test_vectorized():
    """Should handle arrays of different sizes."""
    z1 = jnp.array([0.0])
    z10 = jnp.linspace(0.0, 50000.0, 10)
    p1, t1 = get_standard_atmosphere(z1)
    p10, t10 = get_standard_atmosphere(z10)
    assert p1.shape == (1,)
    assert p10.shape == (10,)
    # First element of z10 is 0 = sea level
    npt.assert_allclose(float(p10[0]), float(p1[0]), rtol=1e-10)


def test_high_altitude():
    """Should handle altitudes up to ~85 km without error."""
    z = jnp.array([80000.0])
    p, t = get_standard_atmosphere(z)
    assert jnp.isfinite(p[0])
    assert jnp.isfinite(t[0])
    assert float(p[0]) > 0
    assert float(t[0]) > 0
