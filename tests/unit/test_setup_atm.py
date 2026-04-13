"""Tests for atmospheric property setup."""

import jax.numpy as jnp
import numpy.testing as npt

from carma.atmosphere_std import get_standard_atmosphere
from carma.constants import GRAV, R_AIR, RM2CGS, RPA2CGS, T0
from carma.enums import GridType
from carma.precision import DTYPE
from carma.setup_atm import setup_atm


def _make_3level_atm():
    """Create a simple 3-level Cartesian atmosphere for testing."""
    z_m = jnp.array([500.0, 1500.0, 2500.0])
    zl_m = jnp.array([0.0, 1000.0, 2000.0, 3000.0])
    p_pa, t = get_standard_atmosphere(z_m)
    pl_pa, _ = get_standard_atmosphere(zl_m)
    return t, p_pa * RPA2CGS, pl_pa * RPA2CGS, z_m * RM2CGS, zl_m * RM2CGS


def test_density_ideal_gas():
    """rhoa should satisfy ideal gas law (before zmet scaling)."""
    t, p, pl, zc, zl = _make_3level_atm()
    rhoa, dz, zmet, zmetl, rmu, thcond, rhoa_wet = setup_atm(
        t, p, pl, zc, zl, GridType.I_CART
    )
    # For Cartesian, zmet=1, so rhoa = p / (R_AIR * t)
    expected = p / (R_AIR * t)
    npt.assert_allclose(rhoa, expected, rtol=1e-14)


def test_cartesian_zmet_is_one():
    t, p, pl, zc, zl = _make_3level_atm()
    _, _, zmet, _, _, _, _ = setup_atm(t, p, pl, zc, zl, GridType.I_CART)
    npt.assert_allclose(zmet, 1.0, rtol=1e-14)


def test_layer_thickness():
    t, p, pl, zc, zl = _make_3level_atm()
    _, dz, _, _, _, _, _ = setup_atm(t, p, pl, zc, zl, GridType.I_CART)
    # 1000m = 100000cm spacing
    npt.assert_allclose(dz, 100000.0, rtol=1e-14)


def test_viscosity_positive():
    t, p, pl, zc, zl = _make_3level_atm()
    _, _, _, _, rmu, _, _ = setup_atm(t, p, pl, zc, zl, GridType.I_CART)
    assert jnp.all(rmu > 0)


def test_viscosity_sutherland():
    """Verify Sutherland's law at known temperature."""
    # At T=300K, Sutherland gives ~1.846e-4 g/cm/s
    t = jnp.array([300.0])
    p = jnp.array([1.01325e6])  # 1 atm in dyne/cm^2
    pl = jnp.array([1.02e6, 1.00e6])
    zc = jnp.array([50000.0])  # 500m in cm
    zl = jnp.array([0.0, 100000.0])
    _, _, _, _, rmu, _, _ = setup_atm(t, p, pl, zc, zl, GridType.I_CART)
    # Sutherland: rmu = 1.8325e-4 * 416.16 / (300+120) * (300/296.16)^1.5
    expected = 1.8325e-4 * 416.16 / 420.0 * (300.0 / 296.16) ** 1.5
    npt.assert_allclose(float(rmu[0]), expected, rtol=1e-10)


def test_thermal_conductivity_at_t0():
    """At T=T0 (273.16K), thcond = 5.69 * 418.6."""
    t = jnp.array([T0])
    p = jnp.array([1.01325e6])
    pl = jnp.array([1.02e6, 1.00e6])
    zc = jnp.array([50000.0])
    zl = jnp.array([0.0, 100000.0])
    _, _, _, _, _, thcond, _ = setup_atm(t, p, pl, zc, zl, GridType.I_CART)
    expected = 5.69 * 4.186e2
    npt.assert_allclose(float(thcond[0]), expected, rtol=1e-10)


def test_output_shapes():
    t, p, pl, zc, zl = _make_3level_atm()
    rhoa, dz, zmet, zmetl, rmu, thcond, rhoa_wet = setup_atm(
        t, p, pl, zc, zl, GridType.I_CART
    )
    assert rhoa.shape == (3,)
    assert dz.shape == (3,)
    assert zmet.shape == (3,)
    assert zmetl.shape == (4,)
    assert rmu.shape == (3,)
    assert thcond.shape == (3,)
    assert rhoa_wet.shape == (3,)
