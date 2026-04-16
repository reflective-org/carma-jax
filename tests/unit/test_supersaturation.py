"""Tests for supersaturation calculation."""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy.testing as npt

from carma.supersaturation import supersat
from carma.constants import RGAS, WTMOL_H2O
from carma.precision import DTYPE


def test_saturated_gives_zero():
    """When gc exactly equals saturation, supersaturation should be 0."""
    t = jnp.array([273.16], dtype=DTYPE)
    pvapl = jnp.array([6112.0], dtype=DTYPE)  # ~sat vapor pressure at 273K
    pvapi = jnp.array([6112.0], dtype=DTYPE)
    zmet = jnp.array([1.0], dtype=DTYPE)
    gwtmol = WTMOL_H2O

    # gc that gives exactly saturation: gc * Rv * T = pvapl
    rvap = RGAS / gwtmol
    gc_sat = pvapl / (rvap * t)

    ssl, ssi = supersat(t, gc_sat, pvapl, pvapi, gwtmol, zmet)
    npt.assert_allclose(float(ssl[0]), 0.0, atol=1e-12)
    npt.assert_allclose(float(ssi[0]), 0.0, atol=1e-12)


def test_supersaturated_positive():
    """Excess gas should give positive supersaturation."""
    t = jnp.array([273.16], dtype=DTYPE)
    pvapl = jnp.array([6112.0], dtype=DTYPE)
    pvapi = jnp.array([6112.0], dtype=DTYPE)
    zmet = jnp.array([1.0], dtype=DTYPE)
    rvap = RGAS / WTMOL_H2O
    gc_excess = pvapl * 1.5 / (rvap * t)  # 50% supersaturated

    ssl, ssi = supersat(t, gc_excess, pvapl, pvapi, WTMOL_H2O, zmet)
    npt.assert_allclose(float(ssl[0]), 0.5, rtol=1e-12)


def test_subsaturated_negative():
    """Deficit gas should give negative supersaturation."""
    t = jnp.array([273.16], dtype=DTYPE)
    pvapl = jnp.array([6112.0], dtype=DTYPE)
    pvapi = jnp.array([6112.0], dtype=DTYPE)
    zmet = jnp.array([1.0], dtype=DTYPE)
    rvap = RGAS / WTMOL_H2O
    gc_deficit = pvapl * 0.5 / (rvap * t)  # 50% subsaturated

    ssl, ssi = supersat(t, gc_deficit, pvapl, pvapi, WTMOL_H2O, zmet)
    npt.assert_allclose(float(ssl[0]), -0.5, rtol=1e-12)


def test_zmet_scaling():
    """zmet should scale gc correctly."""
    t = jnp.array([273.16], dtype=DTYPE)
    pvapl = jnp.array([6112.0], dtype=DTYPE)
    pvapi = jnp.array([6112.0], dtype=DTYPE)
    zmet1 = jnp.array([1.0], dtype=DTYPE)
    zmet2 = jnp.array([2.0], dtype=DTYPE)
    rvap = RGAS / WTMOL_H2O
    gc = pvapl / (rvap * t)

    ssl1, _ = supersat(t, gc, pvapl, pvapi, WTMOL_H2O, zmet1)
    ssl2, _ = supersat(t, gc * 2, pvapl, pvapi, WTMOL_H2O, zmet2)
    npt.assert_allclose(float(ssl1[0]), float(ssl2[0]), rtol=1e-12)
