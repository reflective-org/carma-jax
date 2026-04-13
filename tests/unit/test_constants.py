"""Tests for physical constants matching Fortran values."""

import jax.numpy as jnp

from carma.constants import (
    AVG, BK, CP, GRAV, PI, R_AIR, RGAS, RHO_I, RHO_W, RLHE_CNST, RLHM_CNST,
    RM2CGS, RPA2CGS, SMALL_PC, FEW_PC, T0, WTMOL_AIR, WTMOL_H2O,
)
from carma.precision import DTYPE


def test_fundamental_constants():
    assert GRAV == DTYPE(980.6)
    assert AVG == DTYPE(6.02252e23)
    assert BK == DTYPE(1.38054e-16)
    assert RGAS == DTYPE(8.31430e7)
    assert PI == DTYPE(3.14159265358979)


def test_derived_constants():
    assert jnp.isclose(R_AIR, RGAS / WTMOL_AIR, rtol=1e-14)


def test_water_constants():
    assert RHO_W == DTYPE(1.0)
    assert RHO_I == DTYPE(0.93)
    assert RLHE_CNST == DTYPE(2.501e10)
    assert RLHM_CNST == DTYPE(3.337e9)


def test_conversion_factors():
    assert RM2CGS == DTYPE(100.0)
    assert RPA2CGS == DTYPE(10.0)


def test_thresholds():
    assert SMALL_PC == DTYPE(1e-50)
    assert jnp.isclose(FEW_PC, SMALL_PC * 1e6, rtol=1e-14)


def test_triple_point():
    assert T0 == DTYPE(273.16)


def test_molecular_weights():
    assert WTMOL_AIR == DTYPE(28.966)
    assert WTMOL_H2O == DTYPE(18.016)
