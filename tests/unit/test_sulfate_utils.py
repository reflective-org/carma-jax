"""Unit tests for carma.sulfate_utils."""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from carma.constants import AVG, BK, WTMOL_H2O
from carma.sulfate_utils import wtpct_tabaz, sulfate_density, sulfate_surf_tens
from carma.vapor_pressure import vaporp_h2o_murphy2005


def _h2o_mass(T, RH):
    """Helper: water-vapour mass concentration [g/cm³] at (T, RH)."""
    pvapl = float(vaporp_h2o_murphy2005(jnp.array(T))[0])
    return RH * pvapl * float(WTMOL_H2O) / (float(BK) * T * float(AVG)), pvapl


# --- wtpct_tabaz ---

def test_wtpct_clamped_in_unit_range():
    """Output stays in [1, 100] across full input domain."""
    for T in [185.0, 220.0, 260.0]:
        for RH in [0.001, 0.05, 0.4, 0.85, 1.0]:
            h2o_mass, pvapl = _h2o_mass(T, RH)
            wtp = float(wtpct_tabaz(T, h2o_mass, pvapl))
            assert 1.0 <= wtp <= 100.0


def test_wtpct_decreases_with_humidity():
    """At fixed T, more humidity dilutes the aerosol."""
    T = 220.0
    wtps = []
    for RH in [0.05, 0.2, 0.5, 0.85]:
        h2o_mass, pvapl = _h2o_mass(T, RH)
        wtps.append(float(wtpct_tabaz(T, h2o_mass, pvapl)))
    assert all(wtps[i] > wtps[i + 1] for i in range(len(wtps) - 1)), (
        f"wt% must decrease monotonically with RH, got {wtps}")


def test_wtpct_branch_jumps_are_small():
    """Tabazadeh's piecewise fits have small intentional discontinuities at
    activity = 0.05 and 0.85 (the published fits weren't C0-constrained).
    Verify the jumps stay within physical bounds (≤ 5 wt%)."""
    T = 220.0
    pvapl = float(vaporp_h2o_murphy2005(jnp.array(T))[0])
    for activ in [0.05, 0.85]:
        eps = 1e-4
        wtps = []
        for a in [activ - eps, activ + eps]:
            h2o_mass = a * pvapl * float(WTMOL_H2O) / (float(BK) * T * float(AVG))
            wtps.append(float(wtpct_tabaz(T, h2o_mass, pvapl)))
        assert abs(wtps[0] - wtps[1]) < 5.0, (
            f"unexpected discontinuity at activ={activ}: {wtps}")


def test_wtpct_canonical_strat():
    """Stratospheric aerosol (T=220K, RH=10%) ≈ 55–65 wt% per Carslaw 1995."""
    h2o_mass, pvapl = _h2o_mass(220.0, 0.10)
    wtp = float(wtpct_tabaz(220.0, h2o_mass, pvapl))
    assert 55.0 < wtp < 65.0


# --- sulfate_density ---

def test_density_pure_water():
    """At wt% = 0, density ≈ 1.0 g/cm³ regardless of T (within table precision)."""
    for T in [200.0, 260.0, 320.0]:
        rho = float(sulfate_density(0.0, T))
        assert 0.99 < rho < 1.01


def test_density_pure_h2so4():
    """At wt% = 100, density ≈ 1.83 g/cm³ at 300 K (handbook value 1.84)."""
    rho = float(sulfate_density(100.0, 300.0))
    assert 1.80 < rho < 1.86


def test_density_monotonic_below_peak():
    """Density rises monotonically with wt% up to the table peak near
    96 wt%, then dips slightly to pure-acid value (well-known feature of
    H2SO4/H2O density data — the maximum is around 95–96 wt% at warm T)."""
    T = 250.0
    wtps = jnp.linspace(0, 95, 50)  # below the high-wt% peak
    rhos = jax.vmap(lambda w: sulfate_density(w, T))(wtps)
    rhos = np.asarray(rhos)
    assert (np.diff(rhos) > 0).all(), "density should be monotonic below 95 wt%"
    assert rhos[-1] > rhos[0]
    # Pure-acid endpoint slightly below the ~96 wt% peak — sanity:
    rho_peak = float(sulfate_density(96.0, T))
    rho_pure = float(sulfate_density(100.0, T))
    assert rho_pure < rho_peak  # the well-known dip


def test_density_temperature_clamp():
    """T outside [180, 380] K must give same result as the clamp."""
    T_low_clamp = float(sulfate_density(50.0, 100.0))
    T_at_180 = float(sulfate_density(50.0, 180.0))
    assert T_low_clamp == T_at_180

    T_hi_clamp = float(sulfate_density(50.0, 500.0))
    T_at_380 = float(sulfate_density(50.0, 380.0))
    assert T_hi_clamp == T_at_380


# --- sulfate_surf_tens ---

def test_surf_tens_pure_water():
    """At wt% = 0, T = 298 K, surface tension ≈ 72 dyn/cm (handbook 72.0)."""
    sig = float(sulfate_surf_tens(0.0, 298.0))
    assert 70.0 < sig < 75.0


def test_surf_tens_high_wtp_dropoff():
    """Surface tension of binary H2SO4/H2O actually rises slightly above
    water for moderate wt% (peak around 50–60 wt%) before dropping for
    pure acid. Verify only the high-wt% drop where physics is unambiguous."""
    T = 298.0
    sig_50 = float(sulfate_surf_tens(50.0, T))
    sig_100 = float(sulfate_surf_tens(100.0, T))
    assert sig_100 < sig_50, f"expected drop from {sig_50} to {sig_100}"


# --- JIT / vmap composition ---

def test_jit_round_trip():
    T = 220.0
    h2o_mass, pvapl = _h2o_mass(T, 0.3)
    eager = wtpct_tabaz(T, h2o_mass, pvapl)
    jit = jax.jit(wtpct_tabaz)(T, h2o_mass, pvapl)
    assert jnp.allclose(eager, jit)


def test_vmap_over_temperature():
    Ts = jnp.array([190.0, 220.0, 260.0])
    pvapl_arr = vaporp_h2o_murphy2005(Ts)[0]
    h2o_mass_arr = 0.3 * pvapl_arr * WTMOL_H2O / (BK * Ts * AVG)
    out = jax.vmap(wtpct_tabaz)(Ts, h2o_mass_arr, pvapl_arr)
    assert out.shape == (3,)
    assert (out >= 1.0).all() and (out <= 100.0).all()


def test_density_vmap():
    wtps = jnp.linspace(0, 100, 10)
    rhos = jax.vmap(lambda w: sulfate_density(w, 250.0))(wtps)
    assert rhos.shape == (10,)
    assert (rhos > 0.9).all()


def test_surf_tens_vmap():
    wtps = jnp.linspace(0, 100, 10)
    sigs = jax.vmap(lambda w: sulfate_surf_tens(w, 250.0))(wtps)
    assert sigs.shape == (10,)
    assert (sigs > 50.0).all() and (sigs < 130.0).all()
