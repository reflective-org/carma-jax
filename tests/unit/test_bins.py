"""Tests for bin structure computation."""

import jax
import jax.numpy as jnp
import numpy.testing as npt

from carma.bins import setup_bins
from carma.constants import PI
from carma.precision import DTYPE


def test_bin_count():
    r, rmass, vol, dr, dm, rup, rlow, rmassup = setup_bins(1e-4, 2.0, 8, 1.0)
    assert r.shape == (8,)
    assert rmass.shape == (8,)
    assert rup.shape == (8,)
    assert rlow.shape == (8,)


def test_mass_ratio():
    """Adjacent bin masses should differ by rmrat."""
    rmrat = 2.0
    _, rmass, *_ = setup_bins(1e-4, rmrat, 10, 1.0)
    ratios = rmass[1:] / rmass[:-1]
    npt.assert_allclose(ratios, rmrat, rtol=1e-14)


def test_radius_mass_consistency():
    """r = (3*rmass/(4*pi*rho))^(1/3)."""
    rho = 2.65
    r, rmass, vol, *_ = setup_bins(1e-4, 2.0, 8, rho)
    r_from_mass = (3.0 * rmass / (4.0 * PI * rho)) ** (1.0 / 3.0)
    npt.assert_allclose(r, r_from_mass, rtol=1e-14)


def test_bin_ordering():
    """Bins should be strictly increasing."""
    r, rmass, vol, dr, dm, rup, rlow, rmassup = setup_bins(1e-4, 2.0, 8, 1.0)
    assert jnp.all(jnp.diff(r) > 0)
    assert jnp.all(jnp.diff(rmass) > 0)
    assert jnp.all(rlow < r)
    assert jnp.all(r < rup)


def test_first_bin_radius():
    """First bin center radius should match rmin."""
    rmin = 7.5e-4
    r, *_ = setup_bins(rmin, 2.0, 8, 2.65)
    npt.assert_allclose(float(r[0]), rmin, rtol=1e-10)


def test_positive_widths():
    """All bin widths should be positive."""
    _, _, _, dr, dm, *_ = setup_bins(1e-4, 2.0, 8, 1.0)
    assert jnp.all(dr > 0)
    assert jnp.all(dm > 0)
