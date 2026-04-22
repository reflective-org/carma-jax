"""Unit tests for evap_mono + evap_poly (Phase 8.2)."""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from carma.growth.evap_mono import evap_mono
from carma.growth.evap_poly import evap_poly


def _grid(nbin=8, rho=1.8, rmrat=2.0, rmin_cm=1e-7):
    vmin = (4.0 / 3.0) * np.pi * rmin_cm**3 * rho
    rmass = vmin * rmrat ** np.arange(nbin)
    dm = rmass * (rmrat - 1.0) / 2.0
    return jnp.asarray(rmass[:, None]), jnp.asarray(dm[:, None])   # (nbin, 1)


def _diffmass_1grp(rmass):
    """rmass is (nbin, 1). Build diffmass[target_bin, target_grp,
    src_bin, src_grp] with shape (nbin, 1, nbin, 1)."""
    tgt = rmass[:, :, None, None]
    src = rmass[None, None, :, :]
    return tgt - src


def _icorelem_ievp2elem(nelem, ncore):
    """Minimal core-element setup for tests.
    Element 0 = number (target & source share ieto=0).
    Elements 1..ncore-1 = core masses; each maps to ievp2elem[ic] = ic
    (core goes into same-index element of target group)."""
    icorelem = np.asarray([[ic] for ic in range(ncore)])   # (ncore, 1)
    ievp2elem = np.asarray(list(range(nelem)))             # identity
    return icorelem, ievp2elem


# --- evap_mono: interior split ---

def test_mono_interior_number_conserved():
    nbin = 8
    rmass, _ = _grid(nbin=nbin)
    diffmass = _diffmass_1grp(rmass)
    icorelem, ievp2elem = _icorelem_ievp2elem(nelem=2, ncore=2)

    iavg = 4
    coreavg = 0.5 * (float(rmass[iavg - 1, 0]) + float(rmass[iavg, 0]))

    evdrop = 100.0
    evcore = jnp.asarray([0.0, 2.0])     # evcore[1] = core mass rate

    delta = evap_mono(
        evdrop=evdrop, evcore=evcore, coreavg=coreavg, iavg=iavg,
        ieto=0, igto=0,
        too_big=False, too_small=False, nuc_small=False,
        ncore_ig=2, icorelem=icorelem, ievp2elem=ievp2elem, ig=0,
        rmass=rmass, diffmass=diffmass,
        nbin=nbin, nelem=2,
    )
    # Total number is conserved exactly.
    total_num_in = evdrop
    total_num_out = float(jnp.sum(delta[:, 0]))
    assert abs(total_num_out - total_num_in) < 1e-12


def test_mono_interior_core_mass_conserved():
    """Sum over target bins of core_mass_added should equal
    evcore[ic] · coreavg by construction (Fortran conserves mean core
    mass, not individual per-particle mass — but total number × mean
    = evdrop · coreavg for evcore[ic] = evdrop · pc_core/pc_tot)."""
    nbin = 8
    rmass, _ = _grid(nbin=nbin)
    diffmass = _diffmass_1grp(rmass)
    icorelem, ievp2elem = _icorelem_ievp2elem(nelem=2, ncore=2)

    iavg = 5
    coreavg = 0.5 * (float(rmass[iavg - 1, 0]) + float(rmass[iavg, 0]))

    evdrop = 100.0
    evcore = jnp.asarray([0.0, 2.5])

    delta = evap_mono(
        evdrop=evdrop, evcore=evcore, coreavg=coreavg, iavg=iavg,
        ieto=0, igto=0,
        too_big=False, too_small=False, nuc_small=False,
        ncore_ig=2, icorelem=icorelem, ievp2elem=ievp2elem, ig=0,
        rmass=rmass, diffmass=diffmass,
        nbin=nbin, nelem=2,
    )
    # core mass sum = evcore[1] · coreavg (by the fracmass construction)
    total_core_out = float(jnp.sum(delta[:, 1]))
    expected = 2.5 * coreavg
    assert abs(total_core_out - expected) / expected < 1e-12


# --- evap_mono: boundary cases ---

def test_mono_too_small_puts_mass_in_bin_0():
    nbin = 8
    rmass, _ = _grid(nbin=nbin)
    diffmass = _diffmass_1grp(rmass)
    icorelem, ievp2elem = _icorelem_ievp2elem(nelem=2, ncore=2)

    coreavg = float(rmass[0, 0]) * 0.1   # smaller than smallest bin
    evdrop = 100.0
    evcore = jnp.asarray([0.0, 1.0])

    delta = evap_mono(
        evdrop=evdrop, evcore=evcore, coreavg=coreavg, iavg=0,
        ieto=0, igto=0,
        too_big=False, too_small=True, nuc_small=False,
        ncore_ig=2, icorelem=icorelem, ievp2elem=ievp2elem, ig=0,
        rmass=rmass, diffmass=diffmass,
        nbin=nbin, nelem=2,
    )
    # All number should land in bin 0, with mass factor = coreavg/rmass[0]
    factor = coreavg / float(rmass[0, 0])
    expected_num = factor * evdrop
    assert abs(float(delta[0, 0]) - expected_num) < 1e-12
    assert float(jnp.sum(delta[1:, 0])) == 0.0


def test_mono_too_big_puts_mass_in_top_bin():
    nbin = 8
    rmass, _ = _grid(nbin=nbin)
    diffmass = _diffmass_1grp(rmass)
    icorelem, ievp2elem = _icorelem_ievp2elem(nelem=2, ncore=2)

    coreavg = float(rmass[-1, 0]) * 10.0
    evdrop = 100.0
    evcore = jnp.asarray([0.0, 1.0])

    delta = evap_mono(
        evdrop=evdrop, evcore=evcore, coreavg=coreavg, iavg=0,
        ieto=0, igto=0,
        too_big=True, too_small=False, nuc_small=False,
        ncore_ig=2, icorelem=icorelem, ievp2elem=ievp2elem, ig=0,
        rmass=rmass, diffmass=diffmass,
        nbin=nbin, nelem=2,
    )
    # All number in top bin
    assert float(delta[-1, 0]) > 0
    assert float(jnp.sum(delta[:-1, 0])) == 0.0


def test_mono_conserve_mass_off_uses_unit_factor():
    nbin = 8
    rmass, _ = _grid(nbin=nbin)
    diffmass = _diffmass_1grp(rmass)
    icorelem, ievp2elem = _icorelem_ievp2elem(nelem=2, ncore=2)

    coreavg = float(rmass[0, 0]) * 0.1
    evdrop = 100.0
    evcore = jnp.asarray([0.0, 1.0])

    delta = evap_mono(
        evdrop=evdrop, evcore=evcore, coreavg=coreavg, iavg=0,
        ieto=0, igto=0,
        too_big=False, too_small=True, nuc_small=False,
        ncore_ig=2, icorelem=icorelem, ievp2elem=ievp2elem, ig=0,
        rmass=rmass, diffmass=diffmass,
        nbin=nbin, nelem=2,
        conserve_mass=False,
    )
    # conserve_mass=False → factor = 1; bin 0 number = evdrop exactly
    assert abs(float(delta[0, 0]) - evdrop) < 1e-12


# --- evap_mono: ievp2elem=-1 branch ---

def test_mono_skips_unmapped_core():
    nbin = 8
    rmass, _ = _grid(nbin=nbin)
    diffmass = _diffmass_1grp(rmass)
    # core element 1 is not mapped (ievp2elem[1] = -1)
    icorelem = np.asarray([[0], [1]])
    ievp2elem = np.asarray([0, -1])

    iavg = 4
    coreavg = 0.5 * (float(rmass[iavg - 1, 0]) + float(rmass[iavg, 0]))
    evdrop = 100.0
    evcore = jnp.asarray([0.0, 1.0])

    delta = evap_mono(
        evdrop=evdrop, evcore=evcore, coreavg=coreavg, iavg=iavg,
        ieto=0, igto=0,
        too_big=False, too_small=False, nuc_small=False,
        ncore_ig=2, icorelem=icorelem, ievp2elem=ievp2elem, ig=0,
        rmass=rmass, diffmass=diffmass,
        nbin=nbin, nelem=2,
    )
    # Number still scatters; core is dropped (element 1 all zero)
    assert float(jnp.sum(delta[:, 0])) > 0
    assert float(jnp.sum(delta[:, 1])) == 0.0


# --- evap_poly ---

def test_poly_number_conserved():
    nbin = 8
    rmass, dm = _grid(nbin=nbin)
    icorelem, ievp2elem = _icorelem_ievp2elem(nelem=2, ncore=2)

    iavg = 4
    coreavg = 0.5 * (float(rmass[iavg - 1, 0]) + float(rmass[iavg, 0]))
    coresig = 1.0      # log(std²) — moderate spread
    evdrop = 100.0
    evcore = jnp.asarray([0.0, 2.0])

    delta = evap_poly(
        evdrop=evdrop, evcore=evcore, coreavg=coreavg, coresig=coresig,
        iavg=iavg, ieto=0, igto=0,
        ncore_ig=2, icorelem=icorelem, ievp2elem=ievp2elem, ig=0,
        rmass=rmass, dm=dm, nbin=nbin, nelem=2,
    )
    total_num_out = float(jnp.sum(delta[:, 0]))
    assert abs(total_num_out - evdrop) / evdrop < 1e-12


def test_poly_zero_coreavg_gives_zero_delta():
    """Dispatcher checks coreavg > 0; our guard must also zero out
    the contribution in that edge case."""
    nbin = 8
    rmass, dm = _grid(nbin=nbin)
    icorelem, ievp2elem = _icorelem_ievp2elem(nelem=2, ncore=2)

    delta = evap_poly(
        evdrop=100.0, evcore=jnp.zeros(2),
        coreavg=0.0, coresig=1.0,
        iavg=4, ieto=0, igto=0,
        ncore_ig=2, icorelem=icorelem, ievp2elem=ievp2elem, ig=0,
        rmass=rmass, dm=dm, nbin=nbin, nelem=2,
    )
    assert float(jnp.abs(delta).max()) == 0.0


def test_poly_spreads_across_multiple_bins():
    """The polydisperse distribution should put non-zero weight in
    bins on both sides of iavg."""
    nbin = 8
    rmass, dm = _grid(nbin=nbin)
    icorelem, ievp2elem = _icorelem_ievp2elem(nelem=2, ncore=2)

    iavg = 4
    coreavg = float(rmass[iavg, 0])        # exactly at the target bin
    delta = evap_poly(
        evdrop=100.0, evcore=jnp.asarray([0.0, 0.0]),
        coreavg=coreavg, coresig=1.0,
        iavg=iavg, ieto=0, igto=0,
        ncore_ig=2, icorelem=icorelem, ievp2elem=ievp2elem, ig=0,
        rmass=rmass, dm=dm, nbin=nbin, nelem=2,
    )
    num_per_bin = np.asarray(delta[:, 0])
    below = (num_per_bin[:iavg] > 0).sum()
    above = (num_per_bin[iavg:] > 0).sum()
    assert below > 0 and above > 0


# --- JIT ---

def test_mono_jits():
    nbin = 8
    rmass, _ = _grid(nbin=nbin)
    diffmass = _diffmass_1grp(rmass)
    icorelem, ievp2elem = _icorelem_ievp2elem(nelem=2, ncore=2)

    iavg = 4
    coreavg = float(rmass[iavg, 0]) * 0.9
    evdrop = 100.0
    evcore = jnp.asarray([0.0, 1.0])

    def _wrap(evdrop, coreavg):
        return evap_mono(
            evdrop=evdrop, evcore=evcore, coreavg=coreavg, iavg=iavg,
            ieto=0, igto=0,
            too_big=False, too_small=False, nuc_small=False,
            ncore_ig=2, icorelem=icorelem, ievp2elem=ievp2elem, ig=0,
            rmass=rmass, diffmass=diffmass, nbin=nbin, nelem=2,
        )

    eager = _wrap(evdrop, coreavg)
    jit_out = jax.jit(_wrap)(evdrop, coreavg)
    assert jnp.allclose(eager, jit_out)


def test_poly_jits():
    nbin = 8
    rmass, dm = _grid(nbin=nbin)
    icorelem, ievp2elem = _icorelem_ievp2elem(nelem=2, ncore=2)
    iavg = 4
    coreavg = float(rmass[iavg, 0])

    def _wrap(evdrop, coresig):
        return evap_poly(
            evdrop=evdrop, evcore=jnp.asarray([0.0, 1.0]),
            coreavg=coreavg, coresig=coresig,
            iavg=iavg, ieto=0, igto=0,
            ncore_ig=2, icorelem=icorelem, ievp2elem=ievp2elem, ig=0,
            rmass=rmass, dm=dm, nbin=nbin, nelem=2,
        )

    eager = _wrap(100.0, 1.0)
    jit_out = jax.jit(_wrap)(100.0, 1.0)
    assert jnp.allclose(eager, jit_out)
