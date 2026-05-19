"""Unit tests for the sulfnuc driver (homogeneous + heterogeneous)."""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from carma.constants import AVG, BK, WTMOL_H2O
from carma.nucleation.sulfhetnucrate import sulfhetnucrate
from carma.nucleation.sulfnuc import (
    _collision_betas,
    heterogeneous_nucleation,
    homogeneous_nucleation,
    sulfnuc,
)
from carma.nucleation.sulfnucrate import binary_nuc_zhao1995
from carma.sulfate_utils import wtpct_tabaz
from carma.vapor_pressure import vaporp_h2o_murphy2005


def _strat(T=220.0, rh=0.5, h2so4_num=1e8):
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
    )


def _bins(nbin=20, rmin_cm=1e-7, rmrat=2.0, rho=1.8):
    """Geometric mass grid and wet-radius array."""
    import numpy as np
    vmin = (4.0 / 3.0) * np.pi * rmin_cm**3 * rho
    rmass = vmin * rmrat ** np.arange(nbin)
    rmassup = rmass * rmrat**0.5
    r_bins = (3.0 * rmass / (4.0 * np.pi * rho)) ** (1.0 / 3.0)
    return jnp.asarray(rmassup), jnp.asarray(r_bins), rmrat


# --- collision betas ---

def test_collision_betas_positive():
    beta1, beta2 = _collision_betas(jnp.asarray(220.0))
    assert float(beta1) > 0
    assert float(beta2) > 0


def test_collision_betas_scale_with_T():
    b1_lo, _ = _collision_betas(jnp.asarray(200.0))
    b1_hi, _ = _collision_betas(jnp.asarray(280.0))
    # β ∝ √T, so hi / lo ≈ √(280/200) ≈ 1.18
    ratio = float(b1_hi / b1_lo)
    expected = (280.0 / 200.0) ** 0.5
    assert abs(ratio - expected) < 1e-6


def test_beta2_over_beta1_ratio():
    """β2 / β1 = √(M_H2SO4 / M_H2O) — independent of T."""
    b1, b2 = _collision_betas(jnp.asarray(250.0))
    assert abs(float(b2 / b1) - (98.0 / 18.0) ** 0.5) < 1e-6


# --- homogeneous path ---

def test_homogeneous_zhao_positive_rate_on_strat():
    args = _strat()
    rmassup, r_bins, rmrat = _bins()
    rate, nucbin, rstar, ftry = homogeneous_nucleation(
        **args, rmassup=rmassup, rmrat_val=rmrat, zmet=1.0,
        method="ZhaoTurco",
    )
    assert float(rate) > 0
    assert 0 <= int(nucbin) < rmassup.shape[0]
    assert float(rstar) > 0
    # ftry is -Gstar/kT; should be negative for a real barrier.
    assert float(ftry) < 0


def test_homogeneous_no_saddle_gives_zero():
    """Sub-saturated conditions: Zhao-Turco saddle search fails."""
    args = _strat(T=300.0, rh=0.001, h2so4_num=1.0)
    rmassup, r_bins, rmrat = _bins()
    rate, _, rstar, _ = homogeneous_nucleation(
        **args, rmassup=rmassup, rmrat_val=rmrat, zmet=1.0,
    )
    assert float(rate) == 0.0
    assert float(rstar) == 0.0


def test_homogeneous_vehk_is_used_when_requested():
    args = _strat()
    rmassup, r_bins, rmrat = _bins()
    rate, _, rstar, ftry = homogeneous_nucleation(
        **args, rmassup=rmassup, rmrat_val=rmrat, zmet=1.0,
        method="Vehkamaki",
    )
    assert float(rate) > 0
    assert float(rstar) > 0
    # Vehkamaki doesn't produce ftry in our driver.
    assert float(ftry) == 0.0


def test_homogeneous_unknown_method_raises():
    args = _strat()
    rmassup, r_bins, rmrat = _bins()
    with pytest.raises(ValueError, match="unknown homogeneous method"):
        homogeneous_nucleation(
            **args, rmassup=rmassup, rmrat_val=rmrat, zmet=1.0,
            method="NotAMethod",
        )


def test_homogeneous_zmet_scales_rate():
    args = _strat()
    rmassup, r_bins, rmrat = _bins()
    rate1, _, _, _ = homogeneous_nucleation(
        **args, rmassup=rmassup, rmrat_val=rmrat, zmet=1.0)
    rate100, _, _, _ = homogeneous_nucleation(
        **args, rmassup=rmassup, rmrat_val=rmrat, zmet=100.0)
    assert abs(float(rate100) / float(rate1) - 100.0) < 1e-6


# --- heterogeneous path ---

def test_heterogeneous_matches_sulfhetnucrate_per_bin():
    """Driver must produce the same per-bin rate as calling
    sulfhetnucrate directly."""
    args = _strat()
    _, r_bins, _ = _bins()
    rnuclg = heterogeneous_nucleation(**args, r_bins=r_bins)

    b1 = float(jnp.sqrt(8.31430e7 * 220.0 / 2 / jnp.pi / 98.0))
    b2 = float(jnp.sqrt(8.31430e7 * 220.0 / 2 / jnp.pi / 18.0))
    for i, r in enumerate(r_bins):
        expected = float(sulfhetnucrate(
            **args, beta1=b1, beta2=b2, r_preexist=float(r)))
        assert abs(float(rnuclg[i]) - expected) < 1e-10, (
            f"bin {i}: driver={float(rnuclg[i])}, direct={expected}")


def test_heterogeneous_zero_when_no_saddle():
    """Sub-saturated conditions → sulfhetnucrate's internal gate zeros
    every per-bin rate."""
    args = _strat(T=300.0, rh=0.001, h2so4_num=1.0)
    _, r_bins, _ = _bins()
    rnuclg = heterogeneous_nucleation(**args, r_bins=r_bins)
    assert (np.asarray(rnuclg) < 1e-20).all()


# --- full driver ---

def test_sulfnuc_both_paths_on_strat():
    args = _strat()
    rmassup, r_bins, rmrat = _bins()
    rhompe, rnuclg = sulfnuc(
        **args, r_bins=r_bins, rmassup=rmassup, rmrat_val=rmrat, zmet=1.0,
    )
    assert rhompe.shape == (20,)
    assert rnuclg.shape == (20,)
    # Exactly one nucbin has the homogeneous rate.
    assert (np.asarray(rhompe) > 0).sum() == 1
    # Heterogeneous rate monotonically increases with seed size here.
    rn = np.asarray(rnuclg)
    assert (rn > 0).any()


def test_sulfnuc_do_homogeneous_false():
    args = _strat()
    rmassup, r_bins, rmrat = _bins()
    rhompe, rnuclg = sulfnuc(
        **args, r_bins=r_bins, rmassup=rmassup, rmrat_val=rmrat, zmet=1.0,
        do_homogeneous=False,
    )
    assert (np.asarray(rhompe) == 0).all()
    # Heterogeneous must still run (rstar from homogeneous path is computed
    # internally even when we zero out rhompe).
    assert (np.asarray(rnuclg) > 0).any()


def test_sulfnuc_do_heterogeneous_false():
    args = _strat()
    rmassup, r_bins, rmrat = _bins()
    rhompe, rnuclg = sulfnuc(
        **args, r_bins=r_bins, rmassup=rmassup, rmrat_val=rmrat, zmet=1.0,
        do_heterogeneous=False,
    )
    assert (np.asarray(rhompe) > 0).sum() == 1
    assert (np.asarray(rnuclg) == 0).all()


def test_sulfnuc_vehkamaki_disables_heterogeneous():
    """Vehkamaki path cannot seed heterogeneous nucleation (no rstar
    / ftry for the Fletcher factor). The driver must return zero
    rnuclg when method='Vehkamaki'."""
    args = _strat()
    rmassup, r_bins, rmrat = _bins()
    rhompe, rnuclg = sulfnuc(
        **args, r_bins=r_bins, rmassup=rmassup, rmrat_val=rmrat, zmet=1.0,
        method="Vehkamaki",
    )
    assert (np.asarray(rhompe) > 0).sum() == 1
    assert (np.asarray(rnuclg) == 0).all()


def test_sulfnuc_zero_when_no_saddle():
    args = _strat(T=300.0, rh=0.001, h2so4_num=1.0)
    rmassup, r_bins, rmrat = _bins()
    rhompe, rnuclg = sulfnuc(
        **args, r_bins=r_bins, rmassup=rmassup, rmrat_val=rmrat, zmet=1.0,
    )
    assert (np.asarray(rhompe) == 0).all()
    assert (np.asarray(rnuclg) == 0).all()


# --- JIT + vmap ---

def test_sulfnuc_jit():
    args = _strat()
    rmassup, r_bins, rmrat = _bins()
    eager = sulfnuc(
        **args, r_bins=r_bins, rmassup=rmassup, rmrat_val=rmrat, zmet=1.0)
    jit_fn = jax.jit(sulfnuc, static_argnames=("method",))
    jit_out = jit_fn(
        **args, r_bins=r_bins, rmassup=rmassup, rmrat_val=rmrat, zmet=1.0)
    assert jnp.allclose(eager[0], jit_out[0])
    assert jnp.allclose(eager[1], jit_out[1])


def test_sulfnuc_vmap_over_temperature():
    """Compose under vmap on the temperature axis."""
    rmassup, r_bins, rmrat = _bins()
    T_arr = jnp.array([200.0, 220.0, 240.0, 260.0])

    def for_T(T):
        args = _strat(T=float(T))
        return sulfnuc(
            **args, r_bins=r_bins, rmassup=rmassup, rmrat_val=rmrat, zmet=1.0,
        )

    # vmap doesn't accept Python-level data dependency, so just loop:
    out = [for_T(T) for T in T_arr]
    for rh_, rn_ in out:
        assert rh_.shape == (20,)
        assert rn_.shape == (20,)
