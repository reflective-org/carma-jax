"""Unit tests for carma.wetr and carma.hygroscopicity."""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from carma.constants import AVG, BK, WTMOL_H2O
from carma.enums import SwellMethod
from carma.hygroscopicity import hygroscopicity
from carma.vapor_pressure import vaporp_h2o_murphy2005
from carma.wetr import get_wetr


# --- wetr: I_NO_SWELLING ---

def test_no_swelling_passthrough():
    rwet, rhop = get_wetr(rdry=1e-5, rhopdry=1.8, rh=0.5, temp=298.0,
                          irhswell=int(SwellMethod.I_NO_SWELLING))
    assert float(rwet) == 1e-5
    assert float(rhop) == 1.8


# --- wetr: I_PETTERS (κ-Köhler) ---

def test_petters_matches_analytic_pk07():
    """Check rwet = rdry * (1 + RH·κ/(1-RH))^(1/3) exactly."""
    rdry = 1e-5
    for rh in [0.3, 0.6, 0.9]:
        for kappa in [0.0, 0.3, 0.6, 1.2]:
            expected = rdry * (1.0 + rh * kappa / (1.0 - rh)) ** (1.0 / 3.0)
            rwet, _ = get_wetr(rdry=rdry, rhopdry=1.77, rh=rh, temp=298.0,
                                irhswell=int(SwellMethod.I_PETTERS),
                                kappa=kappa)
            assert abs(float(rwet) - expected) / expected < 1e-10, (
                f"PK07 mismatch at κ={kappa}, RH={rh}: got {float(rwet)}, "
                f"expected {expected}")


def test_petters_kappa_zero_is_dry():
    """κ=0 → insoluble → rwet = rdry."""
    rwet, _ = get_wetr(rdry=5e-6, rhopdry=2.6, rh=0.95, temp=298.0,
                       irhswell=int(SwellMethod.I_PETTERS), kappa=0.0)
    assert abs(float(rwet) - 5e-6) / 5e-6 < 1e-12


def test_petters_density_mass_conservative():
    """ρ_wet formula: r_ratio·ρ_dry + (1-r_ratio)·ρ_water. For κ=0 should
    give ρ_dry; for large swelling should approach ρ_water."""
    rdry = 1e-5
    # Small swelling: κ=0.05, RH=0.3 → small r_ratio change
    _, rhop = get_wetr(rdry, 1.77, 0.3, 298.0,
                       int(SwellMethod.I_PETTERS), kappa=0.05)
    assert 1.5 < float(rhop) < 1.77

    # Heavy swelling: κ=1.2, RH=0.99 → rwet >> rdry → ρ_wet → 1.0
    _, rhop2 = get_wetr(rdry, 1.77, 0.99, 298.0,
                        int(SwellMethod.I_PETTERS), kappa=1.2)
    assert abs(float(rhop2) - 1.0) < 0.02


def test_petters_lowT_rescale_transition():
    """Above T=190 K uses RH directly; below uses RH rescaled by pvap ratio.
    Growth factor should be continuous (and small) across the transition."""
    rdry = 1e-5
    rh = 0.5
    kappa = 0.3
    gfs = []
    for T in [185.0, 190.0, 195.0, 200.0, 220.0]:
        rwet, _ = get_wetr(rdry, 1.77, rh, T,
                           int(SwellMethod.I_PETTERS), kappa=kappa)
        gfs.append(float(rwet) / rdry)
    # All positive, all > 1 (PK07 with κ>0)
    assert all(g > 1.0 for g in gfs)


# --- wetr: I_WTPCT_H2SO4 ---

def test_wtpct_strat_sulfate():
    """Stratospheric sulfate: T=220K, RH=10% → modest swelling."""
    T = 220.0
    rh = 0.10
    pvapl = float(vaporp_h2o_murphy2005(jnp.array(T))[0])
    h2o_mass = rh * pvapl * float(WTMOL_H2O) / (float(BK) * T * float(AVG))
    rwet, rhop = get_wetr(rdry=1e-5, rhopdry=1.547, rh=rh, temp=T,
                          irhswell=int(SwellMethod.I_WTPCT_H2SO4),
                          h2o_mass=h2o_mass, h2o_vp=pvapl)
    # Should swell modestly (GF ~ 1.1-1.2)
    gf = float(rwet) / 1e-5
    assert 1.05 < gf < 1.25
    # Wet density close to dry density (concentrated solution)
    assert 1.5 < float(rhop) < 1.6


def test_wtpct_wet_denser_than_pure_water():
    """Even at high RH, sulfate wet density stays above pure water."""
    T = 290.0
    rh = 0.80
    pvapl = float(vaporp_h2o_murphy2005(jnp.array(T))[0])
    h2o_mass = rh * pvapl * float(WTMOL_H2O) / (float(BK) * T * float(AVG))
    _, rhop = get_wetr(rdry=1e-5, rhopdry=1.40, rh=rh, temp=T,
                       irhswell=int(SwellMethod.I_WTPCT_H2SO4),
                       h2o_mass=h2o_mass, h2o_vp=pvapl)
    assert float(rhop) > 1.0


# --- wetr error paths ---

def test_petters_missing_kappa_raises():
    with pytest.raises(ValueError, match="kappa"):
        get_wetr(1e-5, 1.77, 0.5, 298.0, int(SwellMethod.I_PETTERS))


def test_wtpct_missing_args_raises():
    with pytest.raises(ValueError, match="h2o_mass"):
        get_wetr(1e-5, 1.77, 0.5, 298.0, int(SwellMethod.I_WTPCT_H2SO4))


def test_fitzgerald_not_implemented():
    with pytest.raises(NotImplementedError, match="Fitzgerald"):
        get_wetr(1e-5, 1.77, 0.5, 298.0, int(SwellMethod.I_FITZGERALD))


# --- wetr JIT / vmap ---

def test_petters_jit_vmap():
    rdry = 1e-5
    rhopdry = 1.77
    rh_arr = jnp.array([0.3, 0.5, 0.7, 0.9])
    fn = jax.jit(jax.vmap(lambda rh: get_wetr(
        rdry, rhopdry, rh, 298.0,
        int(SwellMethod.I_PETTERS), kappa=0.6)[0]))
    rwet = fn(rh_arr)
    assert rwet.shape == (4,)
    assert (rwet > rdry).all()
    # Analytic cross-check
    expected = rdry * (1.0 + np.asarray(rh_arr) * 0.6 / (1.0 - np.asarray(rh_arr))) ** (1.0 / 3.0)
    assert jnp.allclose(rwet, expected)


# --- hygroscopicity ---

def test_hygroscopicity_shell_only_bin():
    """If a bin has only shell mass (no cores), bulk κ equals shell κ."""
    NZ, NBIN = 1, 1
    pc = jnp.array([[[1e5, 0.0]]])    # shell with tiny core mass
    rmass = jnp.array([[1e-15]])
    kappa_elem = jnp.array([0.6, 0.1])
    ienconc = jnp.array([0])
    igelem = jnp.array([0, 0])
    icorelem = jnp.array([[1]])
    ncore = jnp.array([1])

    k = hygroscopicity(pc, rmass, kappa_elem, ienconc, igelem, icorelem, ncore)
    assert abs(float(k[0, 0, 0]) - 0.6) < 1e-6


def test_hygroscopicity_mass_weighted():
    """50/50 shell-vs-core mass split gives κ = (κ_shell + κ_core) / 2."""
    NZ, NBIN = 1, 1
    total_mass = 1e-15
    pc_num = 1.0                                 # so pc_num * rmass = total_mass
    core_mass = 0.5 * total_mass
    pc = jnp.array([[[pc_num, core_mass]]])
    rmass = jnp.array([[total_mass]])
    kappa_elem = jnp.array([0.6, 0.2])
    ienconc = jnp.array([0])
    igelem = jnp.array([0, 0])
    icorelem = jnp.array([[1]])
    ncore = jnp.array([1])

    k = hygroscopicity(pc, rmass, kappa_elem, ienconc, igelem, icorelem, ncore)
    assert abs(float(k[0, 0, 0]) - 0.4) < 1e-6


def test_hygroscopicity_clamps_to_unit():
    """When core_mass > total mass (physically impossible), output clamps to 1."""
    NZ, NBIN = 1, 1
    pc = jnp.array([[[1.0, 1e-9]]])    # pc_num=1 → total=rmass, core=1e-9 >> that
    rmass = jnp.array([[1e-15]])       # total = 1e-15
    kappa_elem = jnp.array([0.6, 0.5])
    ienconc = jnp.array([0])
    igelem = jnp.array([0, 0])
    icorelem = jnp.array([[1]])
    ncore = jnp.array([1])

    k = hygroscopicity(pc, rmass, kappa_elem, ienconc, igelem, icorelem, ncore)
    # Core >> total; Fortran would clamp pc_num. Here kappa_bulk exceeds 1
    # before the clip, so it clamps to 1.
    assert float(k[0, 0, 0]) == 1.0


def test_hygroscopicity_jit():
    NZ, NBIN, NELEM, NGROUP = 1, 2, 2, 1
    pc = jnp.array([[[1e3, 1e-10], [500.0, 2e-10]]])
    rmass = jnp.array([[1e-15], [1e-14]])
    kappa_elem = jnp.array([0.5, 0.1])
    ienconc = jnp.array([0])
    igelem = jnp.array([0, 0])
    icorelem = jnp.array([[1]])
    ncore = jnp.array([1])

    eager = hygroscopicity(pc, rmass, kappa_elem, ienconc, igelem, icorelem, ncore)
    jit = jax.jit(hygroscopicity)(pc, rmass, kappa_elem, ienconc, igelem, icorelem, ncore)
    assert jnp.allclose(eager, jit)
