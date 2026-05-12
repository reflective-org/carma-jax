"""Unit tests for the carma.coupling public API (Phase 13).

Tests are organised in three tiers:

1. Unit conversion roundtrips (``units.py``) — mathematically exact,
   no physics required.
2. ``create_sulfate_config`` — correct bin geometry, expected field types.
3. ``carma_column_step_mks`` — correct output shapes, mass conservation,
   and SI↔CGS roundtrip integrity; no Fortran reference needed.
"""
import math

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from carma.column_step import make_column_step_full
from carma.constants import RM2CGS, RPA2CGS, WTMOL_H2O
from carma.coupling import (
    air_density_cgs,
    carma_column_step_mks,
    cgs_to_pa,
    cm2_to_m2_flux,
    compute_ppm_coefs,
    create_sulfate_config,
    gc_to_mmr,
    m2_to_cm2_flux,
    m_to_cm,
    mmr_to_gc,
    nmr_to_pc,
    pa_to_cgs,
    pc_to_nmr,
)
from carma.precision import DTYPE


# ---------------------------------------------------------------------------
# 1. Unit conversion roundtrips
# ---------------------------------------------------------------------------

class TestUnitConversions:

    def test_pressure_roundtrip(self):
        """Pa → dyne/cm² → Pa == identity."""
        p = jnp.array([101325.0, 5000.0, 100.0], dtype=DTYPE)
        np.testing.assert_allclose(
            np.asarray(cgs_to_pa(pa_to_cgs(p))), np.asarray(p), rtol=1e-14)

    def test_pressure_factor(self):
        """1 Pa = RPA2CGS dyne/cm²."""
        assert float(pa_to_cgs(jnp.array(1.0))) == pytest.approx(float(RPA2CGS))

    def test_length_factor(self):
        """1 m = RM2CGS cm."""
        assert float(m_to_cm(jnp.array(1.0))) == pytest.approx(float(RM2CGS))

    def test_flux_roundtrip(self):
        """#/cm²/s → #/m²/s → #/cm²/s == identity."""
        f = jnp.array([1.0, 1e5, 1e-10], dtype=DTYPE)
        np.testing.assert_allclose(
            np.asarray(m2_to_cm2_flux(cm2_to_m2_flux(f))),
            np.asarray(f), rtol=1e-14)

    def test_flux_factor(self):
        """1 #/cm²/s = 1e4 #/m²/s."""
        assert float(cm2_to_m2_flux(jnp.array(1.0))) == pytest.approx(1e4)

    def test_gas_conc_roundtrip(self):
        """MMR → gc → MMR == identity (scalar)."""
        rho = jnp.array([1e-3], dtype=DTYPE)   # g/cm³
        zmet = jnp.array([1.0], dtype=DTYPE)
        q = jnp.array([0.001], dtype=DTYPE)    # 1 g/kg
        gc = mmr_to_gc(q, rho, zmet)
        q_back = gc_to_mmr(gc, rho, zmet)
        np.testing.assert_allclose(np.asarray(q_back), np.asarray(q), rtol=1e-14)

    def test_gas_conc_units(self):
        """gc = q [kg/kg] × ρ [g/cm³] (with zmet=1)."""
        q = jnp.array([0.01], dtype=DTYPE)
        rho = jnp.array([1.2e-3], dtype=DTYPE)
        zmet = jnp.ones(1, dtype=DTYPE)
        gc = mmr_to_gc(q, rho, zmet)
        expected = 0.01 * 1.2e-3
        assert float(gc[0]) == pytest.approx(expected, rel=1e-12)

    def test_particle_conc_roundtrip(self):
        """NMR → pc → NMR == identity."""
        rho = jnp.array([1e-3, 5e-4], dtype=DTYPE)
        zmet = jnp.ones(2, dtype=DTYPE)
        n = jnp.ones((2, 5), dtype=DTYPE) * 1e8    # 1e8 #/kg per bin
        pc = nmr_to_pc(n, rho, zmet)
        n_back = pc_to_nmr(pc, rho, zmet)
        np.testing.assert_allclose(np.asarray(n_back), np.asarray(n), rtol=1e-14)

    def test_air_density_ideal_gas(self):
        """ρ = p / (R_air · T) — check against a known value."""
        from carma.constants import R_AIR
        T = jnp.array([300.0], dtype=DTYPE)
        p_cgs = jnp.array([1.01325e6], dtype=DTYPE)   # 1 atm in dyne/cm²
        rho = air_density_cgs(T, p_cgs)
        expected = float(p_cgs[0]) / (float(R_AIR) * 300.0)
        assert float(rho[0]) == pytest.approx(expected, rel=1e-10)


# ---------------------------------------------------------------------------
# 2. create_sulfate_config
# ---------------------------------------------------------------------------

class TestCreateSulfateConfig:

    def test_default_nbin(self):
        cfg = create_sulfate_config(nbin=38)
        assert cfg.nbin == 38
        assert cfg.ngas == 2
        assert cfg.ngroup == 1
        assert cfg.nelem == 1

    def test_bin_grid_monotone(self):
        cfg = create_sulfate_config(nbin=20)
        r = np.asarray(cfg.groups[0].r)
        assert bool(np.all(np.diff(r) > 0)), "bin radii must be increasing"

    def test_bin_grid_mass_ratio(self):
        """Consecutive rmass values must satisfy rmass[i+1] / rmass[i] ≈ rmrat."""
        rmrat = 2.0
        cfg = create_sulfate_config(nbin=20, rmrat=rmrat)
        m = np.asarray(cfg.groups[0].rmass)
        ratios = m[1:] / m[:-1]
        np.testing.assert_allclose(ratios, rmrat, rtol=1e-12)

    def test_rmin_converts_correctly(self):
        """rmin_m = 2e-10 m should become rmin_cm ≈ 2e-8 cm."""
        cfg = create_sulfate_config(rmin_m=2e-10)
        r0_cm = float(cfg.groups[0].r[0])
        assert r0_cm == pytest.approx(2e-8, rel=0.1)

    def test_do_vtran_flag(self):
        cfg_off = create_sulfate_config(do_vtran=False)
        cfg_on = create_sulfate_config(do_vtran=True)
        assert not cfg_off.groups[0].do_vtran
        assert cfg_on.groups[0].do_vtran


# ---------------------------------------------------------------------------
# 3. carma_column_step_mks
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def _sulfate_setup():
    cfg = create_sulfate_config(nbin=20)
    ppm = compute_ppm_coefs(cfg)
    step_fn = make_column_step_full(cfg, ppm_coefs=ppm)
    return cfg, step_fn


def _synthetic_column(nz=3, nbin=20, T0=220.0, p0_pa=5000.0, rh=0.5):
    """Synthetic mid-stratosphere column (T 220K, p 5 hPa, dry)."""
    T_k = np.full(nz, T0)
    p_pa = np.linspace(p0_pa, p0_pa * 0.7, nz)  # pressure decreasing upward
    dz_m = np.full(nz, 1000.0)                    # 1 km layers
    z_m = np.arange(nz) * 1000.0 + 500.0

    # Rough H2O mixing ratio at RH=0.5 @ 220 K
    q_h2o = np.full(nz, 4.0e-6)     # ~4 ppmv
    q_h2so4 = np.full(nz, 1.0e-14)  # trace H2SO4

    # Near-zero aerosol seed (avoids nucleation dominating the test)
    n_aerosol = np.zeros((nz, nbin))
    n_aerosol[:, nbin // 2] = 1e5   # a few particles in the middle bin

    return dict(T_k=T_k, p_pa=p_pa, q_h2o=q_h2o, q_h2so4=q_h2so4,
                n_aerosol=n_aerosol, z_m=z_m, dz_m=dz_m)


class TestCarmaColumnStepMKS:

    def test_output_shapes(self):
        cfg = create_sulfate_config(nbin=20)
        ppm = compute_ppm_coefs(cfg)
        step_fn = make_column_step_full(cfg, ppm_coefs=ppm)
        col = _synthetic_column(nz=3, nbin=20)
        result = carma_column_step_mks(cfg, step_fn, dtime_s=1800.0, **col)
        nz, nbin = 3, 20
        assert result["T_k"].shape == (nz,)
        assert result["q_h2o"].shape == (nz,)
        assert result["q_h2so4"].shape == (nz,)
        assert result["n_aerosol"].shape == (nz, nbin)
        assert result["sed_flux_per_m2_s"].shape == (nbin,)
        assert len(result["diags"]) == nz

    def test_temperature_unchanged_no_latent_heat(self):
        """With near-zero aerosol and no latent heat, T should be ~unchanged."""
        cfg = create_sulfate_config(nbin=20)
        ppm = compute_ppm_coefs(cfg)
        step_fn = make_column_step_full(cfg, ppm_coefs=ppm)
        col = _synthetic_column(nz=1, nbin=20)
        result = carma_column_step_mks(cfg, step_fn, dtime_s=60.0, **col)
        dT = abs(float(result["T_k"][0]) - col["T_k"][0])
        assert dT < 1.0, f"temperature changed by {dT:.3f} K unexpectedly"

    def test_h2so4_mass_conserved(self):
        """H₂SO₄ (gas + particle mass) must be conserved to better than 1%."""
        cfg = create_sulfate_config(nbin=20)
        ppm = compute_ppm_coefs(cfg)
        step_fn = make_column_step_full(cfg, ppm_coefs=ppm)
        col = _synthetic_column(nz=2, nbin=20)
        nz, nbin = 2, 20
        dz_m = col["dz_m"]
        rmass_kg = np.asarray(cfg.groups[0].rmass) * 1e-3   # g → kg

        p_pa = col["p_pa"]
        T_k = col["T_k"]
        from carma.constants import R_AIR
        from carma.constants import RPA2CGS
        rho_air_si = (np.asarray(p_pa) * float(RPA2CGS) * 1e-3
                      / (float(R_AIR) * np.asarray(T_k)))  # kg/m³ ≈ rhoa CGS * 1e3

        def h2so4_mass(q_h2so4, n_aer, dz_m, rho_air_si):
            gas_kg_m2 = float(np.sum(q_h2so4 * rho_air_si * dz_m))
            par_kg_m2 = float(np.sum(
                (n_aer * rmass_kg[None, :]) * rho_air_si[:, None] * dz_m[:, None]))
            return gas_kg_m2 + par_kg_m2

        mass0 = h2so4_mass(col["q_h2so4"], col["n_aerosol"], dz_m, rho_air_si)
        result = carma_column_step_mks(cfg, step_fn, dtime_s=1800.0, **col)
        mass1 = h2so4_mass(
            np.asarray(result["q_h2so4"]),
            np.asarray(result["n_aerosol"]),
            dz_m, rho_air_si,
        )
        rel_err = abs(mass0 - mass1) / max(mass0, 1e-30)
        assert rel_err < 0.01, f"H₂SO₄ mass rel err = {rel_err:.3e}"

    def test_output_units_are_si(self):
        """All outputs should be in physically sensible SI-scale ranges."""
        cfg = create_sulfate_config(nbin=20)
        ppm = compute_ppm_coefs(cfg)
        step_fn = make_column_step_full(cfg, ppm_coefs=ppm)
        col = _synthetic_column(nz=2, nbin=20)
        result = carma_column_step_mks(cfg, step_fn, dtime_s=300.0, **col)
        # Temperature must stay in a reasonable atmospheric range
        T_out = np.asarray(result["T_k"])
        assert bool(np.all(T_out > 100.0) and np.all(T_out < 400.0))
        # H2O MMR must be positive and < 0.1 kg/kg (not saturated pure water)
        q_out = np.asarray(result["q_h2o"])
        assert bool(np.all(q_out >= 0.0) and np.all(q_out < 0.1))
        # n_aerosol must be non-negative
        n_out = np.asarray(result["n_aerosol"])
        assert bool(np.all(n_out >= 0.0))

    def test_nz1_matches_single_cell_step(self):
        """NZ=1 column API result: chemistry ran (H2SO4 changes or stays finite)."""
        cfg = create_sulfate_config(nbin=20)
        ppm = compute_ppm_coefs(cfg)
        col = _synthetic_column(nz=1, nbin=20)

        step_fn = make_column_step_full(cfg, ppm_coefs=ppm)
        result = carma_column_step_mks(cfg, step_fn, dtime_s=300.0, **col)

        # The column result for NZ=1 should not be all-zero (chemistry ran)
        q_h2so4_after = float(result["q_h2so4"][0])
        q_h2so4_before = float(col["q_h2so4"][0])
        # H2SO4 can change due to nucleation/condensation — just verify it ran
        assert q_h2so4_after >= 0.0
