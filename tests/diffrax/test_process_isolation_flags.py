"""Smoke tests for the diffrax physics-isolation flags.

The diffrax sulfate path now exposes three orthogonal toggles so we
can run controlled experiments where only one physics process is
active at a time:

  - ``make_rhs(shape, do_growth=…, do_homogeneous=…)``
  - ``diffrax_step(..., coag=...)``

This test file is the minimum-viable check that each toggle does the
right thing on a trivial integration. The physics-isolation 3-way
plots (Test A coag, Test B condensation, Test C nucleation) run in
separate benchmark scripts and reuse these toggles.
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from carma.precision import DTYPE
from carma_diffrax.rhs import make_rhs, FrozenEnv
from carma_diffrax.state import StateShape, pack, unpack
from tests.diffrax._env_builder import make_env_and_state


def _build_env(rh=0.30, T_K=298.0, h2so4_g_per_cm3=1e-15, M=2.0,
                mu_nm=100.0, sigma_g=1.6):
    """Construct a FrozenEnv + initial state for a single test."""
    env, shape, pc, gc, T = make_env_and_state(
        T_K=T_K, p_hPa=1013.0, rh=rh,
        h2so4_g_per_cm3=h2so4_g_per_cm3,
        mu_nm=mu_nm, sigma_g=sigma_g, M_ug_m3=M,
    )
    return env, pc, gc, T


def test_do_growth_false_zeros_pc_drift_from_advection():
    """With do_growth=False, the only thing that can change pc is
    nucleation (which puts mass in the bottom bin). The PPM advection
    contribution (which would otherwise redistribute pc across bins)
    must be zero.
    """
    env, pc, gc, T = _build_env(h2so4_g_per_cm3=1e-13)  # nontrivial H2SO4
    nbin = pc.shape[0]
    shape = StateShape(nbin=nbin, nelem=1, ngas=2)

    # Both growth and nuc off → dpc/dt MUST be exactly zero.
    rhs_off = make_rhs(shape, do_growth=False, do_homogeneous=False)
    y = pack(pc, gc, T)
    dy = rhs_off(0.0, y, env)
    dpc_dt, dgc_dt, dT_dt = unpack(dy, shape)
    np.testing.assert_array_equal(np.asarray(dpc_dt), 0.0)
    # Gas should be a passive spectator (zero in both columns) since the
    # growth + nuc mass sinks are both off.
    np.testing.assert_array_equal(np.asarray(dgc_dt), 0.0)


def test_do_homogeneous_false_no_new_particles():
    """With do_homogeneous=False, nucleation does not put any number
    flux into bin 0. Verify by zeroing out the existing aerosol and
    confirming dpc/dt[bin 0] is exactly 0 (it would otherwise be
    positive from sulfnuc).
    """
    env, _pc, gc, T = _build_env(h2so4_g_per_cm3=1e-12)  # high H2SO4 to drive nuc
    nbin = 38
    shape = StateShape(nbin=nbin, nelem=1, ngas=2)

    # Empty aerosol → growth contribution to bin 0 is zero, only nuc
    # could put particles there.
    pc = jnp.zeros((nbin, 1), dtype=DTYPE)

    rhs_off = make_rhs(shape, do_homogeneous=False)
    y = pack(pc, gc, T)
    dpc_dt, _, _ = unpack(rhs_off(0.0, y, env), shape)
    assert float(dpc_dt[0, 0]) == 0.0


def test_do_homogeneous_true_does_seed_bin_0():
    """Sanity: with do_homogeneous=True, the SAME setup as above
    *does* produce a nonzero positive nucleation flux into some bin.
    Confirms the flag is genuinely off vs on, not always off.
    """
    env, _pc, gc, T = _build_env(h2so4_g_per_cm3=1e-12)
    nbin = 38
    shape = StateShape(nbin=nbin, nelem=1, ngas=2)
    pc = jnp.zeros((nbin, 1), dtype=DTYPE)

    rhs_on = make_rhs(shape, do_homogeneous=True)
    y = pack(pc, gc, T)
    dpc_dt, _, _ = unpack(rhs_on(0.0, y, env), shape)
    # Total dpc/dt across all bins should be positive (nucleation is a
    # net source of particles).
    assert float(jnp.sum(dpc_dt)) > 0


def test_apply_coag_reduces_total_number():
    """One operator-split coag step on a moderately dense lognormal
    should monotonically reduce total particle number (coagulation
    only merges; it never creates particles).
    """
    from carma_diffrax.coag_step import (
        CoagBundle, apply_coag, build_microslow_for_sulfate,
    )
    from jax_ensemble import _minimal_config, _refresh_env, _compute_ppm_coefs
    from carma.constants import RPA2CGS, R_AIR

    cfg = _minimal_config()._replace(do_coag=True, do_grow=False)
    ppm = _compute_ppm_coefs(cfg)
    microslow = build_microslow_for_sulfate(cfg)

    T = jnp.asarray([298.0], dtype=DTYPE)
    p_cgs = jnp.asarray([1013.0 * 100.0 * float(RPA2CGS)], dtype=DTYPE)
    rho_air = float(p_cgs[0]) / (float(R_AIR) * float(T[0]))
    gc = jnp.asarray([[1e-7 * rho_air, 0.0]], dtype=DTYPE)
    env_d = _refresh_env(T, p_cgs, gc, cfg, ppm)

    # Very-dense lognormal so coag has something to do over 1800 s.
    grp = cfg.groups[0]
    r = np.asarray(grp.r)
    rmass = np.asarray(grp.rmass)
    log_mu = np.log(50e-7)         # 50 nm GMD
    log_s = np.log(1.6)
    pdf = (np.exp(-0.5 * ((np.log(r) - log_mu) / log_s) ** 2)
            / (r * log_s * np.sqrt(2 * np.pi)) * rmass)
    pdf = pdf / pdf.sum() * (1e3 * 1e-15 / np.sum(pdf * rmass))
    pc = jnp.asarray(pdf, dtype=DTYPE)[:, None] * 1e10  # boost density

    coag = CoagBundle(
        microslow_jit=microslow,
        ckernel=env_d["ckernel"],
        zmet=env_d["zmet"],
    )

    n_before = float(jnp.sum(pc))
    pc_after = apply_coag(pc, coag, dtime=1800.0)
    n_after = float(jnp.sum(pc_after))
    print(f"\n  N before: {n_before:.4e}  after: {n_after:.4e}  "
          f"Δ = {(n_after-n_before)/n_before:.2%}")
    assert n_after < n_before, (
        f"coag should reduce total N; got before={n_before:.3e}, "
        f"after={n_after:.3e}"
    )


def test_apply_coag_none_is_no_op():
    """Sanity: passing coag=None returns the input pc unchanged."""
    from carma_diffrax.coag_step import apply_coag
    pc = jnp.array([[1.0], [2.0], [3.0]], dtype=DTYPE)
    pc_out = apply_coag(pc, None, dtime=1800.0)
    np.testing.assert_array_equal(np.asarray(pc_out), np.asarray(pc))
