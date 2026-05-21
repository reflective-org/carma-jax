"""Unit tests for :mod:`carma_diffrax.ppm_advection`.

What's covered here
-------------------
- **Shape/finiteness**: output has shape ``(nbin,)`` and no NaN/Inf.
- **Constant n**: when ``pc/dm`` is uniform, every interior boundary
  value equals that constant to within roundoff. This is the most
  important sanity gate — PPM must reduce to the cell average when
  the spectrum is flat.
- **Upwind selection**: positive ``dmdt`` picks ``ar[b]``; negative
  picks ``al[b+1]``. We verify this with a constructed contrast (one
  bin spiked) and the rest of the parabola flat.
- **Monotonicity preservation**: in a monotonically-increasing
  spectrum, ``n_at_boundary[i]`` stays between
  ``dpc[i-1]`` and ``dpc[i+1]`` (no overshoot above the local max
  or below the local min of the four neighbours).
- **Trailing zero**: ``n_at_boundary[nbin-1] == 0`` always (there's
  no boundary above the top bin).
- **Spike test (smoke, not strict)**: a single localised peak under
  PPM yields a tighter distribution after one Euler step than under
  first-order upwind. The numerical value of "tighter" depends on
  the time step; this test only checks the *direction* of the bias.
"""
from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from carma.precision import DTYPE
from carma_diffrax.ppm_advection import ppm_n_at_boundary


def _toy_ppm_coefs(nbin, rmrat=2.0):
    """Build PPM coefs for a log-spaced mass grid with the given rmrat.

    Mirrors the formulae in ``scripts/jax_ensemble.py:_compute_ppm_coefs``
    so the tests don't have to import the heavy ``CarmaConfig`` stack.
    """
    rmass = rmrat ** np.arange(nbin, dtype=float)
    # dm[i] = rmassup[i] - rmasslow[i]; with rmrat=2 and the Fortran
    # convention rmassup[i] = 2·rmrat/(rmrat+1)·rmass[i], this gives
    # dm = (2/3)·rmass for rmrat=2.
    rmassup = (2.0 * rmrat / (rmrat + 1.0)) * rmass
    rmasslow = np.concatenate(([rmass[0] / np.sqrt(rmrat)], rmassup[:-1]))
    dm = rmassup - rmasslow

    pratt = np.zeros((3, nbin))
    prat = np.zeros((4, nbin))
    pden1 = np.ones(nbin)
    palr = np.zeros(4)
    for ibin in range(1, nbin - 1):
        dm_im1, dm_i, dm_ip1 = dm[ibin - 1], dm[ibin], dm[ibin + 1]
        pratt[0, ibin] = dm_i / (dm_im1 + dm_i + dm_ip1)
        pratt[1, ibin] = (2.0 * dm_im1 + dm_i) / (dm_ip1 + dm_i)
        pratt[2, ibin] = (2.0 * dm_ip1 + dm_i) / (dm_im1 + dm_i)
    for ibin in range(1, nbin - 2):
        dm_im1, dm_i = dm[ibin - 1], dm[ibin]
        dm_ip1 = dm[ibin + 1]
        dm_ip2 = dm[min(ibin + 2, nbin - 1)]
        prat[0, ibin] = dm_i / (dm_i + dm_ip1)
        prat[1, ibin] = 2.0 * dm_ip1 * dm_i / (dm_i + dm_ip1)
        prat[2, ibin] = (dm_im1 + dm_i) / (2.0 * dm_i + dm_ip1)
        prat[3, ibin] = (dm_ip2 + dm_ip1) / (2.0 * dm_ip1 + dm_i)
        pden1[ibin] = dm_im1 + dm_i + dm_ip1 + dm_ip2
    denom_low = rmass[1] - rmass[0]
    denom_high = rmass[nbin - 1] - rmass[nbin - 2]
    palr[0] = (rmassup[0] - rmass[0]) / denom_low
    palr[1] = (rmassup[0] / rmrat - rmass[0]) / denom_low
    palr[2] = (rmassup[nbin - 2] - rmass[nbin - 2]) / denom_high
    palr[3] = (rmassup[nbin - 1] - rmass[nbin - 2]) / denom_high

    return (
        jnp.asarray(dm, dtype=DTYPE),
        jnp.asarray(pratt, dtype=DTYPE),
        jnp.asarray(prat, dtype=DTYPE),
        jnp.asarray(pden1, dtype=DTYPE),
        jnp.asarray(palr, dtype=DTYPE),
    )


@pytest.fixture
def grid():
    nbin = 12
    dm, pratt, prat, pden1, palr = _toy_ppm_coefs(nbin)
    return dict(nbin=nbin, dm=dm, pratt=pratt, prat=prat,
                 pden1=pden1, palr=palr)


def test_shape_and_finite(grid):
    nbin = grid["nbin"]
    pc = jnp.linspace(1.0, 5.0, nbin, dtype=DTYPE) * grid["dm"]
    dmdt = jnp.ones(nbin, dtype=DTYPE)
    n_b = ppm_n_at_boundary(
        pc, grid["dm"], grid["pratt"], grid["prat"], grid["pden1"],
        grid["palr"], dmdt,
    )
    assert n_b.shape == (nbin,)
    assert jnp.all(jnp.isfinite(n_b))


def test_constant_n_recovers_constant(grid):
    """If n is uniform, every boundary value should equal n.

    This is the core sanity test: PPM on a flat spectrum is the
    identity (modulo the trailing zero at the top boundary).
    """
    nbin = grid["nbin"]
    c = DTYPE(3.14)
    pc = c * grid["dm"]                          # dpc = pc/dm = c
    dmdt = jnp.ones(nbin, dtype=DTYPE)
    n_b = ppm_n_at_boundary(
        pc, grid["dm"], grid["pratt"], grid["prat"], grid["pden1"],
        grid["palr"], dmdt,
    )
    # Interior boundaries — every one of them
    np.testing.assert_allclose(np.asarray(n_b[:-1]), float(c), rtol=1e-12)
    # Trailing zero
    assert float(n_b[-1]) == 0.0


def test_trailing_zero(grid):
    nbin = grid["nbin"]
    pc = jnp.linspace(1.0, 5.0, nbin, dtype=DTYPE) * grid["dm"]
    n_b = ppm_n_at_boundary(
        pc, grid["dm"], grid["pratt"], grid["prat"], grid["pden1"],
        grid["palr"], jnp.ones(nbin, dtype=DTYPE),
    )
    assert float(n_b[-1]) == 0.0


def test_upwind_selection(grid):
    """Sign of dmdt determines which side of the boundary we pick.

    Build a sharp jump: bin 5 has dpc=10, all others have dpc=1.
    For positive dmdt at boundary 4 (between bin 4 and 5), the upwind
    bin is 4 — n_boundary should be near 1 (the small side).
    For negative dmdt at the same boundary, n_boundary should be near
    10 (drawing from bin 5).
    """
    nbin = grid["nbin"]
    pc_low = jnp.ones(nbin, dtype=DTYPE) * grid["dm"]
    pc = pc_low.at[5].set(10.0 * grid["dm"][5])

    pos = ppm_n_at_boundary(
        pc, grid["dm"], grid["pratt"], grid["prat"], grid["pden1"],
        grid["palr"], jnp.ones(nbin, dtype=DTYPE),
    )
    neg = ppm_n_at_boundary(
        pc, grid["dm"], grid["pratt"], grid["prat"], grid["pden1"],
        grid["palr"], -jnp.ones(nbin, dtype=DTYPE),
    )
    # Positive dmdt at b=4 → upwind = bin 4 (low). Expect near 1.
    # Negative dmdt at b=4 → upwind = bin 5 (high). Expect near 10.
    assert float(pos[4]) < 3.0,  f"pos upwind at b=4 should be near 1, got {pos[4]}"
    assert float(neg[4]) > 5.0,  f"neg upwind at b=4 should be near 10, got {neg[4]}"


def test_monotonicity_in_monotone_field(grid):
    """For a monotonically increasing dpc, the PPM boundary value
    must stay within the [min, max] range of the immediate neighbours.

    This is the Colella-Woodward monotonicity guarantee. We test it
    on a smooth ramp; the limiter should NOT kick in (we expect
    near-third-order accuracy) but the bounds still hold.
    """
    nbin = grid["nbin"]
    dpc_target = jnp.arange(1, nbin + 1, dtype=DTYPE) ** 0.5
    pc = dpc_target * grid["dm"]
    n_b = ppm_n_at_boundary(
        pc, grid["dm"], grid["pratt"], grid["prat"], grid["pden1"],
        grid["palr"], jnp.ones(nbin, dtype=DTYPE),
    )
    # At interior boundaries 1..nbin-3, n_b[b] should sit between
    # dpc[b] and dpc[b+1] (since dpc is monotone increasing).
    dpc = pc / grid["dm"]
    for b in range(1, nbin - 2):
        lo = float(min(dpc[b], dpc[b + 1]))
        hi = float(max(dpc[b], dpc[b + 1]))
        # Allow a tiny epsilon for floating-point at the bound.
        eps = 1e-9 * hi
        assert lo - eps <= float(n_b[b]) <= hi + eps, (
            f"boundary {b}: n_b={n_b[b]:.6g} not in "
            f"[{lo:.6g}, {hi:.6g}]"
        )


def test_ppm_less_diffusive_than_upwind(grid):
    """One-step semi-discrete advection: PPM produces a sharper
    distribution than first-order upwind for a localized spike.

    This is a *direction-of-bias* check, not a strict numerical
    target. We set up a spike in bin 5, advect for one tiny Euler
    step with uniform dmdt, and measure peak retention (max
    pc[bin]/initial peak). PPM should keep more of the peak.
    """
    nbin = grid["nbin"]
    dm = grid["dm"]
    pc0 = jnp.zeros(nbin, dtype=DTYPE).at[5].set(1.0)
    dmdt = jnp.full(nbin, 0.1, dtype=DTYPE)

    # 1st-order upwind n_at_boundary
    n_upwind = jnp.where(dmdt > 0, pc0 / dm,
                         jnp.concatenate([pc0[1:], jnp.zeros(1, dtype=DTYPE)])
                         / jnp.concatenate([dm[1:], jnp.ones(1, dtype=DTYPE)]))
    n_upwind = n_upwind.at[-1].set(0.0)

    # PPM n_at_boundary
    n_ppm = ppm_n_at_boundary(
        pc0, dm, grid["pratt"], grid["prat"], grid["pden1"],
        grid["palr"], dmdt,
    )

    # Apply one Euler step: dpc/dt = F_below - F at each boundary
    def euler_step(n_at_boundary):
        F = dmdt * n_at_boundary
        F_below = jnp.concatenate([jnp.zeros(1, dtype=DTYPE), F[:-1]])
        return pc0 + (F_below - F) * 0.5    # tiny dt

    pc_upwind = euler_step(n_upwind)
    pc_ppm = euler_step(n_ppm)

    peak_upwind = float(pc_upwind.max())
    peak_ppm = float(pc_ppm.max())
    print(f"\n  peak retention: PPM={peak_ppm:.4f}  upwind={peak_upwind:.4f}")
    assert peak_ppm >= peak_upwind, (
        f"PPM peak ({peak_ppm}) should be >= upwind peak ({peak_upwind}); "
        "if not, the PPM reconstruction is more diffusive than upwind."
    )
