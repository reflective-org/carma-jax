"""Isolate the NaN-cotangent source in binary_nuc_zhao1995.

The Phase 3.1 gradient PoC found that `jax.jvp` through the full
diffrax pipeline produces NaN cotangents when nucleation is on, and
worked around it by setting `do_homogeneous=False`. This file
narrows down WHERE the NaN actually comes from so we can patch the
right `jnp.where` guards.
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

from carma.nucleation.sulfnucrate import binary_nuc_zhao1995


def _typical_inputs():
    """Test-point inputs from Phase 6.6c (strat39 first call)."""
    return dict(
        temp=jnp.asarray(250.0),
        weight_percent=jnp.asarray(72.5),
        rh=jnp.asarray(0.015),
        h2so4=jnp.asarray(2.7e8),
        h2so4_cgs=jnp.asarray(4.45e-14),
        h2o=jnp.asarray(4.2e14),
        h2o_cgs=jnp.asarray(1.25e-8),
        beta1=jnp.asarray(5807.7),
        gwtmol_h2so4=98.078479,
        gwtmol_h2o=18.01528,
    )


def test_forward_call_finite():
    """Baseline: forward call returns finite values."""
    inp = _typical_inputs()
    rate, m, r, ftry = binary_nuc_zhao1995(**inp)
    assert all(np.isfinite([float(rate), float(m), float(r), float(ftry)]))
    assert float(rate) > 0


@pytest.mark.parametrize("which", ["temp", "h2so4_cgs", "h2o_cgs"])
def test_jvp_no_nan(which):
    """jax.jvp w.r.t. each scalar input should return finite tangents."""
    inp = _typical_inputs()
    def f(x):
        new = dict(inp); new[which] = x
        return binary_nuc_zhao1995(**new)[0]   # just nucrate_cgs
    primal, tangent = jax.jvp(f, (inp[which],), (jnp.asarray(1.0),))
    print(f"\n  jvp w.r.t. {which}: primal={float(primal):.3e}, "
          f"tangent={float(tangent):.3e}")
    assert np.isfinite(float(primal)), f"primal non-finite for {which}"
    assert np.isfinite(float(tangent)), (
        f"tangent non-finite for {which} — this is the NaN-cotangent bug"
    )


@pytest.mark.parametrize("which", ["temp", "h2so4_cgs", "h2o_cgs"])
def test_grad_no_nan(which):
    """jax.grad (reverse mode) should also return finite cotangents."""
    inp = _typical_inputs()
    def f(x):
        new = dict(inp); new[which] = x
        return binary_nuc_zhao1995(**new)[0]
    g = jax.grad(f)(inp[which])
    print(f"  grad w.r.t. {which}: {float(g):.3e}")
    assert np.isfinite(float(g)), (
        f"grad non-finite for {which} — reverse-mode NaN"
    )


def test_grad_through_zero_branch():
    """At a saddle-fail input (h2so4 below ~1e4), forward returns 0.
    Check that the gradient through that branch is also finite (and
    likely 0), not NaN."""
    inp = _typical_inputs()
    inp["h2so4"] = jnp.asarray(1.0e3)   # below the 1e4 cutoff
    inp["h2so4_cgs"] = jnp.asarray(1.6e-19)
    def f(x):
        new = dict(inp); new["h2so4_cgs"] = x
        return binary_nuc_zhao1995(**new)[0]
    primal, tangent = jax.jvp(f, (inp["h2so4_cgs"],), (jnp.asarray(1.0),))
    print(f"\n  zero-branch jvp: primal={float(primal):.3e}, "
          f"tangent={float(tangent):.3e}")
    assert np.isfinite(float(primal))
    assert np.isfinite(float(tangent))


if __name__ == "__main__":
    test_forward_call_finite()
    for w in ["temp", "h2so4_cgs", "h2o_cgs"]:
        test_jvp_no_nan(w)
        test_grad_no_nan(w)
    test_grad_through_zero_branch()
    print("\nAll passed")
