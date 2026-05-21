"""Phase 3.1: gradient PoC validation.

Validates ``jax.grad(f)(prod_rate)`` against finite difference for
``f(prod_rate) = final_total_aerosol_mass`` on scenario 21.

What we check:
  1. Forward pass works: ``f(prod_rate)`` returns a positive scalar.
  2. Analytical gradient: ``jax.grad(f)`` works without raising.
  3. FD validation at 1 % and 5 % perturbations agree with the
     analytical gradient within a target relative error.

How to interpret:
  - The forward pass is the same physics the production runner does.
  - The gradient says "if I increase the H2SO4 production rate by
    1 %, final aerosol mass changes by ``grad · 0.01 · prod_rate``".
  - Finite-difference vs analytical should agree to ~0.5 % at 1 %
    perturbation; larger error at 5 % is expected (nonlinearity).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from carma_diffrax.sensitivity import make_final_mass_wrt_prod_rate


SCENARIOS_PATH = (
    REPO / "benchmark_final" / "scenarios" / "realistic_scenarios_100.npz"
)


def _scen_21():
    d = np.load(SCENARIOS_PATH)
    return {k: float(d[k][21]) for k in
            ("T", "p", "rh", "h2so4_prod_rate", "M_total_ug_m3",
             "aerosol_mu_nm", "aerosol_sigma_g")}


@pytest.mark.slow
def test_forward_pass_runs():
    """Smoke: f(prod_rate) returns a positive finite scalar."""
    scen = _scen_21()
    f = make_final_mass_wrt_prod_rate(scen, nstep=1, dtime=1800.0)
    m = float(f(scen["h2so4_prod_rate"]))
    print(f"\n  scen 21 prod_rate = {scen['h2so4_prod_rate']:.4e} #/cm³/s")
    print(f"  final mass after 1×1800s = {m:.4e} g/cm³")
    assert m > 0
    assert np.isfinite(m)


@pytest.mark.slow
def test_gradient_vs_finite_difference():
    """Analytical autodiff agrees with FD at 1 % perturbation.

    Uses ``jax.jvp`` (forward-mode) rather than ``jax.grad`` (reverse-
    mode). Forward-mode is both more efficient for single-input
    single-output functions AND avoids diffrax's adjoint linear
    solver, which can hit NaN cotangents when the RHS contains
    ``jnp.where`` safety guards (the standard "double-where" pitfall).

    Two perturbations (1 % and 5 %) for sanity. Target: 1 % FD
    matches analytical to within ~5 % relative; 5 % FD matches to
    within ~20 % (nonlinearity tolerated).
    """
    scen = _scen_21()
    p0 = scen["h2so4_prod_rate"]
    f = make_final_mass_wrt_prod_rate(scen, nstep=1, dtime=1800.0)

    t = time.perf_counter()
    f0 = float(f(p0))
    t_forward = time.perf_counter() - t
    print(f"\n  baseline f(p0)         = {f0:.6e}  ({t_forward:.1f}s)")

    t = time.perf_counter()
    f0_jvp, grad_analytical_arr = jax.jvp(f, (p0,), (1.0,))
    grad_analytical = float(grad_analytical_arr)
    t_grad = time.perf_counter() - t
    print(f"  jax.jvp(f, p0)        = {grad_analytical:.6e}  "
          f"({t_grad:.1f}s, {t_grad/t_forward:.1f}× forward)")
    assert abs(float(f0_jvp) - f0) / f0 < 1e-10, (
        "jvp primal output should match plain forward call"
    )

    # 1 % perturbation
    p1 = p0 * 1.01
    f1 = float(f(p1))
    grad_fd_1pct = (f1 - f0) / (0.01 * p0)
    rel_1pct = (grad_fd_1pct - grad_analytical) / grad_analytical
    print(f"\n  FD @ 1 %:   ({f1:.6e} - {f0:.6e})/({0.01:.2f}·{p0:.2e})")
    print(f"             = {grad_fd_1pct:.6e}  "
          f"(rel err vs analytical = {rel_1pct:+.2%})")

    # 5 % perturbation
    p5 = p0 * 1.05
    f5 = float(f(p5))
    grad_fd_5pct = (f5 - f0) / (0.05 * p0)
    rel_5pct = (grad_fd_5pct - grad_analytical) / grad_analytical
    print(f"  FD @ 5 %:   = {grad_fd_5pct:.6e}  "
          f"(rel err vs analytical = {rel_5pct:+.2%})")

    # Gates: 1 % FD should match within 5 %, 5 % FD within 20 %.
    assert abs(rel_1pct) < 0.05, (
        f"1 % FD ({grad_fd_1pct:.3e}) disagrees with analytical "
        f"({grad_analytical:.3e}) by {rel_1pct:+.2%} — expected < 5 %"
    )
    assert abs(rel_5pct) < 0.20, (
        f"5 % FD ({grad_fd_5pct:.3e}) disagrees with analytical "
        f"({grad_analytical:.3e}) by {rel_5pct:+.2%} — expected < 20 %"
    )


if __name__ == "__main__":
    # Allow `python tests/diffrax/test_gradient_poc.py` for ad-hoc runs.
    test_forward_pass_runs()
    test_gradient_vs_finite_difference()
