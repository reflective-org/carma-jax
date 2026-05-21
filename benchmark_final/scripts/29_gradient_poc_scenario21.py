"""Phase 3.1 demo: gradient of final aerosol mass vs h2so4_prod_rate.

For scenario 21 (T=216 K, p=27 hPa, RH=51 %, GMD=681 nm), sweeps
prod_rate around the nominal value and overlays:

  - the forward output ``f(prod_rate)`` — final total particle mass
  - the analytical autodiff gradient at each point — slope of f
  - finite-difference slopes for sanity

Generates ``plots/diff/grad/scen21_dM_dprod.png``.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from carma_diffrax.sensitivity import make_final_mass_wrt_prod_rate


SCENARIOS_PATH = (
    REPO / "benchmark_final" / "scenarios" / "realistic_scenarios_100.npz"
)


def main():
    d = np.load(SCENARIOS_PATH)
    scen = {k: float(d[k][21]) for k in
            ("T", "p", "rh", "h2so4_prod_rate", "M_total_ug_m3",
             "aerosol_mu_nm", "aerosol_sigma_g")}
    p0 = scen["h2so4_prod_rate"]
    print(f"\nScenario 21:")
    for k, v in scen.items():
        print(f"  {k:20s} = {v:.4e}")

    f = make_final_mass_wrt_prod_rate(scen, nstep=1, dtime=1800.0)

    # Sweep prod_rate around the nominal value
    factors = np.linspace(0.5, 1.5, 11)
    f_vals = np.zeros_like(factors)
    grad_vals = np.zeros_like(factors)

    print(f"\nSweep:")
    for i, fac in enumerate(factors):
        p = p0 * fac
        t = time.perf_counter()
        f_v, g_v = jax.jvp(f, (p,), (1.0,))
        dt = time.perf_counter() - t
        f_vals[i] = float(f_v)
        grad_vals[i] = float(g_v)
        print(f"  factor={fac:.2f}  p={p:.3e}  "
              f"f={f_vals[i]:.4e}  ∂f/∂p={grad_vals[i]:.4e}  ({dt:.1f}s)")

    # Finite-difference reference from f_vals (central differences)
    fd_grad = np.gradient(f_vals, factors * p0)
    rel_err = (grad_vals - fd_grad) / fd_grad

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(factors * p0, f_vals, "o-", color="#2c3e50", lw=2)
    ax1.axvline(p0, color="grey", linestyle=":", alpha=0.6,
                  label=f"nominal p₀ = {p0:.2e}")
    ax1.set_xlabel("H₂SO₄ production rate [molec/cm³/s]")
    ax1.set_ylabel("final total aerosol mass [g/cm³]")
    ax1.set_title("Scenario 21 — forward sweep")
    ax1.set_xscale("log")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    ax2.plot(factors * p0, grad_vals, "o-", color="#27ae60", lw=2,
              label="jax.jvp (autodiff)")
    ax2.plot(factors * p0, fd_grad, "s--", color="#c0392b", lw=1,
              alpha=0.7, label="finite difference")
    ax2.axvline(p0, color="grey", linestyle=":", alpha=0.6)
    ax2.set_xlabel("H₂SO₄ production rate [molec/cm³/s]")
    ax2.set_ylabel("∂(final mass)/∂(prod_rate)  [g·cm/molec]")
    ax2.set_title(f"Gradient — autodiff vs FD\n"
                  f"max |rel err| = {np.max(np.abs(rel_err[1:-1])):.2%}")
    ax2.set_xscale("log")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower right")

    plt.tight_layout()
    out = REPO / "plots" / "diff" / "grad" / "scen21_dM_dprod.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=110)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
