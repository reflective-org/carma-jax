"""Phase 3.2: Multi-input sensitivity of final aerosol mass.

Computes ∂(final aerosol mass)/∂(each of 5 inputs) for scenario 21
using forward-mode jax.jvp. Each input gets its own jvp call.

Inputs:
  - prod_rate    [molec/cm³/s]   H2SO4 production rate
  - T0           [K]             initial temperature
  - mu_nm        [nm]            lognormal GMD
  - sigma_g      [dimensionless] lognormal GSD
  - M_ug_m3      [µg/m³]         initial total aerosol mass

Output:
  - final_mass   [g/cm³]         sum of pc[b] · rmass[b] after dtime

Validates each analytical jvp against finite difference at 1 %
perturbation.

Produces a bar chart of relative sensitivities (elasticities):
    E_i = (∂f/∂x_i) × (x_i / f)
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from carma_diffrax.sensitivity import make_final_mass_5d


def main():
    d = np.load(REPO / "benchmark_final" / "scenarios"
                / "realistic_scenarios_100.npz")
    scen = {k: float(d[k][21]) for k in
            ("T", "p", "rh", "h2so4_prod_rate", "M_total_ug_m3",
             "aerosol_mu_nm", "aerosol_sigma_g")}
    print(f"Scenario 21:")
    for k, v in scen.items():
        print(f"  {k:20s} = {v:.4e}")

    f = make_final_mass_5d(scen, nstep=1, dtime=1800.0,
                            do_homogeneous=True)

    # Baseline inputs
    x0 = (
        scen["h2so4_prod_rate"],
        scen["T"],
        scen["aerosol_mu_nm"],
        scen["aerosol_sigma_g"],
        scen["M_total_ug_m3"],
    )
    names = ["prod_rate", "T0", "mu_nm", "sigma_g", "M_ug_m3"]

    t0 = time.perf_counter()
    f0 = float(f(*x0))
    t_fwd = time.perf_counter() - t0
    print(f"\nBaseline final mass = {f0:.6e} g/cm³  ({t_fwd:.1f}s)")

    # 5 jvp calls — one tangent direction at a time.
    print(f"\nForward-mode jvp w.r.t. each input:")
    grads = {}
    for i, name in enumerate(names):
        tangents = [0.0] * 5
        tangents[i] = 1.0
        t0 = time.perf_counter()
        _, g = jax.jvp(f, x0, tuple(tangents))
        dt = time.perf_counter() - t0
        grads[name] = float(g)
        # Elasticity = (∂f/∂x_i) × (x_i / f)
        elast = grads[name] * x0[i] / f0
        print(f"  ∂f/∂{name:10s} = {grads[name]:+.4e}  "
              f"E = {elast:+.4f}  ({dt:.1f}s)")

    # Finite-difference validation at 1 % perturbation
    print(f"\nFinite-difference validation at 1 % perturbation:")
    for i, name in enumerate(names):
        x_pert = list(x0)
        delta = 0.01 * x0[i]
        x_pert[i] = x0[i] + delta
        f1 = float(f(*x_pert))
        fd = (f1 - f0) / delta
        rel = (fd - grads[name]) / grads[name] * 100 if grads[name] != 0 else float('nan')
        print(f"  {name:10s}: jvp={grads[name]:+.4e}  fd={fd:+.4e}  "
              f"rel_err={rel:+.2f}%")

    # Save and plot elasticities.
    out_dir = REPO / "benchmark_final" / "plots" / "phase3_2"
    out_dir.mkdir(parents=True, exist_ok=True)
    elasts = [grads[n] * x0[i] / f0 for i, n in enumerate(names)]
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["tab:blue" if e >= 0 else "tab:red" for e in elasts]
    bars = ax.bar(names, elasts, color=colors, alpha=0.85)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("Elasticity  E_i = (∂f/∂x_i) · (x_i / f)")
    ax.set_xlabel("input parameter")
    ax.set_title(f"Phase 3.2 — Sensitivity of final aerosol mass to 5 inputs\n"
                  f"Scenario 21, nstep=1 × 1800 s, "
                  f"baseline f₀ = {f0:.3e} g/cm³",
                  fontsize=11)
    for bar, e in zip(bars, elasts):
        ax.text(bar.get_x() + bar.get_width() / 2, e,
                 f"{e:+.3f}", ha="center",
                 va="bottom" if e >= 0 else "top", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out_path = out_dir / "sensitivities_scen21.png"
    plt.savefig(out_path, dpi=110)
    print(f"\nSaved {out_path}")

    np.savez(out_dir / "sensitivities_scen21.npz",
             names=np.array(names), x0=np.array(x0),
             grads=np.array([grads[n] for n in names]),
             elasticities=np.array(elasts), f0=f0)


if __name__ == "__main__":
    main()
