"""Phase 3.3: Batched sensitivities across 100 scenarios.

For each of the 100 v1 scenarios, compute the 5 sensitivities
(∂f/∂prod_rate, ∂f/∂T0, ∂f/∂mu, ∂f/∂sigma, ∂f/∂M) using
forward-mode jax.jacfwd. Then plot elasticities vs scenario type
(strat vs trop, separated by pressure level).

Uses make_final_mass_7d so all per-scenario inputs are JAX-traceable;
this lets vmap batch the entire ensemble through a single JIT trace.

Output:
  benchmark_final/plots/phase3_3/
    1_elasticities_by_scenario.png  — bar chart per scenario
    2_elasticities_vs_pressure.png  — scatter
    3_summary.png                    — distribution histograms
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

from carma_diffrax.sensitivity import make_final_mass_7d


def main():
    print("Phase 3.3 — batched sensitivities over 100 scenarios")

    d = np.load(REPO / "benchmark_final" / "scenarios"
                 / "realistic_scenarios_100.npz")
    n = int(d["_n"])
    print(f"Loaded {n} scenarios")

    # Build the 7-input forward function once. JIT cache will be hot
    # after the first call.
    f = make_final_mass_7d(nstep=48, dtime=1800.0, do_homogeneous=True)

    # jacfwd internally vmaps over basis vectors, which hits a NaN in the
    # linear solver for unknown reason. Sidestep: do 5 explicit jvps.
    def jac_f(*args):
        out = []
        for k in range(5):
            tangents = [0.0] * 7
            tangents[k] = 1.0
            _, g = jax.jvp(f, args, tuple(tangents))
            out.append(g)
        return tuple(out)

    # First-call JIT compile timing
    print(f"\nFirst-call JIT compile (scenario 0)...")
    t0 = time.perf_counter()
    s = {k: float(d[k][0]) for k in ("h2so4_prod_rate", "T",
                                       "aerosol_mu_nm", "aerosol_sigma_g",
                                       "M_total_ug_m3", "p", "rh")}
    f0 = float(f(s["h2so4_prod_rate"], s["T"], s["aerosol_mu_nm"],
                  s["aerosol_sigma_g"], s["M_total_ug_m3"], s["p"], s["rh"]))
    print(f"  baseline f0 (scen 0) = {f0:.3e}  ({time.perf_counter()-t0:.1f}s)")

    t0 = time.perf_counter()
    grads0 = jac_f(s["h2so4_prod_rate"], s["T"], s["aerosol_mu_nm"],
                    s["aerosol_sigma_g"], s["M_total_ug_m3"], s["p"], s["rh"])
    print(f"  jacfwd (scen 0) = {[float(g) for g in grads0]}  "
          f"({time.perf_counter()-t0:.1f}s, JIT compile)")

    # Loop over all scenarios. Some scenarios push the solver into
    # Jacobian-degenerate states that produce NaN in linear solver under
    # autodiff; we catch and record these as 'failed' rather than
    # crashing the whole sweep.
    print(f"\nLooping over all {n} scenarios:")
    elasts = np.full((n, 5), np.nan)
    f_vals = np.full(n, np.nan)
    failed_idx: list[int] = []
    t_start = time.perf_counter()
    for i in range(n):
        prod = float(d["h2so4_prod_rate"][i])
        T0 = float(d["T"][i])
        mu = float(d["aerosol_mu_nm"][i])
        sg = float(d["aerosol_sigma_g"][i])
        M = float(d["M_total_ug_m3"][i])
        p = float(d["p"][i])
        rh = float(d["rh"][i])
        try:
            fi = float(f(prod, T0, mu, sg, M, p, rh))
            gs = jac_f(prod, T0, mu, sg, M, p, rh)
            xs = [prod, T0, mu, sg, M]
            for k in range(5):
                elasts[i, k] = float(gs[k]) * xs[k] / fi
            f_vals[i] = fi
        except Exception as e:
            failed_idx.append(i)
            err_short = f"{type(e).__name__}: {str(e)[:60]}"
            print(f"  [scen {i}] FAILED: {err_short}")
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n}] elapsed {time.perf_counter()-t_start:.1f}s, "
                  f"failures so far: {len(failed_idx)}")
    t_total = time.perf_counter() - t_start
    n_ok = n - len(failed_idx)
    print(f"\nTotal: {t_total:.1f}s for {n} scenarios "
          f"({t_total/n:.1f}s/scen), {n_ok}/{n} succeeded")
    if failed_idx:
        print(f"Failed scenario indices: {failed_idx}")

    # Save raw
    out_dir = REPO / "benchmark_final" / "plots" / "phase3_3"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        REPO / "benchmark_final" / "outputs" / "phase3_3" / "elasticities.npz",
        elasticities=elasts, f_vals=f_vals,
        names=np.array(["prod_rate", "T0", "mu_nm", "sigma_g", "M_ug_m3"]),
        p_hPa=d["p"], T=d["T"],
    )

    # ---- Plot 1: bar chart per scenario, faceted by input ----
    fig, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=True)
    names = ["prod_rate", "T0", "mu_nm", "sigma_g", "M_ug_m3"]
    for i, (ax, name) in enumerate(zip(axes, names)):
        colors = ["tab:blue" if e >= 0 else "tab:red" for e in elasts[:, i]]
        ax.bar(np.arange(n), elasts[:, i], color=colors, alpha=0.85)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_ylabel(f"E({name})")
        ax.grid(True, alpha=0.3, axis="y")
    axes[-1].set_xlabel("scenario index")
    fig.suptitle("Phase 3.3 — Per-scenario elasticities (24 h, full physics)",
                  fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.99))
    plt.savefig(out_dir / "1_elasticities_by_scenario.png", dpi=110)
    plt.close()
    print(f"Saved 1_elasticities_by_scenario.png")

    # ---- Plot 2: elasticity vs pressure (regime split) ----
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
    for i, (ax, name) in enumerate(zip(axes, names)):
        ax.scatter(d["p"], elasts[:, i], c=d["T"], cmap="coolwarm",
                    alpha=0.7, s=20)
        ax.set_xlabel("p [hPa]")
        ax.set_xscale("log")
        ax.set_title(f"E({name})")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="black", lw=0.5)
    axes[0].set_ylabel("Elasticity")
    fig.suptitle("Phase 3.3 — Elasticity vs pressure (color = T)", fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(out_dir / "2_elasticities_vs_pressure.png", dpi=110)
    plt.close()
    print(f"Saved 2_elasticities_vs_pressure.png")

    # ---- Plot 3: histograms of elasticities ----
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for ax, name, i in zip(axes, names, range(5)):
        ax.hist(elasts[:, i], bins=30, alpha=0.8)
        ax.axvline(0, color="black", lw=0.5)
        ax.set_title(f"E({name})\nmedian={np.nanmedian(elasts[:, i]):+.3f}")
        ax.set_xlabel("elasticity")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Phase 3.3 — Distribution of elasticities across 100 scenarios",
                  fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(out_dir / "3_summary.png", dpi=110)
    plt.close()
    print(f"Saved 3_summary.png")

    print(f"\nMedian elasticities (over {n_ok} successful scenarios):")
    for i, name in enumerate(names):
        col = elasts[~np.isnan(elasts[:, i]), i]
        if len(col) == 0:
            continue
        print(f"  {name:>11}: {np.median(col):+.4f}  "
              f"(spread: [{np.percentile(col, 5):+.3f}, "
              f"{np.percentile(col, 95):+.3f}])")


if __name__ == "__main__":
    main()
