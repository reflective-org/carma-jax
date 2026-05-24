"""Process-isolation comparison: which process is the source of JAX-Fortran scatter?

Three configurations of the SAME 100 scenarios:
  1. baseline    — both growth+nucleation AND coagulation active
  2. no_coag     — only growth/nucleation/evap (coag disabled)
  3. coag_only   — only coagulation (prod_rate=0 so no H₂SO₄ source)

For each config we compare JAX vs Fortran total N, total M, and per-bin error.
If errors are similar in baseline and no_coag → coag isn't the main contributor.
If coag_only has near-zero errors → coag is well-ported and not the source.

Outputs benchmark_final/plots/07_process_isolation.png.
"""
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "scripts"))
from jax_ensemble import _minimal_config


def bin_diameters_nm(nbin=38, rmin_cm=2e-8, rmrat=2.0, rho=1.923):
    vmin = (4.0 / 3.0) * math.pi * rmin_cm**3 * rho
    rmass = vmin * rmrat ** np.arange(nbin)
    r = (3.0 * rmass / (4.0 * math.pi * rho)) ** (1.0 / 3.0)
    return 2.0 * r * 1e7


def load_set(label, scen_path, F_path, J_path):
    scens = np.load(scen_path)
    F = np.load(F_path)
    J = np.load(J_path)
    return dict(label=label, scens=scens, F=F, J=J)


def per_volume(pc_mmr, scens, rmass):
    from carma.constants import R_AIR
    rho_air = scens["p"] * 100.0 * 10.0 / (float(R_AIR) * scens["T"])
    num = pc_mmr * rho_air[:, None] / rmass[None, :]
    mass = pc_mmr * rho_air[:, None] * 1e12
    return num, mass


def rel_err(F_, J_):
    F_ = np.asarray(F_); J_ = np.asarray(J_)
    denom = np.maximum(np.abs(F_), np.abs(J_))
    denom = np.where(denom > 1e-300, denom, 1.0)
    return np.abs(F_ - J_) / denom


def main():
    cfg = _minimal_config()
    rmass = np.asarray(cfg.groups[0].rmass)
    d_nm = bin_diameters_nm(cfg.nbin)

    base_scens = ROOT / "scenarios" / "realistic_scenarios_100.npz"
    coag_scens = ROOT / "scenarios" / "realistic_scenarios_100_coag_only.npz"

    runs = [
        load_set("baseline",
                 base_scens,
                 ROOT / "outputs" / "fortran_outputs.npz",
                 ROOT / "outputs" / "jax_outputs.npz"),
        load_set("no_coag (growth only)",
                 base_scens,
                 ROOT / "outputs" / "no_coag" / "fortran_outputs.npz",
                 ROOT / "outputs" / "no_coag" / "jax_outputs.npz"),
        load_set("coag_only (no growth)",
                 coag_scens,
                 ROOT / "outputs" / "coag_only" / "fortran_outputs.npz",
                 ROOT / "outputs" / "coag_only" / "jax_outputs.npz"),
    ]

    # Compute per-run stats
    for r in runs:
        num_F, mass_F = per_volume(r["F"]["pc_final"], r["scens"], rmass)
        num_J, mass_J = per_volume(r["J"]["pc_final"], r["scens"], rmass)
        r["num_F"] = num_F; r["num_J"] = num_J
        r["mass_F"] = mass_F; r["mass_J"] = mass_J
        N_F = num_F.sum(axis=1); N_J = num_J.sum(axis=1)
        M_F = mass_F.sum(axis=1); M_J = mass_J.sum(axis=1)
        r["N_err"] = rel_err(N_F, N_J)
        r["M_err"] = rel_err(M_F, M_J)

    print("="*78)
    print("PROCESS ISOLATION RESULTS — JAX vs Fortran error")
    print("="*78)
    print(f"{'config':<25} {'N median':>10} {'N max':>10} {'M median':>10} {'M max':>10}")
    for r in runs:
        n_med = np.median(r["N_err"]); n_max = r["N_err"].max()
        m_med = np.median(r["M_err"]); m_max = r["M_err"].max()
        print(f"{r['label']:<25} {n_med:>10.2e} {n_max:>10.2e} "
              f"{m_med:>10.2e} {m_max:>10.2e}")

    # Three-panel plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    colors = ["steelblue", "darkorange", "forestgreen"]

    # Row 1: total number scatter
    for i, r in enumerate(runs):
        ax = axes[0, i]
        N_F = r["num_F"].sum(axis=1); N_J = r["num_J"].sum(axis=1)
        mask = (N_F > 0) & (N_J > 0)
        if mask.sum() == 0:
            ax.text(0.5, 0.5, "no nonzero data", ha="center", va="center",
                    transform=ax.transAxes)
        else:
            ax.scatter(N_F[mask], N_J[mask], s=18, alpha=0.7,
                        color=colors[i], edgecolor="black", lw=0.4)
            lo, hi = N_F[mask].min(), N_F[mask].max()
            ref = np.logspace(np.log10(max(lo, 1e-30)), np.log10(max(hi, 1e-20)), 5)
            ax.plot(ref, ref, "r--", lw=1)
            ax.plot(ref, ref*1.1, "0.6", ls=":", lw=0.8)
            ax.plot(ref, ref*0.9, "0.6", ls=":", lw=0.8)
            ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Fortran N_total [#/cm³]", fontsize=10)
        ax.set_ylabel("JAX N_total [#/cm³]", fontsize=10)
        n_med = np.median(r["N_err"]); n_max = r["N_err"].max()
        ax.set_title(f"{r['label']}\nN — median {n_med:.1e}, max {n_max:.1e}",
                     fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3, which="both")

    # Row 2: total mass scatter
    for i, r in enumerate(runs):
        ax = axes[1, i]
        M_F = r["mass_F"].sum(axis=1); M_J = r["mass_J"].sum(axis=1)
        mask = (M_F > 0) & (M_J > 0)
        if mask.sum() == 0:
            ax.text(0.5, 0.5, "no nonzero data", ha="center", va="center",
                    transform=ax.transAxes)
        else:
            ax.scatter(M_F[mask], M_J[mask], s=18, alpha=0.7,
                        color=colors[i], edgecolor="black", lw=0.4)
            lo, hi = M_F[mask].min(), M_F[mask].max()
            ref = np.logspace(np.log10(max(lo, 1e-30)), np.log10(max(hi, 1e-20)), 5)
            ax.plot(ref, ref, "r--", lw=1)
            ax.plot(ref, ref*1.1, "0.6", ls=":", lw=0.8)
            ax.plot(ref, ref*0.9, "0.6", ls=":", lw=0.8)
            ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Fortran M_total [µg/m³]", fontsize=10)
        ax.set_ylabel("JAX M_total [µg/m³]", fontsize=10)
        m_med = np.median(r["M_err"]); m_max = r["M_err"].max()
        ax.set_title(f"{r['label']}\nM — median {m_med:.1e}, max {m_max:.1e}",
                     fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3, which="both")

    plt.suptitle(
        "Process-isolation experiment: JAX vs Fortran parity\n"
        "100 scenarios, 60 s × 1440 steps, sulfate-only, 24 h",
        fontsize=12, fontweight="bold", y=1.00,
    )
    out = ROOT / "plots" / "07_process_isolation.png"
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
