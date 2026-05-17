"""Cascade-of-evidence plot: max JAX-Fortran error across 4 configurations.

Shows how each isolation test progressively narrows the error source:

  1. baseline       — both processes, JAX adaptive retry
  2. no_coag        — growth only, JAX adaptive retry
  3. coag_only      — coag only (no H₂SO₄ source)
  4. prescribed     — both processes, JAX forced to Fortran's substep schedule

Each bar shows median and max relative error for total N, total M, and
bin-by-bin per active pair. Log y-axis to show the multi-order-of-magnitude
collapse from baseline (~10⁰) to prescribed (~10⁻⁹).

Output: benchmark_final/plots/10_error_cascade.png
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "scripts"))
from jax_ensemble import _minimal_config


def per_volume(pc_mmr, scens, rmass):
    from carma.constants import R_AIR
    rho_air = scens["p"] * 100.0 * 10.0 / (float(R_AIR) * scens["T"])
    return (pc_mmr * rho_air[:, None] / rmass[None, :],
            pc_mmr * rho_air[:, None] * 1e12)


def rel(F, J):
    F = np.asarray(F); J = np.asarray(J)
    denom = np.maximum(np.abs(F), np.abs(J))
    denom = np.where(denom > 1e-300, denom, 1.0)
    return np.abs(F - J) / denom


def stats(F_arr, J_arr, active=None):
    r = rel(F_arr, J_arr)
    if active is not None:
        r = r[active]
    if r.size == 0:
        return np.nan, np.nan, np.nan
    return np.median(r), np.percentile(r, 95), r.max()


def main():
    cfg = _minimal_config()
    rmass = np.asarray(cfg.groups[0].rmass)

    base_scens = ROOT / "scenarios" / "realistic_scenarios_100.npz"
    coag_scens = ROOT / "scenarios" / "realistic_scenarios_100_coag_only.npz"

    configs = [
        ("baseline\n(adaptive, both)",   base_scens,
         "outputs/fortran_outputs.npz",            "outputs/jax_outputs.npz"),
        ("no_coag\n(growth only)",       base_scens,
         "outputs/no_coag/fortran_outputs.npz",    "outputs/no_coag/jax_outputs.npz"),
        ("coag_only\n(no growth)",       coag_scens,
         "outputs/coag_only/fortran_outputs.npz",  "outputs/coag_only/jax_outputs.npz"),
        ("prescribed\n(forced match)",   base_scens,
         "outputs/prescribed/fortran_outputs.npz", "outputs/prescribed/jax_outputs.npz"),
    ]

    colors = ["#3b78b3", "#e08e3a", "#2ca02c", "#9467bd"]

    results = []   # list of dicts per config
    for label, scen_path, f_rel, j_rel in configs:
        scens = np.load(scen_path)
        F = np.load(ROOT / f_rel); J = np.load(ROOT / j_rel)
        num_F, mass_F = per_volume(F["pc_final"], scens, rmass)
        num_J, mass_J = per_volume(J["pc_final"], scens, rmass)
        N_F = num_F.sum(axis=1); N_J = num_J.sum(axis=1)
        M_F = mass_F.sum(axis=1); M_J = mass_J.sum(axis=1)
        nmed, _, nmax = stats(N_F, N_J)
        mmed, _, mmax = stats(M_F, M_J)
        bin_n = stats(num_F, num_J, active=num_F > 1e-3)
        bin_m = stats(mass_F, mass_J, active=mass_F > 1e-6)
        results.append(dict(label=label,
                             n_med=nmed, n_max=nmax,
                             m_med=mmed, m_max=mmax,
                             bin_n_max=bin_n[2], bin_m_max=bin_m[2]))

    labels = [r["label"] for r in results]
    x = np.arange(len(labels))
    width = 0.32

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # ---- Panel 1: Number errors ----
    ax = axes[0]
    n_meds = np.array([r["n_med"] for r in results])
    n_maxs = np.array([r["n_max"] for r in results])
    bin_n_maxs = np.array([r["bin_n_max"] for r in results])

    b1 = ax.bar(x - width, n_meds, width, color=colors, alpha=0.5,
                edgecolor="black", lw=0.7, label="total N (median)")
    b2 = ax.bar(x,         n_maxs, width, color=colors, alpha=0.85,
                edgecolor="black", lw=0.7, label="total N (max)")
    b3 = ax.bar(x + width, bin_n_maxs, width, color=colors, alpha=1.0,
                edgecolor="black", lw=0.7, hatch="///",
                label="bin-by-bin N (max)")

    for bars, vals in [(b1, n_meds), (b2, n_maxs), (b3, bin_n_maxs)]:
        for bar, v in zip(bars, vals):
            ax.annotate(f"{v:.1e}",
                        xy=(bar.get_x() + bar.get_width() / 2, v),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom",
                        fontsize=7, rotation=90)

    ax.set_yscale("log")
    ax.set_ylim(1e-11, 5.0)
    ax.axhline(1.0, color="red", ls="--", lw=0.8, alpha=0.6, label="100%")
    ax.axhline(0.01, color="green", ls="--", lw=0.8, alpha=0.6, label="1%")
    ax.axhline(1e-9, color="0.5", ls=":", lw=0.8, alpha=0.7, label="float64 ULP scale")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Relative error", fontsize=11)
    ax.set_title("Total NUMBER + bin-by-bin NUMBER\n"
                 "JAX vs Fortran relative error", fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3, which="both")
    ax.legend(loc="lower left", fontsize=8.5, ncol=2)

    # ---- Panel 2: Mass errors ----
    ax = axes[1]
    m_meds = np.array([r["m_med"] for r in results])
    m_maxs = np.array([r["m_max"] for r in results])
    bin_m_maxs = np.array([r["bin_m_max"] for r in results])

    b1 = ax.bar(x - width, m_meds, width, color=colors, alpha=0.5,
                edgecolor="black", lw=0.7, label="total M (median)")
    b2 = ax.bar(x,         m_maxs, width, color=colors, alpha=0.85,
                edgecolor="black", lw=0.7, label="total M (max)")
    b3 = ax.bar(x + width, bin_m_maxs, width, color=colors, alpha=1.0,
                edgecolor="black", lw=0.7, hatch="///",
                label="bin-by-bin M (max)")
    for bars, vals in [(b1, m_meds), (b2, m_maxs), (b3, bin_m_maxs)]:
        for bar, v in zip(bars, vals):
            ax.annotate(f"{v:.1e}",
                        xy=(bar.get_x() + bar.get_width() / 2, v),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom",
                        fontsize=7, rotation=90)
    ax.set_yscale("log")
    ax.set_ylim(1e-11, 5.0)
    ax.axhline(1.0, color="red", ls="--", lw=0.8, alpha=0.6, label="100%")
    ax.axhline(0.01, color="green", ls="--", lw=0.8, alpha=0.6, label="1%")
    ax.axhline(1e-9, color="0.5", ls=":", lw=0.8, alpha=0.7, label="float64 ULP scale")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Relative error", fontsize=11)
    ax.set_title("Total MASS + bin-by-bin MASS\n"
                 "JAX vs Fortran relative error", fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3, which="both")
    ax.legend(loc="lower left", fontsize=8.5, ncol=2)

    plt.suptitle(
        "Cascade of evidence: where does the JAX↔Fortran scatter come from?\n"
        "Reading right to left: each test isolates a possible source. "
        "When both sides do identical arithmetic on identical substep counts\n"
        "(rightmost bars), all errors collapse to the floating-point ULP scale.",
        fontsize=12, fontweight="bold", y=1.04,
    )

    out = ROOT / "plots" / "10_error_cascade.png"
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved: {out}")
    print()
    print("Summary values used:")
    for r in results:
        print(f"  {r['label'].replace(chr(10), ' ')}:")
        print(f"    N: med={r['n_med']:.2e}  max={r['n_max']:.2e}  bin_max={r['bin_n_max']:.2e}")
        print(f"    M: med={r['m_med']:.2e}  max={r['m_max']:.2e}  bin_max={r['bin_m_max']:.2e}")


if __name__ == "__main__":
    main()
