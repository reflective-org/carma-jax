"""Visual JAX-vs-Fortran comparison plots (Phase 10.4 diagnostic).

Three figures:

1. ``fig2_per_bin_distributions.png`` — pc(bin) for JAX and Fortran on
   representative scenarios: best-match, median-error, worst-case.
2. ``fig3_gc_scatter.png`` — scatter of JAX gc_h2so4 vs Fortran
   gc_h2so4 across all 1000 scenarios; 1:1 line for reference.
3. ``fig4_total_mass_scatter.png`` — same scatter for total particle
   mass (Σ pc·rmass).
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))
from generate_sulfate_scenarios import load_scenarios

OUTDIR = _ROOT / "plots" / "phase10_parity"


def _bin_radii_um():
    rho = 1.923
    rmin = 2.0e-8                  # cm
    rmrat = 2.0
    nbin = 38
    vmin = (4.0 / 3.0) * np.pi * rmin**3 * rho
    rmass = vmin * rmrat ** np.arange(nbin)
    r = (3.0 * rmass / (4.0 * np.pi * rho)) ** (1.0 / 3.0)
    return r * 1e4                  # cm → μm


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    j = np.load(_ROOT / "data" / "sulfate_jax_outputs.npz")
    f = np.load(_ROOT / "data" / "sulfate_fortran_outputs.npz")
    s = load_scenarios(_ROOT / "data" / "sulfate_scenarios_1000.npz")

    n = j["T_final"].shape[0]
    r_um = _bin_radii_um()

    # --- per-scenario error to pick representatives ---
    eps = 1e-30
    rel = np.abs(j["pc_final"] - f["pc_final"]) / np.maximum(
        np.maximum(np.abs(j["pc_final"]), np.abs(f["pc_final"])), eps)
    err = np.median(rel, axis=1)

    best = int(np.argmin(err))
    worst = int(np.argmax(err))
    median = int(np.argsort(err)[len(err) // 2])

    # --- Figure 2: per-bin distributions for the three representative scenarios ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    titles = [f"best (idx {best}, err {err[best]:.2e})",
              f"median (idx {median}, err {err[median]:.2e})",
              f"worst (idx {worst}, err {err[worst]:.2e})"]
    for ax, idx, title in zip(axes, [best, median, worst], titles):
        ax.bar(np.arange(38) - 0.2, np.maximum(j["pc_final"][idx], 1e-25),
               width=0.4, color="C0", alpha=0.8, label="JAX")
        ax.bar(np.arange(38) + 0.2, np.maximum(f["pc_final"][idx], 1e-25),
               width=0.4, color="C3", alpha=0.8, label="Fortran")
        ax.set_yscale("log")
        ax.set_xlabel("bin index")
        ax.set_title(title + "\n"
                     f"T={s['T'][idx]:.0f}K, p={s['p'][idx]:.0f}hPa, "
                     f"RH={s['rh'][idx]:.2f}, "
                     f"H2SO4={s['h2so4_pptv'][idx]:.2f}pptv")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("pc per bin [g/g]")
    fig.suptitle("Per-bin mass distributions: JAX vs Fortran")
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig2_per_bin_distributions.png", dpi=140)
    plt.close(fig)

    # --- Figure 3: gc_h2so4 scatter ---
    fig, ax = plt.subplots(figsize=(7, 7))
    gc_j = np.maximum(j["gc_h2so4_final"], 1e-30)
    gc_f = np.maximum(f["gc_h2so4_final"], 1e-30)
    ax.scatter(gc_f, gc_j, s=8, alpha=0.4, c="C0")
    lo, hi = 1e-30, 1e-3
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6, label="1:1")
    ax.fill_between([lo, hi], [lo*0.99, hi*0.99], [lo*1.01, hi*1.01],
                     color="orange", alpha=0.2, label="±1% band")
    ax.set_xlabel("Fortran gc_h2so4 [g/g]")
    ax.set_ylabel("JAX gc_h2so4 [g/g]")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title("H$_2$SO$_4$ final gas concentration: JAX vs Fortran (n=1000)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig3_gc_scatter.png", dpi=140)
    plt.close(fig)

    # --- Figure 4: total particle mass scatter ---
    tot_j = j["pc_final"].sum(axis=1)
    tot_f = f["pc_final"].sum(axis=1)
    tot_j = np.maximum(tot_j, 1e-30)
    tot_f = np.maximum(tot_f, 1e-30)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(tot_f, tot_j, s=8, alpha=0.4, c="C2")
    lo, hi = 1e-25, 1e1
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6, label="1:1")
    # Realistic stratospheric sulfate band
    ax.axvspan(1e-9, 1e-7, color="green", alpha=0.1,
               label="realistic stratosphere")
    ax.axhspan(1e-9, 1e-7, color="green", alpha=0.05)
    ax.set_xlabel("Fortran total pc mmr [g/g]")
    ax.set_ylabel("JAX total pc mmr [g/g]")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title("Total particle mass: JAX vs Fortran (n=1000)")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig4_total_mass_scatter.png", dpi=140)
    plt.close(fig)

    print(f"Figures saved to {OUTDIR}")
    print(f"  best  scenario: idx {best}, error {err[best]:.3e}")
    print(f"  median scenario: idx {median}, error {err[median]:.3e}")
    print(f"  worst scenario: idx {worst}, error {err[worst]:.3e}")
    print(f"\nFortran total mmr percentiles: "
          f"p25={np.percentile(tot_f, 25):.2e}, "
          f"p50={np.percentile(tot_f, 50):.2e}, "
          f"p75={np.percentile(tot_f, 75):.2e}, "
          f"p99={np.percentile(tot_f, 99):.2e}")
    print(f"JAX     total mmr percentiles: "
          f"p25={np.percentile(tot_j, 25):.2e}, "
          f"p50={np.percentile(tot_j, 50):.2e}, "
          f"p75={np.percentile(tot_j, 75):.2e}, "
          f"p99={np.percentile(tot_j, 99):.2e}")


if __name__ == "__main__":
    main()
