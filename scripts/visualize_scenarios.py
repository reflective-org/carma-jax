"""Visualize the Phase 10.1 scenario ensemble — projection plots
across pairs of axes to confirm uniform coverage."""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))
from generate_sulfate_scenarios import _BOUNDS, _LOG_AXES, load_scenarios


OUTDIR = _ROOT / "plots" / "phase10_scenarios"


def _ax_label(k):
    units = {
        "T": "T [K]", "p": "p [hPa]", "rh": "RH",
        "h2so4_pptv": "H$_2$SO$_4$ [pptv]",
        "aerosol_mu_nm": "aerosol μ [nm]",
        "aerosol_sigma_g": "aerosol σ$_g$",
    }
    return units.get(k, k)


def _plot_axis(ax, x, k):
    if k in _LOG_AXES:
        ax.set_xscale("log")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    scenarios = load_scenarios(_ROOT / "data" / "sulfate_scenarios_1000.npz")

    keys = [k for k in _BOUNDS]
    n_keys = len(keys)

    # 1. 1-D histograms
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for i, k in enumerate(keys):
        ax = axes[i // 3, i % 3]
        if k in _LOG_AXES:
            ax.hist(np.log10(scenarios[k]), bins=30, color="C0", alpha=0.7)
            ax.set_xlabel("log10(" + _ax_label(k) + ")")
        else:
            ax.hist(scenarios[k], bins=30, color="C0", alpha=0.7)
            ax.set_xlabel(_ax_label(k))
        ax.set_ylabel("count")
        ax.grid(alpha=0.3)
    fig.suptitle("Phase 10 sulfate ensemble — 1-D coverage (n = 1000, seed = 42)")
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig1_1d_histograms.png", dpi=140)
    plt.close(fig)

    # 2. Key 2-D projections (T vs p, T vs RH, H2SO4 vs aerosol_mu)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    ax.scatter(scenarios["T"], scenarios["p"], s=8, c="C0", alpha=0.5)
    ax.set_xlabel("T [K]")
    ax.set_ylabel("p [hPa]")
    ax.set_title("T × p")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.scatter(scenarios["T"], scenarios["rh"], s=8, c="C1", alpha=0.5)
    ax.set_xlabel("T [K]")
    ax.set_ylabel("RH")
    ax.set_title("T × RH")
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.scatter(scenarios["h2so4_pptv"], scenarios["aerosol_mu_nm"],
               s=8, c="C2", alpha=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("H$_2$SO$_4$ [pptv]")
    ax.set_ylabel("aerosol μ [nm]")
    ax.set_title("H$_2$SO$_4$ × aerosol μ")
    ax.grid(alpha=0.3, which="both")

    fig.suptitle("Phase 10 sulfate ensemble — key 2-D projections")
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig2_2d_projections.png", dpi=140)
    plt.close(fig)

    print(f"Figures saved to {OUTDIR}")


if __name__ == "__main__":
    main()
