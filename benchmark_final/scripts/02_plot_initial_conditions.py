"""Plot the input parameter distributions for the realistic ensemble.

Produces benchmark_final/plots/01_initial_conditions.png — a grid of
histograms + a T-p scatter showing the atmospheric placement.
For inclusion in slides / report.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def main():
    scens = np.load(ROOT / "scenarios" / "realistic_scenarios_100.npz")
    n = int(scens["_n"])

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.35)

    # (variable_key, axis_label, log?, bin_strategy)
    panels = [
        ("altitude_km",      "Altitude [km]",                  False, np.linspace(0, 30, 21)),
        ("T",                "Temperature [K]",                False, np.linspace(180, 320, 21)),
        ("p",                "Pressure [hPa]",                 True,  np.logspace(0.5, 3.1, 21)),
        ("rh",               "Relative humidity",              False, np.linspace(0, 0.6, 21)),
        ("h2so4_prod_rate",  "H₂SO₄ production [molec/cm³/s]", True,  np.logspace(1, 9, 21)),
        ("M_total_ug_m3",    "Initial mass [µg/m³]",            True,  np.logspace(-1, 1.7, 21)),
        ("aerosol_mu_nm",    "Seed mode diameter [nm]",         True,  np.logspace(1, 2.9, 21)),
        ("aerosol_sigma_g",  "Seed σ_g",                        False, np.linspace(1.0, 2.5, 21)),
        ("N_total_cm3",      "Initial N_total [#/cm³]",         True,  np.logspace(-1, 6, 25)),
    ]

    for i, (key, label, logx, bins) in enumerate(panels):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        ax.hist(scens[key], bins=bins, color="steelblue", edgecolor="white", lw=0.5)
        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel("# scenarios", fontsize=9)
        if logx:
            ax.set_xscale("log")
        ax.grid(True, alpha=0.3, which="both")

    # T-p scatter, last 3 cells (atmospheric placement)
    ax = fig.add_subplot(gs[2, 1:])
    sc = ax.scatter(scens["T"], scens["p"], c=scens["altitude_km"],
                     cmap="viridis", s=30, edgecolor="k", lw=0.4)
    ax.invert_yaxis()
    ax.set_yscale("log")
    ax.set_xlabel("Temperature [K]", fontsize=10)
    ax.set_ylabel("Pressure [hPa]  (log, inverted)", fontsize=10)
    ax.set_title("Atmospheric (T, p) placement — coloured by altitude", fontsize=10)
    cb = plt.colorbar(sc, ax=ax, shrink=0.85, label="Altitude [km]")

    # Region labels
    ax.axhline(100, color="red", ls="--", lw=1, alpha=0.6)
    ax.text(310, 50,  "Stratosphere", fontsize=9, color="red", ha="right")
    ax.text(310, 700, "Troposphere",  fontsize=9, color="red", ha="right")
    ax.grid(True, alpha=0.3, which="both")

    fig.suptitle(
        f"CARMA-JAX realistic benchmark — input parameter distributions "
        f"({n} scenarios)\n"
        f"50% stratosphere (12–30 km)  /  50% troposphere (0–12 km)  "
        f"— US Std Atm 1976 + ±5 K perturbation",
        fontsize=12, fontweight="bold", y=0.99,
    )

    out = ROOT / "plots" / "01_initial_conditions.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
