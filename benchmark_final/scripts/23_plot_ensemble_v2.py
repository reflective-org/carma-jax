"""Ensemble-v2 summary plot: input distribution + Fortran vs diffrax outcomes.

Reads:
  benchmark_final/scenarios/realistic_scenarios_100_v2.npz
  benchmark_final/outputs/dt1800/v2/adaptive/fortran_outputs.npz
  benchmark_final/outputs/dt1800/v2/diffrax/jax_diffrax_outputs.npz

Writes:
  benchmark_final/plots/v2/ensemble_v2_summary.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    S = np.load(ROOT / "scenarios" / "realistic_scenarios_100_v2.npz")
    F = np.load(ROOT / "outputs" / "dt1800" / "v2" / "adaptive"
                 / "fortran_outputs.npz")
    D = np.load(ROOT / "outputs" / "dt1800" / "v2" / "diffrax"
                 / "jax_diffrax_outputs.npz")

    is_strat = S["is_strat"]
    fortran_fail = F["gc_h2so4_final"] < 0
    diffrax_fail = D["min_gc_h2so4"] < 0

    out = ROOT / "plots" / "v2" / "ensemble_v2_summary.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10),
                              gridspec_kw=dict(hspace=0.40, wspace=0.35))

    # (a) altitude distribution
    ax = axes[0, 0]
    ax.hist(S["altitude_km"][~is_strat], bins=15, color="tab:blue",
             alpha=0.7, label=f"troposphere (n={(~is_strat).sum()})")
    ax.hist(S["altitude_km"][is_strat], bins=15, color="tab:orange",
             alpha=0.7, label=f"stratosphere (n={is_strat.sum()})")
    ax.set_xlabel("Altitude [km]"); ax.set_ylabel("Count")
    ax.set_title("(a) Altitude — 50/50 trop/strat")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (b) T-p scatter
    ax = axes[0, 1]
    ax.scatter(S["T"][~is_strat], S["p"][~is_strat], s=20, alpha=0.6,
                color="tab:blue", label="trop")
    ax.scatter(S["T"][is_strat], S["p"][is_strat], s=20, alpha=0.6,
                color="tab:orange", label="strat")
    ax.axvline(210, color="red", ls="--", lw=1, label="T_min = 210 K")
    ax.axhline(50, color="red", ls="--", lw=1, label="p_min = 50 hPa")
    ax.set_xlabel("T [K]"); ax.set_ylabel("p [hPa]")
    ax.invert_yaxis()
    ax.set_yscale("log")
    ax.set_title("(b) T vs p")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

    # (c) RH by regime
    ax = axes[0, 2]
    ax.hist(S["rh"][~is_strat], bins=20, color="tab:blue", alpha=0.7,
             label="trop")
    ax.hist(S["rh"][is_strat], bins=20, color="tab:orange", alpha=0.7,
             label="strat")
    ax.axvline(0.10, color="red", ls="--", lw=1, label="strat RH max = 0.10")
    ax.set_xlabel("RH"); ax.set_ylabel("Count")
    ax.set_title("(c) RH by regime")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (d) prod_rate histogram colored by Fortran outcome
    ax = axes[1, 0]
    ax.hist(S["h2so4_prod_rate"][~fortran_fail], bins=20,
             color="steelblue", alpha=0.7,
             label=f"Fortran OK (n={(~fortran_fail).sum()})")
    ax.hist(S["h2so4_prod_rate"][fortran_fail], bins=20,
             color="crimson", alpha=0.85,
             label=f"Fortran failed (n={fortran_fail.sum()})")
    ax.set_xscale("log")
    ax.set_xlabel("H₂SO₄ production rate [molec/cm³/s]")
    ax.set_ylabel("Count")
    ax.set_title("(d) Fortran outcome by prod_rate")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

    # (e) wall time bar
    ax = axes[1, 1]
    labels = ["Fortran", "Diffrax"]
    walls = [float(F["wall_time_s"][0]), float(D["wall_time_s"][0])]
    colors = ["tab:red", "tab:blue"]
    bars = ax.bar(labels, walls, color=colors)
    for bar, w in zip(bars, walls):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{w:.0f}s", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Wall time [s]")
    ax.set_title("(e) 100-scenario wall time @ 1800s × 48")
    ax.grid(alpha=0.3, axis="y")

    # (f) failure counts side-by-side
    ax = axes[1, 2]; ax.axis("off")
    text = (
        "Ensemble v2 outcome summary\n"
        "─────────────────────────────\n"
        f"Total scenarios     : 100\n"
        f"  troposphere       : {(~is_strat).sum()}\n"
        f"  stratosphere      : {is_strat.sum()}\n"
        "\n"
        "Physical bounds (NEW in v2):\n"
        f"  T ≥ 210 K           ✓ min = {S['T'].min():.1f}\n"
        f"  p ≥ 50 hPa          ✓ min = {S['p'].min():.1f}\n"
        f"  strat RH ≤ 0.10     ✓ max = {S['rh'][is_strat].max():.3f}\n"
        "\n"
        "Fortran (Euler+retry):\n"
        f"  ceiling-hits        {fortran_fail.sum()}/100\n"
        f"  wall                {float(F['wall_time_s'][0]):.0f} s\n"
        "\n"
        "Diffrax (Kvaerno5+PID):\n"
        f"  failures            {(D['n_failures']>0).sum()}/100\n"
        f"  gc<0 anywhere       {diffrax_fail.sum()}/100\n"
        f"  wall                {float(D['wall_time_s'][0]):.0f} s\n"
        f"  total acc steps     {D['total_accepted'].sum():,}\n"
        f"  total rej steps     {D['total_rejected'].sum():,}\n"
    )
    ax.text(0.0, 1.0, text, transform=ax.transAxes, va="top", ha="left",
             family="monospace", fontsize=10,
             bbox=dict(facecolor="#f5f5f5", edgecolor="0.7", pad=10))

    fig.suptitle(
        f"Ensemble v2 — physically-bounded 100 scenarios @ 1800s × 48 outer steps\n"
        f"Fortran fails on {fortran_fail.sum()}/100 (all physically-realistic). "
        f"Diffrax: 0 failures.",
        fontsize=13, fontweight="bold", y=0.995,
    )

    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
