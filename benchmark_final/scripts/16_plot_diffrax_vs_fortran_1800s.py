"""Compare diffrax-JAX vs Fortran (adaptive) at 1800s outer timestep.

Three-panel plot:
  panel 0: Number per bin — Fortran (ceiling-hit flagged) vs diffrax
  panel 1: Mass per bin   — same
  panel 2: Summary stats — failure counts, gc<0 counts, runtime

Reads:
  benchmark_final/scenarios/realistic_scenarios_100.npz
  benchmark_final/outputs/dt1800/adaptive/fortran_outputs.npz   (from bench/realistic-ensemble)
  benchmark_final/outputs/dt1800/diffrax/jax_diffrax_outputs.npz

Output: benchmark_final/plots/16_diffrax_vs_fortran_1800s.png
"""
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "scripts"))
from jax_ensemble import _minimal_config


def bin_diameters_nm(nbin=38, rmin_cm=2e-8, rmrat=2.0, rho=1.923):
    vmin = (4.0 / 3.0) * math.pi * rmin_cm ** 3 * rho
    rmass = vmin * rmrat ** np.arange(nbin)
    r = (3.0 * rmass / (4.0 * math.pi * rho)) ** (1.0 / 3.0)
    return 2.0 * r * 1e7


def per_volume(pc_mmr, scens, rmass):
    from carma.constants import R_AIR
    rho_air = scens["p"] * 100.0 * 10.0 / (float(R_AIR) * scens["T"])
    return (pc_mmr * rho_air[:, None] / rmass[None, :],
             pc_mmr * rho_air[:, None] * 1e12)


def panel(ax, F_arr, J_arr, active, ceiling_scens, d_nm,
           threshold_label, title, ylim_low=1e-13):
    rng = np.random.default_rng(0)
    nbin_ = F_arr.shape[1]
    data_for_box = []
    for b in range(nbin_):
        denom = np.maximum(np.abs(F_arr[:, b]), np.abs(J_arr[:, b]))
        denom = np.where(denom > 1e-300, denom, 1.0)
        err = np.abs(F_arr[:, b] - J_arr[:, b]) / denom
        xs = d_nm[b] * np.exp(rng.normal(0, 0.07, size=err.shape))

        inactive = ~active[:, b]
        if inactive.any():
            ax.scatter(xs[inactive], np.clip(err[inactive], ylim_low, 1),
                        s=4, alpha=0.2, color="0.75", edgecolors="none", zorder=1)
        active_conv = active[:, b] & ~ceiling_scens
        active_ceil = active[:, b] & ceiling_scens
        if active_conv.any():
            ax.scatter(xs[active_conv], np.clip(err[active_conv], ylim_low, 1),
                        s=6, alpha=0.55, color="steelblue",
                        edgecolors="none", zorder=2)
        if active_ceil.any():
            ax.scatter(xs[active_ceil], np.clip(err[active_ceil], ylim_low, 1),
                        s=12, alpha=0.85, color="crimson",
                        edgecolors="black", lw=0.4, marker="x", zorder=4)
        vals = err[active_conv]
        data_for_box.append(np.clip(vals, ylim_low, 1) if vals.size
                              else np.array([ylim_low]))

    ax.boxplot(data_for_box, positions=d_nm, widths=d_nm * 0.13,
                showfliers=False, patch_artist=True,
                medianprops=dict(color="darkblue", lw=1.5),
                boxprops=dict(facecolor="white", alpha=0.85, edgecolor="black"),
                whiskerprops=dict(color="black", lw=0.8),
                capprops=dict(color="black", lw=0.8), zorder=3)

    ax.axhline(0.05, color="orange", ls="--", lw=1.2, alpha=0.8)
    ax.axhline(0.5, color="red", ls="--", lw=1.2, alpha=0.8)

    handles = [
        Line2D([0], [0], color="orange", ls="--", lw=1.2, label="5% rel err"),
        Line2D([0], [0], color="red", ls="--", lw=1.2, label="50% rel err"),
        ax.scatter([], [], s=10, color="steelblue", alpha=0.7,
                    label=f"Fortran converged (≥ {threshold_label})"),
        ax.scatter([], [], s=16, color="crimson", marker="x",
                    label="Fortran ceiling-hit"),
        ax.scatter([], [], s=10, color="0.75", alpha=0.5,
                    label=f"empty (< {threshold_label})"),
    ]
    ax.legend(handles=handles, fontsize=8.5, loc="lower right")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(ylim_low, 2.0)
    ax.xaxis.set_minor_locator(LogLocator(numticks=15))
    ax.yaxis.set_minor_locator(LogLocator(numticks=15))
    ax.grid(True, alpha=0.2, which="both")
    ax.set_xlabel("Bin median diameter [nm]", fontsize=11)
    ax.set_ylabel("|diffrax − Fortran| / max", fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")


def stats_panel(ax, F, D, scens):
    ax.axis("off")
    ceiling_F = int((F["gc_h2so4_final"] < 0).sum())
    fails_D = int((D["n_failures"] > 0).sum())
    neg_D = int((D["min_gc_h2so4"] < 0).sum())
    wall_F = float(F["wall_time_s"][0]) if "wall_time_s" in F else float("nan")
    wall_D = float(D["wall_time_s"][0])
    n = int(D["n_scenarios"][0])

    text = f"""
Outer timestep: 1800 s × 48 steps (24 h sim)
Number of scenarios: {n}

{'Fortran (semi-implicit Euler + retry)':<40}
  ceiling-hits (gc_h2so4_final < 0):  {ceiling_F}/{n}
  wall time:                           {wall_F:.0f} s

{'diffrax (Kvaerno5 + PIDController)':<40}
  diffrax max_steps failures:         {fails_D}/{n}
  scenarios with gc<0 anywhere:        {neg_D}/{n}
  wall time:                           {wall_D:.0f} s

Total internal solver steps (across all scens):
  Kvaerno5 accepted:  {int(D["total_accepted"].sum()):,}
  Kvaerno5 rejected:  {int(D["total_rejected"].sum()):,}
  acceptance ratio:   {float(D["total_accepted"].sum() / max(D["total_accepted"].sum() + D["total_rejected"].sum(), 1)):.1%}
""".strip()
    ax.text(0.02, 0.98, text, transform=ax.transAxes, va="top", ha="left",
            family="monospace", fontsize=10,
            bbox=dict(facecolor="#f5f5f5", edgecolor="0.7"))


def main():
    cfg = _minimal_config()
    rmass = np.asarray(cfg.groups[0].rmass)
    d_nm = bin_diameters_nm(cfg.nbin)

    scens = np.load(ROOT / "scenarios" / "realistic_scenarios_100.npz")
    F = np.load(ROOT / "outputs" / "dt1800" / "adaptive" / "fortran_outputs.npz")
    D = np.load(ROOT / "outputs" / "dt1800" / "diffrax"
                  / "jax_diffrax_outputs.npz")

    ceiling_scens = F["gc_h2so4_final"] < 0
    num_F, mass_F = per_volume(F["pc_final"], scens, rmass)
    num_D, mass_D = per_volume(D["pc_final"], scens, rmass)

    NUM_FLOOR, MASS_FLOOR = 1e-3, 1e-6
    active_num = num_F > NUM_FLOOR
    active_mass = mass_F > MASS_FLOOR

    fig = plt.figure(figsize=(18, 13))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.4])
    ax_num = fig.add_subplot(gs[0, 0])
    ax_mass = fig.add_subplot(gs[0, 1])
    ax_stats = fig.add_subplot(gs[1, :])

    panel(ax_num, num_F, num_D, active_num, ceiling_scens, d_nm,
            f"{NUM_FLOOR:.0e} #/cm³",
            "Number per bin — relative error (diffrax vs Fortran, 1800 s × 48)")
    panel(ax_mass, mass_F, mass_D, active_mass, ceiling_scens, d_nm,
            f"{MASS_FLOOR:.0e} µg/m³",
            "Mass per bin — relative error (diffrax vs Fortran, 1800 s × 48)")
    stats_panel(ax_stats, F, D, scens)

    n_ceil = int(ceiling_scens.sum())
    fig.suptitle(
        f"diffrax (Kvaerno5+PIDController) vs Fortran (semi-implicit Euler) "
        f"at 1800 s outer step\n"
        f"Fortran hit retry ceiling on {n_ceil}/100 scenarios; "
        f"red × marks those scenarios where Fortran result is unreliable",
        fontsize=12, fontweight="bold", y=0.995,
    )

    out = ROOT / "plots" / "16_diffrax_vs_fortran_1800s.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
