"""Comparison plots for the realistic-ensemble benchmark.

Produces (in benchmark_final/plots/):
  02_total_number_comparison.png    — N_total(JAX) vs N_total(Fortran)
  03_total_mass_comparison.png      — M_total(JAX) vs M_total(Fortran)
  04_binbybin_number.png            — per-bin rel-err, number
  05_binbybin_mass.png              — per-bin rel-err, mass
  06_timing_summary.txt             — wall time comparison
"""
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "scripts"))
from jax_ensemble import _minimal_config


def bin_diameters_nm(nbin=38, rmin_cm=2e-8, rmrat=2.0, rho=1.923):
    vmin = (4.0 / 3.0) * math.pi * rmin_cm**3 * rho
    rmass = vmin * rmrat ** np.arange(nbin)
    r = (3.0 * rmass / (4.0 * math.pi * rho)) ** (1.0 / 3.0)
    return 2.0 * r * 1e7   # cm → nm


def compute_per_volume(pc_mmr, scens, rmass):
    """pc_final (mmr g/g per bin) → (#/cm³ per bin, µg/m³ per bin) for each scenario."""
    from carma.constants import R_AIR
    rho_air = scens["p"] * 100.0 * 10.0 / (float(R_AIR) * scens["T"])   # g/cm³
    num  = pc_mmr * rho_air[:, None] / rmass[None, :]               # #/cm³
    mass = pc_mmr * rho_air[:, None] * 1e12                          # µg/m³
    return num, mass


def panel_scatter(ax, F, J, label, unit, log=True):
    """One y=x scatter panel comparing totals."""
    mask = (F > 0) & (J > 0)
    ax.scatter(F[mask], J[mask], s=18, alpha=0.7, color="steelblue",
                edgecolor="black", lw=0.4)
    if log:
        ax.set_xscale("log"); ax.set_yscale("log")
    lo = min(F[mask].min(), J[mask].min())
    hi = max(F[mask].max(), J[mask].max())
    ref = np.logspace(np.log10(lo), np.log10(hi), 5) if log else np.linspace(lo, hi, 5)
    ax.plot(ref, ref, "r--", lw=1, label="y = x")
    ax.plot(ref, ref*1.1, "0.6", ls=":", lw=0.8, label="±10%")
    ax.plot(ref, ref*0.9, "0.6", ls=":", lw=0.8)
    ax.set_xlabel(f"Fortran  {label}  [{unit}]", fontsize=10)
    ax.set_ylabel(f"JAX  {label}  [{unit}]", fontsize=10)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9)
    # Stats
    denom = np.maximum(np.abs(F[mask]), np.abs(J[mask]))
    rel = np.abs(F[mask] - J[mask]) / denom
    ax.text(0.03, 0.97,
            f"n = {mask.sum()}/{len(F)}\n"
            f"Median rel err: {np.median(rel):.2e}\n"
            f"P95: {np.percentile(rel, 95):.2e}\n"
            f"Max: {rel.max():.2e}",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=8.5, family="monospace",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.8"))


def panel_binbybin(ax, F_arr, J_arr, active, d_nm, threshold_label,
                   ylabel, title):
    rng = np.random.default_rng(0)
    nbin_ = F_arr.shape[1]
    data_for_box = []
    for b in range(nbin_):
        denom = np.maximum(np.abs(F_arr[:, b]), np.abs(J_arr[:, b]))
        denom = np.where(denom > 1e-300, denom, 1.0)
        err = np.abs(F_arr[:, b] - J_arr[:, b]) / denom
        jitter = rng.normal(0, 0.07, size=err.shape)
        xs = d_nm[b] * np.exp(jitter)
        if (~active[:, b]).any():
            ax.scatter(xs[~active[:, b]], np.clip(err[~active[:, b]], 1e-16, 1),
                       s=2, alpha=0.15, color="0.75", edgecolors="none", zorder=1)
        if active[:, b].any():
            ax.scatter(xs[active[:, b]], np.clip(err[active[:, b]], 1e-16, 1),
                       s=3, alpha=0.4, color="steelblue", edgecolors="none", zorder=2)
        vals = err[active[:, b]]
        data_for_box.append(np.clip(vals, 1e-16, 1) if vals.size else np.array([1e-16]))

    ax.boxplot(data_for_box, positions=d_nm, widths=d_nm * 0.13,
                showfliers=False, patch_artist=True,
                medianprops=dict(color="crimson", lw=1.5),
                boxprops=dict(facecolor="white", alpha=0.85, edgecolor="black"),
                whiskerprops=dict(color="black", lw=0.8),
                capprops=dict(color="black", lw=0.8), zorder=3)

    ax.axhline(0.01, color="green",  ls="--", lw=1.2, alpha=0.8)
    ax.axhline(0.05, color="orange", ls="--", lw=1.2, alpha=0.8)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="green",  ls="--", lw=1.2, label="1%"),
        Line2D([0], [0], color="orange", ls="--", lw=1.2, label="5%"),
        ax.scatter([], [], s=8, color="steelblue", alpha=0.6,
                   label=f"active (≥ {threshold_label})"),
        ax.scatter([], [], s=8, color="0.75", alpha=0.6,
                   label=f"empty  (< {threshold_label})"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="lower right")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(5e-17, 2.0)
    ax.xaxis.set_minor_locator(LogLocator(numticks=15))
    ax.yaxis.set_minor_locator(LogLocator(numticks=15))
    ax.grid(True, alpha=0.2, which="both")
    ax.set_xlabel("Bin median diameter [nm]", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10)

    all_err = []
    for b in range(nbin_):
        denom = np.maximum(np.abs(F_arr[:, b]), np.abs(J_arr[:, b]))
        denom = np.where(denom > 1e-300, denom, 1.0)
        err = np.abs(F_arr[:, b] - J_arr[:, b]) / denom
        all_err.extend(err[active[:, b]].tolist())
    all_err = np.array(all_err)
    if all_err.size:
        pct_pass = (all_err < 0.01).mean() * 100
        ax.text(0.02, 0.97,
                f"Active: {active.sum():,}/{F_arr.size}\n"
                f"Median err: {np.median(all_err):.1e}\n"
                f"< 1%: {pct_pass:.0f}%",
                transform=ax.transAxes, va="top", ha="left",
                fontsize=8.5, family="monospace",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.8"))


def main():
    scens = np.load(ROOT / "scenarios" / "realistic_scenarios_100.npz")
    F     = np.load(ROOT / "outputs" / "fortran_outputs.npz")
    J     = np.load(ROOT / "outputs" / "jax_outputs.npz")

    cfg = _minimal_config()
    rmass = np.asarray(cfg.groups[0].rmass)
    d_nm  = bin_diameters_nm(cfg.nbin)

    pc_F = F["pc_final"]; pc_J = J["pc_final"]

    num_F, mass_F = compute_per_volume(pc_F, scens, rmass)
    num_J, mass_J = compute_per_volume(pc_J, scens, rmass)

    NUM_FLOOR  = 1e-3   # #/cm³
    MASS_FLOOR = 1e-6   # µg/m³
    active_num  = num_F  > NUM_FLOOR
    active_mass = mass_F > MASS_FLOOR

    out_dir = ROOT / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 02 - total number scatter
    fig, ax = plt.subplots(figsize=(8, 7))
    panel_scatter(ax, num_F.sum(axis=1), num_J.sum(axis=1),
                  label="N_total", unit="#/cm³")
    ax.set_title("Total number concentration: JAX vs Fortran\n"
                 "100 scenarios, 60 s × 1440 steps (24 h)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "02_total_number_comparison.png", dpi=130,
                bbox_inches="tight")
    plt.close()

    # 03 - total mass scatter
    fig, ax = plt.subplots(figsize=(8, 7))
    panel_scatter(ax, mass_F.sum(axis=1), mass_J.sum(axis=1),
                  label="M_total", unit="µg/m³")
    ax.set_title("Total mass concentration: JAX vs Fortran\n"
                 "100 scenarios, 60 s × 1440 steps (24 h)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "03_total_mass_comparison.png", dpi=130,
                bbox_inches="tight")
    plt.close()

    # 04 - bin-by-bin number
    fig, ax = plt.subplots(figsize=(11, 6))
    panel_binbybin(ax, num_F, num_J, active_num, d_nm,
                   threshold_label=f"{NUM_FLOOR:.0e} #/cm³",
                   ylabel="|JAX − Fortran| / max(|JAX|, |Fortran|)",
                   title=f"Bin-by-bin number relative error  "
                         f"(active = N > {NUM_FLOOR:.0e} #/cm³)")
    plt.tight_layout()
    plt.savefig(out_dir / "04_binbybin_number.png", dpi=130, bbox_inches="tight")
    plt.close()

    # 05 - bin-by-bin mass
    fig, ax = plt.subplots(figsize=(11, 6))
    panel_binbybin(ax, mass_F, mass_J, active_mass, d_nm,
                   threshold_label=f"{MASS_FLOOR:.0e} µg/m³",
                   ylabel="|JAX − Fortran| / max(|JAX|, |Fortran|)",
                   title=f"Bin-by-bin mass relative error  "
                         f"(active = M > {MASS_FLOOR:.0e} µg/m³)")
    plt.tight_layout()
    plt.savefig(out_dir / "05_binbybin_mass.png", dpi=130, bbox_inches="tight")
    plt.close()

    # 06 - timing summary
    wall_F = float(F.get("wall_time_s", np.array([np.nan]))[0])
    wall_J = float(J.get("wall_time_s", np.array([np.nan]))[0])
    n = len(pc_F)
    lines = [
        "=== CARMA-JAX realistic-ensemble timing ===",
        f"  Setup: 60 s × 1440 steps = 24 h simulation, 100 scenarios",
        "",
        f"  Fortran (8 parallel processes):    {wall_F:.0f} s  ({wall_F/n*1000:.0f} ms/scen wall)",
        f"  JAX (8 parallel processes, fp64):  {wall_J:.0f} s  ({wall_J/n*1000:.0f} ms/scen wall)",
        f"  Ratio (JAX wall / Fortran wall):   {wall_J/wall_F:.1f}×",
        "",
        f"  Both runs used 8 worker processes on the same CPU.",
        f"  Per-scenario effective CPU time:",
        f"    Fortran: ~{wall_F*8/n:.0f} s/scen   JAX: ~{wall_J*8/n:.0f} s/scen",
    ]
    print("\n".join(lines))
    (out_dir / "06_timing_summary.txt").write_text("\n".join(lines) + "\n")
    print(f"\nAll plots in: {out_dir}")


if __name__ == "__main__":
    main()
