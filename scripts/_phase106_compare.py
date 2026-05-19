"""Phase 10.6 comparison: prescribed-substep JAX vs Fortran, with
side-by-side against the adaptive-substep baseline (Phase 10.5).

Tells us how much of the Phase 10.5 residual gap is FP-boundary
retry-decision drift versus something else.
"""
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent

J_adapt = np.load(ROOT / "data/sulfate_jax_realistic_outputs.npz")        # Phase 10.5
J_pres  = np.load(ROOT / "data/sulfate_jax_prescribed_outputs.npz")       # Phase 10.6
F       = np.load(ROOT / "data/sulfate_fortran_realistic_outputs.npz")
sched   = np.load(ROOT / "data/sulfate_fortran_substep_schedules.npz")

nbin, rmin, rmrat, rho = 38, 2.0e-8, 2.0, 1.923
vmin = (4.0 / 3.0) * math.pi * rmin**3 * rho
rmass = vmin * rmrat ** np.arange(nbin)
d_nm = 2.0 * (3.0 * rmass / (4.0 * math.pi * rho)) ** (1.0/3.0) * 1e7


def rel_err(j, f):
    j = np.asarray(j); f = np.asarray(f)
    denom = np.maximum(np.abs(j), np.abs(f))
    denom = np.where(denom > 1e-300, denom, 1.0)
    return np.abs(j - f) / denom


def stats(label, J, F):
    n = J["pc_final"].shape[0]
    pc_J = J["pc_final"]; pc_F = F["pc_final"]
    rel  = rel_err(pc_J, pc_F)
    sum_rel = rel_err(pc_J.sum(axis=1), pc_F.sum(axis=1))
    med_per_scen = np.median(rel, axis=1)
    max_per_scen = rel.max(axis=1)
    peak_match = (pc_J.argmax(axis=1) == pc_F.argmax(axis=1))
    n_strict = ((med_per_scen <= 0.01) & (max_per_scen <= 0.05)).sum()
    n_med1  = (med_per_scen <= 0.01).sum()
    n_max5  = (max_per_scen <= 0.05).sum()

    print(f"\n=== {label} ===")
    print(f"  total mass   p50={np.percentile(sum_rel,50):.2e}  "
          f"p95={np.percentile(sum_rel,95):.2e}  max={sum_rel.max():.2e}")
    print(f"  per-bin med  p50={np.percentile(med_per_scen,50):.2e}  "
          f"p95={np.percentile(med_per_scen,95):.2e}  max={med_per_scen.max():.2e}")
    print(f"  per-bin max  p50={np.percentile(max_per_scen,50):.2e}  "
          f"p95={np.percentile(max_per_scen,95):.2e}  max={max_per_scen.max():.2e}")
    print(f"  peak match: {peak_match.sum()}/{n} = {100*peak_match.mean():.1f}%")
    print(f"  strict gate (med ≤ 1%, max ≤ 5%): {n_strict}/{n} = {100*n_strict/n:.1f}%")
    print(f"  median ≤ 1%:  {n_med1}/{n}   max ≤ 5%:  {n_max5}/{n}")
    return rel, med_per_scen, max_per_scen


rel_a, med_a, max_a = stats("Phase 10.5 (adaptive retry)", J_adapt, F)
rel_p, med_p, max_p = stats("Phase 10.6 (prescribed = Fortran's ntsubsteps)", J_pres, F)

# How many scenarios moved from fail→pass with prescribed schedule?
pass_a = (med_a <= 0.01) & (max_a <= 0.05)
pass_p = (med_p <= 0.01) & (max_p <= 0.05)
moved_to_pass = (~pass_a) & pass_p
moved_to_fail = pass_a & (~pass_p)
print(f"\nPhase 10.5 → 10.6 transitions:")
print(f"  fail → pass: {moved_to_pass.sum()}")
print(f"  pass → fail: {moved_to_fail.sum()}")
print(f"  net change in pass count: {pass_p.sum() - pass_a.sum():+d}")

# Per-bin distribution comparison
fig, ax = plt.subplots(1, 1, figsize=(14, 6))
adapt_box = [rel_a[:, b] for b in range(nbin)]
pres_box  = [rel_p[:, b] for b in range(nbin)]
floor = 1e-16
adapt_for_plot = [np.where(x > floor, x, floor) for x in adapt_box]
pres_for_plot  = [np.where(x > floor, x, floor) for x in pres_box]

w = 0.18
positions_a = d_nm * np.exp(-0.07)
positions_p = d_nm * np.exp(+0.07)
bp_a = ax.boxplot(adapt_for_plot, positions=positions_a, widths=positions_a*w,
                  showfliers=False, patch_artist=True,
                  medianprops=dict(color="darkblue", lw=1.5),
                  boxprops=dict(facecolor="lightblue", alpha=0.6, edgecolor="black"),
                  whiskerprops=dict(color="black", lw=0.8),
                  capprops=dict(color="black", lw=0.8))
bp_p = ax.boxplot(pres_for_plot, positions=positions_p, widths=positions_p*w,
                  showfliers=False, patch_artist=True,
                  medianprops=dict(color="darkred", lw=1.5),
                  boxprops=dict(facecolor="lightyellow", alpha=0.6, edgecolor="black"),
                  whiskerprops=dict(color="black", lw=0.8),
                  capprops=dict(color="black", lw=0.8))
ax.axhline(0.05, color="red", ls="--", lw=1, alpha=0.7)
ax.axhline(0.01, color="blue", ls="--", lw=1, alpha=0.7)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(d_nm[0]*0.6, d_nm[-1]*1.6)
ax.set_ylim(1e-9, 2.0)
ax.set_xlabel("Bin median diameter (nm)")
ax.set_ylabel("per-bin rel err")
ax.set_title("Per-bin parity: adaptive (blue) vs prescribed-substep (yellow). "
             "If prescribed boxes are much lower, residual is retry-boundary FP drift.")
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
xtick_d = [d for d in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000)
           if d_nm[0]*0.6 <= d <= d_nm[-1]*1.6]
ax.set_xticks(xtick_d)
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x)}" if x>=1 else f"{x:g}"))
ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=(2,3,4,5,6,7,8,9), numticks=99))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.grid(True, alpha=0.4, which="major", lw=0.7)
ax.grid(True, alpha=0.2, which="minor", lw=0.4)
ax.legend([bp_a["boxes"][0], bp_p["boxes"][0]],
          ["adaptive retry (Phase 10.5)", "prescribed = Fortran (Phase 10.6)"],
          loc="lower right")
plt.tight_layout()
out_dir = ROOT / "plots" / "diff" / "phase10"
out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(out_dir / "phase106_perbin_compare.png", dpi=110, bbox_inches="tight")
print(f"\nsaved {out_dir / 'phase106_perbin_compare.png'}")
