"""Per-bin parity distribution: dots = each scenario's rel err for
that bin, box plot per bin shows the cross-scenario spread.

Two stacked panels — mass concentration and number concentration. The
underlying rel-err arrays are mathematically identical (the per-bin
rmass[b] factor cancels in numerator and denominator), so the panels
look the same; both are kept for clarity / to confirm the equivalence
visually.
"""
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
J = np.load(ROOT / "data/sulfate_jax_realistic_outputs.npz")
F = np.load(ROOT / "data/sulfate_fortran_realistic_outputs.npz")

# Bin grid (matches jax_ensemble.py constants)
nbin, rmin, rmrat, rho = 38, 2.0e-8, 2.0, 1.923
vmin = (4.0 / 3.0) * math.pi * rmin**3 * rho
rmass = vmin * rmrat ** np.arange(nbin)
r_cm = (3.0 * rmass / (4.0 * math.pi * rho)) ** (1.0 / 3.0)
d_nm = 2.0 * r_cm * 1e7        # bin median diameter [nm]

pc_J = J["pc_final"]            # (n_scen, NBIN) mmr
pc_F = F["pc_final"]
n_scen = pc_J.shape[0]

# Per-bin rel err. Identical for mass-per-bin and number-per-bin since
# the per-bin rmass[b] constant cancels in the |J - F| / max(|J|, |F|)
# definition. Computed once and shown twice.
denom = np.maximum(np.abs(pc_J), np.abs(pc_F))
denom = np.where(denom > 1e-300, denom, 1.0)
rel_per_bin = np.abs(pc_J - pc_F) / denom    # (n_scen, NBIN)

# Floor zeros for log axis
floor = 1e-16
rel_for_plot = np.where(rel_per_bin > floor, rel_per_bin, floor)

# Box-plot widths on log-x: a multiplicative factor of the position so
# every box has the same visual width on the log scale.
log_width_factor = 0.18
widths = d_nm * log_width_factor

# Horizontal jitter (multiplicative on log axis) so the n_scen dots per
# bin don't sit on top of each other. Std is ~7% of the position which
# is well inside the box width.
rng = np.random.default_rng(0)
jitter = rng.normal(loc=0.0, scale=0.07, size=(n_scen, nbin))
jittered_x = d_nm[None, :] * np.exp(jitter)

fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
for ax, label in zip(axes, ["Mass concentration", "Number concentration"]):
    # Scatter every scenario's rel err per bin (n_scen × NBIN dots, jittered)
    ax.scatter(jittered_x.ravel(), rel_for_plot.ravel(),
               s=3, alpha=0.18, color="steelblue", edgecolors="none",
               zorder=1)

    # Box plot per bin (P25/median/P75 boxes; whiskers 1.5×IQR)
    ax.boxplot(
        [rel_for_plot[:, b] for b in range(nbin)],
        positions=d_nm, widths=widths,
        showfliers=False, patch_artist=True,
        medianprops=dict(color="crimson", lw=1.8),
        boxprops=dict(facecolor="white", alpha=0.6, edgecolor="black", lw=1.0),
        whiskerprops=dict(color="black", lw=0.9),
        capprops=dict(color="black", lw=0.9),
        zorder=3,
    )

    ax.axhline(0.05, color="red", ls="--", lw=1, alpha=0.7, label="5% gate")
    ax.axhline(0.01, color="blue", ls="--", lw=1, alpha=0.7, label="1% gate")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(d_nm[0] * 0.6, d_nm[-1] * 1.6)
    ax.set_ylim(1e-7, 2.0)
    ax.set_ylabel(f"{label}\nper-bin rel err  (J vs F)")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_title(
        f"{label}: dot = one of {n_scen} scenarios, box = cross-scenario "
        f"P25/median/P75; whiskers 1.5·IQR"
    )
    ax.legend(loc="lower right", fontsize=9)

# X-axis tick labels in nm. Use FuncFormatter to render cleanly: ints
# without decimal, sub-nm with one decimal. ScalarFormatter rounded
# 0.5 → "0" on the previous render which read as a meaningless "zero"
# tick on a log axis.
from matplotlib.ticker import FuncFormatter, NullLocator
axes[-1].set_xlabel("Bin median diameter (nm)")
xtick_d = [d for d in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000)
           if d_nm[0] * 0.6 <= d <= d_nm[-1] * 1.6]
axes[-1].set_xticks(xtick_d)
axes[-1].xaxis.set_major_formatter(
    FuncFormatter(lambda x, pos: f"{int(x)}" if x >= 1 else f"{x:g}"))
axes[-1].xaxis.set_minor_locator(NullLocator())

plt.tight_layout()
out_dir = ROOT / "plots" / "diff" / "phase10"
out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / "sulfate_per_bin_relerr.png"
plt.savefig(out, dpi=110, bbox_inches="tight")
print(f"saved {out}")

# Also report the cross-scenario percentiles per bin to the console so
# the box-plot can be cross-checked numerically.
print(f"\nPer-bin cross-scenario percentiles ({n_scen} scenarios):")
print(f"  bin   d (nm)    p25       p50       p75       p95       max")
pct_25 = np.percentile(rel_per_bin, 25, axis=0)
pct_50 = np.percentile(rel_per_bin, 50, axis=0)
pct_75 = np.percentile(rel_per_bin, 75, axis=0)
pct_95 = np.percentile(rel_per_bin, 95, axis=0)
mx     = rel_per_bin.max(axis=0)
for b in range(nbin):
    print(f"  {b:>3}  {d_nm[b]:>8.3f}  {pct_25[b]:.2e}  {pct_50[b]:.2e}  "
          f"{pct_75[b]:.2e}  {pct_95[b]:.2e}  {mx[b]:.2e}")
