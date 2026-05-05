"""Phase 10.4 parity statistics: JAX vs Fortran on 1000 scenarios.

Reports parity in three quantity views (per-bin, total mass, total
number) since they answer different questions:
- per-bin rel err is invariant to mass-vs-number conversion (same
  scenario denominators), so it's plotted once.
- total mass = Σ_b pc_mmr[b]  (weights all bins ~ equally)
- total number ∝ Σ_b pc_mmr[b] / rmass[b]  (weights small bins
  heavily — small particles dominate count, big ones dominate mass).

Mass and number parity often look quite different in aerosol modelling.
"""
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
J = np.load(ROOT / "data/sulfate_jax_realistic_outputs.npz")
F = np.load(ROOT / "data/sulfate_fortran_realistic_outputs.npz")
scens = np.load(ROOT / "data/sulfate_scenarios_realistic_1000.npz")
n = J["pc_final"].shape[0]

# Bin grid (matches jax_ensemble.py constants)
nbin, rmin, rmrat, rho = 38, 2.0e-8, 2.0, 1.923
vmin = (4.0 / 3.0) * math.pi * rmin**3 * rho
rmass = vmin * rmrat ** np.arange(nbin)


def rel_err(j, f):
    j = np.asarray(j); f = np.asarray(f)
    denom = np.maximum(np.abs(j), np.abs(f))
    denom = np.where(denom > 1e-300, denom, 1.0)
    return np.abs(j - f) / denom


# Per-scenario summary statistics
T_rel = rel_err(J["T_final"], F["T_final"])
gc_rel = rel_err(J["gc_h2so4_final"], F["gc_h2so4_final"])

# Mass / number per-scenario totals (number ∝ pc_mmr/rmass)
pc_J = J["pc_final"]; pc_F = F["pc_final"]
mass_total_J = pc_J.sum(axis=1)
mass_total_F = pc_F.sum(axis=1)
num_per_bin_J = pc_J / rmass[None, :]
num_per_bin_F = pc_F / rmass[None, :]
num_total_J = num_per_bin_J.sum(axis=1)
num_total_F = num_per_bin_F.sum(axis=1)

mass_sum_rel = rel_err(mass_total_J, mass_total_F)
num_sum_rel = rel_err(num_total_J, num_total_F)

# Per-bin rel err is identical for mass and number (same denominator
# division by the per-bin rmass constant cancels in numerator/denom).
pc_per_bin_rel = rel_err(pc_J, pc_F)
pc_med_per_scen = np.median(pc_per_bin_rel, axis=1)
pc_max_per_scen = np.max(pc_per_bin_rel, axis=1)
pc_peakidx_match = (pc_J.argmax(axis=1) == pc_F.argmax(axis=1))
num_peakidx_match = (num_per_bin_J.argmax(axis=1) == num_per_bin_F.argmax(axis=1))

print(f"=== Parity stats over {n} scenarios ===\n")
for name, x in [
    ("T_final",            T_rel),
    ("gc_h2so4_final",     gc_rel),
    ("total mass mmr",     mass_sum_rel),
    ("total number ∝1/r",  num_sum_rel),
    ("per-bin median",     pc_med_per_scen),
    ("per-bin max",        pc_max_per_scen),
]:
    pcts = np.percentile(x, [50, 90, 95, 99, 100])
    print(f"{name:>22}:  p50={pcts[0]:.2e}  p90={pcts[1]:.2e}  p95={pcts[2]:.2e}  p99={pcts[3]:.2e}  max={pcts[4]:.2e}")

print(f"\nmass peak-bin match: {pc_peakidx_match.sum()}/{n} = {100*pc_peakidx_match.mean():.1f}%")
print(f"num  peak-bin match: {num_peakidx_match.sum()}/{n} = {100*num_peakidx_match.mean():.1f}%")
print(f"|peak_J - peak_F| > 1 (mass): {(np.abs(pc_J.argmax(axis=1) - pc_F.argmax(axis=1)) > 1).sum()}/{n}")
print(f"|peak_J - peak_F| > 1 (num):  {(np.abs(num_per_bin_J.argmax(axis=1) - num_per_bin_F.argmax(axis=1)) > 1).sum()}/{n}")

# Phase 10 acceptance criteria
GATE_PASS_PCT = 95
GATE_MEDIAN = 0.01
GATE_MAX = 0.05
n_pass = ((pc_med_per_scen <= GATE_MEDIAN) & (pc_max_per_scen <= GATE_MAX)).sum()
print(f"\nPhase 10 gate (per-scen per-bin median ≤ {GATE_MEDIAN}, max ≤ {GATE_MAX}):")
print(f"  passing: {n_pass}/{n} = {100*n_pass/n:.1f}%  (target ≥ {GATE_PASS_PCT}%)")

# Save summary npz for downstream plotting
out_npz = ROOT / "data/sulfate_parity_summary.npz"
np.savez_compressed(out_npz,
    T_rel=T_rel, gc_rel=gc_rel,
    mass_sum_rel=mass_sum_rel, num_sum_rel=num_sum_rel,
    pc_med_per_scen=pc_med_per_scen, pc_max_per_scen=pc_max_per_scen,
    pc_peakidx_match=pc_peakidx_match, num_peakidx_match=num_peakidx_match)
print(f"\nsaved {out_npz}")

# Three-panel scatter: total mass parity, total number parity, per-bin scatter
fig, ax = plt.subplots(1, 3, figsize=(17, 5))

# Panel A: total mass y=x
ax[0].loglog(mass_total_J, mass_total_F, "k.", ms=2, alpha=0.5)
xx = np.geomspace(1e-22, 1e-9, 5)
ax[0].loglog(xx, xx, "r--", lw=1, label="y=x")
ax[0].set_xlabel("JAX total mass mmr [g/g]")
ax[0].set_ylabel("Fortran total mass mmr [g/g]")
ax[0].set_title(f"Total mass parity ({n} scenarios)")
ax[0].legend(); ax[0].grid(True, alpha=0.3, which="both")

# Panel B: total number y=x — number ∝ Σ pc/rmass, units [1 / (rhoa·zmet)]
xn = np.geomspace(num_total_J[num_total_J > 0].min() / 10,
                   num_total_J.max() * 10, 5)
ax[1].loglog(num_total_J, num_total_F, "k.", ms=2, alpha=0.5)
ax[1].loglog(xn, xn, "r--", lw=1, label="y=x")
ax[1].set_xlabel("JAX total number ∝ Σ pc/rmass")
ax[1].set_ylabel("Fortran total number ∝ Σ pc/rmass")
ax[1].set_title(f"Total number parity ({n} scenarios)")
ax[1].legend(); ax[1].grid(True, alpha=0.3, which="both")

# Panel C: per-bin scatter — each dot = one scenario, x = median rel err over
# its 38 bins, y = max rel err over those bins.
ax[2].loglog(pc_med_per_scen, pc_max_per_scen, "k.", ms=2, alpha=0.5)
ax[2].axhline(GATE_MAX, color="r", ls="--", lw=1, label=f"max gate {GATE_MAX}")
ax[2].axvline(GATE_MEDIAN, color="b", ls="--", lw=1, label=f"med gate {GATE_MEDIAN}")
ax[2].set_xlabel("Per-scenario per-bin median rel err")
ax[2].set_ylabel("Per-scenario per-bin max rel err")
ax[2].set_title(f"Per-bin parity ({n} scenarios)")
ax[2].legend(); ax[2].grid(True, alpha=0.3, which="both")

plt.tight_layout()
out_dir = ROOT / "plots" / "diff" / "phase10"
out_dir.mkdir(parents=True, exist_ok=True)
out_png = out_dir / "sulfate_parity_scatter.png"
plt.savefig(out_png, dpi=110, bbox_inches="tight")
print(f"saved {out_png}")
