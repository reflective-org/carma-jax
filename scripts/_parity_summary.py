"""Phase 10.4 parity statistics: JAX vs Fortran on 1000 scenarios."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
J = np.load(ROOT / "data/sulfate_jax_realistic_outputs.npz")
F = np.load(ROOT / "data/sulfate_fortran_realistic_outputs.npz")
scens = np.load(ROOT / "data/sulfate_scenarios_realistic_1000.npz")
n = J["pc_final"].shape[0]


def rel_err(j, f):
    j = np.asarray(j); f = np.asarray(f)
    denom = np.maximum(np.abs(j), np.abs(f))
    denom = np.where(denom > 1e-300, denom, 1.0)
    return np.abs(j - f) / denom


# Per-scenario summary statistics
T_rel = rel_err(J["T_final"], F["T_final"])
gc_rel = rel_err(J["gc_h2so4_final"], F["gc_h2so4_final"])

# pc_final per-scenario aggregate metrics
pc_J = J["pc_final"]; pc_F = F["pc_final"]
pc_sum_rel = rel_err(pc_J.sum(axis=1), pc_F.sum(axis=1))
# Bin-by-bin per scenario, then take the median (typical) and max (worst) across bins
pc_per_bin_rel = rel_err(pc_J, pc_F)
pc_med_per_scen = np.median(pc_per_bin_rel, axis=1)
pc_max_per_scen = np.max(pc_per_bin_rel, axis=1)
pc_peakidx_match = (pc_J.argmax(axis=1) == pc_F.argmax(axis=1))

print(f"=== Parity stats over {n} scenarios ===\n")
for name, x in [
    ("T_final", T_rel),
    ("gc_h2so4_final", gc_rel),
    ("pc total mass", pc_sum_rel),
    ("pc per-bin median", pc_med_per_scen),
    ("pc per-bin max", pc_max_per_scen),
]:
    pcts = np.percentile(x, [50, 90, 95, 99, 100])
    print(f"{name:>22}:  p50={pcts[0]:.2e}  p90={pcts[1]:.2e}  p95={pcts[2]:.2e}  p99={pcts[3]:.2e}  max={pcts[4]:.2e}")

print(f"\npc peak-bin match: {pc_peakidx_match.sum()}/{n} = {100*pc_peakidx_match.mean():.1f}%")
print(f"|peak_J - peak_F| > 1: {(np.abs(pc_J.argmax(axis=1) - pc_F.argmax(axis=1)) > 1).sum()}/{n}")

# Phase 10 acceptance criteria
GATE_PASS_PCT = 95
GATE_MEDIAN = 0.01
GATE_MAX = 0.05
n_pass = ((pc_med_per_scen <= GATE_MEDIAN) & (pc_max_per_scen <= GATE_MAX)).sum()
print(f"\nPhase 10 gate (pc per-scen median ≤ {GATE_MEDIAN}, max ≤ {GATE_MAX}):")
print(f"  passing: {n_pass}/{n} = {100*n_pass/n:.1f}%  (target ≥ {GATE_PASS_PCT}%)")

# Save summary npz for downstream plotting
out_npz = ROOT / "data/sulfate_parity_summary.npz"
np.savez_compressed(out_npz,
    T_rel=T_rel, gc_rel=gc_rel, pc_sum_rel=pc_sum_rel,
    pc_med_per_scen=pc_med_per_scen, pc_max_per_scen=pc_max_per_scen,
    pc_peakidx_match=pc_peakidx_match)
print(f"\nsaved {out_npz}")

# Quick scatter for visual sanity
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].loglog(pc_J.sum(axis=1), pc_F.sum(axis=1), "k.", ms=2, alpha=0.5)
xx = np.geomspace(1e-22, 1e-9, 5)
ax[0].loglog(xx, xx, "r--", lw=1, label="y=x")
ax[0].set_xlabel("JAX total pc [g/g]")
ax[0].set_ylabel("Fortran total pc [g/g]")
ax[0].set_title("Total particle mass parity")
ax[0].legend(); ax[0].grid(True, alpha=0.3, which="both")

ax[1].semilogy(pc_med_per_scen, pc_max_per_scen, "k.", ms=2, alpha=0.5)
ax[1].axhline(GATE_MAX, color="r", ls="--", lw=1, label=f"max gate {GATE_MAX}")
ax[1].axvline(GATE_MEDIAN, color="b", ls="--", lw=1, label=f"med gate {GATE_MEDIAN}")
ax[1].set_xscale("log")
ax[1].set_xlabel("Per-scenario per-bin median rel err")
ax[1].set_ylabel("Per-scenario per-bin max rel err")
ax[1].set_title("Per-bin parity (each dot = one scenario)")
ax[1].legend(); ax[1].grid(True, alpha=0.3, which="both")

plt.tight_layout()
out_dir = ROOT / "plots" / "diff" / "phase10"
out_dir.mkdir(parents=True, exist_ok=True)
out_png = out_dir / "sulfate_parity_scatter.png"
plt.savefig(out_png, dpi=110, bbox_inches="tight")
print(f"saved {out_png}")
