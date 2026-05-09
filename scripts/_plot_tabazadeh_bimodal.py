"""Diagnostic figure: per-bin rel-err for freezaerl_tabazadeh2000,
colour-coded by water activity (ssl+1). Shows that the bimodal
clustering at ~1e-15 vs ~1e-10 is driven by FP cancellation in
log(ssl+1) near saturation (ssl → 0).
"""
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import LogNorm, Normalize

ROOT = Path(__file__).parent.parent
d = np.load(ROOT / "data/freezaerl_tabazadeh2000_bench.npz")

F, J = d["rnuclg_F"], d["rnuclg_J"]
ssl = d["ssl"]; r_bins = d["r_bins"]
nbin = r_bins.shape[0]; n_scen = F.shape[0]
d_nm = 2.0 * r_bins * 1e7

denom = np.maximum(np.abs(F), np.abs(J))
denom = np.where(denom > 1e-300, denom, 1.0)
rel = np.abs(F - J) / denom
nonzero = F > 0

# Pre-Kelvin water activity per scenario (same for all bins of one scen)
ssl_clip = np.maximum(-1.0, np.minimum(0.0, ssl))
act_prek = ssl_clip + 1.0    # in [0, 1]

fig, ax = plt.subplots(1, 1, figsize=(13, 6))

# Jittered scatter, coloured by act_prek
rng = np.random.default_rng(0)
xs_all = []
ys_all = []
cols_all = []
for b in range(nbin):
    mask = nonzero[:, b]
    ys = rel[mask, b]
    if ys.size == 0:
        continue
    ys = np.where(ys > 1e-16, ys, 1e-16)
    jitter = rng.normal(loc=0.0, scale=0.07, size=ys.shape)
    xs = d_nm[b] * np.exp(jitter)
    xs_all.append(xs); ys_all.append(ys); cols_all.append(act_prek[mask])

xs_all = np.concatenate(xs_all)
ys_all = np.concatenate(ys_all)
cols_all = np.concatenate(cols_all)
sc = ax.scatter(xs_all, ys_all, c=cols_all, s=8, alpha=0.55,
                 cmap="coolwarm", vmin=0.0, vmax=1.0, edgecolors="none")

ax.set_xscale("log"); ax.set_yscale("log")
ax.axhline(1e-12, color="green", ls="--", lw=1.0, alpha=0.7,
           label="rtol = 1e-12 (unit-test gate)")
ax.axhline(1e-13, color="black", ls=":", lw=1.0, alpha=0.6,
           label="bimodal split (≈ 1e-13)")
ax.set_xlabel("Bin median diameter [nm]")
ax.set_ylabel("|J - F| / max(|J|, |F|)  per bin")
ax.set_title("Phase 11.2 bimodality diagnosis — Tabazadeh 2000 rel-err coloured by "
             "water activity (ssl + 1)\n"
             "High cluster sits where ssl+1 → 1: log(ssl+1) → 0 in the "
             "critical-germ-radius denominator → FP cancellation amplified through ag² → ΔF → exp")
ax.grid(True, alpha=0.3, which="both")
ax.legend(loc="lower left", fontsize=9)

cb = fig.colorbar(sc, ax=ax, pad=0.01, label="water activity (ssl + 1) — pre-Kelvin")
cb.ax.tick_params(labelsize=9)

plt.tight_layout()
out = ROOT / "plots" / "diff" / "phase11" / "freezaerl_tabazadeh2000_bimodal_diagnosis.png"
plt.savefig(out, dpi=110, bbox_inches="tight")
print(f"saved {out}")
