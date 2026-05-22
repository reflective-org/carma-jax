"""One-page summary plot: per-bin rel-err vs Fortran across the 3 isolation tests.

A 3×4 grid of heatmaps:

  rows: Test A (coag), Test B (condensation), Test C (nuc+growth)
  cols: Faithful strat39, Faithful trop96, Diffrax strat39, Diffrax trop96

Each cell: 5 dt rows × (median, p95, max) cols, color = rel err % vs Fortran.

Active bins only — bins where Fortran dN/dlogD > 1e-4 × peak.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "scripts"))

from carma.constants import R_AIR
from jax_ensemble import _minimal_config

OUT = ROOT / "plots" / "iso_summary"
TESTS = ["A", "B", "C"]
TEST_LABEL = {
    "A": "A (coag)",
    "B": "B (condensation)",
    "C": "C (nuc+growth)",
}
ATMS = ["strat39", "trop96"]
DTS = [1, 10, 60, 300, 1800]


def bin_diam_nm(nbin=38, rmin=2e-8, rmrat=2.0, rho=1.923):
    vmin = (4 / 3) * math.pi * rmin ** 3 * rho
    rm = vmin * rmrat ** np.arange(nbin)
    r = (3 * rm / (4 * math.pi * rho)) ** (1 / 3)
    return 2 * r * 1e7


def compute_stats(test, atm, dt):
    fn = ROOT / "outputs" / f"iso_test_{test}" / f"{atm}_dt{dt}" / "all_solvers.npz"
    if not fn.exists():
        return None
    d = np.load(fn)
    scen = json.loads(str(d["scenario_info"]))
    rho_air = scen["p"] * 100.0 * 10.0 / (float(R_AIR) * scen["T"])
    cfg = _minimal_config()
    rmass = np.asarray(cfg.groups[0].rmass)
    dlog10 = math.log10(bin_diam_nm()[1] / bin_diam_nm()[0])
    N_f = (d["fortran_pc"][-1] * rho_air / rmass) / dlog10
    N_j = d["faithful_pc"][-1] / dlog10
    N_d = d["diffrax_pc"][-1] / dlog10
    floor = N_f.max() * 1e-4
    active = N_f > floor
    if not active.any():
        return None
    def stats(N_other):
        denom = np.maximum(N_f[active], 1e-30)
        e = np.abs(N_other[active] - N_f[active]) / denom * 100
        return float(np.median(e)), float(np.percentile(e, 95)), float(e.max())
    j = stats(N_j)
    dd = stats(N_d)
    return j, dd


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # 3 rows (tests) × 4 cols (J-strat, J-trop, D-strat, D-trop).
    fig, axes = plt.subplots(3, 4, figsize=(15, 10),
                              sharex=False, sharey=True)
    col_titles = ["Faithful — strat39", "Faithful — trop96",
                   "Diffrax — strat39", "Diffrax — trop96"]
    stat_labels = ["median", "p95", "max"]

    # Collect all values to set a common color scale.
    all_data = {}
    for test in TESTS:
        for atm in ATMS:
            for dt in DTS:
                s = compute_stats(test, atm, dt)
                if s is None: continue
                all_data[(test, atm, dt)] = s

    # Global vmax for the colormap (clip at 100 % so the heatmap is readable).
    vmax = 50.0

    for itest, test in enumerate(TESTS):
        for icol, (solver_idx, atm) in enumerate([
            (0, "strat39"), (0, "trop96"),
            (1, "strat39"), (1, "trop96"),
        ]):
            ax = axes[itest, icol]
            # Build matrix: rows=dt, cols=stat_labels
            mat = np.zeros((len(DTS), 3))
            for irow, dt in enumerate(DTS):
                s = all_data.get((test, atm, dt))
                if s is None:
                    mat[irow] = np.nan
                else:
                    mat[irow] = s[solver_idx]   # 0=J, 1=D
            im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r",
                           vmin=0, vmax=vmax, origin="lower",
                           extent=(-0.5, 2.5, -0.5, len(DTS) - 0.5))
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(stat_labels)
            ax.set_yticks(range(len(DTS)))
            ax.set_yticklabels([f"{dt} s" for dt in DTS])
            # Annotate each cell with the percentage. With origin='lower'
            # mat[irow] sits at y=irow on the axes.
            for irow in range(len(DTS)):
                for icl in range(3):
                    v = mat[irow, icl]
                    if np.isnan(v): continue
                    color = "black" if v < 25 else "white"
                    ax.text(icl, irow,
                             f"{v:.1f}%",
                             ha="center", va="center",
                             color=color, fontsize=8)
            if itest == 0:
                ax.set_title(col_titles[icol], fontsize=11)
            if icol == 0:
                ax.set_ylabel(f"Test {TEST_LABEL[test]}\nouter-step dt",
                                fontsize=11)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(),
                         shrink=0.5, label="per-bin rel err vs Fortran [%]")
    fig.suptitle("Physics-isolation tests — per-bin dN/dlogD rel err vs Fortran\n"
                  "(active bins only; clipped at 50 %)",
                  fontsize=13, y=0.99)
    out = OUT / "isolation_relerr_summary.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
