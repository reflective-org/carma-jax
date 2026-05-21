"""Plot Test B strat39 dt-spike shape: rel err vs dt across the 13 sweep dts.

Overlays Faithful and Diffrax rel-err (median, p95, max) vs the outer
step `dt`, on a log-x axis. Highlights the Faithful 30-150 s 'wide
band' where Diffrax is essentially flat.

Reads outputs from script 30 across dt in {1, 10, 20, 30, 40, 50, 60,
80, 100, 150, 200, 300, 1800}. Writes to
plots/iso_summary/test_B_strat39_dt_spike.png.
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

DTS = [1, 10, 20, 30, 40, 50, 60, 80, 100, 150, 200, 300, 1800]


def bin_diam_nm(nbin=38, rmin=2e-8, rmrat=2.0, rho=1.923):
    vmin = (4 / 3) * math.pi * rmin ** 3 * rho
    rm = vmin * rmrat ** np.arange(nbin)
    r = (3 * rm / (4 * math.pi * rho)) ** (1 / 3)
    return 2 * r * 1e7


def main():
    cfg = _minimal_config()
    rmass = np.asarray(cfg.groups[0].rmass)
    dlog10 = math.log10(bin_diam_nm()[1] / bin_diam_nm()[0])

    rows = []
    for dt in DTS:
        fn = ROOT / "outputs" / "iso_test_B" / f"strat39_dt{dt}" / "all_solvers.npz"
        if not fn.exists(): continue
        d = np.load(fn)
        scen = json.loads(str(d["scenario_info"]))
        rho_air = scen["p"] * 100.0 * 10.0 / (float(R_AIR) * scen["T"])
        N_f = (d["fortran_pc"][-1] * rho_air / rmass) / dlog10
        N_j = d["faithful_pc"][-1] / dlog10
        N_d = d["diffrax_pc"][-1] / dlog10
        floor = N_f.max() * 1e-4
        active = N_f > floor
        if not active.any(): continue
        denom = np.maximum(N_f[active], 1e-30)
        e_j = np.abs(N_j[active] - N_f[active]) / denom * 100
        e_d = np.abs(N_d[active] - N_f[active]) / denom * 100
        rows.append(dict(
            dt=dt,
            j_med=float(np.median(e_j)),
            j_p95=float(np.percentile(e_j, 95)),
            j_max=float(e_j.max()),
            d_med=float(np.median(e_d)),
            d_p95=float(np.percentile(e_d, 95)),
            d_max=float(e_d.max()),
            nsub=int(d["fortran_nsubsteps"].max()),
        ))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    dts_arr = [r["dt"] for r in rows]
    ax1.plot(dts_arr, [r["j_max"] for r in rows],
             "-o", color="tab:orange", lw=2, ms=6, label="Faithful — max")
    ax1.plot(dts_arr, [r["j_p95"] for r in rows],
             "--", color="tab:orange", lw=1.5, alpha=0.7, label="Faithful — p95")
    ax1.plot(dts_arr, [r["j_med"] for r in rows],
             ":", color="tab:orange", lw=1.2, alpha=0.6, label="Faithful — median")
    ax1.plot(dts_arr, [r["d_max"] for r in rows],
             "-d", color="tab:blue", lw=2, ms=6, label="Diffrax — max")
    ax1.plot(dts_arr, [r["d_p95"] for r in rows],
             "--", color="tab:blue", lw=1.5, alpha=0.7, label="Diffrax — p95")
    ax1.axvspan(30, 150, alpha=0.1, color="red",
                label="Faithful spike band (30–150 s)")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("outer-step dt [s]")
    ax1.set_ylabel("per-bin rel err vs Fortran [%]")
    ax1.set_title("Test B strat39 — error vs dt\n"
                  "Faithful has a wide spike at dt≈60 s; Diffrax is flat")
    ax1.grid(True, alpha=0.3, which="both")
    ax1.legend(loc="lower right", fontsize=8)

    # Right panel: substep count tracking error structure.
    ax2.plot(dts_arr, [r["nsub"] for r in rows], "-o", color="tab:red",
             lw=2, ms=6, label="Fortran substeps")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("outer-step dt [s]")
    ax2.set_ylabel("Fortran substep count (max across 48 outer steps)")
    ax2.set_title("Fortran adaptive substep count")
    ax2.grid(True, alpha=0.3, which="both")
    # Annotate the discontinuities
    for r in rows:
        ax2.annotate(f"{r['nsub']}", (r["dt"], r["nsub"]),
                     textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax2.legend()

    plt.tight_layout()
    out = ROOT / "plots" / "iso_summary" / "test_B_strat39_dt_spike.png"
    plt.savefig(out, dpi=110)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
