"""Plot BC vs ABC 6h dt-sweep — shows that coag suppresses the
nucleation-driven dt-spread.

Two figures:
  1. Total N + total mass vs dt, BC alongside ABC (each with F+D).
  2. Final dN/dlogD overlay across dt, BC and ABC side-by-side.
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

OUT = ROOT / "plots" / "iso_summary" / "BC_vs_ABC_6h"
DTS = [1, 10, 60, 300, 1800]
NBIN = 38


def bin_diam_nm(nbin=NBIN, rmin=2e-8, rmrat=2.0, rho=1.923):
    vmin = (4 / 3) * math.pi * rmin ** 3 * rho
    rm = vmin * rmrat ** np.arange(nbin)
    r = (3 * rm / (4 * math.pi * rho)) ** (1 / 3)
    return 2 * r * 1e7, r


def _load(mode, dt):
    # ABC dt=1 lives at iso_test_ABC_6h/strat39_dt1; others at iso_test_<mode>_6h_dt<dt>
    if mode == "ABC" and dt == 1:
        path = ROOT / "outputs" / "iso_test_ABC_6h" / "strat39_dt1" / "all_solvers.npz"
    else:
        path = (ROOT / "outputs" / f"iso_test_{mode}_6h_dt{dt}"
                / f"strat39_dt{dt}" / "all_solvers.npz")
    return np.load(path) if path.exists() else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = _minimal_config()
    rmass = np.asarray(cfg.groups[0].rmass)
    d_nm, _ = bin_diam_nm()
    dlog10 = math.log10(d_nm[1] / d_nm[0])

    rows = []
    for mode in ["BC", "ABC"]:
        for dt in DTS:
            d = _load(mode, dt)
            if d is None: continue
            scen = json.loads(str(d["scenario_info"]))
            rho_air = scen["p"] * 100.0 * 10.0 / (float(R_AIR) * scen["T"])
            N_F = d["fortran_pc"][-1] * rho_air / rmass
            N_D = d["diffrax_pc"][-1]
            rows.append(dict(
                mode=mode, dt=dt,
                totN_F=N_F.sum(), totN_D=N_D.sum(),
                totM_F=(N_F * rmass).sum(), totM_D=(N_D * rmass).sum(),
                dN_F=N_F / dlog10, dN_D=N_D / dlog10,
            ))

    # ---- Figure 1: totals vs dt, BC vs ABC ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    for ax, key, label in [(ax1, "totN", "Total N [#/cm³]"),
                              (ax2, "totM", "Total mass [g/cm³]")]:
        for mode, color_F, color_D, marker in [
            ("BC",  "tab:orange", "tab:red",   "o"),
            ("ABC", "tab:cyan",   "tab:blue",  "s"),
        ]:
            dts_m = [r["dt"] for r in rows if r["mode"] == mode]
            f_v = [r[f"{key}_F"] for r in rows if r["mode"] == mode]
            d_v = [r[f"{key}_D"] for r in rows if r["mode"] == mode]
            ax.plot(dts_m, f_v, f"-{marker}", color=color_F, lw=2, ms=8,
                     label=f"{mode} Fortran", alpha=0.9)
            ax.plot(dts_m, d_v, f"--{marker}", color=color_D, lw=1.5, ms=6,
                     label=f"{mode} Diffrax", alpha=0.7)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("outer-step dt [s]")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.suptitle("ABC strat39 at 6h — BC (no coag) vs ABC (with coag) across dt\n"
                  "Coag suppresses the nucleation-driven dt-spread", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT / "1_totals_BC_vs_ABC.png", dpi=110); plt.close()
    print("Saved 1_totals_BC_vs_ABC.png")

    # ---- Figure 2: dN/dlogD overlay, BC and ABC side-by-side ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(DTS)))
    for ax, mode in zip(axes, ["BC", "ABC"]):
        for dt, color in zip(DTS, colors):
            d = _load(mode, dt)
            if d is None: continue
            scen = json.loads(str(d["scenario_info"]))
            rho_air = scen["p"] * 100.0 * 10.0 / (float(R_AIR) * scen["T"])
            N_F = (d["fortran_pc"][-1] * rho_air / rmass) / dlog10
            ax.plot(d_nm, N_F, "-o", color=color, lw=1.5, ms=4,
                     label=f"dt={dt}s")
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlabel("diameter [nm]")
        ax.set_ylabel("dN/dlog₁₀(D) [#/cm³]")
        ax.set_title(f"{mode}  (Fortran)")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("ABC strat39 at 6h — dN/dlogD across dt: BC vs ABC\n"
                  "BC (no coag) shows huge dt-spread; ABC (with coag) much tighter", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT / "2_dN_BC_vs_ABC.png", dpi=110); plt.close()
    print("Saved 2_dN_BC_vs_ABC.png")

    # Summary table
    print(f"\n{'dt':>6} {'BC totN':>13} {'ABC totN':>13} {'coag-removed':>14} "
          f"{'BC/ABC ratio':>14}")
    for dt in DTS:
        bc = next((r for r in rows if r["mode"] == "BC" and r["dt"] == dt), None)
        abc = next((r for r in rows if r["mode"] == "ABC" and r["dt"] == dt), None)
        if bc and abc:
            ratio = bc['totN_F'] / abc['totN_F']
            print(f"{dt:>5}s {bc['totN_F']:>13.3e} {abc['totN_F']:>13.3e} "
                  f"{bc['totN_F']-abc['totN_F']:>14.3e} {ratio:>13.1f}x")


if __name__ == "__main__":
    main()
