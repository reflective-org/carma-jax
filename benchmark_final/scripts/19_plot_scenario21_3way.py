"""5-panel comparison plot for scenario 21 across all three solvers.

Panels:
  (a) H2SO4 vapor concentration vs time
  (b) Total particle number concentration vs time
  (c) Initial size distribution dN/dlog10(D)
  (d) Final size distribution dN/dlog10(D)  (24 h)
  (e) Banana plot: dN/dlog10(D) heatmap (diameter × time)

Solvers overlaid: Fortran (red), faithful JAX (orange), diffrax (blue).

Reads:   benchmark_final/outputs/dt1800/scen21_3way/all_solvers.npz
Outputs: benchmark_final/plots/scen21_3way/scen21_3way.png
"""
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "scripts"))
from jax_ensemble import _minimal_config
from carma.constants import R_AIR


DATA = ROOT / "outputs" / "dt1800" / "scen21_3way" / "all_solvers.npz"
OUT_DIR = ROOT / "plots" / "scen21_3way"

DTIME = 1800.0
NSTEP = 48


def bin_diameters_nm(nbin=38, rmin_cm=2e-8, rmrat=2.0, rho=1.923):
    vmin = (4.0 / 3.0) * math.pi * rmin_cm ** 3 * rho
    rmass = vmin * rmrat ** np.arange(nbin)
    r = (3.0 * rmass / (4.0 * math.pi * rho)) ** (1.0 / 3.0)
    return 2.0 * r * 1e7


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = _minimal_config()
    rmass = np.asarray(cfg.groups[0].rmass)
    d_nm = bin_diameters_nm(cfg.nbin)
    dlog10_d = np.log10(d_nm[1] / d_nm[0])

    data = np.load(DATA, allow_pickle=False)
    scen = json.loads(str(data["scenario_info"]))
    rho_air = scen["p"] * 100.0 * 10.0 / (float(R_AIR) * scen["T"])

    SOLVERS = [
        ("fortran",  "Fortran (semi-implicit Euler)",  "tab:red",    "mmr"),
        ("faithful", "Faithful JAX (same algo)",        "tab:orange", "per_cm3"),
        ("diffrax",  "Diffrax (Kvaerno5+PID)",          "tab:blue",   "per_cm3"),
    ]
    sd = {}
    for key, _, _, pc_unit in SOLVERS:
        gc = np.asarray(data[f"{key}_gc"])
        pc = np.asarray(data[f"{key}_pc"])
        T = np.asarray(data[f"{key}_t"])
        wall = float(data[f"{key}_wall"])
        if pc_unit == "mmr":
            N_per_bin = pc * rho_air / rmass[None, :]
            gc_h2so4_cgs = gc[:, 1] * rho_air
        else:
            N_per_bin = pc
            gc_h2so4_cgs = gc[:, 1]
        sd[key] = dict(
            gc_h2so4_cgs=gc_h2so4_cgs,
            N_per_bin=N_per_bin,
            dN_dlogD=N_per_bin / dlog10_d,
            T=T, wall=wall,
        )

    t_h = np.arange(1, NSTEP + 1) * DTIME / 3600.0
    t_edges = np.concatenate([[0.0], t_h])

    # Initial state — recompute to compare against t=0 of each solver
    p_cgs = scen["p"] * 100.0 * 10.0
    rho_air_g_cm3 = p_cgs / (float(R_AIR) * scen["T"])
    mu_nm = scen["aerosol_mu_nm"]
    sigma_g = scen["aerosol_sigma_g"]
    M_ug_m3 = scen["M_total_ug_m3"]
    r = np.asarray(cfg.groups[0].r)
    log_mu_cm = math.log(mu_nm * 1e-7); log_sigma = math.log(sigma_g)
    shape_pdf = (np.exp(-0.5 * ((np.log(r) - log_mu_cm) / log_sigma) ** 2)
                  / (r * log_sigma * math.sqrt(2.0 * math.pi)) * rmass)
    norm = shape_pdf.sum()
    M_target_mmr = (M_ug_m3 * 1e-12) / rho_air_g_cm3
    N_init = shape_pdf * (M_target_mmr / norm) * rho_air_g_cm3 / rmass
    dN_dlogD_init = N_init / dlog10_d

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(
        2, 4, height_ratios=[1.0, 1.0],
        width_ratios=[1.0, 1.0, 1.0, 1.0],
        hspace=0.40, wspace=0.30,
    )

    # --- (a) H2SO4 vapor ---
    ax = fig.add_subplot(gs[0, 0])
    for key, label, c, _ in SOLVERS:
        y = sd[key]["gc_h2so4_cgs"]
        ax.plot(t_h, y, label=label, color=c, lw=2.0)
    ax.axhline(0, color="0.4", lw=0.8, ls="--", alpha=0.7)
    ax.set_yscale("symlog", linthresh=1e-18)
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("gc[H₂SO₄]  [g/cm³]")
    ax.set_title("(a) H₂SO₄ vapor concentration",
                  fontsize=11, fontweight="bold")
    ax.legend(fontsize=8.5, loc="best")
    ax.grid(True, alpha=0.3, which="major")

    # --- (b) Total particle number ---
    ax = fig.add_subplot(gs[0, 1])
    for key, label, c, _ in SOLVERS:
        N_total = sd[key]["N_per_bin"].sum(axis=1)
        ax.plot(t_h, N_total, label=label, color=c, lw=2.0)
    ax.axhline(N_init.sum(), color="k", lw=1.0, ls=":", alpha=0.7,
                label=f"initial = {N_init.sum():.1e}")
    ax.set_yscale("log")
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Σ N  [#/cm³]")
    ax.set_title("(b) Total particle number",
                  fontsize=11, fontweight="bold")
    ax.legend(fontsize=8.5, loc="best")
    ax.grid(True, alpha=0.3, which="both")

    # --- (c) Initial size distribution ---
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(d_nm, dN_dlogD_init, "k-", lw=2.4,
             label=f"input lognormal")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Diameter [nm]")
    ax.set_ylabel("dN/dlog₁₀(D)  [#/cm³]")
    ax.set_title(f"(c) Initial size distribution\nµ={mu_nm:.0f} nm, σ_g={sigma_g:.2f}, M={M_ug_m3:.2f} µg/m³",
                  fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_ylim(max(dN_dlogD_init.max() * 1e-8, 1e-4),
                 dN_dlogD_init.max() * 3.0)

    # --- (d) Final size distribution ---
    ax = fig.add_subplot(gs[0, 3])
    for key, label, c, _ in SOLVERS:
        y = sd[key]["dN_dlogD"][-1, :]
        ax.plot(d_nm, np.where(y > 0, y, np.nan), label=label, color=c, lw=2.0)
    ax.plot(d_nm, dN_dlogD_init, "k--", lw=1.2, alpha=0.5, label="initial")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Diameter [nm]")
    ax.set_ylabel("dN/dlog₁₀(D)  [#/cm³]")
    ax.set_title("(d) Final size distribution (24 h)",
                  fontsize=11, fontweight="bold")
    ax.legend(fontsize=8.5)
    ax.grid(True, alpha=0.3, which="both")

    # --- (e) Banana plots (one column per solver) ---
    all_vals = np.concatenate([
        sd[k]["dN_dlogD"].ravel() for k, _, _, _ in SOLVERS
    ])
    pos = all_vals[all_vals > 0]
    vmax = float(np.percentile(pos, 99.5)) if pos.size else 1.0
    vmin = max(vmax * 1e-8, 1e-2)

    d_edges = np.geomspace(
        d_nm[0] / (d_nm[1] / d_nm[0]) ** 0.5,
        d_nm[-1] * (d_nm[1] / d_nm[0]) ** 0.5,
        len(d_nm) + 1,
    )
    titles = {
        "fortran":  "(e1) Fortran — gc_h2so4 went negative!",
        "faithful": "(e2) Faithful JAX — same algorithm",
        "diffrax":  "(e3) Diffrax (Kvaerno5+PID)",
    }
    for j, (key, _, c, _) in enumerate(SOLVERS):
        ax = fig.add_subplot(gs[1, j])
        Z = sd[key]["dN_dlogD"].T
        Zp = np.where(Z > 0, Z, np.nan)
        im = ax.pcolormesh(t_edges, d_edges, Zp,
                            norm=LogNorm(vmin=vmin, vmax=vmax),
                            cmap="viridis", shading="auto")
        ax.set_yscale("log")
        ax.set_xlabel("Time [h]")
        if j == 0:
            ax.set_ylabel("Diameter [nm]")
        ax.set_title(titles[key], fontsize=11, fontweight="bold", color=c)
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02,
                      label="dN/dlog₁₀(D)  [#/cm³]")

    # --- summary info panel ---
    ax = fig.add_subplot(gs[1, 3]); ax.axis("off")
    info = (
        f"Scenario 21\n"
        f"  T = {scen['T']:.1f} K\n"
        f"  p = {scen['p']:.1f} hPa  (stratosphere)\n"
        f"  RH = {scen['rh']:.2f}\n"
        f"  prod_rate = {scen['h2so4_prod_rate']:.2e}\n"
        f"            molec/cm³/s\n"
        f"\n"
        f"Outer step: 1800 s × 48 = 24 h\n"
        f"\n"
        f"────────  Wall time  ────────\n"
        f"  Fortran:   {sd['fortran']['wall']:6.1f} s\n"
        f"  Faithful:  {sd['faithful']['wall']:6.1f} s\n"
        f"  Diffrax:   {sd['diffrax']['wall']:6.1f} s\n"
        f"\n"
        f"───── Final gc[H₂SO₄] ─────\n"
        f"  Fortran:   {sd['fortran']['gc_h2so4_cgs'][-1]:+.2e}\n"
        f"  Faithful:  {sd['faithful']['gc_h2so4_cgs'][-1]:+.2e}\n"
        f"  Diffrax:   {sd['diffrax']['gc_h2so4_cgs'][-1]:+.2e}\n"
        f"\n"
        f"  (negative gc is unphysical)\n"
        f"\n"
        f"───── Steps with gc<0 ─────\n"
        f"  Fortran:   {int((sd['fortran']['gc_h2so4_cgs']<0).sum())}/48\n"
        f"  Faithful:  {int((sd['faithful']['gc_h2so4_cgs']<0).sum())}/48\n"
        f"  Diffrax:   {int((sd['diffrax']['gc_h2so4_cgs']<0).sum())}/48"
    )
    ax.text(0.0, 1.0, info, transform=ax.transAxes,
             va="top", ha="left", family="monospace", fontsize=9.5,
             bbox=dict(facecolor="#f5f5f5", edgecolor="0.7", pad=8))

    fig.suptitle(
        f"Scenario 21 (Fortran ceiling-hit at 1800 s) — Fortran vs Faithful JAX vs Diffrax",
        fontsize=14, fontweight="bold", y=0.995,
    )

    out = OUT_DIR / "scen21_3way.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
