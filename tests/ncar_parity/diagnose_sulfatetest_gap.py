"""Diagnose where faithful JAX diverges from NCAR Fortran on carma_sulfatetest.

Runs faithful JAX with per-outer-step state saved; loads the bench
history; produces a 4-panel diagnostic plot:

  (a) Total particle mass mmr vs time — F90 vs faithful
  (b) Gas H2SO4 mmr vs time — F90 vs faithful
  (c) Per-bin mmr at three time snapshots (early, mid, late)
  (d) Per-bin rel err vs time

This is a debugging-driven exploration script. Outputs to
benchmark_final/plots/ncar_parity/sulfatetest_diagnose.png
"""
import math
import sys
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))   # so 'tests' package is importable when run directly

from tests.ncar_parity._harness import parse_bench, bench_path


def run_faithful_trajectory():
    """Run faithful JAX with carma_sulfatetest config, save per-step state."""
    from jax_ensemble import _minimal_config, _compute_ppm_coefs, _refresh_env
    from carma.constants import R_AIR, RPA2CGS
    from carma.precision import DTYPE
    from carma.step_full_faithful import make_step_full_faithful
    from carma.prestep import prestep

    P_PA = 90.0 * 100.0
    T_K = 250.0
    DTIME = 1800.0
    NSTEP = 100
    MMR_H2O_INIT = 100e-6
    MMR_H2SO4_INIT = 0.1e-9 * (98.0 / 29.0)

    rho_air = P_PA * 10.0 / (float(R_AIR) * T_K)
    p_cgs = jnp.asarray([P_PA * RPA2CGS], dtype=DTYPE)
    T = jnp.asarray([T_K], dtype=DTYPE)
    cfg = _minimal_config()
    cfg = cfg._replace(do_coag=True, do_grow=True)
    ppm = _compute_ppm_coefs(cfg)

    gc = jnp.asarray([[MMR_H2O_INIT * rho_air,
                        MMR_H2SO4_INIT * rho_air]], dtype=DTYPE)
    pc = jnp.zeros((1, cfg.nbin, cfg.nelem), dtype=DTYPE)
    t = T

    step = make_step_full_faithful(cfg, ppm_coefs=ppm)
    itype_arr = jnp.asarray([e.itype for e in cfg.elements])
    ienconc_arr = jnp.asarray([g.ienconc for g in cfg.groups])
    igelem_arr = jnp.asarray([e.igroup for e in cfg.elements])
    rmass_2d = jnp.stack(
        [jnp.asarray(g.rmass, dtype=DTYPE) for g in cfg.groups], axis=1,
    )

    pc_hist = np.zeros((NSTEP, cfg.nbin))
    gc_hist = np.zeros((NSTEP, 2))
    T_hist = np.zeros(NSTEP)

    for istep in range(NSTEP):
        env_s = _refresh_env(t, p_cgs, gc, cfg, ppm)
        pc, gc, t, pcl, gcl, d_gc, d_t, pconmax = prestep(
            pc, gc, t, pc, gc, t, env_s["zmet"],
            itype_arr, ienconc_arr, igelem_arr, rmass_2d,
            do_substep=True, do_coag=cfg.do_coag,
        )
        pc, gc, t, _diag = step(
            pc=pc, gc=gc, t=t, dtime=float(DTIME),
            rhoa=env_s["rhoa"], zmet=env_s["zmet"],
            akelvin=env_s["akelvin"], akelvini=env_s["akelvini"],
            gro=env_s["gro"], gro1=env_s["gro1"],
            rup_wet=env_s["rup_wet"],
            rlhe=env_s["rlhe"], rlhm=env_s["rlhm"],
            ckernel=env_s["ckernel"], pconmax=pconmax,
            ds_threshold_arr=env_s["ds_threshold_arr"],
            pcl=pcl, gcl=gcl, told=t, d_gc=d_gc, d_t=d_t,
        )
        pc_hist[istep] = np.asarray(pc[0, :, 0])
        gc_hist[istep] = np.asarray(gc[0, :])
        T_hist[istep] = float(t[0])

    rmass = np.asarray(cfg.groups[0].rmass)
    mmr_hist = pc_hist * rmass[None, :] / rho_air         # to mmr (g/g)
    mmr_gas_hist = gc_hist / rho_air
    # Simulation time array (s): writes happen AFTER each step, so step k
    # corresponds to physical time (k+1) * DTIME.
    time_s = (np.arange(NSTEP) + 1) * DTIME
    return mmr_hist, mmr_gas_hist, time_s, rho_air, rmass


def main():
    bench = parse_bench(
        bench_path("carma_sulfatetest"),
        pre_mmr_skip=4, per_step_skip=3,
    )
    mmr_F = bench.mmr_history[:, 0, :]      # (nstep, nbin)
    mmr_gas_F = bench.mmr_gas_history       # (nstep, 2)
    t_F = bench.times                       # (nstep,)

    print("Running faithful JAX trajectory...")
    mmr_J, mmr_gas_J, t_J, rho_air, rmass = run_faithful_trajectory()
    print(f"  rho_air = {rho_air:.3e} g/cm^3")
    print(f"  nstep (F): {len(t_F)},  nstep (J): {len(t_J)}")

    # Per-bin diameter in nm for plotting
    d_nm = 2.0 * (3.0 * rmass / (4 * np.pi * 1.923)) ** (1/3) * 1e7

    OUT_DIR = REPO / "benchmark_final" / "plots" / "ncar_parity"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10),
                              gridspec_kw=dict(hspace=0.35, wspace=0.30))

    # (a) Total mass vs time
    ax = axes[0, 0]
    ax.plot(t_F / 3600, mmr_F.sum(axis=1), "r-", lw=2, label="NCAR Fortran")
    ax.plot(t_J / 3600, mmr_J.sum(axis=1), "b--", lw=2, label="Faithful JAX")
    ax.set_xlabel("Time [h]"); ax.set_ylabel("Σ particle mmr [g/g]")
    ax.set_yscale("log")
    ax.set_title("(a) Total particle mass")
    ax.legend(fontsize=10); ax.grid(alpha=0.3, which="both")

    # (b) Gas H2SO4 vs time
    ax = axes[0, 1]
    ax.plot(t_F / 3600, mmr_gas_F[:, 1], "r-", lw=2, label="NCAR Fortran")
    ax.plot(t_J / 3600, mmr_gas_J[:, 1], "b--", lw=2, label="Faithful JAX")
    ax.set_xlabel("Time [h]"); ax.set_ylabel("gc[H₂SO₄] mmr [g/g]")
    ax.set_yscale("log")
    ax.set_title("(b) H₂SO₄ vapor")
    ax.legend(fontsize=10); ax.grid(alpha=0.3, which="both")

    # (c) Per-bin mmr at three snapshots
    ax = axes[1, 0]
    snapshots = [0, 25, 49, len(t_F) - 1]   # 1, 26, 50, 100 hr roughly
    colors = ["tab:purple", "tab:green", "tab:orange", "tab:red"]
    for i, c in zip(snapshots, colors):
        ax.plot(d_nm, np.where(mmr_F[i] > 0, mmr_F[i], np.nan),
                 color=c, lw=2.0, label=f"F t={t_F[i]/3600:.0f}h")
        ax.plot(d_nm, np.where(mmr_J[i] > 0, mmr_J[i], np.nan),
                 color=c, lw=1.5, ls="--",
                 label=f"J t={t_J[i]/3600:.0f}h")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Diameter [nm]"); ax.set_ylabel("mmr per bin [g/g]")
    ax.set_ylim(1e-30, 1e-9)
    ax.set_title("(c) Per-bin mmr snapshots")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3, which="both")

    # (d) Per-bin rel err vs time (median over active bins)
    ax = axes[1, 1]
    n = min(len(t_F), len(t_J))
    med_err = np.zeros(n); p95_err = np.zeros(n); max_err = np.zeros(n)
    for k in range(n):
        denom = np.maximum(np.abs(mmr_F[k]), np.abs(mmr_J[k]))
        active = mmr_F[k] > 1e-13
        if active.any():
            e = np.abs(mmr_F[k][active] - mmr_J[k][active]) / np.maximum(
                denom[active], 1e-30)
            med_err[k] = np.median(e)
            p95_err[k] = np.percentile(e, 95)
            max_err[k] = e.max()
    ax.plot(t_F[:n] / 3600, med_err, "-",  color="tab:blue",
             lw=2, label="median")
    ax.plot(t_F[:n] / 3600, p95_err, "--", color="tab:orange",
             lw=1.5, label="P95")
    ax.plot(t_F[:n] / 3600, max_err, ":",  color="tab:red",
             lw=1.5, label="max")
    ax.axhline(0.05, color="green", ls="--", lw=1, alpha=0.6, label="5 %")
    ax.set_xlabel("Time [h]"); ax.set_ylabel("|F − J| / max(|F|,|J|)")
    ax.set_yscale("log")
    ax.set_title("(d) Per-bin rel err vs time (active bins)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")

    fig.suptitle(
        "Faithful JAX vs NCAR carma_sulfatetest — divergence diagnosis\n"
        "(NBIN=38, 1800 s × 100, ZhaoTurco + growth + coag)",
        fontsize=13, fontweight="bold", y=0.995,
    )
    out = OUT_DIR / "sulfatetest_diagnose.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved: {out}")

    # Numeric summary
    print(f"\nDivergence summary (per-bin rel err over time):")
    for k in [0, 10, 25, 50, 99]:
        if k < n:
            print(f"  t = {t_F[k]/3600:5.1f} h: "
                   f"median {med_err[k]:.2e}, P95 {p95_err[k]:.2e}, "
                   f"max {max_err[k]:.2e}")


if __name__ == "__main__":
    main()
