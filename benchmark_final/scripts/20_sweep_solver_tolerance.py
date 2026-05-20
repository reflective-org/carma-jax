"""Phase 2A: solver + tolerance sweep on scenario 21.

Sweeps `(solver_name, rtol)` and records wall time, accepted/rejected step
counts, and final-state agreement with the strictest reference run
(Kvaerno5 + rtol=1e-7). Picks the best combo by `wall × correctness`.

Output: benchmark_final/outputs/dt1800/sweep_2A/sweep_results.npz
        benchmark_final/plots/sweep_2A/wall_vs_error.png
"""
import argparse
import math
import sys
import time
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "scripts"))

OUT_DIR = ROOT / "outputs" / "dt1800" / "sweep_2A"
PLOT_DIR = ROOT / "plots" / "sweep_2A"

SCEN_IDX = 21
DTIME = 1800.0
NSTEP = 48

SOLVERS = ["Kvaerno3", "Kvaerno4", "Kvaerno5", "KenCarp4"]
TOLS = [1e-3, 1e-4, 1e-5]
REFERENCE_SOLVER = "Kvaerno5"
REFERENCE_TOL = 1e-7


def _build_init():
    """Build cfg, ppm, initial (pc, gc, T, p_cgs) for scenario 21."""
    from jax_ensemble import _minimal_config, _compute_ppm_coefs
    from carma.constants import RPA2CGS, R_AIR
    from carma.precision import DTYPE

    S = np.load(ROOT / "scenarios" / "realistic_scenarios_100.npz")
    T_K = float(S["T"][SCEN_IDX])
    p_hPa = float(S["p"][SCEN_IDX])
    rh = float(S["rh"][SCEN_IDX])
    prod_rate = float(S["h2so4_prod_rate"][SCEN_IDX])
    M_ug_m3 = float(S["M_total_ug_m3"][SCEN_IDX])
    mu_nm = float(S["aerosol_mu_nm"][SCEN_IDX])
    sigma_g = float(S["aerosol_sigma_g"][SCEN_IDX])

    cfg = _minimal_config()
    ppm = _compute_ppm_coefs(cfg)
    p_cgs_val = p_hPa * 100.0 * float(RPA2CGS)
    rhoa = p_cgs_val / (float(R_AIR) * T_K)
    p_cgs = jnp.asarray([p_cgs_val], dtype=DTYPE)
    T = jnp.asarray([T_K], dtype=DTYPE)

    pvapl_Pa = math.exp(54.842763 - 6763.22 / T_K
                         - 4.210 * math.log(T_K) + 0.000367 * T_K)
    h2o_mmr = rh * pvapl_Pa * 18.0 / (29.0 * p_hPa * 100.0)
    gc = jnp.asarray([[h2o_mmr * rhoa, 0.0]], dtype=DTYPE)

    grp = cfg.groups[0]
    r = np.asarray(grp.r)
    rmass_np = np.asarray(grp.rmass)
    log_mu_cm = math.log(mu_nm * 1e-7)
    log_sigma = math.log(sigma_g)
    shape_pdf = (np.exp(-0.5 * ((np.log(r) - log_mu_cm) / log_sigma) ** 2)
                  / (r * log_sigma * math.sqrt(2.0 * math.pi)) * rmass_np)
    norm = shape_pdf.sum()
    if norm > 0:
        M_target_mmr = (M_ug_m3 * 1e-12) / rhoa
        mmr_per_bin = shape_pdf * (M_target_mmr / norm)
        pc_per_bin = mmr_per_bin * rhoa / rmass_np
    else:
        pc_per_bin = np.zeros_like(r)
    pc0 = jnp.asarray(pc_per_bin, dtype=DTYPE)[:, None]
    dgc_per_step = prod_rate * DTIME * 98.078479 / 6.02252e23
    return cfg, ppm, pc0, gc, T, p_cgs, dgc_per_step


def _run_one(cfg, ppm, pc0, gc0_2d, T_arr, p_cgs, dgc_per_step,
              solver_name, rtol, atol):
    """Run scenario 21 with the given (solver, rtol, atol)."""
    from jax_ensemble import _refresh_env
    from carma.precision import DTYPE
    from carma_diffrax import DiffraxConfig, diffrax_step
    from carma_diffrax.rhs import FrozenEnv
    from carma_diffrax.state import StateShape

    shape = StateShape(nbin=cfg.nbin, nelem=1, ngas=2)
    grp = cfg.groups[0]
    rmass_np = np.asarray(grp.rmass)
    rmassup_np = np.asarray(grp.rmassup)
    rmasslow_np = np.concatenate([[rmass_np[0]/(grp.rmrat**0.5)], rmassup_np[:-1]])
    dm = jnp.asarray(rmassup_np - rmasslow_np, dtype=DTYPE)

    pc = pc0
    gc = gc0_2d
    T_scalar = T_arr[0]
    cfg_d = DiffraxConfig(rtol=rtol, atol=atol, max_steps=50_000,
                            solver_name=solver_name)

    total_acc, total_rej, n_fail = 0, 0, 0
    t_start = time.perf_counter()
    for istep in range(NSTEP):
        gc = gc.at[0, 1].set(gc[0, 1] + dgc_per_step)
        env_d = _refresh_env(jnp.atleast_1d(T_scalar), p_cgs, gc, cfg, ppm)
        r_wet = (env_d["rup_wet"][0, :, 0] + env_d["rlow_wet"][0, :, 0]) / 2.0
        env = FrozenEnv(
            akelvin=env_d["akelvin"], akelvini=env_d["akelvini"],
            gro=env_d["gro"], gro1=env_d["gro1"],
            rup_wet=env_d["rup_wet"], r_wet=r_wet,
            rmass=jnp.asarray(grp.rmass, dtype=DTYPE)[:, None],
            dm=dm[:, None],
            rmassup=jnp.asarray(grp.rmassup, dtype=DTYPE),
            rmrat_val=float(grp.rmrat),
            rhoa=env_d["rhoa"], zmet=env_d["zmet"],
            rlhe=env_d["rlhe"], rlhm=env_d["rlhm"],
        )
        pc, gc_1d, T_scalar, stats = diffrax_step(
            pc, gc[0], T_scalar, DTIME, env, shape, cfg_d,
        )
        gc = gc_1d[None, :]
        total_acc += stats["num_accepted_steps"]
        total_rej += stats["num_rejected_steps"]
        if not stats["successful"]:
            n_fail += 1
    wall = time.perf_counter() - t_start
    return dict(
        wall=wall, total_acc=total_acc, total_rej=total_rej, n_fail=n_fail,
        pc_final=np.asarray(pc[:, 0]),
        gc_final=np.asarray(gc[0, :]),
        T_final=float(T_scalar),
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 2A: solver + tolerance sweep on scenario 21 ===\n")
    cfg, ppm, pc0, gc0, T_arr, p_cgs, dgc_per_step = _build_init()

    # Reference run — strictest tolerance
    print(f"Reference: {REFERENCE_SOLVER} @ rtol={REFERENCE_TOL:.0e} …",
           flush=True)
    ref = _run_one(cfg, ppm, pc0, gc0, T_arr, p_cgs, dgc_per_step,
                    REFERENCE_SOLVER, REFERENCE_TOL, REFERENCE_TOL)
    print(f"  wall={ref['wall']:.1f}s, "
           f"acc={ref['total_acc']}, rej={ref['total_rej']}, "
           f"gc_final={ref['gc_final'][1]:.3e}", flush=True)

    rows = []
    for solver in SOLVERS:
        for tol in TOLS:
            print(f"\n{solver} @ rtol=atol={tol:.0e} …", flush=True)
            try:
                r = _run_one(cfg, ppm, pc0, gc0, T_arr, p_cgs, dgc_per_step,
                              solver, tol, tol)
            except Exception as e:
                print(f"  FAILED: {e}", flush=True)
                continue
            denom = np.maximum(np.abs(ref["pc_final"]), np.abs(r["pc_final"]))
            denom = np.where(denom > 1e-300, denom, 1.0)
            pc_err = np.abs(r["pc_final"] - ref["pc_final"]) / denom
            pc_max_err = float(np.max(pc_err))
            gc_err = abs(r["gc_final"][1] - ref["gc_final"][1]) \
                      / max(abs(ref["gc_final"][1]), 1e-30)
            rej_ratio = r["total_rej"] / max(r["total_acc"] + r["total_rej"], 1)
            rows.append(dict(
                solver=solver, rtol=tol, atol=tol,
                wall=r["wall"], total_acc=r["total_acc"],
                total_rej=r["total_rej"], rej_ratio=rej_ratio,
                pc_max_err=pc_max_err, gc_err=gc_err, n_fail=r["n_fail"],
            ))
            print(f"  wall={r['wall']:6.1f}s | acc={r['total_acc']:5d} "
                   f"rej={r['total_rej']:5d} ({100*rej_ratio:.0f}%) | "
                   f"pc_max_err={pc_max_err:.2e} gc_err={gc_err:.2e}",
                   flush=True)

    # Save and summarize
    np.savez_compressed(
        OUT_DIR / "sweep_results.npz",
        rows=np.array(rows, dtype=object),
        ref_wall=np.array([ref["wall"]]),
        ref_pc=ref["pc_final"], ref_gc=ref["gc_final"],
    )
    print(f"\nSaved: {OUT_DIR/'sweep_results.npz'}")

    # Find best — score = wall × (max_err / 1e-2) clipped at 1.0
    print("\n=== Summary (sorted by wall) ===")
    rows_sorted = sorted(rows, key=lambda r: r["wall"])
    print(f"{'solver':<10} {'rtol':>6} {'wall':>8} {'acc':>7} {'rej%':>5} "
           f"{'pc_max_err':>10} {'gc_err':>10}")
    for r in rows_sorted:
        print(f"{r['solver']:<10} {r['rtol']:>6.0e} {r['wall']:>7.1f}s "
               f"{r['total_acc']:>7d} {100*r['rej_ratio']:>4.0f}% "
               f"{r['pc_max_err']:>10.2e} {r['gc_err']:>10.2e}")


if __name__ == "__main__":
    main()
