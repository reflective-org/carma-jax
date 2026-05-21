"""Run carma_diffrax over all 100 realistic scenarios at 1800s × 48 steps.

Mirrors the structure of 04_run_jax_parallel.py — multiprocessing.Pool
with a per-worker JIT cache — but calls carma_diffrax.diffrax_step
instead of step_full_faithful.

Output: benchmark_final/outputs/dt1800/diffrax/jax_diffrax_outputs.npz
Schema matches the faithful-port outputs (T_final, gc_h2so4_final,
pc_final, plus per-scenario step stats summary).
"""
import argparse
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent


def _worker_run(args):
    """Run one scenario in a worker process. Returns (idx, result_dict)."""
    idx, scen_dict, dtime, nstep, rtol, atol = args

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    sys.path.insert(0, str(REPO / "scripts"))
    from jax_ensemble import _minimal_config, _compute_ppm_coefs, _refresh_env
    from carma.constants import AVG, R_AIR, RPA2CGS
    from carma.precision import DTYPE
    from carma_diffrax import DiffraxConfig, diffrax_step
    from carma_diffrax.rhs import FrozenEnv
    from carma_diffrax.state import StateShape

    # One-time config build per worker
    if getattr(_worker_run, "_cached_cfg", None) is None:
        cfg = _minimal_config()
        ppm = _compute_ppm_coefs(cfg)
        grp = cfg.groups[0]
        rmass_np = np.asarray(grp.rmass)
        rmassup_np = np.asarray(grp.rmassup)
        rmasslow_np = np.concatenate(
            [[rmass_np[0] / (grp.rmrat ** 0.5)], rmassup_np[:-1]]
        )
        dm = jnp.asarray(rmassup_np - rmasslow_np, dtype=DTYPE)
        shape = StateShape(nbin=cfg.nbin, nelem=1, ngas=2)
        _worker_run._cached_cfg = (cfg, ppm, grp, dm, shape)
    cfg, ppm, grp, dm, shape = _worker_run._cached_cfg

    T_K = scen_dict["T"]
    p_hPa = scen_dict["p"]
    rh = scen_dict["rh"]
    prod_rate = scen_dict["h2so4_prod_rate"]
    M_ug_m3 = scen_dict["M_total_ug_m3"]
    mu_nm = scen_dict["aerosol_mu_nm"]
    sigma_g = scen_dict["aerosol_sigma_g"]

    p_cgs_val = p_hPa * 100.0 * float(RPA2CGS)
    rhoa_g_cm3 = p_cgs_val / (float(R_AIR) * T_K)
    p_cgs = jnp.asarray([p_cgs_val], dtype=DTYPE)

    pvapl_Pa = math.exp(54.842763 - 6763.22 / T_K
                         - 4.210 * math.log(T_K) + 0.000367 * T_K)
    h2o_mmr = rh * pvapl_Pa * 18.0 / (29.0 * p_hPa * 100.0)
    gc = jnp.asarray([[h2o_mmr * rhoa_g_cm3, 0.0]], dtype=DTYPE)

    # Initial lognormal aerosol seed
    r = np.asarray(grp.r)
    rmass_np = np.asarray(grp.rmass)
    log_mu_cm = math.log(mu_nm * 1e-7)
    log_sigma = math.log(sigma_g)
    shape_pdf = (np.exp(-0.5 * ((np.log(r) - log_mu_cm) / log_sigma) ** 2)
                  / (r * log_sigma * math.sqrt(2.0 * math.pi)) * rmass_np)
    norm = shape_pdf.sum()
    if norm > 0:
        M_target_mmr = (M_ug_m3 * 1e-12) / rhoa_g_cm3
        mmr_per_bin = shape_pdf * (M_target_mmr / norm)
        pc_per_bin = mmr_per_bin * rhoa_g_cm3 / rmass_np
    else:
        pc_per_bin = np.zeros_like(r)
    pc = jnp.asarray(pc_per_bin, dtype=DTYPE)[:, None]

    dgc_per_step = prod_rate * dtime * 98.078479 / float(AVG)
    cfg_d = DiffraxConfig(rtol=rtol, atol=atol, max_steps=20_000)

    T_scalar = jnp.asarray(T_K, dtype=DTYPE)
    total_accepted = 0
    total_rejected = 0
    n_failures = 0
    min_gc_h2so4 = float("inf")

    for istep in range(nstep):
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
            pratt=env_d["pratt"][..., 0],
            prat=env_d["prat"][..., 0],
            pden1=env_d["pden1"][..., 0],
            palr=env_d["palr"][..., 0],
        )
        pc, gc_1d, T_scalar, stats = diffrax_step(
            pc, gc[0], T_scalar, dtime, env, shape, cfg_d,
        )
        gc = gc_1d[None, :]
        total_accepted += stats["num_accepted_steps"]
        total_rejected += stats["num_rejected_steps"]
        if not stats["successful"]:
            n_failures += 1
        min_gc_h2so4 = min(min_gc_h2so4, float(gc[0, 1]))

    T_final = float(T_scalar)
    gc_final = float(gc[0, 1]) / rhoa_g_cm3
    pc_final = np.asarray(pc[:, 0]) * rmass_np / rhoa_g_cm3

    return (idx, dict(
        T_final=T_final,
        gc_h2so4_final=gc_final,
        pc_final=pc_final.tolist(),
        total_accepted=total_accepted,
        total_rejected=total_rejected,
        n_failures=n_failures,
        min_gc_h2so4_cgs=min_gc_h2so4,
    ))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenarios", type=Path,
                    default=ROOT / "scenarios" / "realistic_scenarios_100.npz")
    p.add_argument("--out", type=Path,
                    default=ROOT / "outputs" / "dt1800" / "diffrax"
                            / "jax_diffrax_outputs.npz")
    p.add_argument("--dtime", type=float, default=1800.0)
    p.add_argument("--nstep", type=int, default=48)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--rtol", type=float, default=1e-5)
    p.add_argument("--atol", type=float, default=1e-5)
    p.add_argument("--n", type=int, default=None,
                    help="run only first N scenarios (for quick sanity)")
    args = p.parse_args()

    scens = np.load(args.scenarios)
    n = int(scens["_n"])
    if args.n is not None:
        n = min(n, args.n)

    work = []
    for i in range(n):
        scen_dict = {k: float(scens[k][i]) for k in
                       ["T", "p", "rh", "h2so4_prod_rate", "M_total_ug_m3",
                        "aerosol_mu_nm", "aerosol_sigma_g"]}
        work.append((i, scen_dict, args.dtime, args.nstep, args.rtol, args.atol))

    print(f"Running diffrax over {n} scenarios "
          f"({args.nstep} × {args.dtime}s = {args.nstep*args.dtime/3600:.0f}h), "
          f"{args.workers} workers, rtol={args.rtol}, atol={args.atol}", flush=True)
    t_start = time.perf_counter()

    nbin = 38
    T_final = np.zeros(n)
    gc_final = np.zeros(n)
    pc_final = np.zeros((n, nbin))
    total_accepted = np.zeros(n, dtype=np.int64)
    total_rejected = np.zeros(n, dtype=np.int64)
    n_failures = np.zeros(n, dtype=np.int64)
    min_gc = np.zeros(n)

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.workers) as pool:
        done = 0
        for idx, out in pool.imap_unordered(_worker_run, work, chunksize=1):
            T_final[idx] = out["T_final"]
            gc_final[idx] = out["gc_h2so4_final"]
            pc_final[idx] = out["pc_final"]
            total_accepted[idx] = out["total_accepted"]
            total_rejected[idx] = out["total_rejected"]
            n_failures[idx] = out["n_failures"]
            min_gc[idx] = out["min_gc_h2so4_cgs"]
            done += 1
            elapsed = time.perf_counter() - t_start
            print(f"  [{done:3d}/{n}] scen {idx:3d} done  "
                  f"(acc={out['total_accepted']:5d} rej={out['total_rejected']:5d} "
                  f"fails={out['n_failures']} min_gc={out['min_gc_h2so4_cgs']:.2e}) "
                  f"{elapsed/done:.1f}s/scen avg, {elapsed:.0f}s elapsed",
                  flush=True)

    elapsed = time.perf_counter() - t_start
    print(f"\ndiffrax wall time: {elapsed:.0f}s "
          f"({elapsed/n*1000:.0f} ms/scenario avg)")
    print(f"Scenarios with any diffrax failure: {int((n_failures>0).sum())}/{n}")
    print(f"Scenarios with gc_h2so4 ever negative: {int((min_gc<0).sum())}/{n}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        T_final=T_final, gc_h2so4_final=gc_final, pc_final=pc_final,
        total_accepted=total_accepted, total_rejected=total_rejected,
        n_failures=n_failures, min_gc_h2so4=min_gc,
        wall_time_s=np.array([elapsed]), n_scenarios=np.array([n]),
    )
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
