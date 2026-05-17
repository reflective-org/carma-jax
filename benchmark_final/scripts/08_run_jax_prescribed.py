"""Run JAX with Fortran's exact substep schedule (Phase 10.6 prescribed mode).

For each (scenario, outer_step) pair, JAX runs EXACTLY the same number of
substeps Fortran chose adaptively. This eliminates the FP retry-decision
drift between JAX and Fortran — if both sides do the same arithmetic in
the same order on the same number of substeps, the only remaining
difference is FMA fusion (which is ULP-level).

Inputs:
  - benchmark_final/scenarios/realistic_scenarios_100.npz
  - benchmark_final/outputs/prescribed/fortran_outputs_schedules.npz
    (must have been produced by the Fortran orchestrator's --out path)

Output:
  - benchmark_final/outputs/prescribed/jax_outputs.npz
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
    idx, scen_dict, dtime, nsubsteps_per_step = args

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    sys.path.insert(0, str(REPO / "scripts"))
    from jax_ensemble import _minimal_config, _compute_ppm_coefs, _refresh_env
    from carma.constants import AVG, R_AIR, RPA2CGS
    from carma.step_full_faithful import make_step_full_faithful
    from carma.prestep import prestep
    from carma.precision import DTYPE

    if not hasattr(_worker_run, "_cached"):
        cfg = _minimal_config()
        ppm = _compute_ppm_coefs(cfg)
        step = make_step_full_faithful(cfg, ppm_coefs=ppm)
        itype_arr   = jnp.asarray([e.itype for e in cfg.elements])
        ienconc_arr = jnp.asarray([g.ienconc for g in cfg.groups])
        igelem_arr  = jnp.asarray([e.igroup for e in cfg.elements])
        rmass_2d    = jnp.stack(
            [jnp.asarray(g.rmass, dtype=DTYPE) for g in cfg.groups], axis=1,
        )
        _worker_run._cached = (cfg, ppm, step, itype_arr, ienconc_arr,
                                igelem_arr, rmass_2d)
    cfg, ppm, step, itype_arr, ienconc_arr, igelem_arr, rmass_2d = _worker_run._cached

    T_K = scen_dict["T"]; p_hPa = scen_dict["p"]; rh = scen_dict["rh"]
    prod_rate = scen_dict["h2so4_prod_rate"]; M_ug_m3 = scen_dict["M_total_ug_m3"]
    mu_nm = scen_dict["aerosol_mu_nm"]; sigma_g = scen_dict["aerosol_sigma_g"]

    p_cgs_val = p_hPa * 100.0 * float(RPA2CGS)
    rhoa_g_cm3 = p_cgs_val / (float(R_AIR) * T_K)

    T = jnp.asarray([T_K], dtype=DTYPE)
    p_cgs = jnp.asarray([p_cgs_val], dtype=DTYPE)

    pvapl_Pa = math.exp(54.842763 - 6763.22/T_K - 4.210*math.log(T_K) + 0.000367*T_K)
    h2o_mmr = rh * pvapl_Pa * 18.0 / (29.0 * p_hPa * 100.0)
    gc = jnp.asarray([[h2o_mmr * rhoa_g_cm3, 0.0]], dtype=DTYPE)

    r = np.asarray(cfg.groups[0].r)
    rmass = np.asarray(cfg.groups[0].rmass)
    log_mu_cm = math.log(mu_nm * 1e-7); log_sigma = math.log(sigma_g)
    shape = (np.exp(-0.5 * ((np.log(r) - log_mu_cm) / log_sigma)**2)
              / (r * log_sigma * math.sqrt(2.0 * math.pi)) * rmass)
    norm = shape.sum()
    pc_per_bin = (shape * ((M_ug_m3 * 1e-12) / rhoa_g_cm3 / norm) * rhoa_g_cm3
                   / rmass) if norm > 0 else np.zeros_like(r)
    pc = jnp.zeros((1, cfg.nbin, cfg.nelem), dtype=DTYPE).at[0,:,0].set(
        jnp.asarray(pc_per_bin, dtype=DTYPE))
    t = T

    dgc_per_step = prod_rate * dtime * 98.078479 / float(AVG)
    nstep = len(nsubsteps_per_step)

    for istep in range(nstep):
        gc = gc.at[0, 1].set(gc[0, 1] + dgc_per_step)
        env_s = _refresh_env(t, p_cgs, gc, cfg, ppm)
        pc, gc, t, pcl, gcl, d_gc, d_t, pconmax = prestep(
            pc, gc, t, pc, gc, t, env_s["zmet"],
            itype_arr, ienconc_arr, igelem_arr, rmass_2d,
            do_substep=True, do_coag=cfg.do_coag,
        )
        # Use Fortran's exact substep count for this outer step
        nts = int(nsubsteps_per_step[istep])
        pc, gc, t, _ = step(
            pc=pc, gc=gc, t=t, dtime=float(dtime),
            rhoa=env_s["rhoa"], zmet=env_s["zmet"],
            akelvin=env_s["akelvin"], akelvini=env_s["akelvini"],
            gro=env_s["gro"], gro1=env_s["gro1"],
            rup_wet=env_s["rup_wet"],
            rlhe=env_s["rlhe"], rlhm=env_s["rlhm"],
            ckernel=env_s["ckernel"], pconmax=pconmax,
            ds_threshold_arr=env_s["ds_threshold_arr"],
            pcl=pcl, gcl=gcl, told=t, d_gc=d_gc, d_t=d_t,
            prescribed_ntsubsteps=nts,
        )

    T_final = float(t[0])
    gc_final = float(gc[0, 1]) / rhoa_g_cm3
    pc_final = np.asarray(pc[0, :, 0]) * rmass / rhoa_g_cm3
    return (idx, dict(T_final=T_final, gc_h2so4_final=gc_final,
                       pc_final=pc_final.tolist(), nstep_ran=nstep))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenarios", type=Path,
                   default=ROOT / "scenarios" / "realistic_scenarios_100.npz")
    p.add_argument("--schedules", type=Path,
                   default=ROOT / "outputs" / "prescribed"
                            / "fortran_outputs_schedules.npz")
    p.add_argument("--out", type=Path,
                   default=ROOT / "outputs" / "prescribed" / "jax_outputs.npz")
    p.add_argument("--dtime", type=float, default=60.0)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    scens = np.load(args.scenarios)
    sched = np.load(args.schedules)
    n = int(scens["_n"])
    nsub_history = sched["nsubsteps_history"]   # shape (N, nstep)
    nstep = nsub_history.shape[1]
    print(f"Loaded schedules: ({n}, {nstep}). "
          f"min nts={nsub_history.min()}, max nts={nsub_history.max()}, "
          f"mean nts={nsub_history.mean():.1f}")

    work = []
    for i in range(n):
        scen_dict = {k: float(scens[k][i]) for k in
                     ["T", "p", "rh", "h2so4_prod_rate", "M_total_ug_m3",
                      "aerosol_mu_nm", "aerosol_sigma_g"]}
        work.append((i, scen_dict, args.dtime, nsub_history[i]))

    print(f"Running JAX with prescribed substeps: {n} scenarios, "
          f"{args.workers} workers", flush=True)
    t_start = time.perf_counter()

    nbin = 38
    T_final = np.zeros(n); gc_final = np.zeros(n)
    pc_final = np.zeros((n, nbin))
    nstep_ran = np.full(n, nstep, dtype=np.int64)

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.workers) as pool:
        done = 0
        for idx, out in pool.imap_unordered(_worker_run, work, chunksize=1):
            T_final[idx] = out["T_final"]
            gc_final[idx] = out["gc_h2so4_final"]
            pc_final[idx] = out["pc_final"]
            done += 1
            elapsed = time.perf_counter() - t_start
            if done % 10 == 0 or done == n:
                print(f"  [{done:3d}/{n}] done  "
                      f"({elapsed/done:.1f} s/scen avg, {elapsed:.0f}s)",
                      flush=True)

    elapsed = time.perf_counter() - t_start
    print(f"\nJAX wall time: {elapsed:.0f} s ({elapsed/n*1000:.0f} ms/scen)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        T_final=T_final, gc_h2so4_final=gc_final, pc_final=pc_final,
        nstep_ran=nstep_ran,
        wall_time_s=np.array([elapsed]),
        n_scenarios=np.array([n]),
    )
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
