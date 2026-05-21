"""Run scenario 21 through Fortran, faithful JAX, and diffrax with full time-series.

Scenario 21 is one of the 7 ceiling-hit scenarios at 1800 s in Fortran:
  T=215.6 K, p=27.0 hPa (lower stratosphere)
  RH=0.51, prod_rate=9.6e6 molec/cm^3/s
  M=2.54 ug/m^3 initial aerosol, mu=681 nm, sigma_g=1.46

Saves per-step trajectories of pc(nstep, nbin), gc(nstep, ngas), T(nstep)
for each solver into a single npz, plus a scenario_info dict.

Output: benchmark_final/outputs/dt1800/scen21_3way/all_solvers.npz
"""
import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
SCEN_IDX = 21
DTIME = 1800.0
NSTEP = 48
NBIN = 38
NGAS = 2

OUT_DIR = ROOT / "outputs" / "dt1800" / "scen21_3way"
SCENARIOS = ROOT / "scenarios" / "realistic_scenarios_100.npz"
FORTRAN_BIN = REPO.parent / "original-carma" / "CARMA" / "build" / "test_sulfate_realistic"


def _scen_info():
    S = np.load(SCENARIOS)
    return {k: float(S[k][SCEN_IDX]) for k in
             ["T", "p", "rh", "h2so4_prod_rate", "M_total_ug_m3",
              "aerosol_mu_nm", "aerosol_sigma_g"]}


def _scenario_line(s):
    return (f"{s['T']!r} {s['p']!r} {s['rh']!r} "
             f"{s['h2so4_prod_rate']!r} {s['M_total_ug_m3']!r} "
             f"{s['aerosol_mu_nm']!r} {s['aerosol_sigma_g']!r}\n")


def run_fortran(s):
    """Call the patched Fortran binary and parse its history.bin output."""
    if not FORTRAN_BIN.exists():
        raise FileNotFoundError(
            f"Fortran binary missing at {FORTRAN_BIN}; "
            "run benchmark_final/fortran/build_realistic.sh first."
        )
    with tempfile.TemporaryDirectory(prefix="scen21_fort_") as d:
        d = Path(d)
        scen_path = d / "scen.txt"
        out_path = d / "out.json"
        hist_path = Path(str(out_path) + ".history.bin")
        scen_path.write_text(_scenario_line(s))
        cmd = [str(FORTRAN_BIN), str(scen_path), str(out_path),
                "1", "1", str(DTIME), str(NSTEP)]
        t0 = time.perf_counter()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        wall = time.perf_counter() - t0
        if result.returncode != 0:
            raise RuntimeError(f"Fortran failed: {result.stderr[:500]}")
        # Parse history: NSTEP doubles for t, NGAS*NSTEP for gc, NBIN*NSTEP for pc
        raw = np.fromfile(hist_path, dtype=np.float64)
        i = 0
        t_hist = raw[i:i+NSTEP]; i += NSTEP
        # Fortran column-major: gc_history(nstep, NGAS) → reshape Fortran order
        gc_hist = raw[i:i+NSTEP*NGAS].reshape((NGAS, NSTEP)).T
        i += NSTEP * NGAS
        pc_hist = raw[i:i+NSTEP*NBIN].reshape((NBIN, NSTEP)).T
        # Final state (also in JSON for cross-check)
        meta = json.loads(out_path.read_text())
        # Substep schedule
        sched_path = Path(str(out_path) + ".schedule.bin")
        sched_raw = np.fromfile(sched_path, dtype=np.int64)
        nh = sched_raw.size // 2
        nsubsteps_cum = sched_raw[:nh]
        nretries_cum = sched_raw[nh:]
        # per-step counts (the cumulative-counter bug fix from PR #79)
        first = nsubsteps_cum[:1]
        nsubsteps_per_step = np.concatenate([first, np.diff(nsubsteps_cum)])
        first_r = nretries_cum[:1]
        nretries_per_step = np.concatenate([first_r, np.diff(nretries_cum)])
    return dict(
        t=t_hist, gc=gc_hist, pc=pc_hist,
        nsubsteps=nsubsteps_per_step, nretries=nretries_per_step,
        wall=wall, meta=meta,
    )


def _initial_state_jax(s, cfg, ppm, _refresh_env):
    """Build initial (pc, gc, T) JAX state matching the Fortran setup."""
    import jax.numpy as jnp
    from carma.constants import RPA2CGS, R_AIR
    from carma.precision import DTYPE

    T_K, p_hPa, rh = s["T"], s["p"], s["rh"]
    M_ug_m3, mu_nm, sigma_g = s["M_total_ug_m3"], s["aerosol_mu_nm"], s["aerosol_sigma_g"]

    p_cgs_val = p_hPa * 100.0 * float(RPA2CGS)
    rhoa = p_cgs_val / (float(R_AIR) * T_K)

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

    p_cgs = jnp.asarray([p_cgs_val], dtype=DTYPE)
    T = jnp.asarray([T_K], dtype=DTYPE)
    return pc_per_bin, gc, p_cgs, T, rhoa


def run_faithful_jax(s):
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    sys.path.insert(0, str(REPO / "scripts"))
    from jax_ensemble import _minimal_config, _compute_ppm_coefs, _refresh_env
    from carma.constants import AVG
    from carma.precision import DTYPE
    from carma.step_full_faithful import make_step_full_faithful
    from carma.prestep import prestep

    cfg = _minimal_config()
    cfg = cfg._replace(do_coag=True, do_grow=True)
    ppm = _compute_ppm_coefs(cfg)
    step = make_step_full_faithful(cfg, ppm_coefs=ppm)
    itype_arr = jnp.asarray([e.itype for e in cfg.elements])
    ienconc_arr = jnp.asarray([g.ienconc for g in cfg.groups])
    igelem_arr = jnp.asarray([e.igroup for e in cfg.elements])
    rmass_2d = jnp.stack(
        [jnp.asarray(g.rmass, dtype=DTYPE) for g in cfg.groups], axis=1,
    )

    pc_per_bin, gc, p_cgs, T, rhoa = _initial_state_jax(s, cfg, ppm, _refresh_env)
    pc = jnp.zeros((1, cfg.nbin, cfg.nelem), dtype=DTYPE)
    pc = pc.at[0, :, 0].set(jnp.asarray(pc_per_bin, dtype=DTYPE))
    t = T
    dgc_per_step = s["h2so4_prod_rate"] * DTIME * 98.078479 / float(AVG)

    pc_hist = np.zeros((NSTEP, NBIN))
    gc_hist = np.zeros((NSTEP, NGAS))
    T_hist = np.zeros(NSTEP)

    t_start = time.perf_counter()
    for istep in range(NSTEP):
        gc = gc.at[0, 1].set(gc[0, 1] + dgc_per_step)
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
        gc_hist[istep, 0] = float(gc[0, 0])
        gc_hist[istep, 1] = float(gc[0, 1])
        T_hist[istep] = float(t[0])
    wall = time.perf_counter() - t_start
    return dict(t=T_hist, gc=gc_hist, pc=pc_hist, wall=wall)


def run_diffrax(s):
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    sys.path.insert(0, str(REPO / "scripts"))
    from jax_ensemble import _minimal_config, _compute_ppm_coefs, _refresh_env
    from carma.constants import AVG
    from carma.precision import DTYPE
    from carma_diffrax import DiffraxConfig, diffrax_step
    from carma_diffrax.rhs import FrozenEnv
    from carma_diffrax.state import StateShape

    cfg = _minimal_config()
    ppm = _compute_ppm_coefs(cfg)
    shape = StateShape(nbin=cfg.nbin, nelem=1, ngas=2)
    grp = cfg.groups[0]
    rmass_np = np.asarray(grp.rmass)
    rmassup_np = np.asarray(grp.rmassup)
    rmasslow_np = np.concatenate([[rmass_np[0]/(grp.rmrat**0.5)], rmassup_np[:-1]])
    dm = jnp.asarray(rmassup_np - rmasslow_np, dtype=DTYPE)

    pc_per_bin, gc, p_cgs, T, rhoa = _initial_state_jax(s, cfg, ppm, _refresh_env)
    pc = jnp.asarray(pc_per_bin, dtype=DTYPE)[:, None]
    T_scalar = T[0]
    dgc_per_step = s["h2so4_prod_rate"] * DTIME * 98.078479 / float(AVG)
    cfg_d = DiffraxConfig(rtol=1e-5, atol=1e-5, max_steps=20_000)

    pc_hist = np.zeros((NSTEP, NBIN))
    gc_hist = np.zeros((NSTEP, NGAS))
    T_hist = np.zeros(NSTEP)
    n_accepted_hist = np.zeros(NSTEP, dtype=np.int64)
    n_rejected_hist = np.zeros(NSTEP, dtype=np.int64)

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
            pratt=env_d["pratt"][..., 0],
            prat=env_d["prat"][..., 0],
            pden1=env_d["pden1"][..., 0],
            palr=env_d["palr"][..., 0],
        )
        pc, gc_1d, T_scalar, stats = diffrax_step(
            pc, gc[0], T_scalar, DTIME, env, shape, cfg_d,
        )
        gc = gc_1d[None, :]
        pc_hist[istep] = np.asarray(pc[:, 0])
        gc_hist[istep, 0] = float(gc[0, 0])
        gc_hist[istep, 1] = float(gc[0, 1])
        T_hist[istep] = float(T_scalar)
        n_accepted_hist[istep] = stats["num_accepted_steps"]
        n_rejected_hist[istep] = stats["num_rejected_steps"]
    wall = time.perf_counter() - t_start
    return dict(t=T_hist, gc=gc_hist, pc=pc_hist, wall=wall,
                 n_accepted=n_accepted_hist, n_rejected=n_rejected_hist)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip", nargs="*", default=[],
                    choices=["fortran", "faithful", "diffrax"],
                    help="solvers to skip (useful when iterating)")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    s = _scen_info()
    print(f"=== Scenario {SCEN_IDX} ===")
    for k, v in s.items():
        print(f"  {k}: {v}")
    print()

    bundle = {}
    if "fortran" not in args.skip:
        print("Running Fortran...")
        F = run_fortran(s)
        print(f"  Fortran wall: {F['wall']:.1f}s "
               f"(per-step nsubsteps: min={F['nsubsteps'].min()}, "
               f"max={F['nsubsteps'].max()}, sum={F['nsubsteps'].sum()})")
        bundle["fortran"] = F
    if "faithful" not in args.skip:
        print("Running faithful JAX...")
        J = run_faithful_jax(s)
        print(f"  Faithful wall: {J['wall']:.1f}s")
        bundle["faithful"] = J
    if "diffrax" not in args.skip:
        print("Running diffrax...")
        D = run_diffrax(s)
        print(f"  Diffrax wall: {D['wall']:.1f}s "
               f"(internal steps: accepted={D['n_accepted'].sum()}, "
               f"rejected={D['n_rejected'].sum()})")
        bundle["diffrax"] = D

    out_path = OUT_DIR / "all_solvers.npz"
    save_args = {"scenario_info": np.array(json.dumps(s))}
    for solver, d in bundle.items():
        for k, v in d.items():
            if k == "meta":
                save_args[f"{solver}_meta"] = np.array(json.dumps(v))
            else:
                save_args[f"{solver}_{k}"] = np.asarray(v)
    np.savez_compressed(out_path, **save_args)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
