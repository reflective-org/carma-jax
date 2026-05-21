"""Combined-physics isolation tests across 3 solvers.

Drives the same 3 atmospheres × 5 dts × 3 solvers sweep used by the
single-process Tests A/B/C, but with two or three processes turned
on. The point: see whether coupling between processes amplifies the
per-process discrepancies we already documented, or whether the
combined-physics regime stays as clean as the isolated tests.

Test modes:

  - AB   (coag + condensation): do_coag=1, do_grow=1, do_nuc=0
  - BC   (condensation + nucleation): do_coag=0, do_grow=1, do_nuc=1
  - ABC  (all three): do_coag=1, do_grow=1, do_nuc=1

A+C alone (coag + nucleation without growth) isn't included because
nucleated particles never leave bin 0 without growth — would be a
degenerate test.

Common params (match Tests B/C):
  M_total = 2 µg/m³, GMD = 20 nm, GSD = 1.2
  Fixed [H2SO4] = 1e7 molec/cm³ (reset each outer step in JAX paths;
  passed as argv[7] to Fortran).

Usage:
  python 38_isolation_combined.py --mode AB
  python 38_isolation_combined.py --mode BC
  python 38_isolation_combined.py --mode ABC

Output: benchmark_final/outputs/iso_test_<MODE>/<atm>_dt<dt>/all_solvers.npz
"""
from __future__ import annotations

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
sys.path.insert(0, str(REPO / "scripts"))


# ---- Common parameters ----
ATMOSPHERES = {
    "strat39": dict(scen_idx=39),
    "trop96":  dict(scen_idx=96),
}
DT_VALUES = [1.0, 10.0, 60.0, 300.0, 1800.0]
NSTEP = 48
NBIN = 38
NGAS = 2
FIXED_H2SO4_MOLEC_CM3 = 1.0e7
SEED_GMD_NM = 20.0
SEED_GSD = 1.2
SEED_M_UG_M3 = 2.0

MODES = {
    "AB":  dict(do_coag=True,  do_grow=True, do_nuc=False),
    "BC":  dict(do_coag=False, do_grow=True, do_nuc=True),
    "ABC": dict(do_coag=True,  do_grow=True, do_nuc=True),
}

SCENARIOS = ROOT / "scenarios" / "realistic_scenarios_100.npz"
FORTRAN_BIN = REPO.parent / "original-carma" / "CARMA" / "build" / "test_sulfate_realistic"


def _scen_info(idx):
    S = np.load(SCENARIOS)
    out = {k: float(S[k][idx]) for k in
            ["T", "p", "rh", "h2so4_prod_rate", "M_total_ug_m3",
             "aerosol_mu_nm", "aerosol_sigma_g"]}
    out["aerosol_mu_nm"] = SEED_GMD_NM
    out["aerosol_sigma_g"] = SEED_GSD
    out["M_total_ug_m3"] = SEED_M_UG_M3
    return out


def _scenario_line(s):
    # prod_rate forced to 0; fixed [H2SO4] supplies everything.
    return (f"{s['T']!r} {s['p']!r} {s['rh']!r} "
             f"{0.0!r} {s['M_total_ug_m3']!r} "
             f"{s['aerosol_mu_nm']!r} {s['aerosol_sigma_g']!r}\n")


def run_fortran(s, dtime, nstep, flags):
    """Fortran with arbitrary (do_coag, do_grow, do_nuc) toggles."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        scen_path = td / "scen.txt"
        scen_path.write_text(_scenario_line(s))
        out_path = td / "out.json"
        hist_path = td / "out.json.history.bin"
        env = os.environ.copy()
        cmd = [str(FORTRAN_BIN), str(scen_path), str(out_path),
               "1" if flags["do_coag"] else "0",
               "1" if flags["do_grow"] else "0",
               str(dtime), str(nstep),
               str(FIXED_H2SO4_MOLEC_CM3),
               "1" if flags["do_nuc"] else "0",
               ]
        t0 = time.perf_counter()
        result = subprocess.run(cmd, capture_output=True, text=True,
                                 env=env, timeout=900)
        wall = time.perf_counter() - t0
        if result.returncode != 0:
            raise RuntimeError(f"Fortran failed: {result.stderr[:500]}")
        raw = np.fromfile(hist_path, dtype=np.float64)
        i = 0
        t_hist = raw[i:i+nstep]; i += nstep
        gc_hist = raw[i:i+nstep*NGAS].reshape((NGAS, nstep)).T
        i += nstep * NGAS
        pc_hist = raw[i:i+nstep*NBIN].reshape((NBIN, nstep)).T
        sched_path = Path(str(out_path) + ".schedule.bin")
        sched_raw = np.fromfile(sched_path, dtype=np.int64)
        nh = sched_raw.size // 2
        nsubsteps_cum = sched_raw[:nh]
        nretries_cum = sched_raw[nh:]
        first = nsubsteps_cum[:1]
        nsubs = np.concatenate([first, np.diff(nsubsteps_cum)])
        first_r = nretries_cum[:1]
        nret = np.concatenate([first_r, np.diff(nretries_cum)])
    return dict(t=t_hist, gc=gc_hist, pc=pc_hist,
                 nsubsteps=nsubs, nretries=nret, wall=wall)


def _build_initial(s, cfg):
    import jax.numpy as jnp
    from carma.constants import R_AIR, RPA2CGS
    from carma.precision import DTYPE
    T_K = s["T"]; p_hPa = s["p"]
    p_cgs_val = p_hPa * 100.0 * float(RPA2CGS)
    rhoa = p_cgs_val / (float(R_AIR) * T_K)
    grp = cfg.groups[0]
    r = np.asarray(grp.r)
    rmass_np = np.asarray(grp.rmass)
    log_mu_cm = math.log(s["aerosol_mu_nm"] * 1e-7)
    log_sigma = math.log(s["aerosol_sigma_g"])
    pdf = (np.exp(-0.5 * ((np.log(r) - log_mu_cm) / log_sigma) ** 2)
            / (r * log_sigma * math.sqrt(2.0 * math.pi)) * rmass_np)
    M_target_mmr = (s["M_total_ug_m3"] * 1e-12) / rhoa
    mmr_per_bin = pdf * (M_target_mmr / pdf.sum())
    pc_per_bin = mmr_per_bin * rhoa / rmass_np
    pvapl_Pa = math.exp(54.842763 - 6763.22 / T_K
                         - 4.210 * math.log(T_K) + 0.000367 * T_K)
    h2o_mmr = s["rh"] * pvapl_Pa * 18.0 / (29.0 * p_hPa * 100.0)
    gc_h2so4_target = FIXED_H2SO4_MOLEC_CM3 * 98.078479 / 6.02214e23
    return pc_per_bin, h2o_mmr, gc_h2so4_target, rhoa, p_cgs_val, T_K


def run_faithful_jax(s, dtime, nstep, flags):
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax_ensemble import _minimal_config, _compute_ppm_coefs, _refresh_env
    from carma.precision import DTYPE
    from carma.step_full_faithful import make_step_full_faithful
    from carma.prestep import prestep

    cfg = _minimal_config()._replace(
        do_coag=flags["do_coag"], do_grow=flags["do_grow"],
    )
    ppm = _compute_ppm_coefs(cfg)
    pc_per_bin, h2o_mmr, gc_h2so4_target, rhoa, p_cgs_val, T_K = _build_initial(s, cfg)

    pc = jnp.asarray(pc_per_bin, dtype=DTYPE)[None, :, None]
    gc = jnp.asarray([[h2o_mmr * rhoa, gc_h2so4_target]], dtype=DTYPE)
    T = jnp.asarray([T_K], dtype=DTYPE)
    p_cgs = jnp.asarray([p_cgs_val], dtype=DTYPE)

    step = make_step_full_faithful(
        cfg, ppm_coefs=ppm, do_homogeneous=flags["do_nuc"],
    )
    itype_arr = jnp.asarray([e.itype for e in cfg.elements])
    ienconc_arr = jnp.asarray([g.ienconc for g in cfg.groups])
    igelem_arr = jnp.asarray([e.igroup for e in cfg.elements])
    rmass_2d = jnp.stack(
        [jnp.asarray(g.rmass, dtype=DTYPE) for g in cfg.groups], axis=1,
    )

    t_hist = np.zeros(nstep)
    gc_hist = np.zeros((nstep, NGAS))
    pc_hist = np.zeros((nstep, NBIN))
    t0_wall = time.perf_counter()
    for istep in range(nstep):
        gc = gc.at[0, 1].set(gc_h2so4_target)
        env_s = _refresh_env(T, p_cgs, gc, cfg, ppm)
        pc_p, gc_p, T_p, pcl, gcl, d_gc, d_t, pconmax = prestep(
            pc, gc, T, pc, gc, T, env_s["zmet"],
            itype_arr, ienconc_arr, igelem_arr, rmass_2d,
            do_substep=True, do_coag=flags["do_coag"],
        )
        pc, gc, T, _diag = step(
            pc=pc_p, gc=gc_p, t=T_p, dtime=float(dtime),
            rhoa=env_s["rhoa"], zmet=env_s["zmet"],
            akelvin=env_s["akelvin"], akelvini=env_s["akelvini"],
            gro=env_s["gro"], gro1=env_s["gro1"],
            rup_wet=env_s["rup_wet"],
            rlhe=env_s["rlhe"], rlhm=env_s["rlhm"],
            ckernel=env_s["ckernel"], pconmax=pconmax,
            ds_threshold_arr=env_s["ds_threshold_arr"],
            pcl=pcl, gcl=gcl, told=T_p, d_gc=d_gc, d_t=d_t,
        )
        t_hist[istep] = (istep + 1) * dtime
        gc_hist[istep] = np.asarray(gc[0])
        pc_hist[istep] = np.asarray(pc[0, :, 0])
    pc.block_until_ready()
    return dict(t=t_hist, gc=gc_hist, pc=pc_hist,
                 wall=time.perf_counter() - t0_wall)


def run_diffrax(s, dtime, nstep, flags):
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax_ensemble import _minimal_config, _compute_ppm_coefs, _refresh_env
    from carma.precision import DTYPE
    from carma_diffrax.config import DiffraxConfig
    from carma_diffrax.rhs import FrozenEnv, make_rhs
    from carma_diffrax.state import StateShape, pack, unpack
    from carma_diffrax.coag_step import (
        CoagBundle, apply_coag, build_microslow_for_sulfate,
    )
    import diffrax as diffrax_mod
    import optimistix as optx

    cfg = _minimal_config()._replace(
        do_coag=flags["do_coag"], do_grow=flags["do_grow"],
    )
    ppm = _compute_ppm_coefs(cfg)
    microslow = (build_microslow_for_sulfate(cfg)
                  if flags["do_coag"] else None)
    pc_per_bin, h2o_mmr, gc_h2so4_target, rhoa, p_cgs_val, T_K = _build_initial(s, cfg)

    pc = jnp.asarray(pc_per_bin, dtype=DTYPE)[:, None]
    gc = jnp.asarray([h2o_mmr * rhoa, gc_h2so4_target], dtype=DTYPE)
    T_scalar = jnp.asarray(T_K, dtype=DTYPE)
    p_cgs = jnp.asarray([p_cgs_val], dtype=DTYPE)

    grp = cfg.groups[0]
    rmass_np = np.asarray(grp.rmass)
    rmassup_np = np.asarray(grp.rmassup)
    rmasslow_np = np.concatenate(
        [[rmass_np[0] / (grp.rmrat ** 0.5)], rmassup_np[:-1]]
    )
    dm = jnp.asarray(rmassup_np - rmasslow_np, dtype=DTYPE)
    shape = StateShape(nbin=cfg.nbin, nelem=1, ngas=NGAS)
    cfg_d = DiffraxConfig(rtol=1e-5, atol=1e-5, max_steps=20_000)

    term = diffrax_mod.ODETerm(
        make_rhs(shape,
                  do_growth=flags["do_grow"],
                  do_homogeneous=flags["do_nuc"])
    )
    solver = diffrax_mod.Kvaerno5(
        root_finder=optx.Chord(rtol=cfg_d.rtol, atol=cfg_d.atol),
    )
    controller = diffrax_mod.PIDController(
        rtol=cfg_d.rtol, atol=cfg_d.atol,
        pcoeff=0.3, icoeff=0.3, dcoeff=0.0,
        factormin=0.5, factormax=5.0, safety=0.8,
    )

    @jax.jit
    def step_inner(pc0, gc0, T0, dt_arg, env):
        y0 = pack(pc0, gc0, T0)
        sol = diffrax_mod.diffeqsolve(
            term, solver,
            t0=0.0, t1=dt_arg, dt0=None, y0=y0,
            args=env,
            stepsize_controller=controller,
            max_steps=20_000,
            saveat=diffrax_mod.SaveAt(t1=True),
        )
        pc_e, gc_e, T_e = unpack(sol.ys[-1], shape)
        return pc_e, gc_e, T_e, sol.stats, sol.result

    t_hist = np.zeros(nstep)
    gc_hist = np.zeros((nstep, NGAS))
    pc_hist = np.zeros((nstep, NBIN))
    n_acc = 0; n_rej = 0
    t0_wall = time.perf_counter()
    for istep in range(nstep):
        gc = gc.at[1].set(gc_h2so4_target)
        env_d = _refresh_env(
            jnp.atleast_1d(T_scalar), p_cgs, gc[None, :], cfg, ppm,
        )
        # Operator-split coag pre-step.
        if microslow is not None:
            coag = CoagBundle(
                microslow_jit=microslow,
                ckernel=env_d["ckernel"],
                zmet=env_d["zmet"],
            )
            pc = apply_coag(pc, coag, dtime=dtime)
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
            pratt=env_d["pratt"][..., 0], prat=env_d["prat"][..., 0],
            pden1=env_d["pden1"][..., 0], palr=env_d["palr"][..., 0],
        )
        pc, gc, T_scalar, stats, _result = step_inner(
            pc, gc, T_scalar, DTYPE(dtime), env,
        )
        n_acc += int(stats["num_accepted_steps"])
        n_rej += int(stats["num_rejected_steps"])
        t_hist[istep] = (istep + 1) * dtime
        gc_hist[istep] = np.asarray(gc)
        pc_hist[istep] = np.asarray(pc[:, 0])
    pc.block_until_ready()
    return dict(t=t_hist, gc=gc_hist, pc=pc_hist,
                 n_accepted=n_acc, n_rejected=n_rej,
                 wall=time.perf_counter() - t0_wall)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=list(MODES.keys()), required=True)
    p.add_argument("--atm", nargs="*", default=list(ATMOSPHERES.keys()),
                    choices=list(ATMOSPHERES.keys()))
    p.add_argument("--dt", nargs="*", type=float, default=DT_VALUES)
    p.add_argument("--skip", nargs="*", default=[],
                    choices=["fortran", "faithful", "diffrax"])
    args = p.parse_args()
    flags = MODES[args.mode]
    out_root = ROOT / "outputs" / f"iso_test_{args.mode}"
    out_root.mkdir(parents=True, exist_ok=True)

    for atm_key in args.atm:
        scen_idx = ATMOSPHERES[atm_key]["scen_idx"]
        s = _scen_info(scen_idx)
        print(f"\n=== Test {args.mode} ({flags}) — {atm_key}: scen {scen_idx} ===")
        for k, v in s.items():
            print(f"  {k}: {v}")
        for dt in args.dt:
            print(f"\n--- dt = {dt} s × {NSTEP} steps ---")
            out_dir = out_root / f"{atm_key}_dt{int(dt)}"
            out_dir.mkdir(parents=True, exist_ok=True)
            bundle = {"scenario_info": np.array(json.dumps(
                dict(s, dt=dt, nstep=NSTEP, atm=atm_key,
                      fixed_h2so4_molec_cm3=FIXED_H2SO4_MOLEC_CM3,
                      test=args.mode, flags=flags),
            ))}
            if "fortran" not in args.skip:
                print("  Fortran...")
                F = run_fortran(s, dt, NSTEP, flags)
                print(f"    wall = {F['wall']:.1f}s, "
                      f"nsubsteps {F['nsubsteps'].min()}-{F['nsubsteps'].max()}, "
                      f"retries {F['nretries'].max()}")
                for k, v in F.items():
                    bundle[f"fortran_{k}"] = np.asarray(v)
            if "faithful" not in args.skip:
                print("  Faithful JAX...")
                J = run_faithful_jax(s, dt, NSTEP, flags)
                print(f"    wall = {J['wall']:.1f}s")
                for k, v in J.items():
                    bundle[f"faithful_{k}"] = np.asarray(v)
            if "diffrax" not in args.skip:
                print("  Diffrax...")
                D = run_diffrax(s, dt, NSTEP, flags)
                print(f"    wall = {D['wall']:.1f}s, "
                      f"accepted={D['n_accepted']}, rejected={D['n_rejected']}")
                for k, v in D.items():
                    bundle[f"diffrax_{k}"] = np.asarray(v)
            out_path = out_dir / "all_solvers.npz"
            np.savez_compressed(out_path, **bundle)
            print(f"  Saved {out_path}")


if __name__ == "__main__":
    main()
