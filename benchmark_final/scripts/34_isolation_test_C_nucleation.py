"""Test C — nucleation-only physics isolation across 3 solvers.

Starts from an *empty* aerosol (no seed) and lets fixed [H2SO4]
drive homogeneous nucleation. Growth stays on so freshly nucleated
particles can move across bins (otherwise they'd all stick in bin 0
and the per-bin distribution comparison would be trivial). Coag is
off.

Process toggles per solver:

  - Fortran:  test_sulfate_realistic do_coag=0, do_grow=1, do_nuc=1,
              prod_rate=0 in scenario (no extra H2SO4 injection — the
              fixed-H2SO4 mechanism supplies all of it), fixed_h2so4
              = target value reset every step. M_total=0 in scenario.
  - Faithful: cfg.do_coag=False, cfg.do_grow=True, step's
              do_homogeneous=True. Reset gc[H2SO4] each step.
  - Diffrax:  make_rhs(do_growth=True, do_homogeneous=True), coag=None.
              Reset gc[H2SO4] each step.

Same atmospheres / timesteps / nsteps as Tests A and B.
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


ATMOSPHERES = {
    "strat39": dict(scen_idx=39, label="Stratosphere (scen 39)"),
    "trop96":  dict(scen_idx=96, label="Troposphere (scen 96)"),
}
DT_VALUES = [1.0, 10.0, 60.0, 300.0, 1800.0]
NSTEP = 48
NBIN = 38
NGAS = 2
FIXED_H2SO4_MOLEC_CM3 = 1.0e7   # user-fixed common with Test B.
                                  # Lower than the earlier 1e8 → more
                                  # tractable nucleation rate, lets
                                  # all 3 solvers converge cleanly.
# User-requested: background aerosol of 2 µg/m³ at GMD=20 nm GSD=1.2.
# This makes Test C "fresh nucleation in the presence of a pre-existing
# background", which is a more realistic atmospheric setup than the
# previous pc=0 isolation (pure homogeneous nucleation onto empty air).
TEST_C_GMD_NM = 20.0
TEST_C_GSD = 1.2
TEST_C_M_UG_M3 = 2.0

OUT_ROOT = ROOT / "outputs" / "iso_test_C"
SCENARIOS = ROOT / "scenarios" / "realistic_scenarios_100.npz"
FORTRAN_BIN = REPO.parent / "original-carma" / "CARMA" / "build" / "test_sulfate_realistic"


def _scen_info(idx):
    S = np.load(SCENARIOS)
    return {k: float(S[k][idx]) for k in
             ["T", "p", "rh", "h2so4_prod_rate", "M_total_ug_m3",
              "aerosol_mu_nm", "aerosol_sigma_g"]}


def _scenario_line(s):
    # prod_rate → 0 so the only H2SO4 source is the fixed-H2SO4 reset.
    # M_total → TEST_C_M_UG_M3 so we seed a background aerosol.
    # Aerosol GMD/GSD: TEST_C_GMD_NM / TEST_C_GSD.
    return (f"{s['T']!r} {s['p']!r} {s['rh']!r} "
             f"{0.0!r} {TEST_C_M_UG_M3!r} "
             f"{TEST_C_GMD_NM!r} {TEST_C_GSD!r}\n")


def run_fortran(s, dtime, nstep, out_dir):
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        scen_path = td / "scen.txt"
        scen_path.write_text(_scenario_line(s))
        out_path = td / "out.json"
        hist_path = td / "out.json.history.bin"
        env = os.environ.copy()
        cmd = [str(FORTRAN_BIN), str(scen_path), str(out_path),
               "0",                              # do_coag
               "1",                              # do_grow
               str(dtime), str(nstep),
               str(FIXED_H2SO4_MOLEC_CM3),       # fixed H2SO4
               "1",                              # do_nuc
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
    """Lognormal background aerosol at TEST_C_M_UG_M3 µg/m³, water at
    scenario RH, H2SO4 at the fixed target."""
    import jax.numpy as jnp
    from carma.constants import R_AIR, RPA2CGS
    from carma.precision import DTYPE
    T_K = s["T"]; p_hPa = s["p"]
    p_cgs_val = p_hPa * 100.0 * float(RPA2CGS)
    rhoa = p_cgs_val / (float(R_AIR) * T_K)
    grp = cfg.groups[0]
    r = np.asarray(grp.r)
    rmass_np = np.asarray(grp.rmass)
    log_mu_cm = math.log(TEST_C_GMD_NM * 1e-7)
    log_sigma = math.log(TEST_C_GSD)
    pdf = (np.exp(-0.5 * ((np.log(r) - log_mu_cm) / log_sigma) ** 2)
            / (r * log_sigma * math.sqrt(2.0 * math.pi)) * rmass_np)
    M_target_mmr = (TEST_C_M_UG_M3 * 1e-12) / rhoa
    mmr_per_bin = pdf * (M_target_mmr / pdf.sum())
    pc_per_bin = mmr_per_bin * rhoa / rmass_np
    pvapl_Pa = math.exp(54.842763 - 6763.22 / T_K
                         - 4.210 * math.log(T_K) + 0.000367 * T_K)
    h2o_mmr = s["rh"] * pvapl_Pa * 18.0 / (29.0 * p_hPa * 100.0)
    gc_h2so4_target = FIXED_H2SO4_MOLEC_CM3 * 98.078479 / 6.02214e23
    return pc_per_bin, h2o_mmr, gc_h2so4_target, rhoa, p_cgs_val, T_K


def run_faithful_jax(s, dtime, nstep):
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax_ensemble import _minimal_config, _compute_ppm_coefs, _refresh_env
    from carma.precision import DTYPE
    from carma.step_full_faithful import make_step_full_faithful
    from carma.prestep import prestep

    cfg = _minimal_config()._replace(do_coag=False, do_grow=True)
    ppm = _compute_ppm_coefs(cfg)
    pc_per_bin, h2o_mmr, gc_h2so4_target, rhoa, p_cgs_val, T_K = _build_initial(s, cfg)

    pc = jnp.asarray(pc_per_bin, dtype=DTYPE)[None, :, None]
    gc = jnp.asarray([[h2o_mmr * rhoa, gc_h2so4_target]], dtype=DTYPE)
    T = jnp.asarray([T_K], dtype=DTYPE)
    p_cgs = jnp.asarray([p_cgs_val], dtype=DTYPE)

    step = make_step_full_faithful(cfg, ppm_coefs=ppm, do_homogeneous=True)

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
            do_substep=True, do_coag=False,
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


def run_diffrax(s, dtime, nstep):
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax_ensemble import _minimal_config, _compute_ppm_coefs, _refresh_env
    from carma.precision import DTYPE
    from carma_diffrax.config import DiffraxConfig
    from carma_diffrax.rhs import FrozenEnv, make_rhs
    from carma_diffrax.state import StateShape, pack, unpack
    import diffrax as diffrax_mod
    import optimistix as optx

    cfg = _minimal_config()._replace(do_coag=False, do_grow=True)
    ppm = _compute_ppm_coefs(cfg)
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

    # Nucleation + growth, no coag.
    term = diffrax_mod.ODETerm(
        make_rhs(shape, do_growth=True, do_homogeneous=True)
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
    def step(pc0, gc0, T0, dt_arg, env):
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
        pc, gc, T_scalar, stats, _result = step(
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
    p.add_argument("--atm", nargs="*", default=list(ATMOSPHERES.keys()),
                    choices=list(ATMOSPHERES.keys()))
    p.add_argument("--dt", nargs="*", type=float, default=DT_VALUES)
    p.add_argument("--skip", nargs="*", default=[],
                    choices=["fortran", "faithful", "diffrax"])
    args = p.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for atm_key in args.atm:
        scen_idx = ATMOSPHERES[atm_key]["scen_idx"]
        s = _scen_info(scen_idx)
        print(f"\n=== Test C (nuc + growth) — {atm_key}: scen {scen_idx} ===")
        for k, v in s.items():
            print(f"  {k}: {v}")
        for dt in args.dt:
            print(f"\n--- dt = {dt} s × {NSTEP} steps "
                  f"({dt*NSTEP:.0f} s = {dt*NSTEP/3600:.2f} h) ---")
            out_dir = OUT_ROOT / f"{atm_key}_dt{int(dt)}"
            out_dir.mkdir(parents=True, exist_ok=True)
            bundle = {"scenario_info": np.array(json.dumps(
                dict(s, dt=dt, nstep=NSTEP, atm=atm_key,
                      fixed_h2so4_molec_cm3=FIXED_H2SO4_MOLEC_CM3,
                      test="nuc_only"),
            ))}
            if "fortran" not in args.skip:
                print("  Fortran...")
                F = run_fortran(s, dt, NSTEP, out_dir)
                print(f"    wall = {F['wall']:.1f}s, "
                      f"nsubsteps {F['nsubsteps'].min()}-{F['nsubsteps'].max()}, "
                      f"retries {F['nretries'].max()}")
                for k, v in F.items():
                    bundle[f"fortran_{k}"] = np.asarray(v)
            if "faithful" not in args.skip:
                print("  Faithful JAX...")
                J = run_faithful_jax(s, dt, NSTEP)
                print(f"    wall = {J['wall']:.1f}s")
                for k, v in J.items():
                    bundle[f"faithful_{k}"] = np.asarray(v)
            if "diffrax" not in args.skip:
                print("  Diffrax...")
                D = run_diffrax(s, dt, NSTEP)
                print(f"    wall = {D['wall']:.1f}s, "
                      f"accepted={D['n_accepted']}, rejected={D['n_rejected']}")
                for k, v in D.items():
                    bundle[f"diffrax_{k}"] = np.asarray(v)
            out_path = out_dir / "all_solvers.npz"
            np.savez_compressed(out_path, **bundle)
            print(f"  Saved {out_path}")


if __name__ == "__main__":
    main()
