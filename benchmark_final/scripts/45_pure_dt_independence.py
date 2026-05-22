"""Pure diffrax dt-independence test using fix_h2so4 RHS flag.

Compares diffrax results across different outer-loop dt values when
[H2SO4] is held fixed *inside the RHS* (via the new fix_h2so4 flag),
removing the outer-step gas-reset confound from earlier tests.

Test setup:
  ABC physics (coag operator-split + condensation + nucleation),
  strat39 atmosphere, M_total=2 µg/m³ seed, GMD=20 nm, GSD=1.2,
  fixed [H2SO4]=1e7 #/cm³ enforced as an RHS constraint.

Three configurations, all targeting 6h physical:
  A. Outer loop dt=10s × 2160 steps    (many outer steps)
  B. Outer loop dt=1800s × 12 steps    (few outer steps)
  C. Single diffeqsolve, t1=21600s     (no outer loop)

If diffrax adaptive substepping is solid, A, B, and C should agree
within solver tolerance regardless of outer dt. Only coag's
operator-split semantics will differ (more coag-step calls = more
fine-grained coag dynamics).

Output: benchmark_final/outputs/iso_pure_dt_indep/<tag>/all_solvers.npz
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "scripts"))


def run(outer_dt_s, nstep, tag, do_coag=True):
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax_ensemble import _minimal_config, _compute_ppm_coefs, _refresh_env
    from carma.constants import R_AIR, RPA2CGS, AVG
    from carma.precision import DTYPE
    from carma_diffrax.config import DiffraxConfig
    from carma_diffrax.rhs import FrozenEnv, make_rhs
    from carma_diffrax.state import StateShape, pack, unpack
    from carma_diffrax.coag_step import (
        CoagBundle, apply_coag, build_microslow_for_sulfate,
    )
    import diffrax as diffrax_mod
    import optimistix as optx

    cfg = _minimal_config()._replace(do_coag=do_coag, do_grow=True)
    ppm = _compute_ppm_coefs(cfg)
    microslow = build_microslow_for_sulfate(cfg) if do_coag else None

    # Scenario: strat39 + common params
    T_K = 217.116948
    p_hPa = 52.107758
    rh = 0.197017
    M_ug_m3 = 2.0
    GMD = 20.0
    GSD = 1.2
    FIXED_H2SO4 = 1.0e7

    p_cgs_val = p_hPa * 100.0 * float(RPA2CGS)
    rhoa = p_cgs_val / (float(R_AIR) * T_K)
    grp = cfg.groups[0]
    r = np.asarray(grp.r)
    rmass_np = np.asarray(grp.rmass)
    log_mu = math.log(GMD * 1e-7)
    log_s = math.log(GSD)
    pdf = (np.exp(-0.5 * ((np.log(r) - log_mu) / log_s) ** 2)
            / (r * log_s * math.sqrt(2 * math.pi)) * rmass_np)
    M_target = (M_ug_m3 * 1e-12) / rhoa
    pc_init = pdf * (M_target / pdf.sum()) * rhoa / rmass_np
    gc_h2so4_target = FIXED_H2SO4 * 98.078479 / float(AVG)

    pvapl_Pa = math.exp(54.842763 - 6763.22 / T_K
                         - 4.210 * math.log(T_K) + 0.000367 * T_K)
    h2o_mmr = rh * pvapl_Pa * 18.0 / (29.0 * p_hPa * 100.0)
    pc = jnp.asarray(pc_init, dtype=DTYPE)[:, None]
    gc = jnp.asarray([h2o_mmr * rhoa, gc_h2so4_target], dtype=DTYPE)
    T_scalar = jnp.asarray(T_K, dtype=DTYPE)
    p_cgs = jnp.asarray([p_cgs_val], dtype=DTYPE)

    rmassup_np = np.asarray(grp.rmassup)
    rmasslow_np = np.concatenate(
        [[rmass_np[0] / (grp.rmrat ** 0.5)], rmassup_np[:-1]]
    )
    dm = jnp.asarray(rmassup_np - rmasslow_np, dtype=DTYPE)
    shape = StateShape(nbin=cfg.nbin, nelem=1, ngas=2)
    cfg_d = DiffraxConfig(rtol=1e-5, atol=1e-5, max_steps=200_000)

    # Build RHS with fix_h2so4=True — no gas reset needed.
    term = diffrax_mod.ODETerm(
        make_rhs(shape, do_growth=True, do_homogeneous=True,
                 fix_h2so4=True)
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
            max_steps=200_000,
            saveat=diffrax_mod.SaveAt(t1=True),
        )
        pc_e, gc_e, T_e = unpack(sol.ys[-1], shape)
        return pc_e, gc_e, T_e, sol.stats

    n_acc_total = 0; n_rej_total = 0
    t0_wall = time.perf_counter()
    for istep in range(nstep):
        env_d = _refresh_env(
            jnp.atleast_1d(T_scalar), p_cgs, gc[None, :], cfg, ppm,
        )
        # Operator-split coag pre-step (only if do_coag).
        if microslow is not None:
            coag = CoagBundle(
                microslow_jit=microslow,
                ckernel=env_d["ckernel"],
                zmet=env_d["zmet"],
            )
            pc = apply_coag(pc, coag, dtime=outer_dt_s)
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
        pc, gc, T_scalar, stats = step_inner(
            pc, gc, T_scalar, DTYPE(outer_dt_s), env,
        )
        n_acc_total += int(stats["num_accepted_steps"])
        n_rej_total += int(stats["num_rejected_steps"])
    pc.block_until_ready()
    wall = time.perf_counter() - t0_wall

    pc_final = np.asarray(pc[:, 0])
    out_dir = ROOT / "outputs" / "iso_pure_dt_indep" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "result.npz",
        pc_final=pc_final,
        gc_final=np.asarray(gc),
        wall=wall, n_accepted=n_acc_total, n_rejected=n_rej_total,
        outer_dt=outer_dt_s, nstep=nstep, total_time=outer_dt_s * nstep,
    )
    return dict(pc=pc_final, gc=np.asarray(gc), wall=wall,
                 n_acc=n_acc_total, n_rej=n_rej_total)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--no-coag", action="store_true",
                    help="disable coag — true dt-independence test")
    args = p.parse_args()
    do_coag = not args.no_coag
    configs = [
        (10.0,    2160, "outer_dt10"),
        (60.0,     360, "outer_dt60"),
        (300.0,     72, "outer_dt300"),
        (1800.0,    12, "outer_dt1800"),
        (21600.0,    1, "outer_single"),
    ]
    print(f"Pure diffrax dt-independence test (fix_h2so4=True, "
          f"do_coag={do_coag})")
    print("All target 6 h = 21600 s physical, strat39.\n")
    results = {}
    suffix = "" if do_coag else "_nocoag"
    for dt, nstep, tag in configs:
        full_tag = tag + suffix
        print(f"=== {full_tag}: dt={dt}s × {nstep} ===")
        r = run(dt, nstep, full_tag, do_coag=do_coag)
        print(f"  wall = {r['wall']:.1f}s, "
              f"diffrax accepted={r['n_acc']}, rejected={r['n_rej']}")
        results[full_tag] = r

    # Compare totals
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    from jax_ensemble import _minimal_config
    cfg = _minimal_config()
    rmass = np.asarray(cfg.groups[0].rmass)
    print(f"\n{'tag':>15} {'totN':>13} {'totM':>13} {'rel N vs single':>16} {'wall':>8}")
    ref_tag = "outer_single" + suffix
    ref_N = results[ref_tag]["pc"].sum()
    ref_M = (results[ref_tag]["pc"] * rmass).sum()
    for _, _, tag in configs:
        full_tag = tag + suffix
        r = results[full_tag]
        N = r["pc"].sum()
        M = (r["pc"] * rmass).sum()
        relN = (N - ref_N) / ref_N * 100
        print(f"{full_tag:>20} {N:>13.3e} {M:>13.3e} {relN:>+15.3f}% {r['wall']:>7.1f}s")


if __name__ == "__main__":
    main()
