"""Phase 2B: PID controller gain sweep.

After Phase 2A picks a (solver, tolerance) combo, this script sweeps
the PID gains and `factormin` / `factormax` to reduce the 56% step
rejection rate observed at the baseline.

Baseline default: pcoeff=0, icoeff=1, dcoeff=0, factormin=0.2, factormax=10.
"""
import argparse
import math
import sys
import time
from pathlib import Path

import diffrax
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))

OUT_DIR = ROOT / "outputs" / "dt1800" / "sweep_2B"

SCEN_IDX = 21
DTIME = 1800.0
NSTEP = 48


# Candidate PID gain settings.
# Format: (pcoeff, icoeff, dcoeff, factormin, factormax, safety, label)
PID_CONFIGS = [
    (0.0, 1.0, 0.0, 0.2, 10.0, 0.9, "baseline_pure_I"),
    (0.3, 0.3, 0.0, 0.5, 5.0,  0.8, "PI_balanced"),
    (0.2, 0.5, 0.0, 0.5, 5.0,  0.85, "PI_slightly_aggressive"),
    (0.4, 0.4, 0.0, 0.5, 5.0,  0.8, "PI_strong"),
    (0.3, 0.3, 0.0, 0.5, 3.0,  0.8, "PI_balanced_capped_factormax"),
    (0.3, 0.3, 0.0, 0.5, 5.0,  0.7, "PI_balanced_safer"),
]


def _build_init():
    from jax_ensemble import _minimal_config, _compute_ppm_coefs
    from carma.constants import RPA2CGS, R_AIR
    from carma.precision import DTYPE

    S = np.load(ROOT / "scenarios" / "realistic_scenarios_100.npz")
    T_K = float(S["T"][SCEN_IDX]); p_hPa = float(S["p"][SCEN_IDX])
    rh = float(S["rh"][SCEN_IDX])
    prod_rate = float(S["h2so4_prod_rate"][SCEN_IDX])
    M_ug_m3 = float(S["M_total_ug_m3"][SCEN_IDX])
    mu_nm = float(S["aerosol_mu_nm"][SCEN_IDX])
    sigma_g = float(S["aerosol_sigma_g"][SCEN_IDX])

    cfg = _minimal_config(); ppm = _compute_ppm_coefs(cfg)
    p_cgs_val = p_hPa * 100.0 * float(RPA2CGS)
    rhoa = p_cgs_val / (float(R_AIR) * T_K)
    p_cgs = jnp.asarray([p_cgs_val], dtype=DTYPE)
    T = jnp.asarray([T_K], dtype=DTYPE)
    pvapl_Pa = math.exp(54.842763 - 6763.22/T_K - 4.210*math.log(T_K)
                         + 0.000367*T_K)
    h2o_mmr = rh * pvapl_Pa * 18.0 / (29.0 * p_hPa * 100.0)
    gc = jnp.asarray([[h2o_mmr * rhoa, 0.0]], dtype=DTYPE)
    grp = cfg.groups[0]
    r = np.asarray(grp.r); rmass_np = np.asarray(grp.rmass)
    log_mu_cm = math.log(mu_nm * 1e-7); log_sigma = math.log(sigma_g)
    shape_pdf = (np.exp(-0.5 * ((np.log(r) - log_mu_cm) / log_sigma) ** 2)
                  / (r * log_sigma * math.sqrt(2.0 * math.pi)) * rmass_np)
    norm = shape_pdf.sum()
    M_target_mmr = (M_ug_m3 * 1e-12) / rhoa
    mmr_per_bin = shape_pdf * (M_target_mmr / norm)
    pc_per_bin = mmr_per_bin * rhoa / rmass_np
    pc0 = jnp.asarray(pc_per_bin, dtype=DTYPE)[:, None]
    dgc_per_step = prod_rate * DTIME * 98.078479 / 6.02252e23
    return cfg, ppm, pc0, gc, T, p_cgs, dgc_per_step


def _run_pid(cfg, ppm, pc0, gc0_2d, T_arr, p_cgs, dgc_per_step,
              solver_name, rtol, atol,
              pcoeff, icoeff, dcoeff, factormin, factormax, safety):
    from jax_ensemble import _refresh_env
    from carma.precision import DTYPE
    from carma_diffrax.rhs import FrozenEnv, make_rhs
    from carma_diffrax.state import StateShape, pack, unpack
    shape = StateShape(nbin=cfg.nbin, nelem=1, ngas=2)
    grp = cfg.groups[0]
    rmass_np = np.asarray(grp.rmass); rmassup_np = np.asarray(grp.rmassup)
    rmasslow_np = np.concatenate([[rmass_np[0]/(grp.rmrat**0.5)], rmassup_np[:-1]])
    dm = jnp.asarray(rmassup_np - rmasslow_np, dtype=DTYPE)
    solver_cls = {"Kvaerno3": diffrax.Kvaerno3, "Kvaerno4": diffrax.Kvaerno4,
                   "Kvaerno5": diffrax.Kvaerno5, "KenCarp4": diffrax.KenCarp4}
    pc = pc0; gc = gc0_2d; T_scalar = T_arr[0]
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
            pratt=env_d["pratt"][..., 0],
            prat=env_d["prat"][..., 0],
            pden1=env_d["pden1"][..., 0],
            palr=env_d["palr"][..., 0],
        )
        rhs = make_rhs(env, shape)
        term = diffrax.ODETerm(rhs)
        solver = solver_cls[solver_name]()
        controller = diffrax.PIDController(
            rtol=rtol, atol=atol,
            pcoeff=pcoeff, icoeff=icoeff, dcoeff=dcoeff,
            factormin=factormin, factormax=factormax, safety=safety,
        )
        y0 = pack(pc, gc[0], T_scalar)
        sol = diffrax.diffeqsolve(
            term, solver, t0=0.0, t1=DTIME, dt0=None, y0=y0,
            stepsize_controller=controller,
            max_steps=50_000,
            saveat=diffrax.SaveAt(t1=True),
        )
        pc, gc_1d, T_scalar = unpack(sol.ys[-1], shape)
        gc = gc_1d[None, :]
        total_acc += int(sol.stats["num_accepted_steps"])
        total_rej += int(sol.stats["num_rejected_steps"])
        if sol.result != diffrax.RESULTS.successful:
            n_fail += 1
    wall = time.perf_counter() - t_start
    return dict(wall=wall, total_acc=total_acc, total_rej=total_rej,
                 n_fail=n_fail, pc_final=np.asarray(pc[:, 0]),
                 gc_final=np.asarray(gc[0, :]), T_final=float(T_scalar))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solver", default="Kvaerno5",
                    help="solver chosen from Phase 2A")
    p.add_argument("--rtol", type=float, default=1e-5,
                    help="rtol chosen from Phase 2A")
    p.add_argument("--atol", type=float, default=1e-5)
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg, ppm, pc0, gc0, T_arr, p_cgs, dgc_per_step = _build_init()

    print(f"=== Phase 2B: PID gain sweep ({args.solver} @ rtol={args.rtol:.0e}) ===")
    rows = []
    for cfg_pid in PID_CONFIGS:
        pcoeff, icoeff, dcoeff, fmin, fmax, safety, label = cfg_pid
        print(f"\n[{label}] pcoeff={pcoeff} icoeff={icoeff} "
               f"factormin={fmin} factormax={fmax} safety={safety} …", flush=True)
        try:
            r = _run_pid(cfg, ppm, pc0, gc0, T_arr, p_cgs, dgc_per_step,
                          args.solver, args.rtol, args.atol,
                          pcoeff, icoeff, dcoeff, fmin, fmax, safety)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True); continue
        rej_ratio = r["total_rej"] / max(r["total_acc"] + r["total_rej"], 1)
        rows.append(dict(
            label=label, pcoeff=pcoeff, icoeff=icoeff, dcoeff=dcoeff,
            factormin=fmin, factormax=fmax, safety=safety,
            wall=r["wall"], total_acc=r["total_acc"],
            total_rej=r["total_rej"], rej_ratio=rej_ratio,
            n_fail=r["n_fail"],
            pc_final=r["pc_final"], gc_final=r["gc_final"],
        ))
        print(f"  wall={r['wall']:.1f}s, acc={r['total_acc']}, "
               f"rej={r['total_rej']} ({100*rej_ratio:.0f}%), n_fail={r['n_fail']}",
               flush=True)

    np.savez_compressed(OUT_DIR / "pid_sweep.npz",
                         rows=np.array(rows, dtype=object))
    print(f"\nSaved: {OUT_DIR/'pid_sweep.npz'}")

    print("\n=== Summary ===")
    print(f"{'label':<32} {'wall':>7} {'acc':>6} {'rej%':>5}")
    for r in sorted(rows, key=lambda x: x["wall"]):
        print(f"{r['label']:<32} {r['wall']:>6.1f}s {r['total_acc']:>6d} "
               f"{100*r['rej_ratio']:>4.0f}%")


if __name__ == "__main__":
    main()
