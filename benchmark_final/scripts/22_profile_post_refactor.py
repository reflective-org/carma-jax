"""Post-refactor profile: where does the remaining time go?

Compares the new env-as-args path against the legacy closure path on the
exact same scenario (scenario 21). Reports JIT-warmup vs steady-state
per-step cost.
"""
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


def build_init():
    from jax_ensemble import _minimal_config, _compute_ppm_coefs
    from carma.constants import RPA2CGS, R_AIR
    from carma.precision import DTYPE

    S = np.load(ROOT / "scenarios" / "realistic_scenarios_100.npz")
    i = 21
    T_K = float(S["T"][i]); p_hPa = float(S["p"][i]); rh = float(S["rh"][i])
    prod_rate = float(S["h2so4_prod_rate"][i])
    M_ug_m3 = float(S["M_total_ug_m3"][i])
    mu_nm = float(S["aerosol_mu_nm"][i])
    sigma_g = float(S["aerosol_sigma_g"][i])

    cfg = _minimal_config(); ppm = _compute_ppm_coefs(cfg)
    p_cgs_val = p_hPa * 100.0 * float(RPA2CGS)
    rhoa = p_cgs_val / (float(R_AIR) * T_K)
    p_cgs = jnp.asarray([p_cgs_val], dtype=DTYPE)
    T = jnp.asarray([T_K], dtype=DTYPE)
    pvapl_Pa = math.exp(54.842763 - 6763.22/T_K - 4.210*math.log(T_K) + 0.000367*T_K)
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
    dgc_per_step = prod_rate * 1800.0 * 98.078479 / 6.02252e23
    return cfg, ppm, pc0, gc, T, p_cgs, dgc_per_step


def main():
    from jax_ensemble import _refresh_env
    from carma.precision import DTYPE
    from carma_diffrax import DiffraxConfig, diffrax_step
    from carma_diffrax.rhs import FrozenEnv
    from carma_diffrax.state import StateShape

    cfg, ppm, pc0, gc0, T_arr, p_cgs, dgc_per_step = build_init()
    shape = StateShape(nbin=cfg.nbin, nelem=1, ngas=2)
    grp = cfg.groups[0]
    rmass_np = np.asarray(grp.rmass); rmassup_np = np.asarray(grp.rmassup)
    rmasslow_np = np.concatenate([[rmass_np[0]/(grp.rmrat**0.5)], rmassup_np[:-1]])
    dm = jnp.asarray(rmassup_np - rmasslow_np, dtype=DTYPE)

    pc = pc0; gc = gc0; T_scalar = T_arr[0]
    cfg_d = DiffraxConfig(rtol=1e-5, atol=1e-5, max_steps=20_000)

    print("=== Per-outer-step breakdown (scenario 21, 1800s outer step) ===")
    print(f"{'step':>4} {'env_build':>10} {'diffrax_step':>13} {'acc':>5} {'rej':>5}")

    env_total_ms = 0.0
    step_total_ms = 0.0
    for istep in range(48):
        gc = gc.at[0, 1].set(gc[0, 1] + dgc_per_step)
        t0 = time.perf_counter()
        env_d = _refresh_env(jnp.atleast_1d(T_scalar), p_cgs, gc, cfg, ppm)
        env_d["akelvin"].block_until_ready()  # force materialization
        env_ms = (time.perf_counter() - t0) * 1000

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

        t0 = time.perf_counter()
        pc, gc_1d, T_scalar, stats = diffrax_step(
            pc, gc[0], T_scalar, 1800.0, env, shape, cfg_d,
        )
        pc.block_until_ready()
        step_ms = (time.perf_counter() - t0) * 1000

        gc = gc_1d[None, :]
        env_total_ms += env_ms
        step_total_ms += step_ms

        if istep < 5 or istep % 10 == 0 or istep == 47:
            print(f"{istep:>4d} {env_ms:>9.1f}ms {step_ms:>12.1f}ms "
                   f"{stats['num_accepted_steps']:>5d} {stats['num_rejected_steps']:>5d}")

    print(f"\nTotals (48 outer steps):")
    print(f"  env build : {env_total_ms/1000:.2f}s ({100*env_total_ms/(env_total_ms+step_total_ms):.0f}%)")
    print(f"  diffrax   : {step_total_ms/1000:.2f}s ({100*step_total_ms/(env_total_ms+step_total_ms):.0f}%)")
    print(f"  total     : {(env_total_ms+step_total_ms)/1000:.2f}s")


if __name__ == "__main__":
    main()
