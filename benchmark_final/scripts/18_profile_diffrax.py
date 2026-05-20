"""Profile diffrax to identify the bottleneck.

Breakdown:
  1. Single RHS evaluation (after JIT warm-up).
  2. Full diffeqsolve outer step.
  3. RHS sub-components (vapor pressure, supersat, sulfnuc rate,
     pheat × nbin, mass-balance reduction).

Run on scenario 21 initial state.
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
from jax_ensemble import _minimal_config, _compute_ppm_coefs, _refresh_env
from carma.constants import AVG, R_AIR, RPA2CGS
from carma.precision import DTYPE
from carma_diffrax import DiffraxConfig, diffrax_step
from carma_diffrax.rhs import FrozenEnv, make_rhs
from carma_diffrax.state import StateShape, pack
from carma.nucleation.sulfnuc import sulfnuc
from carma.growth.pheat import pheat
from carma.sulfate_utils import wtpct_tabaz
from carma.supersaturation import supersat
from carma.vapor_pressure import vaporp_h2o_murphy2005, vaporp_h2so4_ayers1980


def time_block(fn, n_warmup=3, n_runs=20, label=""):
    """Time `fn` (which must call .block_until_ready() on its output)."""
    for _ in range(n_warmup):
        out = fn()
        if hasattr(out, "block_until_ready"):
            out.block_until_ready()
        elif isinstance(out, tuple):
            for x in out:
                if hasattr(x, "block_until_ready"):
                    x.block_until_ready()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        out = fn()
        if hasattr(out, "block_until_ready"):
            out.block_until_ready()
        elif isinstance(out, tuple):
            for x in out:
                if hasattr(x, "block_until_ready"):
                    x.block_until_ready()
    dt = (time.perf_counter() - t0) / n_runs
    print(f"  {label:50s} {dt*1000:8.3f} ms/call")
    return dt


def build_state():
    """Scenario 21 initial state + frozen env."""
    S = np.load(ROOT / "scenarios" / "realistic_scenarios_100.npz")
    i = 21
    T_K = float(S["T"][i]); p_hPa = float(S["p"][i]); rh = float(S["rh"][i])
    M_ug_m3 = float(S["M_total_ug_m3"][i])
    mu_nm = float(S["aerosol_mu_nm"][i])
    sigma_g = float(S["aerosol_sigma_g"][i])

    cfg = _minimal_config()
    ppm = _compute_ppm_coefs(cfg)
    p_cgs_val = p_hPa * 100.0 * float(RPA2CGS)
    rhoa = p_cgs_val / (float(R_AIR) * T_K)
    p_cgs = jnp.asarray([p_cgs_val], dtype=DTYPE)
    T = jnp.asarray([T_K], dtype=DTYPE)
    pvapl_Pa = math.exp(54.842763 - 6763.22 / T_K
                         - 4.210 * math.log(T_K) + 0.000367 * T_K)
    h2o_mmr = rh * pvapl_Pa * 18.0 / (29.0 * p_hPa * 100.0)
    gc = jnp.asarray([[h2o_mmr * rhoa, 1e-13]], dtype=DTYPE)
    env_d = _refresh_env(T, p_cgs, gc, cfg, ppm)

    grp = cfg.groups[0]
    rmass_np = np.asarray(grp.rmass)
    rmassup_np = np.asarray(grp.rmassup)
    rmasslow_np = np.concatenate([[rmass_np[0]/(grp.rmrat**0.5)], rmassup_np[:-1]])
    dm = jnp.asarray(rmassup_np - rmasslow_np, dtype=DTYPE)
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

    # Lognormal initial pc
    r = np.asarray(grp.r)
    log_mu_cm = math.log(mu_nm * 1e-7)
    log_sigma = math.log(sigma_g)
    shape_pdf = (np.exp(-0.5 * ((np.log(r) - log_mu_cm) / log_sigma) ** 2)
                  / (r * log_sigma * math.sqrt(2.0 * math.pi)) * rmass_np)
    norm = shape_pdf.sum()
    M_target_mmr = (M_ug_m3 * 1e-12) / rhoa
    pc_per_bin = shape_pdf * (M_target_mmr / norm) * rhoa / rmass_np
    pc0 = jnp.asarray(pc_per_bin, dtype=DTYPE)[:, None]

    shape = StateShape(nbin=cfg.nbin, nelem=1, ngas=2)
    return env, shape, pc0, gc[0], T[0], cfg


def main():
    env, shape, pc0, gc0, T0, cfg = build_state()
    nbin = shape.nbin
    y0 = pack(pc0, gc0, T0)

    print("=== Diffrax breakdown — scenario 21 initial state ===\n")
    print("== Top-level operations ==")

    # 1. Full diffeqsolve outer step (one 1800s integration)
    cfg_d = DiffraxConfig(rtol=1e-5, atol=1e-5, max_steps=20_000)
    def full_step():
        pc, gc, T, stats = diffrax_step(pc0, gc0, T0, 1800.0, env, shape, cfg_d)
        return pc
    time_block(full_step, n_warmup=1, n_runs=3,
                label="diffrax_step (full 1800s, Kvaerno5+PID@1e-5)")

    # 2. RHS single evaluation
    rhs = make_rhs(env, shape)
    def rhs_call():
        return rhs(0.0, y0, None)
    time_block(rhs_call, label="rhs(t,y) (one Kvaerno5 residual eval)")

    # 3. RHS sub-components
    print("\n== Sub-components ==")
    T = jnp.atleast_1d(T0)
    gc_2d = gc0[None, :]
    pc_3d = pc0[None, :, :]

    f1 = jax.jit(lambda T: vaporp_h2o_murphy2005(T))
    time_block(lambda: f1(T), label="vaporp_h2o_murphy2005")

    pvapl_h2o, pvapi_h2o = f1(T)
    f2 = jax.jit(lambda T, gc_h2o: vaporp_h2so4_ayers1980(
        T, gc_h2o, pvapl_h2o, env.zmet))
    time_block(lambda: f2(T, gc_2d[:, 0]), label="vaporp_h2so4_ayers1980")

    pvap_h2so4, _ = f2(T, gc_2d[:, 0])
    pvapl = jnp.stack([pvapl_h2o, pvap_h2so4], axis=1)
    pvapi = jnp.stack([pvapi_h2o, pvap_h2so4], axis=1)
    f3 = jax.jit(lambda T, gc: supersat(T, gc, pvapl[:, 1], pvapi[:, 1],
                                         DTYPE(98.078479), env.zmet))
    time_block(lambda: f3(T, gc_2d[:, 1]), label="supersat (1 gas)")

    # pheat × nbin via vmap
    ssl_h2so4, ssi_h2so4 = f3(T, gc_2d[:, 1])
    ssl_h2o, _ = supersat(T, gc_2d[:, 0], pvapl[:, 0], pvapi[:, 0],
                           DTYPE(18.01528), env.zmet)
    supsatl = jnp.stack([ssl_h2o, ssl_h2so4], axis=1)
    supsati = jnp.stack([jnp.zeros_like(ssl_h2o), ssi_h2so4], axis=1)

    def _dmdt_at(ibin):
        return pheat(
            pc_3d, supsatl, supsati, pvapl, pvapi,
            env.akelvin, env.akelvini, env.gro, env.gro1,
            env.rup_wet, env.rmass, False, 0, 0, ibin, 1,
        )
    f4 = jax.jit(jax.vmap(_dmdt_at))
    time_block(lambda: f4(jnp.arange(nbin - 1)),
                label="pheat × (nbin-1) via vmap")

    # sulfnuc
    h2o_cgs = gc_2d[0, 0] / env.zmet[0]
    h2so4_cgs = gc_2d[0, 1] / env.zmet[0]
    h2o_n = h2o_cgs * AVG / DTYPE(18.01528)
    h2so4_n = h2so4_cgs * AVG / DTYPE(98.078479)
    rh = ssl_h2o[0] + DTYPE(1.0)
    wtp = wtpct_tabaz(T0, h2o_cgs, pvapl[0, 0])
    f5 = jax.jit(lambda *args: sulfnuc(
        T0, wtp, rh, args[0], args[1], h2o_n, h2o_cgs,
        env.r_wet, env.rmassup, DTYPE(env.rmrat_val), env.zmet[0],
        method="ZhaoTurco",
        do_homogeneous=True, do_heterogeneous=False,
        gwtmol_h2so4=DTYPE(98.078479), gwtmol_h2o=DTYPE(18.01528),
    ))
    time_block(lambda: f5(h2so4_n, h2so4_cgs),
                label="sulfnuc (homogeneous, ZhaoTurco)")


if __name__ == "__main__":
    main()
