"""Test 2x2 = (Kvaerno5, Tsit5) × (DirectAdjoint, BacksolveAdjoint)
combinations to see which one lets jax.grad work through full physics
(do_homogeneous=True) on a 1-outer-step diffrax run.

Reports for each combo:
  - Whether forward call succeeds
  - Whether jax.jvp (forward-mode) succeeds
  - Whether jax.grad (reverse-mode) succeeds
  - Wall time for each
  - Number of accepted internal steps (proxy for solver step count)
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import diffrax as dfx
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import optimistix as optx

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from carma.constants import AVG, R_AIR, RPA2CGS
from carma.precision import DTYPE
from carma_diffrax.rhs import FrozenEnv, make_rhs
from carma_diffrax.state import StateShape, pack, unpack
from jax_ensemble import _minimal_config, _compute_ppm_coefs, _refresh_env


# Scenario 21 settings
SCEN = dict(T=215.6, p=27.0, rh=0.51, h2so4_prod_rate=9.6e6,
            M_total_ug_m3=2.54, aerosol_mu_nm=681.0, aerosol_sigma_g=1.46)
DTIME = 1800.0


def _build_state():
    cfg = _minimal_config()._replace(do_coag=False, do_grow=True)
    ppm = _compute_ppm_coefs(cfg)
    grp = cfg.groups[0]
    shape = StateShape(nbin=cfg.nbin, nelem=1, ngas=2)

    p_cgs_val = SCEN["p"] * 100.0 * float(RPA2CGS)
    rho_air = p_cgs_val / (float(R_AIR) * SCEN["T"])
    pvapl = math.exp(54.842763 - 6763.22 / SCEN["T"]
                      - 4.210 * math.log(SCEN["T"]) + 0.000367 * SCEN["T"])
    h2o_mmr = SCEN["rh"] * pvapl * 18.0 / (29.0 * SCEN["p"] * 100.0)

    r = np.asarray(grp.r)
    rmass_np = np.asarray(grp.rmass)
    log_mu = math.log(SCEN["aerosol_mu_nm"] * 1e-7)
    log_s = math.log(SCEN["aerosol_sigma_g"])
    pdf = (np.exp(-0.5 * ((np.log(r) - log_mu) / log_s) ** 2)
            / (r * log_s * math.sqrt(2 * math.pi)) * rmass_np)
    M_target_mmr = (SCEN["M_total_ug_m3"] * 1e-12) / rho_air
    pc_init = pdf * (M_target_mmr / pdf.sum()) * rho_air / rmass_np
    pc = jnp.asarray(pc_init, dtype=DTYPE)[:, None]
    gc = jnp.asarray([h2o_mmr * rho_air, 0.0], dtype=DTYPE)
    T = jnp.asarray(SCEN["T"], dtype=DTYPE)
    p_cgs = jnp.asarray([p_cgs_val], dtype=DTYPE)
    return cfg, ppm, grp, shape, pc, gc, T, p_cgs


def make_f(solver_name, adjoint_name):
    """Build f(prod_rate) -> final particle mass with chosen solver + adjoint."""
    cfg, ppm, grp, shape, pc0, gc0, T0, p_cgs = _build_state()
    rmass_np = np.asarray(grp.rmass)
    rmassup_np = np.asarray(grp.rmassup)
    rmasslow_np = np.concatenate(
        [[rmass_np[0] / (grp.rmrat ** 0.5)], rmassup_np[:-1]]
    )
    dm = jnp.asarray(rmassup_np - rmasslow_np, dtype=DTYPE)
    rmass_jnp = jnp.asarray(rmass_np, dtype=DTYPE)

    H2SO4_PER_MOLEC_G = 98.078479 / float(AVG)

    rtol, atol = 1e-5, 1e-5
    if solver_name == "Kvaerno5":
        solver = dfx.Kvaerno5(root_finder=optx.Chord(rtol=rtol, atol=atol))
    elif solver_name == "Tsit5":
        solver = dfx.Tsit5()
    else:
        raise ValueError(solver_name)
    if adjoint_name == "DirectAdjoint":
        adjoint = dfx.DirectAdjoint()
    elif adjoint_name == "BacksolveAdjoint":
        adjoint = dfx.BacksolveAdjoint()
    else:
        raise ValueError(adjoint_name)

    term = dfx.ODETerm(make_rhs(shape, do_growth=True, do_homogeneous=True))
    controller = dfx.PIDController(
        rtol=rtol, atol=atol,
        pcoeff=0.3, icoeff=0.3, dcoeff=0.0,
        factormin=0.5, factormax=5.0, safety=0.8,
    )

    @jax.jit
    def step_inner(pc_, gc_, T_, env):
        y0 = pack(pc_, gc_, T_)
        sol = dfx.diffeqsolve(
            term, solver,
            t0=0.0, t1=DTIME, dt0=None, y0=y0, args=env,
            stepsize_controller=controller,
            max_steps=50_000,
            saveat=dfx.SaveAt(t1=True),
            adjoint=adjoint,
        )
        pc_e, gc_e, T_e = unpack(sol.ys[-1], shape)
        return pc_e, gc_e, T_e, sol.stats

    def f(prod_rate):
        prod_rate = jnp.asarray(prod_rate, dtype=DTYPE)
        dgc = prod_rate * DTYPE(DTIME) * DTYPE(H2SO4_PER_MOLEC_G)
        gc = gc0.at[1].set(gc0[1] + dgc)
        env_d = _refresh_env(
            jnp.atleast_1d(T0), p_cgs, gc[None, :], cfg, ppm,
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
        pc, gc_out, T_out, stats = step_inner(pc0, gc, T0, env)
        return jnp.sum(pc[:, 0] * rmass_jnp)

    return f


def try_mode(label, fn):
    t0 = time.perf_counter()
    try:
        result = fn()
        wall = time.perf_counter() - t0
        print(f"    {label}: OK  result={float(result):.3e}  wall={wall:.1f}s")
        return True, float(result), wall
    except Exception as e:
        wall = time.perf_counter() - t0
        print(f"    {label}: FAIL ({wall:.1f}s) {type(e).__name__}: {str(e)[:120]}")
        return False, None, wall


def main():
    combos = [
        ("Kvaerno5", "DirectAdjoint"),     # current default
        ("Kvaerno5", "BacksolveAdjoint"),  # try alt adjoint
        ("Tsit5",    "DirectAdjoint"),     # try explicit solver
        ("Tsit5",    "BacksolveAdjoint"),
    ]
    p0 = SCEN["h2so4_prod_rate"]
    for solver, adjoint in combos:
        print(f"\n=== {solver} + {adjoint} ===")
        f = make_f(solver, adjoint)
        try_mode("forward", lambda: f(p0))
        try_mode("jvp",      lambda: jax.jvp(f, (p0,), (1.0,))[1])
        try_mode("grad",     lambda: jax.grad(f)(p0))


if __name__ == "__main__":
    main()
