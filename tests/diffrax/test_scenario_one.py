"""Run one realistic scenario end-to-end at 1800 s outer timesteps.

Mirrors the structure of `benchmark_final/scripts/04_run_jax_parallel.py`
but uses `carma_diffrax.diffrax_step` instead of the faithful port's
`step_full_faithful.step`. This is the first true integration test of
the diffrax-based solver against the kind of scenario that breaks
Fortran's explicit-Euler retry mechanism.
"""
import math
import sys
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from jax_ensemble import _minimal_config, _compute_ppm_coefs, _refresh_env
from carma.constants import AVG, R_AIR, RPA2CGS
from carma.precision import DTYPE

from carma_diffrax import DiffraxConfig, diffrax_step
from carma_diffrax.rhs import FrozenEnv
from carma_diffrax.state import StateShape


_SCENARIOS = REPO / "benchmark_final" / "scenarios" / "realistic_scenarios_100.npz"


def _env_from_dict(env_dict, cfg):
    grp = cfg.groups[0]
    rmass_np = np.asarray(grp.rmass)
    rmassup_np = np.asarray(grp.rmassup)
    rmasslow_np = np.concatenate(
        [[rmass_np[0] / (grp.rmrat ** 0.5)], rmassup_np[:-1]]
    )
    dm = jnp.asarray(rmassup_np - rmasslow_np, dtype=DTYPE)
    r_wet = (env_dict["rup_wet"][0, :, 0] + env_dict["rlow_wet"][0, :, 0]) / 2.0
    return FrozenEnv(
        akelvin=env_dict["akelvin"], akelvini=env_dict["akelvini"],
        gro=env_dict["gro"], gro1=env_dict["gro1"],
        rup_wet=env_dict["rup_wet"], r_wet=r_wet,
        rmass=jnp.asarray(grp.rmass, dtype=DTYPE)[:, None],
        dm=dm[:, None],
        rmassup=jnp.asarray(grp.rmassup, dtype=DTYPE),
        rmrat_val=float(grp.rmrat),
        rhoa=env_dict["rhoa"], zmet=env_dict["zmet"],
        rlhe=env_dict["rlhe"], rlhm=env_dict["rlhm"],
    )


def _run_scenario(scen_idx, dtime=1800.0, nstep=48,
                    rtol=1e-5, atol=1e-5):
    """Return (pc_final, gc_final, T_final, per_step_stats)."""
    S = np.load(_SCENARIOS)
    T_K = float(S["T"][scen_idx])
    p_hPa = float(S["p"][scen_idx])
    rh = float(S["rh"][scen_idx])
    prod_rate = float(S["h2so4_prod_rate"][scen_idx])      # molec/cm^3/s
    M_ug_m3 = float(S["M_total_ug_m3"][scen_idx])
    mu_nm = float(S["aerosol_mu_nm"][scen_idx])
    sigma_g = float(S["aerosol_sigma_g"][scen_idx])

    cfg = _minimal_config()
    ppm = _compute_ppm_coefs(cfg)
    shape = StateShape(nbin=cfg.nbin, nelem=1, ngas=2)

    p_cgs_val = p_hPa * 100.0 * float(RPA2CGS)
    rhoa_g_cm3 = p_cgs_val / (float(R_AIR) * T_K)
    T = jnp.asarray([T_K], dtype=DTYPE)
    p_cgs = jnp.asarray([p_cgs_val], dtype=DTYPE)

    pvapl_Pa = math.exp(54.842763 - 6763.22 / T_K
                         - 4.210 * math.log(T_K) + 0.000367 * T_K)
    h2o_mmr = rh * pvapl_Pa * 18.0 / (29.0 * p_hPa * 100.0)
    gc = jnp.asarray([[h2o_mmr * rhoa_g_cm3, 0.0]], dtype=DTYPE)

    # Initial lognormal aerosol seed.
    grp = cfg.groups[0]
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

    per_step_stats = []
    T_scalar = T[0]
    for istep in range(nstep):
        # Inject H2SO4 production for this outer step.
        gc = gc.at[0, 1].set(gc[0, 1] + dgc_per_step)
        # Refresh env from current (T, gc).
        env_dict = _refresh_env(jnp.atleast_1d(T_scalar), p_cgs, gc, cfg, ppm)
        env = _env_from_dict(env_dict, cfg)
        # Integrate one outer step.
        pc, gc_1d, T_scalar, stats = diffrax_step(
            pc, gc[0], T_scalar, dtime, env, shape, cfg_d,
        )
        gc = gc_1d[None, :]
        per_step_stats.append(stats)

    return pc, gc, T_scalar, per_step_stats


@pytest.mark.diffrax
@pytest.mark.slow
def test_scenario26_high_prod_rate_1800s():
    """Scenario 26 has prod_rate ~ 1e7 — extreme stiffness expected.

    Verify:
    - integration completes without diffrax failure on any outer step
    - gc[H2SO4] >= 0 throughout (the Fortran failure mode)
    - all pc bins >= 0
    """
    pc, gc, T, stats_list = _run_scenario(scen_idx=26, dtime=1800.0, nstep=48)

    # Report
    total_accepted = sum(s["num_accepted_steps"] for s in stats_list)
    total_rejected = sum(s["num_rejected_steps"] for s in stats_list)
    failures = [i for i, s in enumerate(stats_list) if not s["successful"]]
    print(f"\nscen 26 (1800s × 48):")
    print(f"  total accepted={total_accepted}, rejected={total_rejected}")
    print(f"  failures={failures}")
    print(f"  gc_h2so4_final={float(gc[0, 1]):.3e}")
    print(f"  pc_min={float(jnp.min(pc)):.3e}, pc_max={float(jnp.max(pc)):.3e}")
    print(f"  T_final={float(T):.2f}K")

    assert len(failures) == 0, f"diffrax failed at steps {failures}"
    assert float(gc[0, 1]) >= 0.0, (
        f"gc[H2SO4] went negative: {float(gc[0, 1]):.3e}"
    )
    assert float(jnp.min(pc)) >= -1e-30, (
        f"pc went negative: {float(jnp.min(pc)):.3e}"
    )
