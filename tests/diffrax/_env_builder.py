"""Helpers to build a FrozenEnv for testing.

Bridges between the faithful-port's `_refresh_env` helper in
`scripts/jax_ensemble.py` and the diffrax `FrozenEnv` shape.
"""
import math
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from jax_ensemble import _minimal_config, _compute_ppm_coefs, _refresh_env

from carma.constants import RPA2CGS, R_AIR
from carma.precision import DTYPE
from carma_diffrax.rhs import FrozenEnv
from carma_diffrax.state import StateShape


def make_env_and_state(T_K=240.0, p_hPa=200.0, rh=0.3,
                       h2so4_g_per_cm3=1e-15,
                       mu_nm=50.0, sigma_g=1.8, M_ug_m3=1.0):
    """Build a FrozenEnv plus an initial (pc, gc, T) state for tests.

    Default scenario is loosely strat-like (T=240, p=200hPa, RH=30%) with a
    small lognormal sulfate seed and tiny background H2SO4 vapor — enough
    to exercise the RHS without forcing any tail explosions.
    """
    cfg = _minimal_config()
    ppm = _compute_ppm_coefs(cfg)

    p_cgs_val = p_hPa * 100.0 * float(RPA2CGS)
    rhoa_g_cm3 = p_cgs_val / (float(R_AIR) * T_K)

    T = jnp.asarray([T_K], dtype=DTYPE)
    p_cgs = jnp.asarray([p_cgs_val], dtype=DTYPE)

    pvapl_Pa = math.exp(54.842763 - 6763.22 / T_K
                         - 4.210 * math.log(T_K) + 0.000367 * T_K)
    h2o_mmr = rh * pvapl_Pa * 18.0 / (29.0 * p_hPa * 100.0)
    gc = jnp.asarray(
        [[h2o_mmr * rhoa_g_cm3, h2so4_g_per_cm3]], dtype=DTYPE,
    )

    env_dict = _refresh_env(T, p_cgs, gc, cfg, ppm)

    nbin = cfg.nbin
    grp = cfg.groups[0]
    rmass = jnp.asarray(grp.rmass, dtype=DTYPE)
    rmassup = jnp.asarray(grp.rmassup, dtype=DTYPE)
    rmass_np = np.asarray(grp.rmass)
    rmassup_np = np.asarray(grp.rmassup)
    # True bin width: rmassup[i] - rmasslow[i] where rmasslow[i] = rmassup[i-1]
    # and rmasslow[0] = rmass[0] / sqrt(rmrat). For rmrat=2 this gives
    # dm = (2/3) * rmass.
    rmasslow_np = np.concatenate([
        [rmass_np[0] / (grp.rmrat ** 0.5)], rmassup_np[:-1],
    ])
    dm = jnp.asarray(rmassup_np - rmasslow_np, dtype=DTYPE)
    r_wet = env_dict["rup_wet"][0, :, 0]   # use upper wet radius as ~mid radius proxy
    # Actually the existing _refresh_env produces rup_wet and rlow_wet; the
    # mid-radius for nucleation hetnucl is r_wet (the bin center). Use the
    # arithmetic mean of upper/lower as a reasonable approximation.
    r_wet = (env_dict["rup_wet"][0, :, 0] + env_dict["rlow_wet"][0, :, 0]) / 2.0

    env = FrozenEnv(
        akelvin=env_dict["akelvin"],
        akelvini=env_dict["akelvini"],
        gro=env_dict["gro"],
        gro1=env_dict["gro1"],
        rup_wet=env_dict["rup_wet"],
        r_wet=r_wet,
        rmass=rmass[:, None],                       # (nbin, ngroup=1)
        dm=dm[:, None],                              # (nbin, ngroup=1)
        rmassup=rmassup,
        rmrat_val=float(grp.rmrat),
        rhoa=env_dict["rhoa"],
        zmet=env_dict["zmet"],
        rlhe=env_dict["rlhe"],
        rlhm=env_dict["rlhm"],
        pratt=env_dict["pratt"][..., 0],
        prat=env_dict["prat"][..., 0],
        pden1=env_dict["pden1"][..., 0],
        palr=env_dict["palr"][..., 0],
    )

    # Initial lognormal aerosol seed.
    r = np.asarray(grp.r)
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

    pc0 = jnp.asarray(pc_per_bin, dtype=DTYPE)[:, None]      # (nbin, nelem=1)
    gc0 = gc[0]                                                # (ngas=2,)
    T0 = T[0]                                                  # scalar

    shape = StateShape(nbin=nbin, nelem=1, ngas=2)
    return env, shape, pc0, gc0, T0
