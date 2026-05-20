"""Build a multispecies env + initial state for testing.

Mirrors `tests/diffrax/_env_builder.py` but for 3-group setup.
"""
import math
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from carma.constants import RPA2CGS, R_AIR
from carma.precision import DTYPE
from carma_diffrax.multispecies import (
    MultiSpeciesShape, make_multispecies_config, refresh_env_ms,
)


def make_env_and_state(T_K=240.0, p_hPa=200.0, rh=0.3,
                        h2so4_g_per_cm3=1e-14,
                        M_sulf_ug_m3=1.0, mu_sulf_nm=50.0, sigma_sulf=1.8,
                        M_cw_ug_m3=0.0, M_ice_ug_m3=0.0):
    """Build (ms, shape, env, pc0, gc0, T0).

    By default seeds sulfate only (no initial cloud water / ice); tests can
    pass M_cw_ug_m3 > 0 or M_ice_ug_m3 > 0 to seed those groups.

    Returns:
        ms:    MultiSpeciesConfig
        shape: MultiSpeciesShape
        env:   FrozenEnvMS at the initial (T, p, gc)
        pc0:   (nbin, nelem)
        gc0:   (ngas,)
        T0:    scalar
    """
    ms = make_multispecies_config()
    cfg = ms.cfg
    shape = MultiSpeciesShape(nbin=cfg.nbin, nelem=cfg.nelem, ngas=cfg.ngas)

    p_cgs_val = p_hPa * 100.0 * float(RPA2CGS)
    rhoa = p_cgs_val / (float(R_AIR) * T_K)

    T = jnp.asarray([T_K], dtype=DTYPE)
    p_cgs = jnp.asarray([p_cgs_val], dtype=DTYPE)

    pvapl_Pa = math.exp(54.842763 - 6763.22 / T_K
                         - 4.210 * math.log(T_K) + 0.000367 * T_K)
    h2o_mmr = rh * pvapl_Pa * 18.0 / (29.0 * p_hPa * 100.0)
    gc_h2o_cgs = h2o_mmr * rhoa
    gc = jnp.asarray([[gc_h2o_cgs, h2so4_g_per_cm3]], dtype=DTYPE)

    env = refresh_env_ms(T, p_cgs, gc, ms)

    pc = np.zeros((cfg.nbin, cfg.nelem))

    # Helper to seed a lognormal aerosol in one group.
    def _seed_group(igroup, ielem, M_ug_m3, mu_nm, sigma_g):
        if M_ug_m3 <= 0:
            return
        grp = cfg.groups[igroup]
        r = np.asarray(grp.r)
        rmass = np.asarray(grp.rmass)
        log_mu_cm = math.log(mu_nm * 1e-7)
        log_sigma = math.log(sigma_g)
        shape_pdf = (np.exp(-0.5 * ((np.log(r) - log_mu_cm) / log_sigma) ** 2)
                      / (r * log_sigma * math.sqrt(2.0 * math.pi)) * rmass)
        norm = shape_pdf.sum()
        if norm > 0:
            M_target_mmr = (M_ug_m3 * 1e-12) / rhoa
            mmr_per_bin = shape_pdf * (M_target_mmr / norm)
            pc_per_bin = mmr_per_bin * rhoa / rmass
            pc[:, ielem] = pc_per_bin

    _seed_group(0, 0, M_sulf_ug_m3, mu_sulf_nm, sigma_sulf)
    _seed_group(1, 1, M_cw_ug_m3, mu_nm=10000.0, sigma_g=1.5)   # cloud droplets ~10 µm
    _seed_group(2, 2, M_ice_ug_m3, mu_nm=20000.0, sigma_g=1.5)  # ice crystals ~20 µm

    pc0 = jnp.asarray(pc, dtype=DTYPE)
    gc0 = gc[0]
    T0 = T[0]
    return ms, shape, env, pc0, gc0, T0
