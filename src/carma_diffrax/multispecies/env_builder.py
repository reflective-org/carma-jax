"""Build a FrozenEnvMS for the multispecies RHS from current (T, p, gc).

Mirrors `scripts/jax_ensemble.py:_refresh_env` but with proper multi-group
handling: per-group wet radii (sulfate swells with H2SO4 wtpct; water and
ice don't swell), per-group bin density, and the resulting akelvin / gro /
gro1 / rup_wet stacks for all groups together.

For now we don't recompute akelvin every RHS call — the env is frozen at
the start of each outer step.
"""
import jax.numpy as jnp
import numpy as np

from carma.constants import GRAV, R_AIR
from carma.enums import GridType, SwellMethod
from carma.precision import DTYPE
from carma.setup_atm import setup_atm
from carma.setup_ckern import setup_ckern_jit
from carma.setup_gkern import setup_gkern
from carma.setup_grow import setup_grow
from carma.setup_vf import setup_vf_jit
from carma.sulfate_utils import wtpct_tabaz
from carma.vapor_pressure import vaporp_h2o_murphy2005
from carma.wetr import get_wetr

from carma_diffrax.multispecies.config import MultiSpeciesConfig
from carma_diffrax.multispecies.rhs import FrozenEnvMS


_GWTMOL_H2O = DTYPE(18.01528)
_GWTMOL_H2SO4 = DTYPE(98.078479)


def refresh_env_ms(T, p_cgs, gc, ms: MultiSpeciesConfig) -> FrozenEnvMS:
    """Build a `FrozenEnvMS` for the current atmospheric state.

    Args:
        T: shape (1,) temperature [K]
        p_cgs: shape (1,) pressure [dyne/cm^2]
        gc: shape (1, ngas) gas conc [g/cm^3 × zmet]
        ms: MultiSpeciesConfig

    Returns:
        FrozenEnvMS with all per-group tables stacked.
    """
    cfg = ms.cfg
    nbin = cfg.nbin
    ngroup = cfg.ngroup
    ngas = cfg.ngas

    # 1) Atmosphere — thin layer at current (T, p).
    rho_approx = float(p_cgs[0]) / (float(R_AIR) * float(T[0]))
    deltaz = 1.0e3
    zc = jnp.asarray([1.0e5])
    zl = jnp.asarray([zc[0] - deltaz / 2, zc[0] + deltaz / 2])
    pl = jnp.asarray([
        p_cgs[0] + 0.5 * deltaz * rho_approx * float(GRAV),
        p_cgs[0] - 0.5 * deltaz * rho_approx * float(GRAV),
    ])
    rhoa, dz, zmet, zmetl, rmu, thcond, rhoa_wet = setup_atm(
        T, p_cgs, pl, zc, zl, GridType.I_CART,
    )

    # 2) Diffusivity + latent heats per gas (multi-gas already).
    diffus, rlhe, rlhm = setup_grow(
        T, p_cgs, rhoa, zmet,
        igash2o=cfg.igash2o, igash2so4=cfg.igash2so4,
        ngas=ngas, do_cnst_rlh=False,
    )

    # 3) Wet radii per group. Stack into (nz=1, nbin, ngroup).
    h2o_mass_cgs = gc[0, cfg.igash2o] / zmet[0]
    pvapl_h2o_arr, _ = vaporp_h2o_murphy2005(T)
    h2o_vp_cgs = pvapl_h2o_arr[0]
    relhum = h2o_mass_cgs / (
        h2o_vp_cgs * _GWTMOL_H2O / (jnp.asarray(8.314e7, dtype=DTYPE) * T[0])
    )
    r_wet_per_group = []
    rup_wet_per_group = []
    rlow_wet_per_group = []
    rho_wet_per_group = []
    for ig, grp in enumerate(cfg.groups):
        r_dry = jnp.asarray(grp.r, dtype=DTYPE)
        rup_dry = jnp.asarray(grp.rup, dtype=DTYPE)
        rlow_dry = jnp.asarray(grp.rlow, dtype=DTYPE)
        rho_dry_per_bin = jnp.full(r_dry.shape, jnp.asarray(grp.rho, dtype=DTYPE)
                                    if hasattr(grp, "rho_dry") else
                                    jnp.asarray(_default_rho_for_group(grp),
                                                 dtype=DTYPE))
        swell = int(grp.irhswell)
        if swell == int(SwellMethod.I_WTPCT_H2SO4):
            # Sulfate swelling — wet radius depends on H2SO4 weight-%.
            r_wet_1d, rho_wet = get_wetr(
                r_dry, rho_dry_per_bin, relhum, T[0], swell,
                h2o_mass=h2o_mass_cgs, h2o_vp=h2o_vp_cgs,
                gwtmol_h2so4=_GWTMOL_H2SO4,
            )
            rup_wet_1d, _ = get_wetr(
                rup_dry, rho_dry_per_bin, relhum, T[0], swell,
                h2o_mass=h2o_mass_cgs, h2o_vp=h2o_vp_cgs,
                gwtmol_h2so4=_GWTMOL_H2SO4,
            )
            rlow_wet_1d, _ = get_wetr(
                rlow_dry, rho_dry_per_bin, relhum, T[0], swell,
                h2o_mass=h2o_mass_cgs, h2o_vp=h2o_vp_cgs,
                gwtmol_h2so4=_GWTMOL_H2SO4,
            )
        else:
            # No swelling — water and ice groups carry their dry radius.
            r_wet_1d = r_dry
            rup_wet_1d = rup_dry
            rlow_wet_1d = rlow_dry
            rho_wet = rho_dry_per_bin
        r_wet_per_group.append(r_wet_1d)
        rup_wet_per_group.append(rup_wet_1d)
        rlow_wet_per_group.append(rlow_wet_1d)
        rho_wet_per_group.append(rho_wet)

    # Stack per-group → (nbin, ngroup) and lift to (1, nbin, ngroup).
    r_wet_2d = jnp.stack(r_wet_per_group, axis=1)         # (nbin, ngroup)
    rup_wet_2d = jnp.stack(rup_wet_per_group, axis=1)
    rlow_wet_2d = jnp.stack(rlow_wet_per_group, axis=1)
    rho_wet_2d = jnp.stack(rho_wet_per_group, axis=1)
    r_wet = r_wet_2d[None, :, :]                          # (1, nbin, ngroup)
    rup_wet = rup_wet_2d[None, :, :]
    rlow_wet = rlow_wet_2d[None, :, :]

    # 4) Velocity / Reynolds (passed to ckern; not used by RHS directly).
    rrat_2d = jnp.ones((nbin, ngroup), dtype=DTYPE)
    rprat_2d = jnp.ones((nbin, ngroup), dtype=DTYPE)
    vf, re_real, bpm = setup_vf_jit(
        T, rhoa, zmet, rmu, r_wet, rho_wet_2d, rrat_2d, rprat_2d,
    )

    # 5) Growth kernel per group via setup_gkern (multi-group capable).
    eshape_arr = jnp.asarray([float(g.eshape) for g in cfg.groups], dtype=DTYPE)
    is_ice_arr_np = np.asarray([bool(g.is_ice) for g in cfg.groups])
    gwtmol_arr = jnp.asarray([float(g.wtmol) for g in cfg.gases], dtype=DTYPE)
    igrowgas_arr = ms.igrowgas
    rrat = jnp.ones((nbin, ngroup), dtype=DTYPE)
    wtpct_arr = wtpct_tabaz(
        T, jnp.atleast_1d(h2o_mass_cgs), jnp.atleast_1d(h2o_vp_cgs),
    )
    _, akelvin, akelvini, gro, gro1, gro2, _, _ = setup_gkern(
        T, p_cgs, rhoa, zmet, rmu, thcond, diffus, rlhe, rlhm,
        re_real, r_wet, rlow_wet, rrat, eshape_arr, is_ice_arr_np,
        gwtmol_arr, igrowgas_arr,
        cfg.gstickl, cfg.gsticki, cfg.tstick,
        nbin, ngroup, ngas,
        igash2o=cfg.igash2o, igash2so4=cfg.igash2so4, wtpct=wtpct_arr,
    )

    return FrozenEnvMS(
        akelvin=akelvin, akelvini=akelvini,
        gro=gro, gro1=gro1,
        rup_wet=rup_wet, rlow_wet=rlow_wet,
        r_wet=r_wet_2d,
        rmass_2d=jnp.transpose(ms.rmass_2d),        # (nbin, ngroup) from (ngroup,nbin)
        dm_2d=jnp.transpose(ms.dm_2d),
        rmassup_2d=jnp.transpose(ms.rmassup_2d),
        rhoa=rhoa,
        zmet=zmet,
        rlhe=rlhe,
        rlhm=rlhm,
    )


def _default_rho_for_group(grp):
    """Return a per-bin scalar dry density for a group from its bin geometry."""
    rmass_np = np.asarray(grp.rmass)
    r_np = np.asarray(grp.r)
    # rho = mass / (4/3 π r³).
    rho = rmass_np / ((4.0 / 3.0) * np.pi * r_np ** 3)
    return rho.mean()  # all bins same by construction
