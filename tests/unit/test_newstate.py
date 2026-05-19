"""Unit tests for newstate dispatcher (Phase 9.1)."""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from carma.constants import AVG, BK, WTMOL_H2O
from carma.newstate import newstate
from carma.newstate_calc import newstate_calc_growth


def _growth_cfg():
    """Minimal growth scenario: one volatile H2O group, no cores."""
    nz, nbin, ngroup, ngas, nelem = 1, 10, 1, 1, 1

    # Bin geometry
    rho = 1.0
    rmass = jnp.asarray([(2.0 ** i) * 1e-18 for i in range(nbin)])
    rmass_2d = rmass[:, None]                                  # (nbin, ngroup)
    dm = jnp.asarray([0.5 * rmass[i] for i in range(nbin)])
    dm_2d = dm[:, None]                                        # (nbin, ngroup)
    # rup_wet / rlow_wet must be (nz, nbin, ngroup)
    rup_wet = ((rmass_2d * 1.3 / rho) ** (1.0 / 3.0))[None, :, :]
    rlow_wet = ((rmass_2d * 0.7 / rho) ** (1.0 / 3.0))[None, :, :]

    # Atmospheric state (single level)
    t = jnp.asarray([260.0])
    pvapl = 1000.0  # dyn/cm^2 placeholder
    rh = 1.05
    h2o_cgs = rh * pvapl * float(WTMOL_H2O) / (float(BK) * 260.0 * float(AVG))
    gc = jnp.asarray([[h2o_cgs * 1.0]])
    pc = jnp.zeros((nz, nbin, nelem)).at[0, 3, 0].set(100.0)

    # Growth kernels (tiny so step is non-trivial but bounded)
    gro = jnp.ones((nz, nbin, ngroup)) * 1e-12
    gro1 = jnp.zeros((nz, nbin, ngroup))
    gro2 = jnp.zeros((nz, nbin, ngroup))
    # akelvin / akelvini are (nz, ngas), not (nz, nbin, ngroup)
    akelvin = jnp.zeros((nz, ngas))
    akelvini = jnp.zeros((nz, ngas))
    diffus = jnp.ones((nz,)) * 0.1

    # Density, vertical metric, latent heat
    rhoa = jnp.asarray([1.2e-3])
    zmet = jnp.asarray([1.0])
    rlhe = jnp.ones((nz, ngas)) * 2.5e10
    rlhm = jnp.zeros((nz, ngas))

    # PPM coefficients expected by growevapl:
    #   pratt (3, nbin, ngroup), prat (4, nbin, ngroup),
    #   pden1 (nbin, ngroup),    palr (4, ngroup)
    pratt = jnp.ones((3, nbin, ngroup))
    prat = jnp.ones((4, nbin, ngroup))
    pden1 = jnp.ones((nbin, ngroup))
    palr = jnp.ones((4, ngroup))

    # Static config
    is_ice_arr = (False,)
    igrowgas_arr = (0,)
    ienconc_arr = (0,)
    igroup_arr = (0,)
    gwtmol_arr = (float(WTMOL_H2O),)

    return dict(
        pc=pc, gc=gc, t=t, iz=0, dtime=60.0,
        rhoa=rhoa, zmet=zmet, rlhe=rlhe, rlhm=rlhm, diffus=diffus,
        akelvin=akelvin, akelvini=akelvini,
        gro=gro, gro1=gro1, gro2=gro2,
        rup_wet=rup_wet, rmass_2d=rmass_2d, dm_2d=dm_2d, rlow_wet=rlow_wet,
        pratt=pratt, prat=prat, pden1=pden1, palr=palr,
        is_ice_arr=is_ice_arr, igrowgas_arr=igrowgas_arr,
        ienconc_arr=ienconc_arr, igroup_arr=igroup_arr,
        gwtmol_arr=gwtmol_arr,
        nbin=nbin, ngroup=ngroup, ngas=ngas, nelem=nelem,
    )


def test_newstate_clearsky_matches_newstate_calc_growth():
    """With do_incloud=False, newstate must delegate to
    newstate_calc_growth and return identical output."""
    cfg = _growth_cfg()

    pc_a, gc_a, t_a, rl_a, n_a = newstate(do_incloud=False, **cfg)

    # newstate_calc_growth's kwarg is `dtime_orig` — rebuild the dict
    # so we don't double-pass `dtime`.
    dtime_val = cfg.pop("dtime")
    direct_cfg = dict(cfg, dtime_orig=dtime_val)
    pc_b, gc_b, t_b, rl_b, n_b = newstate_calc_growth(**direct_cfg)

    assert jnp.allclose(pc_a, pc_b)
    assert jnp.allclose(gc_a, gc_b)
    assert jnp.allclose(t_a, t_b)
    assert abs(float(rl_a) - float(rl_b)) < 1e-12
    assert int(n_a) == int(n_b)


def test_newstate_incloud_raises():
    cfg = _growth_cfg()
    with pytest.raises(NotImplementedError, match="in-cloud"):
        newstate(do_incloud=True, **cfg)


def test_newstate_preserves_mass_balance():
    """H2O total (gas + condensate) must be preserved across the
    call (modulo tsolve latent-heat recoupling and floor clipping)."""
    cfg = _growth_cfg()
    rmass_2d = cfg["rmass_2d"]
    # Start: mass in gas only (pc negligible)
    pc = cfg["pc"]
    pc_mass_before = float(jnp.sum(pc[0, :, 0] * rmass_2d[:, 0]))
    gc_mass_before = float(cfg["gc"][0, 0]) * float(cfg["zmet"][0])
    total_before = pc_mass_before + gc_mass_before

    pc_new, gc_new, *_ = newstate(do_incloud=False, **cfg)
    pc_mass_after = float(jnp.sum(pc_new[0, :, 0] * rmass_2d[:, 0]))
    gc_mass_after = float(gc_new[0, 0]) * float(cfg["zmet"][0])
    total_after = pc_mass_after + gc_mass_after

    rel_err = abs(total_after - total_before) / total_before
    assert rel_err < 1e-6
