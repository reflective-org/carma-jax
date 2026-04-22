"""Unit tests for JIT-clean adaptive retry (Phase 9.3).

Parity with the Python-while version, plus JIT compilation check.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from carma.constants import AVG, BK, WTMOL_H2O
from carma.newstate_calc import newstate_calc_growth
from carma.newstate_calc_jit import newstate_calc_growth_jit


def _growth_cfg():
    """Same setup as test_newstate.py, minus the dtime rename."""
    nz, nbin, ngroup, ngas, nelem = 1, 10, 1, 1, 1

    rho = 1.0
    rmass = jnp.asarray([(2.0 ** i) * 1e-18 for i in range(nbin)])
    rmass_2d = rmass[:, None]
    dm = jnp.asarray([0.5 * rmass[i] for i in range(nbin)])
    dm_2d = dm[:, None]
    rup_wet = ((rmass_2d * 1.3 / rho) ** (1.0 / 3.0))[None, :, :]
    rlow_wet = ((rmass_2d * 0.7 / rho) ** (1.0 / 3.0))[None, :, :]

    t = jnp.asarray([260.0])
    pvapl = 1000.0
    rh = 1.05
    h2o_cgs = rh * pvapl * float(WTMOL_H2O) / (float(BK) * 260.0 * float(AVG))
    gc = jnp.asarray([[h2o_cgs * 1.0]])
    pc = jnp.zeros((nz, nbin, nelem)).at[0, 3, 0].set(100.0)

    gro = jnp.ones((nz, nbin, ngroup)) * 1e-12
    gro1 = jnp.zeros((nz, nbin, ngroup))
    gro2 = jnp.zeros((nz, nbin, ngroup))
    akelvin = jnp.zeros((nz, ngas))
    akelvini = jnp.zeros((nz, ngas))
    diffus = jnp.ones((nz,)) * 0.1

    rhoa = jnp.asarray([1.2e-3])
    zmet = jnp.asarray([1.0])
    rlhe = jnp.ones((nz, ngas)) * 2.5e10
    rlhm = jnp.zeros((nz, ngas))

    pratt = jnp.ones((3, nbin, ngroup))
    prat = jnp.ones((4, nbin, ngroup))
    pden1 = jnp.ones((nbin, ngroup))
    palr = jnp.ones((4, ngroup))

    is_ice_arr = (False,)
    igrowgas_arr = (0,)
    ienconc_arr = (0,)
    igroup_arr = (0,)
    gwtmol_arr = (float(WTMOL_H2O),)

    return dict(
        pc=pc, gc=gc, t=t, iz=0, dtime_orig=60.0,
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


def test_jit_version_matches_python_while_version():
    """Both retry implementations must produce the same output on
    a scenario where no retry is needed (rc=OK from the first try)."""
    cfg = _growth_cfg()

    pc_p, gc_p, t_p, rl_p, n_p = newstate_calc_growth(**cfg)
    pc_j, gc_j, t_j, rl_j, n_j = newstate_calc_growth_jit(**cfg)

    assert jnp.allclose(pc_p, pc_j)
    assert jnp.allclose(gc_p, gc_j)
    assert jnp.allclose(t_p, t_j)
    assert abs(float(rl_p) - float(rl_j)) < 1e-12
    assert int(n_p) == int(n_j)


def test_jit_compiles_under_jax_jit():
    """The whole retry wrapper must compile under jax.jit — this is
    the point of the 9.3 port."""
    cfg = _growth_cfg()

    def _call(pc, gc, t, dtime):
        return newstate_calc_growth_jit(
            pc=pc, gc=gc, t=t, iz=cfg["iz"], dtime_orig=dtime,
            rhoa=cfg["rhoa"], zmet=cfg["zmet"], rlhe=cfg["rlhe"],
            rlhm=cfg["rlhm"], diffus=cfg["diffus"],
            akelvin=cfg["akelvin"], akelvini=cfg["akelvini"],
            gro=cfg["gro"], gro1=cfg["gro1"], gro2=cfg["gro2"],
            rup_wet=cfg["rup_wet"], rmass_2d=cfg["rmass_2d"],
            dm_2d=cfg["dm_2d"], rlow_wet=cfg["rlow_wet"],
            pratt=cfg["pratt"], prat=cfg["prat"],
            pden1=cfg["pden1"], palr=cfg["palr"],
            is_ice_arr=cfg["is_ice_arr"], igrowgas_arr=cfg["igrowgas_arr"],
            ienconc_arr=cfg["ienconc_arr"], igroup_arr=cfg["igroup_arr"],
            gwtmol_arr=cfg["gwtmol_arr"],
            nbin=cfg["nbin"], ngroup=cfg["ngroup"],
            ngas=cfg["ngas"], nelem=cfg["nelem"],
        )

    eager = _call(cfg["pc"], cfg["gc"], cfg["t"], 60.0)
    jitted = jax.jit(_call)
    jit_out = jitted(cfg["pc"], cfg["gc"], cfg["t"], 60.0)

    assert jnp.allclose(eager[0], jit_out[0])
    assert jnp.allclose(eager[1], jit_out[1])
    assert jnp.allclose(eager[2], jit_out[2])


def test_jit_mass_conservation():
    """Column H2O mass (gas + condensate) is preserved within 1e-6."""
    cfg = _growth_cfg()
    rmass_2d = cfg["rmass_2d"]
    zmet = cfg["zmet"]

    pc_mass0 = float(jnp.sum(cfg["pc"][0, :, 0] * rmass_2d[:, 0]))
    gc_mass0 = float(cfg["gc"][0, 0]) * float(zmet[0])
    total0 = pc_mass0 + gc_mass0

    pc_n, gc_n, *_ = newstate_calc_growth_jit(**cfg)
    pc_mass1 = float(jnp.sum(pc_n[0, :, 0] * rmass_2d[:, 0]))
    gc_mass1 = float(gc_n[0, 0]) * float(zmet[0])
    total1 = pc_mass1 + gc_mass1

    rel_err = abs(total1 - total0) / total0
    assert rel_err < 1e-6
