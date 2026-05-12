"""Unit tests for make_column_step_full (Phase 12).

Validates three properties of the column step factory:

1. ``nz1_identity``: NZ=1, do_vtran=False → output is bit-identical to
   a direct ``step_full_faithful`` call with the same inputs.
2. ``transport_moves_particles``: NZ=5, do_vtran=True, constant fall
   velocity → particles sediment downward as expected.
3. ``mass_conservation``: NZ=5 with both transport + chemistry active
   over multiple steps → total H₂SO₄ (gas + particle) is conserved.
"""
import math

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from carma.column_step import make_column_step_full
from carma.config import (
    CarmaConfig, CoagConfig, ElementConfig, GasConfig, GroupConfig,
    SoluteConfig,
)
from carma.constants import AVG, BK, RHO_W, WTMOL_H2O
from carma.enums import BoundaryCondition
from carma.precision import DTYPE
from carma.prestep import prestep
from carma.step_full_faithful import make_step_full_faithful
from carma.vapor_pressure import vaporp_h2o_murphy2005

_GWTMOL_H2SO4 = 98.0
_GWTMOL_H2O = float(WTMOL_H2O)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _minimal_config(do_vtran: bool = False):
    """Minimal sulfate-only config, matching the jax_ensemble fixture."""
    nbin = 20
    rho = 1.8
    rmin_cm = 1e-7
    rmrat = 2.0
    vmin = (4.0 / 3.0) * np.pi * rmin_cm**3 * rho
    rmass = vmin * rmrat ** np.arange(nbin)
    rmassup = rmass * rmrat**0.5
    r = (3.0 * rmass / (4.0 * np.pi * rho)) ** (1.0 / 3.0)

    group = GroupConfig(
        name="sulfate", ishape=1, ienconc=0,
        is_ice=False, is_cloud=False, is_sulfate=True,
        do_vtran=do_vtran, do_drydep=False,
        ifallrtn=1, irhswell=3, rmrat=rmrat, eshape=1.0, rmin=rmin_cm,
        r=jnp.asarray(r), rmass=jnp.asarray(rmass),
        vol=jnp.asarray(rmass / rho),
        dr=jnp.asarray(r * 0.1), dm=jnp.asarray(rmass * 0.1),
        rmassup=jnp.asarray(rmassup),
        rup=jnp.asarray(r * 1.2), rlow=jnp.asarray(r * 0.8),
        rrat=jnp.ones(nbin), rprat=jnp.ones(nbin), arat=jnp.ones(nbin),
    )
    element = ElementConfig(
        name="sulfate_num", rho=jnp.full((nbin,), rho), igroup=0,
        itype=2, icomposition=0, isolute=0, kappa=0.65,
    )
    gas_h2o = GasConfig(
        name="H2O", wtmol=_GWTMOL_H2O,
        ivaprtn=2, icomposition=1, dgc_threshold=0.0, ds_threshold=0.0,
    )
    gas_h2so4 = GasConfig(
        name="H2SO4", wtmol=_GWTMOL_H2SO4,
        ivaprtn=4, icomposition=2, dgc_threshold=0.0, ds_threshold=0.0,
    )
    solute = SoluteConfig(name="sulfate", ions=3, wtmol=_GWTMOL_H2SO4, rho=1.8)

    return CarmaConfig(
        nbin=nbin, nelem=1, ngroup=1, ngas=2, nsolute=1,
        elements=(element,), groups=(group,),
        gases=(gas_h2o, gas_h2so4), solutes=(solute,),
        coag=None,
        do_coag=False, do_grow=True, do_vtran=do_vtran, do_vdiff=False,
        do_thermo=False, do_substep=False, do_explised=False,
        do_incloud=False, do_clearsky=True, do_detrain=False,
        do_pheat=False, do_pheatatm=False, do_cnst_rlh=False,
        itbnd_pc=int(BoundaryCondition.I_FIXED_CONC),
        ibbnd_pc=int(BoundaryCondition.I_FIXED_CONC),
        maxsubsteps=4, minsubsteps=1, maxretries=4, conmax=1e-4,
        cstick=1.0, gsticki=1.0, gstickl=1.0, tstick=1.0, dt_threshold=0.0,
        igash2o=0, igash2so4=1, igasso2=-1,
    )


def _build_env(cfg, nz, T_arr, p_cgs_arr, gc_arr, zmet_arr):
    """Build minimal chemistry env for test — mimics ensemble runner structure."""
    nbin = cfg.nbin
    ngroup = cfg.ngroup
    ngas = cfg.ngas

    rhoa = p_cgs_arr / (float(BK) * T_arr / _GWTMOL_H2O * float(AVG))
    rhoa = jnp.asarray(rhoa * zmet_arr)   # g/cm²/z ≈ rhoa × zmet

    ngas_arr = jnp.zeros((nz, ngas), dtype=DTYPE)
    rlhe = jnp.ones((nz, ngas), dtype=DTYPE) * 2.5e10
    rlhm = jnp.zeros((nz, ngas), dtype=DTYPE)
    akelvin = jnp.zeros((nz, ngas), dtype=DTYPE)
    akelvini = jnp.zeros((nz, ngas), dtype=DTYPE)
    gro = jnp.full((nz, nbin, ngroup), 1e-30, dtype=DTYPE)
    gro1 = jnp.zeros((nz, nbin, ngroup), dtype=DTYPE)
    rup_wet = jnp.asarray(
        jnp.broadcast_to(cfg.groups[0].rup[None, :, None], (nz, nbin, ngroup)),
        dtype=DTYPE)
    ckernel = jnp.zeros((nbin, nbin, ngroup, ngroup), dtype=DTYPE)
    pconmax = jnp.zeros((nz, ngroup), dtype=DTYPE)
    ds_threshold_arr = jnp.zeros((ngas,), dtype=DTYPE)

    pratt = jnp.ones((3, nbin, ngroup), dtype=DTYPE)
    prat = jnp.ones((4, nbin, ngroup), dtype=DTYPE)
    pden1 = jnp.ones((nbin, ngroup), dtype=DTYPE)
    palr = jnp.ones((4, ngroup), dtype=DTYPE)

    return dict(
        rhoa=jnp.asarray(rhoa, dtype=DTYPE),
        zmet=jnp.asarray(zmet_arr, dtype=DTYPE),
        akelvin=akelvin, akelvini=akelvini,
        gro=gro, gro1=gro1, rup_wet=rup_wet,
        rlhe=rlhe, rlhm=rlhm,
        ckernel=ckernel, pconmax=pconmax,
        ds_threshold_arr=ds_threshold_arr,
    ), (pratt, prat, pden1, palr)


def _build_transport_arrays(cfg, nz, dz_cm=1.0e4, vfall_cm_s=10.0):
    """Synthetic transport arrays: constant fall velocity, no diffusion."""
    nbin = cfg.nbin
    ngroup = cfg.ngroup
    vf = jnp.full((nz + 1, nbin, ngroup), vfall_cm_s, dtype=DTYPE)
    dkz = jnp.zeros((nz + 1, nbin, ngroup), dtype=DTYPE)
    vd = jnp.zeros((nbin, ngroup), dtype=DTYPE)
    dz = jnp.full((nz,), dz_cm, dtype=DTYPE)
    zc = jnp.arange(nz, dtype=DTYPE) * dz_cm + dz_cm / 2.0
    zl = jnp.arange(nz + 1, dtype=DTYPE) * dz_cm
    rhoa_tr = jnp.full((nz,), 1.2e-3, dtype=DTYPE)
    zmet = jnp.ones((nz,), dtype=DTYPE)
    pc_topbnd = jnp.zeros((nbin, 1), dtype=DTYPE)
    pc_botbnd = jnp.zeros((nbin, 1), dtype=DTYPE)
    ftoppart = jnp.zeros((nbin, 1), dtype=DTYPE)
    fbotpart = jnp.zeros((nbin, 1), dtype=DTYPE)
    return dict(vf=vf, dkz=dkz, vd=vd, dz=dz, zc=zc, zl=zl,
                rhoa=rhoa_tr, zmet=zmet,
                pc_topbnd=pc_topbnd, pc_botbnd=pc_botbnd,
                ftoppart=ftoppart, fbotpart=fbotpart)


# ---------------------------------------------------------------------------
# Test 1: NZ=1 identity
# ---------------------------------------------------------------------------

def test_column_step_nz1_matches_step_full_faithful():
    """NZ=1, do_vtran=False: column_step output == bare step_full_faithful."""
    cfg = _minimal_config(do_vtran=False)
    nz = 1
    nbin = cfg.nbin

    T_arr = np.full(nz, 220.0)
    p_cgs_arr = np.full(nz, 50.0e3)
    zmet_arr = np.ones(nz)
    pvapl = float(vaporp_h2o_murphy2005(jnp.array(220.0))[0])
    h2o_cgs = 0.5 * pvapl * _GWTMOL_H2O / (float(BK) * 220.0 * float(AVG))
    gc = jnp.asarray(
        np.array([[h2o_cgs * zmet_arr[0], 1e-15 * zmet_arr[0]]]), dtype=DTYPE)
    t = jnp.asarray(T_arr, dtype=DTYPE)
    pc = jnp.zeros((nz, nbin, cfg.nelem), dtype=DTYPE)
    env, ppm = _build_env(cfg, nz, T_arr, p_cgs_arr, gc, zmet_arr)

    itype_arr = jnp.array([e.itype for e in cfg.elements])
    ienconc_arr = jnp.array([g.ienconc for g in cfg.groups])
    igelem_arr = jnp.array([e.igroup for e in cfg.elements])
    rmass_2d = jnp.stack([g.rmass for g in cfg.groups], axis=1)

    dtime = 60.0
    pc, gc, t, pcl, gcl, d_gc, d_t, pconmax = prestep(
        pc, gc, t, pc, gc, t, env["zmet"],
        itype_arr, ienconc_arr, igelem_arr, rmass_2d,
        do_substep=True, do_coag=False,
    )
    env["pconmax"] = pconmax

    tr = _build_transport_arrays(cfg, nz)

    # Direct step_full_faithful call (reference)
    # step_full_faithful takes full-column arrays (NZ, ...) + iz selector
    cell_step = make_step_full_faithful(cfg, ppm_coefs=ppm)
    pc_ref, gc_ref, t_ref, _ = cell_step(
        pc, gc, t, dtime,
        pcl=pcl, gcl=gcl, told=t, d_gc=d_gc, d_t=d_t, iz=0,
        **env,
    )

    # Column step (should be identical for NZ=1, no transport)
    col_step = make_column_step_full(cfg, ppm_coefs=ppm)
    pc_col, gc_col, t_col, sedflux, diags = col_step(
        pc, gc, t, dtime, **tr,
        pcl_col=pcl, gcl_col=gcl, told_col=t, d_gc_col=d_gc, d_t_col=d_t,
        env=env,
    )

    np.testing.assert_array_equal(np.asarray(pc_col), np.asarray(pc_ref),
                                   err_msg="pc mismatch")
    np.testing.assert_array_equal(np.asarray(gc_col), np.asarray(gc_ref),
                                   err_msg="gc mismatch")
    np.testing.assert_array_equal(np.asarray(t_col), np.asarray(t_ref),
                                   err_msg="t mismatch")
    assert float(jnp.sum(sedflux)) == 0.0


# ---------------------------------------------------------------------------
# Test 2: Transport moves particles
# ---------------------------------------------------------------------------

def test_column_step_transport_moves_particles():
    """NZ=5, do_vtran=True: particles seeded at mid-column sediment downward."""
    nz = 5
    cfg = _minimal_config(do_vtran=True)
    nbin = cfg.nbin
    T_arr = np.full(nz, 220.0)
    p_cgs_arr = np.full(nz, 50.0e3)
    zmet_arr = np.ones(nz)
    h2o_cgs = 1.0e-14
    gc = jnp.asarray(np.tile([[h2o_cgs, 0.0]], (nz, 1)), dtype=DTYPE)
    t = jnp.asarray(T_arr, dtype=DTYPE)

    # Seed particles only at level 2 (middle of 5-level column)
    pc = jnp.zeros((nz, nbin, cfg.nelem), dtype=DTYPE)
    pc = pc.at[2, 5, 0].set(100.0)   # one bin, middle level

    env, ppm = _build_env(cfg, nz, T_arr, p_cgs_arr, gc, zmet_arr)

    itype_arr = jnp.array([e.itype for e in cfg.elements])
    ienconc_arr = jnp.array([g.ienconc for g in cfg.groups])
    igelem_arr = jnp.array([e.igroup for e in cfg.elements])
    rmass_2d = jnp.stack([g.rmass for g in cfg.groups], axis=1)

    dtime = 300.0
    pc_in, gc_in, t_in, pcl, gcl, d_gc, d_t, pconmax = prestep(
        pc, gc, t, pc, gc, t, env["zmet"],
        itype_arr, ienconc_arr, igelem_arr, rmass_2d,
        do_substep=True, do_coag=False,
    )
    env["pconmax"] = pconmax

    tr = _build_transport_arrays(cfg, nz, dz_cm=1.0e5, vfall_cm_s=100.0)
    col_step = make_column_step_full(cfg, ppm_coefs=ppm)
    pc_out, gc_out, t_out, sedflux, diags = col_step(
        pc_in, gc_in, t_in, dtime, **tr,
        pcl_col=pcl, gcl_col=gcl, told_col=t_in, d_gc_col=d_gc, d_t_col=d_t,
        env=env,
    )
    pc_out = np.asarray(pc_out)
    pc_in_np = np.asarray(pc_in)

    # Particles must leave level 2 (they fall downward toward index 0)
    assert float(pc_out[2, 5, 0]) < float(pc_in_np[2, 5, 0]), (
        "particles did not leave source level")
    # Total column count conserved + sedflux (mass to surface)
    total_in = float(np.sum(pc_in_np[:, 5, 0]))
    total_out = float(np.sum(pc_out[:, 5, 0]))
    sedimented = float(np.asarray(sedflux)[5, 0]) * dtime   # flux × dt = count
    assert abs(total_in - total_out - sedimented) / max(total_in, 1e-30) < 1e-6, (
        "mass conservation violated in transport step")


# ---------------------------------------------------------------------------
# Test 3: Multi-level mass conservation
# ---------------------------------------------------------------------------

def test_column_step_mass_conservation_multistep():
    """NZ=3, both transport + chemistry: H₂SO₄ (gas+particle) conserved."""
    nz = 3
    cfg = _minimal_config(do_vtran=True)
    nbin = cfg.nbin
    T_arr = np.full(nz, 220.0)
    p_cgs_arr = np.full(nz, 50.0e3)
    zmet_arr = np.ones(nz)
    pvapl = float(vaporp_h2o_murphy2005(jnp.array(220.0))[0])
    h2o_cgs = 0.5 * pvapl * _GWTMOL_H2O / (float(BK) * 220.0 * float(AVG))
    h2so4_cgs = 5e-15

    gc = jnp.asarray(
        np.tile([[h2o_cgs, h2so4_cgs]], (nz, 1)), dtype=DTYPE)
    t = jnp.asarray(T_arr, dtype=DTYPE)
    pc = jnp.zeros((nz, nbin, cfg.nelem), dtype=DTYPE)

    env, ppm = _build_env(cfg, nz, T_arr, p_cgs_arr, gc, zmet_arr)

    itype_arr = jnp.array([e.itype for e in cfg.elements])
    ienconc_arr = jnp.array([g.ienconc for g in cfg.groups])
    igelem_arr = jnp.array([e.igroup for e in cfg.elements])
    rmass_2d = jnp.stack([g.rmass for g in cfg.groups], axis=1)
    rmass_1d = np.asarray(cfg.groups[0].rmass)
    dz_val = 1.0e5   # 1 km layers [cm] — uniform

    dtime = 300.0
    col_step = make_column_step_full(cfg, ppm_coefs=ppm)
    tr = _build_transport_arrays(cfg, nz, dz_cm=dz_val, vfall_cm_s=10.0)

    def _h2so4_mass_g_per_cm2(pc_, gc_):
        """H₂SO₄ column mass [g/cm²] = (gas + particle) × dz summed over levels."""
        gas_col = float(jnp.sum(gc_[:, 1])) * dz_val
        par_col = float(jnp.sum(
            pc_[:, :, 0] * jnp.asarray(rmass_1d, dtype=DTYPE)[None, :])) * dz_val
        return gas_col + par_col

    mass0 = _h2so4_mass_g_per_cm2(pc, gc)
    cumul_sed_g_cm2 = 0.0

    for _ in range(5):
        pc, gc, t, pcl, gcl, d_gc, d_t, pconmax = prestep(
            pc, gc, t, pc, gc, t, env["zmet"],
            itype_arr, ienconc_arr, igelem_arr, rmass_2d,
            do_substep=True, do_coag=False,
        )
        env["pconmax"] = pconmax
        pc, gc, t, sedflux, _ = col_step(
            pc, gc, t, dtime, **tr,
            pcl_col=pcl, gcl_col=gcl, told_col=t, d_gc_col=d_gc, d_t_col=d_t,
            env=env,
        )
        # sedflux[ibin, ielem] in #/cm²/s; × rmass[ibin] → g/cm²/s; × dtime → g/cm²
        cumul_sed_g_cm2 += float(
            jnp.dot(sedflux[:, 0], jnp.asarray(rmass_1d, dtype=DTYPE))) * dtime

    mass1 = _h2so4_mass_g_per_cm2(pc, gc)
    rel_err = abs(mass0 - mass1 - cumul_sed_g_cm2) / max(mass0, 1e-30)
    assert rel_err < 1e-4, f"H₂SO₄ mass not conserved: rel err = {rel_err:.3e}"
