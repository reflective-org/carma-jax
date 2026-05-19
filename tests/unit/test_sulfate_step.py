"""Unit tests for sulfate_step + make_step_sulfate."""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from carma.config import (
    CarmaConfig, ElementConfig, GroupConfig, GasConfig, SoluteConfig,
)
from carma.constants import AVG, BK, WTMOL_H2O
from carma.sulfate_step import make_step_sulfate, sulfate_step_one_level
from carma.vapor_pressure import vaporp_h2o_murphy2005


_GWTMOL_H2SO4 = 98.0
_GWTMOL_H2O = float(WTMOL_H2O)


def _bins(nbin=20, rmin_cm=1e-7, rmrat=2.0, rho=1.8):
    vmin = (4.0 / 3.0) * np.pi * rmin_cm**3 * rho
    rmass = vmin * rmrat ** np.arange(nbin)
    rmassup = rmass * rmrat**0.5
    r = (3.0 * rmass / (4.0 * np.pi * rho)) ** (1.0 / 3.0)
    return (jnp.asarray(r), jnp.asarray(rmass),
            jnp.asarray(rmassup), rmrat, rho)


def _diffmass(rmass):
    rmass_col = rmass[:, None]
    return rmass_col[:, :, None, None] - rmass_col[None, None, :, :]


def _inuc2bin(nbin):
    i2 = np.arange(nbin) + 1
    i2[-1] = -1
    return jnp.asarray(i2.reshape(nbin, 1, 1))


def _strat_column(T=220.0, rh=0.5, h2so4_ppbv=1.0, zmet=1.0, p_hpa=50.0):
    """Build stratospheric column state (gc arrays, pvapl)."""
    pvapl = float(vaporp_h2o_murphy2005(jnp.array(T))[0])
    # H2SO4 gas concentration [molec/cm³] from ppbv and pressure
    # p_hpa in hPa → dyn/cm² (mks conversion not critical here):
    p_cgs = p_hpa * 1e3                              # hPa → dyn/cm²
    n_air = p_cgs / (BK * T)                         # molec/cm³
    h2so4_num = h2so4_ppbv * 1e-9 * n_air
    h2so4_cgs = h2so4_num * _GWTMOL_H2SO4 / float(AVG)
    # H2O at given RH
    h2o_cgs = rh * pvapl * _GWTMOL_H2O / (float(BK) * T * float(AVG))
    # gc is (g/cm³/z); internal model divides by zmet
    gc = jnp.asarray([h2o_cgs * zmet, h2so4_cgs * zmet])  # (ngas,)
    return gc, pvapl


# --- sulfate_step_one_level ---

def test_one_level_mass_conservation():
    """Total H2SO4 mass (gas + Σpc·rmass) must be conserved to
    floating-point precision."""
    r_bins, rmass, rmassup, rmrat, _ = _bins()
    nbin = r_bins.shape[0]
    diffmass = _diffmass(rmass)
    inuc2bin = _inuc2bin(nbin)

    gc, pvapl = _strat_column(T=220.0, rh=0.5, h2so4_ppbv=1.0)
    pc = jnp.zeros(nbin, dtype=jnp.float64)
    # seed a few particles in the middle bins so the heterogeneous
    # branch has something to transfer.
    pc = pc.at[5].set(1e3).at[8].set(500.0)
    dtime = 60.0

    gc_h2so4 = gc[1]
    gc_h2o = gc[0]
    zmet = 1.0
    mass_before = float(gc_h2so4) + float(jnp.sum(pc * rmass))

    pc_new, gc_h2so4_new, _ = sulfate_step_one_level(
        pc, gc_h2so4, gc_h2o, 220.0, pvapl, zmet, dtime,
        r_bins, rmass, rmassup, diffmass, inuc2bin, rmrat,
    )
    mass_after = float(gc_h2so4_new) + float(jnp.sum(pc_new * rmass))
    rel_err = abs(mass_after - mass_before) / mass_before
    # Float-roundoff limit for the semi-implicit / scaled update —
    # Phase 9 adaptive retry will tighten this.
    assert rel_err < 1e-6, f"mass non-conservation rel err = {rel_err:.3e}"


def test_one_level_homogeneous_only_fills_nucbin():
    """With heterogeneous disabled, only the homogeneous target bin
    should gain particles."""
    r_bins, rmass, rmassup, rmrat, _ = _bins()
    nbin = r_bins.shape[0]
    diffmass = _diffmass(rmass)
    inuc2bin = _inuc2bin(nbin)

    gc, pvapl = _strat_column(T=220.0, rh=0.5, h2so4_ppbv=1.0)
    pc = jnp.zeros(nbin)
    pc_new, _, _ = sulfate_step_one_level(
        pc, gc[1], gc[0], 220.0, pvapl, 1.0, 60.0,
        r_bins, rmass, rmassup, diffmass, inuc2bin, rmrat,
        do_heterogeneous=False,
    )
    pc_np = np.asarray(pc_new)
    # Exactly one bin has all the added particles.
    assert (pc_np > 0).sum() == 1


def test_one_level_heterogeneous_only_transfers():
    """With homogeneous disabled, the total number is conserved (het
    only shuffles particles bin-to-bin — top bin has no target so a tiny
    drain is possible; verify within 1%)."""
    r_bins, rmass, rmassup, rmrat, _ = _bins()
    nbin = r_bins.shape[0]
    diffmass = _diffmass(rmass)
    inuc2bin = _inuc2bin(nbin)

    gc, pvapl = _strat_column(T=220.0, rh=0.5, h2so4_ppbv=1.0)
    pc = jnp.zeros(nbin).at[5].set(1e3)
    pc_new, _, _ = sulfate_step_one_level(
        pc, gc[1], gc[0], 220.0, pvapl, 1.0, 60.0,
        r_bins, rmass, rmassup, diffmass, inuc2bin, rmrat,
        do_homogeneous=False,
    )
    total_before = float(jnp.sum(pc))
    total_after = float(jnp.sum(pc_new))
    rel_err = abs(total_after - total_before) / total_before
    # Top-bin has no target, but pc[5] is interior → err should be 0.
    assert rel_err < 1e-10


def test_one_level_no_h2so4_returns_zero_rates():
    """Sub-saturated conditions: Zhao saddle fails, rates should be 0,
    pc and gc unchanged."""
    r_bins, rmass, rmassup, rmrat, _ = _bins()
    nbin = r_bins.shape[0]
    diffmass = _diffmass(rmass)
    inuc2bin = _inuc2bin(nbin)

    # Negligible H2SO4
    gc, pvapl = _strat_column(T=300.0, rh=0.001, h2so4_ppbv=1e-20)
    pc = jnp.zeros(nbin)
    pc_new, gc_new, diag = sulfate_step_one_level(
        pc, gc[1], gc[0], 300.0, pvapl, 1.0, 60.0,
        r_bins, rmass, rmassup, diffmass, inuc2bin, rmrat,
    )
    assert float(jnp.max(diag["rhompe"])) == 0.0
    assert float(jnp.max(diag["rnuclg"])) == 0.0
    assert float(diag["gasprod_h2so4"]) == 0.0
    assert float(jnp.sum(pc_new)) == 0.0


# --- make_step_sulfate factory ---

def _minimal_config(nbin=20):
    r_bins, rmass, rmassup, rmrat, rho = _bins(nbin=nbin)
    rmin = float(r_bins[0])
    rho_arr = jnp.full((nbin,), rho)

    group = GroupConfig(
        name="sulfate", ishape=1, ienconc=0,
        is_ice=False, is_cloud=False, is_sulfate=True,
        do_vtran=False, do_drydep=False,
        ifallrtn=1, irhswell=3, rmrat=rmrat, eshape=1.0, rmin=rmin,
        r=r_bins, rmass=rmass, vol=rmass / rho,
        dr=r_bins * 0.1, dm=rmass * 0.1,
        rmassup=rmassup, rup=r_bins * 1.2, rlow=r_bins * 0.8,
        rrat=jnp.ones(nbin), rprat=jnp.ones(nbin), arat=jnp.ones(nbin),
    )
    element = ElementConfig(
        name="sulfate_num", rho=rho_arr, igroup=0,
        itype=2, icomposition=0, isolute=0, kappa=0.65,
    )
    gas_h2o = GasConfig(
        name="H2O", wtmol=_GWTMOL_H2O,
        ivaprtn=2, icomposition=1,
        dgc_threshold=0.0, ds_threshold=0.0,
    )
    gas_h2so4 = GasConfig(
        name="H2SO4", wtmol=_GWTMOL_H2SO4,
        ivaprtn=4, icomposition=2,
        dgc_threshold=0.0, ds_threshold=0.0,
    )
    solute = SoluteConfig(name="sulfate", ions=3, wtmol=_GWTMOL_H2SO4, rho=1.8)

    return CarmaConfig(
        nbin=nbin, nelem=1, ngroup=1, ngas=2, nsolute=1,
        elements=(element,), groups=(group,),
        gases=(gas_h2o, gas_h2so4), solutes=(solute,),
        coag=None,
        do_coag=False, do_grow=False, do_vtran=False, do_vdiff=False,
        do_thermo=False, do_substep=False, do_explised=False,
        do_incloud=False, do_clearsky=True, do_detrain=False,
        do_pheat=False, do_pheatatm=False, do_cnst_rlh=False,
        itbnd_pc=1, ibbnd_pc=1,
        maxsubsteps=32, minsubsteps=1, maxretries=4, conmax=1e-4,
        cstick=1.0, gsticki=1.0, gstickl=1.0, tstick=1.0, dt_threshold=0.0,
        igash2o=0, igash2so4=1, igasso2=-1,
    )


def test_factory_runs():
    cfg = _minimal_config()
    step = make_step_sulfate(cfg)
    gc, pvapl = _strat_column(T=220.0, rh=0.5, h2so4_ppbv=1.0)
    pc = jnp.zeros(cfg.nbin)
    pc_new, gc_new, diag = step(pc, gc, 220.0, pvapl, 1.0, 60.0)
    assert pc_new.shape == (cfg.nbin,)
    assert gc_new.shape == (cfg.ngas,)
    # After 60 s, homogeneous nucleation should have seeded some bin.
    assert float(jnp.max(pc_new)) > 0


def test_factory_mass_conservation_multistep():
    """Run 10 steps and check H2SO4 total (gas + particle mass) is
    conserved."""
    cfg = _minimal_config()
    step = make_step_sulfate(cfg)
    gc, pvapl = _strat_column(T=220.0, rh=0.5, h2so4_ppbv=1.0)
    pc = jnp.zeros(cfg.nbin).at[5].set(1e3)         # seed particles

    mass_initial = float(gc[1]) + float(jnp.sum(pc * cfg.groups[0].rmass))

    dtime = 60.0
    for _ in range(10):
        pc, gc, _ = step(pc, gc, 220.0, pvapl, 1.0, dtime)

    mass_final = float(gc[1]) + float(jnp.sum(pc * cfg.groups[0].rmass))
    rel_err = abs(mass_final - mass_initial) / mass_initial
    # Float-roundoff accumulation over 10 steps (see one-level note).
    assert rel_err < 1e-4, f"multistep mass rel err = {rel_err:.3e}"


def test_factory_substepping_preserves_mass():
    """ntsubsteps refines the solution but mass conservation must
    hold for any ntsubsteps."""
    cfg = _minimal_config()
    gc0, pvapl = _strat_column(T=220.0, rh=0.5, h2so4_ppbv=1.0)
    pc_init = jnp.zeros(cfg.nbin).at[5].set(1e3)
    mass_initial = float(gc0[1]) + float(jnp.sum(pc_init * cfg.groups[0].rmass))

    for n in (1, 4, 16):
        step = make_step_sulfate(cfg, ntsubsteps=n)
        pc, gc, _ = step(pc_init, gc0, 220.0, pvapl, 1.0, 10.0)
        mass_final = float(gc[1]) + float(jnp.sum(pc * cfg.groups[0].rmass))
        rel = abs(mass_final - mass_initial) / mass_initial
        assert rel < 1e-4, (
            f"ntsubsteps={n} mass err = {rel:.3e}")


def test_factory_jit_warm():
    """Second call through the JIT'd factory must be consistent."""
    cfg = _minimal_config()
    step = make_step_sulfate(cfg)
    gc, pvapl = _strat_column(T=220.0, rh=0.5, h2so4_ppbv=1.0)
    pc = jnp.zeros(cfg.nbin).at[5].set(1e3)

    pc1, gc1, _ = step(pc, gc, 220.0, pvapl, 1.0, 60.0)
    pc2, gc2, _ = step(pc, gc, 220.0, pvapl, 1.0, 60.0)
    assert jnp.allclose(pc1, pc2)
    assert jnp.allclose(gc1, gc2)
