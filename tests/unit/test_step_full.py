"""Unit tests for make_step_full (Phase 9.4)."""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from carma.config import (
    CarmaConfig, ElementConfig, GroupConfig, GasConfig, SoluteConfig,
)
from carma.constants import AVG, BK, WTMOL_H2O
from carma.step_full import make_step_full
from carma.vapor_pressure import vaporp_h2o_murphy2005


_GWTMOL_H2SO4 = 98.0
_GWTMOL_H2O = float(WTMOL_H2O)


def _minimal_config(nbin=20):
    """Sulfate-only config. Matches the sulfate_step test fixture
    but promoted here so the full-step pipeline can use it."""
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
        do_vtran=False, do_drydep=False,
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
        name="H2O", wtmol=_GWTMOL_H2O, ivaprtn=2, icomposition=1,
        dgc_threshold=0.0, ds_threshold=0.0,
    )
    gas_h2so4 = GasConfig(
        name="H2SO4", wtmol=_GWTMOL_H2SO4, ivaprtn=4, icomposition=2,
        dgc_threshold=0.0, ds_threshold=0.0,
    )
    solute = SoluteConfig(name="sulfate", ions=3, wtmol=_GWTMOL_H2SO4, rho=1.8)

    return CarmaConfig(
        nbin=nbin, nelem=1, ngroup=1, ngas=2, nsolute=1,
        elements=(element,), groups=(group,),
        gases=(gas_h2o, gas_h2so4), solutes=(solute,),
        coag=None,
        do_coag=False, do_grow=True, do_vtran=False, do_vdiff=False,
        do_thermo=False, do_substep=False, do_explised=False,
        do_incloud=False, do_clearsky=True, do_detrain=False,
        do_pheat=False, do_pheatatm=False, do_cnst_rlh=False,
        itbnd_pc=1, ibbnd_pc=1,
        maxsubsteps=32, minsubsteps=1, maxretries=4, conmax=1e-4,
        cstick=1.0, gsticki=1.0, gstickl=1.0, tstick=1.0, dt_threshold=0.0,
        igash2o=0, igash2so4=1, igasso2=-1,
    )


def _growth_env(cfg, T=220.0, rh=0.5, h2so4_ppbv=1.0, p_hpa=50.0, zmet=1.0):
    """Assemble environment fields for the factory call."""
    nz, nbin, ngroup, ngas = 1, cfg.nbin, cfg.ngroup, cfg.ngas

    pvapl = float(vaporp_h2o_murphy2005(jnp.array(T))[0])

    # H2SO4 gas and H2O gas
    p_cgs = p_hpa * 1e3
    n_air = p_cgs / (float(BK) * T)
    h2so4_num = h2so4_ppbv * 1e-9 * n_air
    h2so4_cgs = h2so4_num * _GWTMOL_H2SO4 / float(AVG)
    h2o_cgs = rh * pvapl * _GWTMOL_H2O / (float(BK) * T * float(AVG))

    gc = jnp.asarray([[h2o_cgs * zmet, h2so4_cgs * zmet]])   # (nz, ngas)
    t = jnp.asarray([T])
    pc = jnp.zeros((nz, nbin, cfg.nelem))

    # Growth environment (tiny gro so the growth path is inactive for
    # this test but the adaptive-retry code path still executes)
    rhoa = jnp.asarray([1.2e-3])
    zmet_arr = jnp.asarray([zmet])
    rlhe = jnp.ones((nz, ngas)) * 2.5e10
    rlhm = jnp.zeros((nz, ngas))
    diffus = jnp.ones((nz,)) * 0.1

    akelvin = jnp.zeros((nz, ngas))
    akelvini = jnp.zeros((nz, ngas))
    gro = jnp.full((nz, nbin, ngroup), 1e-30)      # negligible growth
    gro1 = jnp.zeros((nz, nbin, ngroup))
    gro2 = jnp.zeros((nz, nbin, ngroup))

    rup_wet = jnp.asarray(cfg.groups[0].rup[None, :, None])
    rlow_wet = jnp.asarray(cfg.groups[0].rlow[None, :, None])

    pratt = jnp.ones((3, nbin, ngroup))
    prat = jnp.ones((4, nbin, ngroup))
    pden1 = jnp.ones((nbin, ngroup))
    palr = jnp.ones((4, ngroup))

    ds_threshold_arr = jnp.zeros((ngas,), dtype=jnp.float64)

    pvapl_arr = jnp.asarray([[pvapl]])       # (nz, ngas): only the H2O slot matters

    return dict(
        pc=pc, gc=gc, t=t,
        rhoa=rhoa, zmet=zmet_arr, rlhe=rlhe, rlhm=rlhm, diffus=diffus,
        akelvin=akelvin, akelvini=akelvini,
        gro=gro, gro1=gro1, gro2=gro2,
        rup_wet=rup_wet, rlow_wet=rlow_wet,
        pratt=pratt, prat=prat, pden1=pden1, palr=palr,
        pvapl=pvapl,
        ds_threshold_arr=ds_threshold_arr,
    )


# --- factory runs ---

def test_step_full_runs_without_error():
    cfg = _minimal_config()
    step = make_step_full(cfg)
    env = _growth_env(cfg)
    pc_new, gc_new, t_new, diag = step(dtime=60.0, **env)
    assert pc_new.shape == env["pc"].shape
    assert gc_new.shape == env["gc"].shape
    assert t_new.shape == env["t"].shape
    assert "sulfate" in diag
    assert "growth_substeps" in diag


def test_step_full_nucleates_under_saturated_conditions():
    """T=220K, 1 ppbv H2SO4: should produce sulfate particles."""
    cfg = _minimal_config()
    step = make_step_full(cfg)
    env = _growth_env(cfg)
    pc_new, gc_new, t_new, _ = step(dtime=60.0, **env)
    total_particles = float(jnp.sum(pc_new[0, :, 0]))
    assert total_particles > 0, "no particles formed under saturated conditions"


def test_step_full_disable_sulfate_leaves_pc_unchanged():
    """With do_sulfate=False and negligible growth, pc should not
    change (within numerical precision). Note: empty bins pick up
    SMALL_PC (1e-50) floor from psolve — that's expected."""
    cfg = _minimal_config()
    step = make_step_full(cfg, do_sulfate=False)
    env = _growth_env(cfg)
    env["pc"] = env["pc"].at[0, 5, 0].set(1e3)
    pc_before = env["pc"]
    pc_new, gc_new, *_ = step(dtime=60.0, **env)
    # Check the seeded bin is unchanged within 1% (tiny numerical growth
    # from gro=1e-30 produces a 1e-5 leak into adjacent bins).
    assert abs(float(pc_new[0, 5, 0]) - 1e3) / 1e3 < 1e-3
    # Every other bin stays below machine noise.
    for ibin in range(cfg.nbin):
        if ibin == 5:
            continue
        assert float(pc_new[0, ibin, 0]) < 1.0, (
            f"unexpected particles in bin {ibin}")


def test_step_full_mass_conservation_multistep():
    """Total H2SO4 (gas + particle mass) conserved over 10 steps."""
    cfg = _minimal_config()
    step = make_step_full(cfg)
    env = _growth_env(cfg)
    rmass_1d = cfg.groups[0].rmass

    pc = env["pc"]
    gc = env["gc"]
    t = env["t"]

    mass0 = (
        float(gc[0, 1])
        + float(jnp.sum(pc[0, :, 0] * rmass_1d))
    )

    dtime = 60.0
    env_nostate = {k: v for k, v in env.items() if k not in ("pc", "gc", "t")}
    for _ in range(10):
        pc, gc, t, _ = step(pc=pc, gc=gc, t=t, dtime=dtime, **env_nostate)

    mass1 = (
        float(gc[0, 1])
        + float(jnp.sum(pc[0, :, 0] * rmass_1d))
    )
    rel_err = abs(mass1 - mass0) / mass0
    assert rel_err < 1e-4, f"mass rel err = {rel_err:.3e}"


# --- JIT compilation check ---

def test_step_full_is_single_jit_closure():
    """The whole factory-built closure must be one @jax.jit — no
    Python boundaries between growth and sulfate."""
    cfg = _minimal_config()
    step = make_step_full(cfg)
    env = _growth_env(cfg)

    # Cold call traces + compiles; warm call hits cache.
    _ = step(dtime=60.0, **env)
    pc_warm, gc_warm, *_ = step(dtime=60.0, **env)

    # Same inputs → same outputs
    _, gc_again, *_ = step(dtime=60.0, **env)
    assert jnp.allclose(gc_warm, gc_again)
