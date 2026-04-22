"""Unit tests for nsubsteps (Phase 8.4)."""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from carma.enums import ElementType, NucProcess
from carma.nsubsteps import aerfreeze_force, nsubsteps


def _base_cfg(ngroup=1, nbin=10, nelem=1, ngas=1):
    """Single-group volatile setup. gro, gro1, supsatl, etc. are tiny
    so the default growth-rate path resolves to minsubsteps."""
    dm = jnp.full((nbin, ngroup), 1e-20)
    gro = jnp.full((1, nbin, ngroup), 1e-3)
    gro1 = jnp.zeros((1, nbin, ngroup))
    pvapl = jnp.asarray([[10.0]])
    pvapi = jnp.asarray([[5.0]])
    supsatl = jnp.asarray([[0.1]])
    supsati = jnp.asarray([[0.1]])
    supsatlold = jnp.zeros((1, ngas))
    scrit = jnp.full((1, nbin, ngroup), 1.0)      # high threshold
    zmet = jnp.asarray([1.0])
    pc = jnp.zeros((1, nbin, nelem))
    pconmax = jnp.full((1, ngroup), 1e6)

    # Config (static arrays)
    ienconc_arr = np.asarray([0])
    itype_arr = np.asarray([int(ElementType.I_VOLATILE)])
    inucgas_arr = np.asarray([-1])                # no nucleation gas
    igrowgas_arr = np.asarray([0])                # grows via gas 0
    nnuc2elem_arr = np.asarray([0])
    inuc2elem_arr = np.zeros((1, 1), dtype=int)
    inucproc_arr = np.zeros((1, 1), dtype=int)
    is_grp_ice_arr = np.asarray([False])

    return dict(
        pc=pc, supsatl=supsatl, supsati=supsati,
        supsatlold=supsatlold, pvapl=pvapl, pvapi=pvapi,
        scrit=scrit, gro=gro, gro1=gro1,
        pconmax=pconmax, zmet=zmet, dm=dm,
        iz=0, minsubsteps=1, maxsubsteps=32, conmax=0.01,
        ngroup=ngroup, nbin=nbin,
        ienconc_arr=ienconc_arr, itype_arr=itype_arr,
        inucgas_arr=inucgas_arr, igrowgas_arr=igrowgas_arr,
        nnuc2elem_arr=nnuc2elem_arr, inuc2elem_arr=inuc2elem_arr,
        inucproc_arr=inucproc_arr, is_grp_ice_arr=is_grp_ice_arr,
    )


def test_do_substep_false_returns_one():
    cfg = _base_cfg()
    n = nsubsteps(dtime_save=60.0, do_substep=False, **cfg)
    assert int(n) == 1


def test_trivial_growth_returns_minsubsteps():
    """Slow growth (dm / dmdt >> dtime_save) → dt_adv unaffected → n = min."""
    cfg = _base_cfg()
    # Slow growth: tiny gro AND large dm
    cfg["gro"] = jnp.full_like(cfg["gro"], 1e-30)
    cfg["dm"] = jnp.full_like(cfg["dm"], 1.0)
    cfg["supsatl"] = jnp.asarray([[1e-10]])       # tiny supersaturation
    n = int(nsubsteps(dtime_save=60.0, **cfg))
    assert n == cfg["minsubsteps"]


def test_fast_growth_returns_higher_count():
    """Make dmdt large; n should exceed minsubsteps."""
    cfg = _base_cfg()
    # dmdt = pvap · ss · g0 / (1 + g0·g1·pvap). Boost g0.
    cfg["gro"] = jnp.full_like(cfg["gro"], 100.0)
    # Give it some particles so ibin_small is small (bin 0)
    cfg["pc"] = cfg["pc"].at[0, 0, 0].set(1e5)
    n = int(nsubsteps(dtime_save=60.0, **cfg))
    assert n > cfg["minsubsteps"]
    assert n <= cfg["maxsubsteps"]


def test_fast_growth_capped_at_maxsubsteps():
    cfg = _base_cfg()
    cfg["gro"] = jnp.full_like(cfg["gro"], 1e20)
    cfg["pc"] = cfg["pc"].at[0, 0, 0].set(1e8)
    n = int(nsubsteps(dtime_save=60.0, **cfg))
    assert n == cfg["maxsubsteps"]


def test_activation_forces_maxsubsteps():
    """Involatile group with drop activation should force maxsubsteps."""
    cfg = _base_cfg()
    cfg["itype_arr"] = np.asarray([int(ElementType.I_INVOLATILE)])
    cfg["inucgas_arr"] = np.asarray([0])
    cfg["nnuc2elem_arr"] = np.asarray([1])
    cfg["inuc2elem_arr"] = np.asarray([[0]])
    cfg["inucproc_arr"] = np.asarray([[int(NucProcess.I_DROPACT)]])
    # Make the activation trigger: supsatl > scrit, enough particles.
    cfg["supsatl"] = jnp.asarray([[2.0]])           # high
    cfg["scrit"] = jnp.full_like(cfg["scrit"], 1.0)
    cfg["pc"] = cfg["pc"].at[0, 0, 0].set(1e10)     # plenty
    # pconmax tuned so pc/zmet > conmax * pconmax
    cfg["pconmax"] = jnp.asarray([[1e6]])
    n = int(nsubsteps(dtime_save=60.0, **cfg))
    assert n == cfg["maxsubsteps"]


def test_no_particles_returns_minsubsteps():
    """pconmax below FEW_PC → growth path skipped → min."""
    cfg = _base_cfg()
    cfg["pconmax"] = jnp.zeros_like(cfg["pconmax"])
    n = int(nsubsteps(dtime_save=60.0, **cfg))
    assert n == cfg["minsubsteps"]


# --- aerfreeze_force ---

def test_aerfreeze_force_triggers_cold_supersaturated():
    cfg = _base_cfg()
    cfg["inucproc_arr"] = np.asarray([[int(NucProcess.I_AERFREEZE)]])
    cfg["nnuc2elem_arr"] = np.asarray([1])
    cfg["inuc2elem_arr"] = np.asarray([[0]])
    cfg["inucgas_arr"] = np.asarray([0])
    supsati = jnp.asarray([[0.5]])                 # > 0.4
    out = aerfreeze_force(
        supsati=supsati, temp=220.0, iz=0,
        inucgas_arr=cfg["inucgas_arr"], ienconc_arr=cfg["ienconc_arr"],
        nnuc2elem_arr=cfg["nnuc2elem_arr"],
        inuc2elem_arr=cfg["inuc2elem_arr"],
        inucproc_arr=cfg["inucproc_arr"],
        ngroup=cfg["ngroup"],
    )
    assert bool(out)


def test_aerfreeze_force_false_when_warm():
    cfg = _base_cfg()
    cfg["inucproc_arr"] = np.asarray([[int(NucProcess.I_AERFREEZE)]])
    cfg["nnuc2elem_arr"] = np.asarray([1])
    cfg["inuc2elem_arr"] = np.asarray([[0]])
    cfg["inucgas_arr"] = np.asarray([0])
    supsati = jnp.asarray([[0.5]])
    out = aerfreeze_force(
        supsati=supsati, temp=260.0, iz=0,                # > 233.16
        inucgas_arr=cfg["inucgas_arr"], ienconc_arr=cfg["ienconc_arr"],
        nnuc2elem_arr=cfg["nnuc2elem_arr"],
        inuc2elem_arr=cfg["inuc2elem_arr"],
        inucproc_arr=cfg["inucproc_arr"],
        ngroup=cfg["ngroup"],
    )
    assert not bool(out)


def test_aerfreeze_force_false_when_undersaturated():
    cfg = _base_cfg()
    cfg["inucproc_arr"] = np.asarray([[int(NucProcess.I_AERFREEZE)]])
    cfg["nnuc2elem_arr"] = np.asarray([1])
    cfg["inuc2elem_arr"] = np.asarray([[0]])
    cfg["inucgas_arr"] = np.asarray([0])
    supsati = jnp.asarray([[0.3]])                       # <= 0.4
    out = aerfreeze_force(
        supsati=supsati, temp=220.0, iz=0,
        inucgas_arr=cfg["inucgas_arr"], ienconc_arr=cfg["ienconc_arr"],
        nnuc2elem_arr=cfg["nnuc2elem_arr"],
        inuc2elem_arr=cfg["inuc2elem_arr"],
        inucproc_arr=cfg["inucproc_arr"],
        ngroup=cfg["ngroup"],
    )
    assert not bool(out)


# --- JIT ---

def test_nsubsteps_jits():
    cfg = _base_cfg()
    cfg["gro"] = jnp.full_like(cfg["gro"], 10.0)
    cfg["pc"] = cfg["pc"].at[0, 0, 0].set(1e5)

    def _wrap(pc, gro, dtime):
        return nsubsteps(
            pc=pc, supsatl=cfg["supsatl"], supsati=cfg["supsati"],
            supsatlold=cfg["supsatlold"], pvapl=cfg["pvapl"], pvapi=cfg["pvapi"],
            scrit=cfg["scrit"], gro=gro, gro1=cfg["gro1"],
            pconmax=cfg["pconmax"], zmet=cfg["zmet"], dm=cfg["dm"],
            iz=0, dtime_save=dtime,
            minsubsteps=1, maxsubsteps=32, conmax=0.01,
            ngroup=1, nbin=10,
            ienconc_arr=cfg["ienconc_arr"], itype_arr=cfg["itype_arr"],
            inucgas_arr=cfg["inucgas_arr"], igrowgas_arr=cfg["igrowgas_arr"],
            nnuc2elem_arr=cfg["nnuc2elem_arr"],
            inuc2elem_arr=cfg["inuc2elem_arr"],
            inucproc_arr=cfg["inucproc_arr"],
            is_grp_ice_arr=cfg["is_grp_ice_arr"],
        )

    eager = _wrap(cfg["pc"], cfg["gro"], 60.0)
    jit_out = jax.jit(_wrap)(cfg["pc"], cfg["gro"], 60.0)
    assert int(eager) == int(jit_out)
