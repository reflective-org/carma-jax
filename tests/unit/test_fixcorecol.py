"""Unit tests for fixcorecol + coremasscheck (Phase 8.1)."""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from carma.utils.coremasscheck import coremasscheck
from carma.utils.fixcorecol import fixcorecol


def _two_elem_cfg():
    """One group, 2 elements (number + core), 3 bins, 4 vertical levels."""
    nbin, nelem, ngroup = 3, 2, 1
    rmass = jnp.asarray([[1e-20], [2e-20], [4e-20]])            # (nbin, ngroup)
    dz = jnp.ones(4)
    icorelem = np.asarray([[1]])          # element 1 is core in group 0
    ienconc = np.asarray([0])             # number element is 0
    ncore = np.asarray([1])
    return dict(
        nz=4, nbin=nbin, nelem=nelem, ngroup=ngroup,
        rmass=rmass, dz=dz,
        icorelem=icorelem, ienconc=ienconc, ncore=ncore,
    )


# ----- fixcorecol -----

def test_fixcorecol_no_fix_when_healthy():
    """If every level has pc[num]·rmass > total_core, pc is unchanged."""
    cfg = _two_elem_cfg()
    nz, nbin, nelem = cfg["nz"], cfg["nbin"], cfg["nelem"]
    pc = jnp.zeros((nz, nbin, nelem))
    # At every level, bin 0 has 10 particles of mass 1e-20 and core 5e-20
    # → concgas_md = 10*1e-20 - 5e-20 = 5e-20 > 0.
    pc = pc.at[:, 0, 0].set(10.0).at[:, 0, 1].set(5e-20)

    pc_new = fixcorecol(
        pc, cfg["rmass"], cfg["dz"],
        cfg["icorelem"], cfg["ienconc"], cfg["ncore"],
    )
    assert jnp.allclose(pc, pc_new)


def test_fixcorecol_conserves_total_column_mass():
    """Set up a deficit in one level, excess in another; the total
    column mass of pc[iepart]·rmass must be conserved after fix."""
    cfg = _two_elem_cfg()
    nz, nbin, nelem = cfg["nz"], cfg["nbin"], cfg["nelem"]
    rmass = cfg["rmass"]

    # bin 0: levels 0,1 healthy; level 2 has core > num*rmass (deficit).
    pc = jnp.zeros((nz, nbin, nelem))
    pc = pc.at[0, 0, 0].set(20.0).at[0, 0, 1].set(5e-20)   # concgas=15e-20
    pc = pc.at[1, 0, 0].set(8.0).at[1, 0, 1].set(3e-20)    # concgas=5e-20
    pc = pc.at[2, 0, 0].set(1.0).at[2, 0, 1].set(10e-20)   # concgas=-9e-20 (bad)
    pc = pc.at[3, 0, 0].set(5.0).at[3, 0, 1].set(4e-20)    # concgas=1e-20

    total_mass_before = float(jnp.sum(pc[:, 0, 0] * rmass[0, 0] * cfg["dz"]))

    pc_new = fixcorecol(
        pc, rmass, cfg["dz"],
        cfg["icorelem"], cfg["ienconc"], cfg["ncore"],
    )

    total_mass_after = float(jnp.sum(pc_new[:, 0, 0] * rmass[0, 0] * cfg["dz"]))
    rel_err = abs(total_mass_after - total_mass_before) / total_mass_before
    assert rel_err < 1e-12, f"column mass rel err = {rel_err}"

    # After fix: no level has pc[num]*rmass < total_core
    concgas_after = pc_new[:, 0, 0] * rmass[0, 0] - pc_new[:, 0, 1]
    assert (concgas_after >= -1e-30).all()


def test_fixcorecol_clamps_when_insufficient_mass():
    """Deficit exceeds available positive mass → Fortran zeroes pc to
    total_core/rmass at every level. Verify."""
    cfg = _two_elem_cfg()
    nz, nbin, nelem = cfg["nz"], cfg["nbin"], cfg["nelem"]
    rmass = cfg["rmass"]

    pc = jnp.zeros((nz, nbin, nelem))
    # Tiny positive at level 0; large deficit at level 1.
    pc = pc.at[0, 0, 0].set(2.0).at[0, 0, 1].set(1e-20)    # concgas=1e-20
    pc = pc.at[1, 0, 0].set(1.0).at[1, 0, 1].set(100e-20)  # huge deficit

    pc_new = fixcorecol(
        pc, rmass, cfg["dz"],
        cfg["icorelem"], cfg["ienconc"], cfg["ncore"],
    )
    # Every level should now have pc[num] = total_core / rmass
    expected = pc[:, 0, 1] / rmass[0, 0]
    assert jnp.allclose(pc_new[:, 0, 0], expected)


def test_fixcorecol_skips_groups_without_cores():
    """If ncore = 0, the group is untouched."""
    cfg = _two_elem_cfg()
    cfg["ncore"] = np.asarray([0])

    pc = jnp.ones((cfg["nz"], cfg["nbin"], cfg["nelem"]))
    pc_new = fixcorecol(
        pc, cfg["rmass"], cfg["dz"],
        cfg["icorelem"], cfg["ienconc"], cfg["ncore"],
    )
    assert jnp.allclose(pc, pc_new)


def test_fixcorecol_jits():
    cfg = _two_elem_cfg()
    nz, nbin, nelem = cfg["nz"], cfg["nbin"], cfg["nelem"]
    pc = jnp.zeros((nz, nbin, nelem))
    pc = pc.at[0, 0, 0].set(20.0).at[0, 0, 1].set(5e-20)
    pc = pc.at[2, 0, 0].set(1.0).at[2, 0, 1].set(10e-20)

    def _wrap(pc):
        return fixcorecol(
            pc, cfg["rmass"], cfg["dz"],
            cfg["icorelem"], cfg["ienconc"], cfg["ncore"],
        )

    eager = _wrap(pc)
    jit_out = jax.jit(_wrap)(pc)
    assert jnp.allclose(eager, jit_out)


# ----- coremasscheck -----

def test_coremasscheck_flags_exceeded():
    cfg = _two_elem_cfg()
    pc = jnp.zeros((cfg["nz"], cfg["nbin"], cfg["nelem"]))
    # Bin 0 at level 1 has 1 particle (mass 1e-20) but core 5e-20.
    pc = pc.at[1, 0, 0].set(1.0).at[1, 0, 1].set(5e-20)

    _, any_exc = coremasscheck(
        pc, 1, cfg["rmass"],
        cfg["icorelem"], cfg["ienconc"], cfg["ncore"],
        behavior="never",
    )
    assert bool(any_exc)


def test_coremasscheck_roundoff_fixes_small_error():
    """Core 1.0000001× total → within 1e-14 roundoff threshold when
    rel_err < 1e-14. Set up a rel_err just inside the threshold."""
    cfg = _two_elem_cfg()
    rmass = cfg["rmass"]
    pc = jnp.zeros((cfg["nz"], cfg["nbin"], cfg["nelem"]))
    # Total mass 1e-20; core 1e-20 · (1 + 1e-15). Tiny overshoot.
    pc = pc.at[2, 0, 0].set(1.0)
    pc = pc.at[2, 0, 1].set(1e-20 * (1.0 + 1e-15))

    pc_new, _ = coremasscheck(
        pc, 2, rmass,
        cfg["icorelem"], cfg["ienconc"], cfg["ncore"],
        behavior="roundoff",
    )
    # pc[num] adjusted so pc[num]*rmass = coremass exactly
    assert jnp.isclose(pc_new[2, 0, 0] * rmass[0, 0], pc[2, 0, 1])


def test_coremasscheck_roundoff_does_not_fix_large_error():
    """Large rel_err (> 1e-14) in roundoff mode → unchanged."""
    cfg = _two_elem_cfg()
    pc = jnp.zeros((cfg["nz"], cfg["nbin"], cfg["nelem"]))
    pc = pc.at[0, 0, 0].set(1.0).at[0, 0, 1].set(5e-20)   # core 5x larger

    pc_new, any_exc = coremasscheck(
        pc, 0, cfg["rmass"],
        cfg["icorelem"], cfg["ienconc"], cfg["ncore"],
        behavior="roundoff",
    )
    assert bool(any_exc)
    assert jnp.allclose(pc, pc_new)


def test_coremasscheck_always_fixes_large_error():
    cfg = _two_elem_cfg()
    rmass = cfg["rmass"]
    pc = jnp.zeros((cfg["nz"], cfg["nbin"], cfg["nelem"]))
    pc = pc.at[0, 0, 0].set(1.0).at[0, 0, 1].set(5e-20)

    pc_new, _ = coremasscheck(
        pc, 0, rmass,
        cfg["icorelem"], cfg["ienconc"], cfg["ncore"],
        behavior="always",
    )
    # pc[num] * rmass must now equal coremass = 5e-20
    assert jnp.isclose(pc_new[0, 0, 0] * rmass[0, 0], pc[0, 0, 1])


def test_coremasscheck_never_never_modifies():
    cfg = _two_elem_cfg()
    pc = jnp.zeros((cfg["nz"], cfg["nbin"], cfg["nelem"]))
    pc = pc.at[0, 0, 0].set(1.0).at[0, 0, 1].set(5e-20)

    pc_new, any_exc = coremasscheck(
        pc, 0, cfg["rmass"],
        cfg["icorelem"], cfg["ienconc"], cfg["ncore"],
        behavior="never",
    )
    assert bool(any_exc)
    assert jnp.allclose(pc, pc_new)


def test_coremasscheck_ignores_zero_num():
    """Fortran skips bins with pc[iepart] == 0."""
    cfg = _two_elem_cfg()
    pc = jnp.zeros((cfg["nz"], cfg["nbin"], cfg["nelem"]))
    pc = pc.at[0, 0, 1].set(5e-20)   # core exists but num is 0

    pc_new, any_exc = coremasscheck(
        pc, 0, cfg["rmass"],
        cfg["icorelem"], cfg["ienconc"], cfg["ncore"],
        behavior="always",
    )
    assert not bool(any_exc)
    assert jnp.allclose(pc, pc_new)


def test_coremasscheck_jits():
    cfg = _two_elem_cfg()
    pc = jnp.zeros((cfg["nz"], cfg["nbin"], cfg["nelem"]))
    pc = pc.at[0, 0, 0].set(1.0).at[0, 0, 1].set(5e-20)

    def _wrap(pc):
        return coremasscheck(
            pc, 0, cfg["rmass"],
            cfg["icorelem"], cfg["ienconc"], cfg["ncore"],
            behavior="always",
        )

    eager = _wrap(pc)
    jit_out = jax.jit(_wrap)(pc)
    assert jnp.allclose(eager[0], jit_out[0])
    assert bool(eager[1]) == bool(jit_out[1])


def test_coremasscheck_unknown_behavior_raises():
    cfg = _two_elem_cfg()
    pc = jnp.zeros((cfg["nz"], cfg["nbin"], cfg["nelem"]))
    with pytest.raises(ValueError, match="unknown behavior"):
        coremasscheck(
            pc, 0, cfg["rmass"],
            cfg["icorelem"], cfg["ienconc"], cfg["ncore"],
            behavior="nonsense",
        )
