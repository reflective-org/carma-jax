"""Unit tests for carma.rhopart."""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from carma.rhopart import rhopart


# --- Single-element groups ---

def test_single_element_no_cores():
    """Group with a single number-concentration element: rhop == rhoelem."""
    NZ, NBIN = 1, 3
    pc = jnp.array([[[1e5], [1e3], [10.0]]])
    rmass = jnp.array([[1e-18], [1e-15], [1e-12]])
    rhoelem = jnp.array([[2.0], [2.0], [2.0]])
    ienconc = jnp.array([0])
    igelem = jnp.array([0])
    icorelem = jnp.full((1, 1), -1)
    ncore = jnp.array([0])

    rhop, pc_new = rhopart(pc, rmass, rhoelem, ienconc, igelem, icorelem, ncore)
    assert jnp.allclose(rhop, 2.0)
    assert jnp.allclose(pc_new, pc)


def test_single_element_variable_rho_per_bin():
    """With different rhoelem per bin, each bin takes its own density."""
    pc = jnp.array([[[100.0], [100.0]]])
    rmass = jnp.array([[1e-15], [1e-14]])
    rhoelem = jnp.array([[1.5], [2.5]])  # per-bin
    ienconc = jnp.array([0])
    igelem = jnp.array([0])
    icorelem = jnp.full((1, 1), -1)
    ncore = jnp.array([0])

    rhop, _ = rhopart(pc, rmass, rhoelem, ienconc, igelem, icorelem, ncore)
    assert float(rhop[0, 0, 0]) == 1.5
    assert float(rhop[0, 1, 0]) == 2.5


# --- Two-element groups (shell + one core) ---

def test_mixed_shell_core_density():
    """Shell rho=1.8, core rho=1.0, mixed 50/50 by mass → mid density."""
    # m_total = pc_num * rmass; set core = half of total
    pc_num = 1.0
    rmass_val = 1e-15
    m_total = pc_num * rmass_val       # 1e-15
    m_core = 0.5 * m_total             # 5e-16
    pc = jnp.array([[[pc_num, m_core]]])
    rmass = jnp.array([[rmass_val]])
    rhoelem = jnp.array([[1.8, 1.0]])
    ienconc = jnp.array([0])
    igelem = jnp.array([0, 0])
    icorelem = jnp.array([[1]])
    ncore = jnp.array([1])

    rhop, _ = rhopart(pc, rmass, rhoelem, ienconc, igelem, icorelem, ncore)
    # v_shell = m_core / 1.8 = 2.78e-16; v_core = m_core / 1.0 = 5e-16
    # rhop = 1e-15 / (2.78e-16 + 5e-16) = 1.286
    expected = m_total / ((m_total - m_core) / 1.8 + m_core / 1.0)
    assert abs(float(rhop[0, 0, 0]) - expected) / expected < 1e-10


def test_zero_core_in_bin_falls_back_to_shell():
    """In a group *with* cores, a bin where all cores happen to be zero
    (numerically) should still return the shell density."""
    pc = jnp.array([[[100.0, 0.0]]])
    rmass = jnp.array([[1e-15]])
    rhoelem = jnp.array([[1.8, 1.0]])
    ienconc = jnp.array([0])
    igelem = jnp.array([0, 0])
    icorelem = jnp.array([[1]])
    ncore = jnp.array([1])

    rhop, _ = rhopart(pc, rmass, rhoelem, ienconc, igelem, icorelem, ncore)
    assert abs(float(rhop[0, 0, 0]) - 1.8) < 1e-12


# --- Core-mass safety clamp ---

def test_core_exceeds_total_clamps_pc():
    """Numerical diffusion producing m_core > m_total: rhop = core density,
    pc_num repaired to m_core / rmass."""
    # pc_num = 1, rmass = 1e-15 → m_total = 1e-15. m_core = 5e-15 (>> total).
    pc = jnp.array([[[1.0, 5e-15]]])
    rmass = jnp.array([[1e-15]])
    rhoelem = jnp.array([[1.8, 1.0]])
    ienconc = jnp.array([0])
    igelem = jnp.array([0, 0])
    icorelem = jnp.array([[1]])
    ncore = jnp.array([1])

    rhop, pc_new = rhopart(pc, rmass, rhoelem, ienconc, igelem, icorelem, ncore)
    assert abs(float(rhop[0, 0, 0]) - 1.0) < 1e-12
    # pc_num should be repaired from 1.0 to m_core / rmass = 5
    assert abs(float(pc_new[0, 0, 0]) - 5.0) < 1e-12
    # Core pc unchanged
    assert float(pc_new[0, 0, 1]) == 5e-15


# --- Multiple cores (shell + 2 cores) ---

def test_two_cores_mass_weighted_density():
    """Shell + two cores; each contributes volume by rhoelem[core]."""
    pc_num = 1.0
    rmass_val = 1e-15
    m_total = pc_num * rmass_val
    m_c1 = 0.2 * m_total     # core 1 = 20% of total
    m_c2 = 0.3 * m_total     # core 2 = 30% of total
    pc = jnp.array([[[pc_num, m_c1, m_c2]]])
    rmass = jnp.array([[rmass_val]])
    rhoelem = jnp.array([[1.8, 2.0, 2.5]])
    ienconc = jnp.array([0])
    igelem = jnp.array([0, 0, 0])
    icorelem = jnp.array([[1], [2]])
    ncore = jnp.array([2])

    rhop, _ = rhopart(pc, rmass, rhoelem, ienconc, igelem, icorelem, ncore)
    m_core = m_c1 + m_c2
    v_core = m_c1 / 2.0 + m_c2 / 2.5
    v_shell = (m_total - m_core) / 1.8
    expected = m_total / (v_shell + v_core)
    assert abs(float(rhop[0, 0, 0]) - expected) / expected < 1e-10


# --- Multi-group ---

def test_multigroup_independent():
    """Two groups processed independently, each with its own cores."""
    # Group 0: single element (sulfate), no cores.
    # Group 1: shell + one core.
    NZ, NBIN, NELEM, NGROUP = 1, 1, 3, 2
    pc = jnp.array([[[100.0, 50.0, 1e-15]]])  # elem0=sulfate, elem1=ice shell, elem2=dust core
    rmass = jnp.array([[1e-15, 2e-15]])      # (NBIN, NGROUP)
    rhoelem = jnp.array([[1.8, 0.92, 2.6]])  # sulfate, ice, dust
    ienconc = jnp.array([0, 1])              # group 0→elem 0, group 1→elem 1
    igelem = jnp.array([0, 1, 1])            # elem 0 in g0; elem 1, 2 in g1
    icorelem = jnp.array([[-1, 2]])          # g0 no cores, g1 has core elem 2
    ncore = jnp.array([0, 1])

    rhop, _ = rhopart(pc, rmass, rhoelem, ienconc, igelem, icorelem, ncore)
    # Group 0: no cores → shell density 1.8
    assert abs(float(rhop[0, 0, 0]) - 1.8) < 1e-12
    # Group 1: core is element 2
    m_total = 50.0 * 2e-15
    m_core = 1e-15
    v_core = 1e-15 / 2.6
    v_shell = (m_total - m_core) / 0.92
    expected = m_total / (v_shell + v_core)
    assert abs(float(rhop[0, 0, 1]) - expected) / expected < 1e-10


# --- JIT / vmap ---

def test_rhopart_jit():
    pc = jnp.array([[[1e5, 1e-15, 2e-16]]])
    rmass = jnp.array([[1e-15]])
    rhoelem = jnp.array([[1.8, 1.0, 2.5]])
    ienconc = jnp.array([0])
    igelem = jnp.array([0, 0, 0])
    icorelem = jnp.array([[1], [2]])
    ncore = jnp.array([2])

    eager = rhopart(pc, rmass, rhoelem, ienconc, igelem, icorelem, ncore)
    jit_out = jax.jit(rhopart)(pc, rmass, rhoelem, ienconc, igelem, icorelem, ncore)
    assert jnp.allclose(eager[0], jit_out[0])
    assert jnp.allclose(eager[1], jit_out[1])


def test_rhopart_result_in_valid_range():
    """rhop should always lie between min and max element density, except
    under the core-truncation branch where it's the pure-core density."""
    pc = jnp.array([[[100.0, 20.0, 5.0]]])
    rmass = jnp.array([[1e-15]])
    rhoelem = jnp.array([[1.8, 1.0, 2.5]])
    ienconc = jnp.array([0])
    igelem = jnp.array([0, 0, 0])
    icorelem = jnp.array([[1], [2]])
    ncore = jnp.array([2])

    rhop, _ = rhopart(pc, rmass, rhoelem, ienconc, igelem, icorelem, ncore)
    # With cores, result should be in [min(rhoelem), max(rhoelem)] = [1.0, 2.5]
    assert 1.0 <= float(rhop[0, 0, 0]) <= 2.5
