"""Unit tests for freezdropl + melticel (Phase 11.6).

Both are placeholder "constant rate when T-gate fires" kernels — the
Fortran source self-identifies as "temporary simple kludge". Testing
checks the gates fire correctly and JAX matches the gfortran-compiled
standalone binary bit-for-bit (the kernel arithmetic is a single
constant, no floating-point chains to drift).
"""
import math
import subprocess
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from carma.constants import T0
from carma.nucleation.freezdropl import freezdropl
from carma.nucleation.melticel import melticel

REPO = Path(__file__).resolve().parents[2]
FREEZ_BIN = (REPO.parent / "original-carma" / "CARMA"
             / "build_standalone" / "freezdropl_standalone")
MELT_BIN  = (REPO.parent / "original-carma" / "CARMA"
             / "build_standalone" / "melticel_standalone")


def _run_fortran_freezdropl(t, pc, work):
    nbin = pc.shape[0]
    in_path = work / "in.txt"
    out_path = work / "out.bin"
    with open(in_path, "w") as f:
        f.write(f"{float(t)!r} {nbin}\n")
        f.write(" ".join(repr(float(x)) for x in pc) + "\n")
    subprocess.run([str(FREEZ_BIN), str(in_path), str(out_path)], check=True)
    return np.fromfile(out_path, dtype=np.float64)


def _run_fortran_melticel(t, pconmax, nbin, work):
    in_path = work / "in.txt"
    out_path = work / "out.bin"
    with open(in_path, "w") as f:
        f.write(f"{float(t)!r} {float(pconmax)!r} {nbin}\n")
    subprocess.run([str(MELT_BIN), str(in_path), str(out_path)], check=True)
    return np.fromfile(out_path, dtype=np.float64)


# ---------------------------- freezdropl ---------------------------- #

def test_freezdropl_cold_fires():
    """T < T0 - 40 K with particles → rate = 100/s on every populated bin."""
    pc = jnp.full((16,), 1e3, dtype=jnp.float64)
    out = freezdropl(jnp.float64(220.0), pc)
    assert bool(jnp.all(out == 100.0))


def test_freezdropl_warm_skip():
    """T ≥ T0 - 40 K → all zeros regardless of pc."""
    pc = jnp.full((16,), 1e3, dtype=jnp.float64)
    for T in (T0 - 40.0, 240.0, 280.0):
        out = freezdropl(jnp.float64(T), pc)
        assert bool(jnp.all(out == 0.0)), f"T={T} did not gate off"


def test_freezdropl_per_bin_pc_gate():
    """Bins with pc ≤ FEW_PC stay at zero; others get 100/s."""
    pc = jnp.array([1e-50, 1e-40, 1e3, 1e-30, 1e6] + [1e2] * 11,
                    dtype=jnp.float64)
    out = freezdropl(jnp.float64(220.0), pc)
    expected = np.array([0.0, 100.0, 100.0, 100.0, 100.0] + [100.0] * 11)
    np.testing.assert_array_equal(np.asarray(out), expected)


@pytest.mark.skipif(not FREEZ_BIN.exists(),
                     reason="freezdropl_standalone not built")
@pytest.mark.parametrize("T,pc_profile", [
    (220.0, "uniform_dense"),  # gate on, all populated
    (240.0, "uniform_dense"),  # T too warm
    (200.0, "mixed"),          # gate on, some bins below FEW_PC
])
def test_freezdropl_matches_fortran(T, pc_profile, tmp_path):
    if pc_profile == "uniform_dense":
        pc = np.full(16, 1e3, dtype=np.float64)
    else:
        pc = np.array([1e-55, 1e-50, 1e-40, 1e3, 1e6] + [1e2] * 11)
    F = _run_fortran_freezdropl(T, pc, tmp_path)
    J = np.asarray(freezdropl(jnp.float64(T), jnp.asarray(pc)))
    np.testing.assert_array_equal(J, F)


# ---------------------------- melticel ---------------------------- #

def test_melticel_warm_fires():
    """T > T0 with particles → rate = 100/s on every bin."""
    out = melticel(jnp.float64(280.0), jnp.float64(1e3), 16)
    assert bool(jnp.all(out == 100.0))


def test_melticel_cold_skip():
    """T ≤ T0 → all zeros regardless of pconmax."""
    for T in (T0, 260.0, 220.0):
        out = melticel(jnp.float64(T), jnp.float64(1e3), 16)
        assert bool(jnp.all(out == 0.0)), f"T={T} did not gate off"


def test_melticel_pconmax_gate():
    """pconmax ≤ FEW_PC → all zeros even at warm temperatures."""
    out = melticel(jnp.float64(280.0), jnp.float64(1e-50), 16)
    assert bool(jnp.all(out == 0.0))


@pytest.mark.skipif(not MELT_BIN.exists(),
                     reason="melticel_standalone not built")
@pytest.mark.parametrize("T,pconmax", [
    (280.0, 1e3),    # warm, populated
    (260.0, 1e3),    # cold, populated
    (280.0, 1e-50),  # warm, empty
    (T0,    1e3),    # boundary
    (300.0, 1e6),    # very warm, dense
])
def test_melticel_matches_fortran(T, pconmax, tmp_path):
    F = _run_fortran_melticel(T, pconmax, 16, tmp_path)
    J = np.asarray(melticel(jnp.float64(T), jnp.float64(pconmax), 16))
    np.testing.assert_array_equal(J, F)
