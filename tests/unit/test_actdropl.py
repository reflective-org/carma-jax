"""Unit tests for actdropl (Phase 11.9).

Cloud droplet activation: constant-rate gate kernel — same family
as freezdropl/melticel. Five gates (T ≥ T0-40, pconmax > FEW_PC,
supsatl > scrit[bin], pc[bin] > SMALL_PC, target bin not evaporating);
rate is exactly 1000 s⁻¹ when all fire.
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
from carma.nucleation.actdropl import actdropl

REPO = Path(__file__).resolve().parents[2]
FORTRAN_BIN = (REPO.parent / "original-carma" / "CARMA"
               / "build_standalone" / "actdropl_standalone")


def _run_fortran(scen, scrit, pc, target_evap, work):
    nbin = scrit.shape[0]
    in_path = work / "in.txt"
    out_path = work / "out.bin"
    with open(in_path, "w") as f:
        f.write(f"{float(scen['T'])!r} {float(scen['supsatl'])!r} "
                f"{float(scen['pconmax'])!r} {nbin}\n")
        f.write(" ".join(repr(float(x)) for x in scrit) + "\n")
        f.write(" ".join(repr(float(x)) for x in pc) + "\n")
        f.write(" ".join(repr(float(x)) for x in target_evap) + "\n")
    subprocess.run([str(FORTRAN_BIN), str(in_path), str(out_path)], check=True)
    return np.fromfile(out_path, dtype=np.float64)


def _run_jax(scen, scrit, pc, target_evap):
    return np.asarray(actdropl(
        t_val=jnp.float64(scen['T']),
        supsatl_val=jnp.float64(scen['supsatl']),
        pconmax_val=jnp.float64(scen['pconmax']),
        scrit_bins=jnp.asarray(scrit),
        pc_bins=jnp.asarray(pc),
        evappe_target_bins=jnp.asarray(target_evap),
    ))


_skip = pytest.mark.skipif(
    not FORTRAN_BIN.exists(),
    reason="actdropl_standalone not built. "
           "Run scripts/fortran_patch/build_actdropl_standalone.sh.")


def _default_arrays(nbin=16):
    # Realistic per-bin scrit decreasing with bin size (bigger bins
    # activate more easily): 1e-2 at smallest, 1e-5 at largest.
    scrit = np.geomspace(1e-2, 1e-5, nbin)
    pc = np.full(nbin, 1e3)
    target_evap = np.zeros(nbin)
    return scrit, pc, target_evap


def test_actdropl_warm_super_activates_large_bins():
    """At supsatl = 1e-3, only bins where scrit < 1e-3 activate."""
    scrit, pc, te = _default_arrays()
    out = actdropl(jnp.float64(280.0), jnp.float64(1e-3),
                    jnp.float64(1e3),
                    jnp.asarray(scrit), jnp.asarray(pc), jnp.asarray(te))
    expected = np.where(scrit < 1e-3, 1000.0, 0.0)
    np.testing.assert_array_equal(np.asarray(out), expected)


def test_actdropl_cold_skip():
    """T < T0 - 40 K → all zero regardless of scrit / supsatl."""
    scrit, pc, te = _default_arrays()
    out = actdropl(jnp.float64(220.0), jnp.float64(1e-2),
                    jnp.float64(1e3),
                    jnp.asarray(scrit), jnp.asarray(pc), jnp.asarray(te))
    assert bool(jnp.all(out == 0))


def test_actdropl_pc_floor_gate():
    """Bins where pc ≤ SMALL_PC stay at zero even with all other gates open."""
    scrit, pc, _ = _default_arrays()
    pc[0:3] = 1e-55                                   # below SMALL_PC = 1e-50
    te = np.zeros(scrit.shape[0])
    out = actdropl(jnp.float64(280.0), jnp.float64(1e-2),
                    jnp.float64(1e3),
                    jnp.asarray(scrit), jnp.asarray(pc), jnp.asarray(te))
    out_np = np.asarray(out)
    assert bool(np.all(out_np[0:3] == 0))


def test_actdropl_target_evap_gate():
    """Bins flagged as having evaporating target droplet stay at zero."""
    scrit, pc, te = _default_arrays()
    te[5:10] = 1.0                                    # target evaporating
    out = actdropl(jnp.float64(280.0), jnp.float64(1e-2),
                    jnp.float64(1e3),
                    jnp.asarray(scrit), jnp.asarray(pc), jnp.asarray(te))
    out_np = np.asarray(out)
    assert bool(np.all(out_np[5:10] == 0))


def test_actdropl_pconmax_gate():
    """pconmax ≤ FEW_PC → all zero."""
    scrit, pc, te = _default_arrays()
    out = actdropl(jnp.float64(280.0), jnp.float64(1e-2),
                    jnp.float64(1e-50),
                    jnp.asarray(scrit), jnp.asarray(pc), jnp.asarray(te))
    assert bool(jnp.all(out == 0))


@_skip
@pytest.mark.parametrize("T,supsatl,pconmax,label", [
    (280.0, 1e-3,  1e3, "warm_low_super"),
    (280.0, 5e-2,  1e3, "warm_high_super"),
    (260.0, 1e-3,  1e3, "cold_active"),
    (220.0, 1e-3,  1e3, "below_T_gate"),
    (280.0, 1e-3,  1e-50, "below_pconmax_gate"),
])
def test_actdropl_matches_fortran(T, supsatl, pconmax, label, tmp_path):
    """JAX bit-matches the gfortran-compiled standalone (constant rate)."""
    scrit, pc, te = _default_arrays()
    scen = dict(T=T, supsatl=supsatl, pconmax=pconmax)
    F = _run_fortran(scen, scrit, pc, te, tmp_path)
    J = _run_jax(scen, scrit, pc, te)
    np.testing.assert_array_equal(J, F)
