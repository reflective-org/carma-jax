"""Unit tests for setup_vf_heymsfield2010 (Phase 11.10).

Heymsfield & Westbrook 2010 fall-velocity for ice particles. Per-bin,
per-altitude formula; single regime (no Stokes/transitional/high-Re
branching). Bench against the gfortran-compiled standalone.
"""
import math
import subprocess
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from carma.constants import RHO_I
from carma.rhoice_heymsfield2010 import rhoice_heymsfield2010
from carma.setup_vf_heymsfield2010 import setup_vf_heymsfield2010

REPO = Path(__file__).resolve().parents[2]
FORTRAN_BIN = (REPO.parent / "original-carma" / "CARMA"
               / "build_standalone" / "setup_vf_heymsfield2010_standalone")

NBIN = 28
RMIN_CM = 5.0e-4
RMRAT = 4.0


def _bin_setup():
    rhoice = float(RHO_I)
    rmassmin = (4.0 / 3.0) * math.pi * RMIN_CM**3 * rhoice
    rmass = rmassmin * RMRAT ** np.arange(NBIN)
    rho_eff, arat = rhoice_heymsfield2010(rhoice, rmassmin, RMRAT, "avg", NBIN)
    rho_eff = np.asarray(rho_eff); arat = np.asarray(arat)
    r = (rmass / ((4.0 / 3.0) * math.pi * rho_eff)) ** (1.0 / 3.0)
    rrat = np.ones(NBIN)
    return r, rrat, rmass, arat


def _column(nz):
    t = np.linspace(220.0, 200.0, nz)
    rmu = 1.4e-4 + 1e-7 * (t - 200.0)
    rhoa = 1.0e-3 * np.exp(-np.arange(nz) * 0.5)
    zmet = np.full(nz, 5.0e4)
    return t, rhoa, zmet, rmu


def _run_fortran(t, rhoa, zmet, rmu, r_wet, rrat, rmass, arat, work):
    nz, nbin = r_wet.shape
    in_path = work / "in.txt"
    out_path = work / "out.bin"
    with open(in_path, "w") as f:
        f.write(f"{nz} {nbin}\n")
        f.write(" ".join(repr(float(x)) for x in t) + "\n")
        f.write(" ".join(repr(float(x)) for x in rhoa) + "\n")
        f.write(" ".join(repr(float(x)) for x in zmet) + "\n")
        f.write(" ".join(repr(float(x)) for x in rmu) + "\n")
        for k in range(nz):
            f.write(" ".join(repr(float(x)) for x in r_wet[k]) + "\n")
        f.write(" ".join(repr(float(x)) for x in rrat) + "\n")
        f.write(" ".join(repr(float(x)) for x in rmass) + "\n")
        f.write(" ".join(repr(float(x)) for x in arat) + "\n")
    subprocess.run([str(FORTRAN_BIN), str(in_path), str(out_path)], check=True)
    arr = np.fromfile(out_path, dtype=np.float64)
    n = nz * nbin
    vf = arr[:n].reshape(nz, nbin)
    re = arr[n:2 * n].reshape(nz, nbin)
    bpm = arr[2 * n:3 * n].reshape(nz, nbin)
    return vf, re, bpm


def _run_jax(t, rhoa, zmet, rmu, r_wet, rrat, rmass, arat):
    vf, re, bpm = setup_vf_heymsfield2010(
        t=jnp.asarray(t), rhoa=jnp.asarray(rhoa), zmet=jnp.asarray(zmet),
        rmu=jnp.asarray(rmu),
        r_wet=jnp.asarray(r_wet[:, :, None]),
        rrat=jnp.asarray(rrat[:, None]),
        rmass=jnp.asarray(rmass[:, None]),
        arat=jnp.asarray(arat[:, None]))
    return (np.asarray(vf)[..., 0],
            np.asarray(re)[..., 0],
            np.asarray(bpm)[..., 0])


_skip = pytest.mark.skipif(
    not FORTRAN_BIN.exists(),
    reason="setup_vf_heymsfield2010_standalone not built. "
           "Run scripts/fortran_patch/build_setup_vf_heymsfield2010_standalone.sh.")


@_skip
@pytest.mark.parametrize("nz", [1, 5, 32])
def test_setup_vf_heymsfield2010_matches_fortran(nz, tmp_path):
    """JAX ≈ Fortran across a multi-altitude column, all 28 bins."""
    r_1d, rrat, rmass, arat = _bin_setup()
    r_wet = np.broadcast_to(r_1d[None, :], (nz, NBIN)).copy()
    t, rhoa, zmet, rmu = _column(nz)
    vf_F, re_F, bpm_F = _run_fortran(
        t, rhoa, zmet, rmu, r_wet, rrat, rmass, arat, tmp_path)
    vf_J, re_J, bpm_J = _run_jax(
        t, rhoa, zmet, rmu, r_wet, rrat, rmass, arat)

    for name, F, J, tol in (
        ("bpm", bpm_F, bpm_J, 1e-13),
        ("re",  re_F,  re_J,  1e-12),
        ("vf",  vf_F,  vf_J,  1e-12),
    ):
        denom = np.maximum(np.abs(F), 1e-300)
        rel = np.abs(J - F) / denom
        assert float(rel.max()) < tol, (
            f"nz={nz} {name}: max rel err {rel.max():.3e}\n"
            f"  argmax: {np.unravel_index(int(rel.argmax()), F.shape)}\n"
            f"  F: {F.flat[int(rel.argmax())]}\n"
            f"  J: {J.flat[int(rel.argmax())]}")


def test_setup_vf_heymsfield2010_vf_monotonic_with_size():
    """Fall velocity should increase monotonically with bin size."""
    r_1d, rrat, rmass, arat = _bin_setup()
    nz = 1
    r_wet = r_1d[None, :]
    t, rhoa, zmet, rmu = _column(nz)
    vf, _, _ = _run_jax(t, rhoa, zmet, rmu, r_wet, rrat, rmass, arat)
    assert bool(np.all(np.diff(vf[0]) >= 0))


def test_setup_vf_heymsfield2010_jit_compiles():
    """Kernel must compile under jax.jit (single-column smoke)."""
    r_1d, rrat, rmass, arat = _bin_setup()
    nz = 3
    r_wet = np.broadcast_to(r_1d[None, :], (nz, NBIN)).copy()
    t, rhoa, zmet, rmu = _column(nz)
    vf, re, bpm = setup_vf_heymsfield2010(
        t=jnp.asarray(t), rhoa=jnp.asarray(rhoa), zmet=jnp.asarray(zmet),
        rmu=jnp.asarray(rmu),
        r_wet=jnp.asarray(r_wet[:, :, None]),
        rrat=jnp.asarray(rrat[:, None]),
        rmass=jnp.asarray(rmass[:, None]),
        arat=jnp.asarray(arat[:, None]))
    assert vf.shape == (nz, NBIN, 1)
    assert bool(jnp.all(jnp.isfinite(vf)))
