"""Unit tests for hetnucl (Phase 11.7).

Bench against the standalone gfortran-compiled kernel
(scripts/fortran_patch/hetnucl_standalone.F90, built via
build_hetnucl_standalone.sh).
"""
import math
import subprocess
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from carma.constants import FEW_PC
from carma.nucleation.hetnucl import hetnucl

REPO = Path(__file__).resolve().parents[2]
FORTRAN_BIN = (REPO.parent / "original-carma" / "CARMA"
               / "build_standalone" / "hetnucl_standalone")


def _bin_grid(nbin=16, rmin_cm=1e-7, rmrat=4.0, rho=1.78):
    vmin = (4.0 / 3.0) * math.pi * rmin_cm**3 * rho
    rmass = vmin * rmrat ** np.arange(nbin)
    r = (3.0 * rmass / (4.0 * math.pi * rho)) ** (1.0 / 3.0)
    return r


def _run_fortran(scen, r_bins, work):
    nbin = r_bins.shape[0]
    in_path = work / "in.txt"
    out_path = work / "out.bin"
    with open(in_path, "w") as f:
        f.write(f"{float(scen['T'])!r} {float(scen['p'])!r} "
                f"{float(scen['ssi'])!r} {float(scen['gc_h2o'])!r} "
                f"{float(scen['gwtmol'])!r} {float(scen['surfctia'])!r} "
                f"{float(scen['pconmax'])!r} {nbin}\n")
        f.write(" ".join(repr(float(x)) for x in r_bins) + "\n")
    subprocess.run([str(FORTRAN_BIN), str(in_path), str(out_path)], check=True)
    return np.fromfile(out_path, dtype=np.float64)


def _run_jax(scen, r_bins):
    return np.asarray(hetnucl(
        t_val=jnp.float64(scen['T']),
        p_val=jnp.float64(scen['p']),
        supsati_val=jnp.float64(scen['ssi']),
        gc_h2o=jnp.float64(scen['gc_h2o']),
        gwtmol_h2o=jnp.float64(scen['gwtmol']),
        surfctia_val=jnp.float64(scen['surfctia']),
        r_bins=jnp.asarray(r_bins),
        pconmax_val=jnp.float64(scen['pconmax']),
    ))


_skip = pytest.mark.skipif(
    not FORTRAN_BIN.exists(),
    reason="hetnucl_standalone not built. Run "
           "scripts/fortran_patch/build_hetnucl_standalone.sh.")


SCENARIOS = [
    # (label, T, p_dyn, ssi, gc_h2o, gwtmol, surfctia, pconmax)
    ("pmc_typical",  150.0, 1.0,   2.0, 1e-12, 18.016, 100.0, 1e3),
    ("very_cold",    110.0, 5.0,   5.0, 1e-13, 18.016, 110.0, 1e4),
    ("near_p_gate",  150.0, 9e2,   2.0, 1e-12, 18.016, 100.0, 1e3),
    ("low_ssi",      150.0, 1.0,   0.01, 1e-12, 18.016, 100.0, 1e3),
    # gate-off
    ("high_p_skip",  150.0, 2e5,   2.0, 1e-12, 18.016, 100.0, 1e3),
    ("ssi_zero",     150.0, 1.0,   -0.5, 1e-12, 18.016, 100.0, 1e3),
    ("no_particles", 150.0, 1.0,   2.0, 1e-12, 18.016, 100.0, 1e-50),
]


@_skip
@pytest.mark.parametrize(
    "label,T,p,ssi,gc_h2o,gwtmol,surfctia,pconmax", SCENARIOS)
def test_hetnucl_matches_fortran(
        label, T, p, ssi, gc_h2o, gwtmol, surfctia, pconmax, tmp_path):
    """JAX matches the gfortran-compiled kernel within 1e-3 relative.

    Looser than other freezaerl_* benches (1e-9) because the
    heterogeneous nucleation geometric factor `fh` has catastrophic
    cancellation at large `x = r/ag` — observed rel err rises with
    bin size from machine ε at sub-100-nm bins (where PMC particles
    actually live) to ~1e-4 at the largest 2 µm bin. Setting the
    tolerance at 1e-3 captures the worst case across the parameter
    sweep; for PMC-relevant bin sizes the agreement is far tighter.
    """
    r_bins = _bin_grid()
    scen = dict(T=T, p=p, ssi=ssi, gc_h2o=gc_h2o,
                gwtmol=gwtmol, surfctia=surfctia, pconmax=pconmax)
    F = _run_fortran(scen, r_bins, tmp_path)
    J = _run_jax(scen, r_bins)
    if np.any(F > 0):
        denom = np.maximum(np.abs(F), np.abs(J))
        denom = np.where(denom > 1e-300, denom, 1.0)
        rel = np.abs(F - J) / denom
        assert float(rel.max()) < 1e-3, (
            f"scenario {label}: max rel err {rel.max():.3e}\n"
            f"  Fortran: {F}\n  JAX:     {J}")
    else:
        np.testing.assert_array_equal(F, J)


def test_hetnucl_pmc_bins_high_precision():
    """At PMC-relevant bin sizes (≤ 200 nm) the rel err is much tighter."""
    r_bins = _bin_grid()
    # Bins 0-8 = ~2-200 nm (where PMC particles actually live)
    pmc_bin_mask = (2.0 * r_bins * 1e7) <= 200.0
    scen = dict(T=150.0, p=1.0, ssi=2.0, gc_h2o=1e-12,
                gwtmol=18.016, surfctia=100.0, pconmax=1e3)
    J = _run_jax(scen, r_bins)
    if FORTRAN_BIN.exists():
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            F = _run_fortran(scen, r_bins, Path(tmp))
        denom = np.maximum(np.abs(F[pmc_bin_mask]), np.abs(J[pmc_bin_mask]))
        denom = np.where(denom > 1e-300, denom, 1.0)
        rel = np.abs(F[pmc_bin_mask] - J[pmc_bin_mask]) / denom
        rel = rel[rel > 0]
        if rel.size:
            assert float(rel.max()) < 1e-7, (
                f"PMC-bin rel err {rel.max():.3e} above 1e-7 — investigate")


def test_hetnucl_gates():
    r_bins = _bin_grid()
    base = dict(
        gc_h2o=jnp.float64(1e-12),
        gwtmol_h2o=jnp.float64(18.016),
        surfctia_val=jnp.float64(100.0),
        r_bins=jnp.asarray(r_bins),
    )
    active = hetnucl(
        t_val=jnp.float64(150.0), p_val=jnp.float64(1.0),
        supsati_val=jnp.float64(2.0),
        pconmax_val=jnp.float64(1e3), **base)
    assert bool(jnp.any(active > 0))

    for kwargs in (
        dict(t_val=jnp.float64(150.0),
             p_val=jnp.float64(2e5),                  # p > 1 hPa
             supsati_val=jnp.float64(2.0),
             pconmax_val=jnp.float64(1e3)),
        dict(t_val=jnp.float64(150.0),
             p_val=jnp.float64(1.0),
             supsati_val=jnp.float64(-0.5),            # ssi ≤ 0
             pconmax_val=jnp.float64(1e3)),
        dict(t_val=jnp.float64(150.0),
             p_val=jnp.float64(1.0),
             supsati_val=jnp.float64(2.0),
             pconmax_val=jnp.float64(1e-50)),          # pconmax < FEW_PC
    ):
        out = hetnucl(**kwargs, **base)
        assert bool(jnp.all(out == 0)), f"gate failed for {kwargs}"
