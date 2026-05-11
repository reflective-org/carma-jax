"""Unit tests for freezglaerl_murray2010 (Phase 11.8).

Murray 2010 glassy-aerosol heterogeneous freezing: linear-in-ssi
fraction-nucleated formula, produces a uniform per-bin rate. Bench
against the gfortran-compiled standalone.
"""
import math
import subprocess
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from carma.nucleation.freezglaerl_murray2010 import freezglaerl_murray2010

REPO = Path(__file__).resolve().parents[2]
FORTRAN_BIN = (REPO.parent / "original-carma" / "CARMA"
               / "build_standalone" / "freezglaerl_murray2010_standalone")
NBIN = 16


def _run_fortran(scen, work):
    in_path = work / "in.txt"
    out_path = work / "out.bin"
    with open(in_path, "w") as f:
        f.write(f"{float(scen['T'])!r} {float(scen['ssi'])!r} "
                f"{float(scen['ssi_old'])!r} {float(scen['pconmax'])!r} "
                f"{float(scen['dtime'])!r} {NBIN}\n")
    subprocess.run([str(FORTRAN_BIN), str(in_path), str(out_path)], check=True)
    return np.fromfile(out_path, dtype=np.float64)


def _run_jax(scen):
    return np.asarray(freezglaerl_murray2010(
        t_val=jnp.float64(scen['T']),
        supsati_val=jnp.float64(scen['ssi']),
        supsati_old_val=jnp.float64(scen['ssi_old']),
        pconmax_val=jnp.float64(scen['pconmax']),
        dtime=jnp.float64(scen['dtime']),
        nbin=NBIN,
    ))


_skip = pytest.mark.skipif(
    not FORTRAN_BIN.exists(),
    reason="freezglaerl_murray2010_standalone not built.")


SCENARIOS = [
    # (label, T, ssi, ssi_old, pconmax, dtime)
    ("cold_active",      200.0, 0.50, 0.10, 1e3, 1.0),
    ("near_ssmin",       200.0, 0.22, 0.10, 1e3, 1.0),
    ("saturated_ssmax",  200.0, 0.95, 0.10, 1e3, 1.0),
    ("ssi_old_active",   200.0, 0.50, 0.30, 1e3, 1.0),
    # gates
    ("warm_skip",        220.0, 0.50, 0.10, 1e3, 1.0),  # T > 212
    ("low_ssi_skip",     200.0, 0.15, 0.10, 1e3, 1.0),  # ssi < ssmin
    ("decreasing_ssi",   200.0, 0.30, 0.50, 1e3, 1.0),  # ssi_old > ssi
    ("no_particles",     200.0, 0.50, 0.10, 1e-50, 1.0),
]


@_skip
@pytest.mark.parametrize(
    "label,T,ssi,ssi_old,pconmax,dtime", SCENARIOS)
def test_freezglaerl_murray2010_matches_fortran(
        label, T, ssi, ssi_old, pconmax, dtime, tmp_path):
    scen = dict(T=T, ssi=ssi, ssi_old=ssi_old, pconmax=pconmax, dtime=dtime)
    F = _run_fortran(scen, tmp_path)
    J = _run_jax(scen)
    if np.any(F != 0):
        denom = np.maximum(np.abs(F), np.abs(J))
        denom = np.where(denom > 1e-300, denom, 1.0)
        rel = np.abs(F - J) / denom
        # 1e-10 leaves headroom over bench-observed max 2e-11.
        assert float(rel.max()) < 1e-10, (
            f"scenario {label}: max rel err {rel.max():.3e}\n"
            f"  Fortran: {F[0]}\n  JAX:     {J[0]}")
    else:
        np.testing.assert_array_equal(F, J)


def test_freezglaerl_murray2010_bin_uniform():
    """Rate is constant across all bins (formula doesn't depend on r)."""
    out = freezglaerl_murray2010(
        t_val=jnp.float64(200.0), supsati_val=jnp.float64(0.5),
        supsati_old_val=jnp.float64(0.1), pconmax_val=jnp.float64(1e3),
        dtime=jnp.float64(1.0), nbin=NBIN)
    assert bool(jnp.all(out == out[0]))
    assert float(out[0]) > 0


def test_freezglaerl_murray2010_gates():
    base = dict(supsati_val=jnp.float64(0.5),
                supsati_old_val=jnp.float64(0.1),
                dtime=jnp.float64(1.0), nbin=NBIN)
    active = freezglaerl_murray2010(t_val=jnp.float64(200.0),
                                     pconmax_val=jnp.float64(1e3), **base)
    assert bool(jnp.any(active > 0))

    for kwargs in (
        dict(t_val=jnp.float64(220.0), pconmax_val=jnp.float64(1e3)),  # T > 212
        dict(t_val=jnp.float64(200.0), pconmax_val=jnp.float64(1e-50)),  # FEW_PC
    ):
        out = freezglaerl_murray2010(**kwargs, **base)
        assert bool(jnp.all(out == 0)), f"gate failed for {kwargs}"

    # Decreasing supsaturation → no rate
    out = freezglaerl_murray2010(
        t_val=jnp.float64(200.0),
        supsati_val=jnp.float64(0.3),
        supsati_old_val=jnp.float64(0.5),    # old > new
        pconmax_val=jnp.float64(1e3),
        dtime=jnp.float64(1.0), nbin=NBIN)
    assert bool(jnp.all(out == 0))
