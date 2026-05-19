"""Unit tests for freezaerl_koop2000 (Phase 11.3).

Bench against the standalone gfortran-compiled kernel
(scripts/fortran_patch/freezaerl_koop2000_standalone.F90).

When the binary isn't available the Fortran-bench tests skip; the
JIT-roundtrip and gate tests still run.
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
from carma.nucleation.freezaerl_koop2000 import freezaerl_koop2000

REPO = Path(__file__).resolve().parents[2]
FORTRAN_BIN = (REPO.parent / "original-carma" / "CARMA"
               / "build_standalone" / "freezaerl_koop2000_standalone")


def _bin_grid(nbin=16, rmin_cm=1e-7, rmrat=4.0, rho=1.78):
    vmin = (4.0 / 3.0) * math.pi * rmin_cm**3 * rho
    rmass = vmin * rmrat ** np.arange(nbin)
    vol = rmass / rho
    r = (3.0 * rmass / (4.0 * math.pi * rho)) ** (1.0 / 3.0)
    return r, vol


def _run_fortran(scen, r_bins, vol_bins, work):
    nbin = r_bins.shape[0]
    in_path = work / "in.txt"
    out_path = work / "out.bin"
    with open(in_path, "w") as f:
        f.write(f"{scen['T']!r}  {scen['p']!r}  {scen['ssi']!r}  {scen['ssl']!r}  "
                f"{scen['akelvin']!r}  {scen['rhosol']!r}  {scen['pconmax']!r}  {nbin}\n")
        f.write(" ".join(repr(float(x)) for x in r_bins) + "\n")
        f.write(" ".join(repr(float(x)) for x in vol_bins) + "\n")
    subprocess.run([str(FORTRAN_BIN), str(in_path), str(out_path)], check=True)
    return np.fromfile(out_path, dtype=np.float64)


def _run_jax(scen, r_bins, vol_bins):
    return np.asarray(freezaerl_koop2000(
        t_val=jnp.float64(scen['T']),
        p_val=jnp.float64(scen['p']),
        supsati_val=jnp.float64(scen['ssi']),
        supsatl_val=jnp.float64(scen['ssl']),
        akelvin_val=jnp.float64(scen['akelvin']),
        r_bins=jnp.asarray(r_bins), vol_bins=jnp.asarray(vol_bins),
        rhosol_val=jnp.float64(scen['rhosol']),
        pconmax_val=jnp.float64(scen['pconmax']),
        nbin=r_bins.shape[0],
    ))


_skip_no_fortran = pytest.mark.skipif(
    not FORTRAN_BIN.exists(),
    reason="Fortran bench binary not built. "
           "Run scripts/fortran_patch/build_freezaerl_koop2000_standalone.sh.")


SCENARIOS = [
    # (label, T, p_dyn, ssi, ssl, akelvin, rhosol, pconmax)
    ("cirrus_typical",  220.0, 2.0e5, 0.50, -0.40, 1.5e-7, 1.78, 1e3),
    ("very_cold_high",  200.0, 1.0e5, 0.80, -0.20, 1.4e-7, 1.78, 1e4),
    ("low_aw_regime",   220.0, 2.0e5, 0.50, -0.99, 1.5e-7, 1.78, 1e3),
    ("high_aw_regime",  220.0, 2.0e5, 0.50, -0.05, 1.5e-7, 1.78, 1e3),
    ("low_pressure",    210.0, 5.0e4, 0.50, -0.40, 1.5e-7, 1.78, 1e3),
    ("high_pressure",   220.0, 5.0e5, 0.50, -0.40, 1.5e-7, 1.78, 1e3),
    # gates
    ("warm_skip_T",     250.0, 2.0e5, 0.50, -0.40, 1.5e-7, 1.78, 1e3),
    ("low_ssi_skip",    220.0, 2.0e5, 0.10, -0.40, 1.5e-7, 1.78, 1e3),
    ("no_particles",    220.0, 2.0e5, 0.50, -0.40, 1.5e-7, 1.78, 1e-50),
]


@_skip_no_fortran
@pytest.mark.parametrize(
    "label,T,p,ssi,ssl,akelvin,rhosol,pconmax", SCENARIOS)
def test_freezaerl_koop2000_matches_fortran(
        label, T, p, ssi, ssl, akelvin, rhosol, pconmax, tmp_path):
    """JAX matches the gfortran-compiled kernel within 1e-9 relative."""
    r_bins, vol_bins = _bin_grid()
    scen = dict(T=T, p=p, ssi=ssi, ssl=ssl, akelvin=akelvin,
                rhosol=rhosol, pconmax=pconmax)
    F = _run_fortran(scen, r_bins, vol_bins, tmp_path)
    J = _run_jax(scen, r_bins, vol_bins)
    if np.any(F > 0):
        denom = np.maximum(np.abs(F), np.abs(J))
        denom = np.where(denom > 1e-300, denom, 1.0)
        rel = np.abs(F - J) / denom
        # 1e-9 leaves comfortable headroom over the bench-observed max
        # 1.2e-11 on 1000 random scenarios.
        assert float(rel.max()) < 1e-9, (
            f"scenario {label}: max rel err {rel.max():.3e}\n"
            f"  Fortran: {F}\n  JAX:     {J}")
    else:
        np.testing.assert_array_equal(F, J)


def test_freezaerl_koop2000_jit_equivalent():
    r_bins, vol_bins = _bin_grid()
    args = dict(
        t_val=jnp.float64(220.0), p_val=jnp.float64(2.0e5),
        supsati_val=jnp.float64(0.5), supsatl_val=jnp.float64(-0.4),
        akelvin_val=jnp.float64(1.5e-7),
        r_bins=jnp.asarray(r_bins), vol_bins=jnp.asarray(vol_bins),
        rhosol_val=jnp.float64(1.78),
        pconmax_val=jnp.float64(1e3),
        nbin=16,
    )
    out      = freezaerl_koop2000(**args)
    out_jit  = jax.jit(freezaerl_koop2000, static_argnames=("nbin",))(**args)
    np.testing.assert_allclose(np.asarray(out), np.asarray(out_jit),
                                rtol=1e-12, atol=0.0)


def test_freezaerl_koop2000_gates():
    r_bins, vol_bins = _bin_grid()
    base = dict(
        p_val=jnp.float64(2.0e5),
        akelvin_val=jnp.float64(1.5e-7),
        r_bins=jnp.asarray(r_bins), vol_bins=jnp.asarray(vol_bins),
        rhosol_val=jnp.float64(1.78),
        nbin=16,
    )
    active = freezaerl_koop2000(
        t_val=jnp.float64(220.0), supsati_val=jnp.float64(0.5),
        supsatl_val=jnp.float64(-0.4), pconmax_val=jnp.float64(1e3), **base)
    assert bool(jnp.any(active > 0))

    for kwargs in (
        dict(t_val=jnp.float64(245.0),                # T > 240
             supsati_val=jnp.float64(0.5),
             supsatl_val=jnp.float64(-0.4),
             pconmax_val=jnp.float64(1e3)),
        dict(t_val=jnp.float64(220.0),
             supsati_val=jnp.float64(0.1),            # ssi < 0.3
             supsatl_val=jnp.float64(-0.4),
             pconmax_val=jnp.float64(1e3)),
        dict(t_val=jnp.float64(220.0),
             supsati_val=jnp.float64(0.5),
             supsatl_val=jnp.float64(-0.4),
             pconmax_val=jnp.float64(1e-50)),         # pconmax < FEW_PC
    ):
        out = freezaerl_koop2000(**kwargs, **base)
        assert bool(jnp.all(out == 0)), f"gate failed for {kwargs}"
