"""Phase 11.4 dispatch-context unit tests for freezaerl_mohler2010.

Bench against the patched `test_nuc2test_diagnostic` Fortran binary —
a real CARMA cstate (NGROUP=2, NELEM=3, NGAS=1) is set up by the
binary, then scenario inputs are injected into the cstate fields and
the upstream `freezaerl_mohler2010(carma, cstate, iz, rc)` is called
directly. Validates the JAX port matches the kernel running through
its real dispatch wrapper (group/element loops, inucproc gating,
isolelem → solute-density resolution).

Skipped if the diagnostic binary isn't built; build via
  bash scripts/fortran_patch/apply_nuc2_diagnostic_patch.sh
"""
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from carma.nucleation.freezaerl_mohler2010 import freezaerl_mohler2010

REPO = Path(__file__).resolve().parents[2]
DIAG_BIN = (REPO.parent / "original-carma" / "CARMA" / "build"
            / "test_nuc2test_diagnostic")

NBIN = 16
RHO_SOL = 1.38   # Sulfate solute density per nuc2test's CARMASOLUTE_Create
RMIN_CM, RMRAT, RHO_PARTICLE = 1.0e-7, 4.0, 1.78


def _bin_grid():
    vmin = (4.0 / 3.0) * math.pi * RMIN_CM**3 * RHO_PARTICLE
    rmass = vmin * RMRAT ** np.arange(NBIN)
    vol = rmass / RHO_PARTICLE
    r = (3.0 * rmass / (4.0 * math.pi * RHO_PARTICLE)) ** (1.0 / 3.0)
    return r, vol


def _run_fortran_dispatch(scen, work):
    scen_path = work / "scen.txt"
    out_dir = work / "out"
    out_dir.mkdir()
    with open(scen_path, "w") as f:
        f.write(
            "210.0 200.0 0.95 100.0 2.5e-6 1.5  "
            f"{scen['T']!r} {scen['ssi']!r} {scen['ssl']!r}  "
            f"{scen['akelvin']!r} {scen['akelvini']!r} {scen['pconmax']!r}  "
            "0.0 -1.0 -1.0\n"      # Phase 11.11: sentinels for ssi_old/p/gc_h2o
        )
    subprocess.run([str(DIAG_BIN), str(scen_path), str(out_dir), "1"],
                   check=True, capture_output=True)
    arr = np.fromfile(out_dir / "substep_0001_freezaerl_mohler_probe.bin",
                       dtype=np.float64).reshape((NBIN, 2, 2), order='F')
    return arr[:, 0, 1]   # Sulfate IN → Ice Crystal


def _run_jax(scen, r_bins, vol_bins):
    return np.asarray(freezaerl_mohler2010(
        t_val=jnp.float64(scen['T']),
        supsati_val=jnp.float64(scen['ssi']),
        supsatl_val=jnp.float64(scen['ssl']),
        akelvin_val=jnp.float64(scen['akelvin']),
        akelvini_val=jnp.float64(scen['akelvini']),
        r_bins=jnp.asarray(r_bins), vol_bins=jnp.asarray(vol_bins),
        rhosol_val=jnp.float64(RHO_SOL),    # solute density per nuc2test
        pconmax_val=jnp.float64(scen['pconmax']),
    ))


_skip_no_bin = pytest.mark.skipif(
    not DIAG_BIN.exists(),
    reason="nuc2test diagnostic binary not built. "
           "Run scripts/fortran_patch/apply_nuc2_diagnostic_patch.sh.")


SCENARIOS = [
    ("cirrus_typical",  220.0, 0.50, -0.40, 1.5e-7, 2.0e-7, 1e3),
    ("very_cold_high",  200.0, 0.80, -0.20, 1.4e-7, 1.9e-7, 1e4),
    ("low_aw_regime",   220.0, 0.50, -0.99, 1.5e-7, 2.0e-7, 1e3),
    ("high_aw_regime",  220.0, 0.50, -0.10, 1.5e-7, 2.0e-7, 1e3),
]


@_skip_no_bin
@pytest.mark.parametrize("label,T,ssi,ssl,akelvin,akelvini,pconmax", SCENARIOS)
def test_freezaerl_mohler2010_dispatch_matches_jax(
        label, T, ssi, ssl, akelvin, akelvini, pconmax, tmp_path):
    """JAX matches Fortran when both run through the upstream subroutine's
    dispatch wrapper, with cstate-resolved rhosol = 1.38 (solute density)."""
    r_bins, vol_bins = _bin_grid()
    scen = dict(T=T, ssi=ssi, ssl=ssl, akelvin=akelvin,
                akelvini=akelvini, pconmax=pconmax)
    F = _run_fortran_dispatch(scen, tmp_path)
    J = _run_jax(scen, r_bins, vol_bins)
    if np.any(F > 0):
        denom = np.maximum(np.abs(F), np.abs(J))
        denom = np.where(denom > 1e-300, denom, 1.0)
        rel = np.abs(F - J) / denom
        # 1e-9 leaves comfortable headroom over the 200-scenario bench
        # max of 2.2e-13.
        assert float(rel.max()) < 1e-9, (
            f"scenario {label}: max rel err {rel.max():.3e}\n"
            f"  Fortran: {F}\n  JAX:     {J}")
    else:
        np.testing.assert_array_equal(F, J)
