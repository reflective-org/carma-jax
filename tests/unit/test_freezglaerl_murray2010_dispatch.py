"""Phase 11.11 dispatch-context unit tests for freezglaerl_murray2010.

Routes Murray 2010 glassy-aerosol heterogeneous freezing through Fortran's
upstream `subroutine freezglaerl_murray2010(carma, cstate, iz, rc)` rather
than the standalone formula. The cstate inject overrides T, ssi,
supsatiold, pconmax so the gate fires deterministically.

Skipped if the diagnostic binary isn't built; build via
  bash scripts/fortran_patch/apply_nuc2_diagnostic_patch.sh
"""
import subprocess
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from carma.nucleation.freezglaerl_murray2010 import freezglaerl_murray2010

REPO = Path(__file__).resolve().parents[2]
DIAG_BIN = (REPO.parent / "original-carma" / "CARMA" / "build"
            / "test_nuc2test_diagnostic")
NBIN = 16


def _run_fortran(scen, work):
    scen_path = work / "scen.txt"
    out_dir = work / "out"
    out_dir.mkdir()
    with open(scen_path, "w") as f:
        f.write(
            "210.0 200.0 0.95 100.0 2.5e-6 1.5  "
            f"{scen['T']!r} {scen['ssi']!r} {scen['ssl']!r}  "
            f"{scen['akelvin']!r} {scen['akelvini']!r} {scen['pconmax']!r}  "
            f"{scen['ssi_old']!r} -1.0 -1.0\n"
        )
    subprocess.run([str(DIAG_BIN), str(scen_path), str(out_dir),
                    "1", "murray"], check=True, capture_output=True)
    arr = np.fromfile(out_dir / "substep_0001_freezglaerl_murray_probe.bin",
                       dtype=np.float64).reshape((NBIN, 2, 2), order='F')
    return arr[:, 0, 1]


def _run_jax(scen):
    # Diagnostic binary uses dtime = 1.0
    return np.asarray(freezglaerl_murray2010(
        t_val=jnp.float64(scen['T']),
        supsati_val=jnp.float64(scen['ssi']),
        supsati_old_val=jnp.float64(scen['ssi_old']),
        pconmax_val=jnp.float64(scen['pconmax']),
        dtime=jnp.float64(1.0),
        nbin=NBIN,
    ))


_skip = pytest.mark.skipif(
    not DIAG_BIN.exists(),
    reason="nuc2test diagnostic binary not built. "
           "Run scripts/fortran_patch/apply_nuc2_diagnostic_patch.sh.")


SCENARIOS = [
    # label, T, ssi, ssi_old, ssl, akelvin, akelvini, pconmax
    ("cold_rising",       200.0, 0.50, 0.10, -0.50, 2.0e-7, 2.0e-7, 1e3),
    ("near_ssmin",        200.0, 0.22, 0.10, -0.50, 2.0e-7, 2.0e-7, 1e3),
    ("ssmax_clipped",     200.0, 0.95, 0.10, -0.50, 2.0e-7, 2.0e-7, 1e3),
    ("ssi_old_active",    200.0, 0.50, 0.30, -0.50, 2.0e-7, 2.0e-7, 1e3),
    # Gate-off cases
    ("warm_skip",         220.0, 0.50, 0.10, -0.50, 2.0e-7, 2.0e-7, 1e3),
    ("low_ssi_skip",      200.0, 0.15, 0.10, -0.50, 2.0e-7, 2.0e-7, 1e3),
    ("decreasing_skip",   200.0, 0.30, 0.50, -0.50, 2.0e-7, 2.0e-7, 1e3),
]


@_skip
@pytest.mark.parametrize(
    "label,T,ssi,ssi_old,ssl,akelvin,akelvini,pconmax", SCENARIOS)
def test_freezglaerl_murray2010_dispatch_matches_jax(
        label, T, ssi, ssi_old, ssl, akelvin, akelvini, pconmax, tmp_path):
    scen = dict(T=T, ssi=ssi, ssi_old=ssi_old, ssl=ssl,
                akelvin=akelvin, akelvini=akelvini, pconmax=pconmax)
    F = _run_fortran(scen, tmp_path)
    J = _run_jax(scen)
    if np.any(F > 0):
        denom = np.maximum(np.abs(F), np.abs(J))
        denom = np.where(denom > 1e-300, denom, 1.0)
        rel = np.abs(F - J) / denom
        # 1e-13 leaves headroom over 200-scenario bench max 1.4e-14.
        assert float(rel.max()) < 1e-13, (
            f"scenario {label}: max rel err {rel.max():.3e}\n"
            f"  Fortran: {F[0]}\n  JAX:     {J[0]}")
    else:
        np.testing.assert_array_equal(F, J)
