"""Phase 11.11 dispatch-context unit tests for melticel.

Routes the ice-melting kludge through Fortran's upstream
`subroutine melticel(carma, cstate, iz, rc)`. The Fortran kernel registers
the ICE→sulfate I_ICEMELT process (group 2 → element 1) — different
direction than freezaerl_* — so the rnuclg lives in
`rnuclg(:, ignucfrom=2, ignucto=1)`, i.e. Python slice `[:, 1, 0]`.

Rate is exactly 100/s when gated (T > T0, pconmax > FEW_PC), so the
comparison is bit-exact.

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

from carma.nucleation.melticel import melticel

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
            "280.0 200.0 0.95 100.0 2.5e-6 1.5  "
            f"{scen['T']!r} 0.0 0.0  "
            f"2.0e-7 2.0e-7 {scen['pconmax']!r}  "
            "0.0 -1.0 -1.0\n"
        )
    subprocess.run([str(DIAG_BIN), str(scen_path), str(out_dir),
                    "1", "melticel"], check=True, capture_output=True)
    arr = np.fromfile(out_dir / "substep_0001_melticel_probe.bin",
                       dtype=np.float64).reshape((NBIN, 2, 2), order='F')
    # Different rnuclg index: source group=2 (ice), target group=1 (sulfate)
    return arr[:, 1, 0]


def _run_jax(scen):
    return np.asarray(melticel(
        t_val=jnp.float64(scen['T']),
        pconmax_val=jnp.float64(scen['pconmax']),
        nbin=NBIN,
    ))


_skip = pytest.mark.skipif(
    not DIAG_BIN.exists(),
    reason="nuc2test diagnostic binary not built. "
           "Run scripts/fortran_patch/apply_nuc2_diagnostic_patch.sh.")


SCENARIOS = [
    # label, T, pconmax
    ("just_above_T0",   273.5,  1e3),
    ("warm",            300.0,  1e3),
    ("hot",             340.0,  1e4),
    # Gate-off
    ("at_T0",           273.16, 1e3),
    ("below_T0",        260.0,  1e3),
    ("few_particles",   300.0,  1e-50),
]


@_skip
@pytest.mark.parametrize("label,T,pconmax", SCENARIOS)
def test_melticel_dispatch_matches_jax(label, T, pconmax, tmp_path):
    scen = dict(T=T, pconmax=pconmax)
    F = _run_fortran(scen, tmp_path)
    J = _run_jax(scen)
    # Constant 100/s gate kernel → bit-exact
    np.testing.assert_array_equal(F, J, err_msg=f"scenario {label}")
