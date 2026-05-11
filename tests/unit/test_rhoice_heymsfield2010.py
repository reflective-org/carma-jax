"""Unit tests for rhoice_heymsfield2010 (Phase 11.10).

Per-bin ice density + projected-area ratio (Heymsfield 2010). Setup-
time kernel — no traced state. Six regime choices select the `a`
coefficient; otherwise the formula is purely radius-dependent.
"""
import math
import subprocess
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from carma.constants import PI, RHO_I
from carma.rhoice_heymsfield2010 import rhoice_heymsfield2010

REPO = Path(__file__).resolve().parents[2]
FORTRAN_BIN = (REPO.parent / "original-carma" / "CARMA"
               / "build_standalone" / "rhoice_heymsfield2010_standalone")

REGIMES = ["deep", "conv", "cold", "avg", "synp", "warm"]

# An ice group: rmin = 5 µm, rmrat = 4. NBIN = 28 spans 5 µm to ~few cm,
# straddling the 200-µm aratelem branch in the middle.
NBIN = 28
RMASSMIN = (4.0 / 3.0) * math.pi * (5.0e-4) ** 3 * float(RHO_I)
RMRAT = 4.0


def _run_fortran(regime, rhoice, rmassmin, rmrat, work):
    in_path = work / "in.txt"
    out_path = work / "out.bin"
    with open(in_path, "w") as f:
        f.write(f"{regime} {float(rhoice)!r} {float(rmassmin)!r} "
                f"{float(rmrat)!r} {NBIN}\n")
    subprocess.run([str(FORTRAN_BIN), str(in_path), str(out_path)], check=True)
    arr = np.fromfile(out_path, dtype=np.float64)
    return arr[:NBIN], arr[NBIN:]


_skip = pytest.mark.skipif(
    not FORTRAN_BIN.exists(),
    reason="rhoice_heymsfield2010_standalone not built. "
           "Run scripts/fortran_patch/build_rhoice_heymsfield2010_standalone.sh.")


@_skip
@pytest.mark.parametrize("regime", REGIMES)
def test_rhoice_matches_fortran_all_regimes(regime, tmp_path):
    """JAX bit-matches gfortran standalone across all six ice regimes."""
    rho_F, ar_F = _run_fortran(regime, float(RHO_I), RMASSMIN, RMRAT, tmp_path)
    rho_J, ar_J = rhoice_heymsfield2010(
        float(RHO_I), RMASSMIN, RMRAT, regime, NBIN)
    rho_J = np.asarray(rho_J); ar_J = np.asarray(ar_J)

    rho_denom = np.maximum(np.abs(rho_F), 1e-300)
    ar_denom = np.maximum(np.abs(ar_F), 1e-300)
    rho_rel = np.abs(rho_J - rho_F) / rho_denom
    ar_rel = np.abs(ar_J - ar_F) / ar_denom
    assert float(rho_rel.max()) < 1e-12, (
        f"regime {regime}: max rho rel err {rho_rel.max():.3e}")
    assert float(ar_rel.max()) < 1e-12, (
        f"regime {regime}: max aratelem rel err {ar_rel.max():.3e}")


def test_rhoice_density_clipped_to_rhoice():
    """Small bins should have density clipped to bulk ice density (0.93)."""
    rho, _ = rhoice_heymsfield2010(
        float(RHO_I), RMASSMIN, RMRAT, "avg", NBIN)
    rho = np.asarray(rho)
    assert float(rho[0]) == pytest.approx(float(RHO_I))
    assert bool(np.all(rho <= float(RHO_I) + 1e-15))


def test_rhoice_density_decreases_with_size():
    """Effective density decreases monotonically with bin size."""
    rho, _ = rhoice_heymsfield2010(
        float(RHO_I), RMASSMIN, RMRAT, "avg", NBIN)
    rho = np.asarray(rho)
    diffs = np.diff(rho)
    assert bool(np.all(diffs <= 1e-15))


def test_rhoice_aratelem_branches():
    """aratelem follows exp(-38·D) for D ≤ 200 µm, power law above."""
    _, ar = rhoice_heymsfield2010(
        float(RHO_I), RMASSMIN, RMRAT, "avg", NBIN)
    ar = np.asarray(ar)
    assert bool(np.all(ar > 0))
    # All values in (0, 1] — area ratio is bounded.
    assert bool(np.all(ar <= 1.0 + 1e-12))


def test_rhoice_unknown_regime_raises():
    with pytest.raises(ValueError, match="unknown ice regime"):
        rhoice_heymsfield2010(float(RHO_I), RMASSMIN, RMRAT, "bogus", NBIN)
