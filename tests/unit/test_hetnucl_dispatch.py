"""Phase 11.11 dispatch-context unit tests for hetnucl (PMC heterogeneous nucleation).

Routes hetnucl through Fortran's full upstream subroutine, with p/gc/surfctia
overridden in cstate. This is the highest-routing-risk kernel in Phase 11:
hetnucl reads pressure, gas concentration, and surface tension from cstate,
plus constants (RHO_I, BK, RGAS) — exactly the surface the dispatch bench
is designed to cover.

The kernel has known catastrophic cancellation in
  phih = sqrt(1 - 2·m·x + x²)
at large x = r_bin / ag (i.e., bin sizes well above the critical germ).
PMC particles physically live at 20–200 nm (smallest 9 bins), where the
kernel is bit-exact (≤1e-9). Tolerance at 1e-3 covers the large-bin tail
that's not physically populated.

Skipped if the diagnostic binary isn't built; build via
  bash scripts/fortran_patch/apply_nuc2_diagnostic_patch.sh
"""
import math
import subprocess
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from carma.nucleation.hetnucl import hetnucl

REPO = Path(__file__).resolve().parents[2]
DIAG_BIN = (REPO.parent / "original-carma" / "CARMA" / "build"
            / "test_nuc2test_diagnostic")

NBIN = 16
RMIN_CM, RMRAT, RHO_PARTICLE = 1.0e-7, 4.0, 1.78
GWTMOL_H2O = 18.016


def _bin_grid():
    vmin = (4.0 / 3.0) * math.pi * RMIN_CM**3 * RHO_PARTICLE
    rmass = vmin * RMRAT ** np.arange(NBIN)
    r = (3.0 * rmass / (4.0 * math.pi * RHO_PARTICLE)) ** (1.0 / 3.0)
    return r


def _run_fortran(scen, work):
    scen_path = work / "scen.txt"
    out_dir = work / "out"
    out_dir.mkdir()
    with open(scen_path, "w") as f:
        f.write(
            "210.0 200.0 0.95 100.0 2.5e-6 1.5  "
            f"{scen['T']!r} {scen['ssi']!r} {scen['ssl']!r}  "
            f"{scen['akelvin']!r} {scen['akelvini']!r} {scen['pconmax']!r}  "
            f"0.0 {scen['p_cgs']!r} {scen['gc_h2o']!r}\n"
        )
    subprocess.run([str(DIAG_BIN), str(scen_path), str(out_dir),
                    "1", "hetnucl"], check=True, capture_output=True)
    arr = np.fromfile(out_dir / "substep_0001_hetnucl_probe.bin",
                       dtype=np.float64).reshape((NBIN, 2, 2), order='F')
    p_cgs = float(np.fromfile(out_dir / "substep_0001_p.bin",
                                dtype=np.float64)[0])
    gc_cgs = float(np.fromfile(out_dir / "substep_0001_gc.bin",
                                 dtype=np.float64)[0])
    surfctia = float(np.fromfile(out_dir / "substep_0001_surfctia.bin",
                                   dtype=np.float64)[0])
    return arr[:, 0, 1], p_cgs, gc_cgs, surfctia


def _run_jax(scen, r_bins, p_cgs, gc_cgs, surfctia):
    return np.asarray(hetnucl(
        t_val=jnp.float64(scen['T']),
        p_val=jnp.float64(p_cgs),
        supsati_val=jnp.float64(scen['ssi']),
        gc_h2o=jnp.float64(gc_cgs),
        gwtmol_h2o=jnp.float64(GWTMOL_H2O),
        surfctia_val=jnp.float64(surfctia),
        r_bins=jnp.asarray(r_bins),
        pconmax_val=jnp.float64(scen['pconmax']),
    ))


_skip = pytest.mark.skipif(
    not DIAG_BIN.exists(),
    reason="nuc2test diagnostic binary not built. "
           "Run scripts/fortran_patch/apply_nuc2_diagnostic_patch.sh.")


SCENARIOS = [
    # label, T, ssi, ssl, akelvin, akelvini, pconmax, p_cgs, gc_h2o
    ("pmc_typical",      210.0, 1.50, -0.50, 2.0e-7, 2.0e-7, 1e3, 100.0,  1e-10),
    ("very_cold_pmc",    150.0, 2.00, -0.50, 2.0e-7, 2.0e-7, 1e3,  10.0,  1e-11),
    ("warm_pmc_edge",    220.0, 1.00, -0.50, 2.0e-7, 2.0e-7, 1e3, 500.0,  1e-9),
    ("high_ssi",         200.0, 4.00, -0.50, 2.0e-7, 2.0e-7, 1e3,  50.0,  5e-10),
    # Gate-off cases
    ("p_gate_off",       210.0, 1.50, -0.50, 2.0e-7, 2.0e-7, 1e3, 5000.0, 1e-10),
    ("ssi_zero",         210.0, -0.10, -0.50, 2.0e-7, 2.0e-7, 1e3, 100.0, 1e-10),
    ("few_particles",    210.0, 1.50, -0.50, 2.0e-7, 2.0e-7, 1e-50, 100.0, 1e-10),
]


@_skip
@pytest.mark.parametrize(
    "label,T,ssi,ssl,akelvin,akelvini,pconmax,p_cgs,gc_h2o", SCENARIOS)
def test_hetnucl_dispatch_matches_jax(
        label, T, ssi, ssl, akelvin, akelvini, pconmax, p_cgs, gc_h2o,
        tmp_path):
    r_bins = _bin_grid()
    scen = dict(T=T, ssi=ssi, ssl=ssl, akelvin=akelvin, akelvini=akelvini,
                pconmax=pconmax, p_cgs=p_cgs, gc_h2o=gc_h2o)
    F, p_dump, gc_dump, surfctia = _run_fortran(scen, tmp_path)
    J = _run_jax(scen, r_bins, p_dump, gc_dump, surfctia)
    if np.any(F > 0):
        denom = np.maximum(np.abs(F), np.abs(J))
        denom = np.where(denom > 1e-300, denom, 1.0)
        rel = np.abs(F - J) / denom
        # Physically populated bins (0..8 = up to ~200 nm) must be tight;
        # catastrophic-cancellation tail at bins 9..15 gets a looser tol.
        small_bin_rel = rel[:9][F[:9] > 0]
        large_bin_rel = rel[9:][F[9:] > 0]
        if small_bin_rel.size:
            # 1e-8 covers the bin 8 (~200 nm) edge where x = r/ag is
            # large enough to start triggering the phih cancellation;
            # bins 0..7 (≤80 nm) are tight at ~1e-13.
            assert float(small_bin_rel.max()) < 1e-8, (
                f"scenario {label}: small-bin (PMC-relevant) max rel err "
                f"{small_bin_rel.max():.3e}")
        if large_bin_rel.size:
            # 1e-3 leaves headroom over 200-scenario bench max 4.7e-5
            assert float(large_bin_rel.max()) < 1e-3, (
                f"scenario {label}: large-bin (cancellation-affected) max "
                f"rel err {large_bin_rel.max():.3e}")
    else:
        np.testing.assert_array_equal(F, J)
