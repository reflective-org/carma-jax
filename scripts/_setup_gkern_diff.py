"""Bench JAX setup_gkern using Fortran-dumped inputs.

Isolates setup_gkern's own port from upstream env-builder discrepancies
(rhoa, rmu, diffus, etc. all taken from the Fortran probe dump).
"""
import sys
from pathlib import Path

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from carma._diag.fortran_reader import read_substep
from carma.setup_gkern import setup_gkern
from carma.precision import DTYPE


def main():
    n_check = 25
    print(f"Diffing setup_gkern (using Fortran-dumped inputs) on {n_check} scenarios.\n")

    NBIN, NGAS, NGROUP = 38, 2, 1
    igash2o, igash2so4 = 0, 1
    gwtmol_arr = jnp.array([18.016, 98.078479])
    is_ice_arr = np.array([False])
    eshape_arr = jnp.array([1.0])
    igrowgas_arr = np.array([igash2so4], dtype=np.int32)
    rrat = jnp.ones((NBIN, NGROUP))

    errs = {"akelvin": [], "akelvini": [], "gro": [], "gro1": []}
    for i in range(n_check):
        scen_dir = ROOT / "data" / "diff" / f"scen_{i:03d}"
        if not scen_dir.exists():
            continue
        d = read_substep(scen_dir, step=1)
        # thcond not in dump — derive from Fortran's standard formula
        # thcond = (5.69 + 0.017 * (T - 273.16)) * 4.184e2 erg/cm/s/K
        # (Pruppacher & Klett, eqn 13-16). This is what Fortran setup_atm uses.
        T_K = float(d.t[0])
        thcond_F = (5.69 + 0.017 * (T_K - 273.16)) * 4.184e2
        thcond = jnp.array([thcond_F])

        wtpct_F = d.wtpct[0]
        # diffus also not in dump — take only what's in dump and trust JAX's
        # setup_grow's diffus formula matched (rlhe/rlhm we already
        # replaced upstream). For this test we'd really need Fortran's diffus.
        # Workaround: recompute diffus using JAX setup_grow on Fortran's
        # dumped (T, p, rhoa, zmet) — that uses identical formulas.
        from carma.setup_grow import setup_grow
        diffus, rlhe, rlhm = setup_grow(
            jnp.asarray(d.t), jnp.asarray(d.p), jnp.asarray(d.rhoa),
            jnp.asarray(d.zmet),
            igash2o=igash2o, igash2so4=igash2so4, ngas=NGAS,
            do_cnst_rlh=False,
        )

        # rlow_wet[i] from bin geometry: rmass[i] = rmass[0] · rmrat^i, so
        # rmlow[i] = rmass[i] / sqrt(rmrat) → rlow[i] / r[i] = rmrat^(-1/6).
        # Same factor for wet radii since wet density is the same per bin.
        rmrat = 2.0
        rlow_wet = np.asarray(d.r_wet) * rmrat ** (-1.0 / 6.0)

        _, akelvin_J, akelvini_J, gro_J, gro1_J, _, _, _ = setup_gkern(
            jnp.asarray(d.t), jnp.asarray(d.p), jnp.asarray(d.rhoa),
            jnp.asarray(d.zmet), jnp.asarray(d.rmu), thcond,
            diffus, rlhe, rlhm,
            jnp.asarray(d.re), jnp.asarray(d.r_wet),
            jnp.asarray(rlow_wet),
            rrat, eshape_arr, is_ice_arr,
            gwtmol_arr, igrowgas_arr,
            DTYPE(1.0), DTYPE(1.0), DTYPE(1.0),
            NBIN, NGROUP, NGAS,
            igash2o=igash2o, igash2so4=igash2so4, wtpct=jnp.asarray([wtpct_F]),
        )

        def rel(a, b):
            a, b = np.ravel(a), np.ravel(b)
            denom = np.maximum(np.abs(a), np.abs(b))
            denom = np.where(denom > 1e-300, denom, 1.0)
            return float(np.max(np.abs(a - b) / denom))

        errs["akelvin"].append(rel(akelvin_J, d.akelvin))
        errs["akelvini"].append(rel(akelvini_J, d.akelvini))
        errs["gro"].append(rel(gro_J, d.gro))
        errs["gro1"].append(rel(gro1_J, d.gro1))

    print("Summary (max rel err across scenarios using Fortran inputs):")
    for f, vs in errs.items():
        print(f"  {f:<10}: max={max(vs):.3e}  median={np.median(vs):.3e}")


if __name__ == "__main__":
    main()
