"""Term-by-term diff of binary_nuc_zhao1995 (JAX) vs Fortran (patched).

Companion to `diagnose_zhaoturco_rate.py`. Reads
``carma-app/original-carma/CARMA/build/zhaoturco_dump.txt`` produced by
the patched ``sulfnucrate.F90`` (Phase 6.6c), recomputes the same
intermediates in JAX at the exact F90 inputs, and prints a side-by-side
table with relative differences.

Outcome of the previous step
----------------------------
At the carma_sulfatetest initial conditions (T=250 K, [H2SO4]≈2.73e8,
[H2O]≈4.19e14), the patched Fortran's first call reports::

    nucrate_cgs ≈ 7.03e-4  #/cm^3/s

matching the bench-implied effective rate of 7e-4 within 1 %. Our JAX
``binary_nuc_zhao1995`` at the same point gave 4.5e-4 (ratio 0.64), so
the bug is inside the rate routine itself. This script localizes which
term diverges.

Reads
-----
zhaoturco_dump.txt (Fortran)
    Key-value pairs ``name <num>`` per line, produced by the patched
    F90. Path discovered via the FORT_DUMP env var or the canonical
    location ``../original-carma/CARMA/build/zhaoturco_dump.txt``.

Writes
------
stdout
    A table with one row per intermediate: name, Fortran value, JAX
    value, abs diff, rel diff.
"""
import math
import os
import sys
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from carma.constants import AVG, BK, PI, RGAS
from carma.nucleation.sulfnucrate import (
    _DNC0, _DNC1, _DNPOT, _DNWF, _DNWTP, _NTAB,
)
from carma.precision import DTYPE
from carma.sulfate_utils import sulfate_density, sulfate_surf_tens


DEFAULT_DUMP = (
    Path(__file__).resolve().parent / "fixtures" / "zhaoturco_dump_fortran.txt"
)


def parse_fort_dump(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            continue
        out[parts[0]] = float(parts[1])
    return out


def jax_intermediates(temp, weight_percent, h2so4_cgs, h2o_cgs, beta1,
                      gwtmol_h2so4=98.0, gwtmol_h2o=18.0):
    """Mirror binary_nuc_zhao1995 step by step, returning every term."""
    temp = jnp.asarray(temp, dtype=DTYPE)
    weight_percent = jnp.asarray(weight_percent, dtype=DTYPE)
    h2so4_cgs = jnp.asarray(h2so4_cgs, dtype=DTYPE)
    h2o_cgs = jnp.asarray(h2o_cgs, dtype=DTYPE)
    beta1 = jnp.asarray(beta1, dtype=DTYPE)

    h2so4 = h2so4_cgs * float(AVG) / gwtmol_h2so4
    h2o = h2o_cgs * float(AVG) / gwtmol_h2o

    wtmolr = gwtmol_h2so4 / gwtmol_h2o
    h2oln = jnp.log(h2o_cgs * (float(RGAS) / gwtmol_h2o) * temp)
    h2so4ln = jnp.log(h2so4_cgs * (float(RGAS) / gwtmol_h2so4) * temp)

    dens_tab = _DNC0 + _DNC1 * temp
    wtp = _DNWTP
    cw = (DTYPE(22.7490) + DTYPE(0.0424817) * wtp
          - DTYPE(0.0567432) * jnp.sqrt(wtp)
          - DTYPE(0.000621533) * wtp ** 2)
    dw = (DTYPE(-5850.24) + DTYPE(21.9744) * wtp
          - DTYPE(44.5210) * jnp.sqrt(wtp)
          - DTYPE(0.384362) * wtp ** 2)
    wvp = jnp.exp(cw + dw / temp)
    wvpln = jnp.log(wvp * DTYPE(1013250.0) / DTYPE(1013.25))
    pb = h2oln - wvpln

    t0_kulm = DTYPE(340.0)
    t_crit_kulm = DTYPE(905.0)
    seqln_base = DTYPE(-10156.0) / t0_kulm + DTYPE(16.259)
    factor_kulm = (
        DTYPE(-1.0) / temp + DTYPE(1.0) / t0_kulm
        + DTYPE(0.38) / (t_crit_kulm - t0_kulm)
        * (DTYPE(1.0) + jnp.log(t0_kulm / temp) - t0_kulm / temp)
    )
    seqln_pure = seqln_base + DTYPE(10156.0) * factor_kulm
    seqln = seqln_pure - _DNPOT / (DTYPE(8.3143) * temp)
    seqln = seqln + jnp.log(DTYPE(1013250.0))
    pa = h2so4ln - seqln

    c1 = pa - pb * wtmolr
    c2 = pa * _DNWF + pb * (DTYPE(1.0) - _DNWF) * wtmolr

    dwtp = _DNWTP[1:] - _DNWTP[:-1]
    grad = (dens_tab[1:] - dens_tab[:-1]) / dwtp
    dens1_mid = (grad[:-1] * dwtp[1:] + grad[1:] * dwtp[:-1]) / (
        dwtp[:-1] + dwtp[1:]
    )
    dens1_lo = grad[0]
    dens1_hi = grad[-1]
    dens1_full = jnp.concatenate([
        jnp.array([dens1_lo], dtype=DTYPE),
        dens1_mid,
        jnp.array([dens1_hi], dtype=DTYPE),
    ])

    fct = c1 + c2 * DTYPE(100.0) * dens1_full / dens_tab

    sign_products = fct[:-1] * fct[1:]
    has_crossing = sign_products <= DTYPE(0.0)
    rev = has_crossing[::-1]
    first_rev_idx = jnp.argmax(rev)
    i_star = DTYPE(44) - first_rev_idx.astype(DTYPE)
    i_lo = jnp.clip(i_star.astype(jnp.int32), 0, _NTAB - 2)
    i_hi = i_lo + 1

    fct_lo = fct[i_lo]
    fct_hi = fct[i_hi]
    denom = fct_hi - fct_lo
    safe_denom = jnp.where(jnp.abs(denom) > DTYPE(1e-300), denom, DTYPE(1.0))
    xfrac = fct_hi / safe_denom
    xfrac = jnp.clip(xfrac, DTYPE(0.0), DTYPE(1.0))

    wstar = _DNWTP[i_hi] * (DTYPE(1.0) - xfrac) + _DNWTP[i_lo] * xfrac
    dstar = dens_tab[i_hi] * (DTYPE(1.0) - xfrac) + dens_tab[i_lo] * xfrac
    rhln = pb[i_hi] * (DTYPE(1.0) - xfrac) + pb[i_lo] * xfrac
    raln = pa[i_hi] * (DTYPE(1.0) - xfrac) + pa[i_lo] * xfrac
    wfstar = wstar / DTYPE(100.0)

    sigma = sulfate_surf_tens(wstar, temp)

    ystar = dstar * float(RGAS) * temp * (
        wfstar / gwtmol_h2so4 * raln
        + (DTYPE(1.0) - wfstar) / gwtmol_h2o * rhln
    )
    rstar = DTYPE(2.0) * sigma / ystar
    radius_cluster = jnp.maximum(rstar, DTYPE(0.0))
    r2 = radius_cluster * radius_cluster
    gstar = (DTYPE(4.0) * float(PI) / DTYPE(3.0)) * r2 * sigma
    rpr = DTYPE(4.0) * float(PI) * r2 * h2o * beta1
    rpre = rpr * h2so4

    denom_frac = DTYPE(1.0) + wtmolr * (DTYPE(1.0) - wfstar) / jnp.maximum(
        wfstar, DTYPE(1e-30))
    fracmol = DTYPE(1.0) / denom_frac
    zphi = jnp.arctan(fracmol)
    zeld = DTYPE(0.25) / jnp.sin(zphi) ** 2

    ftry = -gstar / float(BK) / temp
    exhom = jnp.exp(jnp.minimum(ftry, DTYPE(28.0)))

    rho_h2so4_wet = sulfate_density(weight_percent, temp)
    mass_cluster_wet = (
        DTYPE(4.0) * float(PI) / DTYPE(3.0)) * rho_h2so4_wet * r2 * radius_cluster
    mass_cluster_dry = mass_cluster_wet * wfstar
    nucrate_cgs = rpre * zeld * exhom

    return {
        "saddle_i": int(i_lo) + 1,  # convert to 1-based to match F90
        "xfrac": float(xfrac),
        "dnwtp_i": float(_DNWTP[i_lo]),
        "dnwtp_ip1": float(_DNWTP[i_hi]),
        "pa_i": float(pa[i_lo]),
        "pa_ip1": float(pa[i_hi]),
        "pb_i": float(pb[i_lo]),
        "pb_ip1": float(pb[i_hi]),
        "c1_i": float(c1[i_lo]),
        "c1_ip1": float(c1[i_hi]),
        "c2_i": float(c2[i_lo]),
        "c2_ip1": float(c2[i_hi]),
        "fct_i": float(fct[i_lo]),
        "fct_ip1": float(fct[i_hi]),
        "dens_i": float(dens_tab[i_lo]),
        "dens_ip1": float(dens_tab[i_hi]),
        "wstar": float(wstar),
        "dstar": float(dstar),
        "rhln": float(rhln),
        "raln": float(raln),
        "sigma": float(sigma),
        "ystar": float(ystar),
        "radius_cluster": float(radius_cluster),
        "gstar": float(gstar),
        "ftry": float(ftry),
        "rho_H2SO4_wet": float(rho_h2so4_wet),
        "mass_cluster_dry": float(mass_cluster_dry),
        "rpr": float(rpr),
        "rpre": float(rpre),
        "fracmol": float(fracmol),
        "zphi": float(zphi),
        "zeld": float(zeld),
        "exhom": float(exhom),
        "nucrate_cgs": float(nucrate_cgs),
    }


def main() -> int:
    dump_path = Path(os.environ.get("FORT_DUMP", str(DEFAULT_DUMP)))
    if not dump_path.exists():
        print(f"ERROR: no Fortran dump at {dump_path}", file=sys.stderr)
        print("Run the patched test_sulfate first, see "
              "docs/decisions/0001-ncar-test-parity.md.", file=sys.stderr)
        return 1
    fort = parse_fort_dump(dump_path)

    # Inputs come from the Fortran dump so we compare on the SAME state.
    temp = fort["temp"]
    weight_percent = fort["weight_percent"]
    h2so4_cgs = fort["h2so4_cgs"]
    h2o_cgs = fort["h2o_cgs"]
    beta1 = fort["beta1"]

    jx = jax_intermediates(temp, weight_percent, h2so4_cgs, h2o_cgs, beta1)

    # Compare every key the JAX side produced; report rel diff.
    keys = list(jx.keys())
    width_name = max(len(k) for k in keys)
    print(f"{'name':<{width_name}}  {'F90':>16}  {'JAX':>16}  "
          f"{'absdiff':>14}  {'reldiff':>10}")
    print("-" * (width_name + 16 + 16 + 14 + 10 + 8))
    for k in keys:
        if k not in fort:
            continue
        f = fort[k]
        j = jx[k]
        absdiff = j - f
        denom = max(abs(f), 1e-30)
        rel = absdiff / denom
        print(f"{k:<{width_name}}  {f:>16.8e}  {j:>16.8e}  "
              f"{absdiff:>14.4e}  {rel:>10.2%}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
