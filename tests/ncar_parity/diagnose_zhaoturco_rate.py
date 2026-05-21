"""Diagnose the ZhaoTurco rate discrepancy vs NCAR Fortran.

Conclusion from `tests/ncar_parity/diagnose_sulfatetest_gap.py`:
the total-particle-mass and gas residuals at end of carma_sulfatetest
differ between faithful JAX (ZhaoTurco) and the NCAR bench by ~5% and
46% respectively, with the particle population centroid shifted by
~+0.8 bin in faithful. Vehkamäki agrees within ~1% on gas.

This script narrows it to a single test point: at the initial conditions
of `carma_sulfatetest` (T=250 K, p=90 hPa, [H2SO4]=2.6e8 cm⁻³,
RH=1.5 %), we compute the ZhaoTurco rate and the rate implied by the
bench's first-step nucleation deposit.

Result (sample, see comments below):
  - Our `binary_nuc_zhao1995` rate         : 4.5e-4 #/cm³/s
  - Bench-implied effective rate (step 1)  : 7.0e-4 #/cm³/s
  - Ratio (ours / Fortran's effective)    : 0.64

So our ZhaoTurco systematically under-predicts the nucleation rate by
~36 % at this point. With the rate proportional to
`exp(-gstar/kT)`, a 36 % factor corresponds to an `ftry` (= -gstar/kT)
difference of only ~0.31 — meaning gstar (~1.36e-12 erg) must differ by
only ~0.8 %.

Vehkamäki on the same conditions gives 6.2e-2 #/cm³/s and matches the
F90 bench's per-test totals within 1 %, so the surrounding `sulfnuc`
dispatcher and nucbin placement are correct — the bug is inside
`binary_nuc_zhao1995` specifically.

Suspect terms (all small, but small in input → small in `ftry` →
amplified by `exp`):
  - `sigma = sulfate_surf_tens(wstar, T)` — surface-tension table lookup
  - `dstar` (density linear blend at wstar)
  - `pa`, `pb` linear blend at saddle (raln, rhln)

Next step (not yet done): build a side-by-side debug print of
`(wstar, dstar, sigma, ystar, rstar, gstar, ftry)` against a hand-run
of the F90 source at the same conditions. The F90 doesn't expose these
intermediates to its bench file, so this requires either patching the
F90 with extra `write()` statements or transcribing the F90 step-by-step
and comparing with a Python mock.

This script is here as a starting point for that work — runs the JAX
ZhaoTurco at the canonical test conditions and prints every
intermediate. Pair with a similar patched-F90 run for the smoking gun.
"""
import math
import sys
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from carma.constants import AVG, BK, PI, RGAS, R_AIR
from carma.nucleation.sulfnucrate import (
    _DNC0, _DNC1, _DNPOT, _DNWTP,
    binary_nuc_vehk2002, binary_nuc_zhao1995,
)
from carma.precision import DTYPE
from carma.sulfate_utils import (
    sulfate_density, sulfate_surf_tens, wtpct_tabaz,
)
from carma.vapor_pressure import vaporp_h2o_murphy2005


# Test conditions from carma_sulfatetest.F90
T = 250.0
P_PA = 9000.0
MMR_H2O = 100e-6
MMR_H2SO4 = 0.1e-9 * (98.0 / 29.0)
GW_H2SO4 = 98.0
GW_H2O = 18.0
RHO_SULFATE = 1.923


def dump_zhaoturco():
    rho_air = P_PA * 10.0 / (float(R_AIR) * T)
    h2so4_cgs = MMR_H2SO4 * rho_air
    h2o_cgs = MMR_H2O * rho_air
    h2so4 = h2so4_cgs * float(AVG) / GW_H2SO4
    h2o = h2o_cgs * float(AVG) / GW_H2O

    T_arr = jnp.atleast_1d(T)
    pvapl_h2o, _ = vaporp_h2o_murphy2005(T_arr)
    h2o_vp = float(pvapl_h2o[0])
    h2o_pp = h2o_cgs * float(RGAS) / GW_H2O * T
    rh = h2o_pp / h2o_vp
    wtpct = float(wtpct_tabaz(T, h2o_cgs, h2o_vp))
    beta1 = math.sqrt(float(RGAS) * T / (2.0 * float(PI) * GW_H2SO4))

    print(f"Conditions: T={T} K, p={P_PA} Pa, RH={rh:.4f}, "
          f"wtpct_bulk={wtpct:.2f}")
    print(f"  h2so4 = {h2so4:.3e} molec/cm^3  ({h2so4_cgs:.3e} g/cm^3)")
    print(f"  h2o   = {h2o:.3e} molec/cm^3  ({h2o_cgs:.3e} g/cm^3)")
    print(f"  beta1 = {beta1:.3e} cm/s")

    # Direct JAX ZhaoTurco
    rate, m, rstar, ftry = binary_nuc_zhao1995(
        T, wtpct, rh, h2so4, h2so4_cgs, h2o, h2o_cgs,
        beta1, GW_H2SO4, GW_H2O,
    )
    print(f"\nbinary_nuc_zhao1995:")
    print(f"  rate     = {float(rate):.3e} #/cm^3/s")
    print(f"  rstar    = {float(rstar):.3e} cm   ({float(rstar)*1e7:.3f} nm)")
    print(f"  m_cluster_dry = {float(m):.3e} g")
    print(f"  ftry     = {float(ftry):.3f}")

    # Vehkamaki for comparison
    rate_v, m_v, rstar_v = binary_nuc_vehk2002(T, rh, h2so4, GW_H2SO4)
    print(f"\nbinary_nuc_vehk2002 (comparison):")
    print(f"  rate     = {float(rate_v):.3e} #/cm^3/s")
    print(f"  rstar    = {float(rstar_v):.3e} cm   ({float(rstar_v)*1e7:.3f} nm)")

    # The bench shows step 1 nucleates 1.27 #/cm^3 into bin 5 over 1800 s.
    # Effective rate (averaged over substep) ≈ 7e-4 #/cm^3/s.
    print(f"\nBench-implied effective rate over step 1 (1800 s):")
    print(f"  pc_bench = 1.27 #/cm^3 → rate ≈ 7.0e-4 #/cm^3/s")
    print(f"  ratio (ours / Fortran's effective): "
          f"{float(rate) / 7.0e-4:.2f}")


if __name__ == "__main__":
    dump_zhaoturco()
