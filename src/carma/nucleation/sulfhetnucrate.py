"""Heterogeneous H2SO4/H2O nucleation on pre-existing particles.

Uses the Zhao & Turco (1995) classical-nucleation parameters
(``rstar``, ``ftry``) from ``binary_nuc_zhao1995`` and applies a
Fletcher heterogeneous correction factor based on the ratio
``xm = r_preexist / rstar`` and a fixed contact angle of 50°.

Ported from: sulfhetnucrate.F90 (CARMA base).

References:
    - Zhao, J. and R.P. Turco (1995). J. Aerosol Sci. 26, 779-795.
    - Fletcher, N.H. (1958). "Size effect in heterogeneous nucleation."
      J. Chem. Phys. 29, 572-576.
"""

import jax
import jax.numpy as jnp

from carma.constants import DEG2RAD, PI, RGAS
from carma.nucleation.sulfnucrate import binary_nuc_zhao1995
from carma.precision import DTYPE


# Contact angle for Fletcher factor (sulfate on pre-existing sulfate-like
# seed surface). Fortran bakes it as a module-level `cos(50° * DEG2RAD)`.
_COS_CONTACT = jnp.cos(DTYPE(50.0) * DEG2RAD)

# Empirical constants from sulfhetnucrate.F90.
_EXPC_PREFACTOR = DTYPE(2.4e-16)
_EXPC_EXPONENT_NUM = DTYPE(4.51872e11)   # erg K⁻¹ mol⁻¹ · K (before /R/T)


def _fletcher_factor(xm, fm=_COS_CONTACT):
    """Fletcher (1958) geometric factor fv1 for heterogeneous nucleation.

    Two branches by size ratio ``xm = r_pre / rstar``:

    - xm < 1: small pre-existing seed relative to critical cluster
    - xm ≥ 1: large pre-existing seed

    Matches the Fortran `sulfhetnucrate` piecewise formula; see
    Fletcher 1958 for the derivation.
    """
    xm = jnp.asarray(xm, dtype=DTYPE)

    # Branch: xm < 1
    fxm_lt = jnp.sqrt(DTYPE(1.0) - DTYPE(2.0) * fm * xm + xm ** 2)
    fxm_lt_safe = jnp.where(fxm_lt > DTYPE(1e-300), fxm_lt, DTYPE(1.0))
    fv2_lt = (xm - fm) / fxm_lt_safe
    fu2_lt = (DTYPE(1.0) - xm * fm) / fxm_lt_safe
    fv3_lt = (DTYPE(2.0) + fv2_lt) * xm ** 3 * (fv2_lt - DTYPE(1.0)) ** 2
    fv4_lt = DTYPE(3.0) * fm * xm ** 2 * (fv2_lt - DTYPE(1.0))

    # Branch: xm >= 1
    # Guard against xm=0 (not physical here but keep expression stable)
    xm_safe = jnp.where(xm > DTYPE(1e-300), xm, DTYPE(1.0))
    xm1 = DTYPE(1.0) / xm_safe
    fxm_ge = jnp.sqrt(DTYPE(1.0) - DTYPE(2.0) * fm * xm1 + xm1 ** 2)
    fxm_ge_safe = jnp.where(fxm_ge > DTYPE(1e-300), fxm_ge, DTYPE(1.0))
    fu2_ge = (xm1 - fm) / fxm_ge_safe
    fv2_ge = (DTYPE(1.0) - xm1 * fm) / fxm_ge_safe
    denom_v1 = (fv2_ge + DTYPE(1.0)) * fxm_ge_safe ** 2
    denom_v1_safe = jnp.where(
        jnp.abs(denom_v1) > DTYPE(1e-300), denom_v1, DTYPE(1.0))
    v1_ge = (fm ** 2 - DTYPE(1.0)) / denom_v1_safe
    fv3_ge = (DTYPE(2.0) + fv2_ge) * xm1 * v1_ge ** 2
    fv4_ge = DTYPE(3.0) * fm * v1_ge

    lt = xm < DTYPE(1.0)
    fu2 = jnp.where(lt, fu2_lt, fu2_ge)
    fv3 = jnp.where(lt, fv3_lt, fv3_ge)
    fv4 = jnp.where(lt, fv4_lt, fv4_ge)

    fv1 = DTYPE(0.5) * (DTYPE(1.0) + fu2 ** 3 + fv3 + fv4)
    return fv1


def sulfhetnucrate(temp, weight_percent, rh,
                    h2so4, h2so4_cgs, h2o, h2o_cgs,
                    beta1, beta2, r_preexist,
                    gwtmol_h2so4=DTYPE(98.0),
                    gwtmol_h2o=DTYPE(18.0)):
    """Heterogeneous nucleation rate on a pre-existing particle of radius
    ``r_preexist``.

    Algorithm (from sulfhetnucrate.F90):
        1. call binary_nuc_zhao1995 → rstar, ftry (from homogeneous path)
        2. cnucl = 4π rstar²
        3. chom  = h2so4 · h2o · β1 · cnucl
        4. expc  = 2.4e-16 · exp(4.51872e11 / (R·T))
        5. chet  = chom · expc · β2
        6. xm    = r_preexist / rstar
        7. Compute Fletcher factor fv1 by xm<1 or xm≥1 branch
        8. ftry1 = ftry · fv1
        9. If ftry1 < -1000: nucrate = 0
           else: nucrate = chet · exp(ftry1) · 4π r_preexist²

    Args:
        temp, weight_percent, rh: atmospheric state. Same role as in
            ``binary_nuc_zhao1995``.
        h2so4, h2so4_cgs: H2SO4 gas concentration in molecules/cm³ and g/cm³.
        h2o, h2o_cgs: H2O gas concentration in molecules/cm³ and g/cm³.
        beta1: gas-kinetic H2SO4 collision rate coefficient [cm/s].
        beta2: diffusion / sticking correction factor for the seed.
        r_preexist: radius of the pre-existing particle [cm].
        gwtmol_h2so4, gwtmol_h2o: molecular weights [g/mol].

    Returns:
        nucrate: heterogeneous nucleation rate [embryos/s].
            Multiply by number of pre-existing particles to get
            embryos per gridpoint per second.

    Output is 0 when the homogeneous Zhao-Turco path finds no saddle
    (rstar = 0) or when the Fletcher-corrected Gibbs exponent goes
    below -1000 (deep thermodynamic barrier — rate below machine
    underflow anyway).
    """
    temp = jnp.asarray(temp, dtype=DTYPE)
    r_preexist = jnp.asarray(r_preexist, dtype=DTYPE)
    beta1 = jnp.asarray(beta1, dtype=DTYPE)
    beta2 = jnp.asarray(beta2, dtype=DTYPE)

    # Homogeneous parameters
    _, _, rstar, ftry = binary_nuc_zhao1995(
        temp, weight_percent, rh, h2so4, h2so4_cgs, h2o, h2o_cgs,
        beta1, gwtmol_h2so4, gwtmol_h2o,
    )

    # Pre-factor chain
    cnucl = DTYPE(4.0) * PI * rstar ** 2
    chom = jnp.asarray(h2so4, DTYPE) * jnp.asarray(h2o, DTYPE) * beta1 * cnucl
    expc = _EXPC_PREFACTOR * jnp.exp(_EXPC_EXPONENT_NUM / (RGAS * temp))
    chet = chom * expc * beta2

    # Fletcher factor
    rstar_safe = jnp.where(rstar > DTYPE(0.0), rstar, DTYPE(1.0))
    xm = r_preexist / rstar_safe
    fv1 = _fletcher_factor(xm)

    # Fletcher-corrected Gibbs exponent
    ftry1 = ftry * fv1
    exp_ok = ftry1 >= DTYPE(-1000.0)
    gg = jnp.where(exp_ok, jnp.exp(jnp.minimum(ftry1, DTYPE(700.0))), DTYPE(0.0))

    # Seed surface area
    rarea = DTYPE(4.0) * PI * r_preexist ** 2
    nucrate = chet * gg * rarea

    # Zero out when the Zhao-Turco path found no saddle
    nucrate = jnp.where(rstar > DTYPE(0.0), nucrate, DTYPE(0.0))
    return nucrate
