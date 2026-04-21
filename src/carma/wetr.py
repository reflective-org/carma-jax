"""Wet radius and wet density for hydrophilic aerosol (CARMA-JAX).

Three swelling parameterisations are ported:

- ``I_NO_SWELLING`` — dry particles, trivial passthrough
- ``I_PETTERS`` — κ-Köhler from Petters & Kreidenweis (ACP 2007) +
  Yu et al. (JAMES 2015) low-T extension. Used for sulfate+dust
  mixtures where a per-element κ is known.
- ``I_WTPCT_H2SO4`` — binary H2SO4/H2O aerosol using the Tabazadeh
  weight-percent (``sulfate_utils.wtpct_tabaz``) + sulfate density
  and a Kelvin correction. Used for pure stratospheric sulfate.

Not ported in this phase (Phase 7.2 scope is sulfate-focused):

- ``I_FITZGERALD`` — Fitzgerald (1975), sea-salt and other ionic.
  9 composition options. Required for sea-salt / tropospheric
  chloride work; defer to a future phase.
- ``I_GERBER`` — Gerber (1985), 4 composition options (NH42SO4,
  urban, rural, sea-salt). Defer with Fitzgerald.

Calling a non-implemented branch raises ``NotImplementedError`` at
trace time (Python-level, since the branch is chosen by the static
``irhswell`` config flag).

Ported from: wetr.F90 (CARMA base).
"""

from functools import partial

import jax
import jax.numpy as jnp

from carma.constants import AVG, BK, PI, RGAS, RHO_W, WTMOL_H2O
from carma.enums import SwellMethod
from carma.precision import DTYPE
from carma.sulfate_utils import sulfate_density, wtpct_tabaz


# Parameters from the Fortran:
_RH_CAP = DTYPE(0.995)   # upper cap on RH for Köhler (singularity avoidance)
_T_KOHLER_LO = DTYPE(190.0)  # below this T, Yu 2015 vapor-pressure rescale


def _pvap_h2o_murphy(T):
    """Murphy & Koop (2005) liquid-water vapour pressure, scaled as the
    Fortran's inline ``pvap_h2o``: ``T / (10 * exp(...))``.

    Returns a dimension identical to the Fortran inline function (not
    pvapl in dyn/cm^2 — this is only used as a *ratio* in the Yu 2015
    low-T correction so the scale factor cancels).
    """
    T = jnp.asarray(T, dtype=DTYPE)
    tt = T
    return T / (DTYPE(10.0) * jnp.exp(
        DTYPE(54.842763)
        - DTYPE(6763.22) / tt
        - DTYPE(4.210) * jnp.log(tt)
        + DTYPE(0.000367) * tt
        + jnp.tanh(DTYPE(0.0415) * (tt - DTYPE(218.8))) * (
            DTYPE(53.878)
            - DTYPE(1331.22) / tt
            - DTYPE(9.44523) * jnp.log(tt)
            + DTYPE(0.014025) * tt
        )
    ))


def _wetr_petters(rdry, rh, temp, kappa):
    """κ-Köhler wet radius (Petters & Kreidenweis 2007, Eq. 6).

    ``rwet = rdry * (1 + RH·κ/(1-RH))^(1/3)``

    At very cold temperatures (T ≤ 190 K), Yu et al. (JAMES 2015)
    rescale RH by the vapour-pressure ratio to avoid the pathology
    where the pure κ-Köhler form produces unphysical shrinking.
    """
    rh_clip = jnp.clip(rh, jnp.finfo(DTYPE).tiny, _RH_CAP)

    # Low-T rescale: RH_eff = RH · pvap(T) / pvap(190)
    rh190 = rh * _pvap_h2o_murphy(temp) / _pvap_h2o_murphy(_T_KOHLER_LO)
    rh190_clip = jnp.clip(rh190, jnp.finfo(DTYPE).tiny, _RH_CAP)

    rh_eff = jnp.where(temp <= _T_KOHLER_LO, rh190_clip, rh_clip)

    return rdry * (DTYPE(1.0) + rh_eff * kappa / (DTYPE(1.0) - rh_eff)) ** (DTYPE(1.0) / DTYPE(3.0))


def _wetr_wtpct(rdry, rhopdry, temp, h2o_mass, h2o_vp, gwtmol_h2so4):
    """Binary H2SO4/H2O wet radius, Kelvin-corrected.

    Mirrors Fortran's ``I_WTPCT_H2SO4`` path. First iteration uses a
    fixed 80 wt% guess for Kelvin; then re-evaluates Tabazadeh wt%
    using the Kelvin-corrected water activity and the corresponding
    density. One Kelvin iteration is what the Fortran does too —
    enough for typical stratospheric conditions.
    """
    wtpkelv0 = DTYPE(80.0)
    den1 = DTYPE(2.00151) - DTYPE(0.000974043) * temp  # density at 79 wt%
    den2 = DTYPE(2.01703) - DTYPE(0.000988264) * temp  # density at 80 wt%
    drho_dwt = den2 - den1
    sig1 = DTYPE(79.3556) - DTYPE(0.0267212) * temp    # σ at 79.432 wt%
    sig2 = DTYPE(75.608) - DTYPE(0.0269204) * temp     # σ at 85.9195 wt%
    dsigma_dwt = (sig2 - sig1) / (DTYPE(85.9195) - DTYPE(79.432))
    sigkelv = sig1 + dsigma_dwt * (DTYPE(80.0) - DTYPE(79.432))

    rwet0 = rdry * (DTYPE(100.0) * rhopdry / wtpkelv0 / den2) ** (DTYPE(1.0) / DTYPE(3.0))
    rkelv_b = (DTYPE(1.0)
               + wtpkelv0 * drho_dwt / den2
               - DTYPE(3.0) * wtpkelv0 * dsigma_dwt / (DTYPE(2.0) * sigkelv))
    rkelv_a = (DTYPE(2.0) * gwtmol_h2so4 * sigkelv
               / (den1 * RGAS * temp * rwet0))
    rkelvinH2O = jnp.exp(rkelv_a * rkelv_b)

    h2o_kelv = h2o_mass / rkelvinH2O
    wtpkelv = wtpct_tabaz(temp, h2o_kelv, h2o_vp)
    rhopwet = sulfate_density(wtpkelv, temp)
    rwet = rdry * (DTYPE(100.0) * rhopdry / wtpkelv / rhopwet) ** (DTYPE(1.0) / DTYPE(3.0))
    return rwet, rhopwet


def get_wetr(rdry, rhopdry, rh, temp, irhswell,
             kappa=None, h2o_mass=None, h2o_vp=None,
             gwtmol_h2so4=DTYPE(98.0)):
    """Wet radius and wet density for a single particle size.

    Args:
        rdry: Dry radius [cm]. Scalar or array.
        rhopdry: Dry particle density [g/cm^3]. Same shape as rdry.
        rh: Relative humidity over pure water, [0, 1]. Same shape.
        temp: Temperature [K]. Same shape.
        irhswell: Swelling method (``SwellMethod`` enum value, *static*).
        kappa: Per-particle hygroscopicity parameter κ (Petters-Kreidenweis).
            Required for ``I_PETTERS``.
        h2o_mass: Water-vapour mass concentration [g/cm^3].
            Required for ``I_WTPCT_H2SO4``.
        h2o_vp: Water saturation vapour pressure over liquid [dyn/cm^2].
            Required for ``I_WTPCT_H2SO4``.
        gwtmol_h2so4: H2SO4 molecular weight. Default 98 g/mol.

    Returns:
        Tuple ``(rwet, rhopwet)``.
    """
    rdry = jnp.asarray(rdry, dtype=DTYPE)
    rhopdry = jnp.asarray(rhopdry, dtype=DTYPE)
    rh = jnp.asarray(rh, dtype=DTYPE)
    temp = jnp.asarray(temp, dtype=DTYPE)

    irhswell = int(irhswell)
    if irhswell == int(SwellMethod.I_NO_SWELLING):
        return rdry, rhopdry

    if irhswell == int(SwellMethod.I_PETTERS):
        if kappa is None:
            raise ValueError("I_PETTERS requires `kappa`")
        kappa = jnp.asarray(kappa, dtype=DTYPE)
        rwet = _wetr_petters(rdry, rh, temp, kappa)
        r_ratio = (rdry / rwet) ** DTYPE(3.0)
        rhopwet = r_ratio * rhopdry + (DTYPE(1.0) - r_ratio) * RHO_W
        return rwet, rhopwet

    if irhswell == int(SwellMethod.I_WTPCT_H2SO4):
        if h2o_mass is None or h2o_vp is None:
            raise ValueError("I_WTPCT_H2SO4 requires `h2o_mass` and `h2o_vp`")
        h2o_mass = jnp.asarray(h2o_mass, dtype=DTYPE)
        h2o_vp = jnp.asarray(h2o_vp, dtype=DTYPE)
        return _wetr_wtpct(rdry, rhopdry, temp, h2o_mass, h2o_vp,
                           jnp.asarray(gwtmol_h2so4, dtype=DTYPE))

    if irhswell in (int(SwellMethod.I_FITZGERALD), int(SwellMethod.I_GERBER)):
        raise NotImplementedError(
            f"irhswell={irhswell} (Fitzgerald/Gerber) is not yet ported — "
            "scheduled for a later phase alongside seasalt physics.")

    raise ValueError(f"unknown irhswell={irhswell}")
