"""Effective ice-particle density per bin (Heymsfield et al. 2010).

Ported from `rhoice_heymsfield2010.F90`. For an ice group, each bin
gets:

    m(D) = a · Dᵇ           (Heymsfield & Schmitt 2010, cgs)
    rbin = ½ · (m / a)^(1/b)
    rho(ibin) = m / (4π/3 · rbin³)        (clipped to rhoice)
    aratelem  = exp(-38 · dmax)           if dmax ≤ 200 µm
              = 0.16 · dmax^(-0.27)       otherwise

where `m` for bin i is `rmassmin · rmrat^(i-1)` (CARMA's geometric
bin grid) and `dmax = 2·rbin`. `b = 2.1`; `a` depends on the chosen
ice regime (Heymsfield et al. 2010 Table 1).

This is a setup-time routine — called once per ice group during
`CARMAELEMENT_Create` to build the bin-resolved density table. The
regime is a static configuration choice (string), so we resolve it
at Python level and inject the appropriate `a` coefficient.
"""
from typing import Tuple

import jax.numpy as jnp

from carma.constants import PI
from carma.precision import DTYPE


# Regime → `a` coefficient (cgs). From rhoice_heymsfield2010.F90:46-57.
_A_BY_REGIME = {
    "deep": DTYPE(1.10e-2),
    "conv": DTYPE(6.33e-3),
    "cold": DTYPE(5.74e-3),
    "avg":  DTYPE(5.28e-3),
    "synp": DTYPE(4.22e-3),
    "warm": DTYPE(3.79e-3),
}
_B = DTYPE(2.1)                       # m(D) = a · D^b exponent
_DMAX_THRESHOLD = DTYPE(200.0e-4)     # 200 µm in cm (aratelem branch)
_ARATELEM_FAR_SCALE = DTYPE(0.16)
_ARATELEM_FAR_EXP = DTYPE(-0.27)
_ARATELEM_NEAR_K = DTYPE(-38.0)
_FOUR_THIRDS_PI = DTYPE(4.0) / DTYPE(3.0) * PI


def rhoice_heymsfield2010(
    rhoice: float,
    rmassmin: float,
    rmrat: float,
    regime: str,
    nbin: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Per-bin ice density and projected-area ratio.

    Args:
        rhoice:    Bulk ice density [g/cm³] — upper clip for `rho(ibin)`.
        rmassmin:  Minimum bin mass for this group [g] (CARMA bin grid).
        rmrat:     Bin-to-bin mass ratio for this group (dimensionless).
        regime:    Ice regime — one of ``{"deep","conv","cold","avg","synp","warm"}``.
                   Selects the `a` coefficient in `m = a·Dᵇ`.
        nbin:      Number of bins.

    Returns:
        Tuple ``(rho, aratelem)`` each of shape ``(NBIN,)``:
            rho:       Effective per-bin density [g/cm³], clipped to ≤ rhoice.
            aratelem:  Projected area ratio (Schmitt & Heymsfield 2009).

    Raises:
        ValueError: if `regime` is not one of the six known names.
    """
    try:
        a = _A_BY_REGIME[regime]
    except KeyError:
        raise ValueError(
            f"rhoice_heymsfield2010: unknown ice regime {regime!r}; "
            f"expected one of {sorted(_A_BY_REGIME)}")

    ibin = jnp.arange(nbin, dtype=DTYPE)
    totalmass = DTYPE(rmassmin) * (DTYPE(rmrat) ** ibin)

    rbin = (totalmass / a) ** (DTYPE(1.0) / _B) / DTYPE(2.0)
    rho_raw = totalmass / (_FOUR_THIRDS_PI * rbin ** 3)
    rho = jnp.minimum(rho_raw, DTYPE(rhoice))

    dmax = DTYPE(2.0) * rbin
    aratelem = jnp.where(
        dmax <= _DMAX_THRESHOLD,
        jnp.exp(_ARATELEM_NEAR_K * dmax),
        _ARATELEM_FAR_SCALE * dmax ** _ARATELEM_FAR_EXP,
    )

    return rho, aratelem
