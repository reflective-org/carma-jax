"""Heterogeneous nucleation of glassy aerosols (Murray et al. 2010).

Computes pseudo-nucleation rates for glassy aerosol particles that
nucleate ice heterogeneously. Based on fraction of aerosol nucleated
as function of ice supersaturation.
Ported from: freezglaerl_murray2010.F90

Reference: Murray, B.J., et al., 2010, Heterogeneous nucleation of ice
    particles on glassy aerosols under cirrus conditions, Nature Geosci.,
    3, 233-237.
"""

import jax.numpy as jnp

from carma.constants import FEW_PC
from carma.precision import DTYPE

# Murray 2010 fit constants
_KICE1 = DTYPE(7.7211e-5)   # Fraction fit slope
_KICE2 = DTYPE(9.2688e-3)   # Fraction fit intercept
_SSMIN = DTYPE(0.21)         # Minimum ice supersaturation for nucleation
_SSMAX = DTYPE(0.7)          # Maximum ice supersaturation
_TGLASS = DTYPE(212.0)       # Maximum temperature for glassy state [K]
_FGLASS = DTYPE(0.5)         # Fraction of aerosols that can be glassy


def freezglaerl_murray2010(t_val, supsati_val, supsati_old_val,
                            pconmax_val, dtime, nbin):
    """Compute Murray (2010) glassy aerosol heterogeneous freezing rates.

    The rate is computed as the difference in nucleated fraction between
    current and previous supersaturation, divided by the timestep.

    fice = kice1 * RHi(%) - kice2   for 121% < RHi < 170%
    rnuclg = fglass * (fice(ssi) - fice(ssi_old)) / dtime

    Args:
        t_val: Temperature [K], scalar.
        supsati_val: Current ice supersaturation, scalar.
        supsati_old_val: Previous ice supersaturation, scalar.
        pconmax_val: Max concentration for source group.
        dtime: Timestep [s].
        nbin: Number of bins.

    Returns:
        rnuclg_add: Additional nucleation loss rates [s^-1], shape (NBIN,).
    """
    rnuclg_add = jnp.zeros(nbin, dtype=DTYPE)

    # Conditions for Murray nucleation
    active = (
        (t_val <= _TGLASS)
        & (pconmax_val > FEW_PC)
        & (supsati_val >= _SSMIN)
        & (supsati_val > supsati_old_val)
    )

    # Fraction nucleated at current supersaturation
    ssi_capped = jnp.minimum(_SSMAX, supsati_val)
    dfice = _KICE1 * (DTYPE(1.0) + ssi_capped) * DTYPE(100.0) - _KICE2

    # Subtract fraction nucleated at previous supersaturation
    ssi_old_capped = jnp.minimum(_SSMAX, supsati_old_val)
    dfice_old = jnp.where(
        supsati_old_val >= _SSMIN,
        _KICE1 * (DTYPE(1.0) + ssi_old_capped) * DTYPE(100.0) - _KICE2,
        DTYPE(0.0),
    )
    dfice = dfice - dfice_old

    # Rate = fglass * dfice / dtime (uniform across all bins)
    rate = jnp.where(active, _FGLASS * dfice / dtime, DTYPE(0.0))

    # Same rate for all bins
    rnuclg_add = jnp.full(nbin, rate, dtype=DTYPE)

    return rnuclg_add
