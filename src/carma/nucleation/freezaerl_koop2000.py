"""Homogeneous aerosol freezing via Koop et al. (2000).

Computes nucleation loss rates for aerosol particles freezing into ice
crystals, parameterized by water activity.
Ported from: freezaerl_koop2000.F90

Reference: Koop, T., B. Luo, A. Tsias, and T. Peter, 2000,
    Water activity as the determinant for homogeneous ice nucleation
    in aqueous solutions, Nature, 406, 611-614.
"""

import jax.numpy as jnp

from carma.constants import RGAS, RHO_W, RHO_I, FEW_PC
from carma.precision import DTYPE, POWMAX


# Physical constants for Koop parameterization
_PRENUC = DTYPE(2.075e33) * RHO_W / RHO_I
_KT0 = DTYPE(1.6)
_DKT0DP = DTYPE(-8.8)
_KTI = DTYPE(0.22)
_DKTIDP = DTYPE(-0.17)
_RSI = RGAS / DTYPE(1e7)  # Gas constant in J/(mol·K)
_SIFREEZE = DTYPE(0.3)     # Minimum ice supersaturation for freezing


def freezaerl_koop2000(t_val, p_val, supsati_val, supsatl_val,
                        akelvin_val, r_bins, vol_bins, rhosol_val,
                        pconmax_val, nbin):
    """Compute Koop (2000) aerosol freezing nucleation rates.

    Args:
        t_val: Temperature [K], scalar.
        p_val: Pressure [dyne/cm^2], scalar.
        supsati_val: Ice supersaturation, scalar.
        supsatl_val: Liquid supersaturation, scalar.
        akelvin_val: Kelvin factor [cm], scalar.
        r_bins: Bin center radii [cm], shape (NBIN,).
        vol_bins: Bin volumes [cm^3], shape (NBIN,).
        rhosol_val: Solute density [g/cm^3].
        pconmax_val: Max concentration for this group.
        nbin: Number of bins.

    Returns:
        rnuclg: Nucleation loss rates [s^-1], shape (NBIN,).
    """
    rnuclg = jnp.zeros(nbin, dtype=DTYPE)

    # Only for T < 240K
    active_T = t_val <= DTYPE(240.0)
    # Only if significant particles present
    active_pc = pconmax_val > FEW_PC
    # Only if ice supersaturated above threshold
    active_ssi = supsati_val > _SIFREEZE

    active = active_T & active_pc & active_ssi

    # Koop parameterization
    td = t_val
    rlnt = jnp.log(td)

    # Eq. 2: chemical potential difference [J/mol]
    dmy = (DTYPE(210368.0) + DTYPE(131.438) * td
           - DTYPE(3.32373e6) / td - DTYPE(41729.1) * rlnt)

    # Water activity of ice at this temperature
    awi = jnp.exp(dmy / (_RSI * td))

    # Eq. 4: molar volume of water [cm^3/mol]
    vw0 = DTYPE(-230.76) - DTYPE(0.1478) * td + DTYPE(4099.2) / td + DTYPE(48.8341) * rlnt

    # Eq. 5: molar volume of ice [cm^3/mol]
    vi = DTYPE(19.43) - DTYPE(2.2e-3) * td + DTYPE(1.08e-5) * td * td

    # Pressure terms [GPa]
    pp = DTYPE(1e-10) * p_val  # dyne/cm^2 to GPa
    pp2 = pp * pp * DTYPE(0.5)
    pp3 = pp2 * pp / DTYPE(3.0)

    # Eq. 3: pressure-volume correction
    riv = (vw0 * (pp - _KT0 * pp2 - _DKT0DP * pp3)
           - vi * (pp - _KTI * pp2 - _DKTIDP * pp3))
    riv = riv * DTYPE(1e3)  # GPa·cm^3/mol to Pa·m^3/mol

    # Water activity (clamp liquid supersaturation)
    ssl_clamped = jnp.clip(supsatl_val, DTYPE(-1.0), DTYPE(0.0))
    aw_base = DTYPE(1.0) + ssl_clamped

    # Per-bin computation
    def compute_bin(ibin):
        # Kelvin effect on water activity
        fkelv = jnp.exp(akelvin_val / r_bins[ibin])
        aw = aw_base / fkelv

        # Eq. 6: delta water activity
        daw = aw * jnp.exp(riv / (_RSI * td)) - awi
        # Eq. 7 validity range: 0.26 < daw < 0.34
        daw = jnp.clip(daw, DTYPE(0.26), DTYPE(0.34))

        # Eq. 7: log10(J) = polynomial in daw
        rlogj = ((DTYPE(29180.0) * daw - DTYPE(26924.0)) * daw + DTYPE(8502.0)) * daw - DTYPE(906.7)
        rlogj = jnp.minimum(rlogj, POWMAX * DTYPE(0.3))
        rjj = DTYPE(10.0) ** rlogj  # J in cm^-3 s^-1

        # H2SO4 molality from water activity (piecewise fit)
        CONTL = jnp.where(
            aw < DTYPE(0.05),
            DTYPE(12.37208932) * aw**DTYPE(-0.16125516114) - DTYPE(30.490657554) * aw - DTYPE(2.1133114241),
            jnp.where(
                aw <= DTYPE(0.85),
                DTYPE(11.820654354) * aw**DTYPE(-0.20786404244) - DTYPE(4.807306373) * aw - DTYPE(5.1727540348),
                DTYPE(-180.06541028) * aw**DTYPE(-0.38601102592) - DTYPE(93.317846778) * aw + DTYPE(273.88132245),
            ))
        CONTH = jnp.where(
            aw < DTYPE(0.05),
            DTYPE(13.455394705) * aw**DTYPE(-0.1921312255) - DTYPE(34.285174604) * aw - DTYPE(1.7620073078),
            jnp.where(
                aw <= DTYPE(0.85),
                DTYPE(12.891938068) * aw**DTYPE(-0.23233847708) - DTYPE(6.4261237757) * aw - DTYPE(4.9005471319),
                DTYPE(-176.95814097) * aw**DTYPE(-0.36257048154) - DTYPE(90.469744201) * aw + DTYPE(267.45509988),
            ))

        # Interpolate between 190K and 260K
        H2SO4m = CONTL + (CONTH - CONTL) * (t_val - DTYPE(190.0)) / DTYPE(70.0)
        WT = DTYPE(98.0) * H2SO4m / (DTYPE(1000.0) + DTYPE(98.0) * H2SO4m)
        WT = jnp.clip(WT, DTYPE(0.0), DTYPE(1.0)) * DTYPE(100.0)

        # Volume ratio wet/dry
        volrat = jnp.where(
            WT <= DTYPE(0.0),
            DTYPE(1e10),
            rhosol_val / RHO_W * ((DTYPE(100.0) - WT) / WT) + DTYPE(1.0),
        )

        # Nucleation rate [s^-1] = J * V_wet
        rate = rjj * volrat * vol_bins[ibin]
        return jnp.where(active, rate, DTYPE(0.0))

    # Vectorized over bins
    rnuclg = jnp.array([compute_bin(i) for i in range(nbin)])

    return rnuclg
