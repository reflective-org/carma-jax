"""Binary homogeneous nucleation rate for H2SO4-H2O.

Implements the Vehkamaki et al. (2002) parameterization.
Ported from: sulfnucrate.F90 (binary_nuc_vehk2002 subroutine)

Reference: Vehkamaki, H., M. Kulmala, I. Napari, K.E.J. Lehtinen,
    C. Timmreck, M. Noppel and A. Laaksonen, 2002,
    An improved parameterization for sulfuric acid-water nucleation
    rates for tropospheric and stratospheric conditions,
    J. Geophys. Res., 107, 4622, doi:10.1029/2002jd002184
"""

import jax.numpy as jnp

from carma.constants import AVG, RGAS, PI
from carma.precision import DTYPE


def binary_nuc_vehk2002(temp, rh, h2so4, gwtmol_h2so4):
    """Compute binary H2SO4-H2O nucleation rate (Vehkamaki 2002).

    Valid for: T: 230.15-305.15 K, RH: 0.01-100%, Na: 1e4-1e11 cm^-3.
    Inputs are bounded to validity range internally.

    Args:
        temp: Temperature [K], scalar or array.
        rh: Relative humidity (0-1), scalar or array.
        h2so4: H2SO4 concentration [molecules/cm^3], scalar or array.
        gwtmol_h2so4: H2SO4 molecular weight [g/mol].

    Returns:
        Tuple of (nucrate_cgs, mass_cluster_dry, radius_cluster) where:
            nucrate_cgs: Nucleation rate [#/cm^3/s].
            mass_cluster_dry: Dry cluster mass [g].
            radius_cluster: Critical cluster radius [cm].
    """
    # Bound inputs to validity range
    T = jnp.clip(temp, DTYPE(230.0), DTYPE(305.0))
    RH = jnp.clip(rh, DTYPE(1e-4), DTYPE(1.0))
    Na = jnp.clip(h2so4, DTYPE(1e4), DTYPE(1e11))

    ln_RH = jnp.log(RH)
    ln_Na = jnp.log(Na)

    # --- Critical cluster mole fraction x* (Eq. 11) ---
    crit_x = (DTYPE(0.740997) - DTYPE(0.00266379) * T
              - DTYPE(0.00349998) * ln_Na
              + DTYPE(0.0000504022) * T * ln_Na
              + DTYPE(0.00201048) * ln_RH
              - DTYPE(0.000183289) * T * ln_RH
              + DTYPE(0.00157407) * ln_RH**2
              - DTYPE(0.0000179059) * T * ln_RH**2
              + DTYPE(0.000184403) * ln_RH**3
              - DTYPE(1.50345e-6) * T * ln_RH**3)

    # --- Nucleation rate coefficients a-j (Eq. 12, page 3-5) ---
    acoe = (DTYPE(0.14309) + DTYPE(2.21956) * T
            - DTYPE(0.0273911) * T**2 + DTYPE(0.0000722811) * T**3
            + DTYPE(5.91822) / crit_x)

    bcoe = (DTYPE(0.117489) + DTYPE(0.462532) * T
            - DTYPE(0.0118059) * T**2 + DTYPE(0.0000404196) * T**3
            + DTYPE(15.7963) / crit_x)

    ccoe = (DTYPE(-0.215554) - DTYPE(0.0810269) * T
            + DTYPE(0.00143581) * T**2 - DTYPE(4.7758e-6) * T**3
            - DTYPE(2.91297) / crit_x)

    dcoe = (DTYPE(-3.58856) + DTYPE(0.049508) * T
            - DTYPE(0.00021382) * T**2 + DTYPE(3.10801e-7) * T**3
            - DTYPE(0.0293333) / crit_x)

    ecoe = (DTYPE(1.14598) - DTYPE(0.600796) * T
            + DTYPE(0.00864245) * T**2 - DTYPE(0.0000228947) * T**3
            - DTYPE(8.44985) / crit_x)

    fcoe = (DTYPE(2.15855) + DTYPE(0.0808121) * T
            - DTYPE(0.000407382) * T**2 - DTYPE(4.01957e-7) * T**3
            + DTYPE(0.721326) / crit_x)

    gcoe = (DTYPE(1.6241) - DTYPE(0.0160106) * T
            + DTYPE(0.0000377124) * T**2 + DTYPE(3.21794e-8) * T**3
            - DTYPE(0.0113255) / crit_x)

    hcoe = (DTYPE(9.71682) - DTYPE(0.115048) * T
            + DTYPE(0.000157098) * T**2 + DTYPE(4.00914e-7) * T**3
            + DTYPE(0.71186) / crit_x)

    icoe = (DTYPE(-1.05611) + DTYPE(0.00903378) * T
            - DTYPE(0.0000198417) * T**2 + DTYPE(2.46048e-8) * T**3
            - DTYPE(0.0579087) / crit_x)

    jcoe = (DTYPE(-0.148712) + DTYPE(0.00283508) * T
            - DTYPE(9.24619e-6) * T**2 + DTYPE(5.00427e-9) * T**3
            - DTYPE(0.0127081) / crit_x)

    # Nucleation rate J [#/cm^3/s] (Eq. 12)
    ln_J = (acoe
            + bcoe * ln_RH
            + ccoe * ln_RH**2
            + dcoe * ln_RH**3
            + ecoe * ln_Na
            + fcoe * ln_RH * ln_Na
            + gcoe * ln_RH**2 * ln_Na
            + hcoe * ln_Na**2
            + icoe * ln_RH * ln_Na**2
            + jcoe * ln_Na**3)

    ln_J = jnp.minimum(ln_J, jnp.log(DTYPE(1e38)))
    nucrate_cgs = jnp.exp(ln_J)

    # --- Total molecules in critical cluster (Eq. 13) ---
    # Coefficients for n_tot (second set in Fortran, different from J)
    acoe2 = (DTYPE(-0.00295413) - DTYPE(0.0976834) * T
             + DTYPE(0.00102485) * T**2 - DTYPE(2.18646e-6) * T**3
             - DTYPE(0.101717) / crit_x)

    bcoe2 = (DTYPE(-0.00205064) - DTYPE(0.00758504) * T
             + DTYPE(0.000192654) * T**2 - DTYPE(6.7043e-7) * T**3
             - DTYPE(0.255774) / crit_x)

    ccoe2 = (DTYPE(0.00322308) + DTYPE(0.000852637) * T
             - DTYPE(0.0000154757) * T**2 + DTYPE(5.66661e-8) * T**3
             + DTYPE(0.0338444) / crit_x)

    dcoe2 = (DTYPE(0.0474323) - DTYPE(0.000625104) * T
             + DTYPE(2.65066e-6) * T**2 - DTYPE(3.67471e-9) * T**3
             - DTYPE(0.000267251) / crit_x)

    ecoe2 = (DTYPE(-0.0125211) + DTYPE(0.00580655) * T
             - DTYPE(0.000101674) * T**2 + DTYPE(2.88195e-7) * T**3
             + DTYPE(0.0942243) / crit_x)

    fcoe2 = (DTYPE(-0.038546) - DTYPE(0.000672316) * T
             + DTYPE(2.60288e-6) * T**2 + DTYPE(1.19416e-8) * T**3
             - DTYPE(0.00851515) / crit_x)

    gcoe2 = (DTYPE(-0.0183749) + DTYPE(0.000172072) * T
             - DTYPE(3.71766e-7) * T**2 - DTYPE(5.14875e-10) * T**3
             + DTYPE(0.00026866) / crit_x)

    hcoe2 = (DTYPE(-0.0619974) + DTYPE(0.000906958) * T
             - DTYPE(9.11728e-7) * T**2 - DTYPE(5.36796e-9) * T**3
             - DTYPE(0.00774234) / crit_x)

    icoe2 = (DTYPE(0.0121827) - DTYPE(0.00010665) * T
             + DTYPE(2.5346e-7) * T**2 - DTYPE(3.63519e-10) * T**3
             + DTYPE(0.000610065) / crit_x)

    jcoe2 = (DTYPE(0.000320184) - DTYPE(0.0000174762) * T
             + DTYPE(6.06504e-8) * T**2 - DTYPE(1.4177e-11) * T**3
             + DTYPE(0.000135751) / crit_x)

    cnum_tot = jnp.exp(
        acoe2
        + bcoe2 * ln_RH
        + ccoe2 * ln_RH**2
        + dcoe2 * ln_RH**3
        + ecoe2 * ln_Na
        + fcoe2 * ln_RH * ln_Na
        + gcoe2 * ln_RH**2 * ln_Na
        + hcoe2 * ln_Na**2
        + icoe2 * ln_RH * ln_Na**2
        + jcoe2 * ln_Na**3
    )

    cnum_h2so4 = cnum_tot * crit_x

    # --- Critical cluster radius (Eq. 14) ---
    radius_cluster = jnp.exp(
        DTYPE(-1.6524245) + DTYPE(0.42316402) * crit_x
        + DTYPE(0.3346648) * jnp.log(cnum_tot)
    )
    radius_cluster = radius_cluster * DTYPE(1e-7)  # nm -> cm

    # Dry cluster mass
    mass_cluster_dry = cnum_h2so4 * DTYPE(gwtmol_h2so4) / AVG  # g

    # Zero out nucleation if below threshold
    nucrate_cgs = jnp.where(h2so4 >= DTYPE(1e4), nucrate_cgs, DTYPE(0.0))

    return nucrate_cgs, mass_cluster_dry, radius_cluster


def sulfnucrate(t_val, rh, h2so4, gwtmol_h2so4, gwtmol_h2o,
                rmassup, rmrat_val, zmet_val, nbin):
    """Compute nucleation rate and target bin.

    Wrapper around binary_nuc_vehk2002 that determines which bin
    the critical cluster falls in and scales to internal units.

    Args:
        t_val: Temperature [K], scalar.
        rh: Relative humidity (0-1), scalar.
        h2so4: H2SO4 concentration [molecules/cm^3], scalar.
        gwtmol_h2so4: H2SO4 molecular weight [g/mol].
        gwtmol_h2o: H2O molecular weight [g/mol].
        rmassup: Upper bin boundary masses [g], shape (NBIN,).
        rmrat_val: Mass ratio between bins.
        zmet_val: Vertical metric, scalar.
        nbin: Number of bins.

    Returns:
        Tuple of (nucrate, nucbin, radius_cluster) where:
            nucrate: Nucleation rate [#/z/s] (internal units).
            nucbin: Target bin index (0-based).
            radius_cluster: Critical cluster radius [cm].
    """
    nucrate_cgs, mass_cluster_dry, radius_cluster = binary_nuc_vehk2002(
        t_val, rh, h2so4, gwtmol_h2so4
    )

    # Determine target bin
    nucbin = jnp.where(
        mass_cluster_dry < rmassup[0],
        0,
        jnp.clip(
            1 + jnp.floor(jnp.log(mass_cluster_dry / rmassup[0]) / jnp.log(DTYPE(rmrat_val))).astype(jnp.int32),
            0, nbin - 1,
        ),
    )

    # Scale to internal units [#/z/s]
    nucrate = nucrate_cgs * DTYPE(zmet_val)

    # Zero if cluster is larger than all bins
    nucrate = jnp.where(nucbin < nbin, nucrate, DTYPE(0.0))

    return nucrate, nucbin, radius_cluster
