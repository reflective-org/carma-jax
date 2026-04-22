"""Binary homogeneous nucleation rate for H2SO4-H2O.

Two parameterisations:

- ``binary_nuc_vehk2002`` — Vehkamaki et al. (2002). Polynomial fits,
  valid T ∈ [230, 305] K, RH ∈ [1e-4, 1], H2SO4 ∈ [1e4, 1e11] cm⁻³.

- ``binary_nuc_zhao1995`` — Zhao & Turco (1995). Classical nucleation
  with Giauque (1960) Gibbs potential table, Ayers + Kulmala vapour
  pressure for H2SO4, Lin-Tabazadeh (2001) for H2O. Used by
  ``sulfhetnucrate`` for heterogeneous nucleation seeding. Also
  provides the critical radius ``rstar`` and exponent ``ftry`` that
  Fletcher-factor heterogeneous rates need.

Ported from: sulfnucrate.F90.
"""

import jax.numpy as jnp

from carma.constants import AVG, BK, RGAS, PI
from carma.precision import DTYPE
from carma.sulfate_utils import (
    _DNC0, _DNC1, _DNWTP, sulfate_density, sulfate_surf_tens,
)


# Gibbs potential table from Giauque 1960 / Mills 1994 fit.
# Computed once at import; matches Fortran data statement.
_DNPOT = DTYPE(4.184) * (
    DTYPE(23624.8)
    - DTYPE(1.14208e8) / ((_DNWTP - DTYPE(105.318)) ** 2 + DTYPE(4798.69))
)
_DNWF = _DNWTP / DTYPE(100.0)    # weight fraction table
_NTAB = 46                        # table length (match Fortran)


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


# ---------------------------------------------------------------------------
# Zhao & Turco (1995) — classical binary H2SO4/H2O nucleation
# ---------------------------------------------------------------------------

def binary_nuc_zhao1995(temp, weight_percent, rh, h2so4, h2so4_cgs, h2o,
                        h2o_cgs, beta1, gwtmol_h2so4, gwtmol_h2o):
    """Binary H2SO4/H2O nucleation rate — Zhao & Turco (1995).

    Classical-nucleation-theory formulation with:
      - Giauque (1960) partial-molal Gibbs free-energy table
      - Ayers (1980) H2SO4 vapour pressure + Kulmala temperature correction
      - Lin & Tabazadeh (2001) water-over-solution vapour pressure
      - Jaecker-Voirol & Mirabel (1988) Zeldovitch non-equilibrium factor

    Args:
        temp: Temperature [K]. Scalar.
        weight_percent: H2SO4 weight-percent of the bulk aerosol [0-100].
            Used only for the critical-cluster density.
        rh: Relative humidity over liquid water (0-1). Currently unused
            internally (kept on the signature to mirror Fortran).
        h2so4: H2SO4 gas concentration [molecules/cm^3].
        h2so4_cgs: H2SO4 gas mass concentration [g/cm^3].
        h2o: H2O gas concentration [molecules/cm^3].
        h2o_cgs: H2O gas mass concentration [g/cm^3].
        beta1: Gas-kinetic collision rate coefficient [cm/s].
        gwtmol_h2so4: H2SO4 molecular weight [g/mol].
        gwtmol_h2o: H2O molecular weight [g/mol].

    Returns:
        (nucrate_cgs, mass_cluster_dry, radius_cluster, ftry) where:
            nucrate_cgs: Binary homogeneous rate [# cm^-3 s^-1]
            mass_cluster_dry: Dry mass of the critical cluster [g]
            radius_cluster: Critical-cluster radius [cm] (0 if none)
            ftry: Gstar / (k·T) — used by heterogeneous nucleation

    All-zero output (``nucrate=0, rstar=0, ftry=0``) is produced when
    the saddle-point search fails (no crossing of the Gibbs-free-energy
    derivative) or the critical surface-tension integrand is too small.

    Matches Fortran ``binary_nuc_zhao1995`` in ``sulfnucrate.F90``.
    """
    temp = jnp.asarray(temp, dtype=DTYPE)
    weight_percent = jnp.asarray(weight_percent, dtype=DTYPE)
    h2so4 = jnp.asarray(h2so4, dtype=DTYPE)
    h2so4_cgs = jnp.asarray(h2so4_cgs, dtype=DTYPE)
    h2o = jnp.asarray(h2o, dtype=DTYPE)
    h2o_cgs = jnp.asarray(h2o_cgs, dtype=DTYPE)
    beta1 = jnp.asarray(beta1, dtype=DTYPE)
    gwtmol_h2so4 = jnp.asarray(gwtmol_h2so4, dtype=DTYPE)
    gwtmol_h2o = jnp.asarray(gwtmol_h2o, dtype=DTYPE)

    wtmolr = gwtmol_h2so4 / gwtmol_h2o

    # ln of H2O and H2SO4 ambient partial pressures [dyn/cm^2]
    h2oln = jnp.log(h2o_cgs * (RGAS / gwtmol_h2o) * temp)
    h2so4ln = jnp.log(h2so4_cgs * (RGAS / gwtmol_h2so4) * temp)

    # Density at each tabulated wt% (linear in T)
    dens_tab = _DNC0 + _DNC1 * temp                    # (46,)

    # Water vapour pressure over H2SO4/H2O (Lin & Tabazadeh 2001 eqn 5),
    # parameterised per-wt-pct.
    wtp = _DNWTP
    cw = (DTYPE(22.7490) + DTYPE(0.0424817) * wtp
          - DTYPE(0.0567432) * jnp.sqrt(wtp)
          - DTYPE(0.000621533) * wtp ** 2)
    dw = (DTYPE(-5850.24) + DTYPE(21.9744) * wtp
          - DTYPE(44.5210) * jnp.sqrt(wtp)
          - DTYPE(0.384362) * wtp ** 2)
    wvp = jnp.exp(cw + dw / temp)                       # mb
    # ln(pH2O|eq in dyn/cm^2); 1013250 dyn/cm^2 == 1013.25 mb
    wvpln = jnp.log(wvp * DTYPE(1013250.0) / DTYPE(1013.25))
    pb = h2oln - wvpln

    # H2SO4 equilibrium vapour pressure (Ayers 1980 + Kulmala correction)
    t0_kulm = DTYPE(340.0)
    t_crit_kulm = DTYPE(905.0)
    seqln_base = DTYPE(-10156.0) / t0_kulm + DTYPE(16.259)
    factor_kulm = (
        DTYPE(-1.0) / temp + DTYPE(1.0) / t0_kulm
        + DTYPE(0.38) / (t_crit_kulm - t0_kulm)
        * (DTYPE(1.0) + jnp.log(t0_kulm / temp) - t0_kulm / temp)
    )
    seqln_pure = seqln_base + DTYPE(10156.0) * factor_kulm
    # Giauque composition adjustment
    seqln = seqln_pure - _DNPOT / (DTYPE(8.3143) * temp)
    seqln = seqln + jnp.log(DTYPE(1013250.0))
    pa = h2so4ln - seqln

    # Activity products for saddle-point search
    c1 = pa - pb * wtmolr
    c2 = pa * _DNWF + pb * (DTYPE(1.0) - _DNWF) * wtmolr

    # dens1 at each i — Fortran uses different gradient formulas at the
    # top endpoint (i=46 in Fortran, 45 in 0-based) vs interior.
    # We build fct[i] for i ∈ {1..46} (Fortran) → {0..45} (0-based).
    #
    # Fortran ordering (1-based):
    #   i = 46: dens1 = (dens[46] - dens[45]) / (wtp[46] - wtp[45])
    #   i in [45, 2]: dens1 = (dens11*dw2 + dens12*dw1) / (dw1+dw2)
    # with dw1, dw2, dens11, dens12 evolving as the loop descends.
    #
    # The interior formula is equivalent to a centered gradient at i:
    #   (dens[i+1]-dens[i])/dw2 and (dens[i]-dens[i-1])/dw1, weighted by
    #   dw2 and dw1 respectively (harmonic-mean-like). Compute vectorised.

    dwtp = _DNWTP[1:] - _DNWTP[:-1]                     # (45,)
    grad = (dens_tab[1:] - dens_tab[:-1]) / dwtp         # (45,)
    # For i in [1, 44] (0-based), the Fortran dens1 is the dw-weighted
    # mean of grad[i-1] (dens11) and grad[i] (dens12):
    #   dens1[i] = (grad[i-1]*dw[i] + grad[i]*dw[i-1]) / (dw[i-1] + dw[i])
    dens1_mid = (grad[:-1] * dwtp[1:] + grad[1:] * dwtp[:-1]) / (
        dwtp[:-1] + dwtp[1:]
    )                                                    # (44,) → indices 1..44
    # For i=0 (Fortran i=1) and i=45 (Fortran i=46), use the local gradient.
    dens1_lo = grad[0]
    dens1_hi = grad[-1]
    dens1_full = jnp.concatenate([
        jnp.array([dens1_lo], dtype=DTYPE),
        dens1_mid,
        jnp.array([dens1_hi], dtype=DTYPE),
    ])                                                   # (46,)

    # fct[i] = c1[i] + c2[i] * 100 * dens1[i] / dens[i]
    fct = c1 + c2 * DTYPE(100.0) * dens1_full / dens_tab

    # Saddle-point search: largest i ∈ [0, 44] where fct[i]*fct[i+1] ≤ 0.
    # Fortran loops from i=45 down to 2 (1-based), exits on first crossing.
    # In 0-based: find largest i in [0, 44] with sign-change between fct[i]
    # and fct[i+1]. If none → no nucleation.
    sign_products = fct[:-1] * fct[1:]                   # (45,)
    has_crossing = sign_products <= DTYPE(0.0)            # (45,)
    any_cross = jnp.any(has_crossing)
    # Largest index with crossing — reverse the mask and use argmax (finds
    # first True, which corresponds to last True in original).
    rev = has_crossing[::-1]
    first_rev_idx = jnp.argmax(rev)                      # 0-based from reversed
    i_star = DTYPE(44) - first_rev_idx.astype(DTYPE)     # original 0-based index

    i_lo = jnp.clip(i_star.astype(jnp.int32), 0, _NTAB - 2)
    i_hi = i_lo + 1

    fct_lo = fct[i_lo]
    fct_hi = fct[i_hi]

    # Interpolate where sign products cross
    # xfrac = fct[i+1] / (fct[i+1] - fct[i])
    denom = fct_hi - fct_lo
    safe_denom = jnp.where(jnp.abs(denom) > DTYPE(1e-300), denom, DTYPE(1.0))
    xfrac = fct_hi / safe_denom
    # Clamp to [0, 1] — outside this the interpolation is meaningless
    xfrac = jnp.clip(xfrac, DTYPE(0.0), DTYPE(1.0))

    wstar = _DNWTP[i_hi] * (DTYPE(1.0) - xfrac) + _DNWTP[i_lo] * xfrac
    dstar = dens_tab[i_hi] * (DTYPE(1.0) - xfrac) + dens_tab[i_lo] * xfrac
    rhln = pb[i_hi] * (DTYPE(1.0) - xfrac) + pb[i_lo] * xfrac
    raln = pa[i_hi] * (DTYPE(1.0) - xfrac) + pa[i_lo] * xfrac

    # If exact crossing (fct_lo == 0), Fortran takes values at i (lo);
    # our linear interpolation produces the same numerical result when
    # xfrac snaps to 0 or 1.

    wfstar = wstar / DTYPE(100.0)

    # Critical surface tension [erg/cm^2]
    sigma = sulfate_surf_tens(wstar, temp)

    # Critical Y (Zhao-Turco 1993 eqn 13) [erg/cm^3]
    ystar = dstar * RGAS * temp * (
        wfstar / gwtmol_h2so4 * raln
        + (DTYPE(1.0) - wfstar) / gwtmol_h2o * rhln
    )

    # Critical-cluster radius: 2·σ / ystar
    ystar_ok = ystar >= DTYPE(1e-20)
    ystar_safe = jnp.where(ystar_ok, ystar, DTYPE(1.0))
    rstar_raw = DTYPE(2.0) * sigma / ystar_safe
    radius_cluster = jnp.maximum(rstar_raw, DTYPE(0.0))
    r2 = radius_cluster * radius_cluster

    # Critical Gibbs free energy [erg]
    gstar = (DTYPE(4.0) * PI / DTYPE(3.0)) * r2 * sigma

    # RPRE = 4π r^2 · H2O · β1 · H2SO4   (Zhao-Turco eqn 16 pre-factor)
    rpr = DTYPE(4.0) * PI * r2 * h2o * beta1
    rpre = rpr * h2so4

    # Zeldovitch non-equilibrium factor
    denom_frac = DTYPE(1.0) + wtmolr * (DTYPE(1.0) - wfstar) / jnp.maximum(
        wfstar, DTYPE(1e-30))
    fracmol = DTYPE(1.0) / denom_frac
    zphi = jnp.arctan(fracmol)
    # sin(zphi); guard against zphi = 0
    sinzphi = jnp.sin(zphi)
    sinzphi_safe = jnp.where(sinzphi ** 2 > DTYPE(1e-60),
                              sinzphi, DTYPE(1e-30))
    zeld = DTYPE(0.25) / sinzphi_safe ** 2

    # Exponential term in Zhao-Turco eqn 16
    ftry = -gstar / BK / temp
    ahom = ftry  # cfac=0
    exhom = jnp.where(ahom < DTYPE(-500.0),
                      DTYPE(0.0),
                      jnp.exp(jnp.minimum(ahom, DTYPE(28.0))))

    # Critical-cluster mass (dry)
    rho_h2so4_wet = sulfate_density(weight_percent, temp)
    mass_cluster_wet = (DTYPE(4.0) * PI / DTYPE(3.0)) * rho_h2so4_wet * r2 * radius_cluster
    mass_cluster_dry = mass_cluster_wet * wfstar

    # Nucleation rate
    nucrate_cgs = rpre * zeld * exhom

    # Gate on saddle existence and ystar positivity
    valid = any_cross & ystar_ok
    nucrate_cgs = jnp.where(valid, nucrate_cgs, DTYPE(0.0))
    radius_cluster = jnp.where(valid, radius_cluster, DTYPE(0.0))
    mass_cluster_dry = jnp.where(valid, mass_cluster_dry, DTYPE(0.0))
    ftry = jnp.where(valid, ftry, DTYPE(0.0))

    return nucrate_cgs, mass_cluster_dry, radius_cluster, ftry
