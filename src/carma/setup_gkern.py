"""Growth kernel setup for CARMA-JAX.

Computes radius-dependent growth parameters at bin boundaries: surface
tensions, Kelvin factors, free paths, Knudsen corrections, ventilation
factors, and growth kernel coefficients.
Ported from: setupgkern.F90
"""

import jax.numpy as jnp

from carma.constants import (
    AVG, BK, CP, PI, RGAS, RHO_W, RHO_I, WTMOL_AIR, T0,
)
from carma.precision import DTYPE


def setup_gkern(t, p, rhoa, zmet, rmu, thcond, diffus, rlhe, rlhm,
                re, r_wet, rlow_wet, rrat, eshape_arr, is_ice_arr,
                gwtmol_arr, igrowgas_arr, gstickl, gsticki, tstick,
                nbin, ngroup, ngas,
                igash2o=-1, igash2so4=-1, wtpct=None):
    """Compute growth kernel coefficients.

    Args:
        t: Temperature [K], shape (NZ,).
        p: Pressure [dyne/cm^2], shape (NZ,).
        rhoa: Air density * zmet [g/cm^2/z], shape (NZ,).
        zmet: Vertical metric, shape (NZ,).
        rmu: Dynamic viscosity [g/cm/s], shape (NZ,).
        thcond: Thermal conductivity [erg/cm/s/K], shape (NZ,).
        diffus: Gas diffusivity [cm^2/s], shape (NZ, NGAS).
        rlhe: Latent heat evaporation [cm^2/s^2], shape (NZ, NGAS).
        rlhm: Latent heat melting [cm^2/s^2], shape (NZ, NGAS).
        re: Reynolds number, shape (NZ, NBIN, NGROUP).
        r_wet: Wet radius [cm], shape (NZ, NBIN, NGROUP).
        rlow_wet: Wet radius at lower bin boundary [cm], shape (NZ, NBIN, NGROUP).
        rrat: Radius ratio for shape, shape (NBIN, NGROUP).
        eshape_arr: Aspect ratio per group, shape (NGROUP,).
        is_ice_arr: Ice flag per group, shape (NGROUP,).
        gwtmol_arr: Gas molecular weight per gas, shape (NGAS,).
        igrowgas_arr: Gas index per element, shape (NELEM,).
        gstickl: Liquid sticking coefficient.
        gsticki: Ice sticking coefficient.
        tstick: Thermal accommodation coefficient.
        nbin, ngroup, ngas: Dimensions.

    Returns:
        Tuple of (surfctwa, akelvin, akelvini, gro, gro1, gro2, ft_arr, thcondnc)
    """
    nz = t.shape[0]
    rhoa_cgs = rhoa / zmet  # (NZ,)

    # --- Surface tensions (for H2O) ---
    surfctwa = DTYPE(76.10) - DTYPE(0.155) * (t - T0)  # water-air [dyne/cm]
    surfctiw = DTYPE(28.5) + DTYPE(0.25) * (t - T0)    # ice-water [dyne/cm]
    surfctia = DTYPE(141.0) - DTYPE(0.15) * t           # ice-air [dyne/cm]

    # --- Kelvin curvature factors ---
    akelvin = jnp.zeros((nz, ngas), dtype=DTYPE)
    akelvini = jnp.zeros((nz, ngas), dtype=DTYPE)

    for igas in range(ngas):
        gwtmol = DTYPE(gwtmol_arr[igas])
        akelvin = akelvin.at[:, igas].set(
            DTYPE(2.0) * gwtmol * surfctwa / (t * RHO_W * RGAS)
        )
        akelvini = akelvini.at[:, igas].set(
            DTYPE(2.0) * gwtmol * surfctia / (t * RHO_W * RGAS)
        )

    # H2SO4 Kelvin overrides — Fortran setupgkern.F90:115-126.
    # akelvin uses sulfate-specific surface tension and density (depend
    # on wtpct = sulfate weight percent in the binary H2SO4/H2O solution);
    # akelvini falls back to the H2O ice-Kelvin value because Fortran
    # does not condense H2SO4 onto ice.
    if igash2so4 >= 0 and wtpct is not None:
        from carma.sulfate_utils import sulfate_density, sulfate_surf_tens
        wtpct_arr = jnp.asarray(wtpct, dtype=DTYPE)
        gwtmol_h2so4 = DTYPE(gwtmol_arr[igash2so4])
        surf_tens_h2so4 = sulfate_surf_tens(wtpct_arr, t)
        rho_h2so4 = sulfate_density(wtpct_arr, t)
        akelvin = akelvin.at[:, igash2so4].set(
            DTYPE(2.0) * gwtmol_h2so4 * surf_tens_h2so4
            / (t * rho_h2so4 * RGAS)
        )
        if igash2o >= 0:
            akelvini = akelvini.at[:, igash2so4].set(akelvini[:, igash2o])

    # --- Free paths ---
    freep = jnp.zeros((nz, ngas), dtype=DTYPE)
    freept = jnp.zeros((nz, ngas), dtype=DTYPE)

    for igas in range(ngas):
        gwtmol = DTYPE(gwtmol_arr[igas])
        freep = freep.at[:, igas].set(
            DTYPE(3.0) * diffus[:, igas]
            * jnp.sqrt(PI * gwtmol / (DTYPE(8.0) * RGAS * t))
        )
        freept = freept.at[:, igas].set(
            freep[:, igas] * thcond
            / (diffus[:, igas] * rhoa_cgs * (CP - RGAS / (DTYPE(2.0) * WTMOL_AIR)))
        )

    # --- Growth kernel arrays ---
    gro = jnp.zeros((nz, nbin, ngroup), dtype=DTYPE)
    gro1 = jnp.zeros((nz, nbin, ngroup), dtype=DTYPE)
    gro2 = jnp.zeros((nz, ngroup), dtype=DTYPE)
    ft_arr = jnp.zeros((nz, nbin, ngroup), dtype=DTYPE)
    thcondnc = jnp.zeros((nz, nbin, ngroup), dtype=DTYPE)

    for ig in range(ngroup):
        is_ice = bool(is_ice_arr[ig])
        gstick = gsticki if is_ice else gstickl

        # Shape corrections (spheres only for now)
        cor = DTYPE(1.0)
        phish = DTYPE(1.0)

        # Find the condensing gas for this group
        # Use first element's igrowgas
        igas = -1
        for ie in range(len(igrowgas_arr)):
            if int(igrowgas_arr[ie]) >= 0:
                igas = int(igrowgas_arr[ie])
                break
        if igas < 0:
            continue

        gwtmol = DTYPE(gwtmol_arr[igas])

        for k in range(nz):
            # Latent heat
            if is_ice:
                rlh = rlhe[k, igas] + rlhm[k, igas]
            else:
                rlh = rlhe[k, igas]

            for i in range(nbin):
                # Bin boundary radius (use lower boundary of wet radius)
                br = rlow_wet[k, i, ig]
                if br <= 0:
                    br = r_wet[k, i, ig] * DTYPE(0.5)

                # Knudsen numbers
                rknudn = freep[k, igas] / br
                rknudnt = freept[k, igas] / br

                # Lambda corrections
                rlam = (
                    (DTYPE(1.33) * rknudn + DTYPE(0.71)) / (rknudn + DTYPE(1.0))
                    + DTYPE(4.0) * (DTYPE(1.0) - DTYPE(gstick)) / (DTYPE(3.0) * DTYPE(gstick))
                )
                rlamt = (
                    (DTYPE(1.33) * rknudnt + DTYPE(0.71)) / (rknudnt + DTYPE(1.0))
                    + DTYPE(4.0) * (DTYPE(1.0) - DTYPE(tstick)) / (DTYPE(3.0) * DTYPE(tstick))
                )

                # Corrected diffusivity and thermal conductivity
                diffus1 = diffus[k, igas] * cor / (DTYPE(1.0) + rlam * rknudn * cor / phish)
                thcond1 = thcond[k] * cor / (DTYPE(1.0) + rlamt * rknudnt * cor / phish)

                thcondnc = thcondnc.at[k, i, ig].set(thcond1)

                # Schmidt and Prandtl numbers
                schn = rmu[k] / (rhoa_cgs[k] * diffus1)
                prnum = rmu[k] * CP / thcond1

                # Reynolds number with shape correction
                reyn_shape = re[k, i, ig]

                # Ventilation factors
                x1 = schn ** (DTYPE(1.0) / DTYPE(3.0)) * jnp.sqrt(reyn_shape)
                x2 = prnum ** (DTYPE(1.0) / DTYPE(3.0)) * jnp.sqrt(reyn_shape)

                if is_ice:
                    fv = jnp.where(x1 <= DTYPE(1.0),
                                   DTYPE(1.0) + DTYPE(0.14) * x1**2,
                                   DTYPE(0.86) + DTYPE(0.28) * x1)
                    ft_val = jnp.where(x2 <= DTYPE(1.0),
                                       DTYPE(1.0) + DTYPE(0.14) * x2**2,
                                       DTYPE(0.86) + DTYPE(0.28) * x2)
                else:
                    fv = jnp.where(x1 <= DTYPE(1.4),
                                   DTYPE(1.0) + DTYPE(0.108) * x1**2,
                                   DTYPE(0.78) + DTYPE(0.308) * x1)
                    ft_val = jnp.where(x2 <= DTYPE(1.4),
                                       DTYPE(1.0) + DTYPE(0.108) * x2**2,
                                       DTYPE(0.78) + DTYPE(0.308) * x2)

                ft_arr = ft_arr.at[k, i, ig].set(ft_val)

                # Growth kernel coefficients
                gro_val = (
                    DTYPE(4.0) * PI * br * diffus1 * fv * gwtmol
                    / (BK * t[k] * AVG)
                )
                gro = gro.at[k, i, ig].set(gro_val)

                gro1_val = (
                    gwtmol * rlh**2
                    / (RGAS * t[k]**2 * ft_val * thcond1)
                    / (DTYPE(4.0) * PI * br)
                )
                gro1 = gro1.at[k, i, ig].set(gro1_val)

                # gro2 computed once per group (at first bin)
                if i == 0:
                    gro2 = gro2.at[k, ig].set(DTYPE(1.0) / rlh)

    return surfctwa, akelvin, akelvini, gro, gro1, gro2, ft_arr, thcondnc
