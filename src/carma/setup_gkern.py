"""Growth kernel setup for CARMA-JAX.

Computes radius-dependent growth parameters at bin boundaries: surface
tensions, Kelvin factors, free paths, Knudsen corrections, ventilation
factors, and growth kernel coefficients.
Ported from: setupgkern.F90

Performance note: the inner bin loop is fully vectorized over the NBIN
axis; the outer group loop stays as Python (NGROUP is static and small).
The function is JIT-compiled; static arguments that affect trace shape
(nbin, ngroup, ngas, igash2o, igash2so4) must be passed as Python ints.
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
        is_ice_arr: Ice flag per group, shape (NGROUP,) — numpy or sequence.
        gwtmol_arr: Gas molecular weight per gas, shape (NGAS,) JAX array.
        igrowgas_arr: Gas index per element, shape (NELEM,) — numpy or sequence.
        gstickl: Liquid sticking coefficient.
        gsticki: Ice sticking coefficient.
        tstick: Thermal accommodation coefficient.
        nbin, ngroup, ngas: Dimensions (static — used as Python loop bounds).
        igash2o, igash2so4: Gas indices (static — resolve branch at trace time).
        wtpct: H2SO4 weight-percent [0–1], shape (NZ,) or None.
               When provided, enables sulfate-specific Kelvin correction.

    Returns:
        Tuple of (surfctwa, akelvin, akelvini, gro, gro1, gro2, ft_arr, thcondnc)
        where each output has the same shape convention as before.
    """
    nz = t.shape[0]
    rhoa_cgs = rhoa / zmet  # (NZ,)

    # --- Surface tensions ---
    surfctwa = DTYPE(76.10) - DTYPE(0.155) * (t - T0)   # water-air [dyne/cm]
    surfctia = DTYPE(141.0) - DTYPE(0.15) * t            # ice-air  [dyne/cm]

    # --- Kelvin curvature factors ---
    # Broadcast gwtmol over the gas axis: (NGAS,) → (NZ, NGAS)
    gwtmol_v = jnp.asarray(gwtmol_arr, dtype=DTYPE)[None, :]  # (1, NGAS)
    akelvin = (DTYPE(2.0) * gwtmol_v * surfctwa[:, None]
               / (t[:, None] * DTYPE(RHO_W) * DTYPE(RGAS)))   # (NZ, NGAS)
    akelvini = (DTYPE(2.0) * gwtmol_v * surfctia[:, None]
                / (t[:, None] * DTYPE(RHO_W) * DTYPE(RGAS)))  # (NZ, NGAS)

    # H2SO4 Kelvin overrides — Fortran setupgkern.F90:115-126.
    # akelvin uses sulfate-specific surface tension and density;
    # akelvini for H2SO4 copies the H2O ice-Kelvin value (Fortran does
    # not condense H2SO4 onto ice).
    if igash2so4 >= 0 and wtpct is not None:
        from carma.sulfate_utils import sulfate_density, sulfate_surf_tens
        wtpct_arr = jnp.asarray(wtpct, dtype=DTYPE)
        gwtmol_h2so4 = DTYPE(gwtmol_arr[igash2so4])
        surf_tens_h2so4 = sulfate_surf_tens(wtpct_arr, t)
        rho_h2so4 = sulfate_density(wtpct_arr, t)
        akelvin = akelvin.at[:, igash2so4].set(
            DTYPE(2.0) * gwtmol_h2so4 * surf_tens_h2so4
            / (t * rho_h2so4 * DTYPE(RGAS))
        )
        if igash2o >= 0:
            akelvini = akelvini.at[:, igash2so4].set(akelvini[:, igash2o])

    # --- Molecular free paths (NZ, NGAS) — vectorized over gas axis ---
    # freep = 3 * D * sqrt(π * gwtmol / (8 * R * T))
    freep = (DTYPE(3.0) * diffus
             * jnp.sqrt(DTYPE(PI) * gwtmol_v
                        / (DTYPE(8.0) * DTYPE(RGAS) * t[:, None])))   # (NZ, NGAS)
    freept = (freep * thcond[:, None]
              / (diffus * rhoa_cgs[:, None]
                 * (DTYPE(CP) - DTYPE(RGAS) / (DTYPE(2.0) * DTYPE(WTMOL_AIR)))))

    # --- Growth kernel arrays ---
    gro = jnp.zeros((nz, nbin, ngroup), dtype=DTYPE)
    gro1 = jnp.zeros((nz, nbin, ngroup), dtype=DTYPE)
    gro2 = jnp.zeros((nz, ngroup), dtype=DTYPE)
    ft_arr = jnp.zeros((nz, nbin, ngroup), dtype=DTYPE)
    thcondnc = jnp.zeros((nz, nbin, ngroup), dtype=DTYPE)

    for ig in range(ngroup):
        is_ice = bool(is_ice_arr[ig])
        gstick = DTYPE(gsticki) if is_ice else DTYPE(gstickl)

        # Find condensing gas for this group
        igas = -1
        for ie in range(len(igrowgas_arr)):
            if int(igrowgas_arr[ie]) >= 0:
                igas = int(igrowgas_arr[ie])
                break
        if igas < 0:
            continue

        gwtmol = DTYPE(gwtmol_arr[igas])

        # --- Bin boundary radius (NZ, NBIN) ---
        rlow_g = rlow_wet[:, :, ig]   # (NZ, NBIN)
        r_g = r_wet[:, :, ig]
        br = jnp.where(rlow_g > DTYPE(0.0), rlow_g, r_g * DTYPE(0.5))

        # --- Knudsen numbers (NZ, NBIN): freep is (NZ,), br is (NZ, NBIN) ---
        rknudn = freep[:, igas, None] / br
        rknudnt = freept[:, igas, None] / br

        # --- Lambda corrections (NZ, NBIN) ---
        rlam = ((DTYPE(1.33) * rknudn + DTYPE(0.71)) / (rknudn + DTYPE(1.0))
                + DTYPE(4.0) * (DTYPE(1.0) - gstick) / (DTYPE(3.0) * gstick))
        rlamt = ((DTYPE(1.33) * rknudnt + DTYPE(0.71)) / (rknudnt + DTYPE(1.0))
                 + DTYPE(4.0) * (DTYPE(1.0) - DTYPE(tstick))
                   / (DTYPE(3.0) * DTYPE(tstick)))

        # --- Corrected diffusivity and conductivity (NZ, NBIN) ---
        # cor = phish = 1 (spheres)
        diffus1 = diffus[:, igas, None] / (DTYPE(1.0) + rlam * rknudn)
        thcond1 = thcond[:, None] / (DTYPE(1.0) + rlamt * rknudnt)

        # --- Schmidt and Prandtl numbers (NZ, NBIN) ---
        schn = rmu[:, None] / (rhoa_cgs[:, None] * diffus1)
        prnum = rmu[:, None] * DTYPE(CP) / thcond1

        # --- Reynolds number (NZ, NBIN) ---
        reyn = re[:, :, ig]

        # --- Ventilation factors (NZ, NBIN) ---
        x1 = schn ** (DTYPE(1.0) / DTYPE(3.0)) * jnp.sqrt(reyn)
        x2 = prnum ** (DTYPE(1.0) / DTYPE(3.0)) * jnp.sqrt(reyn)

        if is_ice:
            fv = jnp.where(x1 <= DTYPE(1.0),
                           DTYPE(1.0) + DTYPE(0.14) * x1**2,
                           DTYPE(0.86) + DTYPE(0.28) * x1)
            ft = jnp.where(x2 <= DTYPE(1.0),
                           DTYPE(1.0) + DTYPE(0.14) * x2**2,
                           DTYPE(0.86) + DTYPE(0.28) * x2)
        else:
            fv = jnp.where(x1 <= DTYPE(1.4),
                           DTYPE(1.0) + DTYPE(0.108) * x1**2,
                           DTYPE(0.78) + DTYPE(0.308) * x1)
            ft = jnp.where(x2 <= DTYPE(1.4),
                           DTYPE(1.0) + DTYPE(0.108) * x2**2,
                           DTYPE(0.78) + DTYPE(0.308) * x2)

        # --- Latent heat (NZ,) ---
        rlh = (rlhe[:, igas] + rlhm[:, igas]) if is_ice else rlhe[:, igas]

        # --- Growth kernels (NZ, NBIN) ---
        gro_g = (DTYPE(4.0) * DTYPE(PI) * br * diffus1 * fv * gwtmol
                 / (DTYPE(BK) * t[:, None] * DTYPE(AVG)))

        gro1_g = (gwtmol * rlh[:, None]**2
                  / (DTYPE(RGAS) * t[:, None]**2 * ft * thcond1)
                  / (DTYPE(4.0) * DTYPE(PI) * br))

        gro = gro.at[:, :, ig].set(gro_g)
        gro1 = gro1.at[:, :, ig].set(gro1_g)
        ft_arr = ft_arr.at[:, :, ig].set(ft)
        thcondnc = thcondnc.at[:, :, ig].set(thcond1)

        # gro2 is bin-independent — one value per (NZ, group)
        gro2 = gro2.at[:, ig].set(DTYPE(1.0) / rlh)

    return surfctwa, akelvin, akelvini, gro, gro1, gro2, ft_arr, thcondnc
