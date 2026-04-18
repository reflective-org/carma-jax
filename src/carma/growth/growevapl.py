"""Growth and evaporation loss rates via PPM for CARMA-JAX.

Implements the Colella-Woodward (1984) Piecewise Parabolic Method for
computing growth and evaporation loss rates in mass space.
Ported from: growevapl.F90
"""

import jax.numpy as jnp

from carma.constants import FEW_PC
from carma.precision import DTYPE
from carma.growth.pheat import pheat


def growevapl(pc, growlg, evaplg,
              supsatl, supsati, pvapl, pvapi,
              akelvin, akelvini, gro, gro1,
              rup_wet, rmass, dm, pconmax,
              pratt, prat, pden1, palr,
              is_ice_arr, igrowgas_arr, ienconc_arr,
              dtime, iz, nbin, ngroup):
    """Compute growth and evaporation loss rates using PPM.

    Args:
        pc: Particle concentrations (NZ, NBIN, NELEM).
        growlg: Growth loss rates (NBIN, NGROUP) to update.
        evaplg: Evaporation loss rates (NBIN, NGROUP) to update.
        supsatl, supsati: Supersaturation (NZ, NGAS).
        pvapl, pvapi: Saturation vapor pressure (NZ, NGAS).
        akelvin, akelvini: Kelvin factors (NZ, NGAS).
        gro: Growth kernel (NZ, NBIN, NGROUP).
        gro1: Growth conduction (NZ, NBIN, NGROUP).
        rup_wet: Wet radius at upper boundary (NZ, NBIN, NGROUP) [cm].
        rmass: Particle mass (NBIN, NGROUP) [g].
        dm: Bin mass width (NBIN, NGROUP) [g].
        pconmax: Max concentration per group (NZ, NGROUP).
        pratt: PPM gradient coefficients (3, NBIN, NGROUP).
        prat: PPM boundary coefficients (4, NBIN, NGROUP).
        pden1: PPM denominator (NBIN, NGROUP).
        palr: PPM edge slope factors (4, NGROUP).
        is_ice_arr: Ice flag per group (NGROUP,).
        igrowgas_arr: Growth gas index per element (NELEM,).
        ienconc_arr: Number concentration element per group (NGROUP,).
        dtime: Timestep [s].
        iz: Vertical level index.
        nbin, ngroup: Dimensions.

    Returns:
        Tuple of (growlg, evaplg) updated arrays.
    """
    for ig in range(ngroup):
        iepart = int(ienconc_arr[ig])
        igas = int(igrowgas_arr[iepart])
        if igas < 0:
            continue

        is_ice = bool(is_ice_arr[ig])
        has_particles = pconmax[iz, ig] > FEW_PC
        # Don't short-circuit on has_particles here — the per-bin
        # bin_has_particles mask in the flux computation below already
        # zeros out contributions from negligible bins. Short-circuiting
        # via Python `if` would break JIT tracing since has_particles is
        # a traced boolean.

        # Compute dmdt for all bin boundaries (bin 0 to NBIN-2)
        dmdt = jnp.zeros(nbin, dtype=DTYPE)
        for ibin in range(nbin - 1):
            dmdt_val = pheat(
                pc, supsatl, supsati, pvapl, pvapi,
                akelvin, akelvini, gro, gro1,
                rup_wet, rmass, is_ice, iz, ig, ibin, igas,
            )
            dmdt = dmdt.at[ibin].set(dmdt_val)

        # PPM polynomial construction
        # Normalized concentrations: dpc = pc / dm
        dpc = jnp.zeros(nbin, dtype=DTYPE)
        for ibin in range(nbin):
            dpc = dpc.at[ibin].set(
                pc[iz, ibin, iepart] / dm[ibin, ig]
            )

        # Step 1: Gradient estimation (bins 1 to NBIN-2, 0-based)
        dela = jnp.zeros(nbin, dtype=DTYPE)
        delma = jnp.zeros(nbin, dtype=DTYPE)

        for ibin in range(1, nbin - 1):
            ratt1 = pratt[0, ibin, ig]
            ratt2 = pratt[1, ibin, ig]
            ratt3 = pratt[2, ibin, ig]

            d = dpc[ibin]
            d1 = dpc[ibin + 1]
            dm1 = dpc[ibin - 1]

            dela_val = ratt1 * (ratt2 * (d1 - d) + ratt3 * (d - dm1))
            dela = dela.at[ibin].set(dela_val)

            # Monotonicity limiter
            cond = (d1 - d) * (d - dm1) > DTYPE(0.0)
            limited = jnp.minimum(
                jnp.abs(dela_val),
                jnp.minimum(
                    DTYPE(2.0) * jnp.abs(d - d1),
                    DTYPE(2.0) * jnp.abs(d - dm1),
                ),
            ) * jnp.sign(dela_val)
            delma = delma.at[ibin].set(jnp.where(cond, limited, DTYPE(0.0)))

        # Step 2: PPM boundary values (bins 1 to NBIN-3, 0-based)
        aju = jnp.zeros(nbin, dtype=DTYPE)
        for ibin in range(1, nbin - 2):
            rat1 = prat[0, ibin, ig]
            rat2 = prat[1, ibin, ig]
            rat3 = prat[2, ibin, ig]
            rat4 = prat[3, ibin, ig]
            den1 = pden1[ibin, ig]

            aju_val = (
                dpc[ibin]
                + rat1 * (dpc[ibin + 1] - dpc[ibin])
                + DTYPE(1.0) / den1 * (
                    rat2 * (rat3 - rat4) * (dpc[ibin + 1] - dpc[ibin])
                    - dm[ibin, ig] * rat3 * delma[ibin + 1]
                    + dm[ibin + 1, ig] * rat4 * delma[ibin]
                )
            )
            aju = aju.at[ibin].set(aju_val)

        # Step 3: Set polynomial endpoints (al, ar)
        al = jnp.zeros(nbin, dtype=DTYPE)
        ar = jnp.zeros(nbin, dtype=DTYPE)

        # Interior bins (2 to NBIN-3, 0-based)
        for ibin in range(2, nbin - 2):
            al = al.at[ibin].set(aju[ibin - 1])
            ar = ar.at[ibin].set(aju[ibin])

        # Edge bins
        # Bin 1 (0-based index 1)
        ar = ar.at[1].set(aju[1])
        al = al.at[1].set(
            dpc[0] + palr[0, ig] * (dpc[1] - dpc[0])
        )

        # Bin 0
        ar = ar.at[0].set(al[1])
        al = al.at[0].set(
            dpc[0] + palr[1, ig] * (dpc[1] - dpc[0])
        )

        # Bin NBIN-2 (0-based)
        al = al.at[nbin - 2].set(aju[nbin - 3])
        ar = ar.at[nbin - 2].set(
            dpc[nbin - 2] + palr[2, ig] * (dpc[nbin - 1] - dpc[nbin - 2])
        )

        # Bin NBIN-1 (0-based)
        al = al.at[nbin - 1].set(ar[nbin - 2])
        ar = ar.at[nbin - 1].set(
            dpc[nbin - 2] + palr[3, ig] * (dpc[nbin - 1] - dpc[nbin - 2])
        )

        # Step 4: Monotonicity enforcement
        for ibin in range(nbin):
            d = dpc[ibin]
            al_v = al[ibin]
            ar_v = ar[ibin]

            # If d is outside [al, ar], clamp to constant
            outside = (ar_v - d) * (d - al_v) <= DTYPE(0.0)
            al_v = jnp.where(outside, d, al_v)
            ar_v = jnp.where(outside, d, ar_v)

            # Test for overshooting
            diff = ar_v - al_v
            test1 = diff * (d - DTYPE(0.5) * (al_v + ar_v))
            test2 = diff**2 / DTYPE(6.0)

            al_v = jnp.where(test1 > test2, DTYPE(3.0) * d - DTYPE(2.0) * ar_v, al_v)
            ar_v = jnp.where(test1 < -test2, DTYPE(3.0) * d - DTYPE(2.0) * al_v, ar_v)

            al = al.at[ibin].set(al_v)
            ar = ar.at[ibin].set(ar_v)

        # Step 5: Compute dela and a6 for flux
        for ibin in range(nbin):
            d = dpc[ibin]
            dela = dela.at[ibin].set(ar[ibin] - al[ibin])

        a6 = jnp.zeros(nbin, dtype=DTYPE)
        for ibin in range(nbin):
            a6 = a6.at[ibin].set(
                DTYPE(6.0) * (dpc[ibin] - DTYPE(0.5) * (ar[ibin] + al[ibin]))
            )

        # Per-bin threshold: max concentration in this group for relative check.
        # The PPM divides dmdt by pc, producing enormous growth rates when
        # pc is tiny. Use a relative threshold to zero out growth in bins
        # that are negligible compared to the peak.
        pc_group_max = jnp.max(pc[iz, :, iepart])
        bin_threshold = jnp.maximum(FEW_PC, pc_group_max * DTYPE(1e-20))

        # Step 6: Flux computation
        for ibin in range(nbin - 1):
            dmdt_val = dmdt[ibin]
            pc_val = pc[iz, ibin, iepart]

            bin_has_particles = pc_val > bin_threshold

            # Growth (dmdt > 0): flux from bin ibin to ibin+1
            x_grow = dmdt_val * dtime / dm[ibin, ig]
            grow_ppm = jnp.where(
                x_grow < DTYPE(1.0),
                dmdt_val / jnp.maximum(pc_val, DTYPE(1e-50)) * (
                    ar[ibin] - DTYPE(0.5) * dela[ibin] * x_grow
                    + (x_grow / DTYPE(2.0) - x_grow**2 / DTYPE(3.0)) * a6[ibin]
                ),
                dmdt_val / dm[ibin, ig],
            )
            grow_ppm = jnp.where(bin_has_particles, grow_ppm, DTYPE(0.0))
            growlg = growlg.at[ibin, ig].set(
                jnp.where(dmdt_val > DTYPE(0.0), grow_ppm, growlg[ibin, ig])
            )

            # Evaporation (dmdt < 0): flux from bin ibin+1 to ibin
            pc_val_next = pc[iz, ibin + 1, iepart]
            next_bin_has_particles = pc_val_next > bin_threshold
            x_evap = -dmdt_val * dtime / dm[ibin + 1, ig]
            evap_ppm = jnp.where(
                x_evap < DTYPE(1.0),
                -dmdt_val / jnp.maximum(pc_val_next, DTYPE(1e-50)) * (
                    al[ibin + 1] + DTYPE(0.5) * dela[ibin + 1] * x_evap
                    + (x_evap / DTYPE(2.0) - x_evap**2 / DTYPE(3.0)) * a6[ibin + 1]
                ),
                -dmdt_val / dm[ibin + 1, ig],
            )
            evap_ppm = jnp.where(next_bin_has_particles, evap_ppm, DTYPE(0.0))
            evaplg = evaplg.at[ibin + 1, ig].set(
                jnp.where(dmdt_val < DTYPE(0.0), evap_ppm, evaplg[ibin + 1, ig])
            )

            # Special boundary: evaporation at bin 0
            evaplg = evaplg.at[0, ig].set(
                jnp.where(
                    (ibin == 0) & (dmdt_val < DTYPE(0.0)),
                    -dmdt_val / dm[0, ig],
                    evaplg[0, ig],
                )
            )

    return growlg, evaplg
