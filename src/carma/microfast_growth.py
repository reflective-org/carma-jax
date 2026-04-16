"""JIT-compiled growth step for CARMA-JAX.

Vectorized: pheat over all bins, PPM as array ops, growp/evapp as shifts,
psolve via jax.lax.scan. Full microfast_growth is @jax.jit.
"""

import jax
import jax.numpy as jnp

from carma.constants import SMALL_PC, FEW_PC, CP, RGAS
from carma.enums import RC_OK, RC_WARNING_RETRY
from carma.precision import DTYPE, POWMAX


def make_microfast_growth(nbin, nelem, ngroup, ngas, gwtmol_val, is_ice):
    """Factory: creates a JIT-compiled growth step function.

    Bakes static config into closure. Returns pure function of dynamic arrays.
    """

    @jax.jit
    def microfast_growth_jit(pc, gc, t, iz_dummy,
                              rhoa, zmet, rlhe, rlhm,
                              akelvin, akelvini, gro, gro1,
                              rup_wet, rmass_2d, dm_2d,
                              pratt, prat, pden1, palr,
                              dtime):
        """One growth step: vapor pressure → supersaturation → PPM → psolve → evapp → gsolve → tsolve.

        All arrays for a single level (NZ=1). iz=0 always.
        """
        iz = 0

        # --- Vapor pressure (Murphy 2005) ---
        tt = t[iz]
        pvapi = DTYPE(10.0) * jnp.exp(
            DTYPE(9.550426) - DTYPE(5723.265) / tt
            + DTYPE(3.53068) * jnp.log(tt) - DTYPE(0.00728332) * tt
        )
        pvapl = DTYPE(10.0) * jnp.exp(
            DTYPE(54.842763) - DTYPE(6763.22) / tt - DTYPE(4.210) * jnp.log(tt)
            + DTYPE(0.000367) * tt
            + jnp.tanh(DTYPE(0.0415) * (tt - DTYPE(218.8)))
            * (DTYPE(53.878) - DTYPE(1331.22) / tt
               - DTYPE(9.44523) * jnp.log(tt) + DTYPE(0.014025) * tt)
        )

        # --- Supersaturation ---
        rvap = RGAS / DTYPE(gwtmol_val)
        gc_cgs = gc[iz, 0] / zmet[iz]
        pvap_use = jnp.where(is_ice, pvapi, pvapl)
        supsatl = (gc_cgs * rvap * tt - pvapl) / pvapl
        supsati = (gc_cgs * rvap * tt - pvapi) / pvapi
        ss = jnp.where(is_ice, supsati, supsatl)

        # --- Total condensate before growth ---
        prev_condensate = jnp.sum(pc[iz, :, 0] * rmass_2d[:, 0])

        # --- Vectorized pheat: dmdt for all bin boundaries ---
        # gro/gro1 at boundary ibin→ibin+1 is stored at index ibin+1
        g0 = gro[iz, 1:, 0]   # (NBIN-1,) boundaries 0→1, 1→2, ..., NBIN-2→NBIN-1
        g1 = gro1[iz, 1:, 0]  # (NBIN-1,)

        # Kelvin factor at rup (upper boundary of each bin)
        akelv = jnp.where(is_ice, akelvini[iz, 0], akelvin[iz, 0])
        r_bound = rup_wet[iz, :nbin - 1, 0]  # (NBIN-1,)
        expon = jnp.clip(akelv / jnp.maximum(r_bound, DTYPE(1e-30)), -POWMAX, POWMAX)
        akas = jnp.exp(expon)

        dmdt_all = pvap_use * (ss + DTYPE(1.0) - akas) * g0 / (DTYPE(1.0) + g0 * g1 * pvap_use)
        # (NBIN-1,): growth rate at each bin boundary

        # --- Vectorized PPM ---
        dm_1d = dm_2d[:, 0]   # (NBIN,)
        pc_1d = pc[iz, :, 0]  # (NBIN,)

        # Normalized concentration
        dpc = pc_1d / dm_1d  # (NBIN,)

        # Gradient estimation (vectorized over interior bins 1..NBIN-2)
        dela = jnp.zeros(nbin, dtype=DTYPE)
        delma = jnp.zeros(nbin, dtype=DTYPE)

        d_center = dpc[1:-1]
        d_right = dpc[2:]
        d_left = dpc[:-2]

        ratt1 = pratt[0, 1:-1, 0]
        ratt2 = pratt[1, 1:-1, 0]
        ratt3 = pratt[2, 1:-1, 0]

        dela_interior = ratt1 * (ratt2 * (d_right - d_center) + ratt3 * (d_center - d_left))
        dela = dela.at[1:-1].set(dela_interior)

        # Monotonicity limiter
        cond = (d_right - d_center) * (d_center - d_left) > DTYPE(0.0)
        limited = jnp.minimum(
            jnp.abs(dela_interior),
            jnp.minimum(DTYPE(2.0) * jnp.abs(d_center - d_right),
                        DTYPE(2.0) * jnp.abs(d_center - d_left))
        ) * jnp.sign(dela_interior)
        delma = delma.at[1:-1].set(jnp.where(cond, limited, DTYPE(0.0)))

        # PPM boundary values (vectorized over bins 1..NBIN-3)
        aju = jnp.zeros(nbin, dtype=DTYPE)
        idx = jnp.arange(1, nbin - 2)
        rat1 = prat[0, 1:-2, 0]
        rat2 = prat[1, 1:-2, 0]
        rat3 = prat[2, 1:-2, 0]
        rat4 = prat[3, 1:-2, 0]
        den1 = pden1[1:-2, 0]

        aju_interior = (
            dpc[1:-2] + rat1 * (dpc[2:-1] - dpc[1:-2])
            + DTYPE(1.0) / den1 * (
                rat2 * (rat3 - rat4) * (dpc[2:-1] - dpc[1:-2])
                - dm_1d[1:-2] * rat3 * delma[2:-1]
                + dm_1d[2:-1] * rat4 * delma[1:-2]
            )
        )
        aju = aju.at[1:-2].set(aju_interior)

        # Polynomial endpoints
        al = jnp.zeros(nbin, dtype=DTYPE)
        ar = jnp.zeros(nbin, dtype=DTYPE)

        # Interior bins: al[i] = aju[i-1], ar[i] = aju[i]
        al = al.at[2:-2].set(aju[1:-3])
        ar = ar.at[2:-2].set(aju[2:-2])

        # Edge bins
        ar = ar.at[1].set(aju[1])
        al = al.at[1].set(dpc[0] + palr[0, 0] * (dpc[1] - dpc[0]))
        ar = ar.at[0].set(al[1])
        al = al.at[0].set(dpc[0] + palr[1, 0] * (dpc[1] - dpc[0]))
        al = al.at[-2].set(aju[-3])
        ar = ar.at[-2].set(dpc[-2] + palr[2, 0] * (dpc[-1] - dpc[-2]))
        al = al.at[-1].set(ar[-2])
        ar = ar.at[-1].set(dpc[-2] + palr[3, 0] * (dpc[-1] - dpc[-2]))

        # Monotonicity enforcement (vectorized)
        outside = (ar - dpc) * (dpc - al) <= DTYPE(0.0)
        al = jnp.where(outside, dpc, al)
        ar = jnp.where(outside, dpc, ar)

        diff = ar - al
        test1 = diff * (dpc - DTYPE(0.5) * (al + ar))
        test2 = diff ** 2 / DTYPE(6.0)
        al = jnp.where(test1 > test2, DTYPE(3.0) * dpc - DTYPE(2.0) * ar, al)
        ar = jnp.where(test1 < -test2, DTYPE(3.0) * dpc - DTYPE(2.0) * al, ar)

        # PPM flux parameters
        dela_flux = ar - al
        a6 = DTYPE(6.0) * (dpc - DTYPE(0.5) * (ar + al))

        # --- Compute growlg and evaplg (vectorized) ---
        pconmax_val = jnp.max(pc_1d / zmet[iz])
        has_particles = pconmax_val > FEW_PC

        # Growth (dmdt > 0)
        x_grow = dmdt_all * dtime / dm_1d[:nbin - 1]
        grow_ppm = (dmdt_all / jnp.maximum(pc_1d[:nbin - 1], DTYPE(1e-50))
                    * (ar[:nbin - 1] - DTYPE(0.5) * dela_flux[:nbin - 1] * x_grow
                       + (x_grow / DTYPE(2.0) - x_grow ** 2 / DTYPE(3.0)) * a6[:nbin - 1]))
        grow_upwind = dmdt_all / dm_1d[:nbin - 1]
        growlg_vals = jnp.where(x_grow < DTYPE(1.0), grow_ppm, grow_upwind)
        growlg_vals = jnp.where(dmdt_all > DTYPE(0.0), growlg_vals, DTYPE(0.0))
        growlg_vals = jnp.where(has_particles, growlg_vals, DTYPE(0.0))
        growlg = jnp.concatenate([growlg_vals, jnp.zeros(1, dtype=DTYPE)])  # (NBIN,)

        # Evaporation (dmdt < 0)
        x_evap = -dmdt_all * dtime / dm_1d[1:nbin]
        evap_ppm = (-dmdt_all / jnp.maximum(pc_1d[1:nbin], DTYPE(1e-50))
                    * (al[1:nbin] + DTYPE(0.5) * dela_flux[1:nbin] * x_evap
                       + (x_evap / DTYPE(2.0) - x_evap ** 2 / DTYPE(3.0)) * a6[1:nbin]))
        evap_upwind = -dmdt_all / dm_1d[1:nbin]
        evaplg_vals = jnp.where(x_evap < DTYPE(1.0), evap_ppm, evap_upwind)
        evaplg_vals = jnp.where(dmdt_all < DTYPE(0.0), evaplg_vals, DTYPE(0.0))
        evaplg_vals = jnp.where(has_particles, evaplg_vals, DTYPE(0.0))
        # Bin 0 evaporation: always upwind
        evaplg_0 = jnp.where(
            (dmdt_all[0] < DTYPE(0.0)) & has_particles,
            -dmdt_all[0] / dm_1d[0], DTYPE(0.0)
        )
        evaplg = jnp.concatenate([evaplg_0[None], evaplg_vals])  # (NBIN,)

        # --- Growth production + psolve via scan ---
        # Sequential: for each bin, growpe[i] = pc[i-1]*growlg[i-1], then psolve
        def psolve_step(pc_1d, ibin):
            # Growth production from bin ibin-1
            growpe_val = jnp.where(
                ibin > 0,
                pc_1d[ibin - 1] * growlg[ibin - 1],
                DTYPE(0.0),
            )
            # psolve: pc_new = (pc_old + dt*ppd) / (1 + pls*dt)
            ppd = growpe_val
            pls = growlg[ibin] + evaplg[ibin]
            pc_new = (pc_1d[ibin] + dtime * ppd) / (DTYPE(1.0) + pls * dtime)
            pc_new = jnp.maximum(pc_new, SMALL_PC)
            pc_1d = pc_1d.at[ibin].set(pc_new)
            return pc_1d, None

        pc_1d, _ = jax.lax.scan(psolve_step, pc_1d, jnp.arange(nbin))

        # --- Evaporation production (explicit) ---
        # evappe[i-1] += pc[i] * evaplg[i] for i=1..NBIN-1
        evappe = pc_1d[1:] * evaplg[1:]  # (NBIN-1,)
        pc_1d = pc_1d.at[:nbin - 1].add(dtime * evappe)
        pc_1d = jnp.maximum(pc_1d, SMALL_PC)

        # Write back to pc
        pc = pc.at[iz, :, 0].set(pc_1d)

        # --- Gas solver ---
        curr_condensate = jnp.sum(pc_1d * rmass_2d[:, 0])
        gasprod = (prev_condensate - curr_condensate) / dtime

        # Latent heat production
        rlh = jnp.where(is_ice, rlhe[iz, 0] + rlhm[iz, 0], rlhe[iz, 0])
        rlprod = -(prev_condensate - curr_condensate) * rlh / (CP * rhoa[iz] * dtime)

        gc = gc.at[iz, 0].add(dtime * gasprod)

        # --- Temperature solver ---
        dt_val = dtime * rlprod
        t = t.at[iz].add(dt_val)

        rlheat_val = rlprod * dtime

        return pc, gc, t, rlheat_val

    return microfast_growth_jit
