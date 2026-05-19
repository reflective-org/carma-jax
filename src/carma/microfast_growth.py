"""JIT-compiled growth step for CARMA-JAX.

Recomputes ALL T-dependent quantities every step (growth kernel,
diffusivity, latent heats, vapor pressure). No frozen physics.

Vectorized: pheat over all bins, PPM as array ops, growp/evapp as shifts,
psolve via jax.lax.scan. Full microfast_growth is @jax.jit.
"""

import jax
import jax.numpy as jnp

from carma.constants import (
    SMALL_PC, FEW_PC, CP, RGAS, RHO_W, AVG, BK, PI, WTMOL_AIR, T0,
)
from carma.precision import DTYPE, POWMAX


def make_microfast_growth(nbin, nelem, ngroup, ngas, gwtmol_val, is_ice,
                          gstick=1.0, tstick=1.0):
    """Factory: creates a JIT-compiled growth step function.

    Recomputes growth kernel from current T every step.
    Bakes static config into closure.
    """

    _gwtmol = DTYPE(gwtmol_val)
    _gstick = DTYPE(gstick)
    _tstick = DTYPE(tstick)

    @jax.jit
    def microfast_growth_jit(pc, gc, t, p, rhoa, zmet, rmu, thcond,
                              re, r_wet, rlow_wet, rup_wet,
                              rmass_2d, dm_2d,
                              pratt, prat, pden1, palr,
                              dtime):
        """One growth step with full physics recomputation.

        Recomputes every step: vapor pressure, diffusivity, latent heats,
        free paths, Knudsen corrections, ventilation, growth kernel.

        Args:
            pc: Particle concentrations (NZ, NBIN, NELEM).
            gc: Gas concentrations (NZ, NGAS).
            t: Temperature (NZ,) [K].
            p: Pressure (NZ,) [dyne/cm^2].
            rhoa: Air density * zmet (NZ,).
            zmet: Vertical metric (NZ,).
            rmu: Dynamic viscosity (NZ,) [g/cm/s].
            thcond: Thermal conductivity (NZ,) [erg/cm/s/K].
            re: Reynolds number (NZ, NBIN, NGROUP).
            r_wet: Wet radius (NZ, NBIN, NGROUP) [cm].
            rlow_wet: Lower boundary wet radius (NZ, NBIN, NGROUP) [cm].
            rup_wet: Upper boundary wet radius (NZ, NBIN, NGROUP) [cm].
            rmass_2d: Particle mass (NBIN, NGROUP) [g].
            dm_2d: Bin mass width (NBIN, NGROUP) [g].
            pratt, prat, pden1, palr: PPM coefficients.
            dtime: Timestep [s].

        Returns:
            Tuple of (pc, gc, t, rlheat_val).
        """
        iz = 0
        tt = t[iz]
        rhoa_cgs = rhoa[iz] / zmet[iz]

        # ============================================================
        # RECOMPUTE T-DEPENDENT QUANTITIES
        # ============================================================

        # --- Vapor pressure (Murphy 2005, inlined) ---
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

        # --- Gas diffusivity (T-dependent) ---
        D = DTYPE(0.211) * (DTYPE(1.01325e6) / p[iz]) * (tt / DTYPE(273.15)) ** DTYPE(1.94)

        # --- Latent heats (T-dependent) ---
        rlhe = (DTYPE(2.5) - DTYPE(0.00239) * (tt - DTYPE(273.16))) * DTYPE(1e10)
        rlhm = (DTYPE(79.7) + DTYPE(0.485) * (tt - DTYPE(273.16))
                 - DTYPE(2.5e-3) * (tt - DTYPE(273.16))**2) * DTYPE(4.186e7)
        rlh = jnp.where(is_ice, rlhe + rlhm, rlhe)

        # --- Thermal conductivity (T-dependent) ---
        thcond_val = (DTYPE(5.69) + DTYPE(0.017) * (tt - T0)) * DTYPE(4.186e2)

        # --- Viscosity (T-dependent, Sutherland) ---
        rmu_val = (DTYPE(1.8325e-4) * (DTYPE(296.16) + DTYPE(120.0))
                   / (tt + DTYPE(120.0)) * (tt / DTYPE(296.16)) ** DTYPE(1.5))

        # --- Surface tensions (T-dependent) ---
        surfctwa = DTYPE(76.10) - DTYPE(0.155) * (tt - T0)
        surfctia = DTYPE(141.0) - DTYPE(0.15) * tt

        # --- Kelvin factors (T-dependent) ---
        akelvin_val = DTYPE(2.0) * _gwtmol * surfctwa / (tt * RHO_W * RGAS)
        akelvini_val = DTYPE(2.0) * _gwtmol * surfctia / (tt * RHO_W * RGAS)

        # --- Free paths (T-dependent) ---
        freep = DTYPE(3.0) * D * jnp.sqrt(PI * _gwtmol / (DTYPE(8.0) * RGAS * tt))
        freept = freep * thcond_val / (D * rhoa_cgs * (CP - RGAS / (DTYPE(2.0) * WTMOL_AIR)))

        # --- Growth kernel over ALL bins (vectorized) ---
        br = rlow_wet[iz, :, 0]  # (NBIN,) bin boundary radii

        rknudn = freep / jnp.maximum(br, DTYPE(1e-30))
        rknudnt = freept / jnp.maximum(br, DTYPE(1e-30))

        rlam = ((DTYPE(1.33) * rknudn + DTYPE(0.71)) / (rknudn + DTYPE(1.0))
                + DTYPE(4.0) * (DTYPE(1.0) - _gstick) / (DTYPE(3.0) * _gstick))
        rlamt = ((DTYPE(1.33) * rknudnt + DTYPE(0.71)) / (rknudnt + DTYPE(1.0))
                 + DTYPE(4.0) * (DTYPE(1.0) - _tstick) / (DTYPE(3.0) * _tstick))

        diffus1 = D / (DTYPE(1.0) + rlam * rknudn)
        thcond1 = thcond_val / (DTYPE(1.0) + rlamt * rknudnt)

        schn = rmu_val / (rhoa_cgs * diffus1)
        prnum = rmu_val * CP / thcond1

        reyn = re[iz, :, 0]
        x1 = schn ** (DTYPE(1.0) / DTYPE(3.0)) * jnp.sqrt(reyn)
        x2 = prnum ** (DTYPE(1.0) / DTYPE(3.0)) * jnp.sqrt(reyn)

        fv = jnp.where(is_ice,
                        jnp.where(x1 <= DTYPE(1.0), DTYPE(1.0) + DTYPE(0.14) * x1**2, DTYPE(0.86) + DTYPE(0.28) * x1),
                        jnp.where(x1 <= DTYPE(1.4), DTYPE(1.0) + DTYPE(0.108) * x1**2, DTYPE(0.78) + DTYPE(0.308) * x1))
        ft = jnp.where(is_ice,
                        jnp.where(x2 <= DTYPE(1.0), DTYPE(1.0) + DTYPE(0.14) * x2**2, DTYPE(0.86) + DTYPE(0.28) * x2),
                        jnp.where(x2 <= DTYPE(1.4), DTYPE(1.0) + DTYPE(0.108) * x2**2, DTYPE(0.78) + DTYPE(0.308) * x2))

        gro = DTYPE(4.0) * PI * br * diffus1 * fv * _gwtmol / (BK * tt * AVG)  # (NBIN,)
        gro1 = _gwtmol * rlh**2 / (RGAS * tt**2 * ft * thcond1) / (DTYPE(4.0) * PI * br)  # (NBIN,)

        # ============================================================
        # GROWTH PHYSICS (same as before)
        # ============================================================

        # --- Supersaturation ---
        rvap = RGAS / _gwtmol
        gc_cgs = gc[iz, 0] / zmet[iz]
        pvap_use = jnp.where(is_ice, pvapi, pvapl)
        ss = jnp.where(is_ice,
                        (gc_cgs * rvap * tt - pvapi) / pvapi,
                        (gc_cgs * rvap * tt - pvapl) / pvapl)

        # --- Total condensate before growth ---
        pc_1d = pc[iz, :, 0]
        prev_condensate = jnp.sum(pc_1d * rmass_2d[:, 0])

        # --- Vectorized pheat: dmdt at bin boundaries ---
        g0 = gro[1:]   # (NBIN-1,) boundary 0→1, 1→2, ...
        g1_b = gro1[1:] # (NBIN-1,)

        akelv = jnp.where(is_ice, akelvini_val, akelvin_val)
        r_bound = rup_wet[iz, :nbin - 1, 0]
        expon = jnp.clip(akelv / jnp.maximum(r_bound, DTYPE(1e-30)), -POWMAX, POWMAX)
        akas = jnp.exp(expon)

        dmdt_all = pvap_use * (ss + DTYPE(1.0) - akas) * g0 / (DTYPE(1.0) + g0 * g1_b * pvap_use)

        # --- Vectorized PPM ---
        dm_1d = dm_2d[:, 0]
        dpc = pc_1d / dm_1d

        # Gradient
        d_center = dpc[1:-1]
        d_right = dpc[2:]
        d_left = dpc[:-2]
        dela = jnp.zeros(nbin, dtype=DTYPE)
        delma = jnp.zeros(nbin, dtype=DTYPE)

        dela_interior = pratt[0, 1:-1, 0] * (pratt[1, 1:-1, 0] * (d_right - d_center) + pratt[2, 1:-1, 0] * (d_center - d_left))
        dela = dela.at[1:-1].set(dela_interior)

        cond = (d_right - d_center) * (d_center - d_left) > DTYPE(0.0)
        limited = jnp.minimum(jnp.abs(dela_interior),
                               jnp.minimum(DTYPE(2.0) * jnp.abs(d_center - d_right),
                                           DTYPE(2.0) * jnp.abs(d_center - d_left))) * jnp.sign(dela_interior)
        delma = delma.at[1:-1].set(jnp.where(cond, limited, DTYPE(0.0)))

        # Boundary values
        aju = jnp.zeros(nbin, dtype=DTYPE)
        aju_interior = (dpc[1:-2] + prat[0, 1:-2, 0] * (dpc[2:-1] - dpc[1:-2])
                        + DTYPE(1.0) / pden1[1:-2, 0] * (
                            prat[1, 1:-2, 0] * (prat[2, 1:-2, 0] - prat[3, 1:-2, 0]) * (dpc[2:-1] - dpc[1:-2])
                            - dm_1d[1:-2] * prat[2, 1:-2, 0] * delma[2:-1]
                            + dm_1d[2:-1] * prat[3, 1:-2, 0] * delma[1:-2]))
        aju = aju.at[1:-2].set(aju_interior)

        # Endpoints
        al = jnp.zeros(nbin, dtype=DTYPE)
        ar = jnp.zeros(nbin, dtype=DTYPE)
        al = al.at[2:-2].set(aju[1:-3])
        ar = ar.at[2:-2].set(aju[2:-2])
        ar = ar.at[1].set(aju[1])
        al = al.at[1].set(dpc[0] + palr[0, 0] * (dpc[1] - dpc[0]))
        ar = ar.at[0].set(al[1])
        al = al.at[0].set(dpc[0] + palr[1, 0] * (dpc[1] - dpc[0]))
        al = al.at[-2].set(aju[-3])
        ar = ar.at[-2].set(dpc[-2] + palr[2, 0] * (dpc[-1] - dpc[-2]))
        al = al.at[-1].set(ar[-2])
        ar = ar.at[-1].set(dpc[-2] + palr[3, 0] * (dpc[-1] - dpc[-2]))

        # Monotonicity
        outside = (ar - dpc) * (dpc - al) <= DTYPE(0.0)
        al = jnp.where(outside, dpc, al)
        ar = jnp.where(outside, dpc, ar)
        diff = ar - al
        test1 = diff * (dpc - DTYPE(0.5) * (al + ar))
        test2 = diff**2 / DTYPE(6.0)
        al = jnp.where(test1 > test2, DTYPE(3.0) * dpc - DTYPE(2.0) * ar, al)
        ar = jnp.where(test1 < -test2, DTYPE(3.0) * dpc - DTYPE(2.0) * al, ar)

        dela_flux = ar - al
        a6 = DTYPE(6.0) * (dpc - DTYPE(0.5) * (ar + al))

        # --- growlg / evaplg ---
        has_particles = jnp.max(pc_1d / zmet[iz]) > FEW_PC

        x_grow = dmdt_all * dtime / dm_1d[:nbin - 1]
        grow_ppm = (dmdt_all / jnp.maximum(pc_1d[:nbin - 1], DTYPE(1e-50))
                    * (ar[:nbin - 1] - DTYPE(0.5) * dela_flux[:nbin - 1] * x_grow
                       + (x_grow / DTYPE(2.0) - x_grow**2 / DTYPE(3.0)) * a6[:nbin - 1]))
        grow_upwind = dmdt_all / dm_1d[:nbin - 1]
        growlg_vals = jnp.where(x_grow < DTYPE(1.0), grow_ppm, grow_upwind)
        growlg_vals = jnp.where(dmdt_all > DTYPE(0.0), growlg_vals, DTYPE(0.0))
        growlg_vals = jnp.where(has_particles, growlg_vals, DTYPE(0.0))
        growlg = jnp.concatenate([growlg_vals, jnp.zeros(1, dtype=DTYPE)])

        x_evap = -dmdt_all * dtime / dm_1d[1:nbin]
        evap_ppm = (-dmdt_all / jnp.maximum(pc_1d[1:nbin], DTYPE(1e-50))
                    * (al[1:nbin] + DTYPE(0.5) * dela_flux[1:nbin] * x_evap
                       + (x_evap / DTYPE(2.0) - x_evap**2 / DTYPE(3.0)) * a6[1:nbin]))
        evap_upwind = -dmdt_all / dm_1d[1:nbin]
        evaplg_vals = jnp.where(x_evap < DTYPE(1.0), evap_ppm, evap_upwind)
        evaplg_vals = jnp.where(dmdt_all < DTYPE(0.0), evaplg_vals, DTYPE(0.0))
        evaplg_vals = jnp.where(has_particles, evaplg_vals, DTYPE(0.0))
        evaplg_0 = jnp.where((dmdt_all[0] < DTYPE(0.0)) & has_particles, -dmdt_all[0] / dm_1d[0], DTYPE(0.0))
        evaplg = jnp.concatenate([evaplg_0[None], evaplg_vals])

        # --- psolve via scan ---
        def psolve_step(pc_1d, ibin):
            growpe_val = jnp.where(ibin > 0, pc_1d[ibin - 1] * growlg[ibin - 1], DTYPE(0.0))
            pls = growlg[ibin] + evaplg[ibin]
            pc_new = (pc_1d[ibin] + dtime * growpe_val) / (DTYPE(1.0) + pls * dtime)
            pc_new = jnp.maximum(pc_new, SMALL_PC)
            pc_1d = pc_1d.at[ibin].set(pc_new)
            return pc_1d, None

        pc_1d, _ = jax.lax.scan(psolve_step, pc_1d, jnp.arange(nbin))

        # --- Evaporation production ---
        evappe = pc_1d[1:] * evaplg[1:]
        pc_1d = pc_1d.at[:nbin - 1].add(dtime * evappe)
        pc_1d = jnp.maximum(pc_1d, SMALL_PC)

        pc = pc.at[iz, :, 0].set(pc_1d)

        # --- Gas solver ---
        curr_condensate = jnp.sum(pc_1d * rmass_2d[:, 0])
        gasprod = (prev_condensate - curr_condensate) / dtime
        rlprod = -(prev_condensate - curr_condensate) * rlh / (CP * rhoa[iz] * dtime)
        gc = gc.at[iz, 0].add(dtime * gasprod)

        # --- Temperature solver ---
        t = t.at[iz].add(dtime * rlprod)
        rlheat_val = rlprod * dtime

        return pc, gc, t, rlheat_val

    @jax.jit
    def microfast_growth_substep(pc, gc, t, p, rhoa, zmet, rmu, thcond,
                                  re, r_wet, rlow_wet, rup_wet,
                                  rmass_2d, dm_2d,
                                  pratt, prat, pden1, palr,
                                  dtime_orig, max_substeps):
        """Growth step with adaptive substepping.

        Starts with 1 substep. If supersaturation changes sign after the step,
        doubles substeps and retries. Repeats up to log2(max_substeps) times.

        Args:
            ... same as microfast_growth_jit ...
            dtime_orig: Full timestep [s].
            max_substeps: Maximum number of substeps (power of 2, e.g. 128).

        Returns:
            Tuple of (pc, gc, t, rlheat_total).
        """
        iz = 0

        # Save initial supersaturation
        tt0 = t[iz]
        pvapi0 = DTYPE(10.0) * jnp.exp(
            DTYPE(9.550426) - DTYPE(5723.265) / tt0
            + DTYPE(3.53068) * jnp.log(tt0) - DTYPE(0.00728332) * tt0)
        rvap = RGAS / _gwtmol
        gc_cgs0 = gc[iz, 0] / zmet[iz]
        ss0 = (gc_cgs0 * rvap * tt0 - pvapi0) / pvapi0

        # Retry loop state: (pc, gc, t, nsub, nretry, rlheat_total, done)
        pc_saved = pc
        gc_saved = gc
        t_saved = t

        def retry_cond(state):
            _, _, _, _, nretry, _, done, _ = state
            return ~done

        def retry_body(state):
            pc_s, gc_s, t_s, nsub, nretry, rlheat_acc, done, max_sub = state

            # Reset to saved state
            pc_r = pc_saved
            gc_r = gc_saved
            t_r = t_saved

            dtime_sub = dtime_orig / nsub
            rlheat_sub = DTYPE(0.0)

            # Run nsub substeps using fori_loop with max_substeps iterations + masking
            def substep_body(i, carry):
                pc_c, gc_c, t_c, rl_c = carry
                active = i < nsub
                pc_new, gc_new, t_new, rl_new = microfast_growth_jit(
                    pc_c, gc_c, t_c, p, rhoa, zmet, rmu, thcond,
                    re, r_wet, rlow_wet, rup_wet,
                    rmass_2d, dm_2d, pratt, prat, pden1, palr, dtime_sub)
                pc_c = jax.tree.map(lambda o, n: jnp.where(active, n, o), pc_c, pc_new)
                gc_c = jax.tree.map(lambda o, n: jnp.where(active, n, o), gc_c, gc_new)
                t_c = jnp.where(active, t_new, t_c)
                rl_c = rl_c + jnp.where(active, rl_new, DTYPE(0.0))
                return pc_c, gc_c, t_c, rl_c

            pc_r, gc_r, t_r, rlheat_sub = jax.lax.fori_loop(
                0, max_sub, substep_body, (pc_r, gc_r, t_r, DTYPE(0.0)))

            # Check convergence: did supersaturation change sign?
            tt_new = t_r[iz]
            pvapi_new = DTYPE(10.0) * jnp.exp(
                DTYPE(9.550426) - DTYPE(5723.265) / tt_new
                + DTYPE(3.53068) * jnp.log(tt_new) - DTYPE(0.00728332) * tt_new)
            gc_cgs_new = gc_r[iz, 0] / zmet[iz]
            ss_new = (gc_cgs_new * rvap * tt_new - pvapi_new) / pvapi_new

            sign_changed = (ss0 * ss_new) < DTYPE(0.0)
            large_change = jnp.abs(ss_new) > DTYPE(0.05)
            needs_retry = sign_changed & large_change & (nsub < max_sub)

            new_nsub = jnp.where(needs_retry, nsub * 2, nsub)
            new_nretry = nretry + jnp.where(needs_retry, 1, 0)
            new_done = ~needs_retry

            return (pc_r, gc_r, t_r, new_nsub, new_nretry, rlheat_sub, new_done, max_sub)

        init_state = (pc, gc, t, jnp.int32(1), jnp.int32(0), DTYPE(0.0),
                      jnp.bool_(False), jnp.int32(max_substeps))

        pc_f, gc_f, t_f, nsub_f, nretry_f, rlheat_f, _, _ = jax.lax.while_loop(
            retry_cond, retry_body, init_state)

        return pc_f, gc_f, t_f, rlheat_f

    return microfast_growth_jit, microfast_growth_substep
