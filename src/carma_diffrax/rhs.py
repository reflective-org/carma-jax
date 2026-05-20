"""rhs.py — continuous-time ODE RHS for sulfate (pc, gc, T).

This builds the right-hand side that `diffrax.diffeqsolve` integrates over one
outer step. The state vector is::

    y = pack(pc, gc, T)   length nbin*nelem + ngas + 1

The faithful CARMA port uses `growevapl`, a Piecewise-Parabolic-Method (PPM)
finite-volume scheme that returns *time-averaged* loss rates over a substep —
not suitable as a continuous-time RHS. Here we use the underlying primitive
`pheat()`, which returns the *instantaneous* mass growth rate at a bin
boundary, and combine it with a first-order upwind bin-transfer scheme to
build a true ODE RHS::

    dpc[i]/dt = grow_in[i] - grow_out[i] + evap_in[i] - evap_out[i] + nucl[i]
    dgc[H2SO4]/dt = - (mass uptake by growth across all boundaries
                       + mass uptake by nucleation events)
    dT/dt = (latent heat from H2SO4 condensation) / (CP * rhoa)

Mass conservation is exact by construction: the gas-side rate is computed
from the same per-particle mass-change rate that drives the bin-spectrum
evolution. Number is conserved by the upwind scheme since every flux
appears once as a "loss from bin b" and once as a "gain to its neighbor."

Environment tables (akelvin, gro, gro1, rup_wet, rmassup, ...) are computed
once at the outer-step boundary from the initial T and held constant inside
the diffeqsolve. Vapor pressures and supersaturation, however, are
recomputed each RHS call from the current (T, gc) — those are the dominant
drivers of nucleation and growth and must follow the trajectory.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from carma.constants import AVG
from carma.growth.pheat import pheat
from carma.nucleation.sulfnuc import sulfnuc
from carma.precision import DTYPE
from carma.sulfate_utils import wtpct_tabaz
from carma.supersaturation import supersat
from carma.vapor_pressure import vaporp_h2o_murphy2005, vaporp_h2so4_ayers1980

from carma_diffrax.state import StateShape, pack, unpack


# Sulfate-scope constants. The faithful port has these in carma.config, but
# we hard-code them here since cut 1 is sulfate-only and they never change.
_GWTMOL_H2SO4 = DTYPE(98.078479)
_GWTMOL_H2O = DTYPE(18.01528)
_IGAS_H2O = 0
_IGAS_H2SO4 = 1
_IGROUP_SULFATE = 0
_IELEM_SULFATE = 0
_IZ = 0   # NZ=1 in this scope
# Specific heat of dry air at constant pressure, [erg / g / K] (CGS).
# Matches Fortran's CARMA_CONSTANTS_MOD CP_DRY_AIR (= 1.004e7 erg/g/K).
_CP_AIR = DTYPE(1.004e7)
# Default nucleation method matches Fortran ensemble setup.
_NUCLEATION_METHOD = "ZhaoTurco"


class FrozenEnv(NamedTuple):
    """Environment tables computed once per outer step.

    All arrays are sized for sulfate-only single-cell scope:
    nelem = ngroup = 1, NZ = 1, nbin = 38 (typically), ngas = 2.
    """
    akelvin: jnp.ndarray         # (1, ngas)
    akelvini: jnp.ndarray        # (1, ngas)
    gro: jnp.ndarray             # (1, nbin, ngroup)
    gro1: jnp.ndarray            # (1, nbin, ngroup)
    rup_wet: jnp.ndarray         # (1, nbin, ngroup) — wet radius at upper bin edge
    r_wet: jnp.ndarray           # (nbin,) — wet radius at bin center (for nucleation)
    rmass: jnp.ndarray           # (nbin, ngroup) — bin-center mass [g]
    dm: jnp.ndarray              # (nbin, ngroup) — bin mass width [g]
    rmassup: jnp.ndarray         # (nbin,) — upper bin-boundary mass [g]
    rmrat_val: float             # bin mass ratio (scalar)
    rhoa: jnp.ndarray            # (1,) [g/cm^3]
    zmet: jnp.ndarray            # (1,) — vertical metric; = 1 in single-column
    rlhe: jnp.ndarray            # (1, ngas) — latent heat of evap [erg/g]
    rlhm: jnp.ndarray            # (1, ngas) — latent heat of melt [erg/g]


def make_rhs(*args):
    """Build a JIT-compiled sulfate RHS.

    Two call signatures supported:
      - ``make_rhs(shape)``        → new: rhs(t, y, env) takes env via args.
      - ``make_rhs(env, shape)``   → legacy: rhs(t, y, _) with env baked in.

    The new form is preferred because it lets JAX reuse the JIT cache
    across outer steps (and across scenarios under vmap) even when env
    changes. Use diffrax's ``args=env`` parameter on ``diffeqsolve``.
    """
    if len(args) == 1 and isinstance(args[0], StateShape):
        return _make_rhs_args(args[0])
    elif len(args) == 2:
        env, shape = args
        rhs_v2 = _make_rhs_args(shape)
        # Bake env into a wrapper matching the legacy (t, y, args) signature.
        def rhs_legacy(t, y, _ignored):
            return rhs_v2(t, y, env)
        return rhs_legacy
    raise TypeError(
        "make_rhs takes either (shape,) or (env, shape); got "
        f"{len(args)} args"
    )


def _make_rhs_args(shape: StateShape):
    """The actual RHS builder. Closes over only the static shape."""
    nbin = shape.nbin
    ig = _IGROUP_SULFATE
    ielem = _IELEM_SULFATE
    iz = _IZ

    @jax.jit
    def rhs(t, y, env):
        del t
        pc, gc, T_scalar = unpack(y, shape)
        # Lift scalars into shapes the existing physics modules expect.
        T = jnp.atleast_1d(T_scalar)            # (1,)
        gc_2d = gc[None, :]                     # (1, ngas)
        pc_3d = pc[None, :, :]                  # (1, nbin, nelem)

        # --- 1) Vapor pressures from current T (and current gc[H2O]) ---
        pvapl_h2o, pvapi_h2o = vaporp_h2o_murphy2005(T)
        pvap_h2so4, _ = vaporp_h2so4_ayers1980(
            T, gc_2d[:, _IGAS_H2O], pvapl_h2o, env.zmet,
        )
        pvapl = jnp.stack([pvapl_h2o, pvap_h2so4], axis=1)   # (1, 2)
        pvapi = jnp.stack([pvapi_h2o, pvap_h2so4], axis=1)

        # --- 2) Supersaturation per gas ---
        ssl_h2o, _ = supersat(
            T, gc_2d[:, _IGAS_H2O], pvapl[:, _IGAS_H2O],
            pvapi[:, _IGAS_H2O], _GWTMOL_H2O, env.zmet,
        )
        ssl_h2so4, ssi_h2so4 = supersat(
            T, gc_2d[:, _IGAS_H2SO4], pvapl[:, _IGAS_H2SO4],
            pvapi[:, _IGAS_H2SO4], _GWTMOL_H2SO4, env.zmet,
        )
        supsatl = jnp.stack([ssl_h2o, ssl_h2so4], axis=1)
        supsati = jnp.stack([jnp.zeros_like(ssl_h2o), ssi_h2so4], axis=1)

        # --- 3) Per-boundary dmdt via vmap over ibin in [0, nbin-2] ---
        # pheat evaluates the mass growth rate at the boundary between
        # bins ibin and ibin+1.
        def _dmdt_at(ibin):
            return pheat(
                pc_3d, supsatl, supsati, pvapl, pvapi,
                env.akelvin, env.akelvini, env.gro, env.gro1,
                env.rup_wet, env.rmass,
                False, iz, ig, ibin, _IGAS_H2SO4,
            )
        dmdt_inner = jax.vmap(_dmdt_at)(jnp.arange(nbin - 1))   # (nbin-1,)
        # Pad with a trailing zero so dmdt has length nbin and we can index
        # boundary "above bin nbin-1" as zero (no boundary exists there).
        dmdt = jnp.concatenate([dmdt_inner, jnp.zeros((1,), dtype=DTYPE)])

        # --- 4) Upwind number-density flux in mass space ---
        # Continuous-time mass-space advection of n(m) at velocity v=dmdt:
        #   ∂n/∂t + ∂(v·n)/∂m = 0
        # Discretized via finite-volume on (pc[b] = ∫_bin n dm), the bin
        # update is d(pc[b])/dt = F[b-1] - F[b] where F[b] is the number
        # flux at boundary b. With upwind reconstruction of n at the
        # boundary:
        #   F[b] = v[b] · n_upwind = v[b] · (pc[b]/dm[b]   if v[b]>0
        #                                    else pc[b+1]/dm[b+1])
        # dm is the true bin width: rmassup[b] - rmasslow[b].
        dm_g = env.dm[:, ig]                                    # (nbin,)
        pc_elem = pc[:, ielem]                                  # (nbin,)
        zero = jnp.zeros((1,), dtype=DTYPE)
        pc_above = jnp.concatenate([pc_elem[1:], zero])         # pc[b+1]
        dm_above = jnp.concatenate([dm_g[1:], jnp.ones((1,), dtype=DTYPE)])

        n_at_boundary = jnp.where(
            dmdt > 0, pc_elem / dm_g, pc_above / dm_above,
        )
        F_n = dmdt * n_at_boundary                              # (nbin,) [#/cm^3/s]
        F_n_below = jnp.concatenate([zero, F_n[:-1]])           # F at b-1

        dpc_dt_ge = F_n_below - F_n                             # (nbin,)

        # --- 5) Nucleation rate (homogeneous; heterogeneous off for sulfate) ---
        h2o_cgs = gc_2d[iz, _IGAS_H2O] / env.zmet[iz]
        h2so4_cgs = gc_2d[iz, _IGAS_H2SO4] / env.zmet[iz]
        h2o_n = h2o_cgs * AVG / _GWTMOL_H2O
        h2so4_n = h2so4_cgs * AVG / _GWTMOL_H2SO4
        rh = ssl_h2o[iz] + DTYPE(1.0)
        wtp = wtpct_tabaz(T[iz], h2o_cgs, pvapl[iz, _IGAS_H2O])
        rhompe_1d, _rnuclg = sulfnuc(
            T[iz], wtp, rh, h2so4_n, h2so4_cgs, h2o_n, h2o_cgs,
            env.r_wet, env.rmassup, DTYPE(env.rmrat_val), env.zmet[iz],
            method=_NUCLEATION_METHOD,
            do_homogeneous=True, do_heterogeneous=False,
            gwtmol_h2so4=_GWTMOL_H2SO4, gwtmol_h2o=_GWTMOL_H2O,
        )
        # sulfnuc returns rates already multiplied by zmet (Fortran's "z" units).
        # Our pc state is in #/cm^3, so divide back out.
        nucl_rate = rhompe_1d / env.zmet[iz]                    # (nbin,)

        # --- 6) Total dpc/dt ---
        dpc_dt = dpc_dt_ge + nucl_rate                          # (nbin,)
        dpc_dt_2d = dpc_dt[:, None]                             # (nbin, nelem=1)

        # --- 7) dgc/dt by exact mass balance with the bin scheme ---
        # The discrete bin scheme is *not* exactly equivalent to the
        # continuous ∫ dmdt·n dm because adjacent bins differ by a factor
        # of rmrat in mass. To guarantee total mass conservation we set
        # dgc/dt = -d(particle mass)/dt computed directly from dpc/dt and
        # the nucleation rate. Whatever the bin scheme actually does to
        # particle mass, the gas absorbs the inverse.
        rmass_g = env.rmass[:, ig]                              # (nbin,)
        dM_p_dt = jnp.sum(rmass_g * dpc_dt_ge) + jnp.sum(rmass_g * nucl_rate)
        dgc_h2so4_cgs = -dM_p_dt
        # Convert to gc-units (mmr × rhoa × zmet). With single-column zmet=1
        # this is a no-op; kept for unit clarity.
        dgc_h2so4 = dgc_h2so4_cgs * env.zmet[iz]
        # H2O is a spectator in cut 1 — sulfate aerosol water uptake is
        # implicit in wtpct, no separate water-mass exchange. Future work
        # will couple H2O via the wet-radius equation.
        dgc_dt = jnp.array([DTYPE(0.0), dgc_h2so4], dtype=DTYPE)

        # --- 8) dT/dt from latent heat released by H2SO4 condensation ---
        # rlprod = (mass condensing per cm^3 per s) × rlhe[H2SO4]
        #        = (-dgc_h2so4_cgs) × rlhe[H2SO4]   [erg / cm^3 / s]
        # dT/dt = rlprod / (CP_air × rhoa)
        rlprod_h2so4 = -dgc_h2so4_cgs * env.rlhe[iz, _IGAS_H2SO4]
        dT_dt = rlprod_h2so4 / (_CP_AIR * env.rhoa[iz])

        return pack(dpc_dt_2d, dgc_dt, dT_dt)

    return rhs
