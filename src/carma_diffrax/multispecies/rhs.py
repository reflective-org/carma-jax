"""Multi-group continuous-time RHS for (pc, gc, T).

Extends the sulfate-only `carma_diffrax.rhs` to handle three groups
(sulfate, cloud_water, ice_crystal) with phase transitions:

  - Sulfate group: gas H2SO4 ↔ particle via condensation/evaporation
                    + homogeneous nucleation (Vehkamäki / Zhao-Turco).
  - Cloud_water group: gas H2O ↔ droplet via condensation/evaporation
                        + CCN activation from sulfate (actdropl)
                        + melting from ice (melticel).
  - Ice_crystal group: gas H2O ↔ ice via deposition/sublimation
                        + freezing from cloud_water (freezdropl).

Mass balance per gas is exact by construction:
    dgc[g]/dt = − d(Σ rmass · pc)_g/dt − (nucleation mass loss from gas)
for the gas g that supplies that condensed phase.

Per-RHS-call:
  1. Recompute vapor pressures and supersaturation from current (T, gc).
  2. For each group: per-bin dmdt via pheat (vmap), then upwind number-density
     flux in mass space → group-local d(pc)/dt.
  3. Add nucleation + phase-transition source/sink between groups.
  4. Mass balance for each gas (sum of bin-mass changes for that gas).
  5. Latent heat → dT/dt.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from carma.constants import AVG
from carma.growth.pheat import pheat
from carma.nucleation.actdropl import actdropl
from carma.nucleation.freezdropl import freezdropl
from carma.nucleation.melticel import melticel
from carma.nucleation.sulfnuc import sulfnuc
from carma.precision import DTYPE
from carma.sulfate_utils import wtpct_tabaz
from carma.supersaturation import supersat
from carma.vapor_pressure import vaporp_h2o_murphy2005, vaporp_h2so4_ayers1980

from carma_diffrax.multispecies.config import (
    MultiSpeciesConfig,
)
from carma_diffrax.multispecies.state import MultiSpeciesShape, pack, unpack


# Sulfate-scope physical constants.
_GWTMOL_H2SO4 = DTYPE(98.078479)
_GWTMOL_H2O = DTYPE(18.01528)
_CP_AIR = DTYPE(1.004e7)        # erg / g / K
_IZ = 0

# Default nucleation method, matches Fortran ensemble setup.
_NUCLEATION_METHOD = "ZhaoTurco"
# Activation supersaturation threshold for actdropl (fraction).
# Per-bin scrit could be computed from Köhler theory; for cut 2 we use a
# constant 1% — small enough that any actual supersaturation activates,
# large enough that the dry strat (s ≈ −0.5) never fires.
_ACTDROPL_SCRIT_VAL = 0.01


class FrozenEnvMS(NamedTuple):
    """Per-outer-step constant environment tables.

    Mirror of the sulfate FrozenEnv but with arrays sized for ngroup ≥ 1.
    Frozen for the duration of one outer step; the user rebuilds it from
    current (T, gc) at the start of each step.
    """
    akelvin: jnp.ndarray         # (1, ngas)
    akelvini: jnp.ndarray        # (1, ngas)
    gro: jnp.ndarray             # (1, nbin, ngroup)
    gro1: jnp.ndarray            # (1, nbin, ngroup)
    rup_wet: jnp.ndarray         # (1, nbin, ngroup)
    rlow_wet: jnp.ndarray        # (1, nbin, ngroup)
    r_wet: jnp.ndarray           # (nbin, ngroup) — bin-center wet radius
    rmass_2d: jnp.ndarray        # (nbin, ngroup) — bin-center mass [g]
    dm_2d: jnp.ndarray           # (nbin, ngroup) — true bin width [g]
    rmassup_2d: jnp.ndarray      # (nbin, ngroup) — upper bin mass [g]
    rhoa: jnp.ndarray            # (1,) [g/cm^3]
    zmet: jnp.ndarray            # (1,)
    rlhe: jnp.ndarray            # (1, ngas) — latent heat of evap [erg/g]
    rlhm: jnp.ndarray            # (1, ngas) — latent heat of melt [erg/g]


def _bin_transfer(dmdt_inner: jnp.ndarray, pc_g: jnp.ndarray,
                  dm_g: jnp.ndarray) -> jnp.ndarray:
    """Upwind number-density flux in mass space for one group.

    Args:
        dmdt_inner: (nbin-1,) — per-particle growth rate at each bin boundary.
        pc_g: (nbin,) — number concentration in this group [#/cm^3].
        dm_g: (nbin,) — true bin width [g].

    Returns:
        d(pc_g)/dt : (nbin,) [#/cm^3/s].
    """
    nbin = pc_g.shape[0]
    zero = jnp.zeros((1,), dtype=DTYPE)
    dmdt = jnp.concatenate([dmdt_inner, zero])         # (nbin,)
    pc_above = jnp.concatenate([pc_g[1:], zero])       # (nbin,) shifted up
    dm_above = jnp.concatenate(
        [dm_g[1:], jnp.ones((1,), dtype=DTYPE)],
    )
    # Number density at boundary b — upwind based on sign of dmdt:
    #   if growth (dmdt > 0): use n_left = pc[b] / dm[b]
    #   if evap (dmdt < 0):  use n_right = pc[b+1] / dm[b+1]
    n_at = jnp.where(dmdt > 0, pc_g / dm_g, pc_above / dm_above)
    F = dmdt * n_at                                     # (nbin,) [#/cm^3/s]
    F_below = jnp.concatenate([zero, F[:-1]])
    dpc_dt = F_below - F                                # d(pc)/dt for this group
    # Bin-0 evap-out sink (mirrors Fortran growevapl L243-249): when
    # boundary 0 is in evap mode, bin 0 also drains "downward" — physically,
    # sub-monomer cluster dissolves back into vapor. Without this, particles
    # entering bin 0 via downward upwind flux accumulate forever.
    evap_bin0_out = jnp.where(
        dmdt[0] < 0,
        -dmdt[0] / dm_g[0] * pc_g[0],
        DTYPE(0.0),
    )
    return dpc_dt.at[0].add(-evap_bin0_out)


def make_rhs_ms(*args, **kwargs):
    """Build a JIT-compiled multispecies RHS.

    Two call signatures supported:

      make_rhs_ms(ms, shape, do_homogeneous_nuc=…, …)
          → new (preferred): rhs(t, y, env) takes env via args. The JIT
          cache is reused across outer steps regardless of env values.

      make_rhs_ms(env, ms, shape, …)
          → legacy: rhs(t, y, _) with env baked in via closure. Triggers
          re-trace whenever env changes.
    """
    # New signature: first positional arg is MultiSpeciesConfig.
    if args and isinstance(args[0], MultiSpeciesConfig):
        return _make_rhs_ms_args(*args, **kwargs)
    # Legacy signature: first positional arg is FrozenEnvMS.
    if args and isinstance(args[0], FrozenEnvMS):
        env_legacy = args[0]
        rest = args[1:]
        rhs_v2 = _make_rhs_ms_args(*rest, **kwargs)
        def rhs_legacy(t, y, _ignored):
            return rhs_v2(t, y, env_legacy)
        return rhs_legacy
    raise TypeError(
        "make_rhs_ms takes (ms, shape, ...) or (env, ms, shape, ...); "
        f"first arg type is {type(args[0]).__name__ if args else 'None'}"
    )


def _make_rhs_ms_args(ms: MultiSpeciesConfig,
                       shape: MultiSpeciesShape,
                       *,
                       do_homogeneous_nuc: bool = True,
                       do_ccn_activation: bool = True,
                       do_droplet_freezing: bool = True,
                       do_ice_melting: bool = True):
    """Actual multispecies RHS builder — env passed at call time via args."""
    cfg = ms.cfg
    nbin = shape.nbin
    ngroup = cfg.ngroup
    ngas = cfg.ngas
    iz = _IZ

    igroup_sulf = int(jnp.where(ms.is_sulfate)[0][0])
    # cloud_water = is_cloud and not is_ice; ice = is_ice.
    cloud_mask = ms.is_cloud & ~ms.is_ice
    ice_mask = ms.is_ice
    igroup_cw = int(jnp.where(cloud_mask)[0][0]) if cloud_mask.any() else -1
    igroup_ice = int(jnp.where(ice_mask)[0][0]) if ice_mask.any() else -1

    igas_h2o = int(cfg.igash2o)
    igas_h2so4 = int(cfg.igash2so4)

    @jax.jit
    def rhs(t, y, env):
        del t
        pc, gc, T_scalar = unpack(y, shape)              # pc: (nbin, nelem)
        T = jnp.atleast_1d(T_scalar)                     # (1,)
        gc_2d = gc[None, :]                              # (1, ngas)

        # 1) Vapor pressures from current (T, gc[H2O]).
        pvapl_h2o, pvapi_h2o = vaporp_h2o_murphy2005(T)
        pvap_h2so4, _ = vaporp_h2so4_ayers1980(
            T, gc_2d[:, igas_h2o], pvapl_h2o, env.zmet,
        )
        pvapl = jnp.stack([pvapl_h2o, pvap_h2so4], axis=1)    # (1, 2)
        pvapi = jnp.stack([pvapi_h2o, pvap_h2so4], axis=1)

        # 2) Supersaturation per gas.
        ssl_h2o, ssi_h2o = supersat(
            T, gc_2d[:, igas_h2o], pvapl[:, igas_h2o],
            pvapi[:, igas_h2o], _GWTMOL_H2O, env.zmet,
        )
        ssl_h2so4, ssi_h2so4 = supersat(
            T, gc_2d[:, igas_h2so4], pvapl[:, igas_h2so4],
            pvapi[:, igas_h2so4], _GWTMOL_H2SO4, env.zmet,
        )
        supsatl = jnp.stack([ssl_h2o, ssl_h2so4], axis=1)
        supsati = jnp.stack([ssi_h2o, ssi_h2so4], axis=1)

        # 3) Per-group growth/evap via pheat + upwind transfer.
        # We loop over ngroup (Python loop, static) and accumulate dpc/dt.
        pc_3d = pc[None, :, :]                           # (1, nbin, nelem)
        dpc_dt = jnp.zeros((nbin, cfg.nelem), dtype=DTYPE)

        for ig in range(ngroup):
            ielem_num = int(ms.ienconc[ig])
            igas = int(ms.igrowgas[ielem_num])
            if igas < 0:
                continue
            is_ice_g = bool(ms.is_ice[ig])

            def _dmdt_at(ibin, _ig=ig, _igas=igas, _is_ice=is_ice_g):
                return pheat(
                    pc_3d, supsatl, supsati, pvapl, pvapi,
                    env.akelvin, env.akelvini, env.gro, env.gro1,
                    env.rup_wet, env.rmass_2d,
                    _is_ice, iz, _ig, ibin, _igas,
                )

            dmdt_inner = jax.vmap(_dmdt_at)(jnp.arange(nbin - 1))   # (nbin-1,)
            pc_g = pc[:, ielem_num]
            dm_g = env.dm_2d[:, ig]
            dpc_g = _bin_transfer(dmdt_inner, pc_g, dm_g)           # (nbin,)
            dpc_dt = dpc_dt.at[:, ielem_num].add(dpc_g)

        # 4) Nucleation / phase transitions.
        # --- 4a) Homogeneous sulfate nucleation (H2SO4 → sulfate group) ---
        h2o_cgs = gc_2d[iz, igas_h2o] / env.zmet[iz]
        h2so4_cgs = gc_2d[iz, igas_h2so4] / env.zmet[iz]
        h2o_n = h2o_cgs * AVG / _GWTMOL_H2O
        h2so4_n = h2so4_cgs * AVG / _GWTMOL_H2SO4
        rh = ssl_h2o[iz] + DTYPE(1.0)
        wtp = wtpct_tabaz(T[iz], h2o_cgs, pvapl[iz, igas_h2o])

        nucl_sulf_rate = jnp.zeros(nbin, dtype=DTYPE)
        if do_homogeneous_nuc:
            rmassup_sulf = env.rmassup_2d[:, igroup_sulf]
            r_wet_sulf = env.r_wet[:, igroup_sulf]
            rhompe_1d, _rnuclg = sulfnuc(
                T[iz], wtp, rh, h2so4_n, h2so4_cgs, h2o_n, h2o_cgs,
                r_wet_sulf, rmassup_sulf,
                DTYPE(ms.cfg.groups[igroup_sulf].rmrat), env.zmet[iz],
                method=_NUCLEATION_METHOD,
                do_homogeneous=True, do_heterogeneous=False,
                gwtmol_h2so4=_GWTMOL_H2SO4, gwtmol_h2o=_GWTMOL_H2O,
            )
            nucl_sulf_rate = rhompe_1d / env.zmet[iz]
            dpc_dt = dpc_dt.at[:, int(ms.ienconc[igroup_sulf])].add(nucl_sulf_rate)

        # Sum particle conc per group for pconmax gates.
        from carma.utils.smallconc import maxconc
        pconmax = maxconc(pc_3d, jnp.asarray(ms.ienconc), env.zmet)   # (1, ngroup)

        # --- 4b) CCN activation: sulfate → cloud_water ---
        ccn_rate = jnp.zeros(nbin, dtype=DTYPE)
        if do_ccn_activation and igroup_cw >= 0:
            pc_sulf = pc[:, int(ms.ienconc[igroup_sulf])]
            scrit_bins = jnp.full((nbin,), DTYPE(_ACTDROPL_SCRIT_VAL))
            evappe_target = jnp.zeros((nbin,), dtype=DTYPE)
            rnuclg = actdropl(
                T[iz], ssl_h2o[iz], pconmax[iz, igroup_sulf],
                scrit_bins, pc_sulf, evappe_target,
            )
            ccn_rate = rnuclg * pc_sulf
            dpc_dt = dpc_dt.at[:, int(ms.ienconc[igroup_sulf])].add(-ccn_rate)
            dpc_dt = dpc_dt.at[:, int(ms.ienconc[igroup_cw])].add(ccn_rate)

        # --- 4c) Droplet freezing: cloud_water → ice ---
        freeze_rate = jnp.zeros(nbin, dtype=DTYPE)
        if do_droplet_freezing and igroup_cw >= 0 and igroup_ice >= 0:
            pc_cw = pc[:, int(ms.ienconc[igroup_cw])]
            rnuclg = freezdropl(T[iz], pc_cw)
            freeze_rate = rnuclg * pc_cw
            dpc_dt = dpc_dt.at[:, int(ms.ienconc[igroup_cw])].add(-freeze_rate)
            dpc_dt = dpc_dt.at[:, int(ms.ienconc[igroup_ice])].add(freeze_rate)

        # --- 4d) Ice melting: ice → cloud_water ---
        melt_rate = jnp.zeros(nbin, dtype=DTYPE)
        if do_ice_melting and igroup_cw >= 0 and igroup_ice >= 0:
            pc_ice = pc[:, int(ms.ienconc[igroup_ice])]
            rnuclg = melticel(T[iz], pconmax[iz, igroup_ice], nbin)
            melt_rate = rnuclg * pc_ice
            dpc_dt = dpc_dt.at[:, int(ms.ienconc[igroup_ice])].add(-melt_rate)
            dpc_dt = dpc_dt.at[:, int(ms.ienconc[igroup_cw])].add(melt_rate)

        # 5) Mass balance per gas — by construction so total mass is conserved
        # to machine precision regardless of the bin scheme's discrete approx.
        # H2SO4: only sulfate group exchanges mass.
        rmass_sulf = env.rmass_2d[:, igroup_sulf]
        dM_sulf = jnp.sum(rmass_sulf * dpc_dt[:, int(ms.ienconc[igroup_sulf])])
        # Subtract phase-transition contributions, which don't exchange mass
        # with H2SO4 vapor (they just move particles between sulfate and water
        # groups). CCN activation transports sulfate-bound mass to cloud_water,
        # not back to gas. So the gas balance must be computed from grow/evap
        # + nucleation only, not from total dpc.
        # We accumulate the net "grow/evap + homog. nucleation" contribution
        # to the sulfate group during 4a; the CCN-out term is subtracted from
        # sulfate but it returns the same mass to cloud_water below.
        # Easier: track the gas-side mass flow explicitly.
        rmass_cw_per_ccn = jnp.sum(rmass_sulf * ccn_rate)
        dM_sulf_no_ccn = dM_sulf + rmass_cw_per_ccn      # add back what went to CW
        dgc_h2so4_cgs = -dM_sulf_no_ccn

        # H2O: cloud_water and ice groups exchange mass with H2O vapor via
        # condensation/evap. Phase transitions (freeze, melt, ccn) between
        # condensed phases don't touch the gas.
        if igroup_cw >= 0:
            rmass_cw = env.rmass_2d[:, igroup_cw]
            # Net cloud_water mass change excluding internal transfers
            #   (freezing out, melting in, ccn in)
            dpc_cw = dpc_dt[:, int(ms.ienconc[igroup_cw])]
            dpc_cw_grow_evap = (dpc_cw + freeze_rate - melt_rate - ccn_rate)
            dM_cw = jnp.sum(rmass_cw * dpc_cw_grow_evap)
        else:
            dM_cw = DTYPE(0.0)
        if igroup_ice >= 0:
            rmass_ice = env.rmass_2d[:, igroup_ice]
            dpc_ice = dpc_dt[:, int(ms.ienconc[igroup_ice])]
            dpc_ice_grow_evap = (dpc_ice - freeze_rate + melt_rate)
            dM_ice = jnp.sum(rmass_ice * dpc_ice_grow_evap)
        else:
            dM_ice = DTYPE(0.0)
        dgc_h2o_cgs = -(dM_cw + dM_ice)

        dgc_dt = jnp.zeros(ngas, dtype=DTYPE)
        dgc_dt = dgc_dt.at[igas_h2o].set(dgc_h2o_cgs * env.zmet[iz])
        dgc_dt = dgc_dt.at[igas_h2so4].set(dgc_h2so4_cgs * env.zmet[iz])

        # 6) Latent heat → dT/dt.
        # rlprod = (mass condensing per cm^3 per s) × rlhe (per gas).
        # For ice growth, use rlhe + rlhm (vap → ice releases both).
        # Approximation: split based on which condensed phase is changing.
        rlprod_h2so4 = -dgc_h2so4_cgs * env.rlhe[iz, igas_h2so4]
        rlprod_h2o_liq = -(dM_cw if igroup_cw >= 0 else DTYPE(0.0)) \
                          * env.rlhe[iz, igas_h2o]
        rlprod_h2o_ice = -(dM_ice if igroup_ice >= 0 else DTYPE(0.0)) \
                          * (env.rlhe[iz, igas_h2o] + env.rlhm[iz, igas_h2o])
        # Freeze ↔ melt latent heat (between liquid and ice condensed phases)
        rlprod_freeze = (jnp.sum(rmass_sulf * 0.0)  # placeholder for typing
                         + jnp.sum((env.rmass_2d[:, igroup_cw] * freeze_rate
                                    if igroup_cw >= 0 else 0.0))
                           * env.rlhm[iz, igas_h2o])
        rlprod_melt = -(jnp.sum((env.rmass_2d[:, igroup_ice] * melt_rate
                                 if igroup_ice >= 0 else 0.0))
                        * env.rlhm[iz, igas_h2o])
        rlprod = rlprod_h2so4 + rlprod_h2o_liq + rlprod_h2o_ice \
                  + rlprod_freeze + rlprod_melt
        dT_dt = rlprod / (_CP_AIR * env.rhoa[iz])

        return pack(dpc_dt, dgc_dt, dT_dt)

    return rhs
