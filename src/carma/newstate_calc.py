"""Adaptive substepping engine for CARMA-JAX.

Implements the retry logic from newstate_calc.F90: if microfast
returns RC_WARNING_RETRY, double the substeps and retry from saved state.
Ported from: newstate_calc.F90
"""

import jax
import jax.numpy as jnp

from carma.constants import SMALL_PC, FEW_PC, CP, RGAS, WTMOL_H2O
from carma.enums import RC_OK, RC_WARNING_RETRY
from carma.precision import DTYPE
from carma.vapor_pressure import vaporp_h2o_murphy2005, vaporp_h2so4_ayers1980
from carma.supersaturation import supersat
from carma.growth.growevapl import growevapl
from carma.growth.growp import growp
from carma.growth.evapp import evapp, downgevapply
from carma.solvers.psolve import psolve
from carma.solvers.gsolve import gsolve
from carma.solvers.tsolve import tsolve
from carma.solvers.totalcondensate import totalcondensate


# Vapor pressure dispatch: build per-gas (NZ, NGAS) arrays based on the
# ivaprtn enum stored in each GasConfig. Mirrors the Fortran prestep's
# call to vaporp_<gas>_<formula>.F90 per gas.
#
# Why this isn't hard-coded to H2O: the sulfate ensemble's growing gas
# is H2SO4 (gas index 1). The previous version of microfast_growth used
# Murphy-Koop H2O for *every* gas slot, which gave H2O equilibrium
# vapor pressure for the H2SO4 column — wildly wrong (off by ~10
# orders of magnitude) and made physically supersaturated H2SO4 look
# subsaturated, so growth evaporated the IC instead of growing it.
def _build_pvapl_pvapi(t, gc, zmet, ivaprtn_arr, igas_h2o, ngas):
    """Build (NZ, NGAS) pvapl, pvapi arrays.

    ``ivaprtn_arr`` selects the saturation formula per gas:
        2 → I_VAPRTN_H2O_MURPHY2005
        4 → I_VAPRTN_H2SO4_AYERS1980
    Other H2O variants (Buck, Goff) fall back to Murphy-2005.
    """
    nz = t.shape[0]
    pvapl_h2o, pvapi_h2o = vaporp_h2o_murphy2005(t)        # (nz,)

    pvapl = jnp.zeros((nz, ngas), dtype=DTYPE)
    pvapi = jnp.zeros((nz, ngas), dtype=DTYPE)
    for igas in range(ngas):
        ivp = int(ivaprtn_arr[igas])
        if ivp == 4:
            # H2SO4 / Ayers-Kulmala — needs H2O gas + pvapl_h2o + zmet.
            pv, _ = vaporp_h2so4_ayers1980(
                t, gc[:, igas_h2o], pvapl_h2o, zmet,
            )
            pvapl = pvapl.at[:, igas].set(pv)
            pvapi = pvapi.at[:, igas].set(pv)
        else:
            # Default H2O Murphy-2005 (covers the sulfate test's H2O slot).
            pvapl = pvapl.at[:, igas].set(pvapl_h2o)
            pvapi = pvapi.at[:, igas].set(pvapi_h2o)
    return pvapl, pvapi


def _build_supsat(t, gc, pvapl, pvapi, gwtmol_arr, zmet, ngas):
    """Per-gas supsatl/supsati of shape (NZ, NGAS)."""
    nz = t.shape[0]
    supsatl = jnp.zeros((nz, ngas), dtype=DTYPE)
    supsati = jnp.zeros((nz, ngas), dtype=DTYPE)
    for igas in range(ngas):
        ssl, ssi = supersat(
            t, gc[:, igas], pvapl[:, igas], pvapi[:, igas],
            gwtmol_arr[igas], zmet,
        )
        supsatl = supsatl.at[:, igas].set(ssl)
        supsati = supsati.at[:, igas].set(ssi)
    return supsatl, supsati


def _convergence_gas(igrowgas_arr, ngas):
    """Pick the gas index used for the substep convergence check.

    Convention: the first element with a valid igrowgas wins. Falls
    back to gas 0 if no element grows (single-gas degenerate case).
    """
    for ielem in range(len(igrowgas_arr)):
        ig = int(igrowgas_arr[ielem])
        if 0 <= ig < ngas:
            return ig
    return 0


def microfast_growth(pc, gc, t, iz, dtime,
                     rhoa, zmet, rlhe, rlhm, diffus,
                     akelvin, akelvini, gro, gro1, gro2,
                     rup_wet, rmass_2d, dm_2d, rlow_wet,
                     pratt, prat, pden1, palr,
                     is_ice_arr, igrowgas_arr, ienconc_arr,
                     igroup_arr, gwtmol_arr,
                     nbin, ngroup, ngas, nelem,
                     dt_threshold, ds_threshold_arr, scale_threshold,
                     itype_arr=None,
                     ivaprtn_arr=None, igas_h2o=0):
    """Execute one microfast growth step at one level.

    Computes: vapor pressure → supersaturation → condensate →
    growth rates → growth production → psolve → gsolve → tsolve →
    check convergence.

    Returns:
        Tuple of (pc, gc, t, rlheat_val, rc)
    """
    # Default: assume every gas is Murphy H2O (legacy single-gas tests).
    if ivaprtn_arr is None:
        ivaprtn_arr = tuple(2 for _ in range(ngas))

    # Vapor pressure at current T — per-gas (NZ, NGAS).
    pvapl, pvapi = _build_pvapl_pvapi(t, gc, zmet, ivaprtn_arr,
                                          igas_h2o, ngas)

    # Supersaturation per gas (NZ, NGAS).
    supsatl, supsati = _build_supsat(t, gc, pvapl, pvapi, gwtmol_arr,
                                          zmet, ngas)

    # Convergence check tracks the *growing* gas (e.g. H2SO4 for the
    # sulfate ensemble), not gas 0 unconditionally.
    iconv = _convergence_gas(igrowgas_arr, ngas)
    prev_supsati = supsati[iz, iconv]

    # Total condensate before growth
    prev_ice, prev_liq = totalcondensate(
        pc, rmass_2d, igroup_arr, is_ice_arr, igrowgas_arr,
        nbin, ngroup, ngas, iz,
    )

    # Growth/evaporation loss rates
    growlg = jnp.zeros((nbin, ngroup), dtype=DTYPE)
    evaplg = jnp.zeros((nbin, ngroup), dtype=DTYPE)
    pconmax = jnp.max(pc[:, :, 0:1] / zmet[:, None, None], axis=1)

    growlg, evaplg = growevapl(
        pc, growlg, evaplg,
        supsatl, supsati, pvapl, pvapi,
        akelvin, akelvini, gro, gro1,
        rup_wet, rmass_2d, dm_2d, pconmax,
        pratt, prat, pden1, palr,
        is_ice_arr, igrowgas_arr, ienconc_arr,
        dtime, iz, nbin, ngroup,
    )

    # Growth production and solve per bin (step 1: growth only via psolve)
    growpe = jnp.zeros((nbin, nelem), dtype=DTYPE)
    evappe_psolve = jnp.zeros((nbin, nelem), dtype=DTYPE)  # zero for psolve (evap applied separately)
    rnucpe = jnp.zeros((nbin, nelem), dtype=DTYPE)
    rhompe = jnp.zeros((nbin, nelem), dtype=DTYPE)
    rnuclg = jnp.zeros((nbin, ngroup, ngroup), dtype=DTYPE)
    pc_nucl = jnp.zeros_like(pc)

    for ibin in range(nbin):
        for ielem in range(nelem):
            ig = int(igroup_arr[ielem])
            igrow = int(igrowgas_arr[ielem])
            growpe = growp(pc, growpe, growlg, pconmax, iz, ibin, ielem, ig, igrow)
            pc, pc_nucl = psolve(
                pc, pc_nucl, growpe, evappe_psolve, rnucpe, rhompe,
                growlg, evaplg, rnuclg, dtime, iz, ibin, ielem, ig, ngroup,
            )

    # Step 2: Evaporation production (particles shrinking from bin i to bin i-1)
    evappe = jnp.zeros((nbin, nelem), dtype=DTYPE)
    # itype_arr default matches the growtest case (single I_VOLATILE element).
    # Callers targeting JIT should pass a Python tuple so evapp's `int()`
    # indexing is trace-time, not traced.
    if itype_arr is None:
        itype_arr = (2,)  # I_VOLATILE
    evappe = evapp(
        pc, evappe, evaplg, pconmax, ienconc_arr, itype_arr,
        igroup_arr, iz, nbin, ngroup, nelem,
    )

    # Step 3: Apply evaporation production
    rnucpe_zero = jnp.zeros((nbin, nelem), dtype=DTYPE)
    pc = downgevapply(pc, evappe, rnucpe_zero, dtime, iz, nbin, nelem)

    # Gas solver
    curr_ice, curr_liq = totalcondensate(
        pc, rmass_2d, igroup_arr, is_ice_arr, igrowgas_arr,
        nbin, ngroup, ngas, iz,
    )

    rlprod = DTYPE(0.0)
    gc, rlprod, rc_gas = gsolve(
        gc, rlprod, prev_ice, prev_liq, curr_ice, curr_liq,
        rlhe, rlhm, rhoa, dtime, iz, ngas,
        ds_threshold_arr, scale_threshold,
    )

    # Temperature solver
    rlheat = jnp.zeros(t.shape[0], dtype=DTYPE)
    partheat = jnp.zeros(t.shape[0], dtype=DTYPE)
    t, rlheat, partheat, rc_t = tsolve(
        t, rlheat, partheat, rlprod, DTYPE(0.0), dtime, iz,
        dt_threshold, scale_threshold,
    )

    # Check supersaturation convergence (all traced) on the growing
    # gas — not gas 0 — so the substep retry tracks H2SO4 saturation
    # for the sulfate column, not H2O.
    pvapl_new, pvapi_new = _build_pvapl_pvapi(
        t, gc, zmet, ivaprtn_arr, igas_h2o, ngas,
    )
    supsatl_new, supsati_new = _build_supsat(
        t, gc, pvapl_new, pvapi_new, gwtmol_arr, zmet, ngas,
    )
    new_supsati = supsati_new[iz, iconv]

    # Retry triggers (any one of these flips rc to RC_WARNING_RETRY):
    #   1. Supersaturation changes sign AND new |ssi| > 1% (sign-flip
    #      heuristic from the original kernel).
    #   2. gsolve flagged dgc_threshold violation on any gas (relative
    #      gas-concentration change exceeds the per-gas threshold).
    #   3. tsolve flagged dt_threshold violation (temperature change
    #      exceeds the threshold from latent heat release).
    # The previous version discarded rc_gas / rc_t entirely — that's why
    # the H2SO4 sulfate runs blew through 100 timesteps without ever
    # substepping, even with dgc_threshold=0.1.
    sign_change = (prev_supsati * new_supsati) < DTYPE(0.0)
    large_change = jnp.abs(new_supsati) > DTYPE(0.01)
    rc_supsat = jnp.where(sign_change & large_change, RC_WARNING_RETRY, RC_OK)
    rc = jnp.where(
        (rc_supsat == RC_WARNING_RETRY)
        | (rc_gas == RC_WARNING_RETRY)
        | (rc_t == RC_WARNING_RETRY),
        RC_WARNING_RETRY,
        RC_OK,
    )

    return pc, gc, t, rlheat[iz], rc


def newstate_calc_growth(pc, gc, t, iz, dtime_orig,
                         rhoa, zmet, rlhe, rlhm, diffus,
                         akelvin, akelvini, gro, gro1, gro2,
                         rup_wet, rmass_2d, dm_2d, rlow_wet,
                         pratt, prat, pden1, palr,
                         is_ice_arr, igrowgas_arr, ienconc_arr,
                         igroup_arr, gwtmol_arr,
                         nbin, ngroup, ngas, nelem,
                         minsubsteps=1, maxsubsteps=128, maxretries=10,
                         dt_threshold=0.0, ds_threshold_arr=None,
                         scale_threshold=1.0):
    """Adaptive substepping wrapper for growth.

    Starts with minsubsteps. If microfast returns RC_WARNING_RETRY,
    doubles substeps and retries from saved state.

    Args:
        ... (same as microfast_growth, plus substepping params)

    Returns:
        Tuple of (pc, gc, t, rlheat_total, ntsubsteps_used)
    """
    if ds_threshold_arr is None:
        ds_threshold_arr = jnp.zeros(max(ngas, 1), dtype=DTYPE)

    # Save state for retry
    pc_saved = pc
    gc_saved = gc
    t_saved = t

    ntsubsteps = minsubsteps
    nretries = 0
    rlheat_total = 0.0

    while True:
        # Reset to saved state
        pc = pc_saved
        gc = gc_saved
        t = t_saved

        dtime = dtime_orig / ntsubsteps
        rlheat_total = 0.0
        rc = RC_OK

        # Run all substeps
        for isubstep in range(ntsubsteps):
            pc, gc, t, rlheat_val, rc = microfast_growth(
                pc, gc, t, iz, dtime,
                rhoa, zmet, rlhe, rlhm, diffus,
                akelvin, akelvini, gro, gro1, gro2,
                rup_wet, rmass_2d, dm_2d, rlow_wet,
                pratt, prat, pden1, palr,
                is_ice_arr, igrowgas_arr, ienconc_arr,
                igroup_arr, gwtmol_arr,
                nbin, ngroup, ngas, nelem,
                dt_threshold, ds_threshold_arr, scale_threshold,
            )
            rlheat_total += rlheat_val

            if rc == RC_WARNING_RETRY:
                break

        if rc != RC_WARNING_RETRY:
            # Success
            break

        # Retry with more substeps
        nretries += 1
        if nretries > maxretries:
            # Give up — use last result
            break

        ntsubsteps = min(ntsubsteps * 2, maxsubsteps)

    return pc, gc, t, rlheat_total, ntsubsteps
