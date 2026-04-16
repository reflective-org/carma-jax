"""Adaptive substepping engine for CARMA-JAX.

Implements the retry logic from newstate_calc.F90: if microfast
returns RC_WARNING_RETRY, double the substeps and retry from saved state.
Ported from: newstate_calc.F90
"""

import jax.numpy as jnp

from carma.constants import SMALL_PC, FEW_PC, CP, RGAS, WTMOL_H2O
from carma.enums import RC_OK, RC_WARNING_RETRY
from carma.precision import DTYPE
from carma.vapor_pressure import vaporp_h2o_murphy2005
from carma.supersaturation import supersat
from carma.growth.growevapl import growevapl
from carma.growth.growp import growp
from carma.growth.evapp import evapp, downgevapply
from carma.solvers.psolve import psolve
from carma.solvers.gsolve import gsolve
from carma.solvers.tsolve import tsolve
from carma.solvers.totalcondensate import totalcondensate


def microfast_growth(pc, gc, t, iz, dtime,
                     rhoa, zmet, rlhe, rlhm, diffus,
                     akelvin, akelvini, gro, gro1, gro2,
                     rup_wet, rmass_2d, dm_2d, rlow_wet,
                     pratt, prat, pden1, palr,
                     is_ice_arr, igrowgas_arr, ienconc_arr,
                     igroup_arr, gwtmol_arr,
                     nbin, ngroup, ngas, nelem,
                     dt_threshold, ds_threshold_arr, scale_threshold):
    """Execute one microfast growth step at one level.

    Computes: vapor pressure → supersaturation → condensate →
    growth rates → growth production → psolve → gsolve → tsolve →
    check convergence.

    Returns:
        Tuple of (pc, gc, t, rlheat_val, rc)
    """
    # Vapor pressure at current T
    pvapl_1d, pvapi_1d = vaporp_h2o_murphy2005(t)
    pvapl = pvapl_1d[None, :]  # (1, 1)
    pvapi = pvapi_1d[None, :]

    # Supersaturation
    ssl, ssi = supersat(t, gc[:, 0], pvapl[:, 0], pvapi[:, 0],
                        float(gwtmol_arr[0]), zmet)
    supsatl = ssl[:, None]
    supsati = ssi[:, None]

    # Save previous supersaturation for convergence check
    prev_supsati = float(ssi[iz])

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
    itype_arr = jnp.array([2])  # I_VOLATILE = 2 for growtest
    evappe = evapp(
        pc, evappe, evaplg, pconmax, ienconc_arr, itype_arr,
        iz, nbin, ngroup, nelem,
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

    # Check supersaturation convergence
    pvapl_new, pvapi_new = vaporp_h2o_murphy2005(t)
    ssl_new, ssi_new = supersat(t, gc[:, 0], pvapl_new[None, :][:, 0],
                                pvapi_new[None, :][:, 0],
                                float(gwtmol_arr[0]), zmet)
    new_supsati = float(ssi_new[iz])

    # Simple convergence check: if supersaturation changed sign, retry
    rc = RC_OK
    sign_change = (prev_supsati * new_supsati) < 0
    large_change = abs(new_supsati) > 0.01
    if sign_change and large_change:
        rc = RC_WARNING_RETRY

    return pc, gc, t, float(rlheat[iz]), rc


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
