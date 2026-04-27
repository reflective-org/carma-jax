"""Adaptive substepping engine for CARMA-JAX.

Implements the retry logic from newstate_calc.F90: if microfast
returns RC_WARNING_RETRY, double the substeps and retry from saved state.
Ported from: newstate_calc.F90
"""

import jax
import jax.numpy as jnp

from carma.constants import SMALL_PC, FEW_PC, CP, RGAS, AVG, BK, WTMOL_H2O
from carma.enums import RC_OK, RC_WARNING_RETRY
from carma.precision import DTYPE
from carma.vapor_pressure import vaporp_h2o_murphy2005, vaporp_h2so4_ayers1980
from carma.supersaturation import supersat
from carma.growth.growevapl import growevapl
from carma.growth.growp import growp
from carma.growth.evapp import evapp, downgevapply
from carma.nucleation.sulfnuc import sulfnuc
from carma.solvers.psolve import psolve
from carma.solvers.gsolve import gsolve
from carma.solvers.tsolve import tsolve
from carma.solvers.totalcondensate import totalcondensate
from carma.sulfate_utils import wtpct_tabaz


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


_GWTMOL_H2SO4 = DTYPE(98.0)
_GWTMOL_H2O = DTYPE(18.0)


def _compute_sulfnuc_rates(
    pc, gc, t, zmet, pvapl,
    sulf_r_bins, sulf_rmassup, sulf_rmrat,
    igroup_sulfate, igas_h2so4, igas_h2o,
    iz, nbin, ngroup, nelem,
    nuc_method="ZhaoTurco",
    do_homogeneous=True, do_heterogeneous=False,
):
    """Compute sulfate nucleation production rates.

    Mirrors Fortran's ``sulfnuc.F90`` placement INSIDE microfast: this
    fills the (nbin, nelem) ``rhompe`` and (nbin, ngroup, ngroup)
    ``rnuclg`` arrays that are then passed to ``psolve``. The previous
    JAX architecture computed nucleation in a separate ``sulfate_step``
    *after* microfast finished, breaking the time-implicit coupling
    between nucleation and growth that Fortran preserves through
    ``psolve``.

    Args:
        pc: (NZ, NBIN, NELEM) particle concentrations.
        gc: (NZ, NGAS) gas concentrations [g/cm³/z].
        t: (NZ,) temperature [K].
        zmet: (NZ,) vertical metric.
        pvapl: (NZ, NGAS) saturation vapor pressure (liquid).
        sulf_r_bins: (NBIN,) wet radius for sulfate group [cm].
        sulf_rmassup: (NBIN,) upper bin-boundary mass [g].
        sulf_rmrat: bin mass ratio (scalar).
        igroup_sulfate, igas_h2so4, igas_h2o: indices.
        iz: vertical level.
        nuc_method: "ZhaoTurco" or "Vehkamaki".
        do_homogeneous, do_heterogeneous: enable/disable branches.

    Returns:
        rhompe: (NBIN, NELEM) homogeneous production.
        rnuclg: (NBIN, NGROUP, NGROUP) heterogeneous loss rates.
    """
    rhompe = jnp.zeros((nbin, nelem), dtype=DTYPE)
    rnuclg = jnp.zeros((nbin, ngroup, ngroup), dtype=DTYPE)
    if igas_h2so4 < 0:
        return rhompe, rnuclg

    # Convert gas to molar densities at this level
    gc_h2so4 = gc[iz, igas_h2so4]
    gc_h2o = gc[iz, igas_h2o]
    zmet_iz = zmet[iz]
    h2so4_cgs = gc_h2so4 / zmet_iz
    h2o_cgs = gc_h2o / zmet_iz
    h2so4_num = h2so4_cgs * AVG / _GWTMOL_H2SO4
    h2o_num = h2o_cgs * AVG / _GWTMOL_H2O

    # Relative humidity and Tabazadeh wt% (Fortran's water activity input
    # to sulfnucrate uses gc_h2o expressed via Boltzmann form, not via
    # gc/pvapl ratio — we mirror sulfate_step._compute_state to get the
    # same numbers).
    rvap = RGAS / _GWTMOL_H2O
    pvapl_h2o = pvapl[iz, igas_h2o]
    rh = (h2o_cgs * rvap * t[iz]) / pvapl_h2o
    h2o_mass_wtp = rh * pvapl_h2o * _GWTMOL_H2O / (BK * t[iz] * AVG)
    wtp = wtpct_tabaz(t[iz], h2o_mass_wtp, pvapl_h2o)

    # Call sulfnuc to get per-bin rates
    rhompe_1d, rnuclg_1d = sulfnuc(
        temp=t[iz], weight_percent=wtp, rh=rh,
        h2so4=h2so4_num, h2so4_cgs=h2so4_cgs,
        h2o=h2o_num, h2o_cgs=h2o_cgs,
        r_bins=sulf_r_bins, rmassup=sulf_rmassup,
        rmrat_val=sulf_rmrat, zmet=zmet_iz,
        method=nuc_method,
        do_homogeneous=do_homogeneous,
        do_heterogeneous=do_heterogeneous,
    )

    # Place rates into the full (nbin, nelem) and (nbin, ngroup, ngroup)
    # arrays. For sulfate: ielem = ienconc(igroup_sulfate); rnuclg
    # source-target both = igroup_sulfate (sulfate-onto-self het).
    # NOTE: sulfate group's number-concentration element index. For our
    # sulfate test ielem == igroup_sulfate (single element).
    rhompe = rhompe.at[:, igroup_sulfate].set(rhompe_1d)
    rnuclg = rnuclg.at[:, igroup_sulfate, igroup_sulfate].set(rnuclg_1d)
    return rhompe, rnuclg


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
                     ivaprtn_arr=None, igas_h2o=0,
                     do_sulfnuc=False,
                     igroup_sulfate=0, igas_h2so4=-1,
                     sulf_r_bins=None, sulf_rmassup=None,
                     sulf_rmrat=2.0,
                     nuc_method="ZhaoTurco",
                     do_homogeneous_nuc=True,
                     do_heterogeneous_nuc=False):
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

    # Sulfate nucleation rates (Fortran microfast.F90 calls sulfnuc HERE,
    # right after supersat and BEFORE growevapl, so the rhompe/rnuclg
    # rates are then time-implicit-coupled with growth via psolve below).
    rhompe = jnp.zeros((nbin, nelem), dtype=DTYPE)
    rnuclg = jnp.zeros((nbin, ngroup, ngroup), dtype=DTYPE)
    if do_sulfnuc and sulf_r_bins is not None:
        rhompe, rnuclg = _compute_sulfnuc_rates(
            pc, gc, t, zmet, pvapl,
            sulf_r_bins, sulf_rmassup, sulf_rmrat,
            igroup_sulfate, igas_h2so4, igas_h2o,
            iz, nbin, ngroup, nelem,
            nuc_method=nuc_method,
            do_homogeneous=do_homogeneous_nuc,
            do_heterogeneous=do_heterogeneous_nuc,
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

    # Growth production and solve per bin (now psolve sees nucleation
    # rates too — rhompe and rnuclg from sulfnuc above).
    growpe = jnp.zeros((nbin, nelem), dtype=DTYPE)
    evappe_psolve = jnp.zeros((nbin, nelem), dtype=DTYPE)  # zero for psolve (evap applied separately)
    rnucpe = jnp.zeros((nbin, nelem), dtype=DTYPE)
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
    new_supsatl = supsatl_new[iz, iconv]
    prev_supsatl = supsatl[iz, iconv]
    # Track the saturation that matches the temperature regime (Fortran
    # uses supsatl above T0, supsati below).
    use_liquid = t[iz] >= DTYPE(273.16)
    old_ssat = jnp.where(use_liquid, prev_supsatl, prev_supsati)
    new_ssat = jnp.where(use_liquid, new_supsatl, new_supsati)

    # Retry triggers (any one of these flips rc to RC_WARNING_RETRY):
    #   1. Sign change AND |new| > 1% (sign-flip heuristic).
    #   2. RELATIVE supsat change > ds_threshold AND |old - new| > 0.1.
    #      Mirrors microfast.F90 lines 187-225 — this is what makes
    #      Fortran sub-step finely when nucleation is fast (H2SO4
    #      saturation drops 10%/substep). Without this, JAX never
    #      sub-steps for the sulfate test (sign never flips, gas only
    #      changes 0.2%/outer-step) so psolve damps all nucleation.
    #   3. gsolve flagged dgc_threshold violation.
    #   4. tsolve flagged dt_threshold violation.
    sign_change = (prev_supsati * new_supsati) < DTYPE(0.0)
    large_change = jnp.abs(new_supsati) > DTYPE(0.01)
    rc_signflip = jnp.where(sign_change & large_change, RC_WARNING_RETRY, RC_OK)

    # Relative-change retry, per Fortran microfast.F90:
    ds_thresh = ds_threshold_arr[iconv] / scale_threshold
    abs_old = jnp.abs(old_ssat)
    abs_new = jnp.abs(new_ssat)
    srat1 = jnp.where(abs_old >= DTYPE(1e-4),
                       jnp.abs(new_ssat / jnp.where(abs_old > 0, old_ssat, DTYPE(1.0)) - DTYPE(1.0)),
                       DTYPE(0.0))
    srat2 = jnp.where(abs_new >= DTYPE(1e-4),
                       jnp.abs(old_ssat / jnp.where(abs_new > 0, new_ssat, DTYPE(1.0)) - DTYPE(1.0)),
                       DTYPE(0.0))
    srat = jnp.maximum(srat1, srat2)
    abs_change = jnp.abs(old_ssat - new_ssat)
    rc_relchange = jnp.where(
        (ds_thresh > DTYPE(0.0))
        & (srat >= ds_thresh)
        & (abs_change > DTYPE(0.1)),
        RC_WARNING_RETRY, RC_OK,
    )

    rc = jnp.where(
        (rc_signflip == RC_WARNING_RETRY)
        | (rc_relchange == RC_WARNING_RETRY)
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
